// pipeline-asr.cpp: wav -> text orchestration. Holds the backend pair, the
// loaded weights, the BPE tokenizer, and runs mel -> conv stem -> audio encoder
// -> audio prefix splice -> Qwen3 decoder greedy loop -> detokenize.

#include "pipeline-asr.h"

#include "audio-enc.h"
#include "audio-mel.h"
#include "audio-resample.h"
#include "audio-tower.h"
#include "backend.h"
#include "bpe.h"
#include "conv-stem.h"
#include "detok-stream.h"
#include "gguf-weights.h"
#include "kv-cache.h"
#include "lang-map.h"
#include "prompt-asr.h"
#include "qa-error.h"
#include "sampling.h"
#include "thinker-forward.h"
#include "thinker-weights.h"
#include "timer.h"

#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Mel frontend runs at this fixed rate, inputs at other rates are resampled.
static const int kModelSampleRate = 16000;

struct pipeline_asr {
  BackendPair backend;
  ggml_backend_sched_t sched;
  ConvStem stem;
  AudioEnc enc;
  ThinkerWeights tw;
  BPETokenizer tok;
  AsrSpecials sp;
  AudioMelConfig mel_cfg;
  AudioTowerConfig tower_cfg;
  bool flash_attn;
  bool clamp_fp16;
  std::string dump_dir;
};

// Reflect pad on the host, Whisper center=True with pad = n_fft / 2. PyTorch
// reflect copies samples [1..pad] then [T-2..T-pad-1] into the edges, the
// boundary sample itself is not duplicated.
static std::vector<float> reflect_pad(const std::vector<float> &x, int pad) {
  const int T = (int)x.size();
  std::vector<float> out((size_t)(T + 2 * pad));
  for (int i = 0; i < pad; i++) {
    out[(size_t)i] = x[(size_t)(pad - i)];
  }
  for (int i = 0; i < T; i++) {
    out[(size_t)(pad + i)] = x[(size_t)i];
  }
  for (int i = 0; i < pad; i++) {
    out[(size_t)(pad + T + i)] = x[(size_t)(T - 2 - i)];
  }
  return out;
}

// Write a host f32 buffer as a raw little endian dump for the cossim harness.
static void dump_f32(const std::string &dir, const char *name,
                     const std::vector<float> &v) {
  const std::string path = dir + "/" + name;
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) {
    qa_log(QA_LOG_WARN, "[Dump] cannot open %s", path.c_str());
    return;
  }
  fwrite(v.data(), sizeof(float), v.size(), f);
  fclose(f);
  qa_log(QA_LOG_INFO, "[Dump] %s [%zu floats]", path.c_str(), v.size());
}

// Mel graph on the shared backend. Returns the normalized mel [n_mels,
// n_frames] and reports n_frames through n_frames_out.
static std::vector<float> run_mel(pipeline_asr *p,
                                  const std::vector<float> &samples,
                                  size_t *n_frames_out) {
  const AudioMelConfig &cfg = p->mel_cfg;
  AudioMelConstants c;
  audio_mel_compute_constants(cfg, c);

  const int pad = cfg.n_fft / 2;
  if ((int)samples.size() < pad + 1) {
    qa_throw("run_mel: audio too short for reflect pad");
  }
  std::vector<float> audio_padded = reflect_pad(samples, pad);
  const int T_pad = (int)audio_padded.size();

  const size_t mem = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                     ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
  struct ggml_init_params gp = {mem, nullptr, true};
  struct ggml_context *gctx = ggml_init(gp);

  struct ggml_tensor *audio_in = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, T_pad);
  struct ggml_tensor *hann_in =
      ggml_new_tensor_1d(gctx, GGML_TYPE_F32, cfg.n_fft);
  struct ggml_tensor *dft_re_in =
      ggml_new_tensor_2d(gctx, GGML_TYPE_F32, cfg.n_fft, c.n_freq);
  struct ggml_tensor *dft_im_in =
      ggml_new_tensor_2d(gctx, GGML_TYPE_F32, cfg.n_fft, c.n_freq);
  struct ggml_tensor *mel_b_in =
      ggml_new_tensor_2d(gctx, GGML_TYPE_F32, c.n_freq, cfg.n_mels);
  ggml_set_input(audio_in);
  ggml_set_input(hann_in);
  ggml_set_input(dft_re_in);
  ggml_set_input(dft_im_in);
  ggml_set_input(mel_b_in);

  struct ggml_tensor *mel = audio_mel_build_graph(
      gctx, audio_in, hann_in, dft_re_in, dft_im_in, mel_b_in, cfg);
  ggml_set_output(mel);

  struct ggml_cgraph *graph =
      ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE, false);
  ggml_build_forward_expand(graph, mel);

  ggml_backend_sched_reset(p->sched);
  if (!ggml_backend_sched_alloc_graph(p->sched, graph)) {
    qa_throw("run_mel: sched alloc failed");
  }
  ggml_backend_tensor_set(audio_in, audio_padded.data(), 0,
                          (size_t)T_pad * sizeof(float));
  ggml_backend_tensor_set(hann_in, c.hann.data(), 0,
                          c.hann.size() * sizeof(float));
  ggml_backend_tensor_set(dft_re_in, c.dft_real.data(), 0,
                          c.dft_real.size() * sizeof(float));
  ggml_backend_tensor_set(dft_im_in, c.dft_imag.data(), 0,
                          c.dft_imag.size() * sizeof(float));
  ggml_backend_tensor_set(mel_b_in, c.mel_basis.data(), 0,
                          c.mel_basis.size() * sizeof(float));
  if (ggml_backend_sched_graph_compute(p->sched, graph) !=
      GGML_STATUS_SUCCESS) {
    qa_throw("run_mel: graph compute failed");
  }

  const size_t n_frames_full = (size_t)mel->ne[0];
  std::vector<float> log10_mel((size_t)cfg.n_mels * n_frames_full);
  ggml_backend_tensor_get(mel, log10_mel.data(), 0,
                          log10_mel.size() * sizeof(float));

  std::vector<float> out;
  *n_frames_out =
      audio_mel_normalize(log10_mel, cfg.n_mels, n_frames_full, out);
  ggml_free(gctx);
  if (!p->dump_dir.empty()) {
    dump_f32(p->dump_dir, "mel.bin", out);
  }
  return out;
}

// Windowed audio tower on the shared backend. Returns the encoder states
// [output_dim, S] host f32 and reports S through n_states_out.
static std::vector<float> run_tower(pipeline_asr *p, int n_mels,
                                    const std::vector<float> &mel,
                                    size_t n_frames, int *n_states_out) {
  const AudioTowerConfig &tcfg = p->tower_cfg;
  std::vector<int> chunk_lengths =
      audio_tower_chunk_lengths((int)n_frames, tcfg);
  const int S = audio_tower_seq_len(chunk_lengths);
  const int window_aftercnn =
      audio_tower_conv_out(tcfg.chunk_mel) * tcfg.window_chunks;
  const int d_model = p->enc.cfg.d_model;

  int pe_rows = 0;
  for (int len : chunk_lengths) {
    const int t_c = audio_tower_conv_out(len);
    if (t_c > pe_rows) {
      pe_rows = t_c;
    }
  }
  std::vector<float> pe;
  audio_enc_compute_pe(d_model, pe_rows, pe);
  std::vector<float> mask = audio_tower_build_mask(S, window_aftercnn);

  const size_t mem = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                     ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
  struct ggml_init_params gp = {mem, nullptr, true};
  struct ggml_context *gctx = ggml_init(gp);

  struct ggml_tensor *mel_in =
      ggml_new_tensor_4d(gctx, GGML_TYPE_F32, (int64_t)n_frames, n_mels, 1, 1);
  struct ggml_tensor *pe_in =
      ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, pe_rows);
  struct ggml_tensor *mask_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, S, S);
  ggml_set_input(mel_in);
  ggml_set_input(pe_in);
  ggml_set_input(mask_in);

  const bool dump = !p->dump_dir.empty();
  struct ggml_tensor *stem_t = nullptr;
  struct ggml_tensor *out = audio_tower_build(
      gctx, p->stem, p->enc, mel_in, pe_in, mask_in, chunk_lengths, &stem_t);
  ggml_set_output(out);
  if (dump) {
    ggml_set_output(stem_t);
  }

  struct ggml_cgraph *graph =
      ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE, false);
  ggml_build_forward_expand(graph, out);

  ggml_backend_sched_reset(p->sched);
  if (!ggml_backend_sched_alloc_graph(p->sched, graph)) {
    qa_throw("run_tower: sched alloc failed");
  }
  ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));
  ggml_backend_tensor_set(pe_in, pe.data(), 0, pe.size() * sizeof(float));
  ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));
  if (ggml_backend_sched_graph_compute(p->sched, graph) !=
      GGML_STATUS_SUCCESS) {
    qa_throw("run_tower: graph compute failed");
  }

  const int64_t out_dim = out->ne[0];
  const int64_t t_out = out->ne[1];
  std::vector<float> states((size_t)out_dim * (size_t)t_out);
  ggml_backend_tensor_get(out, states.data(), 0, states.size() * sizeof(float));
  if (dump) {
    std::vector<float> stem_host((size_t)stem_t->ne[0] * (size_t)stem_t->ne[1]);
    ggml_backend_tensor_get(stem_t, stem_host.data(), 0,
                            stem_host.size() * sizeof(float));
    dump_f32(p->dump_dir, "stem.bin", stem_host);
    dump_f32(p->dump_dir, "windowed.bin", states);
  }
  ggml_free(gctx);
  *n_states_out = (int)t_out;
  return states;
}

// Look up token_embd for the ids via get_rows. Returns [T, hidden] host f32.
static std::vector<float> embed_tokens(pipeline_asr *p,
                                       const std::vector<int> &ids) {
  const int T = (int)ids.size();
  const int hidden = p->tw.hidden_size;

  std::vector<int32_t> ids32(ids.begin(), ids.end());

  const size_t mem =
      ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false);
  struct ggml_init_params gp = {mem, nullptr, true};
  struct ggml_context *gctx = ggml_init(gp);

  struct ggml_tensor *ids_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
  ggml_set_input(ids_in);
  struct ggml_tensor *emb = ggml_get_rows(gctx, p->tw.token_embd, ids_in);
  ggml_set_output(emb);

  struct ggml_cgraph *gf = ggml_new_graph_custom(gctx, 16, false);
  ggml_build_forward_expand(gf, emb);

  ggml_backend_sched_reset(p->sched);
  if (!ggml_backend_sched_alloc_graph(p->sched, gf)) {
    qa_throw("embed_tokens: sched alloc failed");
  }
  ggml_backend_tensor_set(ids_in, ids32.data(), 0, (size_t)T * sizeof(int32_t));
  if (ggml_backend_sched_graph_compute(p->sched, gf) != GGML_STATUS_SUCCESS) {
    qa_throw("embed_tokens: graph compute failed");
  }

  std::vector<float> host((size_t)T * (size_t)hidden);
  ggml_backend_tensor_get(emb, host.data(), 0, host.size() * sizeof(float));
  ggml_free(gctx);
  return host;
}

pipeline_asr *pipeline_asr_load(const pipeline_asr_params &params) {
  pipeline_asr *p = new pipeline_asr();
  p->flash_attn = params.flash_attn;
  p->clamp_fp16 = params.clamp_fp16;
  p->dump_dir = params.dump_dir;

  p->backend = backend_init("Pipeline");
  if (!p->backend.backend) {
    delete p;
    qa_throw("pipeline_asr_load: no backend");
  }
  p->sched = backend_sched_new(p->backend, GGML_DEFAULT_GRAPH_SIZE);

  GGUFModel gf;
  if (!gf_load(&gf, params.model_path.c_str())) {
    qa_throw("pipeline_asr_load: cannot load gguf %s",
             params.model_path.c_str());
  }
  if (!conv_stem_load(&p->stem, gf, p->backend.backend)) {
    qa_throw("pipeline_asr_load: conv stem load failed");
  }
  if (!audio_enc_load(&p->enc, gf, p->backend.backend)) {
    qa_throw("pipeline_asr_load: audio encoder load failed");
  }
  if (!thinker_weights_load(&p->tw, gf, p->backend.backend)) {
    qa_throw("pipeline_asr_load: thinker load failed");
  }
  if (!load_bpe_from_gguf(&p->tok, params.model_path.c_str())) {
    qa_throw("pipeline_asr_load: tokenizer load failed");
  }
  p->sp = asr_resolve_specials(p->tok, gf);
  p->tower_cfg = audio_tower_config_load(gf);
  gf_close(&gf);
  return p;
}

void pipeline_asr_free(pipeline_asr *p) {
  if (!p) {
    return;
  }
  thinker_weights_free(&p->tw);
  audio_enc_free(&p->enc);
  conv_stem_free(&p->stem);
  ggml_backend_sched_free(p->sched);
  backend_release(p->backend.backend, p->backend.cpu_backend);
  delete p;
}

// Per stage wall clock for one transcription. Every span ends on a device
// readback, so the measured time includes the GPU work.
struct AsrPerf {
  double resample_ms; // host resample to 16k mono, zero when input is 16k
  double mel_ms;      // mel frontend graph
  double tower_ms;    // windowed audio tower
  double build_ms;    // prompt build, token embed, audio splice
  double prefill_ms;  // thinker prefill over the prompt span
  double decode_ms;   // thinker decode step, summed
  double total_ms;    // entry to return
  int n_tokens;       // generated tokens
  double audio_sec;   // input audio duration
};

// Route the per stage timings through qa_log at info level.
static void asr_log_perf(const AsrPerf &p) {
  const double rtf =
      p.audio_sec > 0.0 ? (p.total_ms / 1000.0) / p.audio_sec : 0.0;
  const double ms_tok = p.n_tokens > 0 ? p.decode_ms / (double)p.n_tokens : 0.0;
  const double tok_s =
      p.decode_ms > 0.0 ? (double)p.n_tokens / (p.decode_ms / 1000.0) : 0.0;

  qa_log(QA_LOG_INFO, "[Perf] Resample %.1f ms", p.resample_ms);
  qa_log(QA_LOG_INFO, "[Perf] Mel %.1f ms", p.mel_ms);
  qa_log(QA_LOG_INFO, "[Perf] Tower %.1f ms", p.tower_ms);
  qa_log(QA_LOG_INFO, "[Perf] Build %.1f ms (embed + audio splice)",
         p.build_ms);
  qa_log(QA_LOG_INFO, "[Perf] Prefill %.1f ms", p.prefill_ms);
  qa_log(QA_LOG_INFO,
         "[Perf] Decode %.1f ms (%d tokens, %.2f ms/tok, %.1f tok/s)",
         p.decode_ms, p.n_tokens, ms_tok, tok_s);
  qa_log(QA_LOG_INFO, "[Perf] Total %.1f ms (audio %.2f s, RTF %.3f)",
         p.total_ms, p.audio_sec, rtf);
}

qa_status pipeline_asr_run(pipeline_asr *p, const float *pcm, size_t n_samples,
                           int sample_rate, const qa_transcribe_params *params,
                           std::string &text) {
  Timer t_total;
  AsrPerf perf = {};
  perf.audio_sec = (double)n_samples / (double)sample_rate;

  Timer t_re;
  std::vector<float> samples;
  if (sample_rate != kModelSampleRate) {
    int n_out = 0;
    float *r = audio_resample(pcm, (int)n_samples, sample_rate,
                              kModelSampleRate, 1, &n_out);
    samples.assign(r, r + n_out);
    free(r);
  } else {
    samples.assign(pcm, pcm + n_samples);
  }
  perf.resample_ms = t_re.ms();

  if (!p->dump_dir.empty()) {
    dump_f32(p->dump_dir, "pcm16.bin", samples);
  }

  Timer t_mel;
  size_t n_frames = 0;
  std::vector<float> mel = run_mel(p, samples, &n_frames);
  perf.mel_ms = t_mel.ms();

  Timer t_tower;
  int S = 0;
  std::vector<float> states =
      run_tower(p, p->mel_cfg.n_mels, mel, n_frames, &S);
  perf.tower_ms = t_tower.ms();

  Timer t_build;
  const std::string language =
      params->language ? resolve_language(params->language) : std::string();
  const std::string context = params->context ? params->context : "";
  AsrPrompt prompt = asr_build_prompt(p->tok, p->sp, context, language, S);

  const int hidden = p->tw.hidden_size;
  const int T = (int)prompt.ids.size();
  std::vector<float> embed = embed_tokens(p, prompt.ids);
  memcpy(embed.data() + (size_t)prompt.audio_offset * (size_t)hidden,
         states.data(), (size_t)S * (size_t)hidden * sizeof(float));
  perf.build_ms = t_build.ms();

  const int max_new = params->max_new_tokens > 0 ? params->max_new_tokens
                                                 : QA_DEFAULT_MAX_NEW_TOKENS;

  KVCache kv;
  if (!kv_cache_init(&kv, p->tw.num_hidden_layers, p->tw.num_key_value_heads,
                     p->tw.head_dim, T + max_new, p->backend.backend)) {
    qa_set_error("pipeline_asr_run: kv cache init failed");
    return QA_STATUS_GENERATE_FAILED;
  }

  Timer t_pf;
  ThinkerForwardOutput out;
  if (!thinker_forward_prefill(&p->tw, &kv, p->sched, embed.data(), T,
                               p->flash_attn, p->clamp_fp16, &out)) {
    kv_cache_free(&kv);
    qa_set_error("pipeline_asr_run: prefill failed");
    return QA_STATUS_GENERATE_FAILED;
  }
  perf.prefill_ms = t_pf.ms();

  if (!p->dump_dir.empty()) {
    dump_f32(p->dump_dir, "hidden.bin", out.hidden_all);
    dump_f32(p->dump_dir, "logits.bin", out.logits_last);
  }

  // Resolve sampling. seed -1 draws a hardware seed, temperature <= 0 stays
  // greedy argmax. Each token advances the philox subsequence so a fixed seed
  // reproduces the same draw.
  int64_t seed = params->seed;
  if (seed < 0) {
    std::random_device rd;
    seed = ((int64_t)rd() << 32) ^ (int64_t)rd();
  }
  const int vocab = (int)out.logits_last.size();
  int64_t subseq = 0;

  std::vector<int> gen;

  // Streaming detok feeds one token per step and returns the bytes that
  // complete a UTF-8 boundary, so a codepoint split across tokens never streams
  // half. In auto detect mode the decoder emits a "language {Lang}<asr_text>"
  // preamble first, the asr_text marker resets the segment so only the
  // transcript reaches the stream and the final result.
  StreamDetok sd;
  detok_init(&sd, &p->tok, p->sp.asr_text);
  auto sample = [&]() {
    return sample_top_k_p(out.logits_last.data(), vocab, params->temperature,
                          params->top_k, params->top_p,
                          params->repetition_penalty, gen.data(),
                          (int)gen.size(), seed, subseq++, nullptr);
  };

  int next = sample();

  for (int step = 0; step < max_new; step++) {
    if (next == p->sp.im_end || next == p->sp.eos) {
      break;
    }
    gen.push_back(next);

    std::string delta = detok_feed(&sd, next);
    if (params->on_token && !delta.empty()) {
      if (!params->on_token(delta.c_str(), params->user)) {
        kv_cache_free(&kv);
        return QA_STATUS_CANCELLED;
      }
    }

    Timer t_dec;
    if (!thinker_forward_decode(&p->tw, &kv, p->sched, next, p->flash_attn,
                                p->clamp_fp16, &out)) {
      kv_cache_free(&kv);
      qa_set_error("pipeline_asr_run: decode failed");
      return QA_STATUS_GENERATE_FAILED;
    }
    perf.decode_ms += t_dec.ms();
    next = sample();
  }

  text = detok_text(&sd);
  kv_cache_free(&kv);

  perf.n_tokens = (int)gen.size();
  perf.total_ms = t_total.ms();
  asr_log_perf(perf);
  return QA_STATUS_OK;
}
