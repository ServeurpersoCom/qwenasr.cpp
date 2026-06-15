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
#include "gguf-weights.h"
#include "kv-cache.h"
#include "lang-map.h"
#include "prompt-asr.h"
#include "qa-error.h"
#include "sampling.h"
#include "thinker-forward.h"
#include "thinker-weights.h"

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
  bool flash_attn;
  bool clamp_fp16;
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
  return out;
}

// Windowed audio tower on the shared backend. Returns the encoder states
// [output_dim, S] host f32 and reports S through n_states_out.
static std::vector<float> run_tower(pipeline_asr *p, int n_mels,
                                    const std::vector<float> &mel,
                                    size_t n_frames, int *n_states_out) {
  AudioTowerConfig tcfg;
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

  struct ggml_tensor *out = audio_tower_build(gctx, p->stem, p->enc, mel_in,
                                              pe_in, mask_in, chunk_lengths);
  ggml_set_output(out);

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
  bpe_load_qwenasr_specials(&p->tok, params.model_path.c_str());
  p->sp = asr_resolve_specials(p->tok, gf);
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

qa_status pipeline_asr_run(pipeline_asr *p, const float *pcm, size_t n_samples,
                           int sample_rate, const qa_transcribe_params *params,
                           std::string &text) {
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

  size_t n_frames = 0;
  std::vector<float> mel = run_mel(p, samples, &n_frames);

  int S = 0;
  std::vector<float> states =
      run_tower(p, p->mel_cfg.n_mels, mel, n_frames, &S);

  const std::string language =
      params->language ? resolve_language(params->language) : std::string();
  const std::string context = params->context ? params->context : "";
  AsrPrompt prompt = asr_build_prompt(p->tok, p->sp, context, language, S);

  const int hidden = p->tw.hidden_size;
  const int T = (int)prompt.ids.size();
  std::vector<float> embed = embed_tokens(p, prompt.ids);
  memcpy(embed.data() + (size_t)prompt.audio_offset * (size_t)hidden,
         states.data(), (size_t)S * (size_t)hidden * sizeof(float));

  const int max_new = params->max_new_tokens > 0 ? params->max_new_tokens
                                                 : QA_DEFAULT_MAX_NEW_TOKENS;

  KVCache kv;
  if (!kv_cache_init(&kv, p->tw.num_hidden_layers, p->tw.num_key_value_heads,
                     p->tw.head_dim, T + max_new, p->backend.backend)) {
    qa_set_error("pipeline_asr_run: kv cache init failed");
    return QA_STATUS_GENERATE_FAILED;
  }

  ThinkerForwardOutput out;
  if (!thinker_forward_prefill(&p->tw, &kv, p->sched, embed.data(), T,
                               p->flash_attn, p->clamp_fp16, &out)) {
    kv_cache_free(&kv);
    qa_set_error("pipeline_asr_run: prefill failed");
    return QA_STATUS_GENERATE_FAILED;
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
  auto sample = [&]() {
    return sample_top_k_p(out.logits_last.data(), vocab, params->temperature,
                          params->top_k, params->top_p,
                          params->repetition_penalty, gen.data(),
                          (int)gen.size(), seed, subseq++, nullptr);
  };

  int next = sample();

  // In auto detect mode the decoder emits a "language {Lang}<asr_text>"
  // preamble before the transcript. The transcript is everything after the last
  // asr_text marker, so forced language runs (marker already in the prompt)
  // keep the full generation while auto runs drop the preamble.
  auto transcript_text = [&]() {
    int start = 0;
    for (int i = 0; i < (int)gen.size(); i++) {
      if (gen[i] == p->sp.asr_text) {
        start = i + 1;
      }
    }
    std::vector<int> tr(gen.begin() + start, gen.end());
    return bpe_decode(&p->tok, tr);
  };

  for (int step = 0; step < max_new; step++) {
    if (next == p->sp.im_end || next == p->sp.eos) {
      break;
    }
    gen.push_back(next);

    if (params->on_token) {
      std::string full = transcript_text();
      if (full.size() > text.size()) {
        std::string delta = full.substr(text.size());
        text = full;
        if (!params->on_token(delta.c_str(), params->user)) {
          kv_cache_free(&kv);
          return QA_STATUS_CANCELLED;
        }
      }
    }

    std::vector<int> one = {next};
    std::vector<float> e1 = embed_tokens(p, one);
    if (!thinker_forward_decode(&p->tw, &kv, p->sched, e1.data(), p->flash_attn,
                                p->clamp_fp16, &out)) {
      kv_cache_free(&kv);
      qa_set_error("pipeline_asr_run: decode failed");
      return QA_STATUS_GENERATE_FAILED;
    }
    next = sample();
  }

  text = transcript_text();
  kv_cache_free(&kv);
  return QA_STATUS_OK;
}
