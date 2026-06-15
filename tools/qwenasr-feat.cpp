// qwenasr-feat: staged debug dump for the cossim harness. Runs the mel frontend
// on an input file and writes mel.bin. With --model it also runs the conv2d
// stem and writes stem.bin, and with --encoder it chains the audio encoder and
// writes encoder.bin. All stages share one backend and one scheduler. Backend
// follows the GGML_BACKEND env var.

#include "audio-enc.h"
#include "audio-io.h"
#include "audio-mel.h"
#include "audio-tower.h"
#include "backend.h"
#include "conv-stem.h"
#include "qwenasr.h"
#include "utf8.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void print_usage(const char *prog) {
  fprintf(stderr, "qwenasr.cpp %s\n\n", qa_version());
  fprintf(
      stderr,
      "Staged feature dump for the cossim harness. Runs the audio front of "
      "the\n"
      "model up to a chosen stage and writes the f32 tensors so the Python\n"
      "reference can compare them. Each flag adds the next stage in the "
      "chain.\n\n"
      "Usage: %s --file <wav> --dump <dir> [--model <gguf>] [--encoder] "
      "[--windowed] [--pad-30s]\n\n"
      "  --file <wav>   Input audio, any rate (resampled to 16 kHz mono)\n"
      "  --dump <dir>   Existing directory for the f32 dumps\n"
      "  --model <gguf> Run the conv2d stem and dump stem.bin\n"
      "  --encoder      Chain the audio encoder and dump encoder.bin (needs "
      "--model)\n"
      "  --windowed     Run the windowed tower and dump windowed.bin (needs "
      "--model)\n"
      "  --pad-30s      Pad or trim to 480000 samples like the reference "
      "extractor\n\n"
      "Dumps: mel.bin, stem.bin, encoder.bin, windowed.bin.\n",
      prog);
}

static void dump_f32(const char *path, const std::vector<float> &v) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  fwrite(v.data(), sizeof(float), v.size(), f);
  fclose(f);
}

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

// Run the mel graph on the shared backend. Returns the normalized mel
// [n_mels, n_frames] and reports n_frames through n_frames_out.
static std::vector<float> run_mel(BackendPair bp, ggml_backend_sched_t sched,
                                  const AudioMelConfig &cfg,
                                  const std::vector<float> &samples,
                                  size_t *n_frames_out) {
  AudioMelConstants c;
  audio_mel_compute_constants(cfg, c);

  const int pad = cfg.n_fft / 2;
  if ((int)samples.size() < pad + 1) {
    throw std::runtime_error("audio too short for reflect pad");
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
  ggml_set_name(audio_in, "mel.audio_padded");
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

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, graph)) {
    throw std::runtime_error("mel sched alloc failed");
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
  if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("mel graph compute failed");
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

// Run the conv stem on the shared backend. Dumps stem.bin and returns the stem
// states [d_model, t_out] in host memory, reporting the dims through the
// outputs.
static std::vector<float> run_stem(const GGUFModel &gf, BackendPair bp,
                                   ggml_backend_sched_t sched, int n_mels,
                                   const std::vector<float> &mel,
                                   size_t n_frames, const char *dump,
                                   int64_t *d_model_out, int64_t *t_out_out) {
  ConvStem stem;
  if (!conv_stem_load(&stem, gf, bp.backend)) {
    throw std::runtime_error("conv stem load failed");
  }

  const size_t mem = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                     ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
  struct ggml_init_params gp = {mem, nullptr, true};
  struct ggml_context *gctx = ggml_init(gp);

  struct ggml_tensor *mel_in =
      ggml_new_tensor_4d(gctx, GGML_TYPE_F32, (int64_t)n_frames, n_mels, 1, 1);
  ggml_set_name(mel_in, "mel_in");
  ggml_set_input(mel_in);

  struct ggml_tensor *out = conv_stem_build(gctx, stem, mel_in);
  ggml_set_output(out);

  struct ggml_cgraph *graph =
      ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE, false);
  ggml_build_forward_expand(graph, out);

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, graph)) {
    throw std::runtime_error("stem sched alloc failed");
  }
  ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));
  if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("stem graph compute failed");
  }

  const int64_t d_model = out->ne[0];
  const int64_t t_out = out->ne[1];
  std::vector<float> stem_out((size_t)d_model * (size_t)t_out);
  ggml_backend_tensor_get(out, stem_out.data(), 0,
                          stem_out.size() * sizeof(float));

  char path[1024];
  snprintf(path, sizeof(path), "%s/stem.bin", dump);
  dump_f32(path, stem_out);
  fprintf(stderr, "qwenasr-feat %s: stem [%lld, %lld] -> %s\n", qa_version(),
          (long long)t_out, (long long)d_model, path);

  ggml_free(gctx);
  conv_stem_free(&stem);
  *d_model_out = d_model;
  *t_out_out = t_out;
  return stem_out;
}

// Run the audio encoder on the shared backend. Single window, full
// bidirectional attention. Dumps encoder.bin [output_dim, t].
static void run_encoder(const GGUFModel &gf, BackendPair bp,
                        ggml_backend_sched_t sched,
                        const std::vector<float> &stem, int64_t d_model,
                        int64_t t, const char *dump) {
  AudioEnc enc;
  if (!audio_enc_load(&enc, gf, bp.backend)) {
    throw std::runtime_error("audio encoder load failed");
  }

  std::vector<float> pe;
  audio_enc_compute_pe((int)d_model, (int)t, pe);

  const size_t mem = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
                     ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
  struct ggml_init_params gp = {mem, nullptr, true};
  struct ggml_context *gctx = ggml_init(gp);

  struct ggml_tensor *stem_in =
      ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, t);
  struct ggml_tensor *pe_in =
      ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, t);
  ggml_set_name(stem_in, "enc.stem");
  ggml_set_name(pe_in, "enc.pe");
  ggml_set_input(stem_in);
  ggml_set_input(pe_in);

  struct ggml_tensor *seq = ggml_add(gctx, stem_in, pe_in);
  struct ggml_tensor *out = audio_enc_build(gctx, enc, seq, nullptr);
  ggml_set_output(out);

  struct ggml_cgraph *graph =
      ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE, false);
  ggml_build_forward_expand(graph, out);

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, graph)) {
    throw std::runtime_error("encoder sched alloc failed");
  }
  ggml_backend_tensor_set(stem_in, stem.data(), 0, stem.size() * sizeof(float));
  ggml_backend_tensor_set(pe_in, pe.data(), 0, pe.size() * sizeof(float));
  if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("encoder graph compute failed");
  }

  const int64_t out_dim = out->ne[0];
  const int64_t t_out = out->ne[1];
  std::vector<float> enc_out((size_t)out_dim * (size_t)t_out);
  ggml_backend_tensor_get(out, enc_out.data(), 0,
                          enc_out.size() * sizeof(float));

  char path[1024];
  snprintf(path, sizeof(path), "%s/encoder.bin", dump);
  dump_f32(path, enc_out);
  fprintf(stderr, "qwenasr-feat %s: encoder [%lld, %lld] -> %s\n", qa_version(),
          (long long)t_out, (long long)out_dim, path);

  ggml_free(gctx);
  audio_enc_free(&enc);
}

// Run the windowed audio tower on the shared backend. Splits the mel into
// chunk_mel frame chunks, conv per chunk with positional reset, block-diagonal
// attention over windows of window_aftercnn frames. Dumps windowed.bin
// [output_dim, S].
static void run_windowed(const GGUFModel &gf, BackendPair bp,
                         ggml_backend_sched_t sched, int n_mels,
                         const std::vector<float> &mel, size_t n_frames,
                         const char *dump) {
  ConvStem stem;
  if (!conv_stem_load(&stem, gf, bp.backend)) {
    throw std::runtime_error("conv stem load failed");
  }
  AudioEnc enc;
  if (!audio_enc_load(&enc, gf, bp.backend)) {
    throw std::runtime_error("audio encoder load failed");
  }

  AudioTowerConfig tcfg = audio_tower_config_load(gf);
  std::vector<int> chunk_lengths =
      audio_tower_chunk_lengths((int)n_frames, tcfg);
  const int S = audio_tower_seq_len(chunk_lengths);
  const int window_aftercnn =
      audio_tower_conv_out(tcfg.chunk_mel) * tcfg.window_chunks;
  const int d_model = enc.cfg.d_model;

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
  ggml_set_name(mel_in, "tower.mel");
  ggml_set_name(pe_in, "tower.pe");
  ggml_set_name(mask_in, "tower.mask");
  ggml_set_input(mel_in);
  ggml_set_input(pe_in);
  ggml_set_input(mask_in);

  struct ggml_tensor *out =
      audio_tower_build(gctx, stem, enc, mel_in, pe_in, mask_in, chunk_lengths);
  ggml_set_output(out);

  struct ggml_cgraph *graph =
      ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE, false);
  ggml_build_forward_expand(graph, out);

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, graph)) {
    throw std::runtime_error("tower sched alloc failed");
  }
  ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));
  ggml_backend_tensor_set(pe_in, pe.data(), 0, pe.size() * sizeof(float));
  ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(float));
  if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("tower graph compute failed");
  }

  const int64_t out_dim = out->ne[0];
  const int64_t t_out = out->ne[1];
  std::vector<float> tower_out((size_t)out_dim * (size_t)t_out);
  ggml_backend_tensor_get(out, tower_out.data(), 0,
                          tower_out.size() * sizeof(float));

  char path[1024];
  snprintf(path, sizeof(path), "%s/windowed.bin", dump);
  dump_f32(path, tower_out);
  fprintf(stderr, "qwenasr-feat %s: windowed [%lld, %lld] %d chunks -> %s\n",
          qa_version(), (long long)t_out, (long long)out_dim,
          (int)chunk_lengths.size(), path);

  ggml_free(gctx);
  audio_enc_free(&enc);
  conv_stem_free(&stem);
}

static int main_impl(int argc, char **argv) {
  const char *file = nullptr;
  const char *dump = nullptr;
  const char *model = nullptr;
  bool encoder = false;
  bool windowed = false;
  bool pad30 = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
      file = argv[++i];
    } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      dump = argv[++i];
    } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--encoder") == 0) {
      encoder = true;
    } else if (strcmp(argv[i], "--windowed") == 0) {
      windowed = true;
    } else if (strcmp(argv[i], "--pad-30s") == 0) {
      pad30 = true;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", argv[i]);
      print_usage(argv[0]);
      return 2;
    }
  }
  if (!file || !dump) {
    print_usage(argv[0]);
    return 2;
  }
  if (encoder && !model) {
    fprintf(stderr, "[CLI] ERROR: --encoder requires --model\n");
    return 2;
  }
  if (windowed && !model) {
    fprintf(stderr, "[CLI] ERROR: --windowed requires --model\n");
    return 2;
  }

  int n = 0;
  float *pcm = audio_read_mono(file, 16000, &n);
  if (!pcm) {
    fprintf(stderr, "[CLI] ERROR: failed to read audio: %s\n", file);
    return 1;
  }

  std::vector<float> samples;
  if (pad30) {
    const size_t target = 480000;
    samples.assign(target, 0.0f);
    const size_t cnt = (size_t)n < target ? (size_t)n : target;
    for (size_t i = 0; i < cnt; i++) {
      samples[i] = pcm[i];
    }
  } else {
    samples.assign(pcm, pcm + n);
  }
  free(pcm);

  BackendPair bp = backend_init("Audio");
  if (!bp.backend) {
    throw std::runtime_error("no backend");
  }
  ggml_backend_sched_t sched = backend_sched_new(bp, GGML_DEFAULT_GRAPH_SIZE);

  AudioMelConfig cfg;
  size_t n_frames = 0;
  std::vector<float> mel = run_mel(bp, sched, cfg, samples, &n_frames);

  char path[1024];
  snprintf(path, sizeof(path), "%s/mel.bin", dump);
  dump_f32(path, mel);
  fprintf(stderr, "qwenasr-feat %s: mel [%d, %zu] on %s -> %s\n", qa_version(),
          cfg.n_mels, n_frames, ggml_backend_name(bp.backend), path);

  if (model) {
    GGUFModel gf;
    if (!gf_load(&gf, model)) {
      throw std::runtime_error(std::string("cannot load gguf ") + model);
    }
    int64_t d_model = 0;
    int64_t t = 0;
    std::vector<float> stem =
        run_stem(gf, bp, sched, cfg.n_mels, mel, n_frames, dump, &d_model, &t);
    if (encoder) {
      run_encoder(gf, bp, sched, stem, d_model, t, dump);
    }
    if (windowed) {
      run_windowed(gf, bp, sched, cfg.n_mels, mel, n_frames, dump);
    }
    gf_close(&gf);
  }

  ggml_backend_sched_free(sched);
  backend_release(bp.backend, bp.cpu_backend);
  return 0;
}

int main(int argc, char **argv) {
  utf8_init(&argc, &argv);
  try {
    return main_impl(argc, argv);
  } catch (const std::exception &e) {
    fprintf(stderr, "[qwenasr-feat] FATAL: %s\n", e.what());
    return 1;
  }
}
