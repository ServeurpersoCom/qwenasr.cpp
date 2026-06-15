// qwenasr-thinker: staged debug for the thinker decoder cossim. Two input
// modes:
//   --embeds <bin>  feed an input embedding [T, hidden] f32 directly.
//   --ids <bin>     look up token_embd via get_rows for the int32 ids, and with
//                   --audio <bin> --audio-pos <p> splice the audio states
//                   [S, hidden] into rows [p, p + S).
// Runs a KV cached prefill and dumps the final hidden [T, hidden] and the last
// token logits [vocab].

#include "backend.h"
#include "kv-cache.h"
#include "qwenasr.h"
#include "thinker-forward.h"
#include "thinker-weights.h"
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
      "Staged decoder dump for the cossim harness. Feeds the Qwen3 Thinker\n"
      "decoder one prefill and writes the final hidden state and last token\n"
      "logits so the Python reference can compare them. Two input modes:\n\n"
      "Usage: %s --model <gguf> --tokens <T> --dump <dir>\n"
      "          (--embeds <bin> | --ids <bin> [--audio <bin> --audio-pos "
      "<p>]) "
      "[--no-fa]\n\n"
      "  --model <gguf>     Qwen3-ASR GGUF\n"
      "  --tokens <T>       Sequence length of the input\n"
      "  --dump <dir>       Existing directory for hidden.bin and logits.bin\n"
      "  --embeds <bin>     Feed an input embedding [T, hidden] f32 directly\n"
      "  --ids <bin>        Look up token_embd via get_rows for int32 ids\n"
      "  --audio <bin>      Audio states [S, hidden] f32 to splice (with "
      "--ids)\n"
      "  --audio-pos <p>    Row where the audio states splice in, range [p, p "
      "+ "
      "S)\n"
      "  --no-fa            Disable flash attention, use the manual F32 "
      "path\n"
      "  --clamp-fp16       Clamp hidden states to FP16 range\n\n"
      "Dumps: hidden.bin [T, hidden], logits.bin [vocab].\n",
      prog);
}

static std::vector<float> read_f32(const char *path, size_t count) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  std::vector<float> v(count);
  size_t got = fread(v.data(), sizeof(float), count, f);
  fclose(f);
  if (got != count) {
    throw std::runtime_error(std::string("short read on ") + path);
  }
  return v;
}

static std::vector<float> read_f32_all(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<float> v((size_t)n / sizeof(float));
  size_t got = fread(v.data(), sizeof(float), v.size(), f);
  fclose(f);
  if (got != v.size()) {
    throw std::runtime_error(std::string("short read on ") + path);
  }
  return v;
}

static std::vector<int32_t> read_i32(const char *path, size_t count) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  std::vector<int32_t> v(count);
  size_t got = fread(v.data(), sizeof(int32_t), count, f);
  fclose(f);
  if (got != count) {
    throw std::runtime_error(std::string("short read on ") + path);
  }
  return v;
}

static void dump_f32(const char *path, const std::vector<float> &v) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  fwrite(v.data(), sizeof(float), v.size(), f);
  fclose(f);
}

// Look up token_embd for the ids via get_rows. Returns [T, hidden] host f32.
static std::vector<float> embed_tokens(const ThinkerWeights &tw,
                                       ggml_backend_sched_t sched,
                                       const std::vector<int32_t> &ids) {
  const int T = (int)ids.size();
  const int hidden = tw.hidden_size;

  const size_t mem =
      ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false);
  struct ggml_init_params gp = {mem, nullptr, true};
  struct ggml_context *gctx = ggml_init(gp);

  struct ggml_tensor *ids_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
  ggml_set_name(ids_in, "ids");
  ggml_set_input(ids_in);
  struct ggml_tensor *emb =
      ggml_get_rows(gctx, tw.token_embd, ids_in); // [hidden, T]
  ggml_set_output(emb);

  struct ggml_cgraph *gf = ggml_new_graph_custom(gctx, 16, false);
  ggml_build_forward_expand(gf, emb);

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    throw std::runtime_error("embed sched alloc failed");
  }
  ggml_backend_tensor_set(ids_in, ids.data(), 0, (size_t)T * sizeof(int32_t));
  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("embed graph compute failed");
  }

  std::vector<float> emb_host((size_t)T * (size_t)hidden);
  ggml_backend_tensor_get(emb, emb_host.data(), 0,
                          emb_host.size() * sizeof(float));
  ggml_free(gctx);
  return emb_host;
}

static int main_impl(int argc, char **argv) {
  const char *model = nullptr;
  const char *embeds = nullptr;
  const char *ids_path = nullptr;
  const char *audio = nullptr;
  const char *dump = nullptr;
  int T = 0;
  int audio_pos = 0;
  bool no_fa = false;
  bool clamp = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--embeds") == 0 && i + 1 < argc) {
      embeds = argv[++i];
    } else if (strcmp(argv[i], "--ids") == 0 && i + 1 < argc) {
      ids_path = argv[++i];
    } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
      audio = argv[++i];
    } else if (strcmp(argv[i], "--audio-pos") == 0 && i + 1 < argc) {
      audio_pos = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
      T = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      dump = argv[++i];
    } else if (strcmp(argv[i], "--no-fa") == 0) {
      no_fa = true;
    } else if (strcmp(argv[i], "--clamp-fp16") == 0) {
      clamp = true;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", argv[i]);
      print_usage(argv[0]);
      return 2;
    }
  }
  if (!model || !dump || T <= 0 || (!embeds && !ids_path)) {
    print_usage(argv[0]);
    return 2;
  }

  BackendPair bp = backend_init("Thinker");
  if (!bp.backend) {
    throw std::runtime_error("no backend");
  }
  ggml_backend_sched_t sched = backend_sched_new(bp, GGML_DEFAULT_GRAPH_SIZE);

  GGUFModel gf;
  if (!gf_load(&gf, model)) {
    throw std::runtime_error(std::string("cannot load gguf ") + model);
  }
  ThinkerWeights tw;
  if (!thinker_weights_load(&tw, gf, bp.backend)) {
    throw std::runtime_error("thinker load failed");
  }
  const int hidden = tw.hidden_size;

  std::vector<float> embed;
  if (embeds) {
    embed = read_f32(embeds, (size_t)T * (size_t)hidden);
  } else {
    std::vector<int32_t> ids = read_i32(ids_path, (size_t)T);
    embed = embed_tokens(tw, sched, ids);
    if (audio) {
      std::vector<float> states = read_f32_all(audio);
      const int S = (int)(states.size() / (size_t)hidden);
      if (audio_pos < 0 || audio_pos + S > T) {
        throw std::runtime_error("audio splice out of range");
      }
      memcpy(embed.data() + (size_t)audio_pos * (size_t)hidden, states.data(),
             (size_t)S * (size_t)hidden * sizeof(float));
      fprintf(stderr, "qwenasr-thinker %s: spliced %d audio states at pos %d\n",
              qa_version(), S, audio_pos);
    }
  }

  KVCache kv;
  if (!kv_cache_init(&kv, tw.num_hidden_layers, tw.num_key_value_heads,
                     tw.head_dim, T, bp.backend)) {
    throw std::runtime_error("kv cache init failed");
  }

  ThinkerForwardOutput out;
  if (!thinker_forward_prefill(&tw, &kv, sched, embed.data(), T, !no_fa, clamp,
                               &out)) {
    throw std::runtime_error("prefill failed");
  }

  char path[1024];
  snprintf(path, sizeof(path), "%s/hidden.bin", dump);
  dump_f32(path, out.hidden_all);
  snprintf(path, sizeof(path), "%s/logits.bin", dump);
  dump_f32(path, out.logits_last);
  fprintf(stderr, "qwenasr-thinker %s: hidden [%d, %d] logits [%d] on %s\n",
          qa_version(), T, hidden, out.vocab, ggml_backend_name(bp.backend));

  kv_cache_free(&kv);
  thinker_weights_free(&tw);
  gf_close(&gf);
  ggml_backend_sched_free(sched);
  backend_release(bp.backend, bp.cpu_backend);
  return 0;
}

int main(int argc, char **argv) {
  utf8_init(&argc, &argv);
  try {
    return main_impl(argc, argv);
  } catch (const std::exception &e) {
    fprintf(stderr, "[qwenasr-thinker] FATAL: %s\n", e.what());
    return 1;
  }
}
