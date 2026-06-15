// asr-server: OpenAI compatible HTTP server exposing /v1/audio/transcriptions
// over the public ABI. No SSL, meant to sit behind a reverse proxy. One
// transcription context lives GPU resident for the process lifetime and is
// serialized FIFO across connections.
//
// Endpoints:
//   POST /v1/audio/transcriptions  OAI speech-to-text, multipart/form-data
//   GET  /v1/models                single loaded model
//   GET  /health                   liveness probe
//
// Request fields (multipart): file (audio, required), model, language, prompt
// (biasing context), response_format ("json" default or "text"). Response is
// {"text": "..."} for json, the raw transcript for text.

#include "audio-io.h"
#include "httplib.h"
#include "qwenasr.h"
#include "utf8.h"
#include "version.h"
#include "yyjson.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

// Mel frontend rate, every upload is resampled to it.
static const int kModelSampleRate = 16000;

struct server_config {
  std::string host = "127.0.0.1";
  int port = 8090;
  std::string model_path;
  std::string language; // default language hint, empty for auto detect
  bool clamp_fp16 = false;
  bool flash_attn = true;
};

// Single GPU context: transcription is serialized FIFO across connections.
static std::mutex g_infer_mutex;
static httplib::Server *g_svr = nullptr;
static qa_context *g_ctx = nullptr;
static std::string g_model_id;

static void asr_on_signal(int) {
  if (g_svr) {
    g_svr->stop();
  }
}

// Write a JSON error body in the OAI error envelope and set the status.
static void asr_json_error(httplib::Response &res, int status, const char *type,
                           const char *message) {
  yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
  yyjson_mut_val *root = yyjson_mut_obj(doc);
  yyjson_mut_doc_set_root(doc, root);
  yyjson_mut_val *err = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_str(doc, err, "message", message);
  yyjson_mut_obj_add_str(doc, err, "type", type);
  yyjson_mut_obj_add_val(doc, root, "error", err);
  char *json = yyjson_mut_write(doc, 0, NULL);
  res.status = status;
  res.set_content(json ? json : "{}", "application/json");
  if (json) {
    free(json);
  }
  yyjson_mut_doc_free(doc);
}

// Emit {"text": text} as the OAI transcription json response.
static void asr_json_text(httplib::Response &res, const std::string &text) {
  yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
  yyjson_mut_val *root = yyjson_mut_obj(doc);
  yyjson_mut_doc_set_root(doc, root);
  yyjson_mut_obj_add_strn(doc, root, "text", text.c_str(), text.size());
  char *json = yyjson_mut_write(doc, 0, NULL);
  res.set_content(json ? json : "{}", "application/json");
  if (json) {
    free(json);
  }
  yyjson_mut_doc_free(doc);
}

// Read a multipart text field, empty when absent.
static std::string asr_field(const httplib::Request &req, const char *name) {
  if (req.form.has_field(name)) {
    return req.form.get_field(name);
  }
  return std::string();
}

static int asr_status_to_http(qa_status rc) {
  if (rc == QA_STATUS_OK) {
    return 200;
  }
  if (rc == QA_STATUS_INVALID_PARAMS) {
    return 400;
  }
  return 502;
}

static void asr_handle_transcriptions(const server_config &cfg,
                                      const httplib::Request &req,
                                      httplib::Response &res) {
  if (!req.form.has_file("file")) {
    asr_json_error(res, 400, "invalid_request_error",
                   "missing 'file' field with the audio to transcribe");
    return;
  }
  const httplib::FormData file = req.form.get_file("file");

  int n_samples = 0;
  float *pcm =
      audio_read_mono_buf((const uint8_t *)file.content.data(),
                          file.content.size(), kModelSampleRate, &n_samples);
  if (!pcm || n_samples <= 0) {
    free(pcm);
    asr_json_error(res, 400, "invalid_request_error",
                   "cannot decode audio, expects a WAV upload");
    return;
  }

  const std::string language = asr_field(req, "language");
  const std::string prompt = asr_field(req, "prompt");
  std::string format = asr_field(req, "response_format");
  if (format.empty()) {
    format = "json";
  }

  struct qa_transcribe_params tp = qa_transcribe_default_params();
  tp.language = !language.empty() ? language.c_str() : cfg.language.c_str();
  tp.context = !prompt.empty() ? prompt.c_str() : nullptr;

  char *text = nullptr;
  qa_status rc;
  {
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    rc = qa_transcribe(g_ctx, pcm, (size_t)n_samples, kModelSampleRate, &tp,
                       &text);
  }
  free(pcm);

  if (rc != QA_STATUS_OK) {
    asr_json_error(res, asr_status_to_http(rc), "server_error",
                   qa_last_error());
    return;
  }

  if (format == "text") {
    res.set_content(text, "text/plain; charset=utf-8");
  } else {
    asr_json_text(res, text);
  }
  qa_free_text(text);
}

static void asr_handle_models(const httplib::Request &,
                              httplib::Response &res) {
  yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
  yyjson_mut_val *root = yyjson_mut_obj(doc);
  yyjson_mut_doc_set_root(doc, root);
  yyjson_mut_obj_add_str(doc, root, "object", "list");
  yyjson_mut_val *data = yyjson_mut_arr(doc);
  yyjson_mut_val *one = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_str(doc, one, "id", g_model_id.c_str());
  yyjson_mut_obj_add_str(doc, one, "object", "model");
  yyjson_mut_obj_add_str(doc, one, "owned_by", "local");
  yyjson_mut_arr_add_val(data, one);
  yyjson_mut_obj_add_val(doc, root, "data", data);
  char *json = yyjson_mut_write(doc, 0, NULL);
  res.set_content(json ? json : "{}", "application/json");
  if (json) {
    free(json);
  }
  yyjson_mut_doc_free(doc);
}

static void asr_handle_health(const httplib::Request &,
                              httplib::Response &res) {
  res.set_content("{\"status\":\"ok\"}", "application/json");
}

static int asr_server_run(const server_config &cfg) {
  httplib::Server svr;
  g_svr = &svr;

  // read covers a multipart upload, write is short text out.
  svr.set_read_timeout(120);
  svr.set_write_timeout(60);

  // accept sizeable audio uploads.
  svr.set_payload_max_length(128 * 1024 * 1024);

  svr.set_tcp_nodelay(true);
  svr.set_socket_options([](socket_t sock) {
    int one = 1;
#ifdef _WIN32
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
#else
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
  });

  // permissive CORS so a browser client can call the API directly.
  svr.set_default_headers({{"Access-Control-Allow-Origin", "*"}});
  svr.Options("/.*", [](const httplib::Request &, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
  });

  svr.Post("/v1/audio/transcriptions",
           [&cfg](const httplib::Request &req, httplib::Response &res) {
             asr_handle_transcriptions(cfg, req, res);
           });
  svr.Get("/v1/models", asr_handle_models);
  svr.Get("/health", asr_handle_health);

  signal(SIGINT, asr_on_signal);
  signal(SIGTERM, asr_on_signal);

  fprintf(stderr, "[Server] model %s\n", g_model_id.c_str());
  fprintf(stderr, "[Server] listening on %s:%d\n", cfg.host.c_str(), cfg.port);
  if (!svr.listen(cfg.host.c_str(), cfg.port)) {
    fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", cfg.host.c_str(),
            cfg.port);
    return 1;
  }
  return 0;
}

static void print_usage(const char *prog) {
  fprintf(stderr, "qwenasr.cpp %s\n\n", QWENASR_VERSION);
  fprintf(
      stderr,
      "OpenAI-compatible transcription server (/v1/audio/transcriptions).\n\n"
      "Usage: %s --model <gguf> [options]\n\n"
      "Required:\n"
      "  --model <gguf>     Qwen3-ASR GGUF\n\n"
      "Optional:\n"
      "  --host <addr>      Bind address (default: 127.0.0.1)\n"
      "  --port <n>         Bind port (default: 8090)\n"
      "  --lang <code>      Default language hint (default: auto detect)\n"
      "  --clamp-fp16       Clamp hidden states to FP16 range\n"
      "  --no-fa            Disable flash attention\n",
      prog);
}

static int main_impl(int argc, char **argv) {
  server_config cfg;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      cfg.model_path = argv[++i];
    } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
      cfg.host = argv[++i];
    } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      cfg.port = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
      cfg.language = argv[++i];
    } else if (strcmp(argv[i], "--clamp-fp16") == 0) {
      cfg.clamp_fp16 = true;
    } else if (strcmp(argv[i], "--no-fa") == 0) {
      cfg.flash_attn = false;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", argv[i]);
      print_usage(argv[0]);
      return 2;
    }
  }
  if (cfg.model_path.empty()) {
    print_usage(argv[0]);
    return 2;
  }

  struct qa_init_params ip = qa_init_default_params();
  ip.model_path = cfg.model_path.c_str();
  ip.clamp_fp16 = cfg.clamp_fp16;
  ip.flash_attn = cfg.flash_attn;

  g_ctx = qa_init(&ip);
  if (!g_ctx) {
    fprintf(stderr, "[CLI] ERROR: qa_init: %s\n", qa_last_error());
    return 1;
  }
  g_model_id = cfg.model_path;

  const int rc = asr_server_run(cfg);
  qa_free(g_ctx);
  g_ctx = nullptr;
  return rc;
}

int main(int argc, char **argv) {
  utf8_init(&argc, &argv);
  try {
    return main_impl(argc, argv);
  } catch (const std::exception &e) {
    fprintf(stderr, "[asr-server] FATAL: %s\n", e.what());
    return 1;
  }
}
