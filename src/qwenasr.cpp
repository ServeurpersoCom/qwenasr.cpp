// qwenasr.cpp: public ABI implementation. Thin layer over pipeline-asr.h.
// Owns the thread-local error slot, the process-wide log callback, the
// version string, and the init / transcribe / free entries. Exceptions from
// the load and decode paths stop here and become a negative qa_status.

#include "qwenasr.h"
#include "pipeline-asr.h"
#include "qa-error.h"
#include "version.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

static thread_local std::string g_last_error;
static qa_log_cb g_log_cb = nullptr;
static void *g_log_user = nullptr;

const char *qa_version(void) {
  // QWENASR_VERSION is a string literal injected by tools/version.cmake.
  return QWENASR_VERSION;
}

const char *qa_last_error(void) { return g_last_error.c_str(); }

void qa_set_error_v(const char *fmt, va_list ap) {
  if (!fmt) {
    g_last_error.clear();
    return;
  }
  char buf[1024];
  vsnprintf(buf, sizeof(buf), fmt, ap);
  g_last_error.assign(buf);
}

void qa_set_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  qa_set_error_v(fmt, ap);
  va_end(ap);
}

void qa_throw(const char *fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  throw std::runtime_error(buf);
}

void qa_log(enum qa_log_level level, const char *fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (g_log_cb) {
    g_log_cb(level, buf, g_log_user);
  } else {
    fprintf(stderr, "%s\n", buf);
  }
}

void qa_log_set(qa_log_cb cb, void *user) {
  g_log_cb = cb;
  g_log_user = user;
}

struct qa_context {
  pipeline_asr *pipe;
};

struct qa_init_params qa_init_default_params(void) {
  struct qa_init_params p;
  p.abi_version = QA_ABI_VERSION;
  p.model_path = nullptr;
  p.n_threads = 0;
  p.use_gpu = true;
  p.clamp_fp16 = false;
  p.flash_attn = true;
  return p;
}

struct qa_transcribe_params qa_transcribe_default_params(void) {
  struct qa_transcribe_params p;
  p.abi_version = QA_ABI_VERSION;
  p.language = nullptr;
  p.detect_language = true;
  p.max_new_tokens = QA_DEFAULT_MAX_NEW_TOKENS;
  p.on_token = nullptr;
  p.user = nullptr;
  p.context = nullptr;
  return p;
}

qa_context *qa_init(const struct qa_init_params *params) {
  if (!params || !params->model_path) {
    qa_set_error("qa_init: model_path is required");
    return nullptr;
  }
  if (params->abi_version > QA_ABI_VERSION) {
    qa_set_error("qa_init: params->abi_version %d > QA_ABI_VERSION %d",
                 params->abi_version, QA_ABI_VERSION);
    return nullptr;
  }
  try {
    pipeline_asr_params pp;
    pp.model_path = params->model_path;
    pp.n_threads = params->n_threads;
    pp.use_gpu = params->use_gpu;
    pp.clamp_fp16 = params->clamp_fp16;
    pp.flash_attn = params->flash_attn;

    qa_context *ctx = new qa_context{pipeline_asr_load(pp)};
    qa_log(QA_LOG_INFO, "[qwenasr] qwenasr.cpp %s", qa_version());
    return ctx;
  } catch (const std::exception &e) {
    qa_set_error("qa_init: %s", e.what());
    return nullptr;
  }
}

void qa_free(qa_context *ctx) {
  if (!ctx) {
    return;
  }
  pipeline_asr_free(ctx->pipe);
  delete ctx;
}

enum qa_status qa_transcribe(qa_context *ctx, const float *samples,
                             size_t n_samples, int sample_rate,
                             const struct qa_transcribe_params *params,
                             char **out_text) {
  if (!ctx || !samples || !out_text) {
    qa_set_error("qa_transcribe: null argument");
    return QA_STATUS_INVALID_PARAMS;
  }
  struct qa_transcribe_params def = qa_transcribe_default_params();
  const struct qa_transcribe_params *p = params ? params : &def;
  if (p->abi_version > QA_ABI_VERSION) {
    qa_set_error("qa_transcribe: params->abi_version %d too new",
                 p->abi_version);
    return QA_STATUS_INVALID_PARAMS;
  }

  std::string text;
  const qa_status rc =
      pipeline_asr_run(ctx->pipe, samples, n_samples, sample_rate, p, text);
  if (rc != QA_STATUS_OK) {
    return rc;
  }

  char *buf = (char *)malloc(text.size() + 1);
  if (!buf) {
    qa_set_error("qa_transcribe: out of memory for %zu bytes", text.size() + 1);
    return QA_STATUS_OOM;
  }
  memcpy(buf, text.c_str(), text.size() + 1);
  *out_text = buf;
  return QA_STATUS_OK;
}

void qa_free_text(char *text) { free(text); }
