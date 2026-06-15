#pragma once
// qwenasr.h: public ABI for qwenasr.cpp.
//
// Single-header public API. Pure C99, consumable from C and C++ alike.
// Bindings (Python ctypes, Rust bindgen, Go cgo) parse this file directly.
// Style follows whisper.h / llama.h: extern "C" linkage on every entry,
// POD structs only, const char * UTF-8 strings, qa_status enum returns.
//
// The opaque qa_context handle aggregates every module the transcription
// path needs (mel frontend, conv stem, audio encoder, Qwen3 Thinker
// decoder, BPE tokenizer, GGML backend pair). One init, one free, one
// transcribe call covers the full wav -> text path. Lower level
// pipeline_asr_* entries in pipeline-asr.h stay available for the debug
// paths that need partial init, but they are not part of this public ABI.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(QWENASR_STATIC)
#define QA_API
#elif defined(QWENASR_BUILD)
#define QA_API __declspec(dllexport)
#else
#define QA_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define QA_API __attribute__((visibility("default")))
#else
#define QA_API
#endif

// Struct ABI version. Incremented when a public POD struct grows a field at
// its tail. Callers set .abi_version = QA_ABI_VERSION or let the
// qa_*_default_params helpers fill it. Entries reject inputs whose
// abi_version exceeds the build-time constant.
#define QA_ABI_VERSION 1

// Default decode budget, the Qwen3-ASR reference inference default
// (qwen_asr/inference/qwen3_asr.py). The decoder stops at im_end on its own,
// this only caps runaway generation.
#define QA_DEFAULT_MAX_NEW_TOKENS 512

// Returns "<git-hash> (<date>)" for the exact commit this binary came from.
QA_API const char *qa_version(void);

enum qa_status {
  QA_STATUS_OK = 0,
  QA_STATUS_INVALID_PARAMS = -1,
  QA_STATUS_LOAD_FAILED = -2,
  QA_STATUS_GENERATE_FAILED = -3,
  QA_STATUS_OOM = -4,
  QA_STATUS_CANCELLED = -5,
};

// Thread-local errno-style message, only meaningful right after a failure.
QA_API const char *qa_last_error(void);

// Log level for the redirectable diagnostic stream.
enum qa_log_level {
  QA_LOG_ERROR = 0,
  QA_LOG_WARN,
  QA_LOG_INFO,
  QA_LOG_DEBUG,
};

// Installs a callback receiving every internal diagnostic. NULL routes to
// stderr. user carries caller state.
typedef void (*qa_log_cb)(enum qa_log_level level, const char *msg, void *user);
QA_API void qa_log_set(qa_log_cb cb, void *user);

typedef struct qa_context qa_context;

// Init params. model_path points at the GGUF holding the audio encoder plus
// the Qwen3 Thinker decoder; n_threads 0 lets the backend pick. use_gpu
// selects the first available GPU backend, CPU otherwise.
struct qa_init_params {
  int abi_version;
  const char *model_path;
  int n_threads;
  bool use_gpu;
  bool clamp_fp16; // clamp hidden states to the FP16 range on backends that
                   // need it
  bool flash_attn; // enable flash attention in the graph
};

QA_API struct qa_init_params qa_init_default_params(void);
QA_API qa_context *qa_init(const struct qa_init_params *params);
QA_API void qa_free(qa_context *ctx);

// Streaming token callback. Fires once per decoded text chunk as soon as it
// is ready. Returning false requests cancellation and the call unwinds with
// QA_STATUS_CANCELLED. user carries caller state.
typedef bool (*qa_token_cb)(const char *utf8_chunk, void *user);

// Transcribe params. language is an ISO code or NULL for auto detect.
// detect_language asks the model to emit the detected language tag.
struct qa_transcribe_params {
  int abi_version;
  const char *language;
  bool detect_language;
  int max_new_tokens;
  qa_token_cb on_token;
  void *user;
  const char *context; // optional system text for biasing, NULL for none
  // Sampling. temperature <= 0 is greedy argmax, matching the Qwen3-ASR
  // reference default. top_k 0 and top_p 1.0 disable those filters,
  // repetition_penalty 1.0 is a no-op. seed == -1 draws a hardware random seed.
  float temperature;
  int top_k;
  float top_p;
  float repetition_penalty;
  int64_t seed;
};

QA_API struct qa_transcribe_params qa_transcribe_default_params(void);

// Transcribe mono PCM. samples are f32 in [-1, 1], any sample_rate (resampled
// to the model rate internally). On success out_text receives a freshly
// allocated NUL terminated UTF-8 string the caller releases with qa_free_text.
QA_API enum qa_status qa_transcribe(qa_context *ctx, const float *samples,
                                    size_t n_samples, int sample_rate,
                                    const struct qa_transcribe_params *params,
                                    char **out_text);

QA_API void qa_free_text(char *text);

#ifdef __cplusplus
}
#endif
