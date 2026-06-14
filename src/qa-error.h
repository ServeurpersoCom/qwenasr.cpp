#pragma once
// qa-error.h: internal helpers backing the public qa_last_error() entry
// and the qa_log_set callback routing.
//
// Not part of the public ABI. Translation units that emit user-facing
// errors include this header to record a diagnostic on the calling thread
// before they return a negative qa_status (or NULL). The actual storage
// and the public qa_last_error() reader live in qwenasr.cpp.
//
// Storage is thread_local so concurrent qa_transcribe calls on different
// threads never race on each other's messages. The setter is variadic with
// printf semantics; messages longer than the internal buffer are
// truncated, never split. Passing NULL as fmt clears the slot.
//
// qa_throw is the load-path counterpart: functions deep inside the GGUF
// reader and the audio tokenizer load chain cannot return false up 97 call
// sites without a massive cascade. They throw a std::runtime_error instead,
// which the ABI boundary entries (qa_init, qa_transcribe) catch and
// convert into qa_set_error + a negative qa_status. Exceptions never
// cross the extern "C" boundary, so the public API stays pure C.
//
// qa_log routes a formatted message to the user-installed qa_log_cb, or
// to stderr when no callback is installed. Used by every translation unit
// in the lib that wants its diagnostics to be redirectable from a wrapper
// (Python logging, Rust tracing, ...). The level enum lives in qwenasr.h.

#include "qwenasr.h"

#include <cstdarg>

void qa_set_error(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

void qa_set_error_v(const char *fmt, va_list ap);

// Throws std::runtime_error formatted with printf semantics. Tagged
// noreturn so the compiler can prune unreachable branches at the call
// site. Designed for the GGUF / codec load path where any failure means
// the model is unusable and unwinding to the ABI boundary is the only
// sane recovery.
[[noreturn]] void qa_throw(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

// Routes a formatted message at the requested level to the installed
// qa_log_cb. Defaults to stderr (with a trailing newline) when no
// callback is set, so existing fprintf-style call sites can migrate
// one at a time without changing user-visible behaviour. printf
// semantics; messages longer than the internal buffer are truncated.
void qa_log(enum qa_log_level level, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
