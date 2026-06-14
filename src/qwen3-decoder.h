#pragma once
// qwen3-decoder.h: Qwen3 Thinker decoder, the text side of Qwen3-ASR. It is a
// decoder-only Qwen3 LM that consumes the projected audio encoder states as a
// prefix in the embedding stream, then generates text tokens autoregressively
// with a persistent KV cache. This is the same Qwen3 core used by the sibling
// projects, ported from their LM module, with the audio prefix as the only
// ASR-specific entry point.

struct asr_weights;
#include <vector>

struct ggml_tensor;
struct ggml_context;

// Projects audio states [d_model, n_frames] into the LM embedding space and
// returns the prefix embeddings the decoder prepends before generation.
struct ggml_tensor *
qwen3_decoder_audio_prefix(ggml_context *gctx, const asr_weights &w,
                           struct ggml_tensor *audio_states);

// Single autoregressive step over the KV cache, returns next-token logits.
struct ggml_tensor *qwen3_decoder_step(ggml_context *gctx, const asr_weights &w,
                                       struct ggml_tensor *tok_embd, int pos);
