#pragma once
// conv-stem.h: conv2d subsampling stem of the Qwen3-ASR audio encoder. Treats
// the mel matrix as a single channel image [1, num_mel_bins, n_frames]. Three
// conv2d (kernel 3, stride 2, padding 1) with gelu collapse the frequency axis
// 128 -> 16 and downsample time 8x. The result is reshaped to
// [n_out_frames, downsample_hidden * 16 = 7680] with channel outer, frequency
// inner, then projected by conv_out (Linear, no bias) to d_model 896. Built on
// ggml_conv_2d (im2col + matmul), never col2im (that is the synthesis side).

#include "gguf-weights.h"

#include <cstdio>
#include <string>

struct ConvStem {
  struct ggml_tensor *conv1_w;  // [3, 3, 1, 480]
  struct ggml_tensor *conv1_b;  // [480]
  struct ggml_tensor *conv2_w;  // [3, 3, 480, 480]
  struct ggml_tensor *conv2_b;  // [480]
  struct ggml_tensor *conv3_w;  // [3, 3, 480, 480]
  struct ggml_tensor *conv3_b;  // [480]
  struct ggml_tensor *conv_out; // [7680, 896]

  WeightCtx wctx;
};

// Loads the stem weights as f32 (high precision path for the cossim harness).
static bool conv_stem_load(ConvStem *s, const GGUFModel &gf,
                           ggml_backend_t backend) {
  wctx_init(&s->wctx, 8);
  const std::string p = "thinker.audio.";
  s->conv1_w = gf_load_tensor_f32(&s->wctx, gf, p + "conv1.weight");
  s->conv1_b = gf_load_tensor_f32(&s->wctx, gf, p + "conv1.bias");
  s->conv2_w = gf_load_tensor_f32(&s->wctx, gf, p + "conv2.weight");
  s->conv2_b = gf_load_tensor_f32(&s->wctx, gf, p + "conv2.bias");
  s->conv3_w = gf_load_tensor_f32(&s->wctx, gf, p + "conv3.weight");
  s->conv3_b = gf_load_tensor_f32(&s->wctx, gf, p + "conv3.bias");
  s->conv_out = gf_load_tensor_f32(&s->wctx, gf, p + "conv_out.weight");
  bool ok = wctx_alloc(&s->wctx, backend);
  fprintf(stderr, "[ConvStem] Loaded: downsample_hidden %d, d_model %d\n",
          (int)s->conv1_w->ne[3], (int)s->conv_out->ne[1]);
  return ok;
}

static void conv_stem_free(ConvStem *s) {
  if (s->wctx.buffer) {
    ggml_backend_buffer_free(s->wctx.buffer);
    s->wctx.buffer = nullptr;
  }
  if (s->wctx.ctx) {
    ggml_free(s->wctx.ctx);
    s->wctx.ctx = nullptr;
  }
}

// mel input ne = [n_frames, num_mel_bins, 1, 1] f32. Returns [d_model, n_out].
static struct ggml_tensor *conv_stem_build(struct ggml_context *ctx,
                                           const ConvStem &s,
                                           struct ggml_tensor *mel) {
  struct ggml_tensor *x = mel;

  x = ggml_conv_2d(ctx, s.conv1_w, x, 2, 2, 1, 1, 1, 1);
  x = ggml_add(ctx, x,
               ggml_reshape_4d(ctx, s.conv1_b, 1, 1, s.conv1_b->ne[0], 1));
  x = ggml_gelu_erf(ctx, x);

  x = ggml_conv_2d(ctx, s.conv2_w, x, 2, 2, 1, 1, 1, 1);
  x = ggml_add(ctx, x,
               ggml_reshape_4d(ctx, s.conv2_b, 1, 1, s.conv2_b->ne[0], 1));
  x = ggml_gelu_erf(ctx, x);

  x = ggml_conv_2d(ctx, s.conv3_w, x, 2, 2, 1, 1, 1, 1);
  x = ggml_add(ctx, x,
               ggml_reshape_4d(ctx, s.conv3_b, 1, 1, s.conv3_b->ne[0], 1));
  x = ggml_gelu_erf(ctx, x);

  // x: [t_out, freq=16, chan=480, 1] -> [freq, chan, t_out] -> [7680, t_out]
  // contiguous order puts frequency inner and channel outer, matching the
  // reference permute(0,3,1,2).view(b, t, c * f).
  x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
  const int64_t t_out = x->ne[2];
  x = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], t_out);
  x = ggml_mul_mat(ctx, s.conv_out, x); // [896, t_out]
  return x;
}
