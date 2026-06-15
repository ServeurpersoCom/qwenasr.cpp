#pragma once
// audio-enc.h: transformer audio encoder of Qwen3-ASR. Consumes the conv stem
// output [d_model, T] plus a sinusoidal positional input and returns the
// projected audio states [output_dim, T] the decoder reads as prefix
// embeddings. 18 pre-norm layers Whisper style, d_model 896, 14 heads head_dim
// 64, FFN 3584 with gelu, output_dim 1024. Attention is plain bidirectional MHA
// with bias on q/k/v/out, LayerNorm with bias, no RoPE, no q/k norm. A single
// window attends as one dense block, multi window masking lands at the windowed
// pipeline.

#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

struct AudioEncConfig {
  int d_model = 896;
  int n_head = 14;
  int n_layer = 18;
  int ffn = 3584;
  int output_dim = 1024;
  float eps = 1e-5f;
};

struct AudioEncLayer {
  struct ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
  struct ggml_tensor *ln1_w, *ln1_b;
  struct ggml_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b;
  struct ggml_tensor *ln2_w, *ln2_b;
};

struct AudioEnc {
  AudioEncConfig cfg;
  std::vector<AudioEncLayer> layers;
  struct ggml_tensor *ln_post_w, *ln_post_b;
  struct ggml_tensor *proj1_w, *proj1_b;
  struct ggml_tensor *proj2_w, *proj2_b;
  WeightCtx wctx;
};

// Sinusoidal positional table [channels, length] row major, channel inner.
// log_inc = ln(10000) / (channels / 2 - 1), inv[i] = exp(-log_inc * i),
// pe[t, i] = sin(t * inv[i]), pe[t, half + i] = cos(t * inv[i]). Trig in f64 to
// keep the roundoff under the f32 ULP.
static void audio_enc_compute_pe(int channels, int length,
                                 std::vector<float> &out) {
  const int half = channels / 2;
  const double log_inc = std::log(10000.0) / (double)(half - 1);
  std::vector<double> inv((size_t)half);
  for (int i = 0; i < half; i++) {
    inv[(size_t)i] = std::exp(-log_inc * (double)i);
  }
  out.assign((size_t)channels * (size_t)length, 0.0f);
  for (int t = 0; t < length; t++) {
    for (int i = 0; i < half; i++) {
      const double s = (double)t * inv[(size_t)i];
      out[(size_t)t * (size_t)channels + (size_t)i] = (float)std::sin(s);
      out[(size_t)t * (size_t)channels + (size_t)(half + i)] =
          (float)std::cos(s);
    }
  }
}

// Loads the encoder weights as f32 (high precision path for the cossim
// harness).
static bool audio_enc_load(AudioEnc *w, const GGUFModel &gf,
                           ggml_backend_t backend) {
  w->cfg.d_model = (int)gf_get_u32(gf, "qwenasr.audio.d_model");
  w->cfg.n_head = (int)gf_get_u32(gf, "qwenasr.audio.encoder_attention_heads");
  w->cfg.n_layer = (int)gf_get_u32(gf, "qwenasr.audio.encoder_layers");
  w->cfg.ffn = (int)gf_get_u32(gf, "qwenasr.audio.encoder_ffn_dim");
  w->cfg.output_dim = (int)gf_get_u32(gf, "qwenasr.audio.output_dim");
  w->layers.resize((size_t)w->cfg.n_layer);

  wctx_init(&w->wctx, w->cfg.n_layer * 16 + 6);
  const std::string p = "thinker.audio.";
  for (int l = 0; l < w->cfg.n_layer; l++) {
    AudioEncLayer &L = w->layers[l];
    const std::string lp = p + "blk." + std::to_string(l) + ".";
    L.q_w = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_q.weight");
    L.q_b = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_q.bias");
    L.k_w = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_k.weight");
    L.k_b = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_k.bias");
    L.v_w = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_v.weight");
    L.v_b = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_v.bias");
    L.o_w = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_output.weight");
    L.o_b = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_output.bias");
    L.ln1_w = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_norm.weight");
    L.ln1_b = gf_load_tensor_f32(&w->wctx, gf, lp + "attn_norm.bias");
    L.fc1_w = gf_load_tensor_f32(&w->wctx, gf, lp + "ffn_up.weight");
    L.fc1_b = gf_load_tensor_f32(&w->wctx, gf, lp + "ffn_up.bias");
    L.fc2_w = gf_load_tensor_f32(&w->wctx, gf, lp + "ffn_down.weight");
    L.fc2_b = gf_load_tensor_f32(&w->wctx, gf, lp + "ffn_down.bias");
    L.ln2_w = gf_load_tensor_f32(&w->wctx, gf, lp + "ffn_norm.weight");
    L.ln2_b = gf_load_tensor_f32(&w->wctx, gf, lp + "ffn_norm.bias");
  }
  w->ln_post_w = gf_load_tensor_f32(&w->wctx, gf, p + "post_norm.weight");
  w->ln_post_b = gf_load_tensor_f32(&w->wctx, gf, p + "post_norm.bias");
  w->proj1_w = gf_load_tensor_f32(&w->wctx, gf, p + "proj1.weight");
  w->proj1_b = gf_load_tensor_f32(&w->wctx, gf, p + "proj1.bias");
  w->proj2_w = gf_load_tensor_f32(&w->wctx, gf, p + "proj2.weight");
  w->proj2_b = gf_load_tensor_f32(&w->wctx, gf, p + "proj2.bias");
  bool ok = wctx_alloc(&w->wctx, backend);
  fprintf(stderr,
          "[AudioEnc] Loaded: %d layers, d_model %d, heads %d, head_dim %d, "
          "FFN %d, output_dim %d\n",
          w->cfg.n_layer, w->cfg.d_model, w->cfg.n_head,
          w->cfg.d_model / w->cfg.n_head, w->cfg.ffn, w->cfg.output_dim);
  return ok;
}

static void audio_enc_free(AudioEnc *w) {
  if (w->wctx.buffer) {
    ggml_backend_buffer_free(w->wctx.buffer);
    w->wctx.buffer = nullptr;
  }
  if (w->wctx.ctx) {
    ggml_free(w->wctx.ctx);
    w->wctx.ctx = nullptr;
  }
}

// LayerNorm with bias over ne0.
static struct ggml_tensor *audio_enc_ln(struct ggml_context *ctx,
                                        struct ggml_tensor *x,
                                        struct ggml_tensor *g,
                                        struct ggml_tensor *b, float eps) {
  x = ggml_norm(ctx, x, eps);
  x = ggml_mul(ctx, x, g);
  x = ggml_add(ctx, x, b);
  return x;
}

// Linear y = W x + b, weight loaded with ne = [in, out].
static struct ggml_tensor *audio_enc_linear(struct ggml_context *ctx,
                                            struct ggml_tensor *w,
                                            struct ggml_tensor *b,
                                            struct ggml_tensor *x) {
  x = ggml_mul_mat(ctx, w, x);
  x = ggml_add(ctx, x, b);
  return x;
}

// x [d_model, T] is the positional added input sequence, mask is the optional
// additive attention mask [T, T] (NULL for a single full window). Returns the
// projected states [output_dim, T].
static struct ggml_tensor *audio_enc_build(struct ggml_context *ctx,
                                           const AudioEnc &w,
                                           struct ggml_tensor *x,
                                           struct ggml_tensor *mask) {
  const AudioEncConfig &cfg = w.cfg;
  const int head_dim = cfg.d_model / cfg.n_head;
  const float scale = 1.0f / std::sqrt((float)head_dim);
  const int64_t T = x->ne[1];

  for (int l = 0; l < cfg.n_layer; l++) {
    const AudioEncLayer &L = w.layers[l];

    struct ggml_tensor *res = x;
    struct ggml_tensor *h = audio_enc_ln(ctx, x, L.ln1_w, L.ln1_b, cfg.eps);

    struct ggml_tensor *q = audio_enc_linear(ctx, L.q_w, L.q_b, h);
    struct ggml_tensor *k = audio_enc_linear(ctx, L.k_w, L.k_b, h);
    struct ggml_tensor *v = audio_enc_linear(ctx, L.v_w, L.v_b, h);

    q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, cfg.n_head, T), 0,
                     2, 1, 3);
    k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, cfg.n_head, T), 0,
                     2, 1, 3);

    struct ggml_tensor *kq = ggml_mul_mat(ctx, k, q); // [T, T, n_head]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);

    struct ggml_tensor *vt = ggml_cont(
        ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, cfg.n_head, T),
                          1, 2, 0, 3));
    struct ggml_tensor *kqv =
        ggml_mul_mat(ctx, vt, kq);            // [head_dim, T, n_head]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3); // [head_dim, n_head, T]

    struct ggml_tensor *attn = ggml_cont_2d(ctx, kqv, cfg.d_model, T);
    attn = audio_enc_linear(ctx, L.o_w, L.o_b, attn);

    x = ggml_add(ctx, res, attn);

    res = x;
    h = audio_enc_ln(ctx, x, L.ln2_w, L.ln2_b, cfg.eps);
    h = audio_enc_linear(ctx, L.fc1_w, L.fc1_b, h);
    h = ggml_gelu_erf(ctx, h);
    h = audio_enc_linear(ctx, L.fc2_w, L.fc2_b, h);
    x = ggml_add(ctx, res, h);
  }

  x = audio_enc_ln(ctx, x, w.ln_post_w, w.ln_post_b, cfg.eps);
  x = audio_enc_linear(ctx, w.proj1_w, w.proj1_b, x);
  x = ggml_gelu_erf(ctx, x);
  x = audio_enc_linear(ctx, w.proj2_w, w.proj2_b, x);
  return x;
}
