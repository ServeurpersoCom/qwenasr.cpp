#pragma once
// thinker-forward.h: prefill + decode forwards of the Qwen3 Thinker decoder,
// KV cached. Both entry points run the same Qwen3 decoder stack: pre-norm GQA
// attention with per-head QK-norm, 1D NEOX RoPE, SwiGLU MLP, two residuals per
// layer, then a final RMSNorm and the output projection over the vocab.
//
//   thinker_forward_prefill: feeds a [T, hidden] input embedding, rewinds the
//   KV cache to 0 and writes T positions into it.
//   thinker_forward_decode: feeds a token id, gathers its embedding on device
//   with get_rows, appends one position at index kv->cur_len and attends to the
//   [0, cur_len + 1) window.
//
// RoPE is plain 1D NEOX (rotate_half). The Thinker rope index gives the three
// mrope axes the same position id, so the interleaved mrope collapses exactly
// to this. Input embeddings live as [hidden, T] in the graph, fed from a
// [T, hidden] f32 row-major host buffer (rows contiguous on the hidden axis).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "kv-cache.h"
#include "thinker-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct ThinkerForwardOutput {
  // Final hidden state for every position [T, hidden] f32 (post final norm).
  std::vector<float> hidden_all;

  // Output logits for the last position [vocab] f32.
  std::vector<float> logits_last;

  int hidden;
  int vocab;
  int n_tokens;
};

// Manual F32 attention chain (use_flash_attn false): GQA scaled dot product
// with explicit mul_mat / soft_max_ext / mul_mat, F32 accumulators end to end.
// q [hd, T, n_q_heads], k/v [hd, T_full, n_kv], output [hd, n_q_heads, T].
static struct ggml_tensor *
thinker_attn_f32(struct ggml_context *ctx, struct ggml_tensor *q,
                 struct ggml_tensor *k, struct ggml_tensor *v,
                 struct ggml_tensor *mask, float scale) {
  struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);
  scores = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
  struct ggml_tensor *vt = ggml_cont(ctx, ggml_transpose(ctx, v));
  struct ggml_tensor *out = ggml_mul_mat(ctx, vt, scores);
  return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// Per-layer block, KV cached. K and V for the T fresh positions get computed,
// written into the cache at [n_past, n_past + T) on dim 1, and the attention
// reads the contiguous [0, n_past + T) slice. Returns the layer output
// [hidden, T]. use_flash_attn picks the fused kernel (F16 accumulation guarded
// with set_prec F32) or the manual F32 chain. clamp_fp16 inserts
// ggml_clamp(-65504, 65504) on V before attention and on the residual stream.
static struct ggml_tensor *
thinker_layer_forward(struct ggml_context *ctx, const ThinkerWeights *tw,
                      const ThinkerLayer &layer, struct ggml_tensor *x,
                      struct ggml_tensor *positions, struct ggml_tensor *mask,
                      struct ggml_tensor *k_cache, struct ggml_tensor *v_cache,
                      int n_past, int T, bool use_flash_attn, bool clamp_fp16,
                      struct ggml_cgraph *gf) {
  const int n_q_heads = tw->num_attention_heads;
  const int n_kv = tw->num_key_value_heads;
  const int hd = tw->head_dim;
  const float eps = tw->rms_norm_eps;

  struct ggml_tensor *h = ggml_rms_norm(ctx, x, eps);
  h = ggml_mul(ctx, h, layer.input_norm_w);

  struct ggml_tensor *q =
      ggml_mul_mat(ctx, layer.attn.q_proj_w, h); // [n_q_heads*hd, T]
  struct ggml_tensor *k =
      ggml_mul_mat(ctx, layer.attn.k_proj_w, h); // [n_kv*hd, T]
  struct ggml_tensor *v =
      ggml_mul_mat(ctx, layer.attn.v_proj_w, h); // [n_kv*hd, T]

  q = ggml_reshape_3d(ctx, q, hd, n_q_heads, T); // [hd, n_q_heads, T]
  k = ggml_reshape_3d(ctx, k, hd, n_kv, T);
  v = ggml_reshape_3d(ctx, v, hd, n_kv, T);

  // Per-head QK-norm: RMS over hd, then a [hd] gain. Same path for q and k.
  q = ggml_rms_norm(ctx, q, eps);
  q = ggml_mul(ctx, q, layer.attn.q_norm_w);
  k = ggml_rms_norm(ctx, k, eps);
  k = ggml_mul(ctx, k, layer.attn.k_norm_w);

  q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0,
                    tw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0,
                    tw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

  // Write the T fresh K/V into the cache at dim 1 offset n_past.
  struct ggml_tensor *k_perm =
      ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, T, n_kv]
  struct ggml_tensor *v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

  size_t k_off = (size_t)n_past * k_cache->nb[1];
  size_t v_off = (size_t)n_past * v_cache->nb[1];

  struct ggml_tensor *k_dst = ggml_view_3d(
      ctx, k_cache, hd, T, n_kv, k_cache->nb[1], k_cache->nb[2], k_off);
  struct ggml_tensor *v_dst = ggml_view_3d(
      ctx, v_cache, hd, T, n_kv, v_cache->nb[1], v_cache->nb[2], v_off);

  ggml_build_forward_expand(gf, ggml_cpy(ctx, k_perm, k_dst));
  ggml_build_forward_expand(gf, ggml_cpy(ctx, v_perm, v_dst));

  // Read the [0, n_past + T) window for attention.
  const int T_full = n_past + T;
  struct ggml_tensor *k_full = ggml_view_3d(ctx, k_cache, hd, T_full, n_kv,
                                            k_cache->nb[1], k_cache->nb[2], 0);
  struct ggml_tensor *v_full = ggml_view_3d(ctx, v_cache, hd, T_full, n_kv,
                                            v_cache->nb[1], v_cache->nb[2], 0);

  struct ggml_tensor *q_p =
      ggml_permute(ctx, q, 0, 2, 1, 3); // [hd, T, n_q_heads]

  if (clamp_fp16) {
    v_full = ggml_clamp(ctx, v_full, -65504.0f, 65504.0f);
  }

  float scale = 1.0f / sqrtf((float)hd);
  struct ggml_tensor *attn;
  if (use_flash_attn) {
    attn =
        ggml_flash_attn_ext(ctx, q_p, k_full, v_full, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
  } else {
    attn = thinker_attn_f32(ctx, q_p, k_full, v_full, mask, scale);
  }

  attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);
  struct ggml_tensor *o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);

  x = ggml_add(ctx, x, o);
  if (clamp_fp16) {
    x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
  }

  struct ggml_tensor *h2 = ggml_rms_norm(ctx, x, eps);
  h2 = ggml_mul(ctx, h2, layer.post_attn_norm_w);

  struct ggml_tensor *gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, h2);
  struct ggml_tensor *up = ggml_mul_mat(ctx, layer.mlp.up_proj_w, h2);
  gate = ggml_silu(ctx, gate);
  struct ggml_tensor *gu = ggml_mul(ctx, gate, up);
  struct ggml_tensor *mlp = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

  x = ggml_add(ctx, x, mlp);
  if (clamp_fp16) {
    x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
  }
  return x;
}

// Shared core: build the graph, allocate, upload inputs, run, pull the full
// final hidden [T, hidden] and the last position logits [vocab]. T tokens are
// appended to the cache starting at n_past.
static bool thinker_forward_core(const ThinkerWeights *tw, KVCache *kv,
                                 ggml_backend_sched_t sched,
                                 const float *input_embed,
                                 const int32_t *token_ids, int T, int n_past,
                                 bool use_flash_attn, bool clamp_fp16,
                                 ThinkerForwardOutput *out) {
  const int hidden = tw->hidden_size;
  const int n_layers = tw->num_hidden_layers;
  const int vocab = tw->vocab_size;
  const int T_full = n_past + T;

  const int max_nodes = 48 * n_layers + 256;
  const size_t graph_arena_bytes = ggml_tensor_overhead() * max_nodes +
                                   ggml_graph_overhead_custom(max_nodes, false);

  struct ggml_init_params gparams = {graph_arena_bytes, NULL, true};
  struct ggml_context *gctx = ggml_init(gparams);
  if (!gctx) {
    fprintf(stderr, "[ThinkerForward] FATAL: ggml_init failed\n");
    return false;
  }

  // x_in is either an uploaded [hidden, T] embedding (prefill, audio states
  // spliced in) or get_rows(token_embd, ids) gathered on device (decode, no
  // host roundtrip). token_ids selects the decode path.
  struct ggml_tensor *ids_in = NULL;
  struct ggml_tensor *x_in;
  if (token_ids) {
    ids_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(ids_in, "token_ids");
    ggml_set_input(ids_in);
    x_in = ggml_get_rows(gctx, tw->token_embd, ids_in);
  } else {
    x_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hidden, T);
    ggml_set_name(x_in, "input_embed");
    ggml_set_input(x_in);
  }

  struct ggml_tensor *pos_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
  struct ggml_tensor *mask_in =
      ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T_full, T);
  ggml_set_name(pos_in, "positions");
  ggml_set_name(mask_in, "causal_mask");
  ggml_set_input(pos_in);
  ggml_set_input(mask_in);

  struct ggml_cgraph *gf = ggml_new_graph_custom(gctx, max_nodes, false);

  struct ggml_tensor *h = x_in;
  for (int l = 0; l < n_layers; l++) {
    h = thinker_layer_forward(gctx, tw, tw->layers[(size_t)l], h, pos_in,
                              mask_in, kv->k[(size_t)l], kv->v[(size_t)l],
                              n_past, T, use_flash_attn, clamp_fp16, gf);
  }

  struct ggml_tensor *h_final = ggml_rms_norm(gctx, h, tw->rms_norm_eps);
  h_final = ggml_mul(gctx, h_final, tw->output_norm_w);
  ggml_set_output(h_final);

  struct ggml_tensor *logits =
      ggml_mul_mat(gctx, tw->output_w, h_final); // [vocab, T]
  ggml_set_output(logits);

  ggml_build_forward_expand(gf, h_final);
  ggml_build_forward_expand(gf, logits);

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    fprintf(stderr, "[ThinkerForward] FATAL: graph allocation failed\n");
    ggml_backend_sched_reset(sched);
    ggml_free(gctx);
    return false;
  }

  if (token_ids) {
    ggml_backend_tensor_set(ids_in, token_ids, 0, (size_t)T * sizeof(int32_t));
  } else {
    ggml_backend_tensor_set(x_in, input_embed, 0,
                            (size_t)T * (size_t)hidden * sizeof(float));
  }

  {
    std::vector<int32_t> pos((size_t)T);
    for (int i = 0; i < T; i++) {
      pos[(size_t)i] = n_past + i;
    }
    ggml_backend_tensor_set(pos_in, pos.data(), 0, (size_t)T * sizeof(int32_t));
  }

  // Causal mask [T_q, T_k], 0 where k <= n_past + q, -inf otherwise.
  {
    std::vector<ggml_fp16_t> mask((size_t)T * (size_t)T_full);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (size_t i = 0; i < mask.size(); i++) {
      mask[i] = neg_inf;
    }
    for (int q = 0; q < T; q++) {
      const int q_pos = n_past + q;
      for (int kk = 0; kk <= q_pos; kk++) {
        mask[(size_t)q * (size_t)T_full + (size_t)kk] = zero;
      }
    }
    ggml_backend_tensor_set(mask_in, mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));
  }

  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    fprintf(stderr, "[ThinkerForward] FATAL: graph compute failed\n");
    ggml_backend_sched_reset(sched);
    ggml_free(gctx);
    return false;
  }

  out->hidden = hidden;
  out->vocab = vocab;
  out->n_tokens = T;
  out->hidden_all.assign((size_t)T * (size_t)hidden, 0.0f);
  out->logits_last.assign((size_t)vocab, 0.0f);
  ggml_backend_tensor_get(h_final, out->hidden_all.data(), 0,
                          out->hidden_all.size() * sizeof(float));
  {
    size_t row_bytes = (size_t)vocab * sizeof(float);
    ggml_backend_tensor_get(logits, out->logits_last.data(),
                            (size_t)(T - 1) * row_bytes, row_bytes);
  }

  kv->cur_len = T_full;

  ggml_backend_sched_reset(sched);
  ggml_free(gctx);
  return true;
}

// Prefill: reset the cache and write T positions in one shot. input_embed is
// [T, hidden] f32 row-major.
static bool thinker_forward_prefill(const ThinkerWeights *tw, KVCache *kv,
                                    ggml_backend_sched_t sched,
                                    const float *input_embed, int T,
                                    bool use_flash_attn, bool clamp_fp16,
                                    ThinkerForwardOutput *out) {
  kv_cache_reset(kv);
  if (T > kv->max_seq_len) {
    fprintf(
        stderr,
        "[ThinkerForward] FATAL: prefill T=%d exceeds cache max_seq_len=%d\n",
        T, kv->max_seq_len);
    return false;
  }
  return thinker_forward_core(tw, kv, sched, input_embed, NULL, T, 0,
                              use_flash_attn, clamp_fp16, out);
}

// Decode: feed one token id, gather its embedding on device, append one
// position to the cache, attend to the [0, cur_len + 1) window.
static bool thinker_forward_decode(const ThinkerWeights *tw, KVCache *kv,
                                   ggml_backend_sched_t sched, int32_t token_id,
                                   bool use_flash_attn, bool clamp_fp16,
                                   ThinkerForwardOutput *out) {
  if (kv->cur_len + 1 > kv->max_seq_len) {
    fprintf(
        stderr,
        "[ThinkerForward] FATAL: decode would overflow cache (%d + 1 > %d)\n",
        kv->cur_len, kv->max_seq_len);
    return false;
  }
  return thinker_forward_core(tw, kv, sched, NULL, &token_id, 1, kv->cur_len,
                              use_flash_attn, clamp_fp16, out);
}
