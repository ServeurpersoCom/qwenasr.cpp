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
#include "graph-arena.h"
#include "kv-cache.h"
#include "thinker-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// Node budget for one thinker graph, sizes both the persistent arenas in the
// pipeline and the graph allocated per forward.
static int thinker_graph_max_nodes(int n_layers) { return 48 * n_layers + 256; }

struct ThinkerForwardOutput {
  // Final hidden state for every position [T, hidden] f32 (post final norm).
  // Filled only when read_hidden_host is set (dump path); the decode hot loop
  // reads back the logits alone.
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
// then written into the cache at the rows carried by kv_rows via set_rows: the
// destination positions travel as data, so the graph topology stays identical
// across decode steps and the captured CUDA graph replays without an update.
// The attention reads the padded [0, n_kv_pad) slice. Returns the layer output
// [hidden, T]. use_flash_attn picks the fused kernel (F16 accumulation guarded
// with set_prec F32) or the manual F32 chain. clamp_fp16 inserts
// ggml_clamp(-65504, 65504) on V before attention and on the residual stream.
static struct ggml_tensor *
thinker_layer_forward(struct ggml_context *ctx, const ThinkerWeights *tw,
                      const ThinkerLayer &layer, struct ggml_tensor *x,
                      struct ggml_tensor *positions, struct ggml_tensor *mask,
                      struct ggml_tensor *kv_rows, struct ggml_tensor *k_cache,
                      struct ggml_tensor *v_cache, int T, int n_kv_pad,
                      bool use_flash_attn, bool clamp_fp16,
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

  // Write the T fresh K/V into the cache via set_rows. K and V are
  // [hd, n_kv, T] at this point and the cache lives as [hd, max_T, n_kv] so we
  // permute to [hd, T, n_kv]; the row ids broadcast across the n_kv head dim.
  struct ggml_tensor *k_perm =
      ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, T, n_kv]
  struct ggml_tensor *v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

  ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k_perm, kv_rows));
  ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v_perm, kv_rows));

  // Read the padded [0, n_kv_pad) window for attention. The width stays
  // constant across consecutive decode steps so the CUDA graph executable
  // replays; the mask carries neg inf past the causal context and the cache
  // buffer is zero initialized, so the padded tail contributes nothing.
  struct ggml_tensor *k_full = ggml_view_3d(ctx, k_cache, hd, n_kv_pad, n_kv,
                                            k_cache->nb[1], k_cache->nb[2], 0);
  struct ggml_tensor *v_full = ggml_view_3d(ctx, v_cache, hd, n_kv_pad, n_kv,
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

// Shared core: build the graph, allocate, upload inputs, run, pull the last
// position logits [vocab]. The full final hidden [T, hidden] comes back to
// host only under read_hidden_host (dump path). T tokens are appended to the
// cache starting at n_past. The graph metadata lives in the caller owned
// persistent arena.
static bool thinker_forward_core(const ThinkerWeights *tw, KVCache *kv,
                                 ggml_backend_sched_t sched, GraphArena *arena,
                                 const float *input_embed,
                                 const int32_t *token_ids, int T, int n_past,
                                 bool use_flash_attn, bool clamp_fp16,
                                 bool read_hidden_host,
                                 ThinkerForwardOutput *out) {
  const int hidden = tw->hidden_size;
  const int n_layers = tw->num_hidden_layers;
  const int vocab = tw->vocab_size;
  const int T_full = n_past + T;

  // Attention window rounded up to 256 and clamped to the cache size: fixed
  // shapes over spans of 256 decode steps keep the CUDA graph executable
  // updatable in place. Ternary instead of std::min: windows.h min/max macros
  // break the latter in headers on MSVC.
  const int kv_pad_raw = (int)GGML_PAD(T_full, 256);
  const int n_kv_pad =
      kv_pad_raw < kv->max_seq_len ? kv_pad_raw : kv->max_seq_len;

  const int max_nodes = thinker_graph_max_nodes(n_layers);
  struct ggml_context *gctx = graph_arena_begin(arena);

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
      ggml_new_tensor_2d(gctx, GGML_TYPE_F16, n_kv_pad, T);
  ggml_set_name(pos_in, "positions");
  ggml_set_name(mask_in, "causal_mask");
  ggml_set_input(pos_in);
  ggml_set_input(mask_in);

  // KV write positions as data: identical topology at every step, pure CUDA
  // graph replay across the decode loop.
  struct ggml_tensor *rows_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I64, T);
  ggml_set_name(rows_in, "kv_rows");
  ggml_set_input(rows_in);

  struct ggml_cgraph *gf = ggml_new_graph_custom(gctx, max_nodes, false);

  struct ggml_tensor *h = x_in;
  for (int l = 0; l < n_layers; l++) {
    h = thinker_layer_forward(gctx, tw, tw->layers[(size_t)l], h, pos_in,
                              mask_in, rows_in, kv->k[(size_t)l],
                              kv->v[(size_t)l], T, n_kv_pad, use_flash_attn,
                              clamp_fp16, gf);
  }

  struct ggml_tensor *h_final = ggml_rms_norm(gctx, h, tw->rms_norm_eps);
  h_final = ggml_mul(gctx, h_final, tw->output_norm_w);
  if (read_hidden_host) {
    ggml_set_output(h_final);
  }

  struct ggml_tensor *logits =
      ggml_mul_mat(gctx, tw->output_w, h_final); // [vocab, T]
  ggml_set_output(logits);

  ggml_build_forward_expand(gf, logits);

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    fprintf(stderr, "[ThinkerForward] FATAL: graph allocation failed\n");
    ggml_backend_sched_reset(sched);
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

    std::vector<int64_t> rows((size_t)T);
    for (int i = 0; i < T; i++) {
      rows[(size_t)i] = (int64_t)(n_past + i);
    }
    ggml_backend_tensor_set(rows_in, rows.data(), 0,
                            (size_t)T * sizeof(int64_t));
  }

  // Causal mask [T_q, n_kv_pad], 0 where k <= n_past + q, neg inf otherwise,
  // padded tail included.
  {
    std::vector<ggml_fp16_t> mask((size_t)T * (size_t)n_kv_pad);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (size_t i = 0; i < mask.size(); i++) {
      mask[i] = neg_inf;
    }
    for (int q = 0; q < T; q++) {
      const int q_pos = n_past + q;
      for (int kk = 0; kk <= q_pos; kk++) {
        mask[(size_t)q * (size_t)n_kv_pad + (size_t)kk] = zero;
      }
    }
    ggml_backend_tensor_set(mask_in, mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));
  }

  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    fprintf(stderr, "[ThinkerForward] FATAL: graph compute failed\n");
    ggml_backend_sched_reset(sched);
    return false;
  }

  out->hidden = hidden;
  out->vocab = vocab;
  out->n_tokens = T;
  out->logits_last.assign((size_t)vocab, 0.0f);
  {
    size_t row_bytes = (size_t)vocab * sizeof(float);
    ggml_backend_tensor_get(logits, out->logits_last.data(),
                            (size_t)(T - 1) * row_bytes, row_bytes);
  }
  if (read_hidden_host) {
    out->hidden_all.assign((size_t)T * (size_t)hidden, 0.0f);
    ggml_backend_tensor_get(h_final, out->hidden_all.data(), 0,
                            out->hidden_all.size() * sizeof(float));
  }

  // Advance the cache write head. The arena and the sched allocation persist
  // into the next forward.
  kv->cur_len = T_full;
  return true;
}

// Prefill: reset the cache and write T positions in one shot. input_embed is
// [T, hidden] f32 row-major. read_hidden_host pulls the full final hidden back
// to host for the dump path.
static bool thinker_forward_prefill(const ThinkerWeights *tw, KVCache *kv,
                                    ggml_backend_sched_t sched,
                                    GraphArena *arena, const float *input_embed,
                                    int T, bool use_flash_attn, bool clamp_fp16,
                                    bool read_hidden_host,
                                    ThinkerForwardOutput *out) {
  kv_cache_reset(kv);
  if (T > kv->max_seq_len) {
    fprintf(
        stderr,
        "[ThinkerForward] FATAL: prefill T=%d exceeds cache max_seq_len=%d\n",
        T, kv->max_seq_len);
    return false;
  }
  return thinker_forward_core(tw, kv, sched, arena, input_embed, NULL, T, 0,
                              use_flash_attn, clamp_fp16, read_hidden_host,
                              out);
}

// Decode: feed one token id, gather its embedding on device, append one
// position to the cache, attend to the [0, cur_len + 1) window. The logits
// are the only readback on the hot path.
static bool thinker_forward_decode(const ThinkerWeights *tw, KVCache *kv,
                                   ggml_backend_sched_t sched,
                                   GraphArena *arena, int32_t token_id,
                                   bool use_flash_attn, bool clamp_fp16,
                                   ThinkerForwardOutput *out) {
  if (kv->cur_len + 1 > kv->max_seq_len) {
    fprintf(
        stderr,
        "[ThinkerForward] FATAL: decode would overflow cache (%d + 1 > %d)\n",
        kv->cur_len, kv->max_seq_len);
    return false;
  }
  return thinker_forward_core(tw, kv, sched, arena, NULL, &token_id, 1,
                              kv->cur_len, use_flash_attn, clamp_fp16, false,
                              out);
}
