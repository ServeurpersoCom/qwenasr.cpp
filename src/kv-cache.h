#pragma once
// kv-cache.h: persistent per-layer KV cache for the Qwen3 Thinker decoder.
// Mirrors the standard llama.cpp ring approach but kept minimal: the cache
// is sized at init for a fixed max sequence length and never reallocates.
// Reset just rewinds cur_len to 0.
//
// Layout per layer, both K and V :
//   ggml_tensor [hd, max_seq_len, n_kv] f32, contiguous on hd
// This matches the layout the attention block already uses for K, so
// the write path is a ggml_cpy of the freshly RoPE'd K into a view of
// the cache, and the read path is just a view spanning [0, cur_len+T)
// on dim 1. V uses the same layout and gets permuted to [T, n_kv, hd]
// at read time for the value matmul, identical to the prefill path.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct KVCache {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    int cur_len;

    // One pair per layer, both tensors live in `buffer` allocated below.
    std::vector<struct ggml_tensor *> k;
    std::vector<struct ggml_tensor *> v;

    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

// Allocate a fresh KV cache backed by a dedicated buffer on `backend`.
// Tensors are zero initialised (the attention path never reads past
// cur_len so this is mostly cosmetic, but it keeps debug dumps clean).
static bool kv_cache_init(KVCache *      kv,
                          int            n_layers,
                          int            n_kv_heads,
                          int            head_dim,
                          int            max_seq_len,
                          ggml_backend_t backend) {
    kv->n_layers    = n_layers;
    kv->n_kv_heads  = n_kv_heads;
    kv->head_dim    = head_dim;
    kv->max_seq_len = max_seq_len;
    kv->cur_len     = 0;
    kv->k.assign((size_t) n_layers, NULL);
    kv->v.assign((size_t) n_layers, NULL);

    struct ggml_init_params gp = {
        ggml_tensor_overhead() * (size_t) (2 * n_layers + 4),
        NULL,
        true,
    };
    kv->ctx = ggml_init(gp);
    if (!kv->ctx) {
        fprintf(stderr, "[KVCache] FATAL: ggml_init failed\n");
        return false;
    }

    for (int l = 0; l < n_layers; l++) {
        kv->k[(size_t) l] = ggml_new_tensor_3d(kv->ctx, GGML_TYPE_F32, head_dim, max_seq_len, n_kv_heads);
        kv->v[(size_t) l] = ggml_new_tensor_3d(kv->ctx, GGML_TYPE_F32, head_dim, max_seq_len, n_kv_heads);
        char name[64];
        snprintf(name, sizeof(name), "kv_k_l%d", l);
        ggml_set_name(kv->k[(size_t) l], name);
        snprintf(name, sizeof(name), "kv_v_l%d", l);
        ggml_set_name(kv->v[(size_t) l], name);
    }

    kv->buffer = ggml_backend_alloc_ctx_tensors(kv->ctx, backend);
    if (!kv->buffer) {
        fprintf(stderr, "[KVCache] FATAL: backend allocation failed\n");
        ggml_free(kv->ctx);
        kv->ctx = NULL;
        return false;
    }

    // Zero-init the buffer so any out of bounds read returns a known value.
    ggml_backend_buffer_clear(kv->buffer, 0);

    size_t bytes_per_layer = (size_t) head_dim * (size_t) max_seq_len * (size_t) n_kv_heads * sizeof(float);
    size_t total_mb        = (size_t) (2 * n_layers) * bytes_per_layer / (1024 * 1024);
    fprintf(stderr,
            "[KVCache] Allocated: %d layers, %d KV heads, head_dim %d, "
            "max_seq_len %d -> %zu MB\n",
            n_layers, n_kv_heads, head_dim, max_seq_len, total_mb);
    return true;
}

// Rewind the cache so the next forward starts a fresh sequence.
static void kv_cache_reset(KVCache * kv) {
    kv->cur_len = 0;
}

static void kv_cache_free(KVCache * kv) {
    if (kv->buffer) {
        ggml_backend_buffer_free(kv->buffer);
        kv->buffer = NULL;
    }
    if (kv->ctx) {
        ggml_free(kv->ctx);
        kv->ctx = NULL;
    }
    kv->k.clear();
    kv->v.clear();
}
