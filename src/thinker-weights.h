#pragma once
// thinker-weights.h: Qwen3 Thinker decoder weights, the text side of Qwen3-ASR.
// Decoder-only Qwen3 LM with pre-norm GQA attention, per-head QK-norm and a
// pre-norm SwiGLU MLP. Config comes from the qwenasr.* GGUF metadata, tensor
// names follow the llama.cpp convention. Loaded f32 for the cossim harness.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cstdio>
#include <string>
#include <vector>

struct ThinkerAttention {
    struct ggml_tensor * q_proj_w;
    struct ggml_tensor * k_proj_w;
    struct ggml_tensor * v_proj_w;
    struct ggml_tensor * o_proj_w;
    struct ggml_tensor * q_norm_w;
    struct ggml_tensor * k_norm_w;
};

struct ThinkerMLP {
    struct ggml_tensor * gate_proj_w;
    struct ggml_tensor * up_proj_w;
    struct ggml_tensor * down_proj_w;
};

struct ThinkerLayer {
    struct ggml_tensor * input_norm_w;
    ThinkerAttention     attn;
    struct ggml_tensor * post_attn_norm_w;
    ThinkerMLP           mlp;
};

struct ThinkerWeights {
    int   hidden_size;
    int   intermediate_size;
    int   num_hidden_layers;
    int   num_attention_heads;
    int   num_key_value_heads;
    int   head_dim;
    int   vocab_size;
    int   max_position_embeddings;
    float rope_theta;
    float rms_norm_eps;
    bool  tie_word_embeddings;

    struct ggml_tensor * token_embd;
    struct ggml_tensor * output_norm_w;
    struct ggml_tensor * output_w;

    std::vector<ThinkerLayer> layers;

    WeightCtx wctx;
};

static bool thinker_weights_load(ThinkerWeights * tw, const GGUFModel & gf, ggml_backend_t backend) {
    tw->hidden_size             = (int) gf_get_u32(gf, "qwenasr.embedding_length");
    tw->intermediate_size       = (int) gf_get_u32(gf, "qwenasr.feed_forward_length");
    tw->num_hidden_layers       = (int) gf_get_u32(gf, "qwenasr.block_count");
    tw->num_attention_heads     = (int) gf_get_u32(gf, "qwenasr.attention.head_count");
    tw->num_key_value_heads     = (int) gf_get_u32(gf, "qwenasr.attention.head_count_kv");
    tw->head_dim                = (int) gf_get_u32(gf, "qwenasr.attention.key_length");
    tw->vocab_size              = (int) gf_get_u32(gf, "qwenasr.vocab_size");
    tw->max_position_embeddings = (int) gf_get_u32(gf, "qwenasr.context_length");
    tw->rope_theta              = gf_get_f32(gf, "qwenasr.rope.freq_base");
    tw->rms_norm_eps            = gf_get_f32(gf, "qwenasr.attention.layer_norm_rms_epsilon");
    tw->tie_word_embeddings     = gf_get_bool(gf, "qwenasr.tie_word_embeddings");

    if (tw->num_hidden_layers <= 0 || tw->hidden_size <= 0) {
        fprintf(stderr, "[Thinker] FATAL: invalid hyperparameters (layers=%d hidden=%d)\n", tw->num_hidden_layers,
                tw->hidden_size);
        return false;
    }

    tw->layers.resize((size_t) tw->num_hidden_layers);

    // 3 top-level + per layer (2 norms + 4 attn + 2 qk norms + 3 mlp = 11)
    wctx_init(&tw->wctx, 3 + tw->num_hidden_layers * 11);

    tw->token_embd    = gf_load_tensor_f32(&tw->wctx, gf, "thinker.token_embd.weight");
    tw->output_norm_w = gf_load_tensor_f32(&tw->wctx, gf, "thinker.output_norm.weight");
    tw->output_w      = gf_load_tensor_f32(&tw->wctx, gf, "thinker.output.weight");

    for (int l = 0; l < tw->num_hidden_layers; l++) {
        ThinkerLayer &    layer = tw->layers[(size_t) l];
        const std::string p     = "thinker.blk." + std::to_string(l) + ".";
        layer.input_norm_w      = gf_load_tensor_f32(&tw->wctx, gf, p + "attn_norm.weight");
        layer.post_attn_norm_w  = gf_load_tensor_f32(&tw->wctx, gf, p + "ffn_norm.weight");
        layer.attn.q_proj_w     = gf_load_tensor_f32(&tw->wctx, gf, p + "attn_q.weight");
        layer.attn.k_proj_w     = gf_load_tensor_f32(&tw->wctx, gf, p + "attn_k.weight");
        layer.attn.v_proj_w     = gf_load_tensor_f32(&tw->wctx, gf, p + "attn_v.weight");
        layer.attn.o_proj_w     = gf_load_tensor_f32(&tw->wctx, gf, p + "attn_output.weight");
        layer.attn.q_norm_w     = gf_load_tensor_f32(&tw->wctx, gf, p + "attn_q_norm.weight");
        layer.attn.k_norm_w     = gf_load_tensor_f32(&tw->wctx, gf, p + "attn_k_norm.weight");
        layer.mlp.gate_proj_w   = gf_load_tensor_f32(&tw->wctx, gf, p + "ffn_gate.weight");
        layer.mlp.up_proj_w     = gf_load_tensor_f32(&tw->wctx, gf, p + "ffn_up.weight");
        layer.mlp.down_proj_w   = gf_load_tensor_f32(&tw->wctx, gf, p + "ffn_down.weight");
    }

    if (!wctx_alloc(&tw->wctx, backend)) {
        fprintf(stderr, "[Thinker] FATAL: backend allocation failed\n");
        return false;
    }

    fprintf(stderr,
            "[Thinker] Loaded: %d layers, hidden %d, heads %d/%d, head_dim %d, "
            "FFN %d, RoPE theta %.0f, vocab %d, tie_embd %d\n",
            tw->num_hidden_layers, tw->hidden_size, tw->num_attention_heads, tw->num_key_value_heads, tw->head_dim,
            tw->intermediate_size, (double) tw->rope_theta, tw->vocab_size, (int) tw->tie_word_embeddings);
    return true;
}

static void thinker_weights_free(ThinkerWeights * tw) {
    if (tw->wctx.buffer) {
        ggml_backend_buffer_free(tw->wctx.buffer);
        tw->wctx.buffer = nullptr;
    }
    if (tw->wctx.ctx) {
        ggml_free(tw->wctx.ctx);
        tw->wctx.ctx = nullptr;
    }
    tw->layers.clear();
}
