#pragma once
// sampling.h: token sampling for the Qwen3 Thinker decoder. Pipeline matches
// the HuggingFace generate() chain in F32 :
//   repetition_penalty -> temperature -> top_k -> top_p -> softmax ->
//   multinomial
// The multinomial uniform draw comes from philox_uniform_fill so the
// sequence stays byte for byte aligned with the patched torch.multinomial
// in tests/debug-asr-cossim.py. temperature <= 0 short circuits to greedy
// argmax, the Qwen3-ASR reference default.

#include "philox.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

struct TokenProb {
    int   id;
    float prob;
};

// Repetition penalty over unique tokens in history (HF rule):
//   if score >= 0 -> score / penalty
//   if score <  0 -> score * penalty
// Each token is touched at most once per call.
static inline void apply_repetition_penalty(float *         logits,
                                            int             V,
                                            const int32_t * history,
                                            int             n_history,
                                            float           penalty) {
    if (penalty == 1.0f || n_history <= 0) {
        return;
    }
    static thread_local std::vector<uint8_t> seen_buf;
    seen_buf.assign((size_t) V, 0);
    for (int h = 0; h < n_history; h++) {
        int32_t tok = history[h];
        if (tok < 0 || tok >= V) {
            continue;
        }
        if (seen_buf[(size_t) tok]) {
            continue;
        }
        seen_buf[(size_t) tok] = 1;
        float s                = logits[tok];
        logits[tok]            = (s < 0.0f) ? s * penalty : s / penalty;
    }
}

// Stochastic sampler. Pipeline mirrors HF generate() in F32 :
//   1. repetition_penalty(history)
//   2. temperature divide
//   3. top_k mask (skipped when k <= 0 or k >= V)
//   4. top_p nucleus mask (skipped when p >= 1.0)
//   5. softmax
//   6. multinomial via philox_uniform_fill(seed, philox_subseq, 0)
//
// Greedy path: temperature <= 0 returns argmax over the suppressed
// logits, no rep_pen, no philox draw.
//
// Buffers are thread_local to avoid alloc per token.
static int sample_top_k_p(float *         logits,
                          int             V,
                          float           temperature,
                          int             top_k,
                          float           top_p,
                          float           rep_pen,
                          const int32_t * history,
                          int             n_history,
                          int64_t         seed,
                          int64_t         philox_subseq,
                          float *         dump_u_out) {
    if (temperature <= 0.0f) {
        if (dump_u_out) {
            *dump_u_out = -1.0f;
        }
        return (int) (std::max_element(logits, logits + V) - logits);
    }

    apply_repetition_penalty(logits, V, history, n_history, rep_pen);

    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) {
        logits[i] *= inv_temp;
    }

    static thread_local std::vector<float>     tmp_buf;
    static thread_local std::vector<TokenProb> sorted_buf;

    if (top_k > 0 && top_k < V) {
        tmp_buf.resize((size_t) V);
        std::memcpy(tmp_buf.data(), logits, (size_t) V * sizeof(float));
        std::nth_element(tmp_buf.begin(), tmp_buf.begin() + (top_k - 1), tmp_buf.end(), std::greater<float>());
        float threshold = tmp_buf[(size_t) (top_k - 1)];
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) {
                logits[i] = -INFINITY;
            }
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        // Full vocab softmax for an exact cumsum boundary (matches HF
        // TopPLogitsWarper which softmaxes the sorted tensor in place).
        float max_logit = -INFINITY;
        for (int i = 0; i < V; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
            }
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < V; i++) {
            sum_exp += expf(logits[i] - max_logit);
        }
        float inv_sum = 1.0f / sum_exp;

        // Compact only tokens above a relative cutoff. exp(-16) is ~1e-7
        // so the dropped mass stays well below any reachable nucleus
        // boundary, and the sort runs on a small set (typically less
        // than top_k entries).
        float cutoff = max_logit - 16.0f;
        sorted_buf.clear();
        for (int i = 0; i < V; i++) {
            if (logits[i] >= cutoff) {
                float prob = expf(logits[i] - max_logit) * inv_sum;
                sorted_buf.push_back({ i, prob });
            } else {
                logits[i] = -INFINITY;
            }
        }

        int K = (int) sorted_buf.size();
        if (K > 0) {
            std::sort(sorted_buf.begin(), sorted_buf.end(),
                      [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });
            // HF convention: keep tokens until the cumulative probability
            // crosses top_p, drop the rest. Test before accumulate so the
            // first crossing entry is kept.
            float cum = 0.0f;
            for (int i = 0; i < K; i++) {
                if (i > 0 && cum >= top_p) {
                    logits[sorted_buf[(size_t) i].id] = -INFINITY;
                }
                cum += sorted_buf[(size_t) i].prob;
            }
        }
    }

    // Final softmax then philox driven multinomial. We keep the
    // unnormalized exponentials and draw u in [0, sum) which is
    // mathematically identical to softmax + u in [0, 1) but spares a
    // pass.
    float max_val = -INFINITY;
    for (int i = 0; i < V; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    float u = 0.0f;
    philox_uniform_fill(seed, philox_subseq, 0u, &u, 1);
    if (dump_u_out) {
        *dump_u_out = u;
    }

    float r   = u * sum;
    float acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) {
            return i;
        }
    }
    return V - 1;
}
