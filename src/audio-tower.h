#pragma once
// audio-tower.h: windowed forward of the Qwen3-ASR audio tower. Splits the mel
// into chunk_mel = 100 frame chunks, runs the conv stem per chunk with the
// positional table reset per chunk, concatenates the after-cnn frames and runs
// the encoder with a block-diagonal attention mask over windows of
// window_aftercnn = chunk_aftercnn * window_chunks = 104 frames. Ties
// conv-stem.h and audio-enc.h into one graph.

#include "audio-enc.h"
#include "conv-stem.h"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

struct AudioTowerConfig {
  int chunk_mel = 100;   // n_window * 2
  int window_chunks = 8; // n_window_infer / chunk_mel
};

// after-cnn length of a conv chunk of mel_len frames, three conv2d stride 2
// kernel 3 pad 1. Matches the reference _get_feat_extract_output_lengths for
// any chunk length >= 1.
static int audio_tower_conv_out(int mel_len) {
  int o = mel_len;
  for (int i = 0; i < 3; i++) {
    o = (o - 1) / 2 + 1;
  }
  return o;
}

// Mel chunk plan: ceil(n_frames / chunk_mel) chunks, all chunk_mel long, the
// tail chunk carries the remainder (chunk_mel when it divides evenly).
static std::vector<int> audio_tower_chunk_lengths(int n_frames,
                                                  const AudioTowerConfig &cfg) {
  std::vector<int> lengths;
  int off = 0;
  while (off < n_frames) {
    const int len =
        (n_frames - off) < cfg.chunk_mel ? (n_frames - off) : cfg.chunk_mel;
    lengths.push_back(len);
    off += len;
  }
  return lengths;
}

// Total after-cnn length over all chunks.
static int audio_tower_seq_len(const std::vector<int> &chunk_lengths) {
  int s = 0;
  for (int len : chunk_lengths) {
    s += audio_tower_conv_out(len);
  }
  return s;
}

// Additive block-diagonal mask [S, S], memory key inner. 0 inside a window of
// window_aftercnn frames, -inf across windows.
static std::vector<float> audio_tower_build_mask(int seq_len,
                                                 int window_aftercnn) {
  const float neg = -std::numeric_limits<float>::infinity();
  std::vector<float> mask((size_t)seq_len * (size_t)seq_len, neg);
  for (int b0 = 0; b0 < seq_len; b0 += window_aftercnn) {
    const int b1 =
        (b0 + window_aftercnn) < seq_len ? (b0 + window_aftercnn) : seq_len;
    for (int q = b0; q < b1; q++) {
      for (int k = b0; k < b1; k++) {
        mask[(size_t)q * (size_t)seq_len + (size_t)k] = 0.0f;
      }
    }
  }
  return mask;
}

// Build the windowed tower graph. mel_in [n_frames, n_mels, 1, 1], pe_chunk_in
// [d_model, chunk_aftercnn] is the positional table sliced per chunk, mask_in
// [S, S] is the block-diagonal mask. chunk_lengths is the host chunk plan.
// Returns the projected states [output_dim, S].
static struct ggml_tensor *
audio_tower_build(struct ggml_context *ctx, const ConvStem &stem,
                  const AudioEnc &enc, struct ggml_tensor *mel_in,
                  struct ggml_tensor *pe_chunk_in, struct ggml_tensor *mask_in,
                  const std::vector<int> &chunk_lengths) {
  const int64_t n_mels = mel_in->ne[1];
  const int d_model = enc.cfg.d_model;

  struct ggml_tensor *seq = nullptr;
  int64_t off = 0;
  for (int len : chunk_lengths) {
    struct ggml_tensor *chunk =
        ggml_cont(ctx, ggml_view_4d(ctx, mel_in, len, n_mels, 1, 1,
                                    mel_in->nb[1], mel_in->nb[2], mel_in->nb[3],
                                    (size_t)off * mel_in->nb[0]));
    struct ggml_tensor *s = conv_stem_build(ctx, stem, chunk); // [d_model, t_c]
    const int64_t t_c = s->ne[1];
    struct ggml_tensor *pe =
        ggml_view_2d(ctx, pe_chunk_in, d_model, t_c, pe_chunk_in->nb[1], 0);
    s = ggml_add(ctx, s, pe);
    seq = seq ? ggml_concat(ctx, seq, s, 1) : s;
    off += len;
  }

  return audio_enc_build(ctx, enc, seq, mask_in);
}
