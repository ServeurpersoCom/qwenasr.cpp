#pragma once
// audio-mel.h: log mel frontend, WhisperFeatureExtractor equivalent for
// Qwen3-ASR. Same in-graph strategy as the sibling qwentts.cpp audio-mel.h:
// no native FFT op, so the DFT is folded into two real matmuls over im2col
// frames, all on the backend. Differences from the qwentts speaker mel are
// the Qwen3-ASR config and the Whisper specific math:
//
//   center=True reflect pad by n_fft / 2 (done by the caller)
//   torch.stft(n_fft, hop, win=n_fft, hann_periodic), drop the trailing frame
//   power = real^2 + imag^2          (magnitude squared, no sqrt)
//   mel = slaney_filterbank @ power
//   log_spec = log10(max(mel, 1e-10))
//   log_spec = max(log_spec, log_spec.max() - 8)
//   log_spec = (log_spec + 4) / 4
//
// Config: sr=16000, n_fft=400, hop=160, n_mels=128, fmin=0, fmax=8000.
//
// The graph emits the log10 mel laid out [n_frames, n_mels] (ne), which is
// memory [n_mels outer, n_frames inner] and feeds the conv stem directly. The
// drop-last frame and the data dependent max normalization run on the host
// after readback, since GGML has no global max reduction. whisper.cpp computes
// its mel on the host for the same reason.

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

struct AudioMelConfig {
    int   sample_rate = 16000;
    int   n_fft       = 400;
    int   hop         = 160;
    int   n_mels      = 128;
    float fmin        = 0.0f;
    float fmax        = 8000.0f;
};

struct AudioMelConstants {
    AudioMelConfig     cfg;
    int                n_freq;
    std::vector<float> hann;       // [n_fft]
    std::vector<float> dft_real;   // [n_freq, n_fft] row major  ne=(n_fft, n_freq)
    std::vector<float> dft_imag;   // [n_freq, n_fft] row major
    std::vector<float> mel_basis;  // [n_mels, n_freq] row major  ne=(n_freq, n_mels)
};

// Slaney mel scale, the default of librosa.filters.mel(htk=False).
static inline float audio_mel_hz_to_mel(float hz) {
    const float f_sp        = 200.0f / 3.0f;
    const float min_log_hz  = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep     = std::log(6.4f) / 27.0f;
    if (hz < min_log_hz) {
        return hz / f_sp;
    }
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

static inline float audio_mel_mel_to_hz(float mel) {
    const float f_sp        = 200.0f / 3.0f;
    const float min_log_hz  = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep     = std::log(6.4f) / 27.0f;
    if (mel < min_log_mel) {
        return f_sp * mel;
    }
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

// Bake the Hann window, the cos/sin DFT matrices and the Slaney filterbank.
// Trig is evaluated in f64 to keep the roundoff under the f32 ULP.
static void audio_mel_compute_constants(const AudioMelConfig & cfg, AudioMelConstants & c) {
    c.cfg    = cfg;
    c.n_freq = cfg.n_fft / 2 + 1;

    c.hann.assign((size_t) cfg.n_fft, 0.0f);
    for (int i = 0; i < cfg.n_fft; i++) {
        c.hann[(size_t) i] = 0.5f * (1.0f - (float) std::cos(2.0 * M_PI * (double) i / (double) cfg.n_fft));
    }

    c.dft_real.assign((size_t) c.n_freq * (size_t) cfg.n_fft, 0.0f);
    c.dft_imag.assign((size_t) c.n_freq * (size_t) cfg.n_fft, 0.0f);
    for (int k = 0; k < c.n_freq; k++) {
        for (int n = 0; n < cfg.n_fft; n++) {
            const double th = 2.0 * M_PI * (double) k * (double) n / (double) cfg.n_fft;
            c.dft_real[(size_t) k * (size_t) cfg.n_fft + (size_t) n] = (float) std::cos(th);
            c.dft_imag[(size_t) k * (size_t) cfg.n_fft + (size_t) n] = (float) (-std::sin(th));
        }
    }

    const float fmax = (cfg.fmax <= 0.0f) ? (float) cfg.sample_rate * 0.5f : cfg.fmax;
    const float mmin = audio_mel_hz_to_mel(cfg.fmin);
    const float mmax = audio_mel_hz_to_mel(fmax);

    std::vector<float> hz_pts((size_t) cfg.n_mels + 2);
    for (int i = 0; i < cfg.n_mels + 2; i++) {
        const float mel    = mmin + (mmax - mmin) * (float) i / (float) (cfg.n_mels + 1);
        hz_pts[(size_t) i] = audio_mel_mel_to_hz(mel);
    }
    std::vector<float> fft_freqs((size_t) c.n_freq);
    for (int k = 0; k < c.n_freq; k++) {
        fft_freqs[(size_t) k] = (float) k * (float) cfg.sample_rate / (float) cfg.n_fft;
    }

    c.mel_basis.assign((size_t) cfg.n_mels * (size_t) c.n_freq, 0.0f);
    for (int m = 0; m < cfg.n_mels; m++) {
        const float lo    = hz_pts[(size_t) m];
        const float md    = hz_pts[(size_t) m + 1];
        const float hi    = hz_pts[(size_t) m + 2];
        const float enorm = 2.0f / (hi - lo);
        for (int k = 0; k < c.n_freq; k++) {
            const float f    = fft_freqs[(size_t) k];
            const float up   = (f - lo) / (md - lo);
            const float down = (hi - f) / (hi - md);
            float       w    = std::fmin(up, down);
            if (w < 0.0f) {
                w = 0.0f;
            }
            c.mel_basis[(size_t) m * (size_t) c.n_freq + (size_t) k] = w * enorm;
        }
    }
}

// Build the mel graph. The constants live as caller owned graph inputs,
// matching the sibling qwentts.cpp frontend.
//
// Inputs:
//   audio_padded [T_padded]          f32, reflect padded by the caller
//   hann         [n_fft]             f32 host constant
//   dft_real     [n_fft, n_freq]     f32 host constant
//   dft_imag     [n_fft, n_freq]     f32 host constant
//   mel_basis    [n_freq, n_mels]    f32 host constant
//
// Returns the log10 mel with ne = [n_frames_full, n_mels] (memory n_mels outer,
// frame inner). The caller drops the trailing frame and applies the Whisper max
// normalization on the host.
static struct ggml_tensor * audio_mel_build_graph(struct ggml_context *  ctx,
                                                  struct ggml_tensor *   audio_padded,
                                                  struct ggml_tensor *   hann,
                                                  struct ggml_tensor *   dft_real,
                                                  struct ggml_tensor *   dft_imag,
                                                  struct ggml_tensor *   mel_basis,
                                                  const AudioMelConfig & cfg) {
    const int n_fft = cfg.n_fft;
    const int hop   = cfg.hop;

    struct ggml_tensor * a4d   = ggml_reshape_4d(ctx, audio_padded, audio_padded->ne[0], 1, 1, 1);
    struct ggml_tensor * dummy = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_fft, 1, 1, 1);
    ggml_set_name(dummy, "mel.im2col_dummy_kernel");

    struct ggml_tensor * frames = ggml_im2col(ctx, dummy, a4d, hop, 1, 0, 0, 1, 1, false, GGML_TYPE_F32);
    frames                      = ggml_reshape_2d(ctx, frames, n_fft, frames->ne[1]);

    frames = ggml_mul(ctx, frames, ggml_reshape_2d(ctx, hann, n_fft, 1));

    struct ggml_tensor * spec_re = ggml_mul_mat(ctx, dft_real, frames);  // [n_freq, n_frames]
    struct ggml_tensor * spec_im = ggml_mul_mat(ctx, dft_imag, frames);
    ggml_mul_mat_set_prec(spec_re, GGML_PREC_F32);
    ggml_mul_mat_set_prec(spec_im, GGML_PREC_F32);

    // Whisper uses the power spectrum (magnitude squared), no sqrt, no eps.
    struct ggml_tensor * power = ggml_add(ctx, ggml_sqr(ctx, spec_re), ggml_sqr(ctx, spec_im));

    // power [n_freq, n_frames] x mel_basis [n_freq, n_mels] -> [n_frames,
    // n_mels], memory n_mels outer / frame inner, the layout the conv stem reads.
    struct ggml_tensor * mel = ggml_mul_mat(ctx, power, mel_basis);
    ggml_mul_mat_set_prec(mel, GGML_PREC_F32);

    mel = ggml_clamp(ctx, mel, 1e-10f, 1e30f);
    mel = ggml_log(ctx, mel);
    mel = ggml_scale(ctx, mel, (float) (1.0 / 2.302585092994046));  // ln -> log10
    ggml_set_name(mel, "mel.log10");
    return mel;
}

// Host tail, matching WhisperFeatureExtractor: drop the trailing frame, then
// log_spec = (max(log_spec, log_spec.max() - 8) + 4) / 4. Input is the graph
// output [n_frames_full, n_mels] (n_mels outer, frame inner). Returns the
// normalized mel [n_mels, n_frames_full - 1] in the same memory convention.
static size_t audio_mel_normalize(const std::vector<float> & log10_mel,
                                  int                        n_mels,
                                  size_t                     n_frames_full,
                                  std::vector<float> &       out) {
    const size_t n_frames = n_frames_full - 1;  // Whisper drops the last frame
    out.assign((size_t) n_mels * n_frames, 0.0f);

    double gmax = -1.0e30;
    for (int m = 0; m < n_mels; m++) {
        const size_t src = (size_t) m * n_frames_full;
        const size_t dst = (size_t) m * n_frames;
        for (size_t f = 0; f < n_frames; f++) {
            const float v = log10_mel[src + f];
            out[dst + f]  = v;
            if (v > gmax) {
                gmax = v;
            }
        }
    }
    const float floor = (float) (gmax - 8.0);
    for (size_t i = 0; i < out.size(); i++) {
        float v = out[i] < floor ? floor : out[i];
        out[i]  = (v + 4.0f) / 4.0f;
    }
    return n_frames;
}
