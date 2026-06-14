#pragma once
// philox.h  Philox4x32-10 PRNG, uniform distribution
//
// Matches PyTorch CUDA torch.rand() output (cuRAND Philox4_32_10).
// Used by the multinomial sampler in sampling.h to stay byte-exact
// with the upstream Python pipeline. Zero dependencies beyond <cstdint>.

#include <cstdint>

// Philox constants (same as cuRAND / Random123)
static constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
static constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
static constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;

// cuRAND uniform conversion
static constexpr float CURAND_2POW32_INV = 2.3283064365386963e-10f; // 1 / 2^32

struct Philox4 {
  uint32_t x, y, z, w;
};

// 32x32 -> (hi32, lo32)
static inline void mulhilo32(uint32_t a, uint32_t b, uint32_t *hi,
                             uint32_t *lo) {
  uint64_t prod = (uint64_t)a * (uint64_t)b;
  *lo = (uint32_t)prod;
  *hi = (uint32_t)(prod >> 32);
}

// Single Philox round
static inline Philox4 philox_round(Philox4 ctr, uint32_t k0, uint32_t k1) {
  uint32_t hi0, lo0, hi1, lo1;
  mulhilo32(PHILOX_M0, ctr.x, &hi0, &lo0);
  mulhilo32(PHILOX_M1, ctr.z, &hi1, &lo1);
  return {
      hi1 ^ ctr.y ^ k0,
      lo1,
      hi0 ^ ctr.w ^ k1,
      lo0,
  };
}

// Philox4x32-10: 10 rounds
static inline Philox4 philox4x32_10(Philox4 ctr, uint32_t seed_lo,
                                    uint32_t seed_hi) {
  uint32_t k0 = seed_lo;
  uint32_t k1 = seed_hi;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  k0 += PHILOX_W0;
  k1 += PHILOX_W1;
  ctr = philox_round(ctr, k0, k1);
  return ctr;
}

// Fill array with uniform [0, 1) drawn from Philox4x32-10. Matches
// PyTorch CUDA torch.rand kernels.
//
// Required by the multinomial sampler used during stochastic
// generation: torch.multinomial(probs, 1) decomposes mathematically
// as u ~ Uniform[0, 1); cdf = cumsum(probs); argmin{ i: cdf[i] >= u }.
// To stay byte-exact with the upstream Python pipeline, both sides
// must consume the same u from the same Philox state, hence we need a
// uniform draw, not a Box-Muller normal.
//
// Convention :
//   key       = seed
//   subseq    = subseq_start + k (one subsequence per element)
//   ctr       = (ctr_lo, 0, subseq_lo, subseq_hi)
//   output    = (r.x + 0.5) * 2^-32  in (0, 1)
//
// ctr_lo is the cumulative Philox block counter the caller advances
// across kernels. On the first call after manual_seed, ctr_lo = 0.
static inline void philox_uniform_fill(int64_t seed, int64_t subseq_start,
                                       uint32_t ctr_lo, float *out, int n) {
  uint32_t slo = (uint32_t)seed;
  uint32_t shi = (uint32_t)((uint64_t)seed >> 32);
  for (int k = 0; k < n; k++) {
    uint64_t s = (uint64_t)(subseq_start + k);
    Philox4 ctr = {ctr_lo, 0u, (uint32_t)s, (uint32_t)(s >> 32)};
    Philox4 r = philox4x32_10(ctr, slo, shi);
    out[k] = ((float)r.x + 0.5f) * CURAND_2POW32_INV;
  }
}
