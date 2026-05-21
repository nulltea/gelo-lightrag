//! Bulk Gaussian generator for shield-row population (GELO §4.2).
//!
//! Produces `N(0, σ²)` f32 samples for the shield stack using a bulk
//! RNG draw + vectorised Box-Muller. ~6× faster at the decode shape
//! (k=15 rows × d=2560) on Zen 5 vs. the per-element
//! `rand_distr::StandardNormal::sample` loop the previous code path
//! used (commit `0cbe858` baseline: 486 µs/call).
//!
//! Algorithm:
//! 1. Bulk-fill a `u32` buffer from the executor RNG (one
//!    `RngCore::fill_bytes` invocation; ChaCha20 vectorises internally).
//! 2. Convert each `u32` pair `(b1, b2)` to a uniform pair
//!    `(u1, u2) ∈ (0, 1]` via top-24-bit truncation + small positive
//!    bias (keeps `u1 > 0`, so `ln(u1)` stays finite at all input bits).
//! 3. Box-Muller: `r = σ · √(−2·ln(u1))`, emit
//!    `(r·cos(2π·u2), r·sin(2π·u2))` as two i.i.d. Gaussians.
//! 4. Inner loop processes **eight** Box-Muller pairs (16 outputs) per
//!    iteration via `wide::f32x8`. `ln`, `sqrt`, and `sin_cos` all
//!    vectorise on `wide`; LLVM lowers to AVX-512 on Zen 5
//!    (`is_x86_feature_detected!("avx512f")` host) or AVX2 / NEON
//!    elsewhere.
//!
//! Security note: this routine produces **shield noise**, which is
//! distribution-quality material (must look Gaussian, must have σ
//! tracking `energy_scale × mean‖h‖`). It is NOT cryptographic key
//! material — the per-batch orthogonal mask `A` is. The shield RNG
//! source is the executor's `ChaCha20Rng`, same as the mask, so we
//! inherit the same forward-secrecy properties the protocol assumes.
//! Changing the per-sample bit consumption order vs. the prior scalar
//! code is therefore protocol-safe: the existing
//! `shield_rows_have_expected_energy` unit test (20 %-tolerance on mean
//! row norm) is the right semantic gate, and we keep it passing.

use rand::RngCore;
use wide::f32x8;

/// Scale factor for converting a 24-bit unsigned integer to `[0, 1)`.
const UNIFORM_SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
/// Half-LSB additive bias so the converted uniform lands in
/// `(0.5·2⁻²⁴, 1 − 0.5·2⁻²⁴]` — never zero, never one.  Keeps
/// `ln(u1)` finite.
const UNIFORM_BIAS: f32 = 0.5 / ((1u32 << 24) as f32);
const TAU: f32 = std::f32::consts::TAU;

/// Fill `dest` with i.i.d. `N(0, sigma²)` f32 samples.
///
/// The RNG is consumed by `RngCore::fill_bytes`, so for an
/// `n`-element destination we pull `4·ceil(n/2)·2 + pad` bytes
/// (rounded up to the next 16-pair boundary so the SIMD inner loop has
/// a clean tail).
pub fn fill_gaussian<R: RngCore + ?Sized>(dest: &mut [f32], sigma: f32, rng: &mut R) {
    let n = dest.len();
    if n == 0 {
        return;
    }
    if sigma == 0.0 {
        dest.fill(0.0);
        return;
    }

    // One Box-Muller draw uses two uniforms and emits two Gaussians.
    // For `n` outputs we need `ceil(n/2)` pairs.  Round up to a
    // multiple of eight pairs (= 16 u32s) so the SIMD body has no
    // partial-tail branching; the tail-trailing scalar loop only fires
    // on the last few outputs.
    let n_pairs = n.div_ceil(2);
    let n_u32 = (n_pairs * 2).next_multiple_of(16);

    // Heap buffer: at k=15, d=2560 this is 38_416 u32 ≈ 150 KiB,
    // beyond a safe stack slot.  Allocation is amortised by the
    // scratch-reuse path upstream in the executor (one `Vec` per
    // shield call; ~38 k Bytes is well within malloc cache hits).
    let mut u32_buf: Vec<u32> = vec![0; n_u32];
    {
        // SAFETY: `u32_buf` is contiguously allocated and aligned to
        // 4-byte boundaries by `Vec<u32>`; reinterpreting as a `[u8]`
        // of equal byte-length is sound. The slice is not aliased
        // because `u32_buf` is borrowed mutably only here.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                u32_buf.as_mut_ptr().cast::<u8>(),
                n_u32 * std::mem::size_of::<u32>(),
            )
        };
        rng.fill_bytes(bytes);
    }

    let scale_v = f32x8::splat(UNIFORM_SCALE);
    let bias_v = f32x8::splat(UNIFORM_BIAS);
    let neg_two_v = f32x8::splat(-2.0);
    let tau_v = f32x8::splat(TAU);
    let sigma_v = f32x8::splat(sigma);

    let mut idx: usize = 0;
    // Each SIMD iteration emits 16 outputs from 16 u32 inputs (8
    // Box-Muller pairs).  We keep the (u1, u2) layout
    // u32_buf[base + 2i + 0] = u1_bits[i], u32_buf[base + 2i + 1] =
    // u2_bits[i] so that the consumed RNG bytes form one contiguous
    // run per chunk.
    while idx + 16 <= n {
        let base = idx;
        let u1_arr: [f32; 8] = [
            (u32_buf[base] >> 8) as f32,
            (u32_buf[base + 2] >> 8) as f32,
            (u32_buf[base + 4] >> 8) as f32,
            (u32_buf[base + 6] >> 8) as f32,
            (u32_buf[base + 8] >> 8) as f32,
            (u32_buf[base + 10] >> 8) as f32,
            (u32_buf[base + 12] >> 8) as f32,
            (u32_buf[base + 14] >> 8) as f32,
        ];
        let u2_arr: [f32; 8] = [
            (u32_buf[base + 1] >> 8) as f32,
            (u32_buf[base + 3] >> 8) as f32,
            (u32_buf[base + 5] >> 8) as f32,
            (u32_buf[base + 7] >> 8) as f32,
            (u32_buf[base + 9] >> 8) as f32,
            (u32_buf[base + 11] >> 8) as f32,
            (u32_buf[base + 13] >> 8) as f32,
            (u32_buf[base + 15] >> 8) as f32,
        ];
        let u1 = f32x8::from(u1_arr) * scale_v + bias_v;
        let u2 = f32x8::from(u2_arr) * scale_v + bias_v;

        let r = sigma_v * (neg_two_v * u1.ln()).sqrt();
        let (sin_t, cos_t) = (tau_v * u2).sin_cos();
        let cos_arr = (r * cos_t).to_array();
        let sin_arr = (r * sin_t).to_array();

        // Interleave (c0, s0, c1, s1, …, c7, s7) so the consumed-byte
        // ↔ output-index mapping is the natural pair layout (the
        // existing energy test inspects mean row norm, which is order-
        // invariant; this keeps the convention obvious for future
        // debugging).
        for i in 0..8 {
            dest[idx + 2 * i] = cos_arr[i];
            dest[idx + 2 * i + 1] = sin_arr[i];
        }
        idx += 16;
    }

    // Tail: fewer than 16 outputs left.  Scalar Box-Muller pairs.
    while idx < n {
        let u_idx = idx;
        let u1_bits = u32_buf[u_idx];
        let u2_bits = u32_buf[u_idx + 1];
        let u1 = (u1_bits >> 8) as f32 * UNIFORM_SCALE + UNIFORM_BIAS;
        let u2 = (u2_bits >> 8) as f32 * UNIFORM_SCALE + UNIFORM_BIAS;
        let r = sigma * (-2.0_f32 * u1.ln()).sqrt();
        let (sin_t, cos_t) = (TAU * u2).sin_cos();
        dest[idx] = r * cos_t;
        if idx + 1 < n {
            dest[idx + 1] = r * sin_t;
        }
        idx += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// With σ = 1 the sample mean over many draws should be near zero
    /// and the sample variance near 1.  Tolerances chosen for n =
    /// 100_000 (sample-mean std ≈ 1/√n ≈ 3.2e-3).
    #[test]
    fn unit_variance_zero_mean() {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let n = 100_000;
        let mut buf = vec![0.0_f32; n];
        fill_gaussian(&mut buf, 1.0, &mut rng);

        let mean: f64 = buf.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var: f64 =
            buf.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 0.02, "mean {mean} too far from 0");
        assert!((var - 1.0).abs() < 0.05, "var {var} too far from 1");
    }

    /// With σ = 3 the row energy `E[‖row‖²] = d · σ²` should match
    /// within 5 % at d = 4096.
    #[test]
    fn sigma_scales_energy_linearly() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let d = 4096;
        let sigma = 3.0_f32;
        let mut buf = vec![0.0_f32; d];
        fill_gaussian(&mut buf, sigma, &mut rng);

        let energy: f64 =
            buf.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / d as f64;
        let expected = (sigma * sigma) as f64;
        let rel = (energy - expected).abs() / expected;
        assert!(rel < 0.05, "energy {energy} vs expected {expected} (rel {rel})");
    }

    /// Zero σ short-circuits to all-zero output.
    #[test]
    fn zero_sigma_yields_zero() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let mut buf = vec![1.0_f32; 64];
        fill_gaussian(&mut buf, 0.0, &mut rng);
        assert!(buf.iter().all(|v| *v == 0.0));
    }

    /// Odd-length destinations (tail path) must still satisfy
    /// distribution shape.
    #[test]
    fn odd_length_tail() {
        let mut rng = ChaCha20Rng::from_seed([17u8; 32]);
        let n = 999; // not a multiple of 16
        let mut buf = vec![0.0_f32; n];
        fill_gaussian(&mut buf, 2.0, &mut rng);
        let energy: f64 = buf.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64;
        let expected = 4.0;
        assert!((energy - expected).abs() / expected < 0.15);
    }

    /// Two independent invocations with the same seed must produce
    /// identical output — necessary for executor determinism (greedy
    /// decode replay).
    #[test]
    fn deterministic_per_seed() {
        let mut rng_a = ChaCha20Rng::from_seed([23u8; 32]);
        let mut rng_b = ChaCha20Rng::from_seed([23u8; 32]);
        let mut a = vec![0.0_f32; 512];
        let mut b = vec![0.0_f32; 512];
        fill_gaussian(&mut a, 1.5, &mut rng_a);
        fill_gaussian(&mut b, 1.5, &mut rng_b);
        assert_eq!(a, b);
    }
}
