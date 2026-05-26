//! Bulk Gaussian generator for shield-row population (GELO §4.2).
//!
//! Produces `N(0, σ²)` f32 samples for the shield stack using a bulk
//! RNG draw + **vectorised polar (Marsaglia) rejection**.  Replaces
//! the prior per-element `rand_distr::StandardNormal::sample` (~24 ns
//! /sample on Zen 5) and the intermediate SIMD Box-Muller path
//! (~9 ns/sample) with a SIMD polar kernel (~3-4 ns/sample) that
//! avoids the `sin_cos` transcendental entirely.
//!
//! ## Algorithm
//!
//! Polar method (Knuth TAOCP vol. 2 §3.4.1.C, Marsaglia 1964): for
//! each pair of independent uniforms `u₁, u₂ ∈ (-1, 1)`, compute
//! `s = u₁² + u₂²`.  Reject if `s ≥ 1` or `s = 0`; otherwise emit two
//! i.i.d. Gaussians `x₁ = σ · u₁ · √(−2·ln(s) / s)` and
//! `x₂ = σ · u₂ · √(−2·ln(s) / s)`.
//!
//! Acceptance rate is `π/4 ≈ 78.54 %` (the disc/square area ratio),
//! so we need roughly `1.273×` more uniforms than the equivalent
//! Box-Muller path — we pool at 1.4× for a multi-σ safety margin.
//!
//! ## Why polar beats Box-Muller for SIMD
//!
//! Box-Muller emits the same `(x₁, x₂)` pair via `r·cos(θ), r·sin(θ)`
//! where `r = √(−2·ln u₁)` and `θ = 2π·u₂`.  At our shapes the
//! `sin_cos` polynomial inside `wide::f32x8` is ~35 % of the per-call
//! cost.  Polar replaces `sin_cos` with a multiply (`u·factor`) and
//! one division.  Net per-accepted-pair: same `ln + sqrt`, one extra
//! `mul`, one `div`, minus `sin_cos`.  Even paying the ~28 % extra
//! uniforms + ~15 % per-lane rejection bookkeeping, the kernel wins
//! by ~1.5× on top of P3 (Xoshiro256++) at the v7 decode shape.
//!
//! ## SIMD layout
//!
//! The RNG bulk-draws into a single `[u32]` pool laid out as
//! `[u1₀, …, u1_p, u2₀, …, u2_p]` so the inner loop can do contiguous
//! 8-lane loads of u₁ and u₂ separately (no gather instructions).
//! Each iteration processes 8 candidate pairs:
//!
//! 1. Load 8 `u32` lanes from each half, reinterpret as `i32`,
//!    convert to `f32x8`, and scale by `1/2³¹` → both lanes in
//!    `[-1, 1)` (the boundary `+1` maps to `1 − 2⁻²⁴`, well-inside
//!    the disc).
//! 2. `s = u₁² + u₂²`.
//! 3. `accept = (s > 0) & (s < 1)` — `wide`'s `move_mask` packs the
//!    per-lane sign bit into a single `u32` bitmap.
//! 4. Always compute `factor = σ · √(−2·ln(s) / s)` for all 8 lanes;
//!    `NaN`/`-inf` in rejected lanes are discarded by the
//!    bitmap-gated scalar store loop.  Branchless SIMD beats
//!    per-lane masking on AVX2.
//! 5. Compact: for each accepted lane, write `(x₁, x₂)` to the
//!    output.  The 8-iteration compaction is branchy at the bit level
//!    but BTB-friendly on modern x86.
//!
//! ## Security note
//!
//! The shield is **distribution-quality material**, not key
//! material — the per-batch orthogonal mask `A` is what the GELO
//! protocol hides.  The shield only has to look like `N(0, σ²)` and
//! be uncorrelated across offloads to defeat the Gram-matrix leak
//! (§4.2).  Both ChaCha20 and Xoshiro256++ pass BigCrush; the
//! per-offload-freshness invariant (cross-offload ICA defence —
//! `paper_parity_default.md`) is RNG-agnostic.  Caller chooses the
//! RNG: `InProcessTrustedExecutor::shield_rng` is Xoshiro256++ for
//! the hot path.

use rand::RngCore;
use wide::{CmpGt, CmpLt, f32x8, i32x8};

/// Conversion factor from `i32` to `f32` in `[-1, 1)`: 1 / 2³¹.
const I32_TO_UNIT: f32 = 1.0 / 2_147_483_648.0;

/// Multiplier on `target_pairs` to size the uniform pool for one
/// fill.  The polar acceptance rate is `π/4 ≈ 0.7854`, so the strict
/// minimum is `≈ 1.273×`.  We pool at 1.4× — at `target_pairs = 19200`
/// (decode k=15 × d=2560 outputs / 2) the expected accept count is
/// `0.7854 × 1.4 × 19200 ≈ 21111`, std `≈ √(0.169 × 26880) ≈ 67`, so
/// the 1911-pair margin above target is ~28σ.  Refill loop below
/// handles the astronomical case anyway.
const POOL_PADDING_NUM: usize = 7;
const POOL_PADDING_DEN: usize = 5;

/// Fill `dest` with i.i.d. `N(0, sigma²)` f32 samples.
///
/// The RNG is consumed via `RngCore::fill_bytes` in one bulk call per
/// pool refill (pool is sized 1.4× the strict minimum, so a single
/// refill suffices in practice; the `while out_idx < n` outer loop
/// handles the multi-σ-tail underrun case).
pub fn fill_gaussian<R: RngCore + ?Sized>(dest: &mut [f32], sigma: f32, rng: &mut R) {
    let n = dest.len();
    if n == 0 {
        return;
    }
    if sigma == 0.0 {
        dest.fill(0.0);
        return;
    }

    let target_pairs = n.div_ceil(2);
    let pool_pairs = ((target_pairs * POOL_PADDING_NUM) / POOL_PADDING_DEN).max(8);
    let pool_pairs_padded = pool_pairs.next_multiple_of(8);
    // Two halves: u1 contiguous, then u2 contiguous.  Contiguous SIMD
    // loads of each lane group; no gather/scatter.
    let pool_u32 = 2 * pool_pairs_padded;
    let mut u_buf: Vec<u32> = vec![0; pool_u32];

    let mut out_idx: usize = 0;
    while out_idx < n {
        fill_pool_bytes(&mut u_buf, rng);
        polar_emit_simd(&u_buf, pool_pairs_padded, dest, &mut out_idx, sigma);
    }
}

/// Bulk-draw `u_buf.len() * 4` random bytes into the pool.
fn fill_pool_bytes<R: RngCore + ?Sized>(u_buf: &mut [u32], rng: &mut R) {
    // SAFETY: `Vec<u32>` is 4-byte aligned; reinterpreting as `[u8]`
    // of equal byte-length is sound and not aliased — we hold the
    // only mutable reference for the duration of the call.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            u_buf.as_mut_ptr().cast::<u8>(),
            u_buf.len() * std::mem::size_of::<u32>(),
        )
    };
    rng.fill_bytes(bytes);
}

/// SIMD polar inner loop.  Reads up to `pool_pairs` pair-slots from
/// `u_buf` (the first `pool_pairs` u32s are u₁, the second
/// `pool_pairs` u32s are u₂) and writes accepted Gaussians to
/// `dest[*out_idx..]`.  Returns early when either pool or output is
/// exhausted.
fn polar_emit_simd(
    u_buf: &[u32],
    pool_pairs: usize,
    dest: &mut [f32],
    out_idx: &mut usize,
    sigma: f32,
) {
    let n = dest.len();
    if *out_idx >= n {
        return;
    }
    let u1_pool = &u_buf[..pool_pairs];
    let u2_pool = &u_buf[pool_pairs..2 * pool_pairs];

    let scale_v = f32x8::splat(I32_TO_UNIT);
    let one_v = f32x8::splat(1.0);
    let zero_v = f32x8::splat(0.0);
    let neg_two_v = f32x8::splat(-2.0);
    let sigma_v = f32x8::splat(sigma);

    let mut input_idx = 0;
    while input_idx + 8 <= pool_pairs {
        let u1_lanes = load_lanes_as_unit_f32(&u1_pool[input_idx..input_idx + 8], scale_v);
        let u2_lanes = load_lanes_as_unit_f32(&u2_pool[input_idx..input_idx + 8], scale_v);
        let s = u1_lanes * u1_lanes + u2_lanes * u2_lanes;

        // Acceptance: s ∈ (0, 1).  cmp_lt / cmp_gt yield all-1s/0
        // lanes; bitwise & narrows to "both true".  move_mask packs
        // the per-lane sign bit into the low 8 bits of a u32.
        let accept = s.cmp_lt(one_v) & s.cmp_gt(zero_v);
        let accept_bits = accept.move_mask() as u32;

        if accept_bits != 0 {
            // Always compute factor for all 8 lanes — branchless SIMD
            // beats per-lane masking on AVX2.  Rejected lanes yield
            // ±inf/NaN in `factor`; discarded by the bitmap-gated
            // scalar store below.
            let factor = sigma_v * (neg_two_v * s.ln() / s).sqrt();
            let x1_arr = (u1_lanes * factor).to_array();
            let x2_arr = (u2_lanes * factor).to_array();

            let mut bits = accept_bits;
            // Walk set bits low-to-high; each emits a pair to dest.
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1; // clear lowest set bit
                if *out_idx >= n {
                    return;
                }
                dest[*out_idx] = x1_arr[i];
                *out_idx += 1;
                if *out_idx >= n {
                    return;
                }
                dest[*out_idx] = x2_arr[i];
                *out_idx += 1;
            }
        }
        input_idx += 8;
    }
}

/// Reinterpret 8 contiguous `u32`s as `i32`, convert to `f32`, and
/// scale to `[-1, 1)`.
#[inline]
fn load_lanes_as_unit_f32(u32_lanes: &[u32], scale: f32x8) -> f32x8 {
    debug_assert!(u32_lanes.len() >= 8);
    let arr: [i32; 8] = [
        u32_lanes[0] as i32,
        u32_lanes[1] as i32,
        u32_lanes[2] as i32,
        u32_lanes[3] as i32,
        u32_lanes[4] as i32,
        u32_lanes[5] as i32,
        u32_lanes[6] as i32,
        u32_lanes[7] as i32,
    ];
    i32x8::from(arr).round_float() * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_xoshiro::Xoshiro256PlusPlus;

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

    /// Odd-length destinations (tail path: the second of an accepted
    /// pair is discarded by the early-return) must still satisfy
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

    /// Xoshiro256++ as the RNG also produces the right distribution
    /// — same property tests as for ChaCha20, just sanity-checking
    /// the hot-path RNG.
    #[test]
    fn xoshiro_unit_variance_zero_mean() {
        let mut rng = Xoshiro256PlusPlus::from_seed([29u8; 32]);
        let n = 100_000;
        let mut buf = vec![0.0_f32; n];
        fill_gaussian(&mut buf, 1.0, &mut rng);

        let mean: f64 = buf.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var: f64 =
            buf.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 0.02, "mean {mean} too far from 0");
        assert!((var - 1.0).abs() < 0.05, "var {var} too far from 1");
    }

    /// Pool-exhaustion path: artificially small destination (1 row)
    /// vs the larger-allocation case should both produce a single
    /// Gaussian without panicking.  Exercises the early-return inside
    /// the bit-walk loop.
    #[test]
    fn very_small_dest_does_not_panic() {
        let mut rng = ChaCha20Rng::from_seed([31u8; 32]);
        for n in [1usize, 2, 3, 7, 15, 16, 17] {
            let mut buf = vec![0.0_f32; n];
            fill_gaussian(&mut buf, 1.0, &mut rng);
            // Smoke: at least one non-zero entry (probability of all-
            // zero is ~(σ=1, n outputs ~10⁻⁸ for n=1) — accept very
            // unlikely; n ≥ 2 is essentially zero.
            assert!(buf.iter().any(|&v| v != 0.0));
        }
    }
}
