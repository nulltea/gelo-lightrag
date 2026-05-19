//! HD₃ Hadamard-cascade mask — structured-orthogonal alternative to
//! [`crate::mask::GeloMask`] (dense Haar) for the GELO protocol's
//! token-axis obfuscation.
//!
//! The mask matrix is `A = D₃ · H · D₂ · H · D₁ · H` where each `H` is
//! the orthonormal Walsh-Hadamard transform at the padded size `s_pad
//! ≥ s` (with `s_pad` the next power of two), and each `Dᵢ ∈ {-1,+1}^{s_pad}`
//! is a fresh ±1 diagonal sampled per forward pass. `A` is exactly
//! orthogonal: `Aᵀ · A = I` to f32 noise.
//!
//! ## Why HD₃
//!
//! At our long-context shape (s = 2 056 at n = 2 048 + 8 shield rows,
//! d ∈ {2 048, 6 144} per call) the dense Haar mask costs
//! `O(s³)` for the per-forward QR sample (~3.1 s of wall time at
//! threads=16) and `O(s² · d)` per `apply` / `unapply` call (~12 ms each
//! at threads=16). HD₃ replaces both:
//!
//! | op | dense Haar | HD₃ | asymptotic factor |
//! |---|---|---|---|
//! | sample (per forward) | `O(s³)` Householder QR | `O(s)` bit generation | `s²` |
//! | apply / unapply (per call) | `O(s²·d)` dense GEMM | `O(s·d·log s)` FWHT-cascade | `s/log s` |
//! | storage | `O(s²)` (~17 MB at s=2056) | `O(s)` (~6 KB) | `s` |
//!
//! Per-batch freshness: `3·s_pad` random bits = `2^{3·s_pad}` distinct
//! masks (≈ 2^{6144} at s_pad=2048; ≈ 2^{12288} at s_pad=4096) — a
//! discrete orbit inside `O(s)` rather than the continuous Haar
//! measure. The orbit is large enough that brute-force enumeration is
//! infeasible; the question of whether the orbit's *structure*
//! degrades the BSS distinguishing game vs dense Haar is the load-
//! bearing security gate (see the round-3 doc §2.1 plan B.3, attack-
//! suite reproduction).
//!
//! ## Power-of-two contract
//!
//! The Walsh-Hadamard transform is defined only at power-of-two
//! sizes; `Hd3Mask::fresh(s)` requires `s.is_power_of_two()` and
//! `apply` / `unapply` both expect operands of shape `(s, d)` where
//! `s == self.n()`. **Padding non-pow2 inputs is the caller's
//! responsibility** — `Hd3Mask` never modifies the row dimension.
//!
//! Why this API: orthogonality of `A` *is* preserved under
//! zero-padding (`Aᵀ · A · pad(x) = pad(x)`), but the round-trip
//! requires the full padded vector to flow through the GPU and back.
//! If the caller strips padding rows between apply and unapply, the
//! orthogonal mixing populates those rows with non-zero values that
//! get discarded — breaking the round-trip identity. Encoding the
//! pad as an explicit caller responsibility makes that data-flow
//! requirement visible.
//!
//! At our long-context shape `n = 2 048` + `k_shield = 8` → `s = 2 056`,
//! the caller pads to `s_pad = 4 096` (the next power of two). The
//! 2 040 padding rows can either be zero (no extra security; relies on
//! shield rows for Gram-leak mitigation) or Gaussian shield rows
//! (subsumes the existing `k_shield = 8` choice). The protocol
//! transmits `s_pad` rows to the engine — a `s_pad/s ≈ 2×` overhead
//! on the GPU matmul vs the dense-Haar baseline. The CPU mask cost
//! drops from `O(s²·d)` to `O(s·d·log s)` so the CPU side is still a
//! net win at long n.
//!
//! ## References
//!
//! - Tseng et al., *QuIP#*, ICML '24 ([arXiv:2402.04396](https://arxiv.org/abs/2402.04396)) — same cascade for LLM weight quantisation; proves Haar-like incoherence bounds.
//! - Ashkboos et al., *QuaRot* ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) — production CUDA kernels for the cascade.
//! - Ailon-Chazelle, *Fast JL Transform*, STOC '06 — single-stage randomized Hadamard transform, the building block.
//! - GELO paper §3.2 — security argument we inherit unchanged (shield rows + per-batch freshness + orthogonal mask = BSS-distinguishing-game hardness on the protected quantities).

use ndarray::{Array2, ArrayView2};
use rand::RngCore;
use rayon::prelude::*;

pub use crate::rng::MaskSeed;

/// Total butterfly work (`n_rows · d_cols`) above which `fwht_rows_inplace`
/// uses `rayon::par_chunks_mut` to parallelise butterfly pairs across
/// cores. Below this threshold the per-call rayon spawn overhead
/// (~100 µs) dominates the actual butterfly work, so sequential wins
/// — matches the embedder cliff measured under
/// `memory/blis_default_on_and_layer_skip_regression.md`.
///
/// Picked at 65 536 elements = 64 KB of f32 data per FWHT stage: a
/// 32×2048 stage or 256×256 stage is the smallest where parallelism
/// amortises spawn cost.
const FWHT_RAYON_WORK_THRESHOLD: usize = 65_536;

/// HD₃ Hadamard-cascade mask. Stores three ±1 diagonal vectors of
/// length `n` (power of two); the explicit mask matrix is never
/// materialised. See module docs for the math.
#[derive(Debug, Clone)]
pub struct Hd3Mask {
    /// Side length the mask operates on. **Must be a power of two.**
    /// Padding non-pow2 inputs is the caller's responsibility (see
    /// module docs).
    n: usize,
    /// First diagonal `D₁`, length `n`, ±1.0 values.
    d1: Vec<f32>,
    /// Second diagonal `D₂`, length `n`, ±1.0 values.
    d2: Vec<f32>,
    /// Third diagonal `D₃`, length `n`, ±1.0 values.
    d3: Vec<f32>,
    /// `1 / n^{3/2}` — the orthonormal-FWHT scaling collected from
    /// three Walsh-Hadamard transforms. Applied once at the end of
    /// apply/unapply so the inner butterfly loops stay
    /// integer-shift-add.
    inv_norm: f32,
}

impl Hd3Mask {
    /// Sample a fresh HD₃ mask at side length `n` (must be a power
    /// of two). Consumes `3·n` random bits. At n=4 096 (covering the
    /// padded long-context shape) that's ~1.5 kB —
    /// sub-microsecond on any modern RNG.
    pub fn fresh<R: RngCore>(n: usize, rng: &mut R) -> Self {
        assert!(
            n.is_power_of_two() && n > 0,
            "Hd3Mask::fresh: n must be a positive power of two, got {n}"
        );
        let sample_diag = |rng: &mut R| -> Vec<f32> {
            // Pack 32 random bits per RNG call.
            let mut out = Vec::with_capacity(n);
            let mut i = 0;
            while i < n {
                let mut bits = rng.next_u32();
                let take = (n - i).min(32);
                for _ in 0..take {
                    out.push(if bits & 1 == 0 { -1.0_f32 } else { 1.0_f32 });
                    bits >>= 1;
                }
                i += take;
            }
            out
        };
        let d1 = sample_diag(rng);
        let d2 = sample_diag(rng);
        let d3 = sample_diag(rng);
        let inv_norm = 1.0_f32 / (n as f32).powf(1.5);
        Self {
            n,
            d1,
            d2,
            d3,
            inv_norm,
        }
    }

    /// Deterministic constructor for tests.
    pub fn from_seed(n: usize, seed: MaskSeed) -> Self {
        let mut rng = seed.rng();
        Self::fresh(n, &mut rng)
    }

    /// Side length `n` (power of two). Matches the `GeloMask::n` API
    /// for drop-in interchangeability at pow2 shapes.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Apply the mask: `U = A · H`. `hidden` must have shape `(n, d)`
    /// where `n == self.n()` (power of two). Output has shape `(n, d)`.
    pub fn apply(&self, hidden: ArrayView2<'_, f32>) -> Array2<f32> {
        assert_eq!(
            hidden.nrows(),
            self.n,
            "Hd3Mask::apply: hidden has {} rows, expected {}",
            hidden.nrows(),
            self.n
        );
        let d = hidden.ncols();
        let mut buf = Array2::<f32>::zeros((self.n, d));
        if d > 0 {
            buf.assign(&hidden);
        }
        // A · x = D₃ · H · D₂ · H · D₁ · H · x.
        // Apply right-to-left: H first, then D₁, then H, then D₂, then H, then D₃.
        fwht_rows_inplace(&mut buf);
        apply_diag_inplace(&mut buf, &self.d1);
        fwht_rows_inplace(&mut buf);
        apply_diag_inplace(&mut buf, &self.d2);
        fwht_rows_inplace(&mut buf);
        apply_diag_inplace(&mut buf, &self.d3);
        scale_inplace(&mut buf, self.inv_norm);
        buf
    }

    /// Remove the mask: `H·W = Aᵀ · (U·W)`. `masked_output` must have
    /// shape `(n, p)` where `n == self.n()`. Output has shape `(n, p)`.
    pub fn unapply(&self, masked_output: ArrayView2<'_, f32>) -> Array2<f32> {
        assert_eq!(
            masked_output.nrows(),
            self.n,
            "Hd3Mask::unapply: masked_output has {} rows, expected {}",
            masked_output.nrows(),
            self.n
        );
        let p = masked_output.ncols();
        let mut buf = Array2::<f32>::zeros((self.n, p));
        if p > 0 {
            buf.assign(&masked_output);
        }
        // Aᵀ = (D₃·H·D₂·H·D₁·H)ᵀ = Hᵀ·D₁ᵀ·Hᵀ·D₂ᵀ·Hᵀ·D₃ᵀ
        //    = H·D₁·H·D₂·H·D₃   (since H = Hᵀ and Dᵢ = Dᵢᵀ).
        // Apply right-to-left: D₃ first, then H, then D₂, then H, then D₁, then H.
        apply_diag_inplace(&mut buf, &self.d3);
        fwht_rows_inplace(&mut buf);
        apply_diag_inplace(&mut buf, &self.d2);
        fwht_rows_inplace(&mut buf);
        apply_diag_inplace(&mut buf, &self.d1);
        fwht_rows_inplace(&mut buf);
        scale_inplace(&mut buf, self.inv_norm);
        buf
    }
}

/// In-place Walsh-Hadamard transform applied along axis 0 of a
/// row-major `Array2`. Equivalent to multiplying by the (unscaled)
/// Walsh-Hadamard matrix from the left for each column independently.
/// The scaling factor `1/sqrt(n)` per H is collected at the end of
/// `apply`/`unapply` via `inv_norm`, so this function leaves the data
/// in "raw FWHT" form.
///
/// Requires `n.is_power_of_two()`. Cost: `O(n · d · log₂ n)` add/sub
/// operations.
///
/// **Kernel dispatch**:
/// - x86_64 with AVX-512F: 16 f32 per inst, `_mm512_add_ps` + `_mm512_sub_ps`
/// - x86_64 with AVX2: 8 f32 per inst, `_mm256_add_ps` + `_mm256_sub_ps`
/// - else: scalar fallback (LLVM may auto-vectorise to SSE2)
///
/// **Parallelism**: when total work (`n · d`) ≥
/// [`FWHT_RAYON_WORK_THRESHOLD`], butterfly pairs within each stage
/// are processed via `rayon::par_chunks_mut` (chunks of 2·h rows).
/// Late stages (large `h`) get fewer chunks and so less rayon
/// parallelism, but those stages are also memory-bandwidth-bound so
/// adding threads past ~4 saturates DRAM regardless.
fn fwht_rows_inplace(m: &mut Array2<f32>) {
    let n = m.nrows();
    let d = m.ncols();
    debug_assert!(
        n.is_power_of_two(),
        "fwht_rows_inplace: row count {} must be a power of two",
        n
    );
    if n < 2 || d == 0 {
        return;
    }
    let slice = m
        .as_slice_mut()
        .expect("fwht_rows_inplace: matrix must be standard layout");

    let use_avx512 = avx512f_supported();
    let use_avx2 = !use_avx512 && avx2_supported();
    let total_work = n.saturating_mul(d);
    let use_rayon = total_work >= FWHT_RAYON_WORK_THRESHOLD;

    let mut h = 1;
    while h < n {
        let chunk_size = 2 * h * d;
        // SAFETY of inner butterfly calls: each butterfly's two
        // mutable slices (r0, r1) are disjoint (`r1` starts at offset
        // `h·d` past `r0` and both have length `d ≤ h·d`). Across
        // butterflies within a stage the slices are also disjoint
        // (different `(i + j)` ranges). `par_chunks_mut` guarantees
        // disjoint chunks across rayon threads.
        if use_rayon {
            slice.par_chunks_mut(chunk_size).for_each(|chunk| {
                process_stage_chunk(chunk, h, d, use_avx512, use_avx2);
            });
        } else {
            for chunk in slice.chunks_mut(chunk_size) {
                process_stage_chunk(chunk, h, d, use_avx512, use_avx2);
            }
        }
        h *= 2;
    }
}

/// Process one rayon chunk = `2·h` rows worth of buffer. Performs
/// `h` butterflies between rows `(j, j+h)` for `j in 0..h`.
#[inline]
fn process_stage_chunk(chunk: &mut [f32], h: usize, d: usize, use_avx512: bool, use_avx2: bool) {
    for j in 0..h {
        let r0_off = j * d;
        let r1_off = (j + h) * d;
        if r1_off + d > chunk.len() {
            break;
        }
        debug_assert!(r0_off + d <= r1_off);
        let (lo, hi) = chunk.split_at_mut(r1_off);
        let r0 = &mut lo[r0_off..r0_off + d];
        let r1 = &mut hi[..d];
        butterfly_pair(r0, r1, use_avx512, use_avx2);
    }
}

/// One butterfly pair: `(r0, r1) ← (r0 + r1, r0 - r1)`. Dispatches
/// to AVX-512, AVX-2, or scalar based on the runtime feature flags
/// (checked once per `fwht_rows_inplace` call and passed in).
#[inline]
fn butterfly_pair(r0: &mut [f32], r1: &mut [f32], use_avx512: bool, use_avx2: bool) {
    debug_assert_eq!(r0.len(), r1.len());
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512 {
            // SAFETY: caller checked `is_x86_feature_detected!("avx512f")`.
            unsafe {
                butterfly_pair_avx512(r0, r1);
            }
            return;
        }
        if use_avx2 {
            // SAFETY: caller checked `is_x86_feature_detected!("avx2")`.
            unsafe {
                butterfly_pair_avx2(r0, r1);
            }
            return;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Silence "unused" warnings on non-x86 builds.
        let _ = use_avx512;
        let _ = use_avx2;
    }
    butterfly_pair_scalar(r0, r1);
}

#[inline]
fn butterfly_pair_scalar(r0: &mut [f32], r1: &mut [f32]) {
    let d = r0.len();
    for k in 0..d {
        let x = r0[k];
        let y = r1[k];
        r0[k] = x + y;
        r1[k] = x - y;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn butterfly_pair_avx512(r0: &mut [f32], r1: &mut [f32]) {
    use std::arch::x86_64::*;
    let d = r0.len();
    let mut k = 0;
    while k + 16 <= d {
        // SAFETY: bounds checked by `k + 16 <= d` and r0.len() == r1.len() == d.
        unsafe {
            let p0 = r0.as_mut_ptr().add(k);
            let p1 = r1.as_mut_ptr().add(k);
            let v0 = _mm512_loadu_ps(p0);
            let v1 = _mm512_loadu_ps(p1);
            let sum = _mm512_add_ps(v0, v1);
            let diff = _mm512_sub_ps(v0, v1);
            _mm512_storeu_ps(p0, sum);
            _mm512_storeu_ps(p1, diff);
        }
        k += 16;
    }
    // Scalar tail for d % 16.
    while k < d {
        // SAFETY: k < d, both slices have length d.
        unsafe {
            let x = *r0.get_unchecked(k);
            let y = *r1.get_unchecked(k);
            *r0.get_unchecked_mut(k) = x + y;
            *r1.get_unchecked_mut(k) = x - y;
        }
        k += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn butterfly_pair_avx2(r0: &mut [f32], r1: &mut [f32]) {
    use std::arch::x86_64::*;
    let d = r0.len();
    let mut k = 0;
    while k + 8 <= d {
        unsafe {
            let p0 = r0.as_mut_ptr().add(k);
            let p1 = r1.as_mut_ptr().add(k);
            let v0 = _mm256_loadu_ps(p0);
            let v1 = _mm256_loadu_ps(p1);
            let sum = _mm256_add_ps(v0, v1);
            let diff = _mm256_sub_ps(v0, v1);
            _mm256_storeu_ps(p0, sum);
            _mm256_storeu_ps(p1, diff);
        }
        k += 8;
    }
    while k < d {
        unsafe {
            let x = *r0.get_unchecked(k);
            let y = *r1.get_unchecked(k);
            *r0.get_unchecked_mut(k) = x + y;
            *r1.get_unchecked_mut(k) = x - y;
        }
        k += 1;
    }
}

/// Cached AVX-512F support — `is_x86_feature_detected!` is fast but
/// still does a cpuid check internally. We call it many times per
/// forward pass; once-per-process is enough.
#[cfg(target_arch = "x86_64")]
fn avx512f_supported() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::is_x86_feature_detected!("avx512f"))
}

#[cfg(not(target_arch = "x86_64"))]
fn avx512f_supported() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn avx2_supported() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_supported() -> bool {
    false
}

/// In-place row-wise sign flip: `m[i, *] *= d[i]` for each row `i`.
/// `d[i]` is expected to be exactly ±1.0; when `d[i] = +1` the row is
/// untouched; when `d[i] = -1` the row is negated.
///
/// LLVM auto-vectorises the inner negation loop reliably on both
/// AVX-2 and AVX-512 (just a sign-bit XOR per element); no SIMD
/// intrinsics needed here. Rayon-parallel above the work threshold —
/// at small shapes the inner loop is already fast enough that
/// spawn overhead dominates.
fn apply_diag_inplace(m: &mut Array2<f32>, d: &[f32]) {
    let n_rows = m.nrows();
    let cols = m.ncols();
    debug_assert_eq!(
        d.len(),
        n_rows,
        "apply_diag_inplace: diagonal length {} ≠ row count {}",
        d.len(),
        n_rows
    );
    if cols == 0 {
        return;
    }
    let slice = m
        .as_slice_mut()
        .expect("apply_diag_inplace: matrix must be standard layout");
    let total_work = n_rows.saturating_mul(cols);
    if total_work >= FWHT_RAYON_WORK_THRESHOLD {
        // Parallelise row-by-row. Each row is independent.
        slice
            .par_chunks_mut(cols)
            .zip(d.par_iter())
            .for_each(|(row, &sign)| {
                if sign < 0.0 {
                    for v in row.iter_mut() {
                        *v = -*v;
                    }
                }
            });
    } else {
        for (row_idx, &sign) in d.iter().enumerate() {
            if sign < 0.0 {
                let row_offset = row_idx * cols;
                for v in &mut slice[row_offset..row_offset + cols] {
                    *v = -*v;
                }
            }
        }
    }
}

/// In-place scalar multiplication of every element by `factor`.
/// Rayon-parallel above the work threshold; LLVM auto-vectorises the
/// inner multiply for both AVX-2 and AVX-512.
fn scale_inplace(m: &mut Array2<f32>, factor: f32) {
    if factor == 1.0 {
        return;
    }
    let slice = m
        .as_slice_mut()
        .expect("scale_inplace: matrix must be standard layout");
    if slice.len() >= FWHT_RAYON_WORK_THRESHOLD {
        slice.par_iter_mut().for_each(|v| *v *= factor);
    } else {
        for v in slice.iter_mut() {
            *v *= factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn sample_normal(rng: &mut ChaCha20Rng, n: usize, d: usize) -> Array2<f32> {
        let normal = StandardNormal;
        Array2::from_shape_fn((n, d), |_| normal.sample(rng))
    }

    fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// `unapply(apply(H) · W) ≈ H · W` to f32 noise. The HD₃ cascade
    /// preserves the round-trip identity exactly in real arithmetic;
    /// f32 noise comes from FWHT accumulation depth `log₂ n` per H
    /// times three Hs, plus the matmul depth `k`.
    #[test]
    fn hd3_round_trip_preserves_matmul() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        for &(n, d, p) in &[
            (8usize, 8, 8),
            (16, 12, 8),
            (64, 64, 32),
            (256, 128, 128),
            (4096, 2048, 2048),   // long-context Q/K/V apply at pre-padded shape
            (4096, 6144, 2048),   // FfnDown apply at pre-padded shape
            (4096, 2048, 6144),   // gate/up unapply (output width 6144)
        ] {
            let mask = Hd3Mask::fresh(n, &mut rng);
            let h = sample_normal(&mut rng, n, d);
            let w = sample_normal(&mut rng, d, p);
            let target = h.dot(&w);

            let u = mask.apply(h.view());
            let v = u.dot(&w);
            let recovered = mask.unapply(v.view());

            assert_eq!(recovered.dim(), target.dim());
            // Tolerance: matmul depth d plus FWHT depth log₂(n) per H
            // times three Hs. Scale by max-abs of the target to
            // accommodate the worst element.
            let depth = d as f32 + 3.0 * (n as f32).log2();
            let target_max = target
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1.0);
            let tol = 8.0 * depth * f32::EPSILON * target_max;
            let err = max_abs(&recovered, &target);
            assert!(
                err <= tol,
                "round-trip max abs error at (n={n}, d={d}, p={p}): {err:.3e} > tol {tol:.3e}"
            );
        }
    }

    /// `Aᵀ · A == I` to f32 noise. The HD₃ cascade is orthogonal by
    /// construction; this checks that the implementation preserves
    /// the property in floating-point.
    #[test]
    fn hd3_orthogonality() {
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        for &n in &[8usize, 16, 32, 64, 128, 256] {
            let mask = Hd3Mask::fresh(n, &mut rng);
            // Materialise A = mask · I.
            let id = Array2::<f32>::eye(n);
            let a = mask.apply(id.view());
            let ata = a.t().dot(&a);
            let id_target = Array2::<f32>::eye(n);
            let depth = n as f32 + 3.0 * (n as f32).log2();
            let tol = 16.0 * depth * f32::EPSILON;
            let err = max_abs(&ata, &id_target);
            assert!(
                err <= tol,
                "AᵀA - I max abs error at n={n}: {err:.3e} > tol {tol:.3e}"
            );
        }
    }

    /// Different seeds produce different masks; same seed reproduces.
    #[test]
    fn hd3_deterministic_from_seed() {
        let seed_a = MaskSeed::from_bytes([42u8; 32]);
        let seed_b = MaskSeed::from_bytes([43u8; 32]);
        let m_a1 = Hd3Mask::from_seed(64, seed_a);
        let m_a2 = Hd3Mask::from_seed(64, seed_a);
        let m_b = Hd3Mask::from_seed(64, seed_b);
        assert_eq!(m_a1.d1, m_a2.d1);
        assert_eq!(m_a1.d2, m_a2.d2);
        assert_eq!(m_a1.d3, m_a2.d3);
        assert_ne!(m_a1.d1, m_b.d1);
    }

    /// `Hd3Mask::fresh` rejects non-power-of-two `n`.
    #[test]
    #[should_panic(expected = "must be a positive power of two")]
    fn hd3_rejects_non_pow2() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let _ = Hd3Mask::fresh(2056, &mut rng);
    }

    /// Round-trip stays accurate to ~1e-4 relative at the realistic
    /// long-context padded shape (n=4096, d=2048).
    #[test]
    fn hd3_round_trip_relative_error_at_long_n() {
        let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
        let n = 4096;
        let d = 2048;
        let p = 1024;
        let mask = Hd3Mask::fresh(n, &mut rng);
        let h = sample_normal(&mut rng, n, d);
        let w = sample_normal(&mut rng, d, p);
        let target = h.dot(&w);
        let u = mask.apply(h.view());
        let v = u.dot(&w);
        let recovered = mask.unapply(v.view());
        let target_rms = (target.iter().map(|v| v * v).sum::<f32>() / target.len() as f32).sqrt();
        let err_rms = ((recovered
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>())
            / target.len() as f32)
            .sqrt();
        assert!(
            err_rms / target_rms < 1e-4,
            "round-trip relative rms at long-n: {:.3e}",
            err_rms / target_rms
        );
    }
}
