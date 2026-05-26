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
pub(crate) const FWHT_RAYON_WORK_THRESHOLD: usize = 65_536;

/// When `h * 8 <= n`, `fwht_rows_inplace` fuses three radix-2 stages
/// (at distances `h`, `2h`, `4h`) into one radix-8 pass. The fused
/// butterfly takes 8 rows, runs three levels of add/sub in registers,
/// and writes 8 rows back — cutting memory traffic 3× vs the unfused
/// path. See `butterfly_oct_*` for the SIMD kernels.
///
/// Choice of radix-8: AVX-512 has 32 ZMM registers, so 8 input vectors
/// (16 floats each) + 8 intermediates fit comfortably. Radix-16 would
/// spill on AVX-512 and force per-stage stores anyway.
///
/// Tail handling: when `h * 8 > n` for the current `h`, fall back to
/// radix-2 for the remaining 1–2 stages.
const FWHT_RADIX8_FACTOR: usize = 8;

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
    ///
    /// Allocates a fresh `(n, d)` output buffer. For hot-path use the
    /// caller should prefer [`Self::apply_in_place`], which operates
    /// on a caller-supplied buffer and avoids the ~32 MB allocation +
    /// 32 MB hidden→buf copy per call at long-context shapes (see the
    /// `stacked_scratch` reuse path in [`crate::sim`]).
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
        self.apply_in_place(&mut buf);
        buf
    }

    /// Apply the mask in place: `buf ← A · buf`. The caller-supplied
    /// buffer must already contain the operand and have shape `(n, *)`
    /// where `n == self.n()` (power of two). Saves the allocation and
    /// hidden→buf copy that the allocating [`Self::apply`] pays per
    /// call — at our long-context shape (s_pad=4096, d=2048) that's
    /// ~32 MB of memory traffic eliminated per call.
    pub fn apply_in_place(&self, buf: &mut Array2<f32>) {
        assert_eq!(
            buf.nrows(),
            self.n,
            "Hd3Mask::apply_in_place: buf has {} rows, expected {}",
            buf.nrows(),
            self.n
        );
        // A · x = D₃ · H · D₂ · H · D₁ · H · x.
        // Apply right-to-left: H first, then D₁, then H, then D₂, then H, then D₃.
        // The trailing scalar `inv_norm` is fused into the final D₃ pass
        // (each row gets ±inv_norm) — saves one full-tensor read-write
        // vs a separate `scale_inplace`.
        fwht_rows_inplace(buf);
        apply_diag_inplace(buf, &self.d1);
        fwht_rows_inplace(buf);
        apply_diag_inplace(buf, &self.d2);
        fwht_rows_inplace(buf);
        apply_diag_scaled_inplace(buf, &self.d3, self.inv_norm);
    }

    /// Remove the mask: `H·W = Aᵀ · (U·W)`. `masked_output` must have
    /// shape `(n, p)` where `n == self.n()`. Output has shape `(n, p)`.
    ///
    /// As with [`Self::apply`], allocates a fresh output buffer. Hot
    /// paths that already own the engine output should prefer
    /// [`Self::unapply_in_place`] on that buffer directly.
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
        self.unapply_in_place(&mut buf);
        buf
    }

    /// Remove the mask in place: `buf ← Aᵀ · buf`. Same shape contract
    /// as [`Self::unapply`]. Avoids the per-call allocation + copy by
    /// mutating the engine's output buffer directly.
    /// Slice-flavored variant of [`Self::apply_in_place`] that operates
    /// directly on a row-major `(n, cols)` `&mut [f32]`. Used by the
    /// batched per-block dispatch in `crate::sim` to avoid materialising
    /// a per-block `Array2` (which would force `to_owned` + `assign`,
    /// doubling memory traffic on the rayon-parallel mask apply).
    pub fn apply_in_place_slice(&self, buf: &mut [f32], cols: usize) {
        assert_eq!(
            buf.len(),
            self.n.saturating_mul(cols),
            "Hd3Mask::apply_in_place_slice: buf has {} f32s, expected n={} * cols={}",
            buf.len(), self.n, cols,
        );
        fwht_rows_inplace_slice(buf, self.n, cols);
        apply_diag_inplace_slice(buf, &self.d1, cols);
        fwht_rows_inplace_slice(buf, self.n, cols);
        apply_diag_inplace_slice(buf, &self.d2, cols);
        fwht_rows_inplace_slice(buf, self.n, cols);
        apply_diag_scaled_inplace_slice(buf, &self.d3, cols, self.inv_norm);
    }

    pub fn unapply_in_place(&self, buf: &mut Array2<f32>) {
        assert_eq!(
            buf.nrows(),
            self.n,
            "Hd3Mask::unapply_in_place: buf has {} rows, expected {}",
            buf.nrows(),
            self.n
        );
        // Aᵀ = (D₃·H·D₂·H·D₁·H)ᵀ = Hᵀ·D₁ᵀ·Hᵀ·D₂ᵀ·Hᵀ·D₃ᵀ
        //    = H·D₁·H·D₂·H·D₃   (since H = Hᵀ and Dᵢ = Dᵢᵀ).
        // Apply right-to-left: D₃ first, then H, then D₂, then H, then D₁, then H.
        // The trailing `scale_inplace` is fused into the leading D₃ —
        // `inv_norm` commutes with the linear FWHT/diag cascade, so
        // scaling at the input gives identical output with one fewer
        // full-tensor pass.
        apply_diag_scaled_inplace(buf, &self.d3, self.inv_norm);
        fwht_rows_inplace(buf);
        apply_diag_inplace(buf, &self.d2);
        fwht_rows_inplace(buf);
        apply_diag_inplace(buf, &self.d1);
        fwht_rows_inplace(buf);
    }

    /// Slice-flavored variant of [`Self::unapply_in_place`]. See
    /// [`Self::apply_in_place_slice`] for the design motivation.
    pub fn unapply_in_place_slice(&self, buf: &mut [f32], cols: usize) {
        assert_eq!(
            buf.len(),
            self.n.saturating_mul(cols),
            "Hd3Mask::unapply_in_place_slice: buf has {} f32s, expected n={} * cols={}",
            buf.len(), self.n, cols,
        );
        apply_diag_scaled_inplace_slice(buf, &self.d3, cols, self.inv_norm);
        fwht_rows_inplace_slice(buf, self.n, cols);
        apply_diag_inplace_slice(buf, &self.d2, cols);
        fwht_rows_inplace_slice(buf, self.n, cols);
        apply_diag_inplace_slice(buf, &self.d1, cols);
        fwht_rows_inplace_slice(buf, self.n, cols);
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
    let slice = m
        .as_slice_mut()
        .expect("fwht_rows_inplace: matrix must be standard layout");
    fwht_rows_inplace_slice(slice, n, d);
}

/// Slice-flavored body of [`fwht_rows_inplace`]. Operates on a row-major
/// `(n, d)` buffer with `n` a power of two. Used by both the `Array2`
/// public wrapper and the `_slice` per-block path in
/// `crate::sim::build_per_sequence_masked` (avoids the `to_owned` +
/// `assign` that materialising a temporary `Array2` would force).
fn fwht_rows_inplace_slice(slice: &mut [f32], n: usize, d: usize) {
    debug_assert!(
        n.is_power_of_two(),
        "fwht_rows_inplace_slice: row count {} must be a power of two",
        n
    );
    debug_assert_eq!(
        slice.len(),
        n.saturating_mul(d),
        "fwht_rows_inplace_slice: slice has {} f32s, expected n={} * d={}",
        slice.len(), n, d,
    );
    if n < 2 || d == 0 {
        return;
    }
    let use_avx512 = avx512f_supported();
    let use_avx2 = !use_avx512 && avx2_supported();
    let total_work = n.saturating_mul(d);
    let use_rayon = total_work >= FWHT_RAYON_WORK_THRESHOLD;

    let mut h = 1;
    while h < n {
        // Prefer radix-8 (3 stages fused per pass) when the remaining
        // log-levels and the row count both allow it. This cuts memory
        // traffic 3× vs three separate radix-2 stages — the dominant
        // cost at our s_pad ≥ 2048 shapes.
        if h.checked_mul(FWHT_RADIX8_FACTOR).is_some_and(|next| next <= n) {
            let chunk_size = 8 * h * d;
            // SAFETY (per chunk): each radix-8 group processes 8 rows
            // at positions (j, j+h, ..., j+7h) within the chunk; all 8
            // are disjoint slices of length `d` separated by `h·d`.
            // Across groups (different `j`), the row-index sets are
            // pairwise disjoint. Across chunks, `par_chunks_mut`
            // guarantees disjointness.
            if use_rayon {
                slice.par_chunks_mut(chunk_size).for_each(|chunk| {
                    process_stage_chunk_radix8(chunk, h, d, use_avx512, use_avx2);
                });
            } else {
                for chunk in slice.chunks_mut(chunk_size) {
                    process_stage_chunk_radix8(chunk, h, d, use_avx512, use_avx2);
                }
            }
            h *= FWHT_RADIX8_FACTOR;
            continue;
        }
        // Tail: 1–2 remaining log-levels at this `h`. Fall back to the
        // original radix-2 path.
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

/// Process one rayon chunk = `8·h` rows worth of buffer for a radix-8
/// fused stage. For each `j in 0..h`, takes the 8 rows at positions
/// `(j, j+h, j+2h, …, j+7h)` and runs three levels of butterflies
/// (at distances `h`, `2h`, `4h`) in-register, writing back 8 rows.
///
/// Output ordering matches three sequential radix-2 stages at
/// `h`, `2h`, `4h`: see `radix8_matches_radix2` test for parity.
#[inline]
fn process_stage_chunk_radix8(
    chunk: &mut [f32],
    h: usize,
    d: usize,
    use_avx512: bool,
    use_avx2: bool,
) {
    let chunk_ptr = chunk.as_mut_ptr();
    let chunk_len = chunk.len();
    for j in 0..h {
        // Compute the 8 row offsets in the chunk.
        let off = [
            j * d,
            (j + h) * d,
            (j + 2 * h) * d,
            (j + 3 * h) * d,
            (j + 4 * h) * d,
            (j + 5 * h) * d,
            (j + 6 * h) * d,
            (j + 7 * h) * d,
        ];
        // Skip incomplete groups at the tail of an under-sized chunk
        // (only relevant if the caller passes a partial chunk; with
        // `chunks_mut(8·h·d)` the last chunk is always full when
        // `n` is a power of two and divisible by `8·h`).
        if off[7] + d > chunk_len {
            break;
        }
        // SAFETY: the 8 offsets are pairwise disjoint (j + i·h for
        // i = 0..8 are distinct since h > 0) and each row spans
        // `[off[i], off[i] + d)` which is contained in the chunk by
        // the bound check above. `chunk_ptr` is unique to this rayon
        // chunk so no aliasing with other workers.
        unsafe {
            let rows: [*mut f32; 8] = [
                chunk_ptr.add(off[0]),
                chunk_ptr.add(off[1]),
                chunk_ptr.add(off[2]),
                chunk_ptr.add(off[3]),
                chunk_ptr.add(off[4]),
                chunk_ptr.add(off[5]),
                chunk_ptr.add(off[6]),
                chunk_ptr.add(off[7]),
            ];
            butterfly_oct(rows, d, use_avx512, use_avx2);
        }
    }
}

/// Three radix-2 butterfly stages fused over 8 rows. Output ordering
/// matches three sequential radix-2 passes at `h`, `2h`, `4h`:
///
/// ```text
/// y0 = x0+x1+x2+x3 + x4+x5+x6+x7
/// y1 = x0-x1+x2-x3 + x4-x5+x6-x7
/// y2 = (x0+x1)-(x2+x3) + (x4+x5)-(x6+x7)
/// y3 = (x0-x1)-(x2-x3) + (x4-x5)-(x6-x7)
/// y4 = x0+x1+x2+x3 - (x4+x5+x6+x7)
/// y5 = (x0-x1+x2-x3) - (x4-x5+x6-x7)
/// y6 = ((x0+x1)-(x2+x3)) - ((x4+x5)-(x6+x7))
/// y7 = ((x0-x1)-(x2-x3)) - ((x4-x5)-(x6-x7))
/// ```
///
/// SAFETY: caller must pass 8 disjoint, valid-for-`d`-f32-writes row
/// pointers. See `process_stage_chunk_radix8` for the disjointness
/// argument.
#[inline]
unsafe fn butterfly_oct(rows: [*mut f32; 8], d: usize, use_avx512: bool, use_avx2: bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512 {
            // SAFETY: caller checked `is_x86_feature_detected!("avx512f")`.
            unsafe {
                butterfly_oct_avx512(rows, d);
            }
            return;
        }
        if use_avx2 {
            // SAFETY: caller checked `is_x86_feature_detected!("avx2")`.
            unsafe {
                butterfly_oct_avx2(rows, d);
            }
            return;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = use_avx512;
        let _ = use_avx2;
    }
    // SAFETY: forwarded from caller.
    unsafe {
        butterfly_oct_scalar(rows, d);
    }
}

/// Scalar fallback for `butterfly_oct`. LLVM may auto-vectorise, but
/// AVX-2/512 paths above are explicit for predictability.
#[inline]
unsafe fn butterfly_oct_scalar(rows: [*mut f32; 8], d: usize) {
    for k in 0..d {
        // SAFETY: `rows[i]` is valid for `d` writes (caller invariant);
        // `k < d` so each `add(k)` stays in-bounds.
        unsafe {
            let x0 = *rows[0].add(k);
            let x1 = *rows[1].add(k);
            let x2 = *rows[2].add(k);
            let x3 = *rows[3].add(k);
            let x4 = *rows[4].add(k);
            let x5 = *rows[5].add(k);
            let x6 = *rows[6].add(k);
            let x7 = *rows[7].add(k);
            // Stage 1: butterfly pairs (0,1), (2,3), (4,5), (6,7)
            let s01 = x0 + x1;
            let d01 = x0 - x1;
            let s23 = x2 + x3;
            let d23 = x2 - x3;
            let s45 = x4 + x5;
            let d45 = x4 - x5;
            let s67 = x6 + x7;
            let d67 = x6 - x7;
            // Stage 2: butterflies (0,2), (1,3), (4,6), (5,7) over Stage-1 outputs
            let p0 = s01 + s23;
            let p2 = s01 - s23;
            let p1 = d01 + d23;
            let p3 = d01 - d23;
            let p4 = s45 + s67;
            let p6 = s45 - s67;
            let p5 = d45 + d67;
            let p7 = d45 - d67;
            // Stage 3: butterflies (0,4), (1,5), (2,6), (3,7) over Stage-2 outputs
            *rows[0].add(k) = p0 + p4;
            *rows[4].add(k) = p0 - p4;
            *rows[1].add(k) = p1 + p5;
            *rows[5].add(k) = p1 - p5;
            *rows[2].add(k) = p2 + p6;
            *rows[6].add(k) = p2 - p6;
            *rows[3].add(k) = p3 + p7;
            *rows[7].add(k) = p3 - p7;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn butterfly_oct_avx512(rows: [*mut f32; 8], d: usize) {
    use std::arch::x86_64::*;
    let mut k = 0;
    while k + 16 <= d {
        // SAFETY: bounds checked by `k + 16 <= d`; each row valid for d writes.
        unsafe {
            let v0 = _mm512_loadu_ps(rows[0].add(k));
            let v1 = _mm512_loadu_ps(rows[1].add(k));
            let v2 = _mm512_loadu_ps(rows[2].add(k));
            let v3 = _mm512_loadu_ps(rows[3].add(k));
            let v4 = _mm512_loadu_ps(rows[4].add(k));
            let v5 = _mm512_loadu_ps(rows[5].add(k));
            let v6 = _mm512_loadu_ps(rows[6].add(k));
            let v7 = _mm512_loadu_ps(rows[7].add(k));
            // Stage 1
            let s01 = _mm512_add_ps(v0, v1);
            let d01 = _mm512_sub_ps(v0, v1);
            let s23 = _mm512_add_ps(v2, v3);
            let d23 = _mm512_sub_ps(v2, v3);
            let s45 = _mm512_add_ps(v4, v5);
            let d45 = _mm512_sub_ps(v4, v5);
            let s67 = _mm512_add_ps(v6, v7);
            let d67 = _mm512_sub_ps(v6, v7);
            // Stage 2
            let p0 = _mm512_add_ps(s01, s23);
            let p2 = _mm512_sub_ps(s01, s23);
            let p1 = _mm512_add_ps(d01, d23);
            let p3 = _mm512_sub_ps(d01, d23);
            let p4 = _mm512_add_ps(s45, s67);
            let p6 = _mm512_sub_ps(s45, s67);
            let p5 = _mm512_add_ps(d45, d67);
            let p7 = _mm512_sub_ps(d45, d67);
            // Stage 3
            let q0 = _mm512_add_ps(p0, p4);
            let q4 = _mm512_sub_ps(p0, p4);
            let q1 = _mm512_add_ps(p1, p5);
            let q5 = _mm512_sub_ps(p1, p5);
            let q2 = _mm512_add_ps(p2, p6);
            let q6 = _mm512_sub_ps(p2, p6);
            let q3 = _mm512_add_ps(p3, p7);
            let q7 = _mm512_sub_ps(p3, p7);
            _mm512_storeu_ps(rows[0].add(k), q0);
            _mm512_storeu_ps(rows[1].add(k), q1);
            _mm512_storeu_ps(rows[2].add(k), q2);
            _mm512_storeu_ps(rows[3].add(k), q3);
            _mm512_storeu_ps(rows[4].add(k), q4);
            _mm512_storeu_ps(rows[5].add(k), q5);
            _mm512_storeu_ps(rows[6].add(k), q6);
            _mm512_storeu_ps(rows[7].add(k), q7);
        }
        k += 16;
    }
    // Scalar tail for d % 16.
    while k < d {
        // SAFETY: k < d, all rows valid for d writes.
        unsafe {
            butterfly_oct_one_scalar_lane(rows, k);
        }
        k += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn butterfly_oct_avx2(rows: [*mut f32; 8], d: usize) {
    use std::arch::x86_64::*;
    let mut k = 0;
    while k + 8 <= d {
        // SAFETY: bounds checked.
        unsafe {
            let v0 = _mm256_loadu_ps(rows[0].add(k));
            let v1 = _mm256_loadu_ps(rows[1].add(k));
            let v2 = _mm256_loadu_ps(rows[2].add(k));
            let v3 = _mm256_loadu_ps(rows[3].add(k));
            let v4 = _mm256_loadu_ps(rows[4].add(k));
            let v5 = _mm256_loadu_ps(rows[5].add(k));
            let v6 = _mm256_loadu_ps(rows[6].add(k));
            let v7 = _mm256_loadu_ps(rows[7].add(k));
            let s01 = _mm256_add_ps(v0, v1);
            let d01 = _mm256_sub_ps(v0, v1);
            let s23 = _mm256_add_ps(v2, v3);
            let d23 = _mm256_sub_ps(v2, v3);
            let s45 = _mm256_add_ps(v4, v5);
            let d45 = _mm256_sub_ps(v4, v5);
            let s67 = _mm256_add_ps(v6, v7);
            let d67 = _mm256_sub_ps(v6, v7);
            let p0 = _mm256_add_ps(s01, s23);
            let p2 = _mm256_sub_ps(s01, s23);
            let p1 = _mm256_add_ps(d01, d23);
            let p3 = _mm256_sub_ps(d01, d23);
            let p4 = _mm256_add_ps(s45, s67);
            let p6 = _mm256_sub_ps(s45, s67);
            let p5 = _mm256_add_ps(d45, d67);
            let p7 = _mm256_sub_ps(d45, d67);
            let q0 = _mm256_add_ps(p0, p4);
            let q4 = _mm256_sub_ps(p0, p4);
            let q1 = _mm256_add_ps(p1, p5);
            let q5 = _mm256_sub_ps(p1, p5);
            let q2 = _mm256_add_ps(p2, p6);
            let q6 = _mm256_sub_ps(p2, p6);
            let q3 = _mm256_add_ps(p3, p7);
            let q7 = _mm256_sub_ps(p3, p7);
            _mm256_storeu_ps(rows[0].add(k), q0);
            _mm256_storeu_ps(rows[1].add(k), q1);
            _mm256_storeu_ps(rows[2].add(k), q2);
            _mm256_storeu_ps(rows[3].add(k), q3);
            _mm256_storeu_ps(rows[4].add(k), q4);
            _mm256_storeu_ps(rows[5].add(k), q5);
            _mm256_storeu_ps(rows[6].add(k), q6);
            _mm256_storeu_ps(rows[7].add(k), q7);
        }
        k += 8;
    }
    while k < d {
        unsafe {
            butterfly_oct_one_scalar_lane(rows, k);
        }
        k += 1;
    }
}

/// Single-lane scalar octet butterfly used by the SIMD tails of
/// `butterfly_oct_avx512` / `butterfly_oct_avx2`. Inlined to match the
/// load/compute/store pattern of the scalar fallback.
#[inline]
unsafe fn butterfly_oct_one_scalar_lane(rows: [*mut f32; 8], k: usize) {
    // SAFETY: caller guarantees k < d and rows valid for d writes.
    unsafe {
        let x0 = *rows[0].add(k);
        let x1 = *rows[1].add(k);
        let x2 = *rows[2].add(k);
        let x3 = *rows[3].add(k);
        let x4 = *rows[4].add(k);
        let x5 = *rows[5].add(k);
        let x6 = *rows[6].add(k);
        let x7 = *rows[7].add(k);
        let s01 = x0 + x1;
        let d01 = x0 - x1;
        let s23 = x2 + x3;
        let d23 = x2 - x3;
        let s45 = x4 + x5;
        let d45 = x4 - x5;
        let s67 = x6 + x7;
        let d67 = x6 - x7;
        let p0 = s01 + s23;
        let p2 = s01 - s23;
        let p1 = d01 + d23;
        let p3 = d01 - d23;
        let p4 = s45 + s67;
        let p6 = s45 - s67;
        let p5 = d45 + d67;
        let p7 = d45 - d67;
        *rows[0].add(k) = p0 + p4;
        *rows[4].add(k) = p0 - p4;
        *rows[1].add(k) = p1 + p5;
        *rows[5].add(k) = p1 - p5;
        *rows[2].add(k) = p2 + p6;
        *rows[6].add(k) = p2 - p6;
        *rows[3].add(k) = p3 + p7;
        *rows[7].add(k) = p3 - p7;
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
    let cols = m.ncols();
    let slice = m
        .as_slice_mut()
        .expect("apply_diag_inplace: matrix must be standard layout");
    apply_diag_inplace_slice(slice, d, cols);
}

fn apply_diag_inplace_slice(slice: &mut [f32], d: &[f32], cols: usize) {
    let n_rows = d.len();
    debug_assert_eq!(
        slice.len(),
        n_rows.saturating_mul(cols),
        "apply_diag_inplace_slice: slice has {} f32s, expected d.len()={} * cols={}",
        slice.len(), n_rows, cols,
    );
    if cols == 0 {
        return;
    }
    let total_work = n_rows.saturating_mul(cols);
    if total_work >= FWHT_RAYON_WORK_THRESHOLD {
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

/// Fused diagonal + scalar pass: `m[i, *] *= d[i] * factor` for each
/// row `i`. Replaces the `apply_diag_inplace(m, d); scale_inplace(m, factor)`
/// pair with a single full-tensor pass — saves one read-modify-write
/// cycle (~32 MB at the long-context shape n=4096 d=2048).
///
/// `d[i]` is expected to be ±1.0; the effective per-row multiplier is
/// therefore `±factor`. Unlike `apply_diag_inplace` (which can skip
/// rows where `d[i] = +1`), this pass always touches every row because
/// `factor != 1`.
fn apply_diag_scaled_inplace(m: &mut Array2<f32>, d: &[f32], factor: f32) {
    let cols = m.ncols();
    let slice = m
        .as_slice_mut()
        .expect("apply_diag_scaled_inplace: matrix must be standard layout");
    apply_diag_scaled_inplace_slice(slice, d, cols, factor);
}

fn apply_diag_scaled_inplace_slice(slice: &mut [f32], d: &[f32], cols: usize, factor: f32) {
    let n_rows = d.len();
    debug_assert_eq!(
        slice.len(),
        n_rows.saturating_mul(cols),
        "apply_diag_scaled_inplace_slice: slice has {} f32s, expected d.len()={} * cols={}",
        slice.len(), n_rows, cols,
    );
    if cols == 0 {
        return;
    }
    let total_work = n_rows.saturating_mul(cols);
    if total_work >= FWHT_RAYON_WORK_THRESHOLD {
        slice
            .par_chunks_mut(cols)
            .zip(d.par_iter())
            .for_each(|(row, &sign)| {
                let m = sign * factor;
                for v in row.iter_mut() {
                    *v *= m;
                }
            });
    } else {
        for (row_idx, &sign) in d.iter().enumerate() {
            let row_offset = row_idx * cols;
            let mult = sign * factor;
            for v in &mut slice[row_offset..row_offset + cols] {
                *v *= mult;
            }
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

    /// Direct comparison of `fwht_rows_inplace` (radix-8 dispatch path)
    /// against an explicit Walsh-Hadamard reference `H[i,j] =
    /// (-1)^popcount(i & j)`. This pins the radix-8 output ordering
    /// — orthogonality + round-trip tests alone allow any consistent
    /// orthogonal transform, so they wouldn't catch a wrong-but-self-
    /// consistent radix-8.
    ///
    /// Covers all shape regimes:
    /// - n=2, 4: radix-2 only (no radix-8 stage fires).
    /// - n=8, 64, 512: radix-8 only (no radix-2 tail).
    /// - n=16, 32: radix-8 + 1–2 radix-2 tail stages.
    /// - n=128, 256: radix-8 + radix-2 mixed.
    #[test]
    fn radix8_matches_walsh_reference() {
        for &n in &[2usize, 4, 8, 16, 32, 64, 128, 256, 512] {
            for &d in &[1usize, 7, 16, 17, 32] {
                let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
                let m = sample_normal(&mut rng, n, d);
                let mut buf = m.clone();
                fwht_rows_inplace(&mut buf);
                // Reference: H @ M, computed column-wise from the
                // explicit Walsh-Hadamard matrix in natural order.
                let mut expected = Array2::<f32>::zeros((n, d));
                for i in 0..n {
                    for j in 0..n {
                        let sign: f32 = if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                        if sign != 0.0 {
                            for c in 0..d {
                                expected[[i, c]] += sign * m[[j, c]];
                            }
                        }
                    }
                }
                let err = max_abs(&buf, &expected);
                // FWHT accumulation depth = log₂(n). Plus a few ULPs
                // for SIMD reordering.
                let tol = 16.0 * (n as f32).log2().max(1.0) * f32::EPSILON
                    * expected
                        .iter()
                        .map(|v| v.abs())
                        .fold(0.0_f32, f32::max)
                        .max(1.0);
                assert!(
                    err <= tol,
                    "fwht radix-8 vs Walsh reference at (n={n}, d={d}): err {err:.3e} > tol {tol:.3e}"
                );
            }
        }
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
