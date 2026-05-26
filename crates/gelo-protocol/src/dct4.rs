//! DCT-IV cascade mask — non-pow2 alternative to [`crate::hd3::Hd3Mask`].
//!
//! `A = D₃ · C · D₂ · C · D₁ · C` where each `C` is the orthonormal
//! DCT-IV transform of side length `n` and each `Dᵢ ∈ {-1,+1}^{n}` is
//! a fresh ±1 diagonal sampled per forward pass. The construction
//! mirrors [`crate::hd3::Hd3Mask`] with two structural differences:
//!
//! 1. **Works at arbitrary `n`** (no power-of-two requirement). DCT-IV
//!    has `O(n log n)` algorithms at any `n` (Bluestein / Lee-Wang).
//!    This eliminates the `s_pad = n.next_power_of_two()` pad penalty
//!    that costs ~50 % of TTFT for HD₃ at non-pow2 shapes (e.g.,
//!    n=2056 at Qwen3-4B: HD₃ pads to 4096 and pays 2× GPU GEMM cost;
//!    DCT-IV operates directly at n=2056).
//!
//! 2. **DCT-IV is self-inverse** (`C · C = I` exactly in the orthonormal
//!    form), so apply and unapply use the same transform with reversed
//!    cascade order. Same as Hadamard.
//!
//! ## Why DCT-IV (not DCT-II)
//!
//! DCT-II rows include a constant first row `C[0, :] = 1/√n` —
//! after `D₁ · C · x`, the first output coordinate is
//! `(D₁)_0 · (sum of x components) / √n`, which leaks information about
//! the input row-sum and one diagonal sign. DCT-IV has no constant
//! row; all rows are balanced cosine sequences with entries bounded
//! by `√(2/n)` (matching HD₃'s `1/√n` incoherence bound).
//!
//! See `docs/research/hd3-non-pow2-fix.md` §6 for the security
//! argument and threat-model survey that motivated DCT-IV over DCT-II,
//! block-diagonal HD₃, or other candidates.
//!
//! ## Cost (4B at n=2048, s = n + k_shield = 2056)
//!
//! | op | dense Haar | HD₃ at pow2 (s=2048) | HD₃ pad→4096 | DCT-IV (s=2056) |
//! |---|---|---|---|---|
//! | sample (per forward) | `O(s³)` | `O(s)` | `O(s)` | `O(s)` |
//! | apply / unapply (per call) | `O(s²·d)` | `O(s·d·log s)` | `O(s·d·log s_pad)` | `O(s·d·log s)` |
//! | GPU rows transmitted | `s` | `s` | `s_pad` (2× regression) | `s` (no pad) |
//!
//! DCT-IV per-call is ~3× slower than FWHT on CPU (one DCT-IV ≈ one
//! length-n real FFT plus pre/post twiddles via Bluestein at non-pow2
//! `n`), but eliminates the GPU pad regression entirely.
//!
//! ## References
//!
//! - Wang, *On Computing the Discrete Fourier and Cosine Transforms*, 1985 — O(N log N) DCT-IV recursion at any N.
//! - Tolimieri-An-Lu, *Algorithms for Discrete Fourier Transform and Convolution*, 1997 — FFT-via-DCT-IV equivalence.
//! - `rustdct` 0.7 — production DCT-IV implementation we delegate to.

use std::sync::Arc;

use ndarray::Array2;
use rand::RngCore;
use rayon::prelude::*;
use rustdct::{Dct4, DctPlanner};

pub use crate::rng::MaskSeed;

/// Inner-DCT-IV work threshold above which `apply_in_place` parallelises
/// across columns via rayon. Below this the per-call rayon spawn
/// overhead (~100 µs) dominates the per-column DCT cost. Picked at
/// 2 048 columns × n rows ≈ 4 M FLOPs of inner work — slightly above
/// the threshold used in [`crate::hd3`].
const DCT4_RAYON_COL_THRESHOLD: usize = 64;

/// DCT-IV Hadamard-like cascade mask. Stores three ±1 diagonal vectors
/// of length `n` and an `Arc`-shared `Dct4<f32>` planner output.
///
/// `n` is unconstrained — works at any positive integer.
pub struct Dct4Mask {
    /// Side length the mask operates on.
    n: usize,
    /// First diagonal `D₁`, length `n`, ±1.0 values.
    d1: Vec<f32>,
    /// Second diagonal `D₂`, length `n`, ±1.0 values.
    d2: Vec<f32>,
    /// Third diagonal `D₃`, length `n`, ±1.0 values.
    d3: Vec<f32>,
    /// Per-pass normalisation collected across three DCT-IV invocations.
    /// `rustdct`'s DCT-IV is unnormalised: applying it twice scales
    /// each entry by `n/2` (empirically verified: a unit impulse at
    /// `n=8` recovers `4 · impulse` after two passes). So one pass is
    /// `√(n/2) · C_orthonormal · x`. After three passes the cumulative
    /// factor is `(n/2)^{3/2}`; we apply `(2/n)^{3/2}` once at the end
    /// of apply/unapply so the inner passes stay scale-free.
    inv_norm: f32,
    /// Cached DCT-IV planner output. Shared across calls; rustdct
    /// designs are `Send + Sync` so this is safe across rayon threads.
    dct4: Arc<dyn Dct4<f32> + Send + Sync>,
}

impl std::fmt::Debug for Dct4Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dct4Mask")
            .field("n", &self.n)
            .field("inv_norm", &self.inv_norm)
            // Omit dct4 (no Debug impl).
            .finish_non_exhaustive()
    }
}

impl Clone for Dct4Mask {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            d1: self.d1.clone(),
            d2: self.d2.clone(),
            d3: self.d3.clone(),
            inv_norm: self.inv_norm,
            dct4: Arc::clone(&self.dct4),
        }
    }
}

impl Dct4Mask {
    /// Sample a fresh DCT-IV mask at side length `n` (any positive int).
    /// Consumes `3·n` random bits — same orbit cardinality as HD₃ but
    /// without the pow2 constraint.
    pub fn fresh<R: RngCore>(n: usize, rng: &mut R) -> Self {
        assert!(n > 0, "Dct4Mask::fresh: n must be positive, got {n}");
        let sample_diag = |rng: &mut R| -> Vec<f32> {
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
        // rustdct's DCT-IV is unnormalised. Three passes accumulate
        // `(n/2)^{3/2}`, so `inv_norm = (2/n)^{3/2}` makes the
        // composed operator exactly orthogonal.
        let inv_norm = (2.0_f32 / n as f32).powf(1.5);

        // Planner is cheap; rustdct internally caches plans by length.
        let dct4 = DctPlanner::new().plan_dct4(n);
        Self {
            n,
            d1,
            d2,
            d3,
            inv_norm,
            dct4,
        }
    }

    /// Deterministic constructor for tests.
    pub fn from_seed(n: usize, seed: MaskSeed) -> Self {
        let mut rng = seed.rng();
        Self::fresh(n, &mut rng)
    }

    /// Side length `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Apply the mask: `U = A · H`. `hidden` must have shape `(n, d)`.
    /// Returns a freshly-allocated `(n, d)` output buffer.
    ///
    /// Hot paths should prefer [`Self::apply_in_place`] to avoid the
    /// allocation + copy.
    pub fn apply(&self, hidden: ndarray::ArrayView2<'_, f32>) -> Array2<f32> {
        assert_eq!(
            hidden.nrows(),
            self.n,
            "Dct4Mask::apply: hidden has {} rows, expected {}",
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

    /// Apply the mask in place: `buf ← A · buf` (`A = D₃·C·D₂·C·D₁·C`).
    /// Buffer must have shape `(n, *)`.
    pub fn apply_in_place(&self, buf: &mut Array2<f32>) {
        assert_eq!(
            buf.nrows(),
            self.n,
            "Dct4Mask::apply_in_place: buf has {} rows, expected {}",
            buf.nrows(),
            self.n
        );
        let d = buf.ncols();
        let slice = buf
            .as_slice_mut()
            .expect("Dct4Mask::apply_in_place: buffer must be standard layout");
        dct4_cascade_apply_inplace_slice(
            slice,
            self.n,
            d,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    /// Remove the mask: `H·W = Aᵀ · (U·W)`. `masked_output` must have
    /// shape `(n, p)`. Returns a freshly-allocated `(n, p)` buffer.
    ///
    /// Hot paths should prefer [`Self::unapply_in_place`].
    pub fn unapply(&self, masked_output: ndarray::ArrayView2<'_, f32>) -> Array2<f32> {
        assert_eq!(
            masked_output.nrows(),
            self.n,
            "Dct4Mask::unapply: masked_output has {} rows, expected {}",
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
    /// as [`Self::unapply`].
    ///
    /// `Aᵀ = (D₃·C·D₂·C·D₁·C)ᵀ = Cᵀ·D₁ᵀ·Cᵀ·D₂ᵀ·Cᵀ·D₃ᵀ
    ///     = C·D₁·C·D₂·C·D₃` (since `C = Cᵀ` for DCT-IV and `Dᵢ = Dᵢᵀ`).
    /// Slice-flavored variant of [`Self::apply_in_place`] — see
    /// [`crate::hd3::Hd3Mask::apply_in_place_slice`] for the rationale.
    pub fn apply_in_place_slice(&self, buf: &mut [f32], cols: usize) {
        assert_eq!(
            buf.len(),
            self.n.saturating_mul(cols),
            "Dct4Mask::apply_in_place_slice: buf has {} f32s, expected n={} * cols={}",
            buf.len(), self.n, cols,
        );
        dct4_cascade_apply_inplace_slice(
            buf,
            self.n,
            cols,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    pub fn unapply_in_place(&self, buf: &mut Array2<f32>) {
        assert_eq!(
            buf.nrows(),
            self.n,
            "Dct4Mask::unapply_in_place: buf has {} rows, expected {}",
            buf.nrows(),
            self.n
        );
        let d = buf.ncols();
        let slice = buf
            .as_slice_mut()
            .expect("Dct4Mask::unapply_in_place: buffer must be standard layout");
        dct4_cascade_unapply_inplace_slice(
            slice,
            self.n,
            d,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    /// Slice-flavored variant of [`Self::unapply_in_place`].
    pub fn unapply_in_place_slice(&self, buf: &mut [f32], cols: usize) {
        assert_eq!(
            buf.len(),
            self.n.saturating_mul(cols),
            "Dct4Mask::unapply_in_place_slice: buf has {} f32s, expected n={} * cols={}",
            buf.len(), self.n, cols,
        );
        dct4_cascade_unapply_inplace_slice(
            buf,
            self.n,
            cols,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }
}

/// In-place DCT-IV applied along axis 0 of a row-major `Array2`.
/// Equivalent to multiplying by the (unnormalised) DCT-IV matrix from
/// the left for each column independently.
///
/// Implementation: copy each column into a contiguous length-n scratch,
/// run `rustdct::Dct4::process_dct4` (with its internal scratch), copy
/// back. Rayon-parallel over columns when `d ≥ DCT4_RAYON_COL_THRESHOLD`.
///
/// **Superseded** by the tile-fused cascade
/// `dct4_cascade_apply_inplace_slice` — kept here as a parity-test
/// fallback. Will be removed after the cascade ships measurement
/// validation.
#[allow(dead_code)]
fn dct4_cols_inplace(buf: &mut Array2<f32>, dct4: &(dyn Dct4<f32> + Send + Sync)) {
    let n = buf.nrows();
    let d = buf.ncols();
    let slice = buf
        .as_slice_mut()
        .expect("dct4_cols_inplace: matrix must be standard layout");
    dct4_cols_inplace_slice(slice, n, d, dct4);
}

/// Slice-flavored body of [`dct4_cols_inplace`]. Operates on a row-major
/// `(n, d)` `&mut [f32]` directly so the batched per-block dispatch in
/// `crate::sim` can avoid materialising a per-block `Array2`.
#[allow(dead_code)]
fn dct4_cols_inplace_slice(
    slice: &mut [f32],
    n: usize,
    d: usize,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(
        slice.len(),
        n.saturating_mul(d),
        "dct4_cols_inplace_slice: slice has {} f32s, expected n={} * d={}",
        slice.len(), n, d,
    );
    if n < 2 || d == 0 {
        return;
    }

    // Process columns. Each column is strided (stride `d` in row-major
    // (n, d) layout). We copy-out → DCT → copy-back per column.
    if d >= DCT4_RAYON_COL_THRESHOLD {
        // Rayon over column index. Each thread allocates its own
        // column buffer + DCT scratch (via thread-local cache below).
        (0..d).into_par_iter().for_each(|j| {
            COL_SCRATCH.with(|cell| {
                let mut state = cell.borrow_mut();
                state.col.resize(n, 0.0);
                let dct_scratch_len = dct4.get_scratch_len();
                if state.scratch.len() < dct_scratch_len {
                    state.scratch.resize(dct_scratch_len, 0.0);
                }
                // Split the RefMut into two non-overlapping slice borrows.
                let ColScratch { col, scratch } = &mut *state;
                let col = &mut col[..n];
                let scratch = &mut scratch[..dct_scratch_len];

                // Copy column j out: row i has slice[i·d + j].
                // SAFETY: bounds checked by buf shape; disjoint
                // columns across rayon workers (different j).
                let base = slice.as_ptr();
                for i in 0..n {
                    // SAFETY: `i·d + j < n·d = slice.len()`.
                    unsafe {
                        col[i] = *base.add(i * d + j);
                    }
                }

                dct4.process_dct4_with_scratch(col, scratch);

                // SAFETY: same bounds; writes to disjoint column.
                let base_mut = slice.as_ptr() as *mut f32;
                for i in 0..n {
                    unsafe {
                        *base_mut.add(i * d + j) = col[i];
                    }
                }
            });
        });
    } else {
        let mut col = vec![0.0_f32; n];
        let mut scratch = vec![0.0_f32; dct4.get_scratch_len()];
        for j in 0..d {
            for i in 0..n {
                col[i] = slice[i * d + j];
            }
            dct4.process_dct4_with_scratch(&mut col, &mut scratch);
            for i in 0..n {
                slice[i * d + j] = col[i];
            }
        }
    }
}

thread_local! {
    /// Per-thread reusable DCT scratch + column-copy buffer. Avoids
    /// allocating ~16 KB twice per column on the hot path; over
    /// 144 calls × `d` columns × 3 stages this would be ~140 MB of
    /// allocator churn per forward at Qwen3-4B.
    static COL_SCRATCH: std::cell::RefCell<ColScratch> = std::cell::RefCell::new(ColScratch::default());
}

#[derive(Default)]
struct ColScratch {
    col: Vec<f32>,
    scratch: Vec<f32>,
}

/// Tile width for the fused DCT-IV cascade. Picked so:
///  - 16 f32 = 64 B = one cache line: a tile row of the `(n, d)` input
///    is read/written as exactly one cache line per slice row (no
///    stride-`d` cache waste on copy-in / copy-out).
///  - The full tile working set `tile_d × n × 4 B` fits in L2 at the
///    production shape (16 × 4096 × 4 B = 256 KiB; Strix Halo L2 = 1 MiB
///    per core).
const DCT4_CASCADE_TILE: usize = 16;

thread_local! {
    /// Per-thread reusable scratch for the tile-fused cascade. Layout
    /// is `(tile_d, n)` row-major: each tile row is a contiguous
    /// length-`n` vector ready for DCT-IV. Reused across calls so the
    /// 256 KiB allocation amortises across the whole forward pass.
    static TILE_SCRATCH: std::cell::RefCell<TileScratch> =
        std::cell::RefCell::new(TileScratch::default());
}

#[derive(Default)]
struct TileScratch {
    /// `(tile_d, n)` row-major: `tile[j * n + i] = buf[i * d + tile_start + j]`.
    tile: Vec<f32>,
    /// Internal DCT scratch from `rustdct::Dct4::get_scratch_len`.
    dct_scratch: Vec<f32>,
}

/// Apply the DCT-IV cascade `A = D₃·C·D₂·C·D₁·C` in place, fused over
/// `DCT4_CASCADE_TILE`-column tiles. Equivalent to:
///   dct → ×D₁ → dct → ×D₂ → dct → ×D₃·inv_norm
/// run separately, but all six stages happen on each tile while it's
/// resident in L2 — slashing per-stage RAM round-trips from 6× the
/// buffer to one copy-in + one copy-out per tile (≈ 4× less RAM
/// traffic at the production shape).
///
/// Disjoint column tiles let us write through raw pointers under
/// rayon without data races.
fn dct4_cascade_apply_inplace_slice(
    slice: &mut [f32],
    n: usize,
    d: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(slice.len(), n.saturating_mul(d));
    debug_assert_eq!(d1.len(), n);
    debug_assert_eq!(d2.len(), n);
    debug_assert_eq!(d3.len(), n);
    if n < 2 || d == 0 {
        return;
    }
    let tile = DCT4_CASCADE_TILE;
    let n_tiles = d.div_ceil(tile);
    let slice_addr = slice.as_mut_ptr() as usize;
    let process = |t_idx: usize| {
        let tile_start = t_idx * tile;
        let tile_d = (tile_start + tile).min(d) - tile_start;
        TILE_SCRATCH.with(|cell| {
            let mut state = cell.borrow_mut();
            let dct_scratch_len = dct4.get_scratch_len();
            if state.tile.len() < tile * n {
                state.tile.resize(tile * n, 0.0);
            }
            if state.dct_scratch.len() < dct_scratch_len {
                state.dct_scratch.resize(dct_scratch_len, 0.0);
            }
            let TileScratch { tile: tile_buf, dct_scratch } = &mut *state;
            let tile_buf = &mut tile_buf[..tile_d * n];
            let dct_scratch = &mut dct_scratch[..dct_scratch_len];

            // SAFETY: tiles touch disjoint column ranges of `slice`.
            // Each iteration of `i` reads `tile_d` contiguous f32s
            // (one cache line at tile_d=16) from row `i` of slice.
            let slice_ptr = slice_addr as *mut f32;
            unsafe {
                copy_tile_in(slice_ptr, n, d, tile_start, tile_d, tile_buf);
            }

            // Cascade — 3 × (DCT + diag), all in-tile.
            cascade_apply_in_tile(
                tile_buf, tile_d, n, d1, d2, d3, inv_norm, dct4, dct_scratch,
            );

            unsafe {
                copy_tile_out(tile_buf, tile_d, n, slice_ptr, d, tile_start);
            }
        });
    };

    if d >= DCT4_RAYON_COL_THRESHOLD {
        (0..n_tiles).into_par_iter().for_each(process);
    } else {
        (0..n_tiles).for_each(process);
    }
}

/// Inverse cascade: `Aᵀ = C·D₁·C·D₂·C·D₃·inv_norm`. Same tile-fused
/// structure as `dct4_cascade_apply_inplace_slice`.
fn dct4_cascade_unapply_inplace_slice(
    slice: &mut [f32],
    n: usize,
    d: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(slice.len(), n.saturating_mul(d));
    debug_assert_eq!(d1.len(), n);
    debug_assert_eq!(d2.len(), n);
    debug_assert_eq!(d3.len(), n);
    if n < 2 || d == 0 {
        return;
    }
    let tile = DCT4_CASCADE_TILE;
    let n_tiles = d.div_ceil(tile);
    let slice_addr = slice.as_mut_ptr() as usize;
    let process = |t_idx: usize| {
        let tile_start = t_idx * tile;
        let tile_d = (tile_start + tile).min(d) - tile_start;
        TILE_SCRATCH.with(|cell| {
            let mut state = cell.borrow_mut();
            let dct_scratch_len = dct4.get_scratch_len();
            if state.tile.len() < tile * n {
                state.tile.resize(tile * n, 0.0);
            }
            if state.dct_scratch.len() < dct_scratch_len {
                state.dct_scratch.resize(dct_scratch_len, 0.0);
            }
            let TileScratch { tile: tile_buf, dct_scratch } = &mut *state;
            let tile_buf = &mut tile_buf[..tile_d * n];
            let dct_scratch = &mut dct_scratch[..dct_scratch_len];

            let slice_ptr = slice_addr as *mut f32;
            unsafe {
                copy_tile_in(slice_ptr, n, d, tile_start, tile_d, tile_buf);
            }

            cascade_unapply_in_tile(
                tile_buf, tile_d, n, d1, d2, d3, inv_norm, dct4, dct_scratch,
            );

            unsafe {
                copy_tile_out(tile_buf, tile_d, n, slice_ptr, d, tile_start);
            }
        });
    };

    if d >= DCT4_RAYON_COL_THRESHOLD {
        (0..n_tiles).into_par_iter().for_each(process);
    } else {
        (0..n_tiles).for_each(process);
    }
}

/// SAFETY: caller must ensure
///   - `slice_ptr` points to at least `n * d` f32s,
///   - `tile_start + tile_d ≤ d`,
///   - `tile_buf.len() ≥ tile_d * n`,
///   - this thread is the only one writing to `slice[*, tile_start..tile_start+tile_d]`.
#[inline]
unsafe fn copy_tile_in(
    slice_ptr: *const f32,
    n: usize,
    d: usize,
    tile_start: usize,
    tile_d: usize,
    tile_buf: &mut [f32],
) {
    for i in 0..n {
        // SAFETY: `i * d + tile_start + tile_d ≤ n * d`.
        let src = unsafe { slice_ptr.add(i * d + tile_start) };
        for j in 0..tile_d {
            // SAFETY: `j < tile_d`, src slice valid for `tile_d` reads.
            tile_buf[j * n + i] = unsafe { *src.add(j) };
        }
    }
}

/// SAFETY: same preconditions as `copy_tile_in`; this thread is the only
/// one writing to `slice[*, tile_start..tile_start+tile_d]`.
#[inline]
unsafe fn copy_tile_out(
    tile_buf: &[f32],
    tile_d: usize,
    n: usize,
    slice_ptr: *mut f32,
    d: usize,
    tile_start: usize,
) {
    for i in 0..n {
        // SAFETY: `i * d + tile_start + tile_d ≤ n * d`.
        let dst = unsafe { slice_ptr.add(i * d + tile_start) };
        for j in 0..tile_d {
            // SAFETY: `j < tile_d`, dst valid for `tile_d` writes.
            unsafe { *dst.add(j) = tile_buf[j * n + i] };
        }
    }
}

/// Apply cascade: 3× DCT + 3× diag on the tile's `(tile_d, n)` row-major
/// buffer.
#[inline]
fn cascade_apply_in_tile(
    tile_buf: &mut [f32],
    tile_d: usize,
    n: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
    dct_scratch: &mut [f32],
) {
    // C
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    // ×D₁ (±1 sign flip)
    apply_sign_diag_in_tile(tile_buf, tile_d, n, d1);
    // C
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    // ×D₂
    apply_sign_diag_in_tile(tile_buf, tile_d, n, d2);
    // C
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    // ×D₃ · inv_norm
    apply_scaled_diag_in_tile(tile_buf, tile_d, n, d3, inv_norm);
}

/// Inverse cascade: `inv_norm·D₃ → C → D₂ → C → D₁ → C`.
#[inline]
fn cascade_unapply_in_tile(
    tile_buf: &mut [f32],
    tile_d: usize,
    n: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
    dct_scratch: &mut [f32],
) {
    apply_scaled_diag_in_tile(tile_buf, tile_d, n, d3, inv_norm);
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    apply_sign_diag_in_tile(tile_buf, tile_d, n, d2);
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    apply_sign_diag_in_tile(tile_buf, tile_d, n, d1);
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
}

/// In-tile diag with ±1 entries: per `(tile_d, n)` row, flip the sign
/// of `tile[j, i]` when `diag[i] < 0`. No multiply when `diag[i] = +1`.
/// Mirrors `apply_diag_inplace_slice`'s sign-skip optimisation but in
/// the transposed layout (inner loop iterates `i`, contiguous in row j).
#[inline]
fn apply_sign_diag_in_tile(tile_buf: &mut [f32], tile_d: usize, n: usize, diag: &[f32]) {
    for j in 0..tile_d {
        let row = &mut tile_buf[j * n..j * n + n];
        for i in 0..n {
            if diag[i] < 0.0 {
                row[i] = -row[i];
            }
        }
    }
}

/// In-tile scaled diag: `tile[j, i] *= diag[i] * factor`. Always
/// multiplies (factor may be non-trivial — `inv_norm = (2/n)^{3/2}`).
#[inline]
fn apply_scaled_diag_in_tile(
    tile_buf: &mut [f32],
    tile_d: usize,
    n: usize,
    diag: &[f32],
    factor: f32,
) {
    for j in 0..tile_d {
        let row = &mut tile_buf[j * n..j * n + n];
        for i in 0..n {
            row[i] *= diag[i] * factor;
        }
    }
}

/// In-place row-wise sign flip: `m[i, *] *= d[i]`. Identical contract
/// to [`crate::hd3::apply_diag_inplace`]; copied here to avoid a
/// cross-module re-export. Superseded by `apply_sign_diag_in_tile`.
#[allow(dead_code)]
fn apply_diag_inplace(m: &mut Array2<f32>, d: &[f32]) {
    let cols = m.ncols();
    let slice = m
        .as_slice_mut()
        .expect("apply_diag_inplace: matrix must be standard layout");
    apply_diag_inplace_slice(slice, d, cols);
}

#[allow(dead_code)]
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
    if total_work >= crate::hd3::FWHT_RAYON_WORK_THRESHOLD {
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

/// Fused diagonal + scalar pass — mirror of `crate::hd3::apply_diag_scaled_inplace`.
/// Multiplies row `i` by `d[i] * factor` in one full-tensor pass,
/// replacing the `apply_diag_inplace + scale_inplace` pair at the
/// D₃ boundary of `apply` / `unapply`. Superseded by
/// `apply_scaled_diag_in_tile`.
#[allow(dead_code)]
fn apply_diag_scaled_inplace(m: &mut Array2<f32>, d: &[f32], factor: f32) {
    let cols = m.ncols();
    let slice = m
        .as_slice_mut()
        .expect("apply_diag_scaled_inplace: matrix must be standard layout");
    apply_diag_scaled_inplace_slice(slice, d, cols, factor);
}

#[allow(dead_code)]
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
    if total_work >= crate::hd3::FWHT_RAYON_WORK_THRESHOLD {
        slice
            .par_chunks_mut(cols)
            .zip(d.par_iter())
            .for_each(|(row, &sign)| {
                let mult = sign * factor;
                for v in row.iter_mut() {
                    *v *= mult;
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

    /// `unapply(apply(H) · W) ≈ H · W` to f32 noise at non-pow2 sizes
    /// (the whole point of DCT-IV) and pow2 sizes (sanity).
    #[test]
    fn dct4_round_trip_preserves_matmul() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        for &(n, d, p) in &[
            (8usize, 8, 8),
            (12, 8, 8),       // non-pow2
            (17, 13, 19),     // non-pow2 prime-ish
            (64, 64, 32),
            (257, 128, 64),   // non-pow2 prime — Bluestein-DCT path
            (256, 128, 128),
            (2056, 2048, 1024), // Qwen3-4B QKV-like shape, non-pow2
        ] {
            let mask = Dct4Mask::fresh(n, &mut rng);
            let h = sample_normal(&mut rng, n, d);
            let w = sample_normal(&mut rng, d, p);
            let target = h.dot(&w);

            let u = mask.apply(h.view());
            let v = u.dot(&w);
            let recovered = mask.unapply(v.view());

            assert_eq!(recovered.dim(), target.dim());
            // Tolerance: matmul depth d plus DCT-IV accumulation depth
            // (rustdct's algorithm depth is O(log n) for pow2, O(log²n)
            // for Bluestein non-pow2) times three cascade stages. Use a
            // conservative bound similar to HD₃.
            let depth = d as f32 + 3.0 * (n as f32).log2().max(1.0) * 4.0;
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

    /// `Aᵀ · A == I` to f32 noise. DCT-IV cascade is orthogonal by
    /// construction.
    #[test]
    fn dct4_orthogonality() {
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        for &n in &[8usize, 12, 17, 32, 64, 257] {
            let mask = Dct4Mask::fresh(n, &mut rng);
            let id = Array2::<f32>::eye(n);
            let a = mask.apply(id.view());
            let ata = a.t().dot(&a);
            let id_target = Array2::<f32>::eye(n);
            let depth = n as f32 + 3.0 * (n as f32).log2().max(1.0) * 4.0;
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
    fn dct4_deterministic_from_seed() {
        let seed_a = MaskSeed::from_bytes([42u8; 32]);
        let seed_b = MaskSeed::from_bytes([43u8; 32]);
        let m_a1 = Dct4Mask::from_seed(64, seed_a);
        let m_a2 = Dct4Mask::from_seed(64, seed_a);
        let m_b = Dct4Mask::from_seed(64, seed_b);
        assert_eq!(m_a1.d1, m_a2.d1);
        assert_eq!(m_a1.d2, m_a2.d2);
        assert_eq!(m_a1.d3, m_a2.d3);
        assert_ne!(m_a1.d1, m_b.d1);
    }

    /// `Dct4Mask::fresh(0)` rejects zero.
    #[test]
    #[should_panic(expected = "n must be positive")]
    fn dct4_rejects_zero() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let _ = Dct4Mask::fresh(0, &mut rng);
    }

    /// Round-trip relative RMS at the realistic Qwen3-4B
    /// non-pow2 long-context shape (n=2056, d=2560).
    #[test]
    fn dct4_round_trip_relative_error_at_long_n() {
        let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
        let n = 2056;
        let d = 2560;
        let p = 1024;
        let mask = Dct4Mask::fresh(n, &mut rng);
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
            err_rms / target_rms < 1e-3,
            "round-trip relative rms at long-n (n={n}): {:.3e}",
            err_rms / target_rms
        );
    }
}
