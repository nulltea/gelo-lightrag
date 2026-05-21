use ndarray::{Array2, ArrayView2, Axis, s};
use rand::RngCore;
use rand_distr::{Distribution, StandardNormal};

pub use crate::rng::MaskSeed;

use crate::dct4::Dct4Mask;
use crate::hd3::Hd3Mask;

/// Mask family used by [`crate::sim::InProcessTrustedExecutor`].
/// `Haar` is the dense Householder-QR-sampled orthogonal mask
/// described in the GELO paper §3.2 (per-forward `O(s³)` sample +
/// `O(s²·d)` per apply/unapply). `Hd3` is the QuIP#/QuaRot-style
/// structured-orthogonal cascade `D₃·H·D₂·H·D₁·H` with `O(s)` sample
/// and `O(s·d·log s)` per apply/unapply at power-of-two `s` — see
/// [`crate::hd3`] for the math + security trade. `Dct4` is the
/// arbitrary-order analog `D₃·C·D₂·C·D₁·C` with DCT-IV as the inner
/// orthogonal — works at any `n` so the caller skips the pow2 pad —
/// see [`crate::dct4`] for the math + security argument.
#[derive(Debug, Clone)]
pub enum MaskFamily {
    Haar(GeloMask),
    Hd3(Hd3Mask),
    Dct4(Dct4Mask),
}

impl MaskFamily {
    /// Side length the mask operates on. For `Haar` this matches the
    /// stacked-with-shield row count `n + k`; for `Hd3` this is
    /// `(n + k).next_power_of_two()` (the caller must arrange the
    /// pow2 padding before constructing the mask); for `Dct4` it is
    /// exactly `n + k` (no padding needed).
    pub fn n(&self) -> usize {
        match self {
            Self::Haar(m) => m.n(),
            Self::Hd3(m) => m.n(),
            Self::Dct4(m) => m.n(),
        }
    }

    pub fn apply(&self, hidden: ArrayView2<'_, f32>) -> Array2<f32> {
        match self {
            Self::Haar(m) => m.apply(hidden),
            Self::Hd3(m) => m.apply(hidden),
            Self::Dct4(m) => m.apply(hidden),
        }
    }

    pub fn unapply(&self, masked: ArrayView2<'_, f32>) -> Array2<f32> {
        match self {
            Self::Haar(m) => m.unapply(masked),
            Self::Hd3(m) => m.unapply(masked),
            Self::Dct4(m) => m.unapply(masked),
        }
    }

    /// Consume `masked` and return the unmasked output. For [`Self::Hd3`]
    /// and [`Self::Dct4`] runs `Aᵀ` in place on the input buffer —
    /// saving the `(s, p)` allocation + copy that the allocating
    /// [`Self::unapply`] pays per call. For [`Self::Haar`] falls back
    /// to the allocating path because the dense `Aᵀ · M` GEMM needs a
    /// separate output workspace anyway.
    pub fn unapply_take(&self, masked: Array2<f32>) -> Array2<f32> {
        match self {
            Self::Haar(m) => m.unapply(masked.view()),
            Self::Hd3(m) => {
                let mut buf = masked;
                m.unapply_in_place(&mut buf);
                buf
            }
            Self::Dct4(m) => {
                let mut buf = masked;
                m.unapply_in_place(&mut buf);
                buf
            }
        }
    }

    /// `&'static str` profile category for `gelo:mask_apply` that
    /// splits by mask family. Used by `InProcessTrustedExecutor` to
    /// make the per-stage profile dump distinguish Haar / HD₃ /
    /// DCT-IV cost — otherwise Auto's runtime choice is invisible
    /// in the breakdown.
    pub fn apply_profile_category(&self) -> &'static str {
        match self {
            Self::Haar(_) => "gelo:mask_apply:haar",
            Self::Hd3(_) => "gelo:mask_apply:hd3",
            Self::Dct4(_) => "gelo:mask_apply:dct4",
        }
    }

    /// Same as [`Self::apply_profile_category`] but for the unapply
    /// path.
    pub fn unapply_profile_category(&self) -> &'static str {
        match self {
            Self::Haar(_) => "gelo:mask_unapply:haar",
            Self::Hd3(_) => "gelo:mask_unapply:hd3",
            Self::Dct4(_) => "gelo:mask_unapply:dct4",
        }
    }
}

/// Which mask family `InProcessTrustedExecutor` should use. Default
/// is [`MaskKind::Haar`] (paper-parity). Switch to [`MaskKind::Hd3`]
/// via [`crate::sim::InProcessTrustedExecutor::with_hd3_mask`], to
/// [`MaskKind::Dct4`] via `with_dct4_mask` for arbitrary-order
/// (non-pow2) shapes, or to [`MaskKind::Auto`] via `with_auto_mask`
/// for shape-adaptive dispatch (HD₃ when pad penalty is small,
/// DCT-IV otherwise). See `docs/research/hd3-non-pow2-fix.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaskKind {
    #[default]
    Haar,
    Hd3,
    Dct4,
    /// Per-call dispatch between [`Self::Hd3`] and [`Self::Dct4`]
    /// based on the pad penalty `s_pad / s` where `s = n + k_shield`
    /// and `s_pad = s.next_power_of_two()`. HD₃ is picked when the
    /// pad ratio is ≤ [`HD3_AUTO_MAX_PAD_RATIO_NUM`] /
    /// [`HD3_AUTO_MAX_PAD_RATIO_DEN`] (default 4/3 ≈ 33 % pad max,
    /// empirically derived for Qwen3-4B on Strix Halo iGPU); DCT-IV
    /// otherwise. The choice is per-forward-pass (recorded in the
    /// session mask) or per-offload depending on the executor's
    /// `per_forward_mask` setting.
    Auto,
}

/// Numerator of the HD₃-vs-DCT-IV pad-ratio threshold used by
/// [`MaskKind::Auto`]. HD₃ is selected when
/// `s_pad * HD3_AUTO_MAX_PAD_RATIO_DEN <= s * HD3_AUTO_MAX_PAD_RATIO_NUM`,
/// i.e., when the pad fraction `(s_pad − s) / s_pad ≤ 1/4`.
///
/// Empirical crossover from the 2026-05-20 Qwen3-4B bench is ~1.39;
/// 4/3 ≈ 1.333 is a conservative pick that ensures HD₃ is selected
/// only when clearly faster, with a small "either is fine" zone
/// between 1.333 and 1.4.
// 2026-05-21: relaxed from 4/3 (1.333) to 7/5 (1.4). The empirical
// crossover documented in `qwen3_4b_perf_2026_05_20.md` is ~1.39; at
// 4/3 Auto rejected HD₃ at the prefill ratio 1.36 observed on Qwen3-4B
// (s=753, s_pad=1024) even though HD₃ was the faster choice. 7/5 puts
// the empirical crossover inside the "HD₃ wins" band with ~1 % margin.
pub const HD3_AUTO_MAX_PAD_RATIO_NUM: usize = 7;
pub const HD3_AUTO_MAX_PAD_RATIO_DEN: usize = 5;

/// Resolve a configured [`MaskKind`] (possibly [`MaskKind::Auto`]) to
/// a concrete physical kind given the stacked size `s = n + k_shield`.
/// For non-Auto kinds this is the identity; for Auto it applies the
/// pad-ratio rule documented above.
pub fn resolve_mask_kind_for_shape(kind: MaskKind, s: usize) -> MaskKind {
    match kind {
        MaskKind::Auto => {
            let s_pad = s.next_power_of_two().max(2);
            let picked = if s_pad.saturating_mul(HD3_AUTO_MAX_PAD_RATIO_DEN)
                <= s.saturating_mul(HD3_AUTO_MAX_PAD_RATIO_NUM)
            {
                MaskKind::Hd3
            } else {
                MaskKind::Dct4
            };
            // Diagnostic — surfaces what Auto picked for every
            // begin_forward_pass(n). Enable with
            // `RUST_LOG=gelo_protocol=debug` (or trace). Cheap: one
            // small kv-record per forward, never per offload.
            tracing::debug!(
                target: "gelo_protocol::mask",
                s,
                s_pad,
                pad_ratio_x1000 = (s_pad * 1000 / s.max(1)) as u64,
                threshold_x1000 = (HD3_AUTO_MAX_PAD_RATIO_NUM * 1000
                    / HD3_AUTO_MAX_PAD_RATIO_DEN) as u64,
                picked = ?picked,
                "auto-mask resolved"
            );
            picked
        }
        kind => kind,
    }
}

/// Token-axis orthogonal mask used to obfuscate hidden states before the
/// untrusted side sees them.
///
/// For hidden states `H ∈ R^(n×d)` the trusted side computes `U = A·H` and
/// ships `U` to the engine. Because `A` is orthogonal, `A⁻¹ = Aᵀ` and
/// recovery is `H·W = Aᵀ·(U·W)`. `A` is sampled fresh per batch from the
/// Haar measure on `O(n)` using the Mezzadri 2007 QR-with-sign-correction
/// trick over a standard normal seed.
#[derive(Debug, Clone)]
pub struct GeloMask {
    /// `(n, n)` orthogonal mask matrix.
    a: Array2<f32>,
}

impl GeloMask {
    /// Sample a fresh Haar-uniform orthogonal `A ∈ R^(n×n)`.
    pub fn fresh<R: RngCore>(n: usize, rng: &mut R) -> Self {
        Self {
            a: sample_haar_orthogonal(n, rng),
        }
    }

    /// Deterministic constructor for tests.
    pub fn from_seed(n: usize, seed: MaskSeed) -> Self {
        let mut rng = seed.rng();
        Self::fresh(n, &mut rng)
    }

    /// `n`, the token-axis dimension this mask operates on.
    pub fn n(&self) -> usize {
        self.a.nrows()
    }

    /// Reference to the underlying `(n, n)` orthogonal matrix.
    pub fn matrix(&self) -> ArrayView2<'_, f32> {
        self.a.view()
    }

    /// Apply the mask: `U = A · H`.
    ///
    /// `hidden` must have shape `(n, d)` with `n == self.n()`.
    pub fn apply(&self, hidden: ArrayView2<'_, f32>) -> Array2<f32> {
        assert_eq!(
            hidden.nrows(),
            self.n(),
            "hidden row count must equal mask n"
        );
        #[cfg(feature = "blas")]
        {
            return sgemm_blis(self.a.view(), hidden, false);
        }
        #[cfg(not(feature = "blas"))]
        self.a.dot(&hidden)
    }

    /// Remove the mask: `H·W = Aᵀ · (U·W)`.
    ///
    /// `masked_output` must have shape `(n, p)` with `n == self.n()`.
    pub fn unapply(&self, masked_output: ArrayView2<f32>) -> Array2<f32> {
        assert_eq!(
            masked_output.nrows(),
            self.n(),
            "masked output row count must equal mask n"
        );
        #[cfg(feature = "blas")]
        {
            return sgemm_blis(self.a.view(), masked_output, true);
        }
        #[cfg(not(feature = "blas"))]
        self.a.t().dot(&masked_output)
    }

}

#[cfg(feature = "blas")]
unsafe extern "C" {
    /// BLIS thread-count setter. The default (BLIS_NUM_THREADS=N if set,
    /// else all cores) over-subscribes here: each mask GEMM is small
    /// enough that BLIS's per-call thread barrier costs more than the
    /// parallel work saves. We pin to 1 at the first sgemm_blis call.
    fn bli_thread_set_num_threads(n_threads: i64);
}

/// Lazily pin BLIS to single-thread on first call. Process-global state
/// (BLIS owns its own thread pool), but a OnceLock guard means we set it
/// exactly once even under concurrent first-call races.
/// Per-thread idempotent pin of BLIS to single-thread. AOCL-BLIS keeps
/// the active thread count in **thread-local** state (despite the
/// `bli_thread_set_num_threads` name) — calls from the main thread
/// don't propagate to rayon workers spawned later, so the right place
/// to pin is *inside* `sgemm_blis` (which every worker thread enters
/// on its own). The thread-local OnceLock means we pay the C-call cost
/// exactly once per thread that ever runs a mask GEMM.
#[cfg(feature = "blas")]
fn blis_init_single_thread() {
    use std::cell::Cell;
    thread_local! {
        static PINNED: Cell<bool> = const { Cell::new(false) };
    }
    PINNED.with(|p| {
        if !p.get() {
            // `GELO_BLIS_THREADS=N` overrides the single-thread auto-pin.
            // At long-n shapes (e.g. n=2048 prefill) each mask GEMM is
            // multi-TFLOP, so multi-thread BLIS amortises its per-call
            // thread-barrier overhead. At small shapes the default
            // (1 thread) is still right.
            let n_threads: i64 = std::env::var("GELO_BLIS_THREADS")
                .ok()
                .and_then(|s| s.parse::<i64>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(1);
            set_blis_num_threads(n_threads);
            p.set(true);
        }
    });
}

/// Eagerly pin BLIS to single-thread on the calling thread. Idempotent:
/// subsequent calls on the same thread are a thread-local atomic read.
/// Note that AOCL-BLIS holds the active thread count per-thread, so
/// every rayon worker needs its own first-call init — which happens
/// automatically inside `mask::apply` / `mask::unapply`. Calling this
/// from your `main()` is harmless but only affects the main thread.
///
/// No-op when the `blas` feature is disabled.
pub fn ensure_blis_single_thread() {
    #[cfg(feature = "blas")]
    blis_init_single_thread();
}

/// Human-readable description of the mask GEMM backend that will be
/// used at runtime. Useful for bench preambles and bug reports — at
/// long-n shapes the BLIS-vs-matrixmultiply difference is 5× and silent
/// fallback to matrixmultiply was the root cause of an earlier mis-run
/// where `GELO_BLIS_THREADS` looked like it had no effect.
pub fn mask_backend_description() -> String {
    #[cfg(feature = "blas")]
    {
        let n_threads: i64 = std::env::var("GELO_BLIS_THREADS")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1);
        format!("AOCL-BLIS (cblas_sgemm), threads={n_threads}")
    }
    #[cfg(not(feature = "blas"))]
    {
        "matrixmultiply (single-thread, pure-Rust; build with default `blas` feature for ~5× faster long-n)".to_string()
    }
}

/// Override BLIS's thread count for the mask SGEMMs. The default
/// (auto-pinned to 1 on first `mask::apply` / `mask::unapply` call) is
/// the right choice for the embedder + rerank paths because each mask
/// GEMM is small enough that BLIS's per-call thread barrier dominates;
/// multi-thread BLIS over-subscribes when rayon already owns the outer
/// loop. If you have a different workload — e.g. a single very large
/// SGEMM benchmark — call this with `n > 1` from `main()` BEFORE the
/// first mask GEMM and the auto-init becomes a no-op (OnceLock fires
/// once).
///
/// No-op when the `blas` feature is disabled (matrixmultiply has no
/// global thread setting; rayon owns its scheduling).
#[cfg(feature = "blas")]
pub fn set_blis_num_threads(n: i64) {
    // SAFETY: `bli_thread_set_num_threads` is thread-safe per BLIS docs.
    unsafe { bli_thread_set_num_threads(n) };
}

/// No-op stub for the non-`blas` build so callers can call this
/// unconditionally without `#[cfg]` gates of their own.
#[cfg(not(feature = "blas"))]
pub fn set_blis_num_threads(_n: i64) {}

/// Shape threshold (rows of `a`) above which `tee_matmul` switches
/// from ndarray's `.dot()` (matrixmultiply single-thread) to the
/// BLIS-backed `matmul_blis`. Matches the parallelisation threshold
/// used by `causal_gqa_attention_cached` so both shape-aware code
/// paths agree on what counts as "long enough to multi-thread."
const TEE_BLIS_THRESHOLD_ROWS: usize = 64;

/// General `C = A · B` matmul with shape-aware dispatch.
///
/// At `a.nrows() >= TEE_BLIS_THRESHOLD_ROWS` (prefill regime) we route
/// through BLIS so the call benefits from `GELO_BLIS_THREADS`. At
/// smaller shapes (decode `n_q = 1`, embedder shapes) we fall back
/// to `ndarray::dot()` which uses matrixmultiply single-thread —
/// avoiding the BLIS per-call thread-barrier overhead that regresses
/// small matmuls.
///
/// Also falls back to `.dot()` if either operand is non-standard
/// layout (cblas requires row-major contiguous; ndarray's `.dot()`
/// tolerates arbitrary strides). The decoder weight tensors are
/// loaded standard-layout in `decoder::weights::read2_t`, so this
/// fallback is just defensive against future callers.
///
/// Use this from the `tee:*_direct` paths in `decoder/forward.rs` so
/// the in-TEE plain matmul also benefits from BLIS-mt at long-n
/// shapes, just like `mask::apply` / `mask::unapply` already do.
///
/// **Known gap (2026-05-19):** at decode shape `m=1`, the fallback
/// `.dot()` path is ~10× slower than the same matmul on the GPU
/// engine: matrixmultiply hits only ~1 GFLOP/s at the GEMV corner
/// case (vs ~125 GFLOP/s nominal). BLIS at `m=1` isn't a clean
/// alternative either — its per-call thread overhead at multi-thread
/// settings dominates the actual GEMM. So Qwen3-decoder layer-skip
/// (paper §3.2 sensitive-layer exclusion) currently regresses
/// decode TPOT by ~234 ms/step at n=2048, even though it saves ~1 s
/// of prefill. Fixing this needs either a hand-rolled AVX-512 GEMV
/// for `m=1` or a per-call BLIS thread-count override. Documented
/// as a future optimisation surface in
/// `memory/tee_direct_m1_gemv_slowness.md` and the round-3 perf doc.
pub fn tee_matmul(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> Array2<f32> {
    #[cfg(feature = "blas")]
    {
        if a.nrows() >= TEE_BLIS_THRESHOLD_ROWS
            && a.is_standard_layout()
            && b.is_standard_layout()
        {
            return matmul_blis(a, b);
        }
    }
    a.dot(&b)
}

/// bf16-weight variant of [`tee_matmul`] for the sensitive-layer
/// carve-out (paper §3.2 skip_first / skip_last). Activations stay
/// f32; the weight matrix is bf16. We widen per-element into a
/// transient `Array2<f32>` then forward to `tee_matmul`.
///
/// **Only called when offload=false for a given layer.** With the
/// project defaults (skip-first / skip-last both off), this path is
/// unreachable at runtime. The per-element widening is acceptable
/// because skip-layer mode is an explicit operator opt-in, and the
/// resulting alloc lifetime is bounded by the matmul call.
pub fn tee_matmul_bf16(a: ArrayView2<'_, f32>, b: ArrayView2<'_, half::bf16>) -> Array2<f32> {
    let b_f32 = b.mapv(|v| v.to_f32());
    tee_matmul(a, b_f32.view())
}

/// BLIS-backed general matmul `C = A · B`. Both operands assumed
/// row-major contiguous (true for `ndarray::Array2` standard layout
/// and any `.view()` of an unsliced array). The function mirrors
/// `sgemm_blis` but takes general (non-square) inputs.
///
/// Public so the `tee_matmul` dispatch helper can pick it up.
/// Callers should prefer `tee_matmul` which handles the shape
/// threshold; call this directly only when you know the shape
/// warrants BLIS.
#[cfg(feature = "blas")]
pub fn matmul_blis(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> Array2<f32> {
    blis_init_single_thread();
    use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    debug_assert_eq!(
        b.nrows(),
        k,
        "matmul_blis: a.ncols() = {} must equal b.nrows() = {}",
        k,
        b.nrows()
    );
    let a_slice = a.to_slice().expect("matmul_blis: a must be row-major contiguous");
    let b_slice = b.to_slice().expect("matmul_blis: b must be row-major contiguous");
    let mut c = Array2::<f32>::zeros((m, n));
    {
        let c_slice = c.as_slice_mut().expect("fresh Array2 is contiguous");
        // SAFETY: cblas_sgemm reads (m × k) from `a_slice`, (k × n) from
        // `b_slice`, and writes (m × n) into `c_slice`; all three are
        // row-major slice views with the right lengths.
        unsafe {
            cblas_sgemm(
                CBLAS_LAYOUT::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_slice.as_ptr(),
                k as i32,
                b_slice.as_ptr(),
                n as i32,
                0.0,
                c_slice.as_mut_ptr(),
                n as i32,
            );
        }
    }
    c
}

/// BLIS-backed `C = α · op(A) · B` for the GELO mask apply/unapply.
///
/// Called only with `α = 1.0, β = 0.0`. `a` is the (n × n) mask matrix
/// (always row-major); `b` is a (n × d) row-major view of the
/// stacked/masked operand. If `transpose_a` is `true`, computes
/// `C = Aᵀ · B` (the unapply path); otherwise `C = A · B` (apply).
///
/// We bypass `ndarray::dot` (which would dispatch to matrixmultiply or
/// to ndarray's `blas` feature globally) and call `cblas_sgemm`
/// directly so only the mask hot path picks up BLIS — the small per-
/// head attention matmuls keep matrixmultiply.
#[cfg(feature = "blas")]
fn sgemm_blis(
    a: ArrayView2<'_, f32>,
    b: ArrayView2<'_, f32>,
    transpose_a: bool,
) -> Array2<f32> {
    blis_init_single_thread();
    use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n, "mask must be square");
    debug_assert_eq!(b.nrows(), n);
    let d = b.ncols();
    // Both operands are row-major standard-layout from the call sites
    // (`a` is the mask; `b` is the stacked scratch or an engine output).
    let a_slice = a
        .to_slice()
        .expect("mask must be row-major contiguous");
    let b_slice = b
        .to_slice()
        .expect("operand must be row-major contiguous");
    let mut c = Array2::<f32>::zeros((n, d));
    {
        let c_slice = c.as_slice_mut().expect("fresh Array2 is contiguous");
        let transa = if transpose_a {
            CBLAS_TRANSPOSE::CblasTrans
        } else {
            CBLAS_TRANSPOSE::CblasNoTrans
        };
        // SAFETY: cblas_sgemm reads (n×n) from `a_slice`, (n×d) from
        // `b_slice`, and writes (n×d) into `c_slice`. All three lengths
        // are guaranteed by the row-major slice views above.
        unsafe {
            cblas_sgemm(
                CBLAS_LAYOUT::CblasRowMajor,
                transa,
                CBLAS_TRANSPOSE::CblasNoTrans,
                n as i32,                // M = rows of op(A)
                d as i32,                // N = cols of B
                n as i32,                // K = cols of op(A) = rows of B
                1.0,                     // alpha
                a_slice.as_ptr(),
                n as i32,                // lda = cols of A (row-major)
                b_slice.as_ptr(),
                d as i32,                // ldb = cols of B
                0.0,                     // beta
                c_slice.as_mut_ptr(),
                d as i32,                // ldc = cols of C
            );
        }
    }
    c
}

/// Haar-uniform orthogonal sampler via Householder QR with Mezzadri-2007
/// sign correction. O(n³) work, O(n²) memory, no LAPACK dep.
///
/// **Algorithm** — identical to the textbook Householder QR; the speedup
/// over the scalar reference comes from:
///   - Outer reduction (`σ² = Σ vᵢ²`) computed via slice sum, LLVM SIMDs it.
///   - The rank-1 sub-matrix update `A[k:, k:] -= 2 v (vᵀ A[k:, k:])`
///     decomposes into a GEMV `vᵀ A` (via `ndarray::dot`, which dispatches
///     to `matrixmultiply` — SIMD-vectorised + cache-tiled) followed by an
///     outer-product subtraction expressed as row-wise slice operations
///     (LLVM auto-vectorises the inner loop because the row is a
///     contiguous `&mut [f32]`).
///   - The Q accumulation has its rank-1 update mirrored on columns (Q's
///     stride is row-major so we operate on `Q[:, k:]` column-wise).
///
/// **Correctness invariant** — the output `Q ∈ R^(n×n)` is Haar-distributed
/// on O(n). Mezzadri-2007 sign correction (after the QR factorisation) is
/// load-bearing: without it, `Q` is orthogonal but not Haar-uniform, which
/// would weaken GELO's information-theoretic privacy argument.
fn sample_haar_orthogonal<R: RngCore>(n: usize, rng: &mut R) -> Array2<f32> {
    let normal = StandardNormal;
    let mut a = Array2::<f32>::from_shape_fn((n, n), |_| normal.sample(rng));
    let mut q = Array2::<f32>::eye(n);
    // Reusable storage for the Householder vector v[k..] and the GEMV
    // result `v^T A[k:, k:]` of length (n-k).
    let mut v_buf = vec![0.0f32; n];
    let mut dot_buf = vec![0.0f32; n];

    for k in 0..n.saturating_sub(1) {
        let m = n - k;
        // σ² = Σ_{i≥k} a[i, k]². Column-stride read in row-major: not
        // ideal, but only n elements per step — O(n²) total work, well
        // below the O(n³) rank-1 update cost we care about.
        let mut sigma_sq = 0.0_f32;
        for i in k..n {
            let x = a[[i, k]];
            sigma_sq += x * x;
        }
        let sigma = sigma_sq.sqrt();
        if sigma < 1e-30 {
            continue;
        }

        let a_kk = a[[k, k]];
        let sign = if a_kk >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * sigma;

        let v0 = a_kk - alpha;
        let mut v_norm_sq = v0 * v0;
        for i in (k + 1)..n {
            let x = a[[i, k]];
            v_norm_sq += x * x;
        }
        let v_norm = v_norm_sq.sqrt();
        if v_norm < 1e-30 {
            continue;
        }

        v_buf[0] = v0 / v_norm;
        for (offset, dst) in v_buf[1..m].iter_mut().enumerate() {
            *dst = a[[k + 1 + offset, k]] / v_norm;
        }

        // A[k:, k:] -= 2 v vᵀ A[k:, k:]
        rank1_householder_update_rows(&mut a, k, &v_buf[..m], &mut dot_buf[..m]);

        // Q[:, k:] -= 2 (Q[:, k:] v) vᵀ. dot_buf needs length n_rows; pass
        // the full buffer (Q is the n×n accumulator so n_rows == n).
        rank1_householder_update_cols(&mut q, k, &v_buf[..m], &mut dot_buf);
    }

    // Mezzadri 2007: normalize so diag(R) ≥ 0, making the orthogonal output
    // Haar-uniform. Without this step Q is orthogonal but biased — would
    // weaken GELO's privacy guarantee. Column-wise sign flip on Q based on
    // the sign of A's diagonal.
    for i in 0..n {
        if a[[i, i]] < 0.0 {
            let mut col = q.slice_mut(s![.., i]);
            for x in col.iter_mut() {
                *x = -*x;
            }
        }
    }

    q
}

/// Rank-1 update of the bottom-right submatrix:
///   `A[k:, k:] -= 2 v vᵀ A[k:, k:]`
/// `v` has length `m = n - k`. `dot_buf[..m]` is reused scratch.
///
/// Cache-friendly via two row-major passes — never reads a column of A
/// out of stride. ndarray's `.dot()` doesn't pull BLAS unless we enable
/// the `blas` feature (we don't); the in-tree fallback has per-call
/// overhead that exceeds our SIMD-friendly hand-rolled version at the
/// shapes we hit (m ≤ 512).
fn rank1_householder_update_rows(
    a: &mut Array2<f32>,
    k: usize,
    v: &[f32],
    dot_buf: &mut [f32],
) {
    let m = v.len();
    let mut sub = a.slice_mut(s![k.., k..]);
    debug_assert_eq!(sub.shape(), &[m, m]);
    let dot = &mut dot_buf[..m];

    // Pass 1: dot[j] += v[i] * sub[i, j], iterating row-by-row so each
    // inner loop touches a contiguous &[f32] slice. LLVM auto-vectorises.
    dot.fill(0.0);
    for (row, &vi) in sub.axis_iter(Axis(0)).zip(v.iter()) {
        let row_slice = row
            .to_slice()
            .expect("row is contiguous in row-major Array2");
        for (d, &x) in dot.iter_mut().zip(row_slice.iter()) {
            *d += vi * x;
        }
    }

    // Pass 2: sub[i, j] -= 2 v[i] dot[j]. Row-by-row, contiguous writes.
    for (row_view, &vi) in sub.axis_iter_mut(Axis(0)).zip(v.iter()) {
        let coef = 2.0 * vi;
        if coef == 0.0 {
            continue;
        }
        let row_slice = row_view
            .into_slice()
            .expect("row is contiguous in row-major Array2");
        for (r, &d) in row_slice.iter_mut().zip(dot.iter()) {
            *r -= coef * d;
        }
    }
}

/// Rank-1 update of the right block of Q:
///   `Q[:, k:] -= 2 (Q[:, k:] v) vᵀ`
/// where v has length `m = n - k`. `dot_buf[..n_rows]` is scratch.
///
/// Same cache strategy: row-major sweeps over `Q[r, k..n]` slices (which
/// are contiguous in row-major). LLVM SIMD-vectorises both inner loops.
fn rank1_householder_update_cols(
    q: &mut Array2<f32>,
    k: usize,
    v: &[f32],
    dot_buf: &mut [f32],
) {
    let m = v.len();
    let n_rows = q.nrows();
    let mut sub = q.slice_mut(s![.., k..]);
    debug_assert_eq!(sub.shape(), &[n_rows, m]);
    let dot = &mut dot_buf[..n_rows];

    // Pass 1: dot[r] = Σ_c sub[r, c] * v[c]. Each row of sub is contiguous.
    for (r_idx, row) in sub.axis_iter(Axis(0)).enumerate() {
        let row_slice = row
            .to_slice()
            .expect("row is contiguous in row-major Array2");
        let mut acc = 0.0_f32;
        for (&x, &vc) in row_slice.iter().zip(v.iter()) {
            acc += x * vc;
        }
        dot[r_idx] = acc;
    }

    // Pass 2: sub[r, c] -= 2 dot[r] v[c]. Same row-major sweep.
    for (row_view, &dot_r) in sub.axis_iter_mut(Axis(0)).zip(dot.iter()) {
        let coef = 2.0 * dot_r;
        if coef == 0.0 {
            continue;
        }
        let row_slice = row_view
            .into_slice()
            .expect("row is contiguous in row-major Array2");
        for (r, &vc) in row_slice.iter_mut().zip(v.iter()) {
            *r -= coef * vc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn orthogonality() {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([7u8; 32]);
        let n = 32;
        let mask = GeloMask::fresh(n, &mut rng);
        let a = mask.matrix();
        let prod = a.t().dot(&a);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(prod[[i, j]], expected, 1e-4),
                    "AᵀA[{i},{j}] = {} expected {}",
                    prod[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn mask_round_trip_preserves_matmul() {
        // Verify that for any H, W:  Aᵀ · ((A·H) · W) ≈ H · W.
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([11u8; 32]);
        let n = 16;
        let d = 12;
        let p = 8;
        let mask = GeloMask::fresh(n, &mut rng);

        let normal = StandardNormal;
        let h = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let w = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));

        let plaintext = h.dot(&w);
        let masked = mask.apply(h.view());
        let masked_out = masked.dot(&w);
        let recovered = mask.unapply(masked_out.view());

        for ((i, j), pt) in plaintext.indexed_iter() {
            assert!(
                approx_eq(*pt, recovered[[i, j]], 1e-3),
                "mismatch at ({i},{j}): plain={pt} recovered={}",
                recovered[[i, j]]
            );
        }
    }

    #[test]
    fn deterministic_from_seed() {
        let seed = MaskSeed::from_bytes([42u8; 32]);
        let m1 = GeloMask::from_seed(8, seed);
        let m2 = GeloMask::from_seed(8, seed);
        let diff = &m1.a - &m2.a;
        let max_abs = diff.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        assert_eq!(max_abs, 0.0);
    }

    /// Haar-uniformity smoke test. The Mezzadri sign correction is what
    /// distinguishes Haar-uniform sampling from "any orthogonal matrix" —
    /// without it we'd get a biased distribution and lose GELO's privacy
    /// argument. We can't directly test "uniform on O(n)" with a finite
    /// sample, but we *can* test invariants that any Haar sample must
    /// satisfy: det(A) ∈ {±1} (exactly orthogonal, no sign bias toward
    /// +1), and the mean entry across many draws should converge to 0.
    #[test]
    fn haar_uniformity_invariants() {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([99u8; 32]);
        let n = 16;
        let n_draws = 64;
        let mut dets: Vec<f32> = Vec::with_capacity(n_draws);
        let mut mean_acc = 0.0_f32;
        let mut count = 0usize;

        for _ in 0..n_draws {
            let mask = GeloMask::fresh(n, &mut rng);
            let a = mask.matrix();

            // Orthogonality: ‖AᵀA − I‖_max < 1e-4 (covered by `orthogonality`
            // test; spot-check here too).
            let ata = a.t().dot(&a);
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (ata[[i, j]] - want).abs() < 1e-3,
                        "AᵀA[{i},{j}] = {} expected {}",
                        ata[[i, j]],
                        want
                    );
                }
            }

            // det(A) via product of eigenvalues = ±1 for orthogonal
            // matrices. Compute via Gauss elimination — simple O(n³).
            let det = determinant(a.to_owned().view());
            dets.push(det);
            assert!(
                (det.abs() - 1.0).abs() < 5e-3,
                "|det(A)| = {} not close to 1; sign correction likely broken",
                det.abs()
            );

            for &x in a.iter() {
                mean_acc += x;
                count += 1;
            }
        }

        // det signs should be roughly balanced; Mezzadri sign correction
        // makes Q span O(n) uniformly (both SO(n) and the reflection
        // coset). Without sign correction, det would always be +1 (since
        // Householder reflectors compose to ±1 and a particular impl
        // detail picks +1).
        let pos = dets.iter().filter(|&&d| d > 0.0).count();
        let neg = dets.iter().filter(|&&d| d < 0.0).count();
        assert!(
            pos > 0 && neg > 0,
            "Haar distribution should hit both det=+1 and det=-1 in {n_draws} draws; \
             got pos={pos} neg={neg} — sign correction likely broken"
        );

        // Mean of all entries across many draws should converge to ~0.
        // With n_draws=64 and n²=256 entries each, we have ~16k samples;
        // the per-entry mean is ~0 with stderr ~ 1/√(n_draws · n²) ≈ 0.008.
        let mean = mean_acc / count as f32;
        assert!(
            mean.abs() < 0.05,
            "mean of mask entries = {mean} — should be near zero for Haar-uniform"
        );
    }

    /// Naive determinant via row reduction. O(n³) — only used in tests.
    fn determinant(a: ArrayView2<f32>) -> f32 {
        let n = a.nrows();
        let mut m = a.to_owned();
        let mut det = 1.0_f32;
        for i in 0..n {
            // Find pivot
            let mut pivot = i;
            for r in (i + 1)..n {
                if m[[r, i]].abs() > m[[pivot, i]].abs() {
                    pivot = r;
                }
            }
            if m[[pivot, i]].abs() < 1e-9 {
                return 0.0;
            }
            if pivot != i {
                for j in 0..n {
                    let tmp = m[[i, j]];
                    m[[i, j]] = m[[pivot, j]];
                    m[[pivot, j]] = tmp;
                }
                det = -det;
            }
            det *= m[[i, i]];
            let pv = m[[i, i]];
            for r in (i + 1)..n {
                let factor = m[[r, i]] / pv;
                for c in i..n {
                    let s = m[[i, c]];
                    m[[r, c]] -= factor * s;
                }
            }
        }
        det
    }

    use rand::SeedableRng;

    /// `tee_matmul` must agree with `ndarray::dot` to within f32 noise
    /// at both small shapes (where it falls back to `.dot()` directly)
    /// and large shapes (where it routes through BLIS).
    #[test]
    fn tee_matmul_parity_with_ndarray_dot() {
        use rand_distr::{Distribution, StandardNormal};
        let normal = StandardNormal;
        for &(m, k, n) in &[
            (16usize, 16, 16),       // small: ndarray.dot path
            (32, 64, 32),            // small: ndarray.dot path
            (128, 256, 128),         // crosses TEE_BLIS_THRESHOLD_ROWS (64)
            (2048, 2048, 1024),      // long-context apply shape (Q/K/V style)
            (2048, 2048, 6144),      // long-context apply shape (gate/up style)
            (2048, 6144, 2048),      // long-context FfnDown shape
        ] {
            let mut rng = rand_chacha::ChaCha20Rng::from_seed([42u8; 32]);
            let a = Array2::<f32>::from_shape_fn((m, k), |_| normal.sample(&mut rng));
            let b = Array2::<f32>::from_shape_fn((k, n), |_| normal.sample(&mut rng));

            let via_ndarray = a.dot(&b);
            let via_tee = tee_matmul(a.view(), b.view());

            assert_eq!(via_ndarray.dim(), via_tee.dim());

            // Max abs error tolerance: scales with k (accumulation depth)
            // and the operand RMS (~1.0 for standard normal). 2 * k * 1e-7
            // is the f32 epsilon-times-depth bound; we give 3× headroom.
            let tol = 6.0 * (k as f32) * f32::EPSILON;
            let mut max_abs = 0.0f32;
            for (x, y) in via_ndarray.iter().zip(via_tee.iter()) {
                max_abs = max_abs.max((x - y).abs());
            }
            assert!(
                max_abs <= tol,
                "tee_matmul vs ndarray.dot mismatch at (m={m}, k={k}, n={n}): max abs = {max_abs:.3e}, tol = {tol:.3e}"
            );
        }
    }
}
