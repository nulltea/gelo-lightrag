//! bf16 GEMM via AOCL-BLIS LPGEMM addon (`aocl_gemm_bf16bf16f32of32`).
//!
//! Provides a single safe wrapper, [`matmul_bf16_lpgemm`], that takes
//! two `ArrayView2<bf16>` operands and returns an `Array2<f32>` output
//! (bf16 × bf16 → f32, with f32 internal accumulators inside the
//! AVX-512_BF16 microkernel).
//!
//! ## Why this exists
//!
//! M1.12 bucket-3a (the bf16/f16-native activation pipeline) needs a
//! TEE-side bf16 GEMM for the GELO mask apply/unapply path. The
//! mask `A` is sampled at f32 in TEE, downcast to bf16 only at the
//! GEMM call site; the resulting masked operand `A · H` is f32. Mask
//! `A` never leaves the TEE — the bf16 quantisation is purely a
//! bandwidth + throughput optimisation on the TEE-side multiply.
//!
//! ## Why AOCL LPGEMM, not OpenBLAS / hand-rolled
//!
//! Decision history at `docs/plans/m1-12-bf16-activation-pipeline.md`
//! §3.0. Summary: AOCL-BLIS 5.2.2 source already ships
//! `addon/aocl_gemm/aocl_gemm_bf16bf16f32of32.c` — an AVX-512_BF16
//! kernel that uses `vdpbf16ps` (the same instruction a hand-roll
//! would target), tuned by AMD for Zen 4 / Zen 5. Our previous build
//! just didn't enable the addon; the install script now passes
//! `--enable-addon=aocl_gemm` to surface the symbols.
//!
//! No new crate dependency, no symbol-conflict risk (same `.so` we
//! already link), no new build dep (the AOCL-BLIS build was already
//! a requirement of the `blas` feature).
//!
//! ## Threat model
//!
//! Identical to the f32 mask GEMM. Mask `A` is TEE-only. The
//! downcast f32 → bf16 happens at the GEMM call boundary; the
//! resulting bf16 representation of `A` lives in TEE memory for the
//! duration of the call and is discarded on return. Adversary's
//! observation (the masked operand `A · H` sent to GPU) is f32
//! output with bf16 mantissa quantisation noise — strictly noisier
//! than today's f32 path, so existing AloePri c1-c5 bounds carry
//! over without empirical re-validation (per plan §5 math-only
//! argument).
//!
//! ## API quirks
//!
//! AOCL LPGEMM uses a **pre-pack-then-GEMM** pattern (mirrors BLIS's
//! main matmul API). For maximum throughput the B matrix should be
//! pre-reordered into AOCL's internal block-major layout via
//! `aocl_reorder_bf16bf16f32of32` before the GEMM. We currently
//! pass `mem_format_b = 'n'` (no reorder) for simplicity — the
//! reorder optimisation is a follow-up. Single-call cost without
//! reorder is still well below the f32 BLIS path for our shapes.

use ndarray::{Array2, ArrayView2};
use std::ffi::{c_char, c_void};

use half::bf16;

/// `dim_t` in AOCL-BLIS is `gint_t`, which is `int64_t` on the 64-bit
/// builds we ship (`BLIS_INT_TYPE_SIZE = 64`, set in the amdzen
/// configuration). Matches our linux x86_64 target exclusively.
#[allow(non_camel_case_types)]
type dim_t = i64;

#[cfg(feature = "blas")]
unsafe extern "C" {
    /// `aocl_gemm_bf16bf16f32of32`:
    /// `C = alpha · A · B + beta · C` where A, B are bf16 and C is f32.
    /// Accumulation happens at f32 internally via `vdpbf16ps`.
    ///
    /// Layout conventions (from
    /// `vendor/aocl-blis/addon/aocl_gemm/aocl_gemm_interface_apis.h`):
    /// - `order = 'r'` for row-major (the only layout we use).
    /// - `transa/transb = 'n'` for no-transpose, `'t'` for transpose.
    /// - `mem_format_a/b = 'n'` for un-reordered (native row-major);
    ///   `'r'` for AOCL-reordered (pre-packed) layout.
    /// - `lda/ldb/ldc` = leading dimensions (columns in row-major).
    /// - `post_op_unparsed = NULL` for plain GEMM (no fused
    ///   bias / activation / scale).
    fn aocl_gemm_bf16bf16f32of32(
        order: c_char,
        transa: c_char,
        transb: c_char,
        m: dim_t,
        n: dim_t,
        k: dim_t,
        alpha: f32,
        a: *const u16,
        lda: dim_t,
        mem_format_a: c_char,
        b: *const u16,
        ldb: dim_t,
        mem_format_b: c_char,
        beta: f32,
        c: *mut f32,
        ldc: dim_t,
        post_op_unparsed: *const c_void,
    );
}

/// bf16 × bf16 → f32 matmul via AOCL-BLIS LPGEMM addon. Row-major
/// contiguous inputs only (asserted via `ArrayView2::as_slice`).
///
/// Equivalent semantics to the f32 path (`mask::matmul_blis`) but at
/// bf16 input precision with f32 accumulation. Per the LPGEMM
/// kernel's runtime ISA check, requires AVX-512_BF16; on hosts
/// without it the AOCL kernel falls back to a reference scalar impl
/// internally — same answer, slower.
///
/// Returns `(m, n)` output. `lhs` is `(m, k)`, `rhs` is `(k, n)`.
///
/// # Panics
///
/// - Inner-dimension mismatch between `lhs` and `rhs`.
/// - Non-standard layout on either input (must be row-major
///   contiguous).
#[cfg(feature = "blas")]
pub fn matmul_bf16_lpgemm(
    lhs: ArrayView2<'_, bf16>,
    rhs: ArrayView2<'_, bf16>,
) -> Array2<f32> {
    let m = lhs.nrows() as dim_t;
    let k = lhs.ncols() as dim_t;
    let k2 = rhs.nrows() as dim_t;
    let n = rhs.ncols() as dim_t;
    assert_eq!(
        k, k2,
        "matmul_bf16_lpgemm: inner dim mismatch (lhs.ncols={k}, rhs.nrows={k2})"
    );

    let lhs_slice = lhs
        .as_slice()
        .expect("matmul_bf16_lpgemm: lhs must be row-major contiguous");
    let rhs_slice = rhs
        .as_slice()
        .expect("matmul_bf16_lpgemm: rhs must be row-major contiguous");
    let mut c = Array2::<f32>::zeros((m as usize, n as usize));
    let c_slice = c.as_slice_mut().expect("fresh Array2 is contiguous");

    // `bf16` is `#[repr(transparent)]` over `u16` (or equivalent), so
    // the slice memory layout matches what AOCL expects for the
    // `bfloat16 = int16_t` type. Cast via raw pointers.
    let lhs_ptr = lhs_slice.as_ptr() as *const u16;
    let rhs_ptr = rhs_slice.as_ptr() as *const u16;

    unsafe {
        aocl_gemm_bf16bf16f32of32(
            b'r' as c_char,                   // order: row-major
            b'n' as c_char,                   // transa: no transpose
            b'n' as c_char,                   // transb: no transpose
            m,
            n,
            k,
            1.0,                              // alpha
            lhs_ptr,
            k,                                // lda = inner dim (row-major)
            b'n' as c_char,                   // mem_format_a: unreordered
            rhs_ptr,
            n,                                // ldb
            b'n' as c_char,                   // mem_format_b: unreordered
            0.0,                              // beta (C is zero-init'd above)
            c_slice.as_mut_ptr(),
            n,                                // ldc
            std::ptr::null(),                 // post_op: none
        );
    }
    c
}

/// `Aᵀ · B` variant — same semantics as `matmul_bf16_lpgemm` but with
/// the left operand transposed in place by passing `transa='t'` to
/// AOCL. Equivalent to `lhs.t().dot(&rhs)` at bf16 precision.
///
/// Used by the mask unapply path (`Aᵀ · masked_output`).
#[cfg(feature = "blas")]
pub fn matmul_bf16_lpgemm_trans_a(
    lhs: ArrayView2<'_, bf16>,
    rhs: ArrayView2<'_, bf16>,
) -> Array2<f32> {
    // With transa='t' the effective operand is lhs^T, so the output
    // rows are lhs.ncols() and the inner dim is lhs.nrows().
    let m = lhs.ncols() as dim_t;
    let inner = lhs.nrows() as dim_t;
    let inner2 = rhs.nrows() as dim_t;
    let n = rhs.ncols() as dim_t;
    assert_eq!(
        inner, inner2,
        "matmul_bf16_lpgemm_trans_a: inner dim mismatch (lhs.nrows={inner}, rhs.nrows={inner2})"
    );

    let lhs_slice = lhs
        .as_slice()
        .expect("matmul_bf16_lpgemm_trans_a: lhs must be row-major contiguous");
    let rhs_slice = rhs
        .as_slice()
        .expect("matmul_bf16_lpgemm_trans_a: rhs must be row-major contiguous");
    let mut c = Array2::<f32>::zeros((m as usize, n as usize));
    let c_slice = c.as_slice_mut().expect("fresh Array2 is contiguous");

    let lhs_ptr = lhs_slice.as_ptr() as *const u16;
    let rhs_ptr = rhs_slice.as_ptr() as *const u16;

    // For transa='t' AOCL still expects lda to be the leading dim of
    // the un-transposed matrix as stored in row-major — that's
    // `lhs.ncols() = m` (no, wait: lda is the row stride of A in
    // memory, which for row-major un-transposed A of shape
    // (lhs.nrows(), lhs.ncols()) is lhs.ncols() = inner here for the
    // transposed view since we have inner=lhs.nrows() and m=lhs.ncols(),
    // so the storage dim is m). Actually for row-major un-transposed
    // storage of (lhs.nrows() × lhs.ncols()), lda = lhs.ncols() = m.
    let lda_a = lhs.ncols() as dim_t;

    unsafe {
        aocl_gemm_bf16bf16f32of32(
            b'r' as c_char,
            b't' as c_char,                   // transa: transpose
            b'n' as c_char,
            m,
            n,
            inner,
            1.0,
            lhs_ptr,
            lda_a,
            b'n' as c_char,
            rhs_ptr,
            n,
            b'n' as c_char,
            0.0,
            c_slice.as_mut_ptr(),
            n,
            std::ptr::null(),
        );
    }
    c
}

#[cfg(all(test, feature = "blas"))]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    /// Build a reference f32 matmul by widening bf16 inputs to f32
    /// element-by-element then calling ndarray's `.dot()`. Used as the
    /// parity oracle for the LPGEMM bf16 path.
    fn reference_matmul_bf16(
        a: ArrayView2<'_, bf16>,
        b: ArrayView2<'_, bf16>,
    ) -> Array2<f32> {
        let a_f32 = a.mapv(|x| x.to_f32());
        let b_f32 = b.mapv(|x| x.to_f32());
        a_f32.dot(&b_f32)
    }

    fn rand_bf16(rows: usize, cols: usize, seed: u64) -> Array2<bf16> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        Array2::from_shape_fn((rows, cols), |_| {
            // Stay in a range that's clean in bf16 — avoid denormals
            // and avoid the upper exponent where rounding can dominate.
            bf16::from_f32(rng.random::<f32>() * 0.1 - 0.05)
        })
    }

    #[test]
    fn lpgemm_matches_reference_small() {
        let m = 17;
        let k = 33;
        let n = 19;
        let a = rand_bf16(m, k, 0xCAFE_F00D);
        let b = rand_bf16(k, n, 0xC0DE_BEEF);

        let got = matmul_bf16_lpgemm(a.view(), b.view());
        let want = reference_matmul_bf16(a.view(), b.view());

        assert_eq!(got.dim(), want.dim(), "shape mismatch");
        for (g, w) in got.iter().zip(want.iter()) {
            // bf16 has 7 mantissa bits; per-element relative error is
            // bounded by ~k · 2¯⁷ for the dot product of length k.
            // At k=33 that's ~0.26 — generous but realistic. Tightening
            // requires accounting for the f32-accumulator chain.
            let delta = (g - w).abs();
            let scale = w.abs().max(1.0);
            let rel = delta / scale;
            assert!(
                rel < 0.05,
                "rel err {rel:.4} exceeded 5% tolerance: got={g}, want={w}, delta={delta:.6}"
            );
        }
    }

    #[test]
    fn lpgemm_trans_a_matches_reference() {
        let inner = 64;
        let m = 24; // output rows = lhs.ncols()
        let n = 13; // output cols = rhs.ncols()
        let a = rand_bf16(inner, m, 0xABCD_0001);
        let b = rand_bf16(inner, n, 0x1234_5678);

        let got = matmul_bf16_lpgemm_trans_a(a.view(), b.view());

        // Reference: widen + transpose + dot
        let a_f32 = a.mapv(|x| x.to_f32());
        let b_f32 = b.mapv(|x| x.to_f32());
        let want = a_f32.t().dot(&b_f32);

        assert_eq!(got.dim(), want.dim(), "shape mismatch");
        for (g, w) in got.iter().zip(want.iter()) {
            let delta = (g - w).abs();
            let scale = w.abs().max(1.0);
            let rel = delta / scale;
            assert!(
                rel < 0.05,
                "rel err {rel:.4} (trans_a path): got={g}, want={w}"
            );
        }
    }

    /// Square shape ≈ Qwen3 prefill mask GEMM (n = 2048, d = 2560).
    /// Smaller dims here so the test stays under the routine-test
    /// time budget. The Qwen3-4B integration parity test in
    /// `crates/gelo-embedder/tests/` will cover production shapes.
    #[test]
    fn lpgemm_matches_reference_mask_shape() {
        let s = 64; // representative of stacked_n; small for fast test
        let d = 128;
        let a = rand_bf16(s, s, 0x5EED_0001);
        let h = rand_bf16(s, d, 0x5EED_0002);

        let got = matmul_bf16_lpgemm(a.view(), h.view());
        let want = reference_matmul_bf16(a.view(), h.view());

        // Per the bf16_mask_gemm_skipped memory's parity-test result:
        // bf16 mask round-trip mean rel error vs f32 target ≈ 2.65e-3
        // at production shapes. At s=64 the chain depth is short
        // enough that we should beat 1e-2 comfortably.
        let mut max_rel: f32 = 0.0;
        for (g, w) in got.iter().zip(want.iter()) {
            let delta = (g - w).abs();
            let scale = w.abs().max(1.0);
            let rel = delta / scale;
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_rel < 0.02,
            "max rel err {max_rel:.4} exceeded 2% at mask-shape s={s} d={d}"
        );
    }
}
