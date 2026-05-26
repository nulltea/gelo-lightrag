//! bf16-native variants of the forward-pass elementwise kernels —
//! phase 1 of the activation-precision migration (roadmap §4.E.3, plan
//! `m1-12-bf16-activation-pipeline.md` §4.3 step 2).
//!
//! Inputs and outputs are bf16. Arithmetic widens to f32 internally
//! for reductions (RMSNorm's sum-of-squares, inverse-sqrt) and for
//! per-element residual sums; the wider accumulator is load-bearing
//! at large `d` (the d=2560 RMS reduction can lose meaningful precision
//! if accumulated in bf16). Output narrows back to bf16 once.
//!
//! Not wired into `decoder::forward` yet — the production path still
//! holds activations as `Array2<f32>`. Phase 2 (separate session) will
//! consume `rms_norm_bf16` from the first decoder block's input flow
//! and validate end-to-end perf + token parity. This module ships the
//! kernel primitives and the parity contract so that phase 2 is a
//! call-site change only.

use half::bf16;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};

/// Element-count threshold above which the kernels rayon-parallelise
/// over rows. Matches the f32 reference `rms_norm`'s threshold so the
/// parallelism behaviour is consistent across precisions.
const PAR_THRESHOLD: usize = 32_768;

/// bf16-in / bf16-out RMSNorm. Per row `x ∈ R^d`:
///
/// ```text
///   ss   = Σᵢ to_f32(xᵢ)²                   (widened reduction)
///   inv  = 1 / sqrt(ss / d + eps)
///   yᵢ   = to_bf16(to_f32(xᵢ) · inv · γᵢ)   (narrow once at output)
/// ```
///
/// `γ` is kept f32 to match the loaded weight precision; the layer-norm
/// scale tensor is small and there is no bandwidth saving from bf16
/// `γ` storage.
pub fn rms_norm_bf16(x: ArrayView2<'_, bf16>, gamma: &[f32], eps: f32) -> Array2<bf16> {
    let d = x.ncols();
    assert_eq!(gamma.len(), d, "rms_norm_bf16: gamma length must equal hidden dim");
    let mut out = Array2::<bf16>::from_elem(x.raw_dim(), bf16::ZERO);
    let inv_d = 1.0_f32 / d as f32;
    let compute = |mut dst: ndarray::ArrayViewMut1<bf16>, row: ndarray::ArrayView1<bf16>| {
        let mut ss = 0.0_f32;
        for &v in row.iter() {
            let f = v.to_f32();
            ss += f * f;
        }
        let inv_denom = (ss * inv_d + eps).sqrt().recip();
        for ((d_v, &x_v), &g) in dst.iter_mut().zip(row.iter()).zip(gamma.iter()) {
            *d_v = bf16::from_f32(x_v.to_f32() * inv_denom * g);
        }
    };
    let elems = x.nrows() * x.ncols();
    if elems >= PAR_THRESHOLD {
        use ndarray::parallel::prelude::*;
        ndarray::Zip::from(out.axis_iter_mut(Axis(0)))
            .and(x.axis_iter(Axis(0)))
            .into_par_iter()
            .for_each(|(dst, row)| compute(dst, row));
    } else {
        for (dst, row) in out.axis_iter_mut(Axis(0)).zip(x.axis_iter(Axis(0))) {
            compute(dst, row);
        }
    }
    out
}

/// In-place per-head RMSNorm for bf16 Q/K projections. Same contract as
/// the f32 `apply_qk_norm` — each head's `head_dim` slice is reduced
/// independently. Sum-of-squares accumulates in f32; values written
/// back as bf16.
pub fn apply_qk_norm_bf16(
    mut qk: ArrayViewMut2<'_, bf16>,
    n_heads: usize,
    head_dim: usize,
    gamma: &[f32],
    eps: f32,
) {
    let n_cols = qk.ncols();
    assert_eq!(
        n_cols,
        n_heads * head_dim,
        "apply_qk_norm_bf16: expected n_heads({n_heads}) * head_dim({head_dim}) = {} cols, got {n_cols}",
        n_heads * head_dim,
    );
    assert_eq!(
        gamma.len(),
        head_dim,
        "apply_qk_norm_bf16: gamma length must equal head_dim"
    );
    let inv_d = 1.0_f32 / head_dim as f32;
    for mut row in qk.axis_iter_mut(Axis(0)) {
        let slice = row
            .as_slice_mut()
            .expect("rows of Array2 are contiguous by construction");
        for h in 0..n_heads {
            let s = h * head_dim;
            let e = s + head_dim;
            let head = &mut slice[s..e];
            let mut ss = 0.0_f32;
            for &v in head.iter() {
                let f = v.to_f32();
                ss += f * f;
            }
            let inv_denom = (ss * inv_d + eps).sqrt().recip();
            for (x, &g) in head.iter_mut().zip(gamma.iter()) {
                *x = bf16::from_f32(x.to_f32() * inv_denom * g);
            }
        }
    }
}

/// bf16-in / bf16-out element-wise sum (residual add). Both operands
/// widen to f32 for the add and the result narrows once. The plan
/// §4.1 notes that direct bf16 add is also safe because residual sums
/// are bounded — keeping the f32 widening here is the conservative
/// choice that pins the parity contract to the same precision as
/// every other elementwise kernel in this module.
pub fn residual_add_bf16(
    a: ArrayView2<'_, bf16>,
    b: ArrayView2<'_, bf16>,
) -> Array2<bf16> {
    assert_eq!(
        a.raw_dim(),
        b.raw_dim(),
        "residual_add_bf16: shape mismatch (got {:?} and {:?})",
        a.raw_dim(),
        b.raw_dim()
    );
    let mut out = Array2::<bf16>::from_elem(a.raw_dim(), bf16::ZERO);
    let compute = |mut dst: ndarray::ArrayViewMut1<bf16>,
                   ra: ndarray::ArrayView1<bf16>,
                   rb: ndarray::ArrayView1<bf16>| {
        for ((d_v, &x_v), &y_v) in dst.iter_mut().zip(ra.iter()).zip(rb.iter()) {
            *d_v = bf16::from_f32(x_v.to_f32() + y_v.to_f32());
        }
    };
    let elems = a.nrows() * a.ncols();
    if elems >= PAR_THRESHOLD {
        use ndarray::parallel::prelude::*;
        ndarray::Zip::from(out.axis_iter_mut(Axis(0)))
            .and(a.axis_iter(Axis(0)))
            .and(b.axis_iter(Axis(0)))
            .into_par_iter()
            .for_each(|(dst, ra, rb)| compute(dst, ra, rb));
    } else {
        for ((dst, ra), rb) in out
            .axis_iter_mut(Axis(0))
            .zip(a.axis_iter(Axis(0)))
            .zip(b.axis_iter(Axis(0)))
        {
            compute(dst, ra, rb);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::rms_norm::{apply_qk_norm, rms_norm};
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    /// Per-element absolute parity bound vs the f32 reference. bf16 has
    /// 8 mantissa bits → ~4e-3 relative error per stored value. For
    /// inputs in the unit-normal range, RMSNorm outputs are ~O(1) and
    /// residual sums are ~O(few); a 2e-2 abs bound clears both the
    /// rounding error on the bf16 inputs and the small accumulated
    /// drift from the f32 reduction reading bf16-truncated inputs.
    const BF16_PARITY_ABS: f32 = 2e-2;

    fn random_f32(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f64, 1.0_f64).unwrap();
        Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng) as f32)
    }

    fn quantise_to_bf16(x: &Array2<f32>) -> Array2<bf16> {
        let mut out = Array2::<bf16>::from_elem(x.raw_dim(), bf16::ZERO);
        for ((i, j), &v) in x.indexed_iter() {
            out[(i, j)] = bf16::from_f32(v);
        }
        out
    }

    fn max_abs_delta_f32_vs_bf16(a: &Array2<f32>, b: &Array2<bf16>) -> f32 {
        let mut max = 0.0_f32;
        for ((i, j), &av) in a.indexed_iter() {
            let bv = b[(i, j)].to_f32();
            let d = (av - bv).abs();
            if d > max {
                max = d;
            }
        }
        max
    }

    /// f32 reference vs bf16 implementation at the production hidden
    /// dim (Qwen3-4B `d = 2560`). Reduction over 2 560 squared terms
    /// is the precision-stress case for the kernel. `γ` is set to
    /// realistic production-weight magnitudes (near 1.0 with small
    /// trained deviations) so the output magnitudes stay O(1) and
    /// the bf16-floor abs error stays at the documented bound.
    #[test]
    fn rms_norm_bf16_parity_at_d_2560() {
        let x_f32 = random_f32(8, 2560, 42);
        let x_bf16 = quantise_to_bf16(&x_f32);
        // For a fair parity check, run the f32 reference on the
        // already-bf16-quantised inputs (widen back to f32) so that
        // only the *kernel*'s precision is being compared, not the
        // input quantisation noise.
        let x_f32_from_bf16 = x_bf16.mapv(|v| v.to_f32());
        // γ near 1.0 ± 5 % — matches the scale of trained Qwen3 RMSNorm
        // weights (the gamma tensor is initialised at 1.0 and drifts
        // slightly during training).
        let gamma: Vec<f32> = (0..2560).map(|i| 1.0 + 0.05 * ((i as f32 * 0.01).sin())).collect();
        let eps = 1e-6_f32;

        let ref_out = rms_norm(x_f32_from_bf16.view(), &gamma, eps);
        let test_out = rms_norm_bf16(x_bf16.view(), &gamma, eps);

        let delta = max_abs_delta_f32_vs_bf16(&ref_out, &test_out);
        assert!(
            delta < BF16_PARITY_ABS,
            "rms_norm_bf16 abs delta {delta} exceeds bf16-floor parity bound {BF16_PARITY_ABS}"
        );
    }

    /// Per-head RMSNorm at Qwen3-4B Q-projection shape (n_heads = 32,
    /// head_dim = 128). Smaller reduction than d=2560 so the parity
    /// gap is narrower in absolute terms but still bf16-floor-bound.
    #[test]
    fn apply_qk_norm_bf16_parity_at_head_dim_128() {
        let n_heads = 32;
        let head_dim = 128;
        let q_seed = random_f32(8, n_heads * head_dim, 7);
        let q_bf16_init = quantise_to_bf16(&q_seed);
        let mut q_f32 = q_bf16_init.mapv(|v| v.to_f32());
        let mut q_bf16 = q_bf16_init.clone();
        // γ near 1.0 ± 5 % per the production-weight magnitudes.
        let gamma: Vec<f32> = (0..head_dim).map(|i| 1.0 + 0.05 * ((i as f32 * 0.1).sin())).collect();
        let eps = 1e-6_f32;

        apply_qk_norm(q_f32.view_mut(), n_heads, head_dim, &gamma, eps);
        apply_qk_norm_bf16(q_bf16.view_mut(), n_heads, head_dim, &gamma, eps);

        let delta = max_abs_delta_f32_vs_bf16(&q_f32, &q_bf16);
        assert!(
            delta < BF16_PARITY_ABS,
            "apply_qk_norm_bf16 abs delta {delta} exceeds bf16-floor parity bound {BF16_PARITY_ABS}"
        );
    }

    /// Residual sum at production hidden-state shape (n=8, d=2560).
    /// The two summands are independent unit-normal so the sum is
    /// O(√2) in magnitude — well within bf16's dynamic range.
    #[test]
    fn residual_add_bf16_parity() {
        let a_f32 = random_f32(8, 2560, 11);
        let b_f32 = random_f32(8, 2560, 13);
        let a_bf16 = quantise_to_bf16(&a_f32);
        let b_bf16 = quantise_to_bf16(&b_f32);
        let a_f32_q = a_bf16.mapv(|v| v.to_f32());
        let b_f32_q = b_bf16.mapv(|v| v.to_f32());

        let ref_out = &a_f32_q + &b_f32_q;
        let test_out = residual_add_bf16(a_bf16.view(), b_bf16.view());

        let delta = max_abs_delta_f32_vs_bf16(&ref_out, &test_out);
        assert!(
            delta < BF16_PARITY_ABS,
            "residual_add_bf16 abs delta {delta} exceeds bf16-floor parity bound {BF16_PARITY_ABS}"
        );
    }

    /// Sanity: the kernels also work below the rayon threshold (small
    /// matrices still produce correct output).
    #[test]
    fn rms_norm_bf16_small_shape_serial() {
        let x_f32 = random_f32(4, 64, 1);
        let x_bf16 = quantise_to_bf16(&x_f32);
        let x_f32_q = x_bf16.mapv(|v| v.to_f32());
        let gamma = vec![1.0_f32; 64];

        let ref_out = rms_norm(x_f32_q.view(), &gamma, 1e-6);
        let test_out = rms_norm_bf16(x_bf16.view(), &gamma, 1e-6);
        let delta = max_abs_delta_f32_vs_bf16(&ref_out, &test_out);
        assert!(delta < BF16_PARITY_ABS);
    }
}
