use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};

/// Root-mean-square normalization (Zhang & Sennrich 2019). Used by Qwen3,
/// LLaMA, Mistral instead of LayerNorm.
///
/// Per row of `x ∈ R^(n, d)`:
///   out = (x / sqrt(mean(x²) + eps)) ⊙ γ
///
/// No mean-centering, no bias. Rows are independent → rayon-parallel
/// when the matrix clears the `n × d ≥ 32 768` cutoff (matches the
/// bert::forward elementwise threshold). Rerank shape (n ≈ 400,
/// d = 1024 = 410 k elements) is comfortably above; embedder query
/// shape (n ≈ 30 × 1024 = 31 k) stays serial.
pub fn rms_norm(x: ArrayView2<'_, f32>, gamma: &[f32], eps: f32) -> Array2<f32> {
    let d = x.ncols();
    assert_eq!(gamma.len(), d, "rms_norm: gamma length must equal hidden dim");
    let mut out = Array2::<f32>::zeros(x.raw_dim());
    let inv_d = 1.0_f32 / d as f32;
    let compute = |mut dst: ndarray::ArrayViewMut1<f32>, row: ndarray::ArrayView1<f32>| {
        let mut ss = 0.0_f32;
        for &v in row.iter() {
            ss += v * v;
        }
        let mean_sq = ss * inv_d;
        let inv_denom = (mean_sq + eps).sqrt().recip();
        for ((d_v, &x_v), &g) in dst.iter_mut().zip(row.iter()).zip(gamma.iter()) {
            *d_v = x_v * inv_denom * g;
        }
    };
    let elems = x.nrows() * x.ncols();
    const PAR_THRESHOLD: usize = 32_768;
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

/// Per-head RMSNorm applied in-place to a `(n_tokens, n_heads * head_dim)`
/// Q or K projection. Used by Qwen3's QK-norm: each head's slice of the
/// projection is independently normalised against `gamma ∈ R^(head_dim,)`.
///
/// Layout assumption matches the Qwen3 / HF transformers `q_proj` /
/// `k_proj` output: heads are contiguous along the last axis, so
/// `qk[t, h*head_dim + d]` is head `h`'s `d`-th channel for token `t`.
pub fn apply_qk_norm(
    mut qk: ArrayViewMut2<'_, f32>,
    n_heads: usize,
    head_dim: usize,
    gamma: &[f32],
    eps: f32,
) {
    let n_cols = qk.ncols();
    assert_eq!(
        n_cols, n_heads * head_dim,
        "apply_qk_norm: expected n_heads({n_heads}) * head_dim({head_dim}) = {} cols, got {n_cols}",
        n_heads * head_dim,
    );
    assert_eq!(gamma.len(), head_dim, "apply_qk_norm: gamma length must equal head_dim");
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
                ss += v * v;
            }
            let inv_denom = (ss * inv_d + eps).sqrt().recip();
            for (x, &g) in head.iter_mut().zip(gamma.iter()) {
                *x = *x * inv_denom * g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn apply_qk_norm_normalises_each_head_independently() {
        // 2 tokens, 3 heads, head_dim = 4 → cols = 12
        let mut q = Array2::<f32>::from_shape_vec(
            (2, 12),
            (0..24).map(|x| x as f32).collect(),
        )
        .unwrap();
        let gamma = vec![1.0_f32; 4];
        // Compare against an explicit per-head RMSNorm reference.
        let mut expected = q.clone();
        let eps = 1e-6_f32;
        for t in 0..2 {
            for h in 0..3 {
                let base = h * 4;
                let head: Vec<f32> = (0..4).map(|d| expected[(t, base + d)]).collect();
                let ss: f32 = head.iter().map(|v| v * v).sum();
                let inv = (ss / 4.0 + eps).sqrt().recip();
                for d in 0..4 {
                    expected[(t, base + d)] = head[d] * inv;
                }
            }
        }
        apply_qk_norm(q.view_mut(), 3, 4, &gamma, eps);
        for t in 0..2 {
            for d in 0..12 {
                let got = q[(t, d)];
                let want = expected[(t, d)];
                assert!((got - want).abs() < 1e-6, "mismatch at ({t},{d}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn apply_qk_norm_applies_gamma_per_channel() {
        let mut q = Array2::<f32>::ones((1, 4));
        let gamma = vec![2.0_f32, 0.5, -1.0, 1.0];
        apply_qk_norm(q.view_mut(), 1, 4, &gamma, 0.0);
        // All inputs equal → RMS = 1 → out[d] = 1 / 1 * gamma[d].
        assert!((q[(0, 0)] - 2.0).abs() < 1e-6);
        assert!((q[(0, 1)] - 0.5).abs() < 1e-6);
        assert!((q[(0, 2)] - (-1.0)).abs() < 1e-6);
        assert!((q[(0, 3)] - 1.0).abs() < 1e-6);
    }
}
