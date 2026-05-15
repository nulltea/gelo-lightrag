use ndarray::{Array2, ArrayView2, Axis};

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
