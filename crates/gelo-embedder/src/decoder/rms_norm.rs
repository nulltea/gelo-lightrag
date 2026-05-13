use ndarray::{Array2, ArrayView2, Axis};

/// Root-mean-square normalization (Zhang & Sennrich 2019). Used by Qwen3,
/// LLaMA, Mistral instead of LayerNorm.
///
/// Per row of `x ∈ R^(n, d)`:
///   out = (x / sqrt(mean(x²) + eps)) ⊙ γ
///
/// No mean-centering, no bias. Single-threaded with precomputed
/// `inv_denom`; rayon costs more than it saves at our per-call shape.
pub fn rms_norm(x: ArrayView2<'_, f32>, gamma: &[f32], eps: f32) -> Array2<f32> {
    let d = x.ncols();
    assert_eq!(gamma.len(), d, "rms_norm: gamma length must equal hidden dim");
    let mut out = Array2::<f32>::zeros(x.raw_dim());
    let inv_d = 1.0_f32 / d as f32;
    for (mut dst, row) in out.axis_iter_mut(Axis(0)).zip(x.axis_iter(Axis(0))) {
        let mut ss = 0.0_f32;
        for &v in row.iter() {
            ss += v * v;
        }
        let mean_sq = ss * inv_d;
        let inv_denom = (mean_sq + eps).sqrt().recip();
        for ((d_v, &x_v), &g) in dst.iter_mut().zip(row.iter()).zip(gamma.iter()) {
            *d_v = x_v * inv_denom * g;
        }
    }
    out
}
