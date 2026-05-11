use ndarray::{Array2, ArrayView2};

/// Root-mean-square normalization (Zhang & Sennrich 2019). Used by Qwen3,
/// LLaMA, Mistral instead of LayerNorm.
///
/// Per row of `x ∈ R^(n, d)`:
///   out = (x / sqrt(mean(x²) + eps)) ⊙ γ
///
/// No mean-centering, no bias.
pub fn rms_norm(x: ArrayView2<'_, f32>, gamma: &[f32], eps: f32) -> Array2<f32> {
    let d = x.ncols();
    assert_eq!(gamma.len(), d, "rms_norm: gamma length must equal hidden dim");
    let mut out = Array2::<f32>::zeros(x.raw_dim());
    let inv_d = 1.0_f32 / d as f32;
    for (i, row) in x.rows().into_iter().enumerate() {
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() * inv_d;
        let denom = (mean_sq + eps).sqrt();
        let mut dst = out.row_mut(i);
        for (j, v) in row.iter().enumerate() {
            dst[j] = (*v / denom) * gamma[j];
        }
    }
    out
}
