use ndarray::{Array1, ArrayView2};

/// Mean pooling over the token axis, followed by L2 normalization.
/// Inputs: `hidden` shape `(n, d)`. Output: `(d,)`.
///
/// Used by sentence-transformers / BGE family. Identical math to
/// `sentence_transformers.models.Pooling(pooling_mode_mean_tokens=True) +
/// models.Normalize()`.
pub fn mean_l2(hidden: ArrayView2<'_, f32>) -> Array1<f32> {
    let n = hidden.nrows() as f32;
    let mut summed = Array1::<f32>::zeros(hidden.ncols());
    for row in hidden.rows() {
        summed += &row;
    }
    summed /= n;
    let norm = summed.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        summed /= norm;
    }
    summed
}
