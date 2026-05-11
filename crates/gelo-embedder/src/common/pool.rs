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
    l2_normalize(&mut summed);
    summed
}

/// Last-token pooling: take the final row of `hidden` and L2-normalize.
/// Used by decoder-LLM-as-embedder models (E5-Mistral, Qwen3-Embedding,
/// NV-Embed). Output: `(d,)`.
pub fn last_l2(hidden: ArrayView2<'_, f32>) -> Array1<f32> {
    let n = hidden.nrows();
    debug_assert!(n > 0, "last_l2: empty hidden state");
    let mut last: Array1<f32> = hidden.row(n - 1).to_owned();
    l2_normalize(&mut last);
    last
}

fn l2_normalize(v: &mut Array1<f32>) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        *v /= norm;
    }
}
