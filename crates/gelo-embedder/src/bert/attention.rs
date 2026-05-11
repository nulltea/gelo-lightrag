use ndarray::{Array2, ArrayView2};

/// TEE-side scaled-dot-product multi-head attention.
///
/// Inputs are already-unmasked `Q`, `K`, `V` each of shape `(n, d)`. Output
/// is `(n, d)` ready for the output-projection offload.
pub fn multi_head_attention(
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    num_heads: usize,
) -> Array2<f32> {
    let n = q.nrows();
    let d = q.ncols();
    assert_eq!(d % num_heads, 0, "hidden dim must divide num_heads");
    let head_dim = d / num_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut output = Array2::<f32>::zeros((n, d));

    for h in 0..num_heads {
        let col_start = h * head_dim;
        let col_end = col_start + head_dim;
        let qh = q.slice(ndarray::s![.., col_start..col_end]);
        let kh = k.slice(ndarray::s![.., col_start..col_end]);
        let vh = v.slice(ndarray::s![.., col_start..col_end]);

        // (n, n) = (n, hd) · (hd, n)
        let mut scores = qh.dot(&kh.t());
        scores *= scale;

        // softmax along last axis (per query row)
        softmax_inplace(&mut scores);

        // (n, hd) = (n, n) · (n, hd)
        let ctx = scores.dot(&vh);

        let mut dst = output.slice_mut(ndarray::s![.., col_start..col_end]);
        dst.assign(&ctx);
    }
    output
}

fn softmax_inplace(scores: &mut Array2<f32>) {
    for mut row in scores.rows_mut() {
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            row /= sum;
        }
    }
}
