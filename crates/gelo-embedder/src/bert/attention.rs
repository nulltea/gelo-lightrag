use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView2, Axis};

use gelo_protocol::{TrustedExecutor, profile};

/// TEE-side scaled-dot-product multi-head attention.
///
/// Inputs are already-unmasked `Q`, `K`, `V` each of shape `(n, d)`. Output
/// is `(n, d)` ready for the output-projection offload.
///
/// Q·Kᵀ and (scores)·V use `ndarray::dot` which dispatches to
/// `matrixmultiply` for matrix-matrix products — SIMD + cache-tiled.
/// The hot softmax inner loop is hand-rolled with a fused max+exp+sum
/// single-pass (cutting 3 row scans down to 2) plus a reciprocal
/// multiply instead of `Array1::/=` per row.
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

        // (n, n) = (n, hd) · (hd, n) — matrixmultiply-backed GEMM.
        let mut scores = qh.dot(&kh.t());
        scores *= scale;

        // softmax along last axis (per query row), in-place.
        softmax_inplace(&mut scores);

        // (n, hd) = (n, n) · (n, hd) — matrixmultiply-backed GEMM.
        let ctx = scores.dot(&vh);

        let mut dst = output.slice_mut(ndarray::s![.., col_start..col_end]);
        dst.assign(&ctx);
    }
    output
}

/// Same as [`multi_head_attention`] but routes the per-head `Q · Kᵀ`
/// matmuls through [`TrustedExecutor::offload_attention_qkt_batched`] —
/// one fused OutAttnMult dispatch covering every head. Softmax and the
/// `attn · V` follow-up matmul stay inside the TEE. The
/// scale-by-`1/sqrt(head_dim)` is applied *after* the offloaded matmul
/// comes back.
///
/// Mirrors `gelo_embedder::decoder::attention::causal_gqa_attention_with_offload`
/// but for bidirectional (BERT-class) attention: same shape contract,
/// no causal mask, no GQA replication (BERT uses full MHA).
pub fn multi_head_attention_with_offload(
    exec: &mut impl TrustedExecutor,
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    num_heads: usize,
) -> Result<Array2<f32>> {
    let n = q.nrows();
    let d = q.ncols();
    assert_eq!(d % num_heads, 0, "hidden dim must divide num_heads");
    let head_dim = d / num_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Pack all heads into (H, n, d_head) and (H, d_head, n) for the
    // batched offload. One per-layer allocation.
    let (q_batched, kt_batched) = profile::time("tee:bert_mha_batch_pack", || {
        let mut qb = Array3::<f32>::zeros((num_heads, n, head_dim));
        let mut ktb = Array3::<f32>::zeros((num_heads, head_dim, n));
        for h in 0..num_heads {
            let off = h * head_dim;
            let q_view = q.slice(ndarray::s![.., off..off + head_dim]);
            let k_view = k.slice(ndarray::s![.., off..off + head_dim]);
            qb.index_axis_mut(Axis(0), h).assign(&q_view);
            ktb.index_axis_mut(Axis(0), h).assign(&k_view.t());
        }
        (qb, ktb)
    });

    // One fused batched OutAttnMult dispatch.
    let scores_batched =
        exec.offload_attention_qkt_batched(q_batched.view(), kt_batched.view())?;

    // Per-head softmax + V multiply, all in TEE. No causal mask.
    let mut output = Array2::<f32>::zeros((n, num_heads * head_dim));
    profile::time("tee:bert_softmax_av", || {
        for h in 0..num_heads {
            let off = h * head_dim;
            let vh_view = v.slice(ndarray::s![.., off..off + head_dim]);

            let mut scores = scores_batched.index_axis(Axis(0), h).to_owned();
            scores *= scale;
            softmax_inplace(&mut scores);
            let ctx = scores.dot(&vh_view);
            output
                .slice_mut(ndarray::s![.., off..off + head_dim])
                .assign(&ctx);
        }
    });
    Ok(output)
}

/// Row-wise softmax over a (rows, cols) matrix.
///
/// Tight inner loops over contiguous &mut [f32] slices. Two passes per
/// row (max+exp+sum fused into the first, normalize via reciprocal
/// multiply in the second). LLVM auto-vectorises the second pass; the
/// first stays scalar because `f32::exp` is a libcall.
fn softmax_inplace(scores: &mut Array2<f32>) {
    for mut row in scores.axis_iter_mut(Axis(0)) {
        let row_slice = row
            .as_slice_mut()
            .expect("Array2 rows are contiguous in row-major");

        // Pass 1: find max (numeric stability), then exp(x − max) and
        // accumulate sum. Two sequential reads of the row, kept in one
        // loop with a branch — branch predictor handles the max update
        // cheaply since max-finds are mostly monotonic in practice.
        let mut max = f32::NEG_INFINITY;
        for &v in row_slice.iter() {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0_f32;
        for v in row_slice.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }

        // Pass 2: normalize by reciprocal multiply (one division total
        // instead of one per element under the old `row /= sum`).
        if sum > 0.0 {
            let inv_sum = 1.0_f32 / sum;
            for v in row_slice.iter_mut() {
                *v *= inv_sum;
            }
        }
    }
}
