use anyhow::Result;
use ndarray::{Array2, ArrayView2};

use gelo_protocol::TrustedExecutor;

/// Causal grouped-query attention. Inputs `q`, `k`, `v` already have
/// position-rotated values applied (RoPE handled by caller).
///
/// Shapes:
///   q: (n, num_q_heads × head_dim)
///   k: (n, num_kv_heads × head_dim)
///   v: (n, num_kv_heads × head_dim)
/// Output: (n, num_q_heads × head_dim)
///
/// GQA: each KV head feeds `num_q_heads / num_kv_heads` Q heads (replicate
/// on the fly, no materialisation).
pub fn causal_gqa_attention(
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
    assert!(num_q_heads >= num_kv_heads);
    assert_eq!(num_q_heads % num_kv_heads, 0);
    let group = num_q_heads / num_kv_heads;
    let n = q.nrows();
    assert_eq!(q.ncols(), num_q_heads * head_dim);
    assert_eq!(k.ncols(), num_kv_heads * head_dim);
    assert_eq!(v.ncols(), num_kv_heads * head_dim);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut output = Array2::<f32>::zeros((n, num_q_heads * head_dim));

    for qh in 0..num_q_heads {
        let kvh = qh / group;
        let q_off = qh * head_dim;
        let kv_off = kvh * head_dim;

        let qh_view = q.slice(ndarray::s![.., q_off..q_off + head_dim]);
        let kh_view = k.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let vh_view = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]);

        let mut scores = qh_view.dot(&kh_view.t());
        scores *= scale;
        apply_causal_mask(&mut scores);
        softmax_inplace(&mut scores);
        let ctx = scores.dot(&vh_view);

        let mut dst = output.slice_mut(ndarray::s![.., q_off..q_off + head_dim]);
        dst.assign(&ctx);
    }
    output
}

/// Same as [`causal_gqa_attention`] but routes each head's `Q · Kᵀ` matmul
/// through [`TrustedExecutor::offload_attention_qkt`] — the TwinShield
/// OutAttnMult path. Softmax, causal mask, and the `attn · V` follow-up
/// matmul stay inside the TEE. The scale-by-`1/sqrt(head_dim)` is applied
/// *after* the offloaded matmul comes back.
pub fn causal_gqa_attention_with_offload(
    exec: &mut impl TrustedExecutor,
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Array2<f32>> {
    assert!(num_q_heads >= num_kv_heads);
    assert_eq!(num_q_heads % num_kv_heads, 0);
    let group = num_q_heads / num_kv_heads;
    let n = q.nrows();
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut output = Array2::<f32>::zeros((n, num_q_heads * head_dim));

    for qh in 0..num_q_heads {
        let kvh = qh / group;
        let q_off = qh * head_dim;
        let kv_off = kvh * head_dim;

        let qh_view = q.slice(ndarray::s![.., q_off..q_off + head_dim]);
        let kh_view = k.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let vh_view = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]);

        // Transpose to materialise K^T (the protocol's right operand). This
        // is an n × d copy per head — small at typical head_dim (64-128).
        let kht = kh_view.t().to_owned();
        let mut scores = exec.offload_attention_qkt(qh_view, kht.view())?;
        scores *= scale;
        apply_causal_mask(&mut scores);
        softmax_inplace(&mut scores);
        let ctx = scores.dot(&vh_view);

        let mut dst = output.slice_mut(ndarray::s![.., q_off..q_off + head_dim]);
        dst.assign(&ctx);
    }
    Ok(output)
}

fn apply_causal_mask(scores: &mut Array2<f32>) {
    let n = scores.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            scores[[i, j]] = f32::NEG_INFINITY;
        }
    }
}

fn softmax_inplace(scores: &mut Array2<f32>) {
    for mut row in scores.rows_mut() {
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        // When the entire row is -inf (shouldn't happen for causal — row 0 still
        // has scores[0,0] valid), fall back to uniform.
        if !max.is_finite() {
            for v in row.iter_mut() {
                *v = 0.0;
            }
            continue;
        }
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
