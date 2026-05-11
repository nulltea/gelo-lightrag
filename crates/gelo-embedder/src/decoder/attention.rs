use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView2, Axis};

use gelo_protocol::profile;
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

/// Same as [`causal_gqa_attention`] but routes the per-head `Q · Kᵀ`
/// matmuls through [`TrustedExecutor::offload_attention_qkt_batched`] —
/// one fused OutAttnMult dispatch covering every Q head. Softmax, causal
/// mask, and the `attn · V` follow-up matmul stay inside the TEE. The
/// scale-by-`1/sqrt(head_dim)` is applied *after* the offloaded matmul
/// comes back.
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

    // Gather all Q heads into a (H, n, d_head) tensor and materialise the
    // K-side as (H, d_head, n) — with GQA, each Q head's K is its
    // KV-group head, replicated. This is a single per-layer allocation.
    let (q_batched, kt_batched) = profile::time("tee:gqa_batch_pack", || {
        let mut qb = Array3::<f32>::zeros((num_q_heads, n, head_dim));
        let mut ktb = Array3::<f32>::zeros((num_q_heads, head_dim, n));
        for qh in 0..num_q_heads {
            let q_off = qh * head_dim;
            let kvh = qh / group;
            let kv_off = kvh * head_dim;
            let q_view = q.slice(ndarray::s![.., q_off..q_off + head_dim]);
            let k_view = k.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
            qb.index_axis_mut(Axis(0), qh).assign(&q_view);
            ktb.index_axis_mut(Axis(0), qh).assign(&k_view.t());
        }
        (qb, ktb)
    });

    // One fused batched OutAttnMult dispatch.
    let scores_batched = exec.offload_attention_qkt_batched(q_batched.view(), kt_batched.view())?;

    // Per-head softmax + causal mask + V multiply, all in TEE.
    let mut output = Array2::<f32>::zeros((n, num_q_heads * head_dim));
    profile::time("tee:softmax_av", || {
        for qh in 0..num_q_heads {
            let q_off = qh * head_dim;
            let kvh = qh / group;
            let kv_off = kvh * head_dim;
            let vh_view = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]);

            let mut scores = scores_batched.index_axis(Axis(0), qh).to_owned();
            scores *= scale;
            apply_causal_mask(&mut scores);
            softmax_inplace(&mut scores);
            let ctx = scores.dot(&vh_view);
            output
                .slice_mut(ndarray::s![.., q_off..q_off + head_dim])
                .assign(&ctx);
        }
    });
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
