use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView2, Axis};

use gelo_protocol::TrustedExecutor;
use gelo_protocol::attention::AttentionMask;
use gelo_protocol::profile;

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

/// Permutation-shielded causal GQA attention (Tier 1). Offloads all
/// three heavy ops — Q·Kᵀ, softmax, and ·V — to the GPU engine under
/// a fresh per-batch row permutation + optional Gaussian noise. Causal
/// mask is transformed by π on the TEE side and added to the score
/// tensor before the engine softmax dispatch.
///
/// Same input/output shapes as [`causal_gqa_attention_with_offload`]:
///   q: (n, num_q_heads * head_dim)
///   k: (n, num_kv_heads * head_dim)
///   v: (n, num_kv_heads * head_dim)
///   → (n, num_q_heads * head_dim)
pub fn causal_gqa_attention_permuted(
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

    // Reshape: (n, num_q_heads * head_dim) → (num_q_heads, n, head_dim).
    // K, V get GQA replication: each KV head feeds `group` Q heads.
    let (q3, k3, v3) = profile::time("tee:perm_attn_pack", || {
        let mut q3 = Array3::<f32>::zeros((num_q_heads, n, head_dim));
        let mut k3 = Array3::<f32>::zeros((num_q_heads, n, head_dim));
        let mut v3 = Array3::<f32>::zeros((num_q_heads, n, head_dim));
        for qh in 0..num_q_heads {
            let q_off = qh * head_dim;
            let kvh = qh / group;
            let kv_off = kvh * head_dim;
            q3.index_axis_mut(Axis(0), qh)
                .assign(&q.slice(ndarray::s![.., q_off..q_off + head_dim]));
            k3.index_axis_mut(Axis(0), qh)
                .assign(&k.slice(ndarray::s![.., kv_off..kv_off + head_dim]));
            v3.index_axis_mut(Axis(0), qh)
                .assign(&v.slice(ndarray::s![.., kv_off..kv_off + head_dim]));
        }
        (q3, k3, v3)
    });

    // One protocol call: permute + (optional) noise + GPU matmul +
    // GPU softmax + GPU matmul, with TEE-side causal mask injection.
    let out3 = exec.offload_attention_permuted(
        q3.view(),
        k3.view(),
        v3.view(),
        scale,
        AttentionMask::Causal,
    )?;

    // Reshape: (num_q_heads, n, head_dim) → (n, num_q_heads * head_dim).
    let mut output = Array2::<f32>::zeros((n, num_q_heads * head_dim));
    profile::time("tee:perm_attn_unpack", || {
        for qh in 0..num_q_heads {
            let q_off = qh * head_dim;
            output
                .slice_mut(ndarray::s![.., q_off..q_off + head_dim])
                .assign(&out3.index_axis(Axis(0), qh));
        }
    });
    Ok(output)
}

fn apply_causal_mask(scores: &mut Array2<f32>) {
    // Mask the strictly upper triangle. Writing through contiguous row
    // slices lets LLVM emit a tight memset-style store (avoiding the
    // per-element bounds-check that `scores[[i, j]]` would force).
    for (i, mut row) in scores.axis_iter_mut(Axis(0)).enumerate() {
        let row_slice = row
            .as_slice_mut()
            .expect("Array2 rows are contiguous in row-major");
        for v in row_slice.iter_mut().skip(i + 1) {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn softmax_inplace(scores: &mut Array2<f32>) {
    // Same tight pattern as BERT's softmax (Tier 2.3.a): contiguous
    // &mut [f32] slice iter for the two passes per row, fused exp+sum,
    // single reciprocal multiply for the normalise pass. The decoder
    // path has a causal-mask `-inf` corner case to handle (row may be
    // all-`-inf` only on a degenerate path; defensive zeroing kept).
    for mut row in scores.axis_iter_mut(Axis(0)) {
        let row_slice = row
            .as_slice_mut()
            .expect("Array2 rows are contiguous in row-major");
        let mut max = f32::NEG_INFINITY;
        for &v in row_slice.iter() {
            if v > max {
                max = v;
            }
        }
        if !max.is_finite() {
            for v in row_slice.iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        let mut sum = 0.0_f32;
        for v in row_slice.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            let inv_sum = 1.0_f32 / sum;
            for v in row_slice.iter_mut() {
                *v *= inv_sum;
            }
        }
    }
}
