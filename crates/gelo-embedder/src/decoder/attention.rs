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

    let per_head = |qh: usize, mut dst: ndarray::ArrayViewMut2<f32>| {
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
        dst.assign(&ctx);
    };

    // Same rayon threshold as `bert::attention::multi_head_attention`:
    // per-head work scales as `2 · n² · head_dim` flops + softmax. At
    // Qwen3 rerank shape (n ≈ 400, head_dim = 128, 16 heads) one head
    // is ~16 M flops ≈ 1 ms — well past rayon break-even. Embedder shape
    // (n ≈ 30, 16 heads) gets ~100 k flops per head ≈ 5 µs — sequential
    // wins.
    if n >= 64 {
        use ndarray::parallel::prelude::*;
        output
            .axis_chunks_iter_mut(Axis(1), head_dim)
            .into_par_iter()
            .enumerate()
            .for_each(|(qh, dst)| per_head(qh, dst));
    } else {
        for qh in 0..num_q_heads {
            let q_off = qh * head_dim;
            let dst = output.slice_mut(ndarray::s![.., q_off..q_off + head_dim]);
            per_head(qh, dst);
        }
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
        let per_head = |qh: usize, mut dst: ndarray::ArrayViewMut2<f32>| {
            let kvh = qh / group;
            let kv_off = kvh * head_dim;
            let vh_view = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]);

            let mut scores = scores_batched.index_axis(Axis(0), qh).to_owned();
            scores *= scale;
            apply_causal_mask(&mut scores);
            softmax_inplace(&mut scores);
            let ctx = scores.dot(&vh_view);
            dst.assign(&ctx);
        };
        if n >= 64 {
            use ndarray::parallel::prelude::*;
            output
                .axis_chunks_iter_mut(Axis(1), head_dim)
                .into_par_iter()
                .enumerate()
                .for_each(|(qh, dst)| per_head(qh, dst));
        } else {
            for qh in 0..num_q_heads {
                let q_off = qh * head_dim;
                let dst = output.slice_mut(ndarray::s![.., q_off..q_off + head_dim]);
                per_head(qh, dst);
            }
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

/// Asymmetric causal GQA attention. Same semantics as
/// [`causal_gqa_attention`] but supports `n_q ≤ n_kv` — the shape that
/// arises during autoregressive decode where one new query token attends
/// to a growing KV cache.
///
/// Shapes:
///   q: (n_q,  num_q_heads  × head_dim)
///   k: (n_kv, num_kv_heads × head_dim)
///   v: (n_kv, num_kv_heads × head_dim)
///   → (n_q, num_q_heads × head_dim)
///
/// `q_pos_offset` is the absolute position of Q row 0 in the full
/// sequence. Q row `i` then sits at absolute position
/// `q_pos_offset + i` and may attend to K rows `0..=(q_pos_offset + i)`.
/// For decode (`n_q = 1`, `q_pos_offset = n_kv - 1`) every Q row sees
/// every K row, so the mask is a no-op. For prefill (`n_q = n_kv`,
/// `q_pos_offset = 0`) the result matches [`causal_gqa_attention`].
pub fn causal_gqa_attention_cached(
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_pos_offset: usize,
) -> Array2<f32> {
    assert!(num_q_heads >= num_kv_heads);
    assert_eq!(num_q_heads % num_kv_heads, 0);
    let group = num_q_heads / num_kv_heads;
    let n_q = q.nrows();
    let n_kv = k.nrows();
    assert_eq!(v.nrows(), n_kv, "K and V must agree on n_kv");
    assert!(
        n_q + q_pos_offset <= n_kv,
        "asymmetric attention: q_pos_offset {q_pos_offset} + n_q {n_q} > n_kv {n_kv}",
    );
    assert_eq!(q.ncols(), num_q_heads * head_dim);
    assert_eq!(k.ncols(), num_kv_heads * head_dim);
    assert_eq!(v.ncols(), num_kv_heads * head_dim);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut output = Array2::<f32>::zeros((n_q, num_q_heads * head_dim));

    let per_head = |qh: usize, mut dst: ndarray::ArrayViewMut2<f32>| {
        let kvh = qh / group;
        let q_off = qh * head_dim;
        let kv_off = kvh * head_dim;

        let qh_view = q.slice(ndarray::s![.., q_off..q_off + head_dim]);
        let kh_view = k.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let vh_view = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]);

        let mut scores = qh_view.dot(&kh_view.t()); // (n_q, n_kv)
        scores *= scale;
        apply_asymmetric_causal_mask(&mut scores, q_pos_offset);
        softmax_inplace(&mut scores);
        let ctx = scores.dot(&vh_view);
        dst.assign(&ctx);
    };

    // Parallelisation threshold: per-head work is `n_q · n_kv · head_dim`.
    // At decode shape (n_q=1) this is tiny — sequential wins; at prefill
    // shape (n_q=n_kv) we reuse the same 64-row threshold as the
    // symmetric path.
    if n_q >= 64 {
        use ndarray::parallel::prelude::*;
        output
            .axis_chunks_iter_mut(Axis(1), head_dim)
            .into_par_iter()
            .enumerate()
            .for_each(|(qh, dst)| per_head(qh, dst));
    } else {
        for qh in 0..num_q_heads {
            let q_off = qh * head_dim;
            let dst = output.slice_mut(ndarray::s![.., q_off..q_off + head_dim]);
            per_head(qh, dst);
        }
    }
    output
}

/// Permutation-shielded asymmetric causal GQA attention. Same shape
/// contract as [`causal_gqa_attention_cached`] but offloads the heavy
/// path (Q·Kᵀ, ·V) through the trusted executor's
/// `offload_attention_permuted_cached`. Causal masking + softmax stay
/// in-TEE per the F1+ resolution (see
/// `docs/plans/m1-10-security-review.md`).
///
/// Engaged when `cfg.use_perm_attention && n_q ≥ perm_attention_threshold`
/// on Global layers; the cost crossover is set by the same threshold
/// the embedder path uses (default 64), since the per-call protocol
/// overhead doesn't depend on n_kv — it's amortised by the Q·K^T
/// matmul whose compute scales with n_q.
pub fn causal_gqa_attention_permuted_cached(
    exec: &mut impl TrustedExecutor,
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_pos_offset: usize,
) -> Result<Array2<f32>> {
    assert!(num_q_heads >= num_kv_heads);
    assert_eq!(num_q_heads % num_kv_heads, 0);
    let group = num_q_heads / num_kv_heads;
    let n_q = q.nrows();
    let n_kv = k.nrows();
    assert_eq!(v.nrows(), n_kv, "K and V must agree on n_kv");
    assert!(
        n_q + q_pos_offset <= n_kv,
        "asymmetric attention: q_pos_offset {q_pos_offset} + n_q {n_q} > n_kv {n_kv}",
    );
    assert_eq!(q.ncols(), num_q_heads * head_dim);
    assert_eq!(k.ncols(), num_kv_heads * head_dim);
    assert_eq!(v.ncols(), num_kv_heads * head_dim);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Reshape (n, num_q_heads · head_dim) → (num_q_heads, n, head_dim),
    // GQA-replicating K/V so each Q head has its own KV-head view.
    let (q3, k3, v3) = profile::time("tee:perm_attn_cached_pack", || {
        let mut q3 = Array3::<f32>::zeros((num_q_heads, n_q, head_dim));
        let mut k3 = Array3::<f32>::zeros((num_q_heads, n_kv, head_dim));
        let mut v3 = Array3::<f32>::zeros((num_q_heads, n_kv, head_dim));
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

    let out3 = exec.offload_attention_permuted_cached(
        q3.view(),
        k3.view(),
        v3.view(),
        scale,
        q_pos_offset,
        AttentionMask::Causal,
    )?;

    let mut output = Array2::<f32>::zeros((n_q, num_q_heads * head_dim));
    profile::time("tee:perm_attn_cached_unpack", || {
        for qh in 0..num_q_heads {
            let q_off = qh * head_dim;
            output
                .slice_mut(ndarray::s![.., q_off..q_off + head_dim])
                .assign(&out3.index_axis(Axis(0), qh));
        }
    });
    Ok(output)
}

fn apply_asymmetric_causal_mask(scores: &mut Array2<f32>, q_pos_offset: usize) {
    // Score row `i` corresponds to Q absolute position `q_pos_offset + i`.
    // Mask out K columns strictly greater than that absolute position.
    for (i, mut row) in scores.axis_iter_mut(Axis(0)).enumerate() {
        let abs_q = q_pos_offset + i;
        let row_slice = row
            .as_slice_mut()
            .expect("Array2 rows are contiguous in row-major");
        for v in row_slice.iter_mut().skip(abs_q + 1) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Sliding-window causal GQA attention (Gemma 4 local layers).
///
/// Same shape contract as [`causal_gqa_attention_cached`] but Q row at
/// absolute position `p = q_pos_offset + i` attends only to K rows in
/// `[max(0, p - window + 1), p]` — i.e. the most recent `window`
/// tokens including itself. Sets the rest of the row to `-inf` before
/// softmax.
///
/// When `window >= n_kv` (no clamping ever fires), the kernel is
/// identical to [`causal_gqa_attention_cached`]; this collapse is the
/// load-bearing property the v1 parity tests check.
///
/// **Why this stays in-TEE.** The band-diagonal mask is not
/// permutation-invariant (per `docs/prototype/gelo-llm.html` §02), so
/// the permuted-attention path doesn't extend to SWA. OutAttnMult's
/// 2n×2n operand destroys the band sparsity. v1 therefore runs SWA
/// on CPU under AOCL-BLIS — cheap at every realistic context length
/// because the per-layer cost is `O(n · window)` rather than
/// `O(n²)`.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_attention_swa_cached(
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_pos_offset: usize,
    window: usize,
) -> Array2<f32> {
    assert!(num_q_heads >= num_kv_heads);
    assert_eq!(num_q_heads % num_kv_heads, 0);
    assert!(window > 0, "swa window must be > 0");
    let group = num_q_heads / num_kv_heads;
    let n_q = q.nrows();
    let n_kv = k.nrows();
    assert_eq!(v.nrows(), n_kv, "K and V must agree on n_kv");
    assert!(
        n_q + q_pos_offset <= n_kv,
        "swa attention: q_pos_offset {q_pos_offset} + n_q {n_q} > n_kv {n_kv}",
    );
    assert_eq!(q.ncols(), num_q_heads * head_dim);
    assert_eq!(k.ncols(), num_kv_heads * head_dim);
    assert_eq!(v.ncols(), num_kv_heads * head_dim);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut output = Array2::<f32>::zeros((n_q, num_q_heads * head_dim));

    let per_head = |qh: usize, mut dst: ndarray::ArrayViewMut2<f32>| {
        let kvh = qh / group;
        let q_off = qh * head_dim;
        let kv_off = kvh * head_dim;

        let qh_view = q.slice(ndarray::s![.., q_off..q_off + head_dim]);
        let kh_view = k.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let vh_view = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]);

        let mut scores = qh_view.dot(&kh_view.t()); // (n_q, n_kv)
        scores *= scale;
        apply_swa_causal_mask(&mut scores, q_pos_offset, window);
        softmax_inplace(&mut scores);
        let ctx = scores.dot(&vh_view);
        dst.assign(&ctx);
    };

    if n_q >= 64 {
        use ndarray::parallel::prelude::*;
        output
            .axis_chunks_iter_mut(Axis(1), head_dim)
            .into_par_iter()
            .enumerate()
            .for_each(|(qh, dst)| per_head(qh, dst));
    } else {
        for qh in 0..num_q_heads {
            let q_off = qh * head_dim;
            let dst = output.slice_mut(ndarray::s![.., q_off..q_off + head_dim]);
            per_head(qh, dst);
        }
    }
    output
}

fn apply_swa_causal_mask(scores: &mut Array2<f32>, q_pos_offset: usize, window: usize) {
    // Score row `i` is Q at absolute position p = q_pos_offset + i. It
    // attends to K columns in [lo, p], where lo = max(0, p - window + 1).
    // Mask everything outside that band.
    for (i, mut row) in scores.axis_iter_mut(Axis(0)).enumerate() {
        let abs_q = q_pos_offset + i;
        let lo = abs_q.saturating_sub(window - 1);
        let hi = abs_q; // inclusive upper bound on K column index
        let row_slice = row
            .as_slice_mut()
            .expect("Array2 rows are contiguous in row-major");
        // Head end: K columns 0..lo are outside the window → mask.
        for v in row_slice.iter_mut().take(lo) {
            *v = f32::NEG_INFINITY;
        }
        // Tail end: K columns hi+1.. are future positions → mask
        // (causal). Re-uses the asymmetric-causal contract.
        for v in row_slice.iter_mut().skip(hi + 1) {
            *v = f32::NEG_INFINITY;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn rand_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        // Deterministic LCG so the test is reproducible without pulling
        // in a full RNG crate dep. The values are not statistically
        // meaningful, only stable.
        let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut m = Array2::<f32>::zeros((rows, cols));
        for v in m.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((state >> 33) as f32 / u32::MAX as f32) - 0.5;
        }
        m
    }

    #[test]
    fn cached_attention_matches_square_for_prefill_shape() {
        // n_q = n_kv, q_pos_offset = 0 → must equal the existing causal
        // attention output bit-for-bit (same code path inside the per-head
        // closure, just with the parameterised mask).
        let n = 8;
        let n_q_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 4;
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let q = rand_matrix(n, q_dim, 1);
        let k = rand_matrix(n, kv_dim, 2);
        let v = rand_matrix(n, kv_dim, 3);

        let sym =
            causal_gqa_attention(q.view(), k.view(), v.view(), n_q_heads, n_kv_heads, head_dim);
        let cached = causal_gqa_attention_cached(
            q.view(),
            k.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            0,
        );
        for (a, b) in sym.iter().zip(cached.iter()) {
            assert!((a - b).abs() < 1e-6, "sym {a} vs cached {b}");
        }
    }

    #[test]
    fn decode_step_matches_corresponding_prefill_row() {
        // Equivalent to the decode-replay property: running prefill on
        // [t_0..t_p] and taking row p must equal running prefill on
        // [t_0..t_{p-1}] then decoding one step at position p with t_p.
        //
        // Here we simulate this at the attention level only — no
        // projections, no RoPE: build (n_kv+1, kv_dim) of K, V; the full
        // prefill output's last row must equal the cached-attention
        // output for n_q=1 with q_pos_offset = n_kv.
        let total = 7;
        let n_q_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 4;
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q_full = rand_matrix(total, q_dim, 11);
        let k_full = rand_matrix(total, kv_dim, 12);
        let v_full = rand_matrix(total, kv_dim, 13);

        let full = causal_gqa_attention(
            q_full.view(),
            k_full.view(),
            v_full.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
        );
        let last_pos = total - 1;
        let last_row_from_full = full.row(last_pos).to_owned();

        // Decode replay: K, V cover everything 0..=last_pos; Q is just
        // the last row; q_pos_offset = last_pos so the mask is wide open.
        let q_decode = q_full
            .slice(ndarray::s![last_pos..last_pos + 1, ..])
            .to_owned();
        let decode = causal_gqa_attention_cached(
            q_decode.view(),
            k_full.view(),
            v_full.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            last_pos,
        );
        let decode_row = decode.row(0).to_owned();

        for (a, b) in last_row_from_full.iter().zip(decode_row.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "decode replay mismatch: full={a} decode={b}",
            );
        }
    }

    #[test]
    fn swa_with_window_ge_seq_matches_dense_causal() {
        // Window >= n_kv → no positions ever masked out at the head;
        // result equals dense causal attention.
        let n = 8;
        let n_q_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 4;
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let q = rand_matrix(n, q_dim, 31);
        let k = rand_matrix(n, kv_dim, 32);
        let v = rand_matrix(n, kv_dim, 33);

        let dense = causal_gqa_attention_cached(
            q.view(),
            k.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            0,
        );
        let swa = causal_gqa_attention_swa_cached(
            q.view(),
            k.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            0,
            999, // window much bigger than n
        );
        for (a, b) in dense.iter().zip(swa.iter()) {
            assert!((a - b).abs() < 1e-6, "swa(W>=n) vs dense: {a} {b}");
        }
    }

    #[test]
    fn swa_window_one_attends_to_self_only() {
        // Window=1 means each Q row sees only its own absolute position
        // in K. The softmax of one element is 1.0; the result for row p
        // is simply V[p] for that head.
        let n = 6;
        let n_q_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q = rand_matrix(n, q_dim, 41);
        let k = rand_matrix(n, kv_dim, 42);
        let v = rand_matrix(n, kv_dim, 43);

        let swa = causal_gqa_attention_swa_cached(
            q.view(),
            k.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            0,
            1,
        );

        // For each Q row i, head qh, the output should equal V row i
        // for the corresponding KV head — modulo any non-causal
        // softmax fallback. Compare against the V row directly.
        for i in 0..n {
            for qh in 0..n_q_heads {
                let kv_off = (qh / 2) * head_dim; // group=2 so kvh=0 for both
                let q_off = qh * head_dim;
                for d in 0..head_dim {
                    let got = swa[[i, q_off + d]];
                    let want = v[[i, kv_off + d]];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "swa(W=1) row {i} head {qh} dim {d}: got {got} want {want}",
                    );
                }
            }
        }
    }

    #[test]
    fn swa_window_below_p_truly_clips_old_keys() {
        // Build a sequence where K row 0 has a wildly different value
        // than the rest. With W = 3 and Q at p = 5, the kernel must
        // not see K row 0; varying K row 0 must NOT change the output.
        let n = 6;
        let n_q_heads = 1;
        let n_kv_heads = 1;
        let head_dim = 4;
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q = rand_matrix(n, q_dim, 51);
        let mut k = rand_matrix(n, kv_dim, 52);
        let v = rand_matrix(n, kv_dim, 53);

        // Run with original K row 0.
        let out_a = causal_gqa_attention_swa_cached(
            q.view(),
            k.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            0,
            3, // window
        );

        // Mutate K row 0 dramatically.
        for d in 0..kv_dim {
            k[[0, d]] = 1e6;
        }
        let out_b = causal_gqa_attention_swa_cached(
            q.view(),
            k.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            0,
            3,
        );

        // Row 0 must differ (K row 0 is in-window for itself when
        // p=0). Rows 3, 4, 5 must NOT change (they don't see K row 0
        // at window=3: row 3 attends to [1,2,3]; row 4 to [2,3,4];
        // row 5 to [3,4,5]).
        let row3_unchanged = (0..q_dim).all(|d| (out_a[[3, d]] - out_b[[3, d]]).abs() < 1e-5);
        let row5_unchanged = (0..q_dim).all(|d| (out_a[[5, d]] - out_b[[5, d]]).abs() < 1e-5);
        assert!(row3_unchanged, "swa window=3 leaked K[0] into row 3");
        assert!(row5_unchanged, "swa window=3 leaked K[0] into row 5");
    }

    #[test]
    fn swa_decode_step_with_offset_attends_to_window_only() {
        // Decode shape: n_q = 1 at absolute position p = 5, full K, V
        // span = 6, window = 3. The single Q row must attend to K rows
        // [3, 4, 5] only. Compare against a full prefill with the same
        // SWA mask and take row 5: must match.
        let total = 6;
        let n_q_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let window = 3;

        let q_full = rand_matrix(total, q_dim, 61);
        let k_full = rand_matrix(total, kv_dim, 62);
        let v_full = rand_matrix(total, kv_dim, 63);

        let full = causal_gqa_attention_swa_cached(
            q_full.view(),
            k_full.view(),
            v_full.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            0,
            window,
        );
        let last_pos = total - 1;
        let full_last = full.row(last_pos).to_owned();

        let q_decode = q_full
            .slice(ndarray::s![last_pos..last_pos + 1, ..])
            .to_owned();
        let decode = causal_gqa_attention_swa_cached(
            q_decode.view(),
            k_full.view(),
            v_full.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            last_pos,
            window,
        );

        for (a, b) in full_last.iter().zip(decode.row(0).iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "swa decode replay: full row {last_pos} {a} vs decode {b}",
            );
        }
    }

    #[test]
    fn asymmetric_mask_blocks_future_positions() {
        // Build scores where the un-masked attention would produce
        // identifiable contributions from later K rows. After masking,
        // the asymmetric path must agree with what the square causal
        // path produces over the same K, V prefix.
        let n_kv = 6;
        let n_q = 3; // covers Q positions q_pos_offset..q_pos_offset+n_q-1
        let n_q_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let q_pos_offset = 2; // Q rows 0,1,2 → abs positions 2,3,4

        let q_band = rand_matrix(n_q, q_dim, 21);
        let k_full = rand_matrix(n_kv, kv_dim, 22);
        let v_full = rand_matrix(n_kv, kv_dim, 23);

        // Reference: build the equivalent square problem of size
        // (q_pos_offset + n_q, …), put `q_band` at rows
        // q_pos_offset..q_pos_offset+n_q, run the symmetric kernel, take
        // the corresponding rows.
        let total = q_pos_offset + n_q;
        let mut q_full = Array2::<f32>::zeros((total, q_dim));
        q_full
            .slice_mut(ndarray::s![q_pos_offset..total, ..])
            .assign(&q_band);
        let k_pref = k_full.slice(ndarray::s![..total, ..]).to_owned();
        let v_pref = v_full.slice(ndarray::s![..total, ..]).to_owned();
        let ref_full = causal_gqa_attention(
            q_full.view(),
            k_pref.view(),
            v_pref.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
        );
        let ref_band = ref_full
            .slice(ndarray::s![q_pos_offset..total, ..])
            .to_owned();

        // Asymmetric kernel over the full K, V (n_kv = 6 includes future
        // positions that the causal mask must zero out).
        let cached = causal_gqa_attention_cached(
            q_band.view(),
            k_full.view(),
            v_full.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            q_pos_offset,
        );

        for (a, b) in ref_band.iter().zip(cached.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "asymmetric mask broke causal: ref={a} cached={b}",
            );
        }
    }
}
