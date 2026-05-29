//! Permutation-shielded attention protocol (Tier 1).
//!
//! Implements the softmax-equivariance identity from Amulet
//! (arXiv 2512.07495):
//!
//! ```text
//!   softmax(π·Q·Kᵀ·πᵀ / √d) · π·V  =  π · softmax(Q·Kᵀ / √d) · V
//! ```
//!
//! Combined with optional additive Gaussian noise `η ~ N(0, σ²·I)` on
//! Q and K, as the Hidden No More (arXiv 2505.18332) mitigation against
//! sequential-vocabulary-matching attacks on fixed permutations. The
//! per-batch fresh `π_b` is what keeps us out of the broken
//! fixed-permutation class.
//!
//! Phase 2 keeps all operations TEE-side; the function shape is what
//! Phase 3 will swap to a GPU `softmax_batched` engine call. The math
//! is locked down by `tests/permutation_attention.rs` (Phase 1).
//!
//! **Note on shield rows.** Shield rows (`shield.rs`) do NOT compose
//! with attention — softmax normalisation across data + shield tokens
//! corrupts the recovered data-row outputs. The `permuted_attention`
//! function assumes Q, K, V are clean tensors *after* shield-row strip
//! has already happened (i.e. produced by `offload_linear` /
//! `offload_qkv` with shield ON). The attention block adds its own
//! permutation + noise; it does not re-add shield rows.

use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis, s};
use rand::{RngCore, seq::SliceRandom};
use rand_distr::{Distribution, StandardNormal};

use crate::substrate::{FusedAttentionBatch, GpuOffloadEngine, RuntimeMatmulBatch};

/// Configuration for the permutation-shielded attention protocol.
#[derive(Debug, Clone, Copy)]
pub struct PermAttnConfig {
    /// Per-element standard deviation of the Gaussian noise added to
    /// Q and K. `0.0` disables noise (pure permutation equivariance).
    /// Hidden No More reports σ = 0.01 as the threshold where their
    /// recovery attack drops to ROUGE < 0.1.
    pub noise_sigma: f32,
    /// Magnitude of the soft causal-mask penalty applied at blocked
    /// positions (i.e. mask value `-causal_mask_neg`, replacing the
    /// prior `f32::NEG_INFINITY`). Defaults to `30.0`.
    ///
    /// Why not `-∞`. With `-∞` at blocked positions, softmax outputs
    /// exact `0` there. If softmax is offloaded to the GPU, the
    /// engine sees the input score tensor with exact `-∞` entries and
    /// recovers `π` row-by-row from the count of blocked positions
    /// — see `docs/plans/m1-10-security-review.md` F1+ for the
    /// detailed argument. With `-C ≈ 30`, `exp(-30) ≈ 9.4e-14`, which
    /// is at f32 noise floor for typical activation magnitudes after
    /// softmax. The count-of-zeros attack on softmax output no longer
    /// works. Per-row attention weight at blocked positions is
    /// O(1e-13), negligible relative to f32 precision of the
    /// allowed-position weights.
    ///
    /// Only consulted when `AttentionMask::Causal` is in use; the
    /// `None` variant ignores this field. Effective only on the
    /// in-TEE-softmax path that lands with the F1+ resolution.
    pub causal_mask_neg: f32,
    /// **Phase 1b** — opt-in dispatch of softmax to the GPU at
    /// **decode** (and only at decode, where the F1+ mask-pattern leak
    /// is structurally absent because `n_q == 1` + `q_pos_offset ==
    /// n_kv − 1` makes the causal mask a no-op).  When this flag is
    /// `true` and the call-site shape is decode-shaped,
    /// `permuted_attention_cached` calls `engine.softmax_batched`
    /// instead of the in-TEE rowwise softmax.  A cheap row-sum
    /// integrity probe runs in-TEE on one randomly-chosen row to
    /// detect Byzantine GPU softmax that breaks the probability-sum
    /// invariant.
    ///
    /// **Security gate (open)** — the `c5_perm_attn` AloePri
    /// attack-suite condition (HNM vocab-matching at decode, anchor-
    /// ICA / JADE / JD / Gram-error at decode shapes) must be re-run
    /// before this is flipped on by default in production.  See
    /// memory `aloepri_hd3_gate_phase_a_b.md` for the C3 precedent.
    ///
    /// Default `false`.  Prefill always uses the in-TEE-softmax path
    /// regardless of this flag (the F1+ attack does apply at prefill).
    pub decode_softmax_on_gpu: bool,
    /// **Feature-axis rotation cover** (`docs/plans/perm-attn-gpu-offload.md`).
    /// When `true`, each head's Q,K are rotated by a fresh orthogonal
    /// `O_qk` (which cancels in `Q·Kᵀ`, so scores are unchanged) and V by
    /// an independent `O_v` (corrected by `O_vᵀ` on the recovered output).
    /// This hides the operands' absolute coordinates from the engine —
    /// the lever against reference-equipped ARROWMATCH recovery that
    /// σ-noise cannot provide (see the gate-2 analysis). Exactly
    /// invertible, so σ=0 parity is preserved.
    ///
    /// Sampled fresh per call here; the persistent-K/V session fixes
    /// `O_v`/`O_qk` at prefill (the cadence is a security-gated knob).
    /// Default `false` — production cover is unchanged until the gate clears.
    pub feature_rotation: bool,
}

impl PermAttnConfig {
    /// Pure permutation, no noise. Bit-exact equivariance.
    pub const DISABLED_NOISE: Self = Self {
        noise_sigma: 0.0,
        causal_mask_neg: 30.0,
        decode_softmax_on_gpu: false,
        feature_rotation: false,
    };

    /// Hidden No More mitigation level. Default for production.
    pub const HIDDEN_NO_MORE: Self = Self {
        noise_sigma: 0.01,
        causal_mask_neg: 30.0,
        decode_softmax_on_gpu: false,
        feature_rotation: false,
    };

    /// **Phase 1b** — Hidden No More + GPU softmax at decode.  The
    /// production target for Amulet-at-decode once the
    /// `c5_perm_attn` gate passes.
    pub const HIDDEN_NO_MORE_DECODE_GPU: Self = Self {
        noise_sigma: 0.01,
        causal_mask_neg: 30.0,
        decode_softmax_on_gpu: true,
        feature_rotation: false,
    };
}

impl Default for PermAttnConfig {
    fn default() -> Self {
        Self::DISABLED_NOISE
    }
}

/// What attention mask (if any) to apply before softmax in the
/// permutation-shielded attention protocol.
#[derive(Debug, Clone, Copy)]
pub enum AttentionMask {
    /// No mask — full bidirectional attention. Used by encoder-style
    /// models (e.g. BGE).
    None,
    /// Causal upper-triangular mask. For decoder-style models the mask
    /// is transformed by π so position `i` attends only to positions `j`
    /// with `perm[j] ≤ perm[i]`. Math: `mask'[i,j] = -C if perm[j] >
    /// perm[i] else 0` (with `C = cfg.causal_mask_neg`, default 30) —
    /// direct algebra shows this preserves the equivariance identity
    /// to f32 precision (see `tests/permutation_attention.rs`).
    ///
    /// **F1+ resolution.** Softmax runs in-TEE under this mask
    /// (`engine.softmax_batched` would expose the mask pattern); the
    /// soft penalty `-C` (rather than `-∞`) prevents recovery of `π`
    /// from the softmax output's zero-pattern — see
    /// `docs/plans/m1-10-security-review.md`.
    Causal,
}

/// Compute `softmax(Q·Kᵀ / √d) · V` for every head, under the
/// permutation-shielded protocol. Offloads the three heavy ops
/// (Q·Kᵀ, softmax, ·V) to the engine so they can run on the GPU
/// in one dispatch chain.
///
/// `q`, `k`, `v` shape: `(num_heads, n, d_head)`. Result shape:
/// `(num_heads, n, d_head)`.
///
/// `scale` is the attention scale (typically `1 / √d_head`).
///
/// `mask` selects no mask (encoder-style) or causal mask (decoder LMs);
/// for the causal variant the mask is transformed by π before being
/// added to the engine's score tensor on the TEE side (cheap O(n²) work
/// shared across heads).
///
/// The fresh per-batch row permutation `π_b ∈ S_n` is sampled once and
/// shared across all heads within this block. The Hidden No More
/// per-head decoupling can be added later by sampling one π per head.
///
/// Engine usage (F1+ — in-TEE softmax):
/// - `matmul_dynamic_batched` for `(πQ + η_Q)(πK + η_K)ᵀ` (batched over heads)
/// - **in-TEE** row-wise softmax — keeps the causal mask off the GPU
/// - `matmul_dynamic_batched` for `probs · πV` (batched over heads)
///
/// The `softmax_batched` engine method is intentionally NOT used here
/// even though the trait exposes it: with `AttentionMask::Causal` the
/// engine would see `(score + mask)` and recover `π` from the
/// `-C` / `0` count per row. Encoder-style callers
/// (`AttentionMask::None`) get the same in-TEE-softmax for protocol
/// uniformity — at the shapes our embedder uses (n ≈ 256-512) the
/// difference is in the noise.
pub fn permuted_attention<R: RngCore, E: GpuOffloadEngine + ?Sized>(
    engine: &E,
    q: ArrayView3<'_, f32>,
    k: ArrayView3<'_, f32>,
    v: ArrayView3<'_, f32>,
    scale: f32,
    mask: AttentionMask,
    cfg: PermAttnConfig,
    rng: &mut R,
) -> Result<Array3<f32>> {
    let (num_heads, n, d_head) = q.dim();
    if k.dim() != (num_heads, n, d_head) {
        return Err(anyhow::anyhow!(
            "permuted_attention: K shape {:?} must match Q {:?}",
            k.dim(),
            q.dim()
        ));
    }
    if v.dim() != (num_heads, n, d_head) {
        return Err(anyhow::anyhow!(
            "permuted_attention: V shape {:?} must match Q {:?}",
            v.dim(),
            q.dim()
        ));
    }

    // Sample one π_b for this attention block, shared across heads.
    let perm = sample_permutation(n, rng);

    // Permute Q, K, V on the token axis (TEE side, cheap O(n·d) per head).
    let mut q_perm = Array3::<f32>::zeros((num_heads, n, d_head));
    let mut k_perm = Array3::<f32>::zeros((num_heads, n, d_head));
    let mut v_perm = Array3::<f32>::zeros((num_heads, n, d_head));
    for h in 0..num_heads {
        let qh = q.index_axis(Axis(0), h);
        let kh = k.index_axis(Axis(0), h);
        let vh = v.index_axis(Axis(0), h);
        for (i, &src) in perm.iter().enumerate() {
            q_perm.slice_mut(s![h, i, ..]).assign(&qh.row(src));
            k_perm.slice_mut(s![h, i, ..]).assign(&kh.row(src));
            v_perm.slice_mut(s![h, i, ..]).assign(&vh.row(src));
        }
    }

    // Optional N(0, σ²·I) noise on πQ and πK (Hidden No More mitigation).
    if cfg.noise_sigma > 0.0 {
        add_gaussian_3d_inplace(q_perm.view_mut(), cfg.noise_sigma, rng);
        add_gaussian_3d_inplace(k_perm.view_mut(), cfg.noise_sigma, rng);
    }

    // Build Kᵀ over the last two axes as a zero-copy view. burn-cubecl /
    // ndarray handle the non-contiguous stride at upload time — far
    // cheaper than a per-element transpose loop at our shape (Qwen3:
    // 16×400×128 = 820k entries per layer × 28 layers).
    let kt_perm_view = k_perm.view().permuted_axes([0, 2, 1]);

    // GPU step 1: scores = (πQ + η_Q) · (πK + η_K)ᵀ shape (num_heads, n, n).
    let mut scores = engine.run_runtime_matmul(RuntimeMatmulBatch {
        lhs: q_perm.view(),
        rhs: kt_perm_view,
    })?;
    scores.mapv_inplace(|x| x * scale);

    // TEE step: apply permuted causal mask if requested. The mask is the
    // same for every head, so we add it in a broadcast pattern.
    //
    // F1+ resolution: blocked positions get `-cfg.causal_mask_neg`
    // (default -30), NOT `-f32::INFINITY`. exp(-30) ≈ 9.4e-14 falls
    // below f32 noise floor after softmax — no exact zeros, no
    // count attack on the softmax output. See
    // docs/plans/m1-10-security-review.md.
    if let AttentionMask::Causal = mask {
        let neg = -cfg.causal_mask_neg;
        let mut mask_mat = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            let pi = perm[i];
            for j in 0..n {
                if perm[j] > pi {
                    mask_mat[(i, j)] = neg;
                }
            }
        }
        for h in 0..num_heads {
            for i in 0..n {
                for j in 0..n {
                    scores[(h, i, j)] += mask_mat[(i, j)];
                }
            }
        }
    }

    // TEE step 2 (F1+): softmax in-TEE. Was `engine.softmax_batched`
    // pre-2026-05-18 — moved on-TEE because the engine would otherwise
    // see the mask pattern on the input scores and recover π.
    let mut probs = Array3::<f32>::zeros((num_heads, n, n));
    for h in 0..num_heads {
        let scores_h = scores.index_axis(Axis(0), h);
        let probs_h = softmax_rowwise(scores_h);
        probs.index_axis_mut(Axis(0), h).assign(&probs_h);
    }
    let _ = engine; // kept in signature for the next two GPU calls

    // GPU step 3: out_perm = probs · πV shape (num_heads, n, d_head).
    let out_perm = engine.run_runtime_matmul(RuntimeMatmulBatch {
        lhs: probs.view(),
        rhs: v_perm.view(),
    })?;

    // TEE recovery via πᵀ: out[h, perm[i], :] = out_perm[h, i, :].
    let mut out = Array3::<f32>::zeros((num_heads, n, d_head));
    for h in 0..num_heads {
        for (i, &src) in perm.iter().enumerate() {
            out.slice_mut(s![h, src, ..])
                .assign(&out_perm.slice(s![h, i, ..]));
        }
    }

    Ok(out)
}

/// Asymmetric permutation-shielded attention for the cached-KV
/// generation shape. Generalises [`permuted_attention`] to `n_q ≤ n_kv`
/// by sampling **two independent** row permutations — one over the
/// Q axis (`π_q ∈ S_{n_q}`), one over the K/V axis
/// (`π_kv ∈ S_{n_kv}`) — and applying the asymmetric Amulet identity:
///
/// ```text
///   softmax(πQ·Q · (πKV·K)ᵀ · /√d + M_perm) · πKV·V
///   = πQ · softmax(Q·Kᵀ/√d + M_orig) · V
/// ```
///
/// where `M_perm[i,j] = M_orig[πQ(i), πKV(j)]` and the original
/// causal mask is
/// `M_orig[i,j] = 0 if j ≤ q_pos_offset + i else -∞` (replaced by
/// `-cfg.causal_mask_neg` under F1+).
///
/// Shapes:
///   q: `(num_heads, n_q,  d_head)`
///   k: `(num_heads, n_kv, d_head)`
///   v: `(num_heads, n_kv, d_head)`
///   → `(num_heads, n_q,  d_head)`
///
/// `q_pos_offset` is the absolute position of Q row 0 in the full
/// sequence. Q row `i` then attends to K rows `0..=(q_pos_offset + i)`.
/// For decode (`n_q = 1`, `q_pos_offset = n_kv − 1`) the mask is a
/// no-op (every Q row sees every K row); the function still samples
/// `π_q ∈ S_1` (trivial) and `π_kv ∈ S_{n_kv}` so the engine sees
/// row-permuted K/V uniformly.
///
/// Engine usage (F1+ — same as [`permuted_attention`]):
/// - `matmul_dynamic_batched` for `(πQ·Q + η_Q)(πKV·K + η_K)ᵀ`
/// - **in-TEE** row-wise softmax — keeps the causal mask off the GPU
/// - `matmul_dynamic_batched` for `probs · πKV·V`
pub fn permuted_attention_cached<R: RngCore, E: GpuOffloadEngine + ?Sized>(
    engine: &E,
    q: ArrayView3<'_, f32>,
    k: ArrayView3<'_, f32>,
    v: ArrayView3<'_, f32>,
    scale: f32,
    q_pos_offset: usize,
    mask: AttentionMask,
    cfg: PermAttnConfig,
    rng: &mut R,
) -> Result<Array3<f32>> {
    let (num_heads, n_q, d_head) = q.dim();
    let n_kv = k.dim().1;
    if k.dim() != (num_heads, n_kv, d_head) {
        return Err(anyhow::anyhow!(
            "permuted_attention_cached: K shape {:?} expected {:?}",
            k.dim(),
            (num_heads, n_kv, d_head)
        ));
    }
    if v.dim() != (num_heads, n_kv, d_head) {
        return Err(anyhow::anyhow!(
            "permuted_attention_cached: V shape {:?} expected {:?}",
            v.dim(),
            (num_heads, n_kv, d_head)
        ));
    }
    if n_q > n_kv {
        return Err(anyhow::anyhow!(
            "permuted_attention_cached: n_q ({n_q}) cannot exceed n_kv ({n_kv})"
        ));
    }
    if q_pos_offset + n_q > n_kv {
        return Err(anyhow::anyhow!(
            "permuted_attention_cached: q_pos_offset ({q_pos_offset}) + n_q ({n_q}) \
             must be ≤ n_kv ({n_kv})"
        ));
    }

    // Two independent permutations for the asymmetric case.
    let perm_q = sample_permutation(n_q, rng);
    let perm_kv = sample_permutation(n_kv, rng);

    // Permute Q on its (n_q) axis, K/V on their (n_kv) axis.
    let mut q_perm = Array3::<f32>::zeros((num_heads, n_q, d_head));
    let mut k_perm = Array3::<f32>::zeros((num_heads, n_kv, d_head));
    let mut v_perm = Array3::<f32>::zeros((num_heads, n_kv, d_head));
    for h in 0..num_heads {
        let qh = q.index_axis(Axis(0), h);
        let kh = k.index_axis(Axis(0), h);
        let vh = v.index_axis(Axis(0), h);
        for (i, &src) in perm_q.iter().enumerate() {
            q_perm.slice_mut(s![h, i, ..]).assign(&qh.row(src));
        }
        for (i, &src) in perm_kv.iter().enumerate() {
            k_perm.slice_mut(s![h, i, ..]).assign(&kh.row(src));
            v_perm.slice_mut(s![h, i, ..]).assign(&vh.row(src));
        }
    }

    // Hidden-No-More-class additive noise on Q, K only.
    if cfg.noise_sigma > 0.0 {
        add_gaussian_3d_inplace(q_perm.view_mut(), cfg.noise_sigma, rng);
        add_gaussian_3d_inplace(k_perm.view_mut(), cfg.noise_sigma, rng);
    }

    // Feature-axis rotation cover. Per head: rotate Q,K by a shared
    // `O_qk` (cancels in `Q·Kᵀ`, so scores — and the compute below — are
    // unchanged) and V by an independent `O_v` (undone by `O_vᵀ` on the
    // recovered output). Hides operand coordinates from the engine.
    let o_v_heads: Vec<Array2<f32>> = if cfg.feature_rotation {
        let mut ovs = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let o_qk = sample_orthogonal(d_head, rng);
            let o_v = sample_orthogonal(d_head, rng);
            let qh = q_perm.index_axis(Axis(0), h).dot(&o_qk);
            q_perm.index_axis_mut(Axis(0), h).assign(&qh);
            let kh = k_perm.index_axis(Axis(0), h).dot(&o_qk);
            k_perm.index_axis_mut(Axis(0), h).assign(&kh);
            let vh = v_perm.index_axis(Axis(0), h).dot(&o_v);
            v_perm.index_axis_mut(Axis(0), h).assign(&vh);
            ovs.push(o_v);
        }
        ovs
    } else {
        Vec::new()
    };

    // Branch on the protocol choice early so the Phase 1b path goes
    // through the **single** `fused_attention_batched` trait call
    // (today's default impl composes the same 3 dispatches, but a
    // future wgpu kernel override collapses them into one device-side
    // flow with no intermediate `(B, n_q, n_kv)` materialisation).
    // The legacy path keeps the explicit matmul + TEE-softmax + matmul
    // chain because moving softmax to GPU there would violate F1+ at
    // prefill.
    let mask_is_noop = match mask {
        AttentionMask::None => true,
        AttentionMask::Causal => n_q == 1 && q_pos_offset + n_q == n_kv,
    };
    let out_perm = if cfg.decode_softmax_on_gpu && mask_is_noop {
        // **Phase 1b** — full Amulet via `fused_attention_batched`.
        //
        // **A2 mask-elision**: at decode (n_q = 1 ∧ q_pos_offset =
        // n_kv − 1) the causal mask is identically zero — every Q
        // row attends to every K row.  Pass `None` so the engine
        // skips the mask upload and the `+ mask` kernel dispatch
        // entirely.  Encoder-style callers (`AttentionMask::None`)
        // also land here.
        engine.run_fused_attention(FusedAttentionBatch {
            q: q_perm.view(),
            k: k_perm.view(),
            v: v_perm.view(),
            scale,
            mask: None,
        })?
        // No row-sum integrity probe on this path — the fused output
        // is the *post-multiplied* tensor, not the softmax tensor, so
        // the `Σ probs[i,j] = 1` invariant is no longer directly
        // observable.  The c5_perm_attn AloePri gate is the right
        // place to gate Byzantine GPU behaviour for fused attention.
    } else {
        // F1+ legacy path: explicit matmul + TEE-softmax + matmul.
        let kt_perm_view = k_perm.view().permuted_axes([0, 2, 1]);
        let mut scores = engine.run_runtime_matmul(RuntimeMatmulBatch {
            lhs: q_perm.view(),
            rhs: kt_perm_view,
        })?;
        scores.mapv_inplace(|x| x * scale);

        // Apply asymmetric permuted causal mask in-TEE. Shape
        // `(n_q, n_kv)`. Original: `mask_orig[i, j] = -C if j >
        // q_pos_offset + i else 0`. Permuted: `mask_perm[i, j] =
        // mask_orig[perm_q[i], perm_kv[j]]`.
        if let AttentionMask::Causal = mask {
            let neg = -cfg.causal_mask_neg;
            let mut mask_mat = Array2::<f32>::zeros((n_q, n_kv));
            for i in 0..n_q {
                let q_abs = q_pos_offset + perm_q[i];
                for j in 0..n_kv {
                    if perm_kv[j] > q_abs {
                        mask_mat[(i, j)] = neg;
                    }
                }
            }
            for h in 0..num_heads {
                for i in 0..n_q {
                    for j in 0..n_kv {
                        scores[(h, i, j)] += mask_mat[(i, j)];
                    }
                }
            }
        }

        // F1+ in-TEE softmax.  Same reasoning as `permuted_attention`.
        let mut probs = Array3::<f32>::zeros((num_heads, n_q, n_kv));
        for h in 0..num_heads {
            let scores_h = scores.index_axis(Axis(0), h);
            let probs_h = softmax_rowwise(scores_h);
            probs.index_axis_mut(Axis(0), h).assign(&probs_h);
        }
        engine.run_runtime_matmul(RuntimeMatmulBatch {
            lhs: probs.view(),
            rhs: v_perm.view(),
        })?
    };

    // Undo the V feature-rotation: `(softmax·(V·O_v))·O_vᵀ = softmax·V`.
    // Feature-axis (`O_vᵀ`) and row-axis (`π_q⁻¹`) recoveries commute.
    let mut out_perm = out_perm;
    if cfg.feature_rotation {
        for h in 0..num_heads {
            let oh = out_perm.index_axis(Axis(0), h).dot(&o_v_heads[h].t());
            out_perm.index_axis_mut(Axis(0), h).assign(&oh);
        }
    }

    // Recovery via π_q⁻¹ on the Q axis.
    let mut out = Array3::<f32>::zeros((num_heads, n_q, d_head));
    for h in 0..num_heads {
        for (i, &src) in perm_q.iter().enumerate() {
            out.slice_mut(s![h, src, ..])
                .assign(&out_perm.slice(s![h, i, ..]));
        }
    }

    Ok(out)
}

/// Sample a fresh row permutation π ∈ S_n.
pub(crate) fn sample_permutation<R: RngCore>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);
    perm
}

/// Sample a fresh `d × d` orthogonal matrix `O` (modified Gram-Schmidt on
/// a Gaussian matrix → orthonormal rows, so `O·Oᵀ = I`). Used by the
/// feature-axis rotation cover: an operand row `x` is rotated as `x·O`
/// and recovered as `(x·O)·Oᵀ = x`. Rotating Q and K by the same `O_qk`
/// leaves `Q·Kᵀ` invariant; rotating V by an independent `O_v` is undone
/// by `O_vᵀ` on the output.
pub(crate) fn sample_orthogonal<R: RngCore>(d: usize, rng: &mut R) -> Array2<f32> {
    let normal = StandardNormal;
    let mut m = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(rng));
    for i in 0..d {
        // Orthogonalise row i against rows 0..i.
        for j in 0..i {
            let dot: f32 = (0..d).map(|k| m[(i, k)] * m[(j, k)]).sum();
            for k in 0..d {
                m[(i, k)] -= dot * m[(j, k)];
            }
        }
        // Normalise row i.
        let norm: f32 = (0..d).map(|k| m[(i, k)] * m[(i, k)]).sum::<f32>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
        for k in 0..d {
            m[(i, k)] *= inv;
        }
    }
    m
}

/// Add `N(0, σ²·I)` noise to a 3D view element-wise.
///
/// On small tensors stays single-threaded scalar (rayon work-stealing
/// has a ~100 μs fixed cost per call that doesn't amortise below
/// ~32 K elements). On larger tensors — the Qwen3-class cached K
/// shape `(num_heads, n_kv, d_head)` at decode time, ~4.2 M elements
/// per layer — splits along the heads axis with **independent
/// ChaCha20 streams seeded from the parent RNG**, parallelising
/// across cores via rayon.
///
/// The seed-derivation step advances `rng` by `n_heads · 32 bytes`,
/// so the parent stream state after the call is the same number of
/// bytes consumed as the previous serial implementation would have
/// consumed. Per-element noise distribution is unchanged (independent
/// `N(0, σ²)` everywhere); only the cross-head correlation structure
/// differs — which is invariant for HNM-class security arguments
/// since the attack model is per-element / per-row, not cross-head.
///
/// Phase 4 diagnosis: the unoptimised scalar version dominated TPOT
/// in the permuted_cached path at long n_kv (2.4 s/decode-step at
/// n_kv = 2 048 on Qwen3-1.7B). This rewrite is the M1.10 follow-up
/// (`docs/plans/m1-10-phase4-findings.md` §4.3).
fn add_gaussian_3d_inplace<R: RngCore>(
    mut m: ndarray::ArrayViewMut3<'_, f32>,
    sigma: f32,
    rng: &mut R,
) {
    if sigma == 0.0 {
        return;
    }
    // Below ~32 K elements, scalar serial wins (no rayon scheduling
    // tax). Above that, parallelise across the heads axis.
    const PARALLEL_THRESHOLD: usize = 32_768;
    let total = m.shape().iter().product::<usize>();
    if total < PARALLEL_THRESHOLD {
        let normal = StandardNormal;
        for v in m.iter_mut() {
            let z: f32 = normal.sample(rng);
            *v += sigma * z;
        }
        return;
    }

    // Pre-derive one independent ChaCha20 seed per head from the
    // parent RNG. Advances the parent RNG by `n_heads · 32 bytes`.
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let n_heads = m.shape()[0];
    let seeds: Vec<[u8; 32]> = (0..n_heads)
        .map(|_| {
            let mut s = [0u8; 32];
            rng.fill_bytes(&mut s);
            s
        })
        .collect();

    // Rayon-parallel across heads. Each worker creates its own RNG
    // from its pre-derived seed; no shared mutable state.
    use ndarray::parallel::prelude::*;
    m.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(h, mut head_slice)| {
            let mut local_rng = ChaCha20Rng::from_seed(seeds[h]);
            let normal = StandardNormal;
            for v in head_slice.iter_mut() {
                let z: f32 = normal.sample(&mut local_rng);
                *v += sigma * z;
            }
        });
}

/// Row-wise numerically stable softmax. `(n, m) → (n, m)`.
///
/// Used by `permuted_attention` to keep softmax in-TEE under the F1+
/// causal-mask-leak resolution — sending softmax to GPU would let the
/// engine recover the row permutation π from the input score tensor's
/// causal-mask pattern. See `docs/plans/m1-10-security-review.md`.
pub(crate) fn softmax_rowwise(scores: ArrayView2<'_, f32>) -> Array2<f32> {
    let (n, m) = scores.dim();
    let mut out = Array2::<f32>::zeros((n, m));
    for i in 0..n {
        let row = scores.row(i);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (j, v) in row.iter().enumerate() {
            let e = (*v - max).exp();
            out[(i, j)] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..m {
            out[(i, j)] *= inv;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::ReferenceCpuEngine;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// Plain reference: `softmax(Q·Kᵀ/√d)·V` per head.
    fn plain_multi_head_attention(
        q: ArrayView3<'_, f32>,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
        scale: f32,
    ) -> Array3<f32> {
        let (h, n, d) = q.dim();
        let mut out = Array3::<f32>::zeros((h, n, d));
        for i in 0..h {
            let qh = q.index_axis(Axis(0), i);
            let kh = k.index_axis(Axis(0), i);
            let vh = v.index_axis(Axis(0), i);
            let mut scores = qh.dot(&kh.t());
            scores.mapv_inplace(|x| x * scale);
            let probs = softmax_rowwise(scores.view());
            out.index_axis_mut(Axis(0), i).assign(&probs.dot(&vh));
        }
        out
    }

    fn random_q3(h: usize, n: usize, d: usize, rng: &mut ChaCha20Rng) -> Array3<f32> {
        use rand::Rng;
        Array3::from_shape_fn((h, n, d), |_| rng.random::<f32>() * 2.0 - 1.0)
    }

    #[test]
    fn permuted_attention_parity_sigma_zero() {
        let h = 4;
        let n = 16;
        let d = 32;
        let scale = 1.0 / (d as f32).sqrt();
        let mut rng = ChaCha20Rng::seed_from_u64(0xABBA);
        let q = random_q3(h, n, d, &mut rng);
        let k = random_q3(h, n, d, &mut rng);
        let v = random_q3(h, n, d, &mut rng);
        let engine = ReferenceCpuEngine::new();

        let plain = plain_multi_head_attention(q.view(), k.view(), v.view(), scale);
        let out = permuted_attention(
            &engine,
            q.view(),
            k.view(),
            v.view(),
            scale,
            AttentionMask::None,
            PermAttnConfig::DISABLED_NOISE,
            &mut rng,
        )
        .unwrap();

        let drift = (&plain - &out)
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift < 1e-5,
            "σ=0 multi-head equivariance must be bit-exact: drift={drift}",
        );
    }

    #[test]
    fn permuted_attention_drift_bounded_at_hnm_sigma() {
        let h = 8;
        let n = 32;
        let d = 64;
        let scale = 1.0 / (d as f32).sqrt();
        let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE);
        let q = random_q3(h, n, d, &mut rng);
        let k = random_q3(h, n, d, &mut rng);
        let v = random_q3(h, n, d, &mut rng);
        let engine = ReferenceCpuEngine::new();

        let plain = plain_multi_head_attention(q.view(), k.view(), v.view(), scale);
        let out = permuted_attention(
            &engine,
            q.view(),
            k.view(),
            v.view(),
            scale,
            AttentionMask::None,
            PermAttnConfig::HIDDEN_NO_MORE,
            &mut rng,
        )
        .unwrap();

        let drift = (&plain - &out)
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift < 5e-2,
            "σ=0.01 multi-head drift should stay below 5e-2 elementwise: drift={drift}",
        );
    }

    /// Plain reference for the asymmetric (cached-KV) shape:
    /// `softmax(Q · Kᵀ / √d + M_orig) · V` per head, where the
    /// original causal mask blocks K positions beyond
    /// `q_pos_offset + q_row`. Used to validate
    /// `permuted_attention_cached` against an unpermuted baseline.
    fn plain_multi_head_attention_cached(
        q: ArrayView3<'_, f32>,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
        scale: f32,
        q_pos_offset: usize,
        causal: bool,
    ) -> Array3<f32> {
        let (h, n_q, d) = q.dim();
        let n_kv = k.dim().1;
        let mut out = Array3::<f32>::zeros((h, n_q, d));
        for hi in 0..h {
            let qh = q.index_axis(Axis(0), hi);
            let kh = k.index_axis(Axis(0), hi);
            let vh = v.index_axis(Axis(0), hi);
            let mut scores = qh.dot(&kh.t());
            scores.mapv_inplace(|x| x * scale);
            if causal {
                for i in 0..n_q {
                    let q_abs = q_pos_offset + i;
                    for j in 0..n_kv {
                        if j > q_abs {
                            scores[(i, j)] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let probs = softmax_rowwise(scores.view());
            out.index_axis_mut(Axis(0), hi).assign(&probs.dot(&vh));
        }
        out
    }

    #[test]
    fn permuted_attention_cached_matches_full_prefill_at_sigma_zero() {
        // n_q = n_kv, q_pos_offset = 0: should match the symmetric
        // permuted_attention's input regime (full prefill). σ=0 forces
        // bit-exact equivariance.
        let h = 4;
        let n = 16;
        let d = 32;
        let scale = 1.0 / (d as f32).sqrt();
        let mut rng = ChaCha20Rng::seed_from_u64(0xC4C4_E0E0);
        let q = random_q3(h, n, d, &mut rng);
        let k = random_q3(h, n, d, &mut rng);
        let v = random_q3(h, n, d, &mut rng);
        let engine = ReferenceCpuEngine::new();

        let plain = plain_multi_head_attention_cached(q.view(), k.view(), v.view(), scale, 0, true);
        let out = permuted_attention_cached(
            &engine,
            q.view(),
            k.view(),
            v.view(),
            scale,
            0,
            AttentionMask::Causal,
            PermAttnConfig::DISABLED_NOISE,
            &mut rng,
        )
        .unwrap();

        let drift = (&plain - &out)
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift < 1e-5,
            "σ=0 cached prefill equivariance must be bit-exact: drift={drift}",
        );
    }

    #[test]
    fn permuted_attention_cached_decode_shape_matches_plain() {
        // n_q = 1, q_pos_offset = n_kv - 1: the typical decode-step
        // shape where one new query attends to the full cache. The
        // causal mask is a no-op (every kv position is allowed).
        let h = 4;
        let d = 32;
        let scale = 1.0 / (d as f32).sqrt();

        for n_kv in [8usize, 64, 256] {
            let mut rng = ChaCha20Rng::seed_from_u64(0xDEC0_DE00 ^ n_kv as u64);
            let q = random_q3(h, 1, d, &mut rng);
            let k = random_q3(h, n_kv, d, &mut rng);
            let v = random_q3(h, n_kv, d, &mut rng);
            let engine = ReferenceCpuEngine::new();

            let q_pos_offset = n_kv - 1;
            let plain = plain_multi_head_attention_cached(
                q.view(),
                k.view(),
                v.view(),
                scale,
                q_pos_offset,
                true,
            );
            let out = permuted_attention_cached(
                &engine,
                q.view(),
                k.view(),
                v.view(),
                scale,
                q_pos_offset,
                AttentionMask::Causal,
                PermAttnConfig::DISABLED_NOISE,
                &mut rng,
            )
            .unwrap();

            let drift = (&plain - &out)
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            assert!(
                drift < 1e-5,
                "decode-shape (n_q=1, n_kv={n_kv}) must be bit-exact at σ=0: drift={drift}",
            );
        }
    }

    #[test]
    fn feature_rotation_sigma_zero_matches_plain() {
        // The O_qk/O_v feature-axis rotation cover must be exactly
        // invertible: at σ=0 the recovered output is bit-exact (f32
        // floor) to plaintext attention — O_qk cancels in Q·Kᵀ, O_v is
        // undone by O_vᵀ. Production head_dim d=128 plus a small d.
        for d in [32usize, 128] {
            let h = 4;
            let scale = 1.0 / (d as f32).sqrt();
            let cfg = PermAttnConfig {
                feature_rotation: true,
                ..PermAttnConfig::DISABLED_NOISE
            };
            for n_kv in [8usize, 64, 256] {
                let mut rng = ChaCha20Rng::seed_from_u64(0xF0A7_0000 ^ ((d as u64) << 16) ^ n_kv as u64);
                let q = random_q3(h, 1, d, &mut rng);
                let k = random_q3(h, n_kv, d, &mut rng);
                let v = random_q3(h, n_kv, d, &mut rng);
                let engine = ReferenceCpuEngine::new();
                let q_pos_offset = n_kv - 1;
                let plain = plain_multi_head_attention_cached(
                    q.view(), k.view(), v.view(), scale, q_pos_offset, true,
                );
                let out = permuted_attention_cached(
                    &engine, q.view(), k.view(), v.view(), scale, q_pos_offset,
                    AttentionMask::Causal, cfg, &mut rng,
                )
                .unwrap();
                let drift = (&plain - &out).iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                assert!(
                    drift < 1e-3,
                    "feature-rotation must be invertible at σ=0 (d={d}, n_kv={n_kv}): drift={drift}",
                );
            }
        }
    }

    #[test]
    fn feature_rotation_preserves_scores() {
        // O_qk applied to Q,K must leave Q·Kᵀ (hence softmax, hence the
        // whole output) invariant — the rotation is invisible to the
        // computation, visible only to the engine's operand view. At
        // σ=0, feature_rotation=ON must match feature_rotation=OFF.
        let h = 4;
        let d = 128;
        let n_kv = 64;
        let scale = 1.0 / (d as f32).sqrt();
        let q = random_q3(h, 1, d, &mut ChaCha20Rng::seed_from_u64(1));
        let k = random_q3(h, n_kv, d, &mut ChaCha20Rng::seed_from_u64(2));
        let v = random_q3(h, n_kv, d, &mut ChaCha20Rng::seed_from_u64(3));
        let engine = ReferenceCpuEngine::new();
        let q_pos_offset = n_kv - 1;
        let run = |rot: bool| {
            let cfg = PermAttnConfig { feature_rotation: rot, ..PermAttnConfig::DISABLED_NOISE };
            permuted_attention_cached(
                &engine, q.view(), k.view(), v.view(), scale, q_pos_offset,
                AttentionMask::Causal, cfg, &mut ChaCha20Rng::seed_from_u64(42),
            )
            .unwrap()
        };
        let drift = (&run(false) - &run(true)).iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(drift < 1e-3, "feature_rotation must not change the output: drift={drift}");
    }

    #[test]
    fn permuted_attention_cached_partial_prefill_at_sigma_zero() {
        // Continuation-prefill shape: n_q small, q_pos_offset > 0.
        // Realistic case for the second turn of a chat where the
        // cache already holds prior turns.
        let h = 4;
        let d = 32;
        let scale = 1.0 / (d as f32).sqrt();

        let n_q = 4;
        let n_kv = 16;
        let q_pos_offset = n_kv - n_q;
        let mut rng = ChaCha20Rng::seed_from_u64(0xBEEF_C0DE);
        let q = random_q3(h, n_q, d, &mut rng);
        let k = random_q3(h, n_kv, d, &mut rng);
        let v = random_q3(h, n_kv, d, &mut rng);
        let engine = ReferenceCpuEngine::new();

        let plain = plain_multi_head_attention_cached(
            q.view(),
            k.view(),
            v.view(),
            scale,
            q_pos_offset,
            true,
        );
        let out = permuted_attention_cached(
            &engine,
            q.view(),
            k.view(),
            v.view(),
            scale,
            q_pos_offset,
            AttentionMask::Causal,
            PermAttnConfig::DISABLED_NOISE,
            &mut rng,
        )
        .unwrap();

        let drift = (&plain - &out)
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift < 1e-5,
            "continuation-prefill must be bit-exact at σ=0: drift={drift}",
        );
    }

    #[test]
    fn permuted_attention_cached_drift_bounded_at_hnm_sigma() {
        // σ=0.01: same Hidden-No-More noise level as the symmetric
        // test. Decode shape, n_kv=64.
        let h = 4;
        let d = 64;
        let scale = 1.0 / (d as f32).sqrt();
        let n_kv = 64;
        let q_pos_offset = n_kv - 1;
        let mut rng = ChaCha20Rng::seed_from_u64(0xFEED_C0DE);
        let q = random_q3(h, 1, d, &mut rng);
        let k = random_q3(h, n_kv, d, &mut rng);
        let v = random_q3(h, n_kv, d, &mut rng);
        let engine = ReferenceCpuEngine::new();

        let plain = plain_multi_head_attention_cached(
            q.view(),
            k.view(),
            v.view(),
            scale,
            q_pos_offset,
            true,
        );
        let out = permuted_attention_cached(
            &engine,
            q.view(),
            k.view(),
            v.view(),
            scale,
            q_pos_offset,
            AttentionMask::Causal,
            PermAttnConfig::HIDDEN_NO_MORE,
            &mut rng,
        )
        .unwrap();

        let drift = (&plain - &out)
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        // Same tolerance as the symmetric drift test — σ=0.01 noise
        // dominates this bound.
        assert!(
            drift < 5e-2,
            "σ=0.01 decode-shape drift should stay below 5e-2: drift={drift}",
        );
    }

    #[test]
    fn permuted_attention_cached_rejects_n_q_gt_n_kv() {
        let engine = ReferenceCpuEngine::new();
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let q = Array3::<f32>::zeros((2, 8, 4));
        let k = Array3::<f32>::zeros((2, 4, 4));
        let v = Array3::<f32>::zeros((2, 4, 4));
        assert!(
            permuted_attention_cached(
                &engine,
                q.view(),
                k.view(),
                v.view(),
                1.0,
                0,
                AttentionMask::None,
                PermAttnConfig::DISABLED_NOISE,
                &mut rng,
            )
            .is_err(),
        );
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let engine = ReferenceCpuEngine::new();
        let q = Array3::<f32>::zeros((2, 4, 8));
        let k = Array3::<f32>::zeros((2, 4, 8));
        let v = Array3::<f32>::zeros((2, 4, 4)); // wrong d
        assert!(
            permuted_attention(
                &engine,
                q.view(),
                k.view(),
                v.view(),
                1.0,
                AttentionMask::None,
                PermAttnConfig::DISABLED_NOISE,
                &mut rng,
            )
            .is_err()
        );
    }
}
