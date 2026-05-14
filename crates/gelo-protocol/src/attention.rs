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
use ndarray::{Array3, ArrayView3, Axis, s};
#[cfg(test)]
use ndarray::{Array2, ArrayView2};
use rand::{RngCore, seq::SliceRandom};
use rand_distr::{Distribution, StandardNormal};

use crate::substrate::GpuOffloadEngine;

/// Configuration for the permutation-shielded attention protocol.
#[derive(Debug, Clone, Copy)]
pub struct PermAttnConfig {
    /// Per-element standard deviation of the Gaussian noise added to
    /// Q and K. `0.0` disables noise (pure permutation equivariance).
    /// Hidden No More reports σ = 0.01 as the threshold where their
    /// recovery attack drops to ROUGE < 0.1.
    pub noise_sigma: f32,
}

impl PermAttnConfig {
    /// Pure permutation, no noise. Bit-exact equivariance.
    pub const DISABLED_NOISE: Self = Self { noise_sigma: 0.0 };

    /// Hidden No More mitigation level. Default for production.
    pub const HIDDEN_NO_MORE: Self = Self { noise_sigma: 0.01 };
}

impl Default for PermAttnConfig {
    fn default() -> Self {
        Self::DISABLED_NOISE
    }
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
/// The fresh per-batch row permutation `π_b ∈ S_n` is sampled once and
/// shared across all heads within this block. The Hidden No More
/// per-head decoupling can be added later by sampling one π per head.
///
/// Engine usage:
/// - `matmul_dynamic_batched` for `(πQ + η_Q)(πK + η_K)ᵀ` (batched over heads)
/// - `softmax_batched` on the last axis of the score tensor
/// - `matmul_dynamic_batched` for `probs · πV` (batched over heads)
pub fn permuted_attention<R: RngCore, E: GpuOffloadEngine + ?Sized>(
    engine: &E,
    q: ArrayView3<'_, f32>,
    k: ArrayView3<'_, f32>,
    v: ArrayView3<'_, f32>,
    scale: f32,
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

    // Build Kᵀ over the last two axes — engine matmul wants (B, K, N).
    let mut kt_perm = Array3::<f32>::zeros((num_heads, d_head, n));
    for h in 0..num_heads {
        for i in 0..n {
            for j in 0..d_head {
                kt_perm[(h, j, i)] = k_perm[(h, i, j)];
            }
        }
    }

    // GPU step 1: scores = (πQ + η_Q) · (πK + η_K)ᵀ shape (num_heads, n, n).
    let mut scores = engine.matmul_dynamic_batched(q_perm.view(), kt_perm.view())?;
    scores.mapv_inplace(|x| x * scale);

    // GPU step 2: softmax along last axis.
    let probs = engine.softmax_batched(scores.view())?;

    // GPU step 3: out_perm = probs · πV shape (num_heads, n, d_head).
    let out_perm = engine.matmul_dynamic_batched(probs.view(), v_perm.view())?;

    // TEE recovery via πᵀ: out[h, perm[i], :] = out_perm[h, i, :].
    let mut out = Array3::<f32>::zeros((num_heads, n, d_head));
    for h in 0..num_heads {
        for (i, &src) in perm.iter().enumerate() {
            out.slice_mut(s![h, src, ..]).assign(&out_perm.slice(s![h, i, ..]));
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

/// Add `N(0, σ²·I)` noise to a 3D view element-wise.
fn add_gaussian_3d_inplace<R: RngCore>(
    mut m: ndarray::ArrayViewMut3<'_, f32>,
    sigma: f32,
    rng: &mut R,
) {
    if sigma == 0.0 {
        return;
    }
    let normal = StandardNormal;
    for v in m.iter_mut() {
        let z: f32 = normal.sample(rng);
        *v += sigma * z;
    }
}

/// Row-wise numerically stable softmax. `(n, m) → (n, m)`. Test-only —
/// production softmax goes through `GpuOffloadEngine::softmax_batched`.
#[cfg(test)]
fn softmax_rowwise(scores: ArrayView2<'_, f32>) -> Array2<f32> {
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
    use crate::sim::RayonCpuEngine;
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
        let engine = RayonCpuEngine::new();

        let plain = plain_multi_head_attention(q.view(), k.view(), v.view(), scale);
        let out = permuted_attention(
            &engine,
            q.view(),
            k.view(),
            v.view(),
            scale,
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
        let engine = RayonCpuEngine::new();

        let plain = plain_multi_head_attention(q.view(), k.view(), v.view(), scale);
        let out = permuted_attention(
            &engine,
            q.view(),
            k.view(),
            v.view(),
            scale,
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

    #[test]
    fn shape_mismatch_returns_error() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let engine = RayonCpuEngine::new();
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
                PermAttnConfig::DISABLED_NOISE,
                &mut rng,
            )
            .is_err()
        );
    }
}
