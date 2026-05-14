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
/// permutation-shielded protocol.
///
/// `q`, `k`, `v` shape: `(num_heads, n, d_head)`. Result shape:
/// `(num_heads, n, d_head)`.
///
/// `scale` is the attention scale (typically `1 / √d_head`).
///
/// The fresh per-batch row permutation `π_b ∈ S_n` is sampled once and
/// shared across all heads within this block. The Hidden No More
/// per-head decoupling can be added in a later phase by sampling one π
/// per head.
pub fn permuted_attention<R: RngCore>(
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

    let mut out = Array3::<f32>::zeros((num_heads, n, d_head));
    for h in 0..num_heads {
        let qh = q.index_axis(Axis(0), h);
        let kh = k.index_axis(Axis(0), h);
        let vh = v.index_axis(Axis(0), h);

        // Permute Q, K, V on the row (token) axis.
        let mut qp = permute_rows(qh, &perm);
        let mut kp = permute_rows(kh, &perm);
        let vp = permute_rows(vh, &perm);

        // Optionally inject N(0, σ²·I) on Q and K. V is left untouched.
        if cfg.noise_sigma > 0.0 {
            add_gaussian_inplace(qp.view_mut(), cfg.noise_sigma, rng);
            add_gaussian_inplace(kp.view_mut(), cfg.noise_sigma, rng);
        }

        // GPU work (Phase 3): currently TEE-side.
        let mut scores = qp.dot(&kp.t());
        scores.mapv_inplace(|x| x * scale);
        let probs = softmax_rowwise(scores.view());
        let op = probs.dot(&vp);

        // Recover via πᵀ: out[perm[i]] = op[i].
        for (i, &src) in perm.iter().enumerate() {
            out.slice_mut(s![h, src, ..]).assign(&op.row(i));
        }
    }

    Ok(out)
}

/// Sample a fresh row permutation π ∈ S_n. Public so adjacent modules
/// (and future per-head variants) can build on it.
pub(crate) fn sample_permutation<R: RngCore>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);
    perm
}

/// Permute rows of `m`: `out[i] = m[perm[i]]`.
pub(crate) fn permute_rows(m: ArrayView2<'_, f32>, perm: &[usize]) -> Array2<f32> {
    let (n, d) = m.dim();
    debug_assert_eq!(perm.len(), n);
    let mut out = Array2::<f32>::zeros((n, d));
    for (i, &src) in perm.iter().enumerate() {
        out.row_mut(i).assign(&m.row(src));
    }
    out
}

/// Add `N(0, σ²·I)` noise to `m` element-wise.
fn add_gaussian_inplace<R: RngCore>(
    mut m: ndarray::ArrayViewMut2<'_, f32>,
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

/// Row-wise numerically stable softmax. `(n, m) → (n, m)`.
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

        let plain = plain_multi_head_attention(q.view(), k.view(), v.view(), scale);
        let out = permuted_attention(
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

        let plain = plain_multi_head_attention(q.view(), k.view(), v.view(), scale);
        let out = permuted_attention(
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
        let q = Array3::<f32>::zeros((2, 4, 8));
        let k = Array3::<f32>::zeros((2, 4, 8));
        let v = Array3::<f32>::zeros((2, 4, 4)); // wrong d
        assert!(
            permuted_attention(
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
