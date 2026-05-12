//! Planar-Laplace query perturbation (RemoteRAG, [arXiv 2412.12775](https://arxiv.org/abs/2412.12775)).
//!
//! ```text
//!   r ~ Gamma(shape=n, scale=1/ε)
//!   z ~ N(0, I_n);  v = z / ||z||         (uniform on the n-sphere)
//!   q_noisy = q + r · v
//! ```
//!
//! **`n` is the embedding dimension**, **NOT** a cluster count. This is the
//! [single most common misreading](https://arxiv.org/abs/2412.12775) of the
//! `(n, ε)`-DistanceDP notion. The corresponding privacy budget is
//! distance-relative; `ε ∈ [10·n, 50·n]` is the paper's recommended range
//! for dims in `[384, 1536]`, i.e. `ε ≈ 4k–80k`. That is **not** comparable
//! to the `ε ≈ 1–10` of standard DP.
//!
//! This module is internal to the `remote-rag` crate. It lives here, not in
//! `dp-forward`, because the planar-Laplace mechanism is from the RemoteRAG
//! paper, not from Yue et al. The two papers solve different problems.

use rand::RngCore;
use rand_distr::{Distribution, Gamma, StandardNormal};

/// Privacy parameters for the `(n, ε)`-DistanceDP planar-Laplace mechanism.
#[derive(Debug, Clone, Copy)]
pub struct PlanarLaplaceConfig {
    /// Privacy budget `ε > 0`. **Distance-relative scale.** Recommended
    /// range: `ε ∈ [10·n, 50·n]` for moderate retrieval utility on
    /// dim ∈ [384, 1536]. Smaller `ε` = larger radius = tighter privacy.
    pub epsilon: f64,
    /// Embedding dimension `n`. Must match the dimension of the query
    /// vectors fed to [`perturb`].
    pub n: usize,
}

impl PlanarLaplaceConfig {
    /// Construct, panicking on invalid budgets to surface misuse early.
    pub fn new(epsilon: f64, n: usize) -> Self {
        assert!(epsilon > 0.0, "epsilon must be > 0 (got {epsilon})");
        assert!(n > 0, "n must be > 0");
        Self { epsilon, n }
    }
}

/// Sample the radial magnitude `r ~ Gamma(shape=n, scale=1/ε)`. Returned as
/// `f64` for downstream precision; the caller casts back to `f32` once it
/// projects onto the direction.
pub fn sample_radius_gamma<R: RngCore>(n: usize, epsilon: f64, rng: &mut R) -> f64 {
    // rand_distr's Gamma takes (shape, scale).
    let gamma = Gamma::new(n as f64, 1.0 / epsilon).expect("shape > 0, scale > 0");
    gamma.sample(rng)
}

/// Sample a unit vector uniformly on `S^{n−1}` via Gaussian normalisation
/// (Marsaglia/Box-Muller direction). Output length is `n`.
pub fn sample_direction<R: RngCore>(n: usize, rng: &mut R) -> Vec<f32> {
    let normal = StandardNormal;
    let mut v: Vec<f32> = (0..n)
        .map(|_| <StandardNormal as Distribution<f32>>::sample(&normal, rng))
        .collect();
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    // Degenerate `norm == 0` is statistically impossible for `n ≥ 1` Gaussian
    // draws; guard anyway so a hostile RNG doesn't NaN-poison downstream math.
    if norm > 0.0 {
        let scale = (1.0 / norm) as f32;
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
    v
}

/// Apply RemoteRAG Stage-1 perturbation in place: `q ← q + r·v`. Length of
/// `q` must equal `cfg.n`; panics otherwise (programmer error).
pub fn perturb<R: RngCore>(q: &mut [f32], cfg: &PlanarLaplaceConfig, rng: &mut R) {
    assert_eq!(
        q.len(),
        cfg.n,
        "query length ({}) must equal cfg.n ({})",
        q.len(),
        cfg.n
    );
    let r = sample_radius_gamma(cfg.n, cfg.epsilon, rng);
    let v = sample_direction(cfg.n, rng);
    let r_f32 = r as f32;
    for (qi, vi) in q.iter_mut().zip(v.iter()) {
        *qi += r_f32 * *vi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn config_panics_on_nonpositive_epsilon() {
        let result = std::panic::catch_unwind(|| PlanarLaplaceConfig::new(0.0, 32));
        assert!(result.is_err());
    }

    #[test]
    fn direction_is_unit_norm() {
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        for _ in 0..50 {
            let v = sample_direction(64, &mut rng);
            let norm = v
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "direction norm {norm} should equal 1"
            );
        }
    }

    #[test]
    fn radius_gamma_mean_matches_n_over_epsilon() {
        // E[Gamma(n, 1/ε)] = n/ε. With N samples, sample mean concentrates.
        let mut rng = ChaCha20Rng::from_seed([2u8; 32]);
        let n = 64;
        let epsilon = 16.0;
        let n_samples = 20_000;
        let sum: f64 = (0..n_samples)
            .map(|_| sample_radius_gamma(n, epsilon, &mut rng))
            .sum();
        let mean = sum / n_samples as f64;
        let expected = n as f64 / epsilon; // = 4.0
        assert!(
            (mean - expected).abs() < 0.1,
            "empirical mean {mean} ≈ n/ε = {expected}"
        );
    }

    #[test]
    fn radius_gamma_variance_matches_n_over_epsilon_squared() {
        // Var[Gamma(n, 1/ε)] = n/ε².
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let n = 128;
        let epsilon = 8.0;
        let n_samples = 20_000;
        let samples: Vec<f64> = (0..n_samples)
            .map(|_| sample_radius_gamma(n, epsilon, &mut rng))
            .collect();
        let mean = samples.iter().sum::<f64>() / n_samples as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
        let expected_var = n as f64 / (epsilon * epsilon);
        assert!(
            (var / expected_var - 1.0).abs() < 0.1,
            "empirical var {var} ≈ n/ε² = {expected_var}"
        );
    }

    #[test]
    fn perturb_changes_query_by_radius_magnitude() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let n = 32;
        let cfg = PlanarLaplaceConfig::new(8.0, n);
        let mut q = vec![0.1_f32; n];
        let q_clean = q.clone();
        perturb(&mut q, &cfg, &mut rng);
        // ||q' - q|| should equal r, which is positive a.s.
        let diff: f64 = q
            .iter()
            .zip(q_clean.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(diff > 0.0, "perturbation must move the query");
    }

    #[test]
    fn perturb_panics_on_dimension_mismatch() {
        let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
        let cfg = PlanarLaplaceConfig::new(8.0, 32);
        let mut q = vec![0.0_f32; 16];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            perturb(&mut q, &cfg, &mut rng);
        }));
        assert!(result.is_err());
    }

    /// Empirical check that perturbation magnitudes match the planar-Laplace
    /// radius distribution: `||q_noisy - q||` should empirically look like
    /// `Gamma(n, 1/ε)`. Use mean as a coarse moment check.
    #[test]
    fn perturbation_magnitude_matches_gamma() {
        let mut rng = ChaCha20Rng::from_seed([6u8; 32]);
        let n = 64;
        let epsilon = 32.0;
        let cfg = PlanarLaplaceConfig::new(epsilon, n);
        let n_samples = 10_000;
        let mut magnitudes = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut q = vec![0.0_f32; n];
            perturb(&mut q, &cfg, &mut rng);
            let mag: f64 = q.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            magnitudes.push(mag);
        }
        let mean = magnitudes.iter().sum::<f64>() / n_samples as f64;
        let expected = n as f64 / epsilon;
        assert!(
            (mean - expected).abs() < 0.1,
            "empirical perturbation-magnitude mean {mean} ≈ n/ε = {expected}"
        );
    }
}
