//! [`DpForwardConfig`] — privacy budget and clipping parameter, plus the
//! `(ε, δ, C, σ)` digest that callers fold into their attested identity.

use sha2::{Digest, Sha256};

use crate::amgm;

/// Privacy parameters for the aMGM mechanism. The `sigma` field is
/// computed once via [`Self::calibrate`] and memoised so the bisection
/// runs once per config, not once per `embed()` call.
#[derive(Debug, Clone, Copy)]
pub struct DpForwardConfig {
    /// Privacy budget `ε > 0`. Smaller = tighter privacy = more noise.
    pub epsilon: f64,
    /// Privacy budget `δ ∈ (0, 1)`. Typically `1e-5` for moderate corpora.
    pub delta: f64,
    /// L2-clipping bound `C`. Pre-clip sensitivity becomes `Δ₂ = 2C`.
    /// Default 1.0 matches the DP-Forward reference for unit-norm embeddings.
    pub clip_c: f32,
    /// Noise scale (standard deviation) calibrated by [`Self::calibrate`].
    pub sigma: f64,
}

impl DpForwardConfig {
    /// Construct a config and run Balle–Wang calibration to compute σ.
    pub fn calibrate(epsilon: f64, delta: f64, clip_c: f32) -> Self {
        let sigma = amgm::calibrate_sigma(epsilon, delta, (2.0 * clip_c) as f64);
        Self {
            epsilon,
            delta,
            clip_c,
            sigma,
        }
    }

    /// 32-byte stable digest over canonical bytes — folded into
    /// `Embedder::model_identity` so a SEV-SNP attestation report binds to
    /// the exact DP parameters in effect.
    ///
    /// The digest commits to `(epsilon, delta, clip_c, sigma)`, which means a
    /// verifier who pins `expected_model_id` can detect both *parameter
    /// substitution* (different ε/δ) and *calibration substitution*
    /// (matching ε/δ but a manipulated σ).
    pub fn config_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"dp-forward.v1");
        hasher.update(self.epsilon.to_le_bytes());
        hasher.update(self.delta.to_le_bytes());
        hasher.update(self.clip_c.to_le_bytes());
        hasher.update(self.sigma.to_le_bytes());
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrate_populates_sigma() {
        // C = 1.0 → Δ₂ = 2C = 2 → σ ≈ 2.1623 (see amgm::tests::calibrate_sigma_at_ref_config).
        let cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
        assert!(cfg.sigma > 0.0);
        assert!((cfg.sigma - 2.1623).abs() < 1e-3);
    }

    #[test]
    fn digest_differs_when_epsilon_differs() {
        let a = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
        let b = DpForwardConfig::calibrate(2.0, 1e-5, 1.0);
        assert_ne!(a.config_digest(), b.config_digest());
    }

    #[test]
    fn digest_differs_when_clip_differs() {
        let a = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
        let b = DpForwardConfig::calibrate(4.0, 1e-5, 0.5);
        assert_ne!(a.config_digest(), b.config_digest());
    }

    #[test]
    fn digest_stable_across_constructions() {
        let a = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
        let b = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
        assert_eq!(a.config_digest(), b.config_digest());
    }
}
