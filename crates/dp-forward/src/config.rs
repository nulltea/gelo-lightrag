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
    /// Position to apply aMGM noise inside the embedder's forward pass.
    ///
    /// - `Some(n)` — apply to each token-row of the hidden state at the
    ///   end of transformer layer `n` (0-indexed; after the layer's
    ///   output residual). For BERT post-norm this is the
    ///   `add_and_norm_2` position from
    ///   [`xiangyue9607/DP-Forward`](https://github.com/xiangyue9607/DP-Forward)
    ///   (paper default: `noise_layer = 10` of 12 in BERT-base). For
    ///   decoder-LLM pre-norm (Qwen3) it is the final residual add at
    ///   the end of the block — analogous position, paper-extrapolated.
    /// - `None` — legacy: apply at the pooled embedding. Sub-1 % CPU cost
    ///   but destroys retrieval at standard ε on unit-norm embeddings (see
    ///   `docs/prototype/dp-forward.md` §6). Kept for backwards
    ///   compatibility; not recommended for retrieval workloads.
    pub layer_index: Option<usize>,
}

impl DpForwardConfig {
    /// Construct a config and run Balle–Wang calibration to compute σ.
    /// Defaults to pooled-output application (`layer_index = None`); use
    /// [`Self::with_layer_index`] to switch to intermediate-layer aMGM
    /// (recommended for retrieval — see module-level doc).
    pub fn calibrate(epsilon: f64, delta: f64, clip_c: f32) -> Self {
        let sigma = amgm::calibrate_sigma(epsilon, delta, (2.0 * clip_c) as f64);
        Self {
            epsilon,
            delta,
            clip_c,
            sigma,
            layer_index: None,
        }
    }

    /// Builder: select an intermediate transformer layer for the noise
    /// hook. See field-level doc on [`Self::layer_index`].
    pub fn with_layer_index(mut self, layer_index: Option<usize>) -> Self {
        self.layer_index = layer_index;
        self
    }

    /// 32-byte stable digest over canonical bytes — folded into
    /// `Embedder::model_identity` so a SEV-SNP attestation report binds to
    /// the exact DP parameters in effect.
    ///
    /// The digest commits to `(epsilon, delta, clip_c, sigma, layer_index)`,
    /// which means a verifier who pins `expected_model_id` can detect
    /// *parameter substitution* (different ε/δ), *calibration substitution*
    /// (matching ε/δ but a manipulated σ), and *position substitution*
    /// (same parameters but applied at a different layer).
    pub fn config_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"dp-forward.v2");
        hasher.update(self.epsilon.to_le_bytes());
        hasher.update(self.delta.to_le_bytes());
        hasher.update(self.clip_c.to_le_bytes());
        hasher.update(self.sigma.to_le_bytes());
        // None ⇒ sentinel u64::MAX (no valid layer index); Some(n) ⇒ n.
        let layer_marker: u64 = self.layer_index.map(|i| i as u64).unwrap_or(u64::MAX);
        hasher.update(layer_marker.to_le_bytes());
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

    #[test]
    fn digest_differs_when_layer_index_differs() {
        let a = DpForwardConfig::calibrate(4.0, 1e-5, 1.0).with_layer_index(Some(10));
        let b = DpForwardConfig::calibrate(4.0, 1e-5, 1.0).with_layer_index(Some(11));
        let pooled = DpForwardConfig::calibrate(4.0, 1e-5, 1.0); // layer_index = None
        assert_ne!(a.config_digest(), b.config_digest());
        assert_ne!(a.config_digest(), pooled.config_digest());
        assert_ne!(b.config_digest(), pooled.config_digest());
    }

    #[test]
    fn with_layer_index_round_trips() {
        let cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0).with_layer_index(Some(10));
        assert_eq!(cfg.layer_index, Some(10));
        let cfg2 = cfg.with_layer_index(None);
        assert_eq!(cfg2.layer_index, None);
    }
}
