use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;

use gelo_protocol::{TrustedExecutor, WeightHandle, WeightKind};
use rag_core::Embedder;

use super::config::BertConfig;
use super::forward;
use super::weights::BertWeights;
use crate::common::pool;
use crate::common::tokenizer::HfTokenizer;

#[cfg(feature = "dp-forward")]
use rand::SeedableRng;
#[cfg(feature = "dp-forward")]
use rand_chacha::ChaCha20Rng;

/// A BERT-class embedding model whose Q/K/V/O + FFN GEMMs are routed
/// through a GELO-style `TrustedExecutor`.
pub struct GeloBertEmbedder<X: TrustedExecutor> {
    cfg: BertConfig,
    tokenizer: HfTokenizer,
    weights: Arc<BertWeights>,
    exec: X,
    max_len: usize,
    /// Hex-encoded `sha256(safetensors_bytes)`, optionally extended by
    /// `sha256(weights ‖ dp_cfg.config_digest())` if [`Self::with_dp_forward`]
    /// is called. Rides through `AttestationEvidence::model_identity` so a
    /// relying party can pin `(weights, ε, δ, C, σ)`.
    model_identity: String,
    /// Raw sha256 of the weights, before any DP-config mixing. Cached so
    /// `with_dp_forward` can re-derive `model_identity` deterministically.
    #[cfg(feature = "dp-forward")]
    weights_identity: [u8; 32],
    /// Recipe-B aMGM config applied to the pooled embedding inside this
    /// embedder before `embed()` returns.
    #[cfg(feature = "dp-forward")]
    dp_forward: Option<dp_forward::DpForwardConfig>,
    /// Dedicated RNG for DP noise sampling. Seeded from `OsRng` at
    /// construction; not deterministic (DP noise must be unique per call).
    #[cfg(feature = "dp-forward")]
    dp_rng: ChaCha20Rng,
}

impl<X: TrustedExecutor> GeloBertEmbedder<X> {
    /// Build from already-loaded artifacts.
    pub fn new(
        cfg: BertConfig,
        tokenizer: HfTokenizer,
        weights: Arc<BertWeights>,
        mut exec: X,
    ) -> Result<Self> {
        for (li, layer) in weights.layers.iter().enumerate() {
            if !cfg.offload_layer(li) {
                continue;
            }
            let li16 = li as u16;
            exec.provision_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view())?;
            exec.provision_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view())?;
            exec.provision_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view())?;
            exec.provision_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view())?;
            exec.provision_weight(
                WeightHandle::new(li16, WeightKind::FfnUp),
                layer.w_ffn_up.view(),
            )?;
            exec.provision_weight(
                WeightHandle::new(li16, WeightKind::FfnDown),
                layer.w_ffn_down.view(),
            )?;
        }
        let max_len = cfg.max_seq_len.min(cfg.max_position_embeddings);
        let model_identity = hex::encode(weights.model_identity);
        Ok(Self {
            cfg,
            tokenizer,
            #[cfg(feature = "dp-forward")]
            weights_identity: weights.model_identity,
            weights,
            exec,
            max_len,
            model_identity,
            #[cfg(feature = "dp-forward")]
            dp_forward: None,
            #[cfg(feature = "dp-forward")]
            dp_rng: ChaCha20Rng::from_os_rng(),
        })
    }

    /// Enable Recipe-B aMGM noise (DP-Forward) on the pooled embedding.
    /// See [`crate::decoder::embedder::GeloQwenEmbedder::with_dp_forward`]
    /// for the full rationale; behaviour is identical for the BERT path.
    #[cfg(feature = "dp-forward")]
    pub fn with_dp_forward(mut self, cfg: dp_forward::DpForwardConfig) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.weights_identity);
        hasher.update(cfg.config_digest());
        let combined: [u8; 32] = hasher.finalize().into();
        self.model_identity = hex::encode(combined);
        self.dp_forward = Some(cfg);
        self
    }

    /// Download `BAAI/bge-small-en-v1.5` from the HuggingFace hub (using the
    /// local cache when already present) and assemble an embedder.
    pub fn from_pretrained(model_id: &str, exec: X) -> Result<Self> {
        let api = ApiBuilder::new()
            .with_progress(false)
            .build()
            .context("building HuggingFace hub API client")?;
        let repo = api.model(model_id.to_string());
        let config_path = repo.get("config.json").context("downloading config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("downloading tokenizer.json")?;
        let safetensors_path = repo
            .get("model.safetensors")
            .context("downloading model.safetensors")?;

        Self::from_local(&config_path, &tokenizer_path, &safetensors_path, exec)
    }

    /// Build from local files (avoids any network access).
    pub fn from_local(
        config_path: &Path,
        tokenizer_path: &Path,
        safetensors_path: &Path,
        exec: X,
    ) -> Result<Self> {
        let cfg_bytes =
            std::fs::read(config_path).with_context(|| format!("reading {}", config_path.display()))?;
        let cfg: BertConfig =
            serde_json::from_slice(&cfg_bytes).context("parsing config.json")?;
        // [SEP] = 102 in bert-base-uncased; BGE inherits that vocab.
        let tokenizer = HfTokenizer::from_file(tokenizer_path)?.with_truncation_token(102);
        let weights = Arc::new(BertWeights::from_safetensors(safetensors_path, &cfg)?);
        Self::new(cfg, tokenizer, weights, exec)
    }

    /// Override the maximum sequence length (clamped to the config limit).
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len.min(self.cfg.max_position_embeddings);
        self
    }

    pub fn config(&self) -> &BertConfig {
        &self.cfg
    }
}

impl<X: TrustedExecutor> Embedder for GeloBertEmbedder<X> {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let ids = self.tokenizer.encode(text, self.max_len)?;
            let hidden = forward::run(&self.cfg, &self.weights, &mut self.exec, &ids)?;
            let pooled = pool::mean_l2(hidden.view());
            #[allow(unused_mut)]
            let mut pooled_vec = pooled.to_vec();
            #[cfg(feature = "dp-forward")]
            if let Some(cfg) = &self.dp_forward {
                dp_forward::amgm::clip_l2_in_place(&mut pooled_vec, cfg.clip_c);
                dp_forward::amgm::add_gaussian_noise(&mut pooled_vec, cfg.sigma, &mut self.dp_rng);
            }
            out.push(pooled_vec);
        }
        Ok(out)
    }

    fn model_identity(&self) -> &[u8] {
        self.model_identity.as_bytes()
    }
}
