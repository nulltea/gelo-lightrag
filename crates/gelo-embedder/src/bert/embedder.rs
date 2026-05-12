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

/// A BERT-class embedding model whose Q/K/V/O + FFN GEMMs are routed
/// through a GELO-style `TrustedExecutor`.
pub struct GeloBertEmbedder<X: TrustedExecutor> {
    cfg: BertConfig,
    tokenizer: HfTokenizer,
    weights: Arc<BertWeights>,
    exec: X,
    max_len: usize,
    /// Hex-encoded `sha256(safetensors_bytes)`. Stored as a UTF-8 string so
    /// it can ride through `AttestationEvidence::model_identity` (which is
    /// `String`). The relying party recomputes the same hex over the
    /// expected weights and compares.
    model_identity: String,
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
            weights,
            exec,
            max_len,
            model_identity,
        })
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
            out.push(pooled.to_vec());
        }
        Ok(out)
    }

    fn model_identity(&self) -> &[u8] {
        self.model_identity.as_bytes()
    }
}
