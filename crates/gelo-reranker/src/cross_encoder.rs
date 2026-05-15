//! BERT-class cross-encoder rerank service.
//!
//! Reuses `gelo_embedder::bert` for the encoder forward — the GELO
//! mask + TwinShield primitives apply unchanged. The reranker is a
//! `(query, doc)` joint encoder followed by a two-layer
//! `XLMRobertaForSequenceClassification`-style head on the `[CLS]` row:
//! `out_proj(tanh(dense(cls)))`.
//!
//! Reference model: `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa-large
//! backbone, Apache-2.0, 568M params). Any other model that exports
//! the standard `classifier.dense.*` + `classifier.out_proj.*` head
//! over a BERT-shaped encoder works the same way.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use gelo_embedder::bert::config::BertConfig;
use gelo_embedder::bert::forward;
use gelo_embedder::bert::weights::BertWeights;
use gelo_embedder::common::tokenizer::HfTokenizer;
use gelo_protocol::profile;
use gelo_protocol::{TrustedExecutor, WeightHandle, WeightKind};

use crate::head::ClassifierHead;
use crate::output::EncryptedRerankBundle;
use crate::score::{ScoredCandidate, top_k_with_tie_shuffle};
use crate::service::{RerankError, RerankRequest, RerankService};
use crate::session::SessionKey;

pub struct CrossEncoderRerankService<X: TrustedExecutor> {
    cfg: BertConfig,
    tokenizer: HfTokenizer,
    weights: Arc<BertWeights>,
    head: ClassifierHead,
    exec: X,
    max_len: usize,
    /// `sha256(weights_identity ‖ head.identity)` — the attestation
    /// report's model binding covers backbone + head as one unit.
    model_identity: Vec<u8>,
}

impl<X: TrustedExecutor> CrossEncoderRerankService<X> {
    /// Build from already-loaded artifacts. Mirrors
    /// [`gelo_embedder::bert::GeloBertEmbedder::new`].
    pub fn new(
        cfg: BertConfig,
        tokenizer: HfTokenizer,
        weights: Arc<BertWeights>,
        head: ClassifierHead,
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
        let mut hasher = Sha256::new();
        hasher.update(weights.model_identity);
        hasher.update(head.identity);
        let model_identity = hasher.finalize().to_vec();
        Ok(Self {
            cfg,
            tokenizer,
            weights,
            head,
            exec,
            max_len,
            model_identity,
        })
    }

    /// Download model + tokenizer + config from the HuggingFace Hub
    /// and construct the service. Caches under the user's default HF
    /// cache directory.
    pub fn from_pretrained(model_id: &str, exec: X) -> Result<Self> {
        let api = ApiBuilder::new()
            .with_progress(false)
            .build()
            .context("constructing HF Hub API")?;
        let repo = api.model(model_id.to_string());
        let config_path = repo.get("config.json").context("fetching config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("fetching tokenizer.json")?;
        let safetensors_path = repo
            .get("model.safetensors")
            .context("fetching model.safetensors")?;

        load_from_paths(&config_path, &tokenizer_path, &safetensors_path, exec)
    }

    /// Load from a local directory holding `config.json`,
    /// `tokenizer.json`, and `model.safetensors`. Used by air-gapped /
    /// CVM-internal deployments.
    pub fn from_local(model_dir: &Path, exec: X) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let safetensors_path = model_dir.join("model.safetensors");
        load_from_paths(&config_path, &tokenizer_path, &safetensors_path, exec)
    }

    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len.min(self.cfg.max_position_embeddings);
        self
    }

    pub fn config(&self) -> &BertConfig {
        &self.cfg
    }

    pub fn head(&self) -> &ClassifierHead {
        &self.head
    }

    /// Score one `(query, doc)` pair through the GELO-protected
    /// forward. Returns a scalar — used internally by [`Self::rerank`].
    pub fn score_pair(&mut self, query: &str, document: &str) -> Result<f32> {
        let ids = profile::time("tee:tokenize", || {
            self.tokenizer.encode_pair(query, document, self.max_len)
        })?;
        self.score_input_ids(&ids)
    }

    /// Same as [`Self::score_pair`] but takes pre-tokenized input ids
    /// directly. Used by the parity tests that bypass tokenization to
    /// keep the model-shape and head-shape concerns isolated from
    /// tokenizer-file dependencies.
    pub fn score_input_ids(&mut self, input_ids: &[u32]) -> Result<f32> {
        let hidden = forward::run(&self.cfg, &self.weights, &mut self.exec, input_ids)?;
        // CLS row is row 0 (the `<s>` token in XLM-R; `[CLS]` in BERT).
        Ok(profile::time("tee:classifier_head", || self.head.score(hidden.row(0))))
    }
}

fn load_from_paths<X: TrustedExecutor>(
    config_path: &Path,
    tokenizer_path: &Path,
    safetensors_path: &Path,
    exec: X,
) -> Result<CrossEncoderRerankService<X>> {
    let cfg_bytes = std::fs::read(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let cfg: BertConfig = serde_json::from_slice(&cfg_bytes).context("parsing config.json")?;
    let weights = Arc::new(BertWeights::from_safetensors(safetensors_path, &cfg)?);
    let head = ClassifierHead::from_safetensors(safetensors_path)?;
    let mut tokenizer = HfTokenizer::from_file(tokenizer_path)?;
    if let Some(sep) = tokenizer.token_id("</s>").or_else(|| tokenizer.token_id("[SEP]")) {
        tokenizer = tokenizer.with_truncation_token(sep);
    }
    CrossEncoderRerankService::new(cfg, tokenizer, weights, head, exec)
}

impl<X: TrustedExecutor> RerankService for CrossEncoderRerankService<X> {
    fn model_identity(&self) -> &[u8] {
        &self.model_identity
    }

    fn family(&self) -> &'static str {
        "cross-encoder"
    }

    fn rerank(
        &mut self,
        session: &SessionKey,
        request: &RerankRequest<'_>,
    ) -> Result<EncryptedRerankBundle, RerankError> {
        if request.top_k == 0 {
            return Err(RerankError::InvalidRequest("top_k must be > 0".into()));
        }
        if request.top_k > request.k_max {
            return Err(RerankError::InvalidRequest(format!(
                "top_k={} exceeds k_max={}",
                request.top_k, request.k_max
            )));
        }

        let mut scored: Vec<ScoredCandidate> = Vec::with_capacity(request.candidates.len());
        for cand in request.candidates {
            let score = self
                .score_pair(request.query, &cand.text)
                .map_err(RerankError::Forward)?;
            scored.push(ScoredCandidate {
                chunk_id: cand.chunk_id.clone(),
                text: cand.text.clone(),
                score,
            });
        }

        // RNG for tie-shuffle + bundle nonce sampling. Derived from
        // the per-query key so two runs against the same session +
        // query_id reproduce — useful for debugging, but every nonce
        // still depends on the key so AEAD remains safe.
        let qkey = session.derive_query_key(&request.query_id);
        let mut rng = ChaCha20Rng::from_seed(*qkey.as_bytes());
        let ranked = top_k_with_tie_shuffle(scored, request.top_k, &mut rng);

        // Decoy text length: match the longest real candidate so the
        // wire shape doesn't reveal which item is a decoy. Falls back
        // to a small constant when there are no candidates (which the
        // empty-`top_k` guard above usually catches).
        let decoy_len = request
            .candidates
            .iter()
            .map(|c| c.text.len())
            .max()
            .unwrap_or(64);

        EncryptedRerankBundle::seal(&qkey, &ranked, request.k_max, &mut rng, decoy_len)
    }
}
