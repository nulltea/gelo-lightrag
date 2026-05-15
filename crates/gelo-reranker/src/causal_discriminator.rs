//! Causal-LM-as-yes/no-discriminator rerank service.
//!
//! Reuses `gelo_embedder::decoder` for the transformer backbone. The
//! reranker tokenizes a chat-templated `{query, doc}` prompt, runs one
//! forward pass under the GELO mask, then gathers the last-token
//! logits at the pinned [`YesNoHead`] vocab IDs and returns
//! `softmax([no, yes])[1]`.
//!
//! Reference model: `Qwen/Qwen3-Reranker-0.6B` (Qwen3-0.6B backbone +
//! `tie_word_embeddings = true`). The "LM head" is the tied input
//! embedding row — we never materialise the full vocab logit vector;
//! the two scalars we need are computed as two dot products against
//! `weights.token_embedding.row(yes_id)` and `.row(no_id)`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use hf_hub::api::sync::ApiBuilder;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use gelo_embedder::common::tokenizer::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_protocol::profile;
use gelo_protocol::{TrustedExecutor, WeightHandle, WeightKind};

use crate::head::YesNoHead;
use crate::output::EncryptedRerankBundle;
use crate::score::{ScoredCandidate, top_k_with_tie_shuffle};
use crate::service::{RerankError, RerankRequest, RerankService};
use crate::session::SessionKey;

/// Qwen3-Reranker's published chat template. Inlined and SHA-pinned
/// so a tokenizer-config drift can't silently change what the model
/// sees. The trailing `<think>\n\n</think>\n\n` block is part of the
/// official template — the model's next-token distribution
/// concentrates mass on `yes` / `no` at exactly that position.
pub const QWEN3_RERANKER_TEMPLATE: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. Answer only \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub struct CausalDiscriminatorRerankService<X: TrustedExecutor> {
    cfg: DecoderConfig,
    tokenizer: HfTokenizer,
    weights: Arc<DecoderWeights>,
    rope: Arc<RopeTables>,
    head: YesNoHead,
    exec: X,
    max_len: usize,
    /// SHA-256(weights_identity || yes_id || no_id || template_bytes).
    /// The yes/no IDs and the chat template are part of the attested
    /// scheme — they directly affect the score the model produces.
    model_identity: Vec<u8>,
}

impl<X: TrustedExecutor> CausalDiscriminatorRerankService<X> {
    /// Build from already-loaded artifacts. Provisioning mirrors
    /// [`gelo_embedder::decoder::GeloQwenEmbedder::new`] one-to-one;
    /// only the post-forward gather differs.
    pub fn new(
        cfg: DecoderConfig,
        tokenizer: HfTokenizer,
        weights: Arc<DecoderWeights>,
        rope: Arc<RopeTables>,
        head: YesNoHead,
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
                WeightHandle::new(li16, WeightKind::FfnGate),
                layer.w_gate.view(),
            )?;
            exec.provision_weight(
                WeightHandle::new(li16, WeightKind::FfnUp),
                layer.w_up.view(),
            )?;
            exec.provision_weight(
                WeightHandle::new(li16, WeightKind::FfnDown),
                layer.w_down.view(),
            )?;
        }
        let max_len = cfg.max_seq_len.min(cfg.max_position_embeddings);
        let mut hasher = Sha256::new();
        hasher.update(weights.model_identity);
        hasher.update(head.yes_token_id.to_le_bytes());
        hasher.update(head.no_token_id.to_le_bytes());
        hasher.update(QWEN3_RERANKER_TEMPLATE.as_bytes());
        let model_identity = hasher.finalize().to_vec();
        Ok(Self {
            cfg,
            tokenizer,
            weights,
            rope,
            head,
            exec,
            max_len,
            model_identity,
        })
    }

    /// Download model + tokenizer + config from the HuggingFace Hub.
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
        let shard_paths = find_safetensors_shards(&repo)?;
        Self::from_local(&config_path, &tokenizer_path, &shard_paths, exec)
    }

    /// Build from local files. `safetensors_paths` may be a single
    /// file or a sharded list. Yes/no token IDs are resolved from the
    /// tokenizer at load — same pinning as the embedder.
    pub fn from_local(
        config_path: &Path,
        tokenizer_path: &Path,
        safetensors_paths: &[PathBuf],
        exec: X,
    ) -> Result<Self> {
        let cfg_bytes = std::fs::read(config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let cfg: DecoderConfig =
            serde_json::from_slice(&cfg_bytes).context("parsing config.json")?;
        let tokenizer = HfTokenizer::from_file(tokenizer_path)?;
        let shard_refs: Vec<&Path> = safetensors_paths.iter().map(|p| p.as_path()).collect();
        let weights = Arc::new(DecoderWeights::from_safetensors(&shard_refs, &cfg)?);
        let rope = Arc::new(RopeTables::new(
            cfg.head_dim_value(),
            cfg.max_position_embeddings,
            cfg.rope_theta,
        ));

        // Qwen tokenisers register single-token "yes" / "no" as
        // distinct vocab entries; if not, the model is not a
        // discriminator reranker and we reject loudly.
        let yes_token_id = tokenizer
            .token_id("yes")
            .ok_or_else(|| anyhow!("tokenizer does not expose token id for 'yes'"))?;
        let no_token_id = tokenizer
            .token_id("no")
            .ok_or_else(|| anyhow!("tokenizer does not expose token id for 'no'"))?;

        Self::new(
            cfg,
            tokenizer,
            weights,
            rope,
            YesNoHead { yes_token_id, no_token_id },
            exec,
        )
    }

    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len.min(self.cfg.max_position_embeddings);
        self
    }

    /// Run the final transformer block fully inside the TEE (no
    /// offload). GELO §3.2 sensitive-layer exclusion: the final block
    /// contains the highest-information hidden state for the chat-
    /// templated prompt, and keeping it TEE-resident raises the bar
    /// for known-plaintext attacks. Mirrors
    /// `CrossEncoderRerankService::with_skip_last_layer`.
    pub fn with_skip_last_layer(mut self, enabled: bool) -> Self {
        self.cfg.skip_last_layer = enabled;
        self
    }

    /// Toggle OutAttnMult routing of `Q·Kᵀ` through the GPU engine
    /// for the decoder attention. Default-on at the
    /// `DecoderConfig::use_out_attn_mult = true` level; this builder
    /// is the explicit toggle on the reranker for shape-specific
    /// tuning. See the cross-encoder counterpart for the iGPU-vs-dGPU
    /// trade-off reasoning.
    pub fn with_out_attn_mult(mut self, enabled: bool) -> Self {
        self.cfg.use_out_attn_mult = enabled;
        self
    }

    /// Override the OutAttnMult auto-switch threshold (`None` resolves
    /// to `hidden_size`).
    pub fn with_out_attn_mult_min_seq_len(mut self, min_seq_len: Option<usize>) -> Self {
        self.cfg.out_attn_mult_min_seq_len = min_seq_len;
        self
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    pub fn head(&self) -> YesNoHead {
        self.head
    }

    /// Score one `(query, doc)` pair using [`QWEN3_RERANKER_TEMPLATE`]
    /// for prompt construction.
    pub fn score_pair(&mut self, query: &str, document: &str) -> Result<f32> {
        let ids = profile::time("tee:tokenize", || {
            let prompt = QWEN3_RERANKER_TEMPLATE
                .replace("{query}", query)
                .replace("{document}", document);
            self.tokenizer.encode(&prompt, self.max_len)
        })?;
        self.score_input_ids(&ids)
    }

    /// Same as [`Self::score_pair`] but takes pre-tokenized input ids.
    /// Used by the parity tests; also the right entry point for
    /// callers that want to swap in a custom prompt template.
    pub fn score_input_ids(&mut self, input_ids: &[u32]) -> Result<f32> {
        let hidden = forward::run(
            &self.cfg,
            &self.weights,
            &self.rope,
            &mut self.exec,
            input_ids,
        )?;
        Ok(profile::time("tee:yesno_head", || {
            let last = hidden.row(hidden.shape()[0] - 1);
            // Tied LM head: project last hidden by token_embedding.row(id)
            // for each of the two vocab IDs we care about. Skips the
            // (1, hidden) × (hidden, vocab) matmul over all 151669 rows.
            let yes_logit =
                last.dot(&self.weights.token_embedding.row(self.head.yes_token_id as usize));
            let no_logit =
                last.dot(&self.weights.token_embedding.row(self.head.no_token_id as usize));
            // Numerically stable softmax([no, yes])[1]:
            let mx = yes_logit.max(no_logit);
            let e_yes = (yes_logit - mx).exp();
            let e_no = (no_logit - mx).exp();
            e_yes / (e_yes + e_no)
        }))
    }
}

impl<X: TrustedExecutor + Clone + Send + Sync> RerankService for CausalDiscriminatorRerankService<X> {
    fn model_identity(&self) -> &[u8] {
        &self.model_identity
    }

    fn family(&self) -> &'static str {
        "causal-discriminator"
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

        let scored = score_candidates_parallel(self, request).map_err(RerankError::Forward)?;

        let qkey = session.derive_query_key(&request.query_id);
        let mut rng = ChaCha20Rng::from_seed(*qkey.as_bytes());
        let ranked = top_k_with_tie_shuffle(scored, request.top_k, &mut rng);

        let decoy_len = request
            .candidates
            .iter()
            .map(|c| c.text.len())
            .max()
            .unwrap_or(64);

        EncryptedRerankBundle::seal(&qkey, &ranked, request.k_max, &mut rng, decoy_len)
    }
}

/// Score every candidate against the query, returning `ScoredCandidate`s
/// in the input order. Mirrors `cross_encoder::score_candidates_parallel`:
/// single-candidate fast path runs sequentially on the service's own
/// executor; multi-candidate fan-out via rayon, one cloned executor per
/// worker with an independent RNG stream.
fn score_candidates_parallel<X: TrustedExecutor + Clone + Send + Sync>(
    svc: &mut CausalDiscriminatorRerankService<X>,
    request: &RerankRequest<'_>,
) -> Result<Vec<ScoredCandidate>> {
    if request.candidates.len() <= 1 {
        let mut scored = Vec::with_capacity(request.candidates.len());
        for cand in request.candidates {
            let score = svc.score_pair(request.query, &cand.text)?;
            scored.push(ScoredCandidate {
                chunk_id: cand.chunk_id.clone(),
                text: cand.text.clone(),
                score,
            });
        }
        return Ok(scored);
    }

    use rayon::prelude::*;
    let query = request.query;
    request
        .candidates
        .par_iter()
        .enumerate()
        .map(|(idx, cand)| {
            let mut worker_exec = svc.exec.clone();
            worker_exec.set_rng_stream(idx as u64);
            let ids = profile::time("tee:tokenize", || {
                let prompt = QWEN3_RERANKER_TEMPLATE
                    .replace("{query}", query)
                    .replace("{document}", &cand.text);
                svc.tokenizer.encode(&prompt, svc.max_len)
            })?;
            let hidden = forward::run(
                &svc.cfg,
                &svc.weights,
                &svc.rope,
                &mut worker_exec,
                &ids,
            )?;
            let score = profile::time("tee:yesno_head", || {
                let last = hidden.row(hidden.shape()[0] - 1);
                let yes_logit =
                    last.dot(&svc.weights.token_embedding.row(svc.head.yes_token_id as usize));
                let no_logit =
                    last.dot(&svc.weights.token_embedding.row(svc.head.no_token_id as usize));
                let mx = yes_logit.max(no_logit);
                let e_yes = (yes_logit - mx).exp();
                let e_no = (no_logit - mx).exp();
                e_yes / (e_yes + e_no)
            });
            Ok(ScoredCandidate {
                chunk_id: cand.chunk_id.clone(),
                text: cand.text.clone(),
                score,
            })
        })
        .collect()
}

fn find_safetensors_shards(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    // First try the single-file convention.
    if let Ok(p) = repo.get("model.safetensors") {
        return Ok(vec![p]);
    }
    // Fall back to the sharded index.
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("neither model.safetensors nor model.safetensors.index.json available")?;
    let raw = std::fs::read(&index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;
    let index: serde_json::Value =
        serde_json::from_slice(&raw).context("parsing model.safetensors.index.json")?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("index.json missing weight_map object"))?;
    let mut shards: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for v in weight_map.values() {
        if let Some(s) = v.as_str() {
            shards.insert(s.to_string());
        }
    }
    let mut paths = Vec::with_capacity(shards.len());
    for shard in shards {
        paths.push(repo.get(&shard).with_context(|| format!("fetching shard {shard}"))?);
    }
    Ok(paths)
}
