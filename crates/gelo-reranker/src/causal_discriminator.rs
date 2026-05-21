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
        mut weights: DecoderWeights,
        rope: Arc<RopeTables>,
        head: YesNoHead,
        mut exec: X,
    ) -> Result<Self> {
        for li in 0..weights.layers.len() {
            if !cfg.offload_layer(li) {
                continue;
            }
            let li16 = li as u16;
            let layer = &mut weights.layers[li];
            // bf16-native + take() — see GeloQwenEmbedder::new for the
            // rationale (host RAM drops once Arc refcount hits 0).
            let wq = layer.wq.take().ok_or_else(|| anyhow::anyhow!("layer {li}: wq already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::Q), wq)?;
            let wk = layer.wk.take().ok_or_else(|| anyhow::anyhow!("layer {li}: wk already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::K), wk)?;
            let wv = layer.wv.take().ok_or_else(|| anyhow::anyhow!("layer {li}: wv already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::V), wv)?;
            let wo = layer.wo.take().ok_or_else(|| anyhow::anyhow!("layer {li}: wo already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::O), wo)?;
            let w_gate = layer.w_gate.take().ok_or_else(|| anyhow::anyhow!("layer {li}: w_gate already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::FfnGate), w_gate)?;
            let w_up = layer.w_up.take().ok_or_else(|| anyhow::anyhow!("layer {li}: w_up already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::FfnUp), w_up)?;
            let w_down = layer.w_down.take().ok_or_else(|| anyhow::anyhow!("layer {li}: w_down already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::FfnDown), w_down)?;
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
            weights: Arc::new(weights),
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
        let weights = DecoderWeights::from_safetensors(&shard_refs, &cfg)?;
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
            // Token embedding is bf16; widen per-element and accumulate
            // in f32 (same precision as the pre-bf16 path).
            let yes_logit: f32 = last
                .iter()
                .zip(
                    self.weights
                        .token_embedding
                        .row(self.head.yes_token_id as usize)
                        .iter(),
                )
                .map(|(a, b)| a * b.to_f32())
                .sum();
            let no_logit: f32 = last
                .iter()
                .zip(
                    self.weights
                        .token_embedding
                        .row(self.head.no_token_id as usize)
                        .iter(),
                )
                .map(|(a, b)| a * b.to_f32())
                .sum();
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

        // M1.11 R2: prefer the batched-prefill path when there are
        // multiple candidates. Falls back to the legacy per-Rayon-worker
        // fan-out when batched fails (e.g. tokenizer error per
        // candidate) so single-pair callers and the migration path
        // keep working. Set `GELO_RERANKER_LEGACY_RAYON=1` to force the
        // legacy path for A/B measurement.
        let force_legacy =
            std::env::var("GELO_RERANKER_LEGACY_RAYON").as_deref() == Ok("1");
        let scored = if force_legacy || request.candidates.len() <= 1 {
            #[allow(deprecated)]
            {
                score_candidates_parallel(self, request).map_err(RerankError::Forward)?
            }
        } else {
            score_candidates_batched(self, request).map_err(RerankError::Forward)?
        };

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

/// **M1.11 R2** — Score N candidates in one batched-prefill forward.
///
/// Replaces the Rayon per-worker fan-out at the rerank entry point.
/// One `forward::run_batched` call covers all N prompts; the substrate
/// samples N per-sequence masks (`PerSequence` session kind, see
/// `m1-11-batched-decode.md` §3.4) and applies them transparently
/// through `offload_qkv` / `offload_linear` / `offload_linear_many`.
///
/// Yes/no scoring runs after the batched forward: gather the
/// `last-valid-row` per sequence (i.e. row `seq_lens[b] − 1`), project
/// against the tied embedding's `yes_id` / `no_id` rows, return
/// `softmax([no, yes])[1]`.
pub(crate) fn score_candidates_batched<X: TrustedExecutor + Clone + Send + Sync>(
    svc: &mut CausalDiscriminatorRerankService<X>,
    request: &RerankRequest<'_>,
) -> Result<Vec<ScoredCandidate>> {
    let query = request.query;
    // 1. Tokenise every candidate up-front; collect prompts and pre-
    //    measure lengths so we know n_max.
    let prompts: Result<Vec<Vec<u32>>> = request
        .candidates
        .iter()
        .map(|cand| {
            profile::time("tee:tokenize", || {
                let prompt = QWEN3_RERANKER_TEMPLATE
                    .replace("{query}", query)
                    .replace("{document}", &cand.text);
                svc.tokenizer.encode(&prompt, svc.max_len)
            })
        })
        .collect();
    let prompts = prompts?;
    if prompts.is_empty() {
        return Ok(Vec::new());
    }

    // 2. One batched forward pass.
    let (hidden_3d, seq_lens) =
        forward::run_batched(&svc.cfg, &svc.weights, &svc.rope, &mut svc.exec, &prompts)?;

    // 3. Per-sequence yes/no gather. Last-valid row is at index
    //    `seq_lens[b] − 1`.
    let yes_row = svc
        .weights
        .token_embedding
        .row(svc.head.yes_token_id as usize);
    let no_row = svc
        .weights
        .token_embedding
        .row(svc.head.no_token_id as usize);

    let mut scored = Vec::with_capacity(prompts.len());
    profile::time("tee:yesno_head", || {
        for (b, cand) in request.candidates.iter().enumerate() {
            let valid_n = seq_lens[b];
            debug_assert!(valid_n > 0, "candidate {b} tokenised to 0 tokens");
            let last = hidden_3d.slice(ndarray::s![b, valid_n - 1, ..]);
            let yes_logit: f32 = last
                .iter()
                .zip(yes_row.iter())
                .map(|(a, b)| a * b.to_f32())
                .sum();
            let no_logit: f32 = last
                .iter()
                .zip(no_row.iter())
                .map(|(a, b)| a * b.to_f32())
                .sum();
            let mx = yes_logit.max(no_logit);
            let e_yes = (yes_logit - mx).exp();
            let e_no = (no_logit - mx).exp();
            let score = e_yes / (e_yes + e_no);
            scored.push(ScoredCandidate {
                chunk_id: cand.chunk_id.clone(),
                text: cand.text.clone(),
                score,
            });
        }
    });
    Ok(scored)
}

/// **M1.11 deprecated** — Rayon per-worker fan-out path. Replaced by
/// [`score_candidates_batched`] which amortises GPU dispatch + mask
/// sampling across the batch. Kept for the `candidates.len() ≤ 1`
/// fast path and the `GELO_RERANKER_LEGACY_RAYON=1` A/B-measurement
/// escape hatch.
///
/// Mirrors `cross_encoder::score_candidates_parallel`:
/// single-candidate fast path runs sequentially on the service's own
/// executor; multi-candidate fan-out via rayon, one cloned executor per
/// worker with an independent RNG stream.
#[deprecated(
    since = "0.1.0",
    note = "M1.11: prefer score_candidates_batched. \
            Override via GELO_RERANKER_LEGACY_RAYON=1 for A/B measurement only."
)]
#[allow(deprecated)]
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
                let yes_logit: f32 = last
                    .iter()
                    .zip(svc.weights.token_embedding.row(svc.head.yes_token_id as usize).iter())
                    .map(|(a, b)| a * b.to_f32())
                    .sum();
                let no_logit: f32 = last
                    .iter()
                    .zip(svc.weights.token_embedding.row(svc.head.no_token_id as usize).iter())
                    .map(|(a, b)| a * b.to_f32())
                    .sum();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use gelo_embedder::common::tokenizer::HfTokenizer;
    use gelo_embedder::decoder::config::DecoderConfig;
    use gelo_embedder::decoder::rope::RopeTables;
    use gelo_embedder::decoder::weights::{DecoderLayerWeights, DecoderWeights};
    use gelo_protocol::rng::MaskSeed;
    use gelo_protocol::{
        GpuOffloadEngine, InProcessTrustedExecutor, RayonCpuEngine, WeightHandle, WeightKind,
    };
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    use rag_core::ChunkId;

    use crate::head::YesNoHead;
    use crate::service::{RerankCandidate, RerankRequest};
    use crate::session::QueryId;

    fn tiny_cfg() -> DecoderConfig {
        DecoderConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(8),
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: true,
            max_seq_len: 64,
            skip_first_layers: 0,
            skip_last_layer: false,
            use_out_attn_mult: false,
            out_attn_mult_min_seq_len: None,
            use_perm_attention: false,
            perm_attention_min_seq_len: None,
            attention_classes: None,
            partial_rope: None,
            kv_shared_in_global: false,
            final_logit_softcapping: None,
        }
    }

    fn rand2(rows: usize, cols: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array2<f32> {
        let normal = StandardNormal;
        Array2::from_shape_fn((rows, cols), |_| {
            <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale
        })
    }

    fn synth_weights(cfg: &DecoderConfig, rng: &mut impl rand::RngCore) -> DecoderWeights {
        let d = cfg.hidden_size;
        let q = cfg.q_dim();
        let kv = cfg.kv_dim();
        let f = cfg.intermediate_size;
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| DecoderLayerWeights {
                norm_attn: Array1::from_elem(d, 1.0),
                wq: Some(Arc::new(rand2(d, q, rng, 0.05).mapv(half::bf16::from_f32))),
                wk: Some(Arc::new(rand2(d, kv, rng, 0.05).mapv(half::bf16::from_f32))),
                wv: Some(Arc::new(rand2(d, kv, rng, 0.05).mapv(half::bf16::from_f32))),
                wo: Some(Arc::new(rand2(q, d, rng, 0.05).mapv(half::bf16::from_f32))),
                norm_ffn: Array1::from_elem(d, 1.0),
                w_gate: Some(Arc::new(rand2(d, f, rng, 0.05).mapv(half::bf16::from_f32))),
                w_up: Some(Arc::new(rand2(d, f, rng, 0.05).mapv(half::bf16::from_f32))),
                w_down: Some(Arc::new(rand2(f, d, rng, 0.05).mapv(half::bf16::from_f32))),
                q_norm: None,
                k_norm: None,
            })
            .collect();
        DecoderWeights {
            token_embedding: rand2(cfg.vocab_size, d, rng, 0.05).mapv(half::bf16::from_f32),
            final_norm: Array1::from_elem(d, 1.0),
            layers,
            model_identity: [0u8; 32],
        }
    }

    fn provision<E: GpuOffloadEngine>(weights: &DecoderWeights, cfg: &DecoderConfig, engine: &mut E) {
        for (li, layer) in weights.layers.iter().enumerate() {
            if !cfg.offload_layer(li) {
                continue;
            }
            let li16 = li as u16;
            for (kind, w) in [
                (WeightKind::Q, layer.wq.as_ref().unwrap()),
                (WeightKind::K, layer.wk.as_ref().unwrap()),
                (WeightKind::V, layer.wv.as_ref().unwrap()),
                (WeightKind::O, layer.wo.as_ref().unwrap()),
                (WeightKind::FfnGate, layer.w_gate.as_ref().unwrap()),
                (WeightKind::FfnUp, layer.w_up.as_ref().unwrap()),
                (WeightKind::FfnDown, layer.w_down.as_ref().unwrap()),
            ] {
                engine
                    .register_weight_bf16(WeightHandle::new(li16, kind), w.view())
                    .unwrap();
            }
        }
    }

    const STUB_TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": { "[UNK]": 0, "alpha": 1, "bravo": 2, "charlie": 3, "delta": 4, "echo": 5, "foxtrot": 6, "golf": 7, "hotel": 8 },
        "unk_token": "[UNK]"
      }
    }"#;

    fn stub_tokenizer() -> HfTokenizer {
        let tmp = std::env::temp_dir().join(format!(
            "gelo-reranker-r2-tok-{}-{}.json",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::write(&tmp, STUB_TOKENIZER_JSON).expect("write stub tokenizer");
        let tok = HfTokenizer::from_file(&tmp).expect("load stub tokenizer");
        let _ = std::fs::remove_file(&tmp);
        tok
    }

    /// **M1.11 R2 acceptance** — per-candidate scores from
    /// `score_candidates_batched` match per-candidate scores from
    /// serial `score_input_ids` calls to within mask round-trip f32
    /// tolerance.
    ///
    /// Mask topology differs (per-sequence A_b vs single A per
    /// candidate), so we expect ~1e-3 numerical drift but the round-
    /// trip math is correct on both paths.
    #[test]
    fn score_candidates_batched_matches_serial_score_input_ids() {
        let cfg = tiny_cfg();
        let mut rng = ChaCha20Rng::from_seed([81u8; 32]);
        let weights = synth_weights(&cfg, &mut rng);
        let rope = Arc::new(RopeTables::new(
            cfg.head_dim_value(),
            cfg.max_position_embeddings,
            cfg.rope_theta,
        ));
        let head = YesNoHead {
            yes_token_id: 5,
            no_token_id: 8,
        };

        let candidates: Vec<RerankCandidate> = vec![
            RerankCandidate {
                chunk_id: ChunkId::from("a"),
                text: "alpha bravo charlie".into(),
            },
            RerankCandidate {
                chunk_id: ChunkId::from("b"),
                text: "delta echo foxtrot golf".into(),
            },
            RerankCandidate {
                chunk_id: ChunkId::from("c"),
                text: "hotel charlie".into(),
            },
        ];
        let query = "alpha echo";

        // Run 1: serial score_input_ids per candidate (legacy semantic).
        let mut serial_engine = RayonCpuEngine::new();
        provision(&weights, &cfg, &mut serial_engine);
        let serial_exec = InProcessTrustedExecutor::with_seed(
            serial_engine,
            MaskSeed::from_bytes([82u8; 32]),
        );
        let mut serial_svc = CausalDiscriminatorRerankService::new(
            cfg.clone(),
            stub_tokenizer(),
            weights.clone(),
            rope.clone(),
            head,
            serial_exec,
        )
        .unwrap();
        let serial_scores: Vec<f32> = candidates
            .iter()
            .map(|c| serial_svc.score_pair(query, &c.text).unwrap())
            .collect();

        // Run 2: score_candidates_batched on the same inputs.
        let mut batched_engine = RayonCpuEngine::new();
        provision(&weights, &cfg, &mut batched_engine);
        let batched_exec = InProcessTrustedExecutor::with_seed(
            batched_engine,
            MaskSeed::from_bytes([83u8; 32]),
        );
        let mut batched_svc = CausalDiscriminatorRerankService::new(
            cfg,
            stub_tokenizer(),
            weights,
            rope,
            head,
            batched_exec,
        )
        .unwrap();
        let req = RerankRequest {
            query,
            candidates: &candidates,
            top_k: 3,
            k_max: 3,
            query_id: QueryId::from("q-r2"),
        };
        let batched_scored = score_candidates_batched(&mut batched_svc, &req).unwrap();

        assert_eq!(batched_scored.len(), serial_scores.len());

        for (b, (cand, batched_item)) in candidates.iter().zip(batched_scored.iter()).enumerate() {
            // Sanity: chunk_id is preserved.
            assert_eq!(batched_item.chunk_id, cand.chunk_id);
            // Score parity: serial and batched should agree to f32
            // mask round-trip floor (~1e-3, matching the existing
            // masked-vs-plaintext tolerance).
            let diff = (serial_scores[b] - batched_item.score).abs();
            assert!(
                diff < 5e-3,
                "b={b} ({}): serial={:.6} batched={:.6} delta={:.6e}",
                cand.chunk_id.0,
                serial_scores[b],
                batched_item.score,
                diff,
            );
            // Both scores must be valid probabilities.
            assert!(
                serial_scores[b] >= 0.0 && serial_scores[b] <= 1.0,
                "serial[b={b}] out of [0,1]: {}",
                serial_scores[b]
            );
            assert!(
                batched_item.score >= 0.0 && batched_item.score <= 1.0,
                "batched[b={b}] out of [0,1]: {}",
                batched_item.score
            );
        }
    }
}
