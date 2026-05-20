use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use hf_hub::api::sync::ApiBuilder;

use gelo_protocol::{TrustedExecutor, WeightHandle, WeightKind};
use rag_core::Embedder;

use super::config::DecoderConfig;
use super::forward;
use super::rope::RopeTables;
use super::weights::DecoderWeights;
use crate::common::pool;
use crate::common::tokenizer::HfTokenizer;

/// Qwen3-class decoder-LLM-as-embedder driven through a GELO `TrustedExecutor`.
///
/// Pooling: last-token + L2 normalize (matches Qwen3-Embedding / E5-Mistral
/// convention).
pub struct GeloQwenEmbedder<X: TrustedExecutor> {
    cfg: DecoderConfig,
    tokenizer: HfTokenizer,
    weights: Arc<DecoderWeights>,
    rope: Arc<RopeTables>,
    exec: X,
    max_len: usize,
    /// Hex-encoded `sha256(concat of all shard bytes)`. Stored as UTF-8 so it
    /// rides through `AttestationEvidence::model_identity` (a `String`); the
    /// relying party recomputes the same hash over the expected weights and
    /// compares.
    model_identity: String,
}

impl<X: TrustedExecutor> GeloQwenEmbedder<X> {
    pub fn new(
        cfg: DecoderConfig,
        tokenizer: HfTokenizer,
        weights: Arc<DecoderWeights>,
        rope: Arc<RopeTables>,
        mut exec: X,
    ) -> Result<Self> {
        for (li, layer) in weights.layers.iter().enumerate() {
            if !cfg.offload_layer(li) {
                continue;
            }
            let li16 = li as u16;
            // Standard offload weights — same handles as BERT.
            exec.provision_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view())?;
            exec.provision_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view())?;
            exec.provision_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view())?;
            exec.provision_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view())?;
            // SwiGLU has three matmuls: gate, up, down.
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
        let _ = rope.head_dim(); // silence "unused field" if dead-code path triggers
        let model_identity = hex::encode(weights.model_identity);
        Ok(Self {
            cfg,
            tokenizer,
            weights,
            rope,
            exec,
            max_len,
            model_identity,
        })
    }

    /// Download from the HuggingFace hub (uses local cache when present).
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

        let shard_paths = find_safetensors_shards(&repo)?;
        Self::from_local(&config_path, &tokenizer_path, &shard_paths, exec)
    }

    /// Build from local files. `safetensors_paths` may be a single-file or a
    /// sharded list (`model-00001-of-NNNNN.safetensors`).
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

        Self::new(cfg, tokenizer, weights, rope, exec)
    }

    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len.min(self.cfg.max_position_embeddings);
        self
    }

    /// Master switch for TwinShield OutAttnMult on the attention `Q · Kᵀ`
    /// matmul. Default: enabled (subject to the length auto-switch — see
    /// [`Self::with_out_attn_mult_min_seq_len`]).
    pub fn with_out_attn_mult(mut self, enabled: bool) -> Self {
        self.cfg.use_out_attn_mult = enabled;
        self
    }

    /// Override the auto-switch threshold (`out_attn_mult_min_seq_len`).
    /// Pass `Some(n)` to force OutAttnMult only at sequence length ≥ `n`,
    /// or `None` to restore the auto default (= `hidden_size`).
    ///
    /// Common settings:
    /// - `Some(0)`         — always on (when the master switch is true).
    /// - `Some(usize::MAX)` — never on, even with master switch true.
    /// - `None`            — auto: on at `n ≥ hidden_size`.
    pub fn with_out_attn_mult_min_seq_len(mut self, min_seq_len: Option<usize>) -> Self {
        self.cfg.out_attn_mult_min_seq_len = min_seq_len;
        self
    }

    /// Master switch for Tier 1 permutation-shielded attention. Default
    /// off; opt in to engage the path between
    /// `perm_attention_min_seq_len` and `out_attn_mult_min_seq_len`.
    pub fn with_perm_attention(mut self, enabled: bool) -> Self {
        self.cfg.use_perm_attention = enabled;
        self
    }

    /// Override the permuted-attention threshold. `None` resolves to 64.
    pub fn with_perm_attention_min_seq_len(mut self, min_seq_len: Option<usize>) -> Self {
        self.cfg.perm_attention_min_seq_len = min_seq_len;
        self
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    /// Borrow the shared weight bundle so additional embedders can be
    /// constructed without re-loading from disk / HF Hub.
    pub fn weights_arc(&self) -> Arc<DecoderWeights> {
        Arc::clone(&self.weights)
    }

    /// Borrow the precomputed RoPE cos/sin tables likewise.
    pub fn rope_arc(&self) -> Arc<RopeTables> {
        Arc::clone(&self.rope)
    }

    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }
}

fn find_safetensors_shards(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    // Try the unsharded file first.
    if let Ok(p) = repo.get("model.safetensors") {
        return Ok(vec![p]);
    }
    // Fall back to the shard index.
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("model neither has model.safetensors nor model.safetensors.index.json")?;
    let index_bytes =
        std::fs::read(&index_path).with_context(|| format!("reading {}", index_path.display()))?;
    let index: serde_json::Value =
        serde_json::from_slice(&index_bytes).context("parsing shard index json")?;
    let map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("shard index has no weight_map object"))?;
    let mut filenames: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    filenames.sort();
    filenames.dedup();
    let mut paths = Vec::with_capacity(filenames.len());
    for name in filenames {
        let p = repo
            .get(&name)
            .with_context(|| format!("downloading shard {name}"))?;
        paths.push(p);
    }
    Ok(paths)
}

impl<X: TrustedExecutor + Clone + Send + Sync> Embedder for GeloQwenEmbedder<X> {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        // Single-text fast path: skip the rayon scope + executor clone.
        // Online-query latency stays identical to the pre-parallel build.
        if texts.len() <= 1 {
            return texts.iter().map(|t| self.embed_one(t, &mut self.exec.clone())).collect();
        }

        // Bulk-ingest path: parallel fan-out via rayon. Each worker gets a
        // freshly-cloned executor whose RNG is moved to its own ChaCha20
        // stream (worker_idx), so the per-text mask `A` differs across
        // workers — no shared-A leak across the batch (see
        // `docs/prototype/future-rnd.md` §5 "Shared-A multi-text
        // batching"). Engine clones share the Arc-backed weight cache,
        // so no weight duplication.
        //
        // Caller is responsible for setting `BLIS_NUM_THREADS=1` if the
        // `blas` feature is on; with BLIS_NUM_THREADS=16 + rayon 16, the
        // 256-way thread oversubscription thrashes more than it helps.
        use rayon::prelude::*;
        texts
            .par_iter()
            .enumerate()
            .map(|(idx, text)| {
                let mut exec = self.exec.clone();
                // Move each worker's RNG to its own ChaCha20 stream so
                // the per-text mask `A` differs across the batch. Without
                // this, all workers would inherit identical RNG state
                // from the clone and sample identical `A` — the
                // shared-A leak that future-rnd.md §5 calls out.
                exec.set_rng_stream(idx as u64);
                self.embed_one(text, &mut exec)
            })
            .collect()
    }

    fn model_identity(&self) -> &[u8] {
        self.model_identity.as_bytes()
    }
}

impl<X: TrustedExecutor + Clone + Send + Sync> GeloQwenEmbedder<X> {
    /// Embed a single text against a caller-supplied executor. Factored
    /// out of `embed` so the parallel path can hand each rayon worker its
    /// own cloned executor without touching `self.exec`.
    ///
    /// `&self` (not `&mut`) because all model state (`cfg`, `tokenizer`,
    /// `weights`, `rope`) is read-only or `Arc`-shared. The mutable bits
    /// (executor session mask + RNG) live on the caller's `exec` argument.
    fn embed_one(&self, text: &str, exec: &mut X) -> anyhow::Result<Vec<f32>> {
        let ids = self.tokenizer.encode(text, self.max_len)?;
        let hidden = forward::run(&self.cfg, &self.weights, &self.rope, exec, &ids)?;
        let pooled = pool::last_l2(hidden.view());
        Ok(pooled.to_vec())
    }
}
