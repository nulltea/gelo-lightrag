//! Warm-loaded extraction-LLM + description-embedder handles used by
//! the `/lightrag/extract_and_build` route.
//!
//! Owned by `AppState` as `Option<ExtractionHandles>` so the runner
//! still boots when the operator didn't supply weights paths; the
//! route returns 503 in that case.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use gelo_embedder::common::tokenizer::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::generation::{
    GenerationConfig, SamplerConfig, generate as decoder_generate,
};
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_embedder::GeloQwenEmbedder;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, TrustedExecutor, WeightHandle, WeightKind,
};
use lightrag_private::extract::{
    DecoderOutput, DecoderTiming, DescriptionEmbedder, ExtractionDecoder,
};

/// Warm-loaded Qwen3-class decoder used for entity-extraction. Holds
/// the executor with all offloadable layer weights provisioned.
///
/// Generic over the offload engine `E` so callers pick the right
/// substrate: bench / production must use
/// `gelo_gpu_wgpu::WgpuVulkanEngine` per
/// `feedback_benches_use_gelo_gpu.md` +
/// `feedback_no_rayon_cpu_engine.md`. The deprecated CPU reference
/// engine is no longer permitted in new wiring.
pub struct DecoderRuntime<E: GpuOffloadEngine> {
    pub cfg: DecoderConfig,
    pub tokenizer: HfTokenizer,
    pub weights: Arc<DecoderWeights>,
    pub rope: Arc<RopeTables>,
    pub exec: InProcessTrustedExecutor<E>,
    /// EOS token ids (resolved once from the tokenizer). Empty when
    /// the tokenizer doesn't expose any of the common EOS names —
    /// generation will then run to `max_tokens` every time.
    pub eos_token_ids: Vec<u32>,
    /// Cap on prompt token count. Computed from
    /// `cfg.max_position_embeddings` minus a safety margin for the
    /// generation budget.
    pub max_prompt_tokens: usize,
}

impl<E: GpuOffloadEngine> DecoderRuntime<E> {
    /// Build a runtime from a directory containing `config.json`,
    /// `tokenizer.json`, and one or more `*.safetensors` shards.
    pub fn from_dir(dir: &Path, engine: E) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        if !cfg_path.exists() {
            return Err(anyhow!(
                "missing {} — use `from_config_and_dir` if you have a pinned \
                 config for this variant",
                cfg_path.display()
            ));
        }
        let cfg_bytes = std::fs::read(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?;
        let cfg: DecoderConfig = serde_json::from_slice(&cfg_bytes)
            .with_context(|| format!("parsing {}", cfg_path.display()))?;
        Self::from_config_and_dir(cfg, dir, engine)
    }

    /// Build with a caller-supplied `DecoderConfig`, loading
    /// `tokenizer.json` and `*.safetensors` from `dir`. Use this when
    /// the snapshot directory lacks `config.json` — e.g. an HF cache
    /// dir where only the weights + tokenizer were materialised —
    /// and pin the config via `Qwen3Variant::Q1_7B.config()` etc.
    pub fn from_config_and_dir(cfg: DecoderConfig, dir: &Path, engine: E) -> Result<Self> {
        let (tok_path, shard_paths) = discover_tokenizer_and_shards(dir)?;
        let tokenizer = HfTokenizer::from_file(&tok_path)?;
        let shard_refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_path()).collect();
        let mut weights = DecoderWeights::from_safetensors(&shard_refs, &cfg)?;
        let rope = Arc::new(RopeTables::new(
            cfg.head_dim_value(),
            cfg.max_position_embeddings,
            cfg.rope_theta,
        ));

        // Build a fresh executor and provision every offloadable layer's
        // weights via the bf16 Arc-shared path. We `take()` each Arc
        // out of `weights` — when the engine (wgpu) consumes and
        // returns, refcount → 0 and the host bytes drop. With
        // skip-first/last layers off (the default), no forward path
        // reads `layer.{wq..w_down}` ever again. See
        // `feedback_memory_efficiency_priority.md`.
        let mut exec = InProcessTrustedExecutor::new(engine);
        for li in 0..weights.layers.len() {
            if !cfg.offload_layer(li) {
                continue;
            }
            let li16 = li as u16;
            let layer = &mut weights.layers[li];
            let wq = layer.wq.take().ok_or_else(|| anyhow!("layer {li}: wq already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::Q), wq)?;
            let wk = layer.wk.take().ok_or_else(|| anyhow!("layer {li}: wk already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::K), wk)?;
            let wv = layer.wv.take().ok_or_else(|| anyhow!("layer {li}: wv already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::V), wv)?;
            let wo = layer.wo.take().ok_or_else(|| anyhow!("layer {li}: wo already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::O), wo)?;
            let w_gate = layer.w_gate.take().ok_or_else(|| anyhow!("layer {li}: w_gate already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::FfnGate), w_gate)?;
            let w_up = layer.w_up.take().ok_or_else(|| anyhow!("layer {li}: w_up already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::FfnUp), w_up)?;
            let w_down = layer.w_down.take().ok_or_else(|| anyhow!("layer {li}: w_down already taken"))?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, WeightKind::FfnDown), w_down)?;
        }
        let weights = Arc::new(weights);

        let eos_token_ids = collect_eos_token_ids(&tokenizer);
        // Reserve at least 64 tokens of headroom for sampling.
        let max_prompt_tokens = cfg.max_position_embeddings.saturating_sub(64);

        Ok(Self {
            cfg,
            tokenizer,
            weights,
            rope,
            exec,
            eos_token_ids,
            max_prompt_tokens,
        })
    }
}

impl<E: GpuOffloadEngine> ExtractionDecoder for DecoderRuntime<E> {
    fn generate_extraction(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> anyhow::Result<DecoderOutput> {
        let t = Instant::now();
        let prompt_ids = self.tokenizer.encode(prompt, self.max_prompt_tokens)?;
        let tokenize = t.elapsed();
        if prompt_ids.is_empty() {
            anyhow::bail!("tokenizer produced empty prompt id list");
        }
        let budget = max_tokens.max(1);
        let prompt_plus_budget = prompt_ids.len().saturating_add(budget);
        if prompt_plus_budget > self.cfg.max_position_embeddings {
            anyhow::bail!(
                "prompt {} + max_tokens {} exceeds model max_position_embeddings {}; \
                 reduce chunk_size or max_tokens_per_chunk",
                prompt_ids.len(),
                budget,
                self.cfg.max_position_embeddings,
            );
        }
        let gen_cfg = GenerationConfig {
            max_tokens: budget,
            eos_token_ids: self.eos_token_ids.clone(),
            sampler: SamplerConfig::Greedy,
        };
        let t = Instant::now();
        let out = decoder_generate(
            &self.cfg,
            &self.weights,
            &self.rope,
            &mut self.exec,
            &prompt_ids,
            &gen_cfg,
        )?;
        let generate_dur = t.elapsed();
        let t = Instant::now();
        let text = self.tokenizer.decode(&out.tokens, true)?;
        let decode = t.elapsed();
        Ok(DecoderOutput {
            text,
            stopped_on_eos: out.stopped_on_eos,
            timing: DecoderTiming {
                tokenize,
                generate: generate_dur,
                decode,
                prompt_tokens: prompt_ids.len(),
                output_tokens: out.tokens.len(),
            },
        })
    }
}

/// Adapter that wraps a `GeloQwenEmbedder` so the orchestrator can
/// drive it through the [`DescriptionEmbedder`] trait without
/// `lightrag-private` depending on `gelo-embedder`. Generic over the
/// offload engine; bench / production use `WgpuVulkanEngine`.
pub struct GeloDescriptionEmbedder<E: GpuOffloadEngine + Clone + Send + Sync> {
    pub inner: GeloQwenEmbedder<InProcessTrustedExecutor<E>>,
}

impl<E: GpuOffloadEngine + Clone + Send + Sync> GeloDescriptionEmbedder<E> {
    pub fn from_dir(dir: &Path, engine: E) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        if !cfg_path.exists() {
            return Err(anyhow!("missing {}", cfg_path.display()));
        }
        let (tok_path, shard_paths) = discover_tokenizer_and_shards(dir)?;
        let exec = InProcessTrustedExecutor::new(engine);
        let inner = GeloQwenEmbedder::from_local(&cfg_path, &tok_path, &shard_paths, exec)?;
        Ok(Self { inner })
    }

    pub fn dim(&self) -> usize {
        self.inner.config().hidden_size
    }
}

impl<E: GpuOffloadEngine + Clone + Send + Sync> DescriptionEmbedder
    for GeloDescriptionEmbedder<E>
{
    fn embed_batch(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        use rag_core::Embedder;
        self.inner.embed(texts)
    }
}

/// Handle bundle held inside `AppState`. Generic over the offload
/// engine so the production runner and the bench can pick different
/// substrates while sharing the route plumbing. Cloning the `Arc`s is
/// cheap; the per-request handler locks both `Mutex`es for the
/// duration of the extraction + ingest work.
pub struct ExtractionHandles<E: GpuOffloadEngine + Clone + Send + Sync> {
    pub decoder: Arc<Mutex<DecoderRuntime<E>>>,
    pub embedder: Arc<Mutex<GeloDescriptionEmbedder<E>>>,
}

impl<E: GpuOffloadEngine + Clone + Send + Sync> Clone for ExtractionHandles<E> {
    fn clone(&self) -> Self {
        Self {
            decoder: Arc::clone(&self.decoder),
            embedder: Arc::clone(&self.embedder),
        }
    }
}

/// Find `tokenizer.json` + every `*.safetensors` shard in `dir`
/// (sorted lexicographically so sharded files load in deterministic
/// order). `config.json` is *not* required here — callers that need
/// it check separately.
fn discover_tokenizer_and_shards(dir: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    if !dir.is_dir() {
        return Err(anyhow!("model dir {} is not a directory", dir.display()));
    }
    let tok = dir.join("tokenizer.json");
    if !tok.exists() {
        return Err(anyhow!("missing {}", tok.display()));
    }
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            shards.push(path);
        }
    }
    if shards.is_empty() {
        return Err(anyhow!(
            "no *.safetensors shards in {}",
            dir.display()
        ));
    }
    shards.sort();
    Ok((tok, shards))
}

/// Resolve any of the common EOS / chat-end token names exposed by
/// HuggingFace tokenizers. Missing names are silently skipped.
fn collect_eos_token_ids(tokenizer: &HfTokenizer) -> Vec<u32> {
    let candidates = [
        "<|im_end|>",       // Qwen3 chat template
        "<|endoftext|>",    // GPT-2 / Qwen base
        "</s>",             // Llama / Mistral
        "<eos>",            // Gemma
        "<|eot_id|>",       // Llama 3
    ];
    let mut out = Vec::new();
    for name in candidates {
        if let Some(id) = tokenizer.token_id(name) {
            if !out.contains(&id) {
                out.push(id);
            }
        }
    }
    out
}
