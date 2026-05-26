//! **R4 async overlap microbench (minimal).**
//!
//! Loads Qwen3-1.7B *once*, then runs the same prefill harness under
//! `GELO_ASYNC_OFFLOAD=0` and `GELO_ASYNC_OFFLOAD=1`, comparing wall
//! time + profile-bucket attribution. Smaller shape than the full
//! `qwen3_4b_batched_mask_sweep` so a single comparison run is
//! < 2 minutes wall, including model load.
//!
//! What this measures: whether the substrate's async dispatch path
//! delivers any wall savings vs the legacy sync path under the strict
//! serial-dependency chain of one forward pass. Expectation per
//! `docs/plans/m1-12-r4-async-overlap.md` §4 risk #1 is "<1% wall on
//! iGPU" — the bench will either confirm that and inform the cutover
//! decision, or surprise us with measurable overlap we didn't predict.
//!
//! Run:
//!
//! ```text
//! cargo test -p gelo-gpu-wgpu --release \
//!     --test r4_async_minibench \
//!     -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use gelo_embedder::common::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::kv_cache::KvCache;
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderWeights, provision_into_shared};
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::profile;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, TrustedExecutor};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};

const VARIANT: Qwen3Variant = Qwen3Variant::Q1_7B;
const BATCH_SIZE: usize = 2;
const N_PROMPT: usize = 256;
const N_WARM: usize = 1;
const N_MEASURE: usize = 3;
const ENV_VAR: &str = "GELO_ASYNC_OFFLOAD";

fn find_safetensors_shards(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    if let Ok(p) = repo.get("model.safetensors") {
        return Ok(vec![p]);
    }
    let index_path = repo.get("model.safetensors.index.json")?;
    let bytes = std::fs::read(&index_path)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    let map = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| anyhow!("index.json: missing weight_map"))?;
    let mut shards: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for val in map.values() {
        if let Some(s) = val.as_str() {
            shards.insert(s.to_string());
        }
    }
    shards
        .into_iter()
        .map(|name| repo.get(&name).map_err(|e| anyhow!("shard {name}: {e}")))
        .collect()
}

fn load_pretrained() -> Result<(DecoderConfig, HfTokenizer, Arc<DecoderWeights>, Arc<RopeTables>)>
{
    let cfg = VARIANT.config();
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("HF hub API client")?;
    let repo = api.model(VARIANT.hf_model_id().to_string());
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;
    let shard_paths = find_safetensors_shards(&repo)?;
    let shard_refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_path()).collect();
    let weights = Arc::new(DecoderWeights::from_safetensors(&shard_refs, &cfg)?);
    let rope = Arc::new(RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    ));
    Ok((cfg, tokenizer, weights, rope))
}

fn build_prompt_ids(tokenizer: &HfTokenizer, target_tokens: usize) -> Result<Vec<u32>> {
    let seed = "The quick brown fox jumps over the lazy dog. \
        Confidential computing keeps the prompt private inside an attested CVM. \
        Rotary position embeddings rotate query and key vectors per position. \
        Grouped-query attention shares one KV head across several Q heads. \
        The trusted executor samples a fresh Haar mask for every forward pass. ";
    let mut text = String::new();
    while {
        let ids = tokenizer
            .encode(&text, target_tokens * 4)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        ids.len() < target_tokens
    } {
        text.push_str(seed);
    }
    let mut ids = tokenizer
        .encode(&text, target_tokens * 4)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    ids.truncate(target_tokens);
    Ok(ids)
}

struct CellTiming {
    label: &'static str,
    walls: Vec<Duration>,
    prefill_profile: profile::Profile,
}

impl CellTiming {
    fn mean_ms(&self) -> f64 {
        let s: f64 = self.walls.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
        s / self.walls.len() as f64
    }

    fn std_ms(&self) -> f64 {
        let mean = self.mean_ms();
        let var: f64 = self
            .walls
            .iter()
            .map(|d| (d.as_secs_f64() * 1000.0 - mean).powi(2))
            .sum::<f64>()
            / self.walls.len() as f64;
        var.sqrt()
    }
}

fn run_condition(
    label: &'static str,
    cfg: &DecoderConfig,
    weights: &Arc<DecoderWeights>,
    rope: &Arc<RopeTables>,
    engine_root: &WgpuVulkanEngine,
    prompts: &[Vec<u32>],
) -> Result<CellTiming> {
    let seed = MaskSeed::from_bytes([0x91u8; 32]);
    let mut exec = InProcessTrustedExecutor::with_seed(engine_root.clone_shared(), seed);
    provision_into_shared(weights, cfg, &mut exec)?;

    // Warm.
    for _ in 0..N_WARM {
        let mut kv = KvCache::new_batched(
            BATCH_SIZE,
            weights.layers.len(),
            N_PROMPT + 4,
            cfg.kv_dim(),
        );
        let _ = forward::run_prefill_batched(cfg, weights, rope, &mut exec, prompts, &mut kv)?;
    }

    // Measure.
    profile::reset_all();
    let mut walls = Vec::with_capacity(N_MEASURE);
    for _ in 0..N_MEASURE {
        let mut kv = KvCache::new_batched(
            BATCH_SIZE,
            weights.layers.len(),
            N_PROMPT + 4,
            cfg.kv_dim(),
        );
        let t0 = Instant::now();
        let _ = forward::run_prefill_batched(cfg, weights, rope, &mut exec, prompts, &mut kv)?;
        walls.push(t0.elapsed());
    }
    profile::aggregate_threads();
    let prefill_profile = profile::snapshot();

    Ok(CellTiming {
        label,
        walls,
        prefill_profile,
    })
}

#[test]
#[ignore = "loads Qwen3-1.7B (~3.4 GB weights); ~1-2 min wall-clock"]
fn r4_async_minibench() -> Result<()> {
    eprintln!(
        "R4 async minibench — Qwen3-1.7B B={BATCH_SIZE} n={N_PROMPT} (warm={N_WARM} measure={N_MEASURE})"
    );
    let (cfg, tokenizer, weights, rope) = load_pretrained()?;
    let gpu_root = WgpuVulkanEngine::new().context("Vulkan adapter")?;
    eprintln!(
        "Vulkan: {} ({:?})",
        gpu_root.adapter_info().name,
        gpu_root.adapter_info().device_type
    );

    let ids = build_prompt_ids(&tokenizer, N_PROMPT)?;
    let prompts: Vec<Vec<u32>> = vec![ids; BATCH_SIZE];

    // SAFETY: single-threaded section; no other thread touches env vars.
    unsafe { std::env::remove_var(ENV_VAR) };
    let sync = run_condition("sync (env unset)", &cfg, &weights, &rope, &gpu_root, &prompts)?;

    unsafe { std::env::set_var(ENV_VAR, "1") };
    let async_r = run_condition("async (env=1)", &cfg, &weights, &rope, &gpu_root, &prompts)?;
    unsafe { std::env::remove_var(ENV_VAR) };

    // Summary
    eprintln!();
    eprintln!("{}", "=".repeat(72));
    eprintln!(
        "{:24}  {:>12}  {:>12}  {:>10}",
        "condition", "mean (ms)", "std (ms)", "n"
    );
    eprintln!("{}", "-".repeat(72));
    for c in [&sync, &async_r] {
        eprintln!(
            "{:24}  {:>12.1}  {:>12.1}  {:>10}",
            c.label,
            c.mean_ms(),
            c.std_ms(),
            c.walls.len(),
        );
    }
    eprintln!("{}", "-".repeat(72));
    let delta_ms = async_r.mean_ms() - sync.mean_ms();
    let delta_pct = 100.0 * delta_ms / sync.mean_ms();
    eprintln!(
        "delta async-sync: {:+.1} ms ({:+.2}%)  (negative = async faster)",
        delta_ms, delta_pct
    );

    eprintln!();
    sync.prefill_profile.dump("sync profile (N_MEASURE runs aggregated)");
    async_r.prefill_profile.dump("async profile (N_MEASURE runs aggregated)");

    Ok(())
}
