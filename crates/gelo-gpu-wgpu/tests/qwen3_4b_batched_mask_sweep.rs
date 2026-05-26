//! Qwen3-4B batched-decode context-size sweep across mask families.
//!
//! Compares three mask configurations end-to-end on real Qwen3-4B
//! weights:
//!
//!   - **Haar** — dense Householder QR mask (paper-parity baseline).
//!   - **Auto** — `MaskKind::Auto`: HD₃ at pow2-aligned stacked_n,
//!     DCT-IV at non-pow2 (the historical Auto-dispatch path).
//!   - **HD₃ (shield-to-pow2)** — `MaskKind::Hd3` forced. At
//!     non-pow2 single-stream prefill this internally zero-pads the
//!     stacked operand to `next_pow2(n + k_shield)`. Equivalent
//!     compute to the documented "shield-to-pow2" strategy: the
//!     mask GEMM cost is dominated by FWHT at `s_pad`, so whether
//!     the pad rows are zeros or shield Gaussians shifts only the
//!     shield_stack CPU bucket (a few percent of wall).
//!
//! Each config runs `forward::run_prefill_batched` + K decode steps
//! via `forward::run_decode_step_batched` at B=2 sequences. n_prompt
//! sweeps {512, 1024, 2048}. Reports TTFT (prefill wall), TPOT
//! (decode step wall), and per-bucket profile breakdown.
//!
//! Run:
//!
//! ```text
//! cargo test -p gelo-gpu-wgpu --release \
//!     --test qwen3_4b_batched_mask_sweep \
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
use gelo_protocol::{InProcessTrustedExecutor, MaskKind, TrustedExecutor};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};

const VARIANT: Qwen3Variant = Qwen3Variant::Q4B;
const BATCH_SIZE: usize = 8;
const DECODE_STEPS: usize = 4;
const DEFAULT_PROMPT_LENGTHS: &[usize] = &[2040];

fn prompt_lengths_from_env() -> Vec<usize> {
    match std::env::var("GELO_BENCH_LENGTHS") {
        Ok(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => DEFAULT_PROMPT_LENGTHS.to_vec(),
    }
}

const LONG_TEXT_SEED: &str = "The quick brown fox jumps over the lazy dog. \
    Confidential computing keeps the prompt private inside an attested CVM. \
    Rotary position embeddings rotate query and key vectors per position. \
    Grouped-query attention shares one KV head across several Q heads. \
    The trusted executor samples a fresh Haar mask for every forward pass. \
    SwiGLU activation uses a sigmoid-weighted linear gate on the up branch. \
    RMSNorm normalises by the root-mean-square of the activation row. \
    The KV cache grows by one position per decode step and lives in CVM DRAM. \
    Attestation reports bind the model identity to the SEV-SNP key. ";

fn rss_bytes() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<usize>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

fn fmt_gib(b: usize) -> String {
    format!("{:.2} GiB", b as f64 / 1024.0 / 1024.0 / 1024.0)
}

fn load_pretrained()
-> Result<(DecoderConfig, HfTokenizer, Arc<DecoderWeights>, Arc<RopeTables>)> {
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

fn find_safetensors_shards(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    if let Ok(p) = repo.get("model.safetensors") {
        return Ok(vec![p]);
    }
    let index_path = repo.get("model.safetensors.index.json")?;
    let bytes = std::fs::read(&index_path)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    let map = v
        .get("weight_map")
        .and_then(|x| x.as_object())
        .ok_or_else(|| anyhow!("shard index has no weight_map object"))?;
    let mut filenames: Vec<String> = map
        .values()
        .filter_map(|v: &serde_json::Value| v.as_str().map(|s| s.to_string()))
        .collect();
    filenames.sort();
    filenames.dedup();
    let mut paths = Vec::with_capacity(filenames.len());
    for name in filenames {
        paths.push(repo.get(&name)?);
    }
    Ok(paths)
}

fn provision_decoder_weights<X: TrustedExecutor>(
    cfg: &DecoderConfig,
    weights: &Arc<DecoderWeights>,
    exec: &mut X,
) -> Result<()> {
    provision_into_shared(weights.as_ref(), cfg, exec)
}

fn build_prompt_ids(tokenizer: &HfTokenizer, target_tokens: usize) -> Result<Vec<u32>> {
    let reps = (target_tokens / 30).max(1) + 1;
    let text = LONG_TEXT_SEED.repeat(reps);
    let ids = tokenizer.encode(&text, target_tokens)?;
    if ids.len() < target_tokens {
        return Err(anyhow!(
            "tokeniser returned {} tokens, expected {}",
            ids.len(),
            target_tokens,
        ));
    }
    Ok(ids)
}

#[derive(Debug, Clone, Copy)]
enum MaskConfig {
    Haar,
    Auto,
    Hd3ShieldToPow2,
}

impl MaskConfig {
    fn label(&self) -> &'static str {
        match self {
            Self::Haar => "Haar",
            Self::Auto => "Auto",
            Self::Hd3ShieldToPow2 => "HD₃ (shield-to-pow2)",
        }
    }
    fn apply<E: gelo_protocol::GpuOffloadEngine>(
        &self,
        exec: InProcessTrustedExecutor<E>,
    ) -> InProcessTrustedExecutor<E> {
        match self {
            Self::Haar => exec.with_haar_mask(),
            Self::Auto => exec.with_auto_mask(),
            Self::Hd3ShieldToPow2 => exec.with_hd3_mask(),
        }
    }
    fn mask_kind(&self) -> MaskKind {
        match self {
            Self::Haar => MaskKind::Haar,
            Self::Auto => MaskKind::Auto,
            Self::Hd3ShieldToPow2 => MaskKind::Hd3,
        }
    }
}

#[derive(Debug, Clone)]
struct CellTiming {
    mask: MaskConfig,
    n_prompt: usize,
    ttft: Duration,
    decode_steps: Vec<Duration>,
    prefill_profile: profile::Profile,
    decode_profile: profile::Profile,
}

impl CellTiming {
    fn decode_mean_ms(&self) -> f64 {
        if self.decode_steps.is_empty() {
            return 0.0;
        }
        self.decode_steps.iter().map(|d| d.as_secs_f64()).sum::<f64>() * 1000.0
            / self.decode_steps.len() as f64
    }
    fn per_seq_total_ms(&self) -> f64 {
        // batched TTFT covers B prompts in parallel; report per-sequence
        // wall by dividing by B for fair comparison vs serial.
        let total = self.ttft + self.decode_steps.iter().copied().sum::<Duration>();
        total.as_secs_f64() * 1000.0 / BATCH_SIZE as f64
    }
}

fn time_batched_generate<E: gelo_protocol::GpuOffloadEngine>(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut InProcessTrustedExecutor<E>,
    prompts: &[Vec<u32>],
    decode_steps: usize,
    mask: MaskConfig,
) -> Result<CellTiming> {
    let n_prompt = prompts[0].len();
    let max_cache_len = n_prompt + decode_steps + 1;

    let mut kv_cache = KvCache::new_batched(
        prompts.len(),
        weights.layers.len(),
        max_cache_len,
        cfg.kv_dim(),
    );

    profile::reset();
    let t_prefill = Instant::now();
    let _hidden = forward::run_prefill_batched(cfg, weights, rope, exec, prompts, &mut kv_cache)?;
    let ttft = t_prefill.elapsed();
    let prefill_profile = profile::snapshot();

    // For decode, feed token 0 every step (we don't sample). The
    // bench measures protocol overhead, not generation quality.
    let next_tokens: Vec<u32> = vec![0u32; prompts.len()];
    let mut step_times = Vec::with_capacity(decode_steps);
    profile::reset();
    for _ in 0..decode_steps {
        let t = Instant::now();
        let _ = forward::run_decode_step_batched(
            cfg,
            weights,
            rope,
            exec,
            &next_tokens,
            &mut kv_cache,
        )?;
        step_times.push(t.elapsed());
    }
    let decode_profile = profile::snapshot();

    Ok(CellTiming {
        mask,
        n_prompt,
        ttft,
        decode_steps: step_times,
        prefill_profile,
        decode_profile,
    })
}

#[test]
#[ignore = "loads Qwen3-4B (~14 GB weights); ~8-15 min wall-clock"]
fn qwen3_4b_batched_mask_sweep() -> Result<()> {
    let prompt_lengths = prompt_lengths_from_env();
    eprintln!("RSS at start: {}", fmt_gib(rss_bytes()));
    eprintln!(
        "Qwen3-4B batched mask sweep — B={BATCH_SIZE} K={DECODE_STEPS} lengths={prompt_lengths:?}",
    );

    let (cfg, tokenizer, weights, rope) = load_pretrained()?;
    eprintln!("RSS after weights load: {}", fmt_gib(rss_bytes()));

    let gpu_root = WgpuVulkanEngine::new().context("Vulkan adapter")?;
    let adapter_line = format!(
        "{} ({:?})",
        gpu_root.adapter_info().name,
        gpu_root.adapter_info().device_type
    );
    assert!(gpu_root.is_real_gpu(), "bench needs real GPU hardware");
    eprintln!("Vulkan: {adapter_line}");

    // Build B identical prompts per target length (same content; what
    // we're measuring is mask + GPU + attention scaling, not
    // tokenisation noise).
    let prompts_per_length: Vec<Vec<Vec<u32>>> = prompt_lengths
        .iter()
        .map(|&n| {
            let ids = build_prompt_ids(&tokenizer, n)?;
            Ok::<Vec<Vec<u32>>, anyhow::Error>(vec![ids; BATCH_SIZE])
        })
        .collect::<Result<_>>()?;

    // Build one executor at a time per mask config — keeping three
    // alive simultaneously on the iGPU caused command-submission
    // OOM at warmup. Each loop iteration: build + provision + warm +
    // measure across all lengths + drop. Weight upload happens 3×
    // total (no clone-shared dedupe across drop/rebuild), but RSS
    // stays bounded.
    // B=8 @ n=2040 — pow2-aligned (s = n+k = 2048, no HD₃ pad). Auto
    // and HD₃ should converge here; the comparison validates that
    // shield-to-pow2 is on-par with Auto when the operand already
    // lands on pow2. Haar skipped — pre-validated as 2-3× slower
    // and unaffected by the M1.11 batching win at this shape.
    let configs = [MaskConfig::Auto, MaskConfig::Hd3ShieldToPow2];
    let mut results: Vec<CellTiming> = Vec::new();

    for (i, mc) in configs.iter().enumerate() {
        eprintln!();
        eprintln!("== {} ==", mc.label());
        let seed = MaskSeed::from_bytes([(0x70 + i as u8); 32]);
        let mut e = InProcessTrustedExecutor::with_seed(gpu_root.clone_shared(), seed);
        e = mc.apply(e);
        eprintln!(
            "[{}] mask_kind={:?}; provisioning Qwen3-4B weights...",
            mc.label(),
            mc.mask_kind(),
        );
        provision_decoder_weights(&cfg, &weights, &mut e)?;
        eprintln!("  RSS after provision: {}", fmt_gib(rss_bytes()));

        // Warm at the shortest length.
        let warm_prompts = &prompts_per_length[0];
        eprintln!("  [warm] one batched generate at n={}...", warm_prompts[0].len());
        let _ = time_batched_generate(&cfg, &weights, &rope, &mut e, warm_prompts, 2, *mc)?;

        // Measured runs across all lengths.
        for (n_target, prompts) in prompt_lengths.iter().zip(prompts_per_length.iter()) {
            let r = time_batched_generate(&cfg, &weights, &rope, &mut e, prompts, DECODE_STEPS, *mc)?;
            eprintln!(
                "  n={n_target}: TTFT {:.0} ms · TPOT {:.1} ms · per-seq total {:.0} ms",
                r.ttft.as_secs_f64() * 1000.0,
                r.decode_mean_ms(),
                r.per_seq_total_ms(),
            );
            results.push(r);
        }

        // Drop the executor to release GPU resources before the next
        // mask config's provisioning.
        drop(e);
        eprintln!("  RSS after drop: {}", fmt_gib(rss_bytes()));
    }

    eprintln!();
    eprintln!("{}", "=".repeat(96));
    eprintln!(
        "Qwen3-4B batched mask sweep · {adapter_line} · B={BATCH_SIZE}, K={DECODE_STEPS}"
    );
    eprintln!("{}", "=".repeat(96));
    eprintln!(
        "{:>6}  {:24}  {:>10}  {:>10}  {:>14}",
        "n", "mask", "TTFT (ms)", "TPOT (ms)", "per-seq tot ms"
    );
    eprintln!("{}", "-".repeat(96));
    for r in &results {
        eprintln!(
            "{:>6}  {:24}  {:>10.0}  {:>10.1}  {:>14.0}",
            r.n_prompt,
            r.mask.label(),
            r.ttft.as_secs_f64() * 1000.0,
            r.decode_mean_ms(),
            r.per_seq_total_ms(),
        );
    }
    eprintln!("{}", "-".repeat(96));

    // Per-cell profile dumps — prefill + decode separately so the
    // mask GEMM (dominant at prefill) vs attention scaling (dominant
    // at decode) buckets are legible.
    for r in &results {
        r.prefill_profile.dump(&format!(
            "{} n={} prefill (TTFT {:.0} ms)",
            r.mask.label(),
            r.n_prompt,
            r.ttft.as_secs_f64() * 1000.0,
        ));
        r.decode_profile.dump(&format!(
            "{} n={} decode (TPOT {:.1} ms × {} steps)",
            r.mask.label(),
            r.n_prompt,
            r.decode_mean_ms(),
            r.decode_steps.len(),
        ));
    }

    Ok(())
}
