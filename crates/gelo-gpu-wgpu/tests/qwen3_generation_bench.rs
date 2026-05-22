//! Qwen3-1.7B autoregressive generation bench on Vulkan GPU.
//!
//! Reports TTFT (time-to-first-token = prefill duration) and TPOT
//! (time-per-output-token = decode-step mean / median / stddev) across
//! the three GPU protocol cells, matching the embedder-side
//! `qwen3_overhead_bench` cell taxonomy:
//!
//!   1. **gpu_plain** — `PlaintextExecutor` + Vulkan engine. No GELO,
//!      no shield, no U-Verify. Baseline.
//!   2. **gpu_gelo** — `InProcessTrustedExecutor::with_seed` paper-parity
//!      defaults: per-forward Haar mask + shield(8, 4.0). Privacy.
//!   3. **gpu_full_stack** — `gpu_gelo` + `with_verify_probes(k=2)`.
//!      Privacy + integrity (Freivalds-style attestation of each
//!      offloaded matmul). `k = 2` chosen to keep TEE-side CPU cost
//!      tractable in a bench; production deployments run `k = 8`
//!      (≈2.4e-7 undetected-tamper soundness) for the same protocol
//!      cost on the offload side.
//!
//! Workload: greedy `generate(max_tokens = 8)` on a fixed 4-token
//! prompt ("The quick brown fox"). Greedy keeps the output
//! deterministic — argmax robustness ensures gpu_plain and the masked
//! cells emit byte-identical token sequences when the protocol works.
//!
//! Memory budget. Qwen3-1.7B in f32 is ~13 GB (bf16 safetensors
//! up-cast on load). Sharing rules below keep peak working set under
//! ~36 GB on a 62 GB box:
//!   - `Arc<DecoderWeights>` cloned across all three executors (no
//!     CPU-side copy).
//!   - One root `WgpuVulkanEngine`, `clone_shared()` per executor →
//!     one shared GPU weight upload across all three cells.
//!   - U-Verify TEE-side weight cache materialises only in
//!     `gpu_full_stack` (verify_probes > 0). Other two cells skip
//!     the extra allocation.
//!
//! Downloads ~3.4 GB on first run (`Qwen/Qwen3-1.7B` bf16
//! safetensors); gated behind `#[ignore]`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use gelo_embedder::common::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::generation::{GenerationConfig, SamplerConfig, generate};
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderWeights, provision_into_shared};
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, PlaintextExecutor, TrustedExecutor};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};

const VARIANT: Qwen3Variant = Qwen3Variant::Q1_7B;
const PROMPT: &str = "The quick brown fox";
const MAX_TOKENS: usize = 8;
/// Force OutAttnMult on at any sequence length so the bench cell
/// actually exercises the TwinShield 4-partition path. The auto-switch
/// threshold (`hidden_size = 2048`) would otherwise keep attention
/// entirely in-TEE at decode shapes (n = 1) and short prefill (n = 4).
const FORCE_OUTATTN_MIN_SEQ_LEN: Option<usize> = Some(0);
const VERIFY_PROBES_BENCH: usize = 2;

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

#[derive(Debug, Clone)]
struct CellTiming {
    name: &'static str,
    ttft: Duration,
    decode_steps: Vec<Duration>,
    tokens: Vec<u32>,
}

impl CellTiming {
    fn total(&self) -> Duration {
        self.ttft + self.decode_steps.iter().copied().sum::<Duration>()
    }
    fn decode_mean_ms(&self) -> f64 {
        if self.decode_steps.is_empty() {
            return 0.0;
        }
        self.decode_steps.iter().map(|d| d.as_secs_f64()).sum::<f64>() * 1000.0
            / self.decode_steps.len() as f64
    }
    fn decode_median_ms(&self) -> f64 {
        if self.decode_steps.is_empty() {
            return 0.0;
        }
        let mut v: Vec<f64> = self
            .decode_steps
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    }
    fn decode_stddev_ms(&self) -> f64 {
        let mean = self.decode_mean_ms();
        let var: f64 = self
            .decode_steps
            .iter()
            .map(|d| {
                let v = d.as_secs_f64() * 1000.0;
                (v - mean).powi(2)
            })
            .sum::<f64>()
            / self.decode_steps.len().max(1) as f64;
        var.sqrt()
    }
    fn tokens_per_sec(&self) -> f64 {
        let decode_total: f64 = self
            .decode_steps
            .iter()
            .map(|d| d.as_secs_f64())
            .sum();
        if decode_total == 0.0 {
            0.0
        } else {
            self.decode_steps.len() as f64 / decode_total
        }
    }
}

/// One timed generation pass. Calls the same prefill + decode loop as
/// `decoder::generation::generate`, but instruments TTFT and per-step
/// decode timing instead of returning only the tokens.
fn time_generate<X: TrustedExecutor>(
    name: &'static str,
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut X,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Result<CellTiming> {
    // We can't trivially split prefill / decode timing through the
    // public `generate()` entry, so reimplement the loop here with
    // explicit instants. The protocol surface is identical — same
    // begin/end forward bracket, same forward path inside.
    use gelo_embedder::decoder::forward::{run_decode_step, run_prefill};
    use gelo_embedder::decoder::kv_cache::KvCache;

    let max_cache_len = prompt_ids.len() + max_tokens;
    let mut kv_cache = KvCache::new(weights.layers.len(), max_cache_len, cfg.kv_dim());

    let t_prefill = Instant::now();
    let hidden = run_prefill(cfg, weights, rope, exec, prompt_ids, &mut kv_cache)?;
    let ttft = t_prefill.elapsed();

    let mut h_last = hidden.row(hidden.nrows() - 1).to_owned();
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut decode_steps = Vec::with_capacity(max_tokens);

    for _ in 0..max_tokens {
        // Greedy sample on the LM head (tied embeddings).
        let logits = compute_logits(cfg, weights, h_last.view());
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        tokens.push(best_idx);

        let t_step = Instant::now();
        h_last = run_decode_step(cfg, weights, rope, exec, best_idx, &mut kv_cache)?;
        decode_steps.push(t_step.elapsed());
    }

    Ok(CellTiming {
        name,
        ttft,
        decode_steps,
        tokens,
    })
}

/// LM head = `h_last · token_embedding.T` for tied-embedding models.
/// Stays in-TEE — same primitive used by `decoder::generation::generate`.
fn compute_logits(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    h_last: ndarray::ArrayView1<'_, f32>,
) -> ndarray::Array1<f32> {
    let vocab = weights.token_embedding.nrows();
    let mut logits = ndarray::Array1::<f32>::zeros(vocab);
    for v in 0..vocab {
        let row = weights.token_embedding.row(v);
        let dot: f32 = h_last.iter().zip(row.iter()).map(|(a, b)| a * b.to_f32()).sum();
        logits[v] = dot;
    }
    if let Some(cap) = cfg.final_logit_softcapping {
        let inv = 1.0_f32 / cap;
        for x in logits.iter_mut() {
            *x = (*x * inv).tanh() * cap;
        }
    }
    logits
}

fn load_pretrained()
-> Result<(DecoderConfig, HfTokenizer, Arc<DecoderWeights>, Arc<RopeTables>)> {
    let cfg = VARIANT.config();
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("building HF hub API client")?;
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
    weights: &DecoderWeights,
    exec: &mut X,
) -> Result<()> {
    provision_into_shared(weights, cfg, exec)
}

fn pct_over(c: &CellTiming, base: &CellTiming) -> String {
    if std::ptr::eq(c as *const _, base as *const _) {
        "(base)".to_string()
    } else {
        let p = 100.0 * (c.total().as_secs_f64() / base.total().as_secs_f64() - 1.0);
        format!("{:+.1}%", p)
    }
}

#[test]
#[ignore = "downloads ~3.4 GB Qwen3-1.7B; requires Vulkan GPU; ~35 GB RAM peak"]
fn qwen3_1_7b_generation_overhead_breakdown() -> Result<()> {
    eprintln!("RSS before any load: {}", fmt_gib(rss_bytes()));
    eprintln!(
        "Qwen3-1.7B generation bench — model={} prompt={:?} max_tokens={MAX_TOKENS}",
        VARIANT.hf_model_id(),
        PROMPT,
    );

    let (cfg, tokenizer, weights, rope) = load_pretrained()?;
    let prompt_ids = tokenizer.encode(PROMPT, 32)?;
    eprintln!(
        "RSS after CPU weights load (Arc<DecoderWeights>): {}",
        fmt_gib(rss_bytes())
    );

    let gpu_root = WgpuVulkanEngine::new().context("Vulkan adapter")?;
    let adapter_line = format!(
        "{} ({:?}, driver={}, info={})",
        gpu_root.adapter_info().name,
        gpu_root.adapter_info().device_type,
        gpu_root.adapter_info().driver,
        gpu_root.adapter_info().driver_info,
    );
    assert!(gpu_root.is_real_gpu(), "bench needs real GPU hardware");

    // Tweak the config per-cell. OutAttnMult forced on at any n so the
    // bench actually exercises the offload path (auto-switch at
    // hidden_size = 2048 would route every decode step in-TEE).
    let mut cfg_offload = cfg.clone();
    cfg_offload.use_out_attn_mult = true;
    cfg_offload.out_attn_mult_min_seq_len = FORCE_OUTATTN_MIN_SEQ_LEN;

    // 1. gpu_plain (no privacy baseline).
    eprintln!("[gpu_plain] provisioning weights to shared GPU engine...");
    let mut gpu_plain = PlaintextExecutor::new(gpu_root.clone_shared());
    provision_decoder_weights(&cfg_offload, &weights, &mut gpu_plain)?;
    eprintln!("RSS after gpu_plain provision: {}", fmt_gib(rss_bytes()));

    // 2. gpu_gelo (paper-parity: per-forward mask + shield(8, 4.0), no
    //    U-Verify). `with_seed` already sets `per_forward_mask = true`
    //    and `shield = ShieldConfig::new(8, 4.0)`.
    eprintln!("[gpu_gelo] provisioning (per-forward A + shield(8,4.0))...");
    let mut gpu_gelo = InProcessTrustedExecutor::with_seed(
        gpu_root.clone_shared(),
        MaskSeed::from_bytes([13u8; 32]),
    );
    provision_decoder_weights(&cfg_offload, &weights, &mut gpu_gelo)?;
    eprintln!("RSS after gpu_gelo provision: {}", fmt_gib(rss_bytes()));

    // 3. gpu_full_stack (paper-parity + U-Verify). The TEE-side weight
    //    cache that U-Verify needs is allocated lazily by
    //    `provision_weight` only when `verify_probes > 0` — so this is
    //    the cell that grows working set.
    eprintln!(
        "[gpu_full_stack] provisioning (per-forward A + shield(8,4.0) + U-Verify k={VERIFY_PROBES_BENCH})..."
    );
    let mut gpu_full_stack = InProcessTrustedExecutor::with_seed(
        gpu_root.clone_shared(),
        MaskSeed::from_bytes([17u8; 32]),
    )
    .with_verify_probes(VERIFY_PROBES_BENCH);
    provision_decoder_weights(&cfg_offload, &weights, &mut gpu_full_stack)?;
    eprintln!("RSS after gpu_full_stack provision: {}", fmt_gib(rss_bytes()));

    // Warm up — one short generate per cell so JIT shader compile,
    // first-touch page faults, and Vulkan command-buffer cache misses
    // don't poison the measured run.
    eprintln!("[warm] one untimed generate(2) per cell...");
    let warmup = GenerationConfig {
        max_tokens: 2,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
        lm_head_via_gpu_offload: false,
    };
    let _ = generate(&cfg_offload, &weights, &rope, &mut gpu_plain, &prompt_ids, &warmup)?;
    let _ = generate(&cfg_offload, &weights, &rope, &mut gpu_gelo, &prompt_ids, &warmup)?;
    let _ = generate(&cfg_offload, &weights, &rope, &mut gpu_full_stack, &prompt_ids, &warmup)?;
    eprintln!("RSS after warmup: {}", fmt_gib(rss_bytes()));

    // Measure — interleave a single timed run per cell so any
    // system-wide noise (filesystem traffic, GPU thermal throttling)
    // hits all three cells equivalently.
    eprintln!("[measure] timed generate({MAX_TOKENS}) per cell...");
    let plain = time_generate(
        "gpu_plain", &cfg_offload, &weights, &rope, &mut gpu_plain, &prompt_ids, MAX_TOKENS,
    )?;
    eprintln!("  gpu_plain: {} tok in {:.2} s", plain.tokens.len(), plain.total().as_secs_f64());
    let gelo = time_generate(
        "gpu_gelo", &cfg_offload, &weights, &rope, &mut gpu_gelo, &prompt_ids, MAX_TOKENS,
    )?;
    eprintln!("  gpu_gelo: {} tok in {:.2} s", gelo.tokens.len(), gelo.total().as_secs_f64());
    let fullstack = time_generate(
        "gpu_full_stack", &cfg_offload, &weights, &rope, &mut gpu_full_stack, &prompt_ids, MAX_TOKENS,
    )?;
    eprintln!(
        "  gpu_full_stack: {} tok in {:.2} s",
        fullstack.tokens.len(),
        fullstack.total().as_secs_f64()
    );

    // Sanity: the three cells must emit identical token sequences
    // under greedy sampling. Any divergence indicates a protocol bug,
    // not noise — argmax is robust to the float drift the Haar mask
    // and U-Verify probes introduce.
    assert_eq!(
        plain.tokens, gelo.tokens,
        "gpu_plain vs gpu_gelo token divergence: {:?} vs {:?}",
        plain.tokens, gelo.tokens,
    );
    assert_eq!(
        plain.tokens, fullstack.tokens,
        "gpu_plain vs gpu_full_stack token divergence: {:?} vs {:?}",
        plain.tokens, fullstack.tokens,
    );

    let decoded = tokenizer.decode(&plain.tokens, true).unwrap_or_default();

    let results = [&plain, &gelo, &fullstack];
    let baseline = &plain;

    eprintln!();
    eprintln!("{}", "=".repeat(118));
    eprintln!("Vulkan adapter: {adapter_line}");
    eprintln!(
        "Model: {} · prompt={:?} → {:?} ({} tokens generated, greedy, identical across cells)",
        VARIANT.hf_model_id(),
        PROMPT,
        decoded,
        plain.tokens.len(),
    );
    eprintln!("{}", "=".repeat(118));
    eprintln!(
        "{:<22} {:>11} {:>12} {:>13} {:>11} {:>14} {:>13}",
        "cell", "total (s)", "TTFT (ms)", "TPOT mean ms", "median ms", "stddev ms", "vs gpu_plain"
    );
    eprintln!("{}", "-".repeat(118));
    for r in results.iter() {
        eprintln!(
            "{:<22} {:>11.3} {:>12.1} {:>13.1} {:>11.1} {:>14.1} {:>13}",
            r.name,
            r.total().as_secs_f64(),
            r.ttft.as_secs_f64() * 1000.0,
            r.decode_mean_ms(),
            r.decode_median_ms(),
            r.decode_stddev_ms(),
            pct_over(r, baseline),
        );
    }
    eprintln!("{}", "=".repeat(118));
    eprintln!(
        "tokens/sec (decode-only): plain={:.2} gelo={:.2} full_stack={:.2}",
        plain.tokens_per_sec(),
        gelo.tokens_per_sec(),
        fullstack.tokens_per_sec(),
    );
    eprintln!(
        "Notes: OutAttnMult forced on at any n (`out_attn_mult_min_seq_len = Some(0)`) so the bench"
    );
    eprintln!(
        "       exercises the 4-partition path even at decode-shape n = 1. U-Verify k = {VERIFY_PROBES_BENCH} in this"
    );
    eprintln!(
        "       run (≈{:.2}% undetected-tamper soundness in the GELO model); production k = 8 → 2.4e-7.",
        100.0 * (1.0 / 6.0_f64).powi(VERIFY_PROBES_BENCH as i32),
    );
    eprintln!("RSS at end: {}", fmt_gib(rss_bytes()));

    Ok(())
}
