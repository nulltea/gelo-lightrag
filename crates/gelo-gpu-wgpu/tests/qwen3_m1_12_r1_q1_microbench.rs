//! **M1.12 R1 + Q#1 microbench.**
//!
//! Two measurements in one pass:
//!
//! 1. **R1 — RSS-after-provision drop.** Calls
//!    `weights::provision_into` (the consuming variant landed in M1.12
//!    R1) and prints `/proc/self/status` VmRSS before / between /
//!    after. The plan's acceptance is ~7 GiB drop on Qwen3-4B between
//!    "weights loaded" and "weights provisioned" — the host bf16
//!    Arcs are released to the wgpu engine, refcount → 0, host bytes
//!    drop.
//!
//! 2. **Q#1 — `tee:compute_logits` bucket.** Calls
//!    `generation::generate` (which threads through the
//!    `profile::time("tee:compute_logits", …)` wrapper landed in
//!    M1.12 alongside R1) and dumps the per-bucket profile snapshot.
//!    The plan's R3 threshold: if the bucket measures ≥ 10 % of decode
//!    wall, R3 (LM-head GPU offload) wins. If < 30 s/forward on the
//!    realistic budget, R3's priority drops and R4 wins better.
//!
//! Default to Qwen3-1.7B for ~3-5 min wall, ~3.4 GB download on first
//! run. Switch to Qwen3-4B (~14 GB) by setting `GELO_BENCH_VARIANT=4b`.
//! Decode budget tunable via `GELO_BENCH_MAX_TOKENS` (default 64;
//! enough to surface the compute_logits bucket but short enough to fit
//! in a few minutes).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use gelo_embedder::common::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::generation::{
    self, GenerationConfig, SamplerConfig,
};
use gelo_embedder::decoder::kv_cache::KvCache;
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{
    DecoderWeights, provision_into, provision_lm_head_into,
};
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::profile;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    InProcessTrustedExecutor, TrustedExecutor, WeightHandle, WeightKind,
};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use ndarray::Axis;

fn variant_from_env() -> Qwen3Variant {
    match std::env::var("GELO_BENCH_VARIANT").as_deref() {
        Ok("4b") | Ok("4B") | Ok("Q4B") => Qwen3Variant::Q4B,
        _ => Qwen3Variant::Q1_7B,
    }
}

fn max_tokens_from_env() -> usize {
    std::env::var("GELO_BENCH_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64)
}

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

/// Tell glibc to release freed-but-cached pages back to the kernel.
/// Without this call, the dropped bf16 weight bytes stay in the
/// allocator's free-list and never show up as an RSS reduction — even
/// though the Rust-side Arcs hit refcount 0. R1's acceptance bench
/// needs the *kernel-visible* drop (VmRSS shrinks) to compare against
/// the handoff's 9.2 → 2.2 GiB target, so we have to force the
/// allocator's hand.
fn glibc_release_freed() {
    unsafe extern "C" {
        fn malloc_trim(pad: libc::size_t) -> libc::c_int;
    }
    unsafe {
        malloc_trim(0);
    }
}

fn fmt_gib(b: usize) -> String {
    format!("{:.2} GiB", b as f64 / 1024.0 / 1024.0 / 1024.0)
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

fn load_pretrained(
    variant: Qwen3Variant,
) -> Result<(DecoderConfig, HfTokenizer, DecoderWeights, Arc<RopeTables>)> {
    let cfg = variant.config();
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("HF hub API client")?;
    let repo = api.model(variant.hf_model_id().to_string());

    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    let shard_paths = find_safetensors_shards(&repo)?;
    let shard_refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_path()).collect();
    let weights = DecoderWeights::from_safetensors(&shard_refs, &cfg)?;

    let rope = Arc::new(RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    ));

    Ok((cfg, tokenizer, weights, rope))
}

#[test]
#[ignore = "real-weight bench: loads Qwen3-1.7B/4B from HF cache; ~3-15 min wall"]
fn m1_12_r1_q1_microbench() -> Result<()> {
    let variant = variant_from_env();
    let max_tokens = max_tokens_from_env();

    eprintln!("=== M1.12 R1 + Q#1 microbench ===");
    eprintln!("variant: {:?} ({})", variant, variant.hf_model_id());
    eprintln!("max_tokens: {max_tokens}");
    eprintln!("RSS at start: {}", fmt_gib(rss_bytes()));

    // --- Load weights ---
    let t_load = Instant::now();
    let (cfg, tokenizer, weights, rope) = load_pretrained(variant)?;
    eprintln!(
        "RSS after weights load (host bf16 Arcs alive): {} (load {:.1}s)",
        fmt_gib(rss_bytes()),
        t_load.elapsed().as_secs_f64(),
    );

    // --- Build executor ---
    let engine = WgpuVulkanEngine::new_fp16().context("Vulkan adapter (fp16)")?;
    eprintln!(
        "Vulkan: {} ({:?})",
        engine.adapter_info().name,
        engine.adapter_info().device_type,
    );
    let mut exec =
        InProcessTrustedExecutor::with_seed(engine, MaskSeed::from_bytes([42u8; 32]));
    eprintln!("RSS after engine init: {}", fmt_gib(rss_bytes()));

    // --- R1: provision_into (consuming) ---
    glibc_release_freed();
    let rss_pre_provision = rss_bytes();
    let t_provision = Instant::now();
    let mut weights = weights;
    provision_into(&mut weights, &cfg, &mut exec)?;
    let provision_dur = t_provision.elapsed();
    let rss_post_provision_raw = rss_bytes();
    glibc_release_freed();
    let rss_post_provision_trimmed = rss_bytes();
    eprintln!();
    eprintln!("--- R1 — provision_into RSS delta ---");
    eprintln!("RSS pre-provision:           {}", fmt_gib(rss_pre_provision));
    eprintln!(
        "RSS post-provision (raw):    {}",
        fmt_gib(rss_post_provision_raw),
    );
    eprintln!(
        "RSS post-provision (trimmed): {}",
        fmt_gib(rss_post_provision_trimmed),
    );
    let delta_raw_gib =
        (rss_pre_provision as f64 - rss_post_provision_raw as f64) / 1024.0 / 1024.0 / 1024.0;
    let delta_trim_gib =
        (rss_pre_provision as f64 - rss_post_provision_trimmed as f64) / 1024.0 / 1024.0 / 1024.0;
    eprintln!(
        "Δ RSS (raw):     {:.2} GiB  (target on Qwen3-4B: ~7 GiB)",
        delta_raw_gib,
    );
    eprintln!(
        "Δ RSS (trimmed): {:.2} GiB  (after malloc_trim(0))",
        delta_trim_gib,
    );
    eprintln!("provision wall: {:.1}s", provision_dur.as_secs_f64());

    // --- R3: provision LM head into VRAM ---
    let t_lmh = Instant::now();
    provision_lm_head_into(&weights, &mut exec)?;
    glibc_release_freed();
    let rss_after_lmh = rss_bytes();
    eprintln!();
    eprintln!("--- R3 prep — provision_lm_head_into ---");
    eprintln!(
        "RSS after LmHead provision (trimmed): {} (provision {:.1}s)",
        fmt_gib(rss_after_lmh),
        t_lmh.elapsed().as_secs_f64(),
    );

    // --- Q#1 + R3: two generates back-to-back ---
    let prompt = "The quick brown fox";
    let prompt_ids = tokenizer.encode(prompt, 32)?;
    eprintln!();
    eprintln!(
        "--- generation::generate baseline (R3 off) vs R3 on ---",
    );
    eprintln!(
        "prompt='{prompt}' n_prompt={} max_tokens={max_tokens}",
        prompt_ids.len(),
    );

    // Phase 1: baseline (lm_head_via_gpu_offload=false). Exercises
    // the in-TEE bf16 vocab × hidden loop.
    let baseline_cfg = GenerationConfig {
        max_tokens,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
        lm_head_via_gpu_offload: false,
    };
    profile::reset();
    let t = Instant::now();
    let baseline_out =
        generation::generate(&cfg, &weights, &rope, &mut exec, &prompt_ids, &baseline_cfg)?;
    let baseline_wall = t.elapsed();
    let baseline_snap = profile::snapshot();

    // Phase 2: R3 (lm_head_via_gpu_offload=true). Routes through
    // exec.offload_linear(LmHead, …) under a per-step begin/end
    // forward-pass bracket, masking the (1, hidden) operand, GPU
    // matmul to (1+k, vocab), unmask, strip shield.
    let r3_cfg = GenerationConfig {
        lm_head_via_gpu_offload: true,
        ..baseline_cfg.clone()
    };
    profile::reset();
    let t = Instant::now();
    let r3_out =
        generation::generate(&cfg, &weights, &rope, &mut exec, &prompt_ids, &r3_cfg)?;
    let r3_wall = t.elapsed();
    let r3_snap = profile::snapshot();

    // --- Reports ---
    eprintln!();
    eprintln!(
        "baseline (R3 off): wall {:.2}s ({:.1} tok/s), tokens={:?}",
        baseline_wall.as_secs_f64(),
        baseline_out.tokens.len() as f64 / baseline_wall.as_secs_f64(),
        baseline_out.tokens,
    );
    eprintln!(
        "R3 on:             wall {:.2}s ({:.1} tok/s), tokens={:?}",
        r3_wall.as_secs_f64(),
        r3_out.tokens.len() as f64 / r3_wall.as_secs_f64(),
        r3_out.tokens,
    );

    baseline_snap.dump(&format!("{:?} baseline (R3 off) profile", variant));
    r3_snap.dump(&format!("{:?} R3 on (LM_HEAD_GPU_OFFLOAD) profile", variant));

    // Compute_logits bucket comparison.
    let bucket_ms = |snap: &profile::Profile| -> Option<(f64, u64, f64)> {
        snap.buckets.get("tee:compute_logits").map(|(d, n)| {
            let ms = d.as_secs_f64() * 1000.0;
            let share = if !snap.total().is_zero() {
                100.0 * d.as_secs_f64() / snap.total().as_secs_f64()
            } else {
                0.0
            };
            (ms, *n, share)
        })
    };
    eprintln!();
    eprintln!("--- R3 verdict ---");
    match (bucket_ms(&baseline_snap), bucket_ms(&r3_snap)) {
        (Some((ms_b, n_b, sh_b)), Some((ms_r, n_r, sh_r))) => {
            eprintln!(
                "tee:compute_logits — baseline: {:.0} ms / {n_b} calls ({:.1}%)",
                ms_b, sh_b,
            );
            eprintln!(
                "tee:compute_logits — R3 on:   {:.0} ms / {n_r} calls ({:.1}%)",
                ms_r, sh_r,
            );
            let delta = ms_b - ms_r;
            let pct = if ms_b > 0.0 { 100.0 * delta / ms_b } else { 0.0 };
            eprintln!(
                "Δ compute_logits: {:+.0} ms ({:+.1}%) under R3",
                -delta, -pct,
            );
        }
        (Some((ms_b, _, _)), None) => {
            eprintln!(
                "baseline compute_logits: {:.0} ms; R3 path has no compute_logits bucket (expected — offload bypasses the in-TEE loop)",
                ms_b,
            );
        }
        _ => eprintln!("compute_logits bucket missing in one of the snapshots"),
    }
    let wall_delta_ms = (baseline_wall.as_secs_f64() - r3_wall.as_secs_f64()) * 1000.0;
    let wall_pct = if baseline_wall.as_secs_f64() > 0.0 {
        100.0 * wall_delta_ms / (baseline_wall.as_secs_f64() * 1000.0)
    } else {
        0.0
    };
    eprintln!(
        "Δ total wall:     {:+.0} ms ({:+.1}%) under R3",
        -wall_delta_ms, -wall_pct,
    );

    // Token-set sanity check: greedy under PlaintextExecutor would be
    // byte-identical, but under masked InProcessTrustedExecutor the
    // R3 path samples extra masks per step (one per compute_logits
    // call), so its decode-step RNG state diverges from the baseline.
    // We report whether the token sequences happen to align; argmax
    // robustness on the prompt's near-deterministic continuation
    // usually keeps the first few tokens identical.
    let prefix_match = baseline_out
        .tokens
        .iter()
        .zip(r3_out.tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "token-prefix match: {} / {} (greedy under masked exec — RNG-state divergence expected after the first compute_logits offload)",
        prefix_match,
        baseline_out.tokens.len(),
    );

    Ok(())
}

// ─── Prefill / decode breakdown ────────────────────────────────────

const LONG_TEXT_SEED: &str = "The quick brown fox jumps over the lazy dog. \
    Confidential computing keeps the prompt private inside an attested CVM. \
    Rotary position embeddings rotate query and key vectors per position. \
    Grouped-query attention shares one KV head across several Q heads. \
    The trusted executor samples a fresh Haar mask for every forward pass. \
    SwiGLU activation uses a sigmoid-weighted linear gate on the up branch. \
    RMSNorm normalises by the root-mean-square of the activation row. \
    Speculative decoding proposes draft tokens that the target verifies. \
    The KV cache grows by one position per decode step and lives in CVM DRAM. \
    Attestation reports bind the model identity to the SEV-SNP key. ";

fn build_prompt_ids(tokenizer: &HfTokenizer, target_tokens: usize) -> Result<Vec<u32>> {
    let reps = (target_tokens / 30).max(1) + 1;
    let text = LONG_TEXT_SEED.repeat(reps);
    let ids = tokenizer.encode(&text, target_tokens)?;
    if ids.len() < target_tokens {
        return Err(anyhow!(
            "tokeniser returned {} tokens, expected {}",
            ids.len(),
            target_tokens
        ));
    }
    Ok(ids)
}

fn prompt_size_from_env() -> usize {
    std::env::var("GELO_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048)
}

/// Replica of the in-TEE `compute_logits` loop, wrapped in the same
/// `profile::time("tee:compute_logits", …)` bucket the library uses.
/// Inlined here so the bench can drive prefill and decode loops
/// itself and split profile snapshots cleanly at the phase boundary.
fn bench_compute_logits_in_tee(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    h_last: ndarray::ArrayView1<'_, f32>,
) -> ndarray::Array1<f32> {
    profile::time("tee:compute_logits", || {
        let vocab = weights.token_embedding.nrows();
        let mut logits = ndarray::Array1::<f32>::zeros(vocab);
        for v in 0..vocab {
            let row = weights.token_embedding.row(v);
            let dot: f32 = h_last
                .iter()
                .zip(row.iter())
                .map(|(a, b)| a * b.to_f32())
                .sum();
            logits[v] = dot;
        }
        if let Some(cap) = cfg.final_logit_softcapping {
            let inv = 1.0_f32 / cap;
            for x in logits.iter_mut() {
                *x = (*x * inv).tanh() * cap;
            }
        }
        logits
    })
}

/// R3 variant of the inline LM-head op — masked GPU offload through
/// `exec.offload_linear(LmHead, …)` inside its own
/// `begin_forward_pass(1)` bracket. Mirrors
/// `generation::compute_logits_gpu` so the bench profile sees the
/// same wrapper bucket.
fn bench_compute_logits_gpu<X: TrustedExecutor>(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    exec: &mut X,
    h_last: ndarray::ArrayView1<'_, f32>,
) -> Result<ndarray::Array1<f32>> {
    profile::time("tee:compute_logits", || -> Result<ndarray::Array1<f32>> {
        let vocab = weights.token_embedding.nrows();
        let h2 = h_last.insert_axis(Axis(0));
        exec.begin_forward_pass(1)?;
        let logits_2d = exec.offload_linear(
            WeightHandle::new(0, WeightKind::LmHead),
            h2,
        )?;
        exec.end_forward_pass()?;
        anyhow::ensure!(
            logits_2d.nrows() == 1 && logits_2d.ncols() == vocab,
            "lm-head offload returned ({}, {}), expected (1, {vocab})",
            logits_2d.nrows(),
            logits_2d.ncols(),
        );
        let mut logits = logits_2d.row(0).to_owned();
        if let Some(cap) = cfg.final_logit_softcapping {
            let inv = 1.0_f32 / cap;
            for x in logits.iter_mut() {
                *x = (*x * inv).tanh() * cap;
            }
        }
        Ok(logits)
    })
}

fn argmax_row(row: ndarray::ArrayView1<'_, f32>) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

struct PhaseTiming {
    wall: std::time::Duration,
    snap: profile::Profile,
    /// Decode-only: per-step wall list. Empty for prefill.
    decode_steps: Vec<std::time::Duration>,
    tokens: Vec<u32>,
}

fn run_prefill_decode<X: TrustedExecutor>(
    label: &str,
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut X,
    prompt_ids: &[u32],
    max_tokens: usize,
    r3_on: bool,
) -> Result<(PhaseTiming, PhaseTiming)> {
    let max_cache_len = prompt_ids.len() + max_tokens + 1;
    let mut kv_cache = KvCache::new(weights.layers.len(), max_cache_len, cfg.kv_dim());

    // --- Prefill phase ---
    profile::reset();
    let t_prefill = Instant::now();
    let hidden = forward::run_prefill(cfg, weights, rope, exec, prompt_ids, &mut kv_cache)?;
    let prefill_wall = t_prefill.elapsed();
    let prefill_snap = profile::snapshot();
    let mut h_last = hidden.row(hidden.nrows() - 1).to_owned();

    eprintln!(
        "[{label}] PREFILL: n={}, wall {:.2}s, prefill {:.1} tok/s",
        prompt_ids.len(),
        prefill_wall.as_secs_f64(),
        prompt_ids.len() as f64 / prefill_wall.as_secs_f64(),
    );

    let prefill = PhaseTiming {
        wall: prefill_wall,
        snap: prefill_snap,
        decode_steps: Vec::new(),
        tokens: prompt_ids.to_vec(),
    };

    // --- Decode phase ---
    profile::reset();
    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut step_walls: Vec<std::time::Duration> = Vec::with_capacity(max_tokens);
    let t_decode_total = Instant::now();
    for _ in 0..max_tokens {
        let t_step = Instant::now();
        // Sample next token (compute_logits + argmax + advance one step).
        let logits = if r3_on {
            bench_compute_logits_gpu(cfg, weights, exec, h_last.view())?
        } else {
            bench_compute_logits_in_tee(cfg, weights, h_last.view())
        };
        let next_tok = argmax_row(logits.view());
        tokens.push(next_tok);
        // Append one new position via decode step.
        h_last = forward::run_decode_step(cfg, weights, rope, exec, next_tok, &mut kv_cache)?;
        step_walls.push(t_step.elapsed());
    }
    let decode_wall = t_decode_total.elapsed();
    let decode_snap = profile::snapshot();

    let step_mean_ms = if !step_walls.is_empty() {
        step_walls.iter().map(|d| d.as_secs_f64()).sum::<f64>() * 1000.0
            / step_walls.len() as f64
    } else {
        0.0
    };
    eprintln!(
        "[{label}] DECODE:  K={max_tokens}, wall {:.2}s, decode {:.1} tok/s, per-step mean {:.0} ms",
        decode_wall.as_secs_f64(),
        max_tokens as f64 / decode_wall.as_secs_f64(),
        step_mean_ms,
    );

    let decode = PhaseTiming {
        wall: decode_wall,
        snap: decode_snap,
        decode_steps: step_walls,
        tokens,
    };

    Ok((prefill, decode))
}

#[test]
#[ignore = "real-weight long-context bench: ~3-5 min on Qwen3-4B at n=2048 + 64-token decode"]
fn m1_12_per_op_breakdown_prefill_decode() -> Result<()> {
    let variant = variant_from_env();
    let n_prompt = prompt_size_from_env();
    let max_tokens = max_tokens_from_env();

    eprintln!("=== M1.12 per-op breakdown — prefill + decode, baseline vs R3 ===");
    eprintln!(
        "variant: {:?} ({})  n_prompt: {n_prompt}  max_tokens: {max_tokens}",
        variant,
        variant.hf_model_id(),
    );
    eprintln!("RSS at start: {}", fmt_gib(rss_bytes()));

    // Load weights, build executor, provision projections + LM head.
    let (cfg, tokenizer, mut weights, rope) = load_pretrained(variant)?;
    eprintln!("RSS after weights load: {}", fmt_gib(rss_bytes()));
    let engine = WgpuVulkanEngine::new_fp16().context("Vulkan adapter (fp16)")?;
    eprintln!(
        "Vulkan: {} ({:?})",
        engine.adapter_info().name,
        engine.adapter_info().device_type,
    );
    let mut exec =
        InProcessTrustedExecutor::with_seed(engine, MaskSeed::from_bytes([42u8; 32]));
    provision_into(&mut weights, &cfg, &mut exec)?;
    provision_lm_head_into(&weights, &mut exec)?;
    glibc_release_freed();
    eprintln!("RSS after provision (projections + LmHead): {}", fmt_gib(rss_bytes()));

    let prompt_ids = build_prompt_ids(&tokenizer, n_prompt)?;
    eprintln!("Tokenised prompt: {} ids", prompt_ids.len());

    // --- Baseline (R3 off): prefill + decode ---
    eprintln!();
    eprintln!("─── BASELINE (R3 off, in-TEE compute_logits) ───");
    let (b_prefill, b_decode) =
        run_prefill_decode("baseline", &cfg, &weights, &rope, &mut exec, &prompt_ids, max_tokens, false)?;

    b_prefill.snap.dump(&format!(
        "{:?} BASELINE prefill profile (n={n_prompt})",
        variant
    ));
    b_decode.snap.dump(&format!(
        "{:?} BASELINE decode profile (K={max_tokens})",
        variant
    ));

    // --- R3 on: prefill + decode ---
    // Note: prefill never calls compute_logits, so R3's prefill numbers
    // should track baseline prefill (with mask-state independence noise).
    eprintln!();
    eprintln!("─── R3 ON (LM_HEAD_GPU_OFFLOAD, GPU LM-head) ───");
    let (r_prefill, r_decode) =
        run_prefill_decode("R3", &cfg, &weights, &rope, &mut exec, &prompt_ids, max_tokens, true)?;

    r_prefill.snap.dump(&format!(
        "{:?} R3 prefill profile (n={n_prompt})",
        variant
    ));
    r_decode.snap.dump(&format!(
        "{:?} R3 decode profile (K={max_tokens})",
        variant
    ));

    // --- Side-by-side decode comparison ---
    eprintln!();
    eprintln!("─── DECODE Δ (R3 vs baseline) ───");
    let b_decode_ms = b_decode.wall.as_secs_f64() * 1000.0;
    let r_decode_ms = r_decode.wall.as_secs_f64() * 1000.0;
    eprintln!(
        "decode wall:  baseline {:.0} ms  R3 {:.0} ms  Δ {:+.1}%",
        b_decode_ms,
        r_decode_ms,
        100.0 * (r_decode_ms - b_decode_ms) / b_decode_ms,
    );
    let b_logit = b_decode
        .snap
        .buckets
        .get("tee:compute_logits")
        .map(|(d, _)| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let r_logit = r_decode
        .snap
        .buckets
        .get("tee:compute_logits")
        .map(|(d, _)| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    eprintln!(
        "tee:compute_logits (decode):  baseline {:.0} ms  R3 {:.0} ms  Δ {:+.1}%",
        b_logit,
        r_logit,
        100.0 * (r_logit - b_logit) / b_logit.max(1e-9),
    );
    let b_step_mean = if !b_decode.decode_steps.is_empty() {
        b_decode_ms / b_decode.decode_steps.len() as f64
    } else {
        0.0
    };
    let r_step_mean = if !r_decode.decode_steps.is_empty() {
        r_decode_ms / r_decode.decode_steps.len() as f64
    } else {
        0.0
    };
    eprintln!(
        "per-step mean: baseline {:.0} ms  R3 {:.0} ms  ({:.2} → {:.2} tok/s)",
        b_step_mean,
        r_step_mean,
        max_tokens as f64 / b_decode.wall.as_secs_f64(),
        max_tokens as f64 / r_decode.wall.as_secs_f64(),
    );

    let prefix_match = b_decode
        .tokens
        .iter()
        .zip(r_decode.tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "decode token-prefix match: {} / {}",
        prefix_match,
        b_decode.tokens.len()
    );

    Ok(())
}
