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
    // R1's RSS verdict is captured above. For Q#1, just dump the
    // profile of one short generate so the `tee:compute_logits`
    // bucket appears alongside the other buckets — confirms the
    // instrumentation fires. Historical comparison (baseline vs R3)
    // is gone: R3 is the only path now. The full prefill/decode
    // breakdown lives in `m1_12_per_op_breakdown_prefill_decode`.
    let prompt = "The quick brown fox";
    let prompt_ids = tokenizer.encode(prompt, 32)?;
    let gen_cfg = GenerationConfig {
        max_tokens,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };
    profile::reset();
    let t = Instant::now();
    let out = generation::generate(&cfg, &weights, &rope, &mut exec, &prompt_ids, &gen_cfg)?;
    let wall = t.elapsed();
    let snap = profile::snapshot();
    eprintln!();
    eprintln!(
        "Q#1 smoke generate: wall {:.2}s ({:.1} tok/s), prompt='{prompt}' n={} K={max_tokens}, tokens={:?}",
        wall.as_secs_f64(),
        out.tokens.len() as f64 / wall.as_secs_f64(),
        prompt_ids.len(),
        out.tokens,
    );
    snap.dump(&format!("{:?} generate({max_tokens}) profile", variant));

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

/// **Default B = 8** for every future bench run in this file —
/// matches the production-shape target from
/// `2026-05-22-q3-4b-b8-mask-sweep.md`. Pass `GELO_BENCH_B=1`
/// explicitly when you want single-stream sanity numbers.
fn batch_size_from_env() -> usize {
    std::env::var("GELO_BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
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
    /// For B=1 path: the decoded tokens. For B>1: per-sequence sample,
    /// only the first sequence's tokens are reported (the bench is
    /// shape-comparison, not output-quality).
    tokens: Vec<u32>,
}

/// Batched (B>1) variant of `run_prefill_decode`. Drives
/// `forward::run_prefill_batched` + `forward::run_decode_step_batched`
/// over `B` identical prompts, calls per-sequence
/// `bench_compute_logits_*` between decode steps. R3 LM-head offload
/// opens its own one-row forward-pass bracket per sequence per step
/// (B brackets per decode step), then the batched decode-step session
/// opens its own per-sequence-A_b bracket — matches the library's
/// `generate_batched` dispatch shape.
fn run_prefill_decode_batched<X: TrustedExecutor>(
    label: &str,
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut X,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    r3_on: bool,
) -> Result<(PhaseTiming, PhaseTiming)> {
    let batch_size = prompts.len();
    assert!(batch_size > 0);
    let n_max = prompts.iter().map(|p| p.len()).max().unwrap();
    let max_cache_len = n_max + max_tokens + 1;
    let mut kv_cache =
        KvCache::new_batched(batch_size, weights.layers.len(), max_cache_len, cfg.kv_dim());

    // --- Prefill phase ---
    profile::reset();
    let t = Instant::now();
    let (hidden_3d, seq_lens) =
        forward::run_prefill_batched(cfg, weights, rope, exec, prompts, &mut kv_cache)?;
    let prefill_wall = t.elapsed();
    let prefill_snap = profile::snapshot();

    let mut last_hidden: Vec<ndarray::Array1<f32>> = (0..batch_size)
        .map(|b| hidden_3d.slice(ndarray::s![b, seq_lens[b] - 1, ..]).to_owned())
        .collect();

    eprintln!(
        "[{label}] PREFILL: B={batch_size} n_max={n_max}, wall {:.2}s, prefill {:.1} tok/s/seq",
        prefill_wall.as_secs_f64(),
        n_max as f64 / prefill_wall.as_secs_f64(),
    );

    let prefill = PhaseTiming {
        wall: prefill_wall,
        snap: prefill_snap,
        decode_steps: Vec::new(),
        tokens: prompts[0].clone(),
    };

    // --- Decode phase ---
    profile::reset();
    let mut tokens: Vec<Vec<u32>> =
        (0..batch_size).map(|_| Vec::with_capacity(max_tokens)).collect();
    let mut step_walls: Vec<std::time::Duration> = Vec::with_capacity(max_tokens);
    let t_decode_total = Instant::now();
    for _ in 0..max_tokens {
        let t_step = Instant::now();
        // Per-sequence compute_logits — B independent calls.
        let mut next_tokens = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let logits = if r3_on {
                bench_compute_logits_gpu(cfg, weights, exec, last_hidden[b].view())?
            } else {
                bench_compute_logits_in_tee(cfg, weights, last_hidden[b].view())
            };
            let next_tok = argmax_row(logits.view());
            tokens[b].push(next_tok);
            next_tokens.push(next_tok);
        }
        // One batched decode step over B tokens.
        let h_step = forward::run_decode_step_batched(
            cfg,
            weights,
            rope,
            exec,
            &next_tokens,
            &mut kv_cache,
        )?;
        for b in 0..batch_size {
            last_hidden[b].assign(&h_step.row(b));
        }
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
        "[{label}] DECODE:  B={batch_size} K={max_tokens}, wall {:.2}s, decode {:.1} tok/s/seq, per-step mean {:.0} ms (= {:.1} ms / tok / seq)",
        decode_wall.as_secs_f64(),
        max_tokens as f64 / decode_wall.as_secs_f64(),
        step_mean_ms,
        step_mean_ms / batch_size as f64,
    );

    let decode = PhaseTiming {
        wall: decode_wall,
        snap: decode_snap,
        decode_steps: step_walls,
        tokens: std::mem::take(&mut tokens[0]),
    };

    Ok((prefill, decode))
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
#[ignore = "real-weight long-context bench: minutes on Qwen3-4B at B=8 n=2048 K=64"]
fn m1_12_per_op_breakdown_prefill_decode() -> Result<()> {
    let variant = variant_from_env();
    let n_prompt = prompt_size_from_env();
    let max_tokens = max_tokens_from_env();
    let batch_size = batch_size_from_env();

    eprintln!("=== M1.12 per-op breakdown — prefill + decode, baseline vs R3 ===");
    eprintln!(
        "variant: {:?} ({})  B: {batch_size}  n_prompt: {n_prompt}  max_tokens: {max_tokens}",
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

    let single_prompt = build_prompt_ids(&tokenizer, n_prompt)?;
    eprintln!("Tokenised prompt: {} ids", single_prompt.len());

    // Build B identical prompts when batching. Identical content keeps
    // the measurement focused on the protocol overhead (mask + GPU +
    // attention) and removes tokenisation noise across sequences.
    let prompts: Vec<Vec<u32>> = (0..batch_size).map(|_| single_prompt.clone()).collect();

    // --- Baseline (R3 off): prefill + decode ---
    eprintln!();
    eprintln!("─── BASELINE (R3 off, in-TEE compute_logits) ───");
    let (b_prefill, b_decode) = if batch_size == 1 {
        run_prefill_decode(
            "baseline", &cfg, &weights, &rope, &mut exec, &single_prompt, max_tokens, false,
        )?
    } else {
        run_prefill_decode_batched(
            "baseline", &cfg, &weights, &rope, &mut exec, &prompts, max_tokens, false,
        )?
    };

    b_prefill.snap.dump(&format!(
        "{:?} BASELINE prefill profile (B={batch_size} n={n_prompt})",
        variant
    ));
    b_decode.snap.dump(&format!(
        "{:?} BASELINE decode profile (B={batch_size} K={max_tokens})",
        variant
    ));

    // --- R3 on: prefill + decode ---
    eprintln!();
    eprintln!("─── R3 ON (LM_HEAD_GPU_OFFLOAD, GPU LM-head) ───");
    let (r_prefill, r_decode) = if batch_size == 1 {
        run_prefill_decode(
            "R3", &cfg, &weights, &rope, &mut exec, &single_prompt, max_tokens, true,
        )?
    } else {
        run_prefill_decode_batched(
            "R3", &cfg, &weights, &rope, &mut exec, &prompts, max_tokens, true,
        )?
    };

    r_prefill.snap.dump(&format!(
        "{:?} R3 prefill profile (B={batch_size} n={n_prompt})",
        variant
    ));
    r_decode.snap.dump(&format!(
        "{:?} R3 decode profile (B={batch_size} K={max_tokens})",
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

// ─── M1.12+ sweep: (B, n, mask_kind) cells ─────────────────────────

/// Mask family selector for the sweep harness.
#[derive(Copy, Clone, Debug)]
enum SweepMaskKind {
    Auto,
    Hd3,
}

impl SweepMaskKind {
    fn from_env() -> Self {
        match std::env::var("GELO_SWEEP_MASK").as_deref() {
            Ok("hd3") | Ok("Hd3") | Ok("HD3") => Self::Hd3,
            _ => Self::Auto,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Hd3 => "hd3",
        }
    }
}

/// Dump only the buckets the sweep cares about — mask apply/unapply
/// (split per family), shield, engine matmul, in-TEE attention,
/// compute_logits, strip. Keeps the output greppable.
fn dump_sweep_buckets(snap: &profile::Profile, label: &str) {
    // Order matters for readability: heaviest first.
    let keys = [
        "gelo:mask_apply:hd3",
        "gelo:mask_apply:dct4",
        "gelo:mask_apply:haar",
        "gelo:mask_unapply:hd3",
        "gelo:mask_unapply:dct4",
        "gelo:mask_unapply:haar",
        "gelo:mask_sample",
        "gelo:shield_stack",
        "gelo:strip_shield",
        "engine:matmul",
        "engine:matmul_many",
        // Batched (B≥2) attention buckets — run_prefill_batched /
        // run_decode_step_batched.
        "tee:attn_inplace_many",
        "tee:attn_cached_inplace_many",
        // B=1 attention buckets — run_prefill / run_decode_step via
        // decoder_block_cached. `tee:attn_permuted_cached` fires at
        // prefill when perm_attention_enabled_for(n_q) is true;
        // `tee:attn_cached` fires at decode (n_q=1, perm off) and at
        // prefill when perm is off; `tee:attn_swa_cached` fires for
        // sliding-window attention layers (Qwen3-4B has none).
        "tee:attn_permuted_cached",
        "tee:attn_cached",
        "tee:attn_swa_cached",
        "tee:compute_logits",
    ];
    eprintln!("--- {label} buckets ---");
    for k in keys {
        if let Some((d, n)) = snap.buckets.get(k) {
            eprintln!(
                "  {:36} {:>10.1} ms  ({} calls)",
                k,
                d.as_secs_f64() * 1000.0,
                n
            );
        }
    }
}

/// One sweep cell: (B, n_per_seq, mask_kind) → prefill + decode walls
/// + per-family mask-bucket walls, dumped in a structured greppable
/// line so the bash driver can collate.
#[test]
#[ignore = "M1.12+ sweep cell: real-weight bench, minutes; driven by scripts/m1-12-hd3-perf-sweep.sh"]
fn m1_12_sweep_cell() -> Result<()> {
    let variant = variant_from_env();
    let batch_size = batch_size_from_env();
    let n_per_seq = prompt_size_from_env();
    let max_tokens = max_tokens_from_env();
    let mask_kind = SweepMaskKind::from_env();

    let cell_label = format!(
        "{:?} B={batch_size} n={n_per_seq} K={max_tokens} mask={}",
        variant,
        mask_kind.label()
    );

    eprintln!("=== M1.12+ sweep cell: {cell_label} ===");
    eprintln!("RSS at start: {}", fmt_gib(rss_bytes()));

    let (cfg, tokenizer, mut weights, rope) = load_pretrained(variant)?;
    eprintln!("RSS after weights load: {}", fmt_gib(rss_bytes()));

    let engine = WgpuVulkanEngine::new_fp16().context("Vulkan adapter (fp16)")?;
    let exec =
        InProcessTrustedExecutor::with_seed(engine, MaskSeed::from_bytes([42u8; 32]));
    // Apply mask-kind override BEFORE provision so the session bracket
    // uses the right family from the first offload.
    let mut exec = match mask_kind {
        SweepMaskKind::Auto => exec.with_auto_mask(),
        SweepMaskKind::Hd3 => exec.with_hd3_mask(),
    };

    provision_into(&mut weights, &cfg, &mut exec)?;
    provision_lm_head_into(&weights, &mut exec)?;
    glibc_release_freed();
    eprintln!(
        "RSS after provision (projections + LmHead): {}",
        fmt_gib(rss_bytes())
    );

    let single_prompt = build_prompt_ids(&tokenizer, n_per_seq)?;
    let prompts: Vec<Vec<u32>> = (0..batch_size).map(|_| single_prompt.clone()).collect();

    let (prefill, decode) = if batch_size == 1 {
        run_prefill_decode(
            &cell_label,
            &cfg,
            &weights,
            &rope,
            &mut exec,
            &single_prompt,
            max_tokens,
            true, // R3 on (production default at time of sweep)
        )?
    } else {
        run_prefill_decode_batched(
            &cell_label,
            &cfg,
            &weights,
            &rope,
            &mut exec,
            &prompts,
            max_tokens,
            true,
        )?
    };

    dump_sweep_buckets(&prefill.snap, &format!("{cell_label} PREFILL"));
    dump_sweep_buckets(&decode.snap, &format!("{cell_label} DECODE"));

    // Per-sequence normalisation: at B>1 the aggregate token count is
    // batch_size×n_per_seq for prefill and batch_size×max_tokens for
    // decode; this is what `2026-05-22-q3-4b-b8-mask-sweep.md` and the
    // M1.12 roadmap §0 quote when comparing across B.
    let prefill_aggregate_tokens = (batch_size * n_per_seq) as f64;
    let decode_aggregate_tokens = (batch_size * max_tokens) as f64;
    let prefill_tps_agg = prefill_aggregate_tokens / prefill.wall.as_secs_f64();
    let decode_tps_agg = decode_aggregate_tokens / decode.wall.as_secs_f64();
    let prefill_tps_per_seq = n_per_seq as f64 / prefill.wall.as_secs_f64();
    let decode_tps_per_seq = max_tokens as f64 / decode.wall.as_secs_f64();

    // Single structured summary line — the bash driver greps for SWEEP_RESULT.
    eprintln!(
        "SWEEP_RESULT variant={:?} B={} n={} K={} mask={} \
         prefill_wall_s={:.3} decode_wall_s={:.3} \
         prefill_tps_agg={:.2} prefill_tps_per_seq={:.2} \
         decode_tps_agg={:.2} decode_tps_per_seq={:.2}",
        variant,
        batch_size,
        n_per_seq,
        max_tokens,
        mask_kind.label(),
        prefill.wall.as_secs_f64(),
        decode.wall.as_secs_f64(),
        prefill_tps_agg,
        prefill_tps_per_seq,
        decode_tps_agg,
        decode_tps_per_seq,
    );

    Ok(())
}
