//! Qwen3-1.7B long-context generation bench on Vulkan GPU.
//!
//! Three prompt lengths, two protocol cells. **Correction
//! 2026-05-18:** an earlier comment claimed the n=2048 row crosses
//! the OutAttnMult auto-switch threshold. That was wrong — the
//! cached generation path (`decoder_block_cached`) explicitly
//! keeps global-layer attention in-TEE at **every** `n` per the
//! locked M1.3 design decision (see
//! `crates/gelo-embedder/src/decoder/forward.rs:355-370`). So at
//! every length below, both `gpu_plain` and `gpu_gelo` run
//! attention compute in-TEE on the CPU under BLIS. The cost
//! discontinuity surfaced at n=2048 is the **GELO mask
//! round-trip on the 6 linear projections per layer × 28 layers**,
//! whose CPU cost scales as `O(n²·d)` — not OutAttnMult engaging.
//! M1.10 (fused permuted attention) is the structural fix; see
//! `docs/plans/m1-10-fused-permuted-attention.md`.
//!
//! Cells:
//!  1. **gpu_plain** — `PlaintextExecutor` + Vulkan engine. Baseline.
//!  2. **gpu_gelo** — `InProcessTrustedExecutor::with_seed`
//!     paper-parity (per-forward Haar `A` + shield(8, 4.0)).
//!
//! `gpu_full_stack` (with U-Verify) is intentionally **omitted** at
//! long contexts: the existing `qwen3_generation_bench` measured
//! ~150× decode slowdown for k=2 probes against 1.7B weights; at
//! n_prompt=2048 the Freivalds probes against ~13 GB of TEE-side
//! weights would push wall-clock into hours per cell. The protocol
//! gap is documented in §09 of `docs/prototype/gelo-llm.html`.
//!
//! **OutAttnMult cadence — production auto-switch, not forced-on.**
//! Unlike `qwen3_generation_bench` (which sets
//! `out_attn_mult_min_seq_len = Some(0)` to exercise the path at
//! every shape), this bench uses the production default (`None`,
//! resolves to `hidden_size = 2048`). That way the n=64 and n=512
//! rows stay in-TEE on the attention side and the n=2048 row
//! engages the offload path naturally.
//!
//! Downloads ~3.4 GB on first run (`Qwen/Qwen3-1.7B` bf16
//! safetensors); gated behind `#[ignore]`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use gelo_embedder::common::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::profile;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    InProcessTrustedExecutor, PlaintextExecutor, TrustedExecutor, WeightHandle, WeightKind,
};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};

const VARIANT: Qwen3Variant = Qwen3Variant::Q1_7B;
const DEFAULT_PROMPT_LENGTHS: &[usize] = &[64, 512, 2048];
const DEFAULT_MAX_TOKENS: usize = 16;

fn prompt_lengths_from_env() -> Vec<usize> {
    match std::env::var("GELO_BENCH_LENGTHS") {
        Ok(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => DEFAULT_PROMPT_LENGTHS.to_vec(),
    }
}

fn max_tokens_from_env() -> usize {
    std::env::var("GELO_BENCH_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

/// Skip the gpu_gelo_permuted cell (the slow one) during perf-lever
/// sweeps where only the GELO mask path is changing. `GELO_BENCH_SKIP_PERMUTED=1`
/// halves wall-clock at long n.
fn skip_permuted_from_env() -> bool {
    std::env::var("GELO_BENCH_SKIP_PERMUTED")
        .ok()
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Long-ish source text we tokenise and truncate to hit a target token count.
/// Content is irrelevant for perf measurement — only the resulting token
/// count matters. Repeated to ensure we always overflow the largest
/// requested prompt length.
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
    cell: &'static str,
    n_prompt: usize,
    ttft: Duration,
    decode_steps: Vec<Duration>,
    /// Profile bucket snapshot for the prefill call (post-`run_prefill`).
    /// All `profile::time` regions executed during the prefill are
    /// aggregated here; `decode_profile` captures the same buckets across
    /// all `run_decode_step` calls in this generate.
    prefill_profile: profile::Profile,
    decode_profile: profile::Profile,
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
        if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
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
}

/// One timed generation pass with per-step decode timing. Same body as
/// the short-context bench's helper; duplicated here to keep both
/// test files self-contained.
fn time_generate<X: TrustedExecutor>(
    cell: &'static str,
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut X,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Result<CellTiming> {
    use gelo_embedder::decoder::forward::{run_decode_step, run_prefill};
    use gelo_embedder::decoder::kv_cache::KvCache;

    let max_cache_len = prompt_ids.len() + max_tokens;
    let mut kv_cache = KvCache::new(weights.layers.len(), max_cache_len, cfg.kv_dim());

    profile::reset();
    let t_prefill = Instant::now();
    let hidden = run_prefill(cfg, weights, rope, exec, prompt_ids, &mut kv_cache)?;
    let ttft = t_prefill.elapsed();
    let prefill_profile = profile::snapshot();

    let mut h_last = hidden.row(hidden.nrows() - 1).to_owned();
    let mut decode_steps = Vec::with_capacity(max_tokens);

    profile::reset();
    for _ in 0..max_tokens {
        // Inline greedy argmax (avoids exporting compute_logits from the
        // generation module just for the bench).
        let vocab = weights.token_embedding.nrows();
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for v in 0..vocab {
            let row = weights.token_embedding.row(v);
            let dot: f32 = h_last.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            if dot > best_val {
                best_val = dot;
                best_idx = v as u32;
            }
        }

        let t_step = Instant::now();
        h_last = run_decode_step(cfg, weights, rope, exec, best_idx, &mut kv_cache)?;
        decode_steps.push(t_step.elapsed());
    }
    let decode_profile = profile::snapshot();

    Ok(CellTiming {
        cell,
        n_prompt: prompt_ids.len(),
        ttft,
        decode_steps,
        prefill_profile,
        decode_profile,
    })
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
    weights: &DecoderWeights,
    exec: &mut X,
) -> Result<()> {
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
    Ok(())
}

/// Build a prompt of exactly `target_tokens` token IDs by tokenising a
/// long source text with `max_len = target_tokens`. The tokenizer
/// truncates to fit, so the result has at most `target_tokens` entries;
/// if the source is too short, we repeat it.
fn build_prompt_ids(tokenizer: &HfTokenizer, target_tokens: usize) -> Result<Vec<u32>> {
    // Start with enough repetitions to comfortably overflow the target.
    let reps = (target_tokens / 30).max(1) + 1;
    let text = LONG_TEXT_SEED.repeat(reps);
    let ids = tokenizer.encode(&text, target_tokens)?;
    if ids.len() < target_tokens {
        return Err(anyhow!(
            "tokeniser returned {} tokens, expected {} — increase source text",
            ids.len(),
            target_tokens,
        ));
    }
    Ok(ids)
}

#[test]
#[ignore = "downloads ~3.4 GB Qwen3-1.7B; requires Vulkan GPU; ~2-4 min wall-clock"]
fn qwen3_1_7b_long_context_breakdown() -> Result<()> {
    let prompt_lengths = prompt_lengths_from_env();
    let max_tokens = max_tokens_from_env();
    let skip_permuted = skip_permuted_from_env();
    eprintln!("RSS before any load: {}", fmt_gib(rss_bytes()));
    eprintln!(
        "Qwen3-1.7B long-context bench — model={} lengths={:?} max_tokens={} skip_permuted={}",
        VARIANT.hf_model_id(),
        prompt_lengths,
        max_tokens,
        skip_permuted,
    );
    eprintln!(
        "Mask GEMM backend: {}",
        gelo_protocol::mask_backend_description()
    );

    let (cfg, tokenizer, weights, rope) = load_pretrained()?;
    eprintln!(
        "RSS after CPU weights load (Arc<DecoderWeights>): {}",
        fmt_gib(rss_bytes())
    );

    // Production auto-switch — OutAttnMult engages naturally at
    // `n >= hidden_size = 2048`. The bench's whole point is to see
    // that discontinuity, so do NOT override the threshold.
    let cfg_offload = cfg.clone();

    let gpu_root = WgpuVulkanEngine::new().context("Vulkan adapter")?;
    let adapter_line = format!(
        "{} ({:?}, driver={}, info={})",
        gpu_root.adapter_info().name,
        gpu_root.adapter_info().device_type,
        gpu_root.adapter_info().driver,
        gpu_root.adapter_info().driver_info,
    );
    assert!(gpu_root.is_real_gpu(), "bench needs real GPU hardware");

    // 1. gpu_plain.
    eprintln!("[gpu_plain] provisioning weights to shared GPU engine...");
    let mut gpu_plain = PlaintextExecutor::new(gpu_root.clone_shared());
    provision_decoder_weights(&cfg_offload, &weights, &mut gpu_plain)?;
    eprintln!("RSS after gpu_plain provision: {}", fmt_gib(rss_bytes()));

    // 2. gpu_gelo (paper-parity defaults: per-forward A + shield(8, 4.0)).
    eprintln!("[gpu_gelo] provisioning (per-forward A + shield(8,4.0))...");
    let mut gpu_gelo = InProcessTrustedExecutor::with_seed(
        gpu_root.clone_shared(),
        MaskSeed::from_bytes([13u8; 32]),
    );
    provision_decoder_weights(&cfg_offload, &weights, &mut gpu_gelo)?;
    eprintln!("RSS after gpu_gelo provision: {}", fmt_gib(rss_bytes()));

    // 3. gpu_gelo_permuted (M1.10 Phase 1 dispatch — permuted_cached
    //    attention via F1+ in-TEE softmax + soft causal mask).
    //    Use a separate config that forces the path on at every n_q
    //    and disables OutAttnMult (cached path doesn't call it anyway,
    //    but `perm_attention_enabled_for` consults the OutAttnMult
    //    threshold as a tiebreaker; explicit disable makes the bench
    //    cell unambiguous). Skipped when GELO_BENCH_SKIP_PERMUTED=1.
    let mut cfg_permuted = cfg_offload.clone();
    cfg_permuted.use_perm_attention = true;
    cfg_permuted.perm_attention_min_seq_len = Some(0);
    cfg_permuted.use_out_attn_mult = false;
    let mut gpu_gelo_permuted_opt = if skip_permuted {
        eprintln!("[gpu_gelo_permuted] skipped (GELO_BENCH_SKIP_PERMUTED=1)");
        None
    } else {
        eprintln!("[gpu_gelo_permuted] provisioning (M1.10 Phase 1 permuted_cached path)...");
        let mut e = InProcessTrustedExecutor::with_seed(
            gpu_root.clone_shared(),
            MaskSeed::from_bytes([29u8; 32]),
        );
        // Hidden-No-More-class noise on Q/K under permutation;
        // F1+ soft causal mask C=30 is enabled by default.
        e.set_perm_attention(gelo_protocol::PermAttnConfig::HIDDEN_NO_MORE);
        provision_decoder_weights(&cfg_permuted, &weights, &mut e)?;
        eprintln!(
            "RSS after gpu_gelo_permuted provision: {}",
            fmt_gib(rss_bytes())
        );
        Some(e)
    };

    // Pre-build all prompts. Reused across both cells so the timing
    // measures protocol overhead, not tokenisation.
    let prompts: Vec<Vec<u32>> = prompt_lengths
        .iter()
        .map(|&n| build_prompt_ids(&tokenizer, n))
        .collect::<Result<Vec<_>>>()?;
    for (n_target, ids) in prompt_lengths.iter().zip(prompts.iter()) {
        eprintln!(
            "  prompt n={n_target}: tokenised to {} tokens",
            ids.len(),
        );
    }

    // Warmup: one short generate per cell to amortise shader compile +
    // first-touch page faults. Using the shortest prompt is enough.
    eprintln!("[warm] one untimed generate(2) per cell at n=64...");
    let warm_ids = &prompts[0];
    let _ = time_generate(
        "warm_plain", &cfg_offload, &weights, &rope, &mut gpu_plain, warm_ids, 2,
    )?;
    let _ = time_generate(
        "warm_gelo", &cfg_offload, &weights, &rope, &mut gpu_gelo, warm_ids, 2,
    )?;
    if let Some(e) = gpu_gelo_permuted_opt.as_mut() {
        let _ = time_generate(
            "warm_gelo_permuted",
            &cfg_permuted,
            &weights,
            &rope,
            e,
            warm_ids,
            2,
        )?;
    }
    eprintln!("RSS after warmup: {}", fmt_gib(rss_bytes()));

    // Measure — interleave (cell, length) so system-wide noise (FS, thermal)
    // hits all cells equivalently at each shape. When skip_permuted=true the
    // chunk size is 2 (plain + gelo) instead of 3.
    eprintln!("[measure] timed generate({max_tokens}) per (cell, length)...");
    let cells_per_n: usize = if skip_permuted { 2 } else { 3 };
    let mut results: Vec<CellTiming> = Vec::with_capacity(prompt_lengths.len() * cells_per_n);
    for (n_target, ids) in prompt_lengths.iter().zip(prompts.iter()) {
        eprintln!("  --- n={n_target} ---");
        let r_plain = time_generate(
            "gpu_plain", &cfg_offload, &weights, &rope, &mut gpu_plain, ids, max_tokens,
        )?;
        eprintln!(
            "    gpu_plain          n={n_target}: TTFT {:.0} ms · TPOT {:.1} ms",
            r_plain.ttft.as_secs_f64() * 1000.0,
            r_plain.decode_mean_ms(),
        );
        let r_gelo = time_generate(
            "gpu_gelo", &cfg_offload, &weights, &rope, &mut gpu_gelo, ids, max_tokens,
        )?;
        eprintln!(
            "    gpu_gelo           n={n_target}: TTFT {:.0} ms · TPOT {:.1} ms",
            r_gelo.ttft.as_secs_f64() * 1000.0,
            r_gelo.decode_mean_ms(),
        );
        results.push(r_plain);
        results.push(r_gelo);
        if let Some(e) = gpu_gelo_permuted_opt.as_mut() {
            let r_permuted = time_generate(
                "gpu_gelo_permuted",
                &cfg_permuted,
                &weights,
                &rope,
                e,
                ids,
                max_tokens,
            )?;
            eprintln!(
                "    gpu_gelo_permuted  n={n_target}: TTFT {:.0} ms · TPOT {:.1} ms",
                r_permuted.ttft.as_secs_f64() * 1000.0,
                r_permuted.decode_mean_ms(),
            );
            results.push(r_permuted);
        }
    }

    eprintln!();
    eprintln!("{}", "=".repeat(118));
    eprintln!("Vulkan adapter: {adapter_line}");
    eprintln!(
        "Model: {} · greedy · max_tokens={max_tokens} · global attention runs in-TEE at every n (cached path, see decoder/forward.rs:355-370)",
        VARIANT.hf_model_id(),
    );
    eprintln!("{}", "=".repeat(118));
    eprintln!(
        "{:<14} {:>9} {:>11} {:>13} {:>11} {:>12} {:>15} {:>13}",
        "cell", "n_prompt", "TTFT (ms)", "TPOT mean ms", "median ms", "stddev ms",
        "total (s)", "vs gpu_plain"
    );
    eprintln!("{}", "-".repeat(118));
    for chunk in results.chunks_exact(cells_per_n) {
        let plain = &chunk[0];
        let gelo = &chunk[1];
        let gelo_overhead =
            100.0 * (gelo.total().as_secs_f64() / plain.total().as_secs_f64() - 1.0);
        eprintln!(
            "{:<20} {:>9} {:>11.1} {:>13.1} {:>11.1} {:>12.1} {:>15.3} {:>13}",
            plain.cell,
            plain.n_prompt,
            plain.ttft.as_secs_f64() * 1000.0,
            plain.decode_mean_ms(),
            plain.decode_median_ms(),
            plain.decode_stddev_ms(),
            plain.total().as_secs_f64(),
            "(base)",
        );
        eprintln!(
            "{:<20} {:>9} {:>11.1} {:>13.1} {:>11.1} {:>12.1} {:>15.3} {:>+12.1}%",
            gelo.cell,
            gelo.n_prompt,
            gelo.ttft.as_secs_f64() * 1000.0,
            gelo.decode_mean_ms(),
            gelo.decode_median_ms(),
            gelo.decode_stddev_ms(),
            gelo.total().as_secs_f64(),
            gelo_overhead,
        );
        if let Some(permuted) = chunk.get(2) {
        let permuted_overhead =
            100.0 * (permuted.total().as_secs_f64() / plain.total().as_secs_f64() - 1.0);
        eprintln!(
            "{:<20} {:>9} {:>11.1} {:>13.1} {:>11.1} {:>12.1} {:>15.3} {:>+12.1}%",
            permuted.cell,
            permuted.n_prompt,
            permuted.ttft.as_secs_f64() * 1000.0,
            permuted.decode_mean_ms(),
            permuted.decode_median_ms(),
            permuted.decode_stddev_ms(),
            permuted.total().as_secs_f64(),
            permuted_overhead,
        );
        }
        eprintln!();
    }
    eprintln!("{}", "=".repeat(118));

    // Per-(cell, n) profile breakdown — one bucket dump per CellTiming.
    // Buckets are populated by `profile::time` regions inside
    // `gelo-protocol` (mask_sample, mask_apply, mask_unapply, shield_*,
    // engine:matmul*, perm_attention_cached, …). The prefill section is
    // the most informative for asymptotic scaling; the decode section is
    // mostly relevant for the permuted_cached path.
    eprintln!();
    eprintln!("{}", "#".repeat(118));
    eprintln!("# Per-bucket profile breakdowns (use these for FLOP / scaling analysis)");
    eprintln!("{}", "#".repeat(118));
    for r in &results {
        let header_pref = format!(
            "{} · n={} · prefill (TTFT={:.1} ms)",
            r.cell,
            r.n_prompt,
            r.ttft.as_secs_f64() * 1000.0,
        );
        r.prefill_profile.dump(&header_pref);
        let header_dec = format!(
            "{} · n={} · decode (TPOT mean={:.1} ms, {} step{})",
            r.cell,
            r.n_prompt,
            r.decode_mean_ms(),
            r.decode_steps.len(),
            if r.decode_steps.len() == 1 { "" } else { "s" },
        );
        r.decode_profile.dump(&header_dec);
    }
    eprintln!();
    eprintln!(
        "Notes: every cell runs global attention in-TEE on CPU/BLIS (cached path, locked M1.3 design)."
    );
    eprintln!(
        "       The +overhead at long n is the GELO mask round-trip on 6 linear projections × 28 layers,"
    );
    eprintln!(
        "       NOT OutAttnMult engagement — `decoder_block_cached` does not call OutAttnMult."
    );
    eprintln!(
        "       Mask-applies + mask-unapplies are O((n+k)²·d) CPU GEMMs at the linear-offload boundaries."
    );
    eprintln!(
        "       M1.10 fused permuted attention is the structural fix (see docs/plans/m1-10-*.md)."
    );
    eprintln!(
        "       gpu_full_stack (U-Verify) is intentionally skipped — see qwen3_generation_bench §08 result for the cost regime."
    );
    eprintln!("RSS at end: {}", fmt_gib(rss_bytes()));

    Ok(())
}
