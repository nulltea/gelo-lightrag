//! Snapshot-capture CLI for the AloePri attack-resistance harness.
//!
//! Loads Qwen3-1.7B, runs prefill (and optionally a few decode steps)
//! over the prompt corpus under one or more of the three control
//! conditions defined in `docs/prototype/aloepri-attack-harness.md`
//! §2.4 (C0 plain / C1 mask-only / C2 default), drains the captured
//! `PcieSnapshot`s and writes one `<condition>.safetensors` plus the
//! sidecar `<condition>.meta.json` per condition into `--output`.
//!
//! All AloePri-vs-GELO TTRSR numbers in the Phase 2 release-gate
//! results JSON consume these files; the Python attack drivers under
//! `evals/aloepri-attacks/attack_drivers/` read them via
//! `snapshots_loader.py`.
//!
//! ## OOM safeguards (post-2026-05-18 incident)
//!
//! See `docs/prototype/aloepri-attack-harness-findings.md` for the
//! root-cause writeup. Four safeguards land here:
//!
//! 1. **`max_tokens=0` → direct `run_prefill`**: `generate()`
//!    short-circuits when `max_tokens=0`, so the previous code
//!    captured zero snapshots while still paying full weight-load
//!    cost. Prefill-only now calls `forward::run_prefill` directly
//!    with a freshly-allocated `KvCache`.
//! 2. **One executor per condition, not per prompt**: the engine +
//!    TEE-side weight cache + GPU device handle build once per
//!    condition; the prompt loop only resets the snapshot capture
//!    buffer between prompts.
//! 3. **`provision_weight_shared` with `Arc<Array2<half::bf16>>`**: the
//!    TEE-side weight cache holds an `Arc::clone` of the embedder's
//!    weight tensor rather than a fresh `.to_owned()` byte copy,
//!    so we pay ~3.4 GB for the cache instead of 6.8 GB on
//!    Qwen3-1.7B.
//! 4. **Pre-flight `MemAvailable` check + bounded snapshot cap**:
//!    abort before allocating if the host has < `--min-mem-gb`
//!    available; cap the per-prompt snapshot buffer to
//!    `--max-snapshots-per-prompt` so a runaway forward can't
//!    fill all of RAM.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use ndarray::Array2;

use aloepri_attack_snapshot_runner::{
    CapturingPlaintextExecutor, Condition, ExportArtifacts, export_snapshots,
};
use gelo_embedder::common::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::generation::{GenerationConfig, SamplerConfig, generate};
use gelo_embedder::decoder::kv_cache::KvCache;
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, PcieSnapshot, ReferenceCpuEngine, SnapshotConfig,
    TrustedExecutor, WeightHandle, WeightKind,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ConditionArg {
    C0,
    C1,
    C2,
    /// HD₃ Hadamard-cascade mask + default shield (round-3 gate B.3
    /// extension; mirrors C2 except `.with_hd3_mask()`).
    C3,
    /// LM-head GPU masked offload at the new
    /// `(1+k, vocab=152 064)` shape (M1.12 R3 gate). Mirrors C2 on
    /// mask family + shield + per-forward cadence; the only variable
    /// is that the LM-head projection rides the masked offload path
    /// instead of running in-TEE. Requires `--max-tokens >= 1` so the
    /// LM-head shape actually appears in the capture; prefill-only
    /// runs (`--max-tokens 0`) silently produce zero LM-head
    /// snapshots and the gate will fail loud.
    C6,
    All,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum EngineArg {
    /// `ReferenceCpuEngine` — pure-CPU rayon-parallel matmul. ~25 s/forward
    /// at Qwen3-1.7B n=32 prefill on this machine.
    Cpu,
    /// `WgpuVulkanEngine` (f32) — drops Qwen3-1.7B prefill to ~1–2 s.
    Gpu,
    /// `WgpuVulkanEngine::new_fp16` — same as `gpu` but with f16 GEMM
    /// kernels. Faster on bandwidth-bound kernels but breaks U-Verify
    /// bit-equality, irrelevant for the harness which doesn't enable
    /// verify probes.
    GpuFp16,
}

impl ConditionArg {
    fn to_conditions(self) -> Vec<Condition> {
        match self {
            ConditionArg::C0 => vec![Condition::C0Plain],
            ConditionArg::C1 => vec![Condition::C1MaskOnly],
            ConditionArg::C2 => vec![Condition::C2Default],
            ConditionArg::C3 => vec![Condition::C3Hd3],
            ConditionArg::C6 => vec![Condition::C6LmHeadOffload],
            ConditionArg::All => vec![
                Condition::C0Plain,
                Condition::C1MaskOnly,
                Condition::C2Default,
                Condition::C3Hd3,
                Condition::C6LmHeadOffload,
            ],
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    about = "Capture PCIe-side snapshots from Qwen3-1.7B for AloePri attack drivers"
)]
struct Args {
    /// Which control condition(s) to run. `all` produces three
    /// separate `(condition).safetensors`/`(condition).meta.json` pairs.
    #[arg(long, value_enum, default_value_t = ConditionArg::All)]
    condition: ConditionArg,

    /// Output directory. Defaults to `./snapshots/qwen3-1.7b/`.
    #[arg(long, default_value = "snapshots/qwen3-1.7b")]
    output: PathBuf,

    /// Prompt corpus file: one prompt per line (UTF-8). When unset
    /// the binary uses its built-in 8-prompt smoke corpus — enough
    /// for the export-roundtrip test, but **not** enough to satisfy
    /// the §2.6 acceptance gate (which calls for ≥ 64 prompts).
    #[arg(long)]
    prompts: Option<PathBuf>,

    /// Hard cap on number of prompts to process. The §2.5 default
    /// (256) holds total wall-clock under 30 minutes on a single
    /// node. Override for fast variants — Phase 3 CI uses
    /// `--max-prompts 64`.
    #[arg(long, default_value_t = 64)]
    max_prompts: usize,

    /// Max new tokens to generate per prompt. **0 = prefill only**
    /// (the path the harness wants for the §2.4 acceptance gate; the
    /// AloePri attacks all consume prefill-shape snapshots, see §4.3
    /// for the decode-regime open question).
    ///
    /// Non-zero values use `generate()` and add `max_tokens` extra
    /// decode-shape forward passes per prompt — every decode step
    /// captures its own (28 × 7) snapshots, so the file grows
    /// linearly. The Python loader exposes `prompt_idx`-keyed
    /// indexing; per-step slicing is not currently surfaced.
    #[arg(long, default_value_t = 0)]
    max_tokens: usize,

    /// Mask seed for the InProcess executor branches. Holding this
    /// stable across runs lets bench results round-trip exactly
    /// when re-running the same condition.
    #[arg(long, default_value_t = 29)]
    seed_byte: u8,

    /// Cap each prompt to this many tokens (after tokenisation,
    /// before BOS handling). Defaults to 32 — keeps the per-prompt
    /// forward pass cheap.
    #[arg(long, default_value_t = 32)]
    max_prompt_tokens: usize,

    /// Offload engine to use. `gpu` requires a Vulkan-capable wgpu
    /// adapter; `cpu` works everywhere but takes ~30× longer.
    #[arg(long, value_enum, default_value_t = EngineArg::Cpu)]
    engine: EngineArg,

    /// Cap on snapshots retained per prompt. Qwen3-1.7B prefill
    /// produces 28 layers × 7 op_kinds = 196 snapshots, so a cap of
    /// 4096 leaves ~20× headroom for decode steps. Set higher if
    /// running large prompts + many decode tokens; set lower to fail
    /// loud on unexpected layer expansion. Bounded by design — see
    /// safeguard #4 in the module header.
    #[arg(long, default_value_t = 4096)]
    max_snapshots_per_prompt: usize,

    /// Pre-flight `MemAvailable` floor in gigabytes. Abort before
    /// touching the GPU if the host has less than this much free
    /// memory. 8 GB covers Qwen3-1.7B f32 weights (~3.4 GB) + engine
    /// upload buffers + safetensors export staging.
    #[arg(long, default_value_t = 8.0)]
    min_mem_gb: f32,

    /// Bypass the pre-flight memory check. Use only when you've
    /// measured the actual headroom and know the §"OOM safeguards"
    /// limits don't apply.
    #[arg(long, default_value_t = false)]
    skip_mem_check: bool,
}

const SMOKE_PROMPTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Privacy-preserving inference matters for medical records.",
    "What is the capital of France and which river runs through it?",
    "Encryption alone does not prevent traffic-analysis side channels.",
    "Large language models can hallucinate plausible-sounding errors.",
    "Open weights make protocol-level defences load-bearing.",
    "Token-frequency analysis remains a problem for static obfuscation.",
    "Hardware attestation binds software identity to a measured boot chain.",
];

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("aloepri capture_snapshots: starting");
    eprintln!("  condition(s)             : {:?}", args.condition);
    eprintln!("  engine                   : {:?}", args.engine);
    eprintln!("  output                   : {}", args.output.display());
    eprintln!("  max_prompts              : {}", args.max_prompts);
    eprintln!("  max_tokens (0 = prefill) : {}", args.max_tokens);
    eprintln!("  max_prompt_tok           : {}", args.max_prompt_tokens);
    eprintln!("  max_snapshots/prompt     : {}", args.max_snapshots_per_prompt);

    pre_flight_mem_check(args.min_mem_gb, args.skip_mem_check)?;

    let prompts = load_prompts(args.prompts.as_deref(), args.max_prompts)?;
    eprintln!("  prompts loaded           : {}", prompts.len());

    eprintln!("loading Qwen3-1.7B (may download ~3.4 GB on first run)…");
    let (cfg, tokenizer, weights, rope) = load_qwen3_pretrained()?;
    eprintln!(
        "  cfg.num_hidden_layers={} hidden_size={} max_pos={}",
        weights.layers.len(),
        cfg.hidden_size,
        cfg.max_position_embeddings
    );

    // Build the per-handle Arc weight pool ONCE. We pay ~3.4 GB here
    // for the canonical TEE-side cache; the engine register_weight()
    // path copies internally so it pays its own ~3.4 GB later. With
    // this, the InProcessTrustedExecutor uses provision_weight_shared
    // (Arc::clone, zero extra bytes) instead of cloning the underlying
    // f32 buffer a second time.
    let weight_arcs = build_weight_arcs(&cfg, &weights);
    eprintln!(
        "  weight Arcs built: {} handles, ~{:.2} GB shared TEE-side cache",
        weight_arcs.len(),
        approx_arc_pool_gb(&weight_arcs)
    );

    let mut prompt_token_ids: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let ids = tokenizer.encode(prompt, args.max_prompt_tokens)?;
        if ids.is_empty() {
            return Err(anyhow!("tokenizer returned empty token list for prompt: {prompt}"));
        }
        prompt_token_ids.push(ids);
    }

    let conditions = args.condition.to_conditions();
    let snapshot_cfg = SnapshotConfig {
        capture_outputs: true,
        max_snapshots: Some(args.max_snapshots_per_prompt),
    };

    // C6 needs at least one decode step so `compute_logits_gpu`
    // dispatches at least once and the (1+k, vocab) shape lands in
    // the capture. Fail loud at startup rather than silently
    // produce a c6 snapshot set with zero LM-head rows.
    if conditions.contains(&Condition::C6LmHeadOffload) && args.max_tokens == 0 {
        return Err(anyhow!(
            "condition c6 requires --max-tokens >= 1 so the LM-head shape \
             appears in the capture (prefill-only never calls compute_logits)"
        ));
    }

    for cond in conditions {
        eprintln!("== condition {} ==", cond.slug());
        let t0 = std::time::Instant::now();
        let artifacts = match args.engine {
            EngineArg::Cpu => run_condition(
                cond,
                &cfg,
                &weights,
                &weight_arcs,
                &rope,
                &prompt_token_ids,
                &prompts,
                snapshot_cfg,
                args.seed_byte,
                args.max_tokens,
                &args.output,
                ReferenceCpuEngine::new(),
            )?,
            EngineArg::Gpu => run_condition(
                cond,
                &cfg,
                &weights,
                &weight_arcs,
                &rope,
                &prompt_token_ids,
                &prompts,
                snapshot_cfg,
                args.seed_byte,
                args.max_tokens,
                &args.output,
                WgpuVulkanEngine::new()?,
            )?,
            EngineArg::GpuFp16 => run_condition(
                cond,
                &cfg,
                &weights,
                &weight_arcs,
                &rope,
                &prompt_token_ids,
                &prompts,
                snapshot_cfg,
                args.seed_byte,
                args.max_tokens,
                &args.output,
                WgpuVulkanEngine::new_fp16()?,
            )?,
        };
        eprintln!(
            "  → {} snapshots in {:.1} s, safetensors: {} meta: {}",
            artifacts.snapshot_count,
            t0.elapsed().as_secs_f32(),
            artifacts.safetensors_path.display(),
            artifacts.meta_path.display(),
        );
        // Engine drops here at the end of each match arm — for the
        // GPU path that releases the device handle and pages out the
        // engine-side weight upload, keeping resident-set under
        // control across the three conditions.
    }
    eprintln!("aloepri capture_snapshots: done");
    Ok(())
}

/// Pre-flight check. Reads `/proc/meminfo` `MemAvailable:` and aborts
/// before we touch the GPU if the host has less than `min_gb`
/// available. Skipped when `skip` is true.
fn pre_flight_mem_check(min_gb: f32, skip: bool) -> Result<()> {
    if skip {
        eprintln!("  pre-flight mem check     : SKIPPED (--skip-mem-check)");
        return Ok(());
    }
    let avail = read_mem_available_gb().unwrap_or_else(|e| {
        eprintln!("  pre-flight mem check     : warn — could not read MemAvailable: {e}");
        f32::INFINITY
    });
    eprintln!("  pre-flight mem check     : MemAvailable ≈ {:.1} GB", avail);
    if avail.is_finite() && avail < min_gb {
        return Err(anyhow!(
            "pre-flight mem check failed: MemAvailable {:.1} GB < required {:.1} GB. \
             Free some memory (close llama-server containers, dockers, browser tabs) \
             or pass --skip-mem-check if you've measured the headroom.",
            avail,
            min_gb,
        ));
    }
    Ok(())
}

fn read_mem_available_gb() -> Result<f32> {
    let text = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| anyhow!("parsing MemAvailable line: {line:?}"))?;
            return Ok((kib as f32) / 1024.0 / 1024.0);
        }
    }
    Err(anyhow!("/proc/meminfo had no MemAvailable line"))
}

fn load_prompts(path: Option<&Path>, max_prompts: usize) -> Result<Vec<String>> {
    let raw: Vec<String> = match path {
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .with_context(|| format!("reading prompt corpus from {}", p.display()))?;
            text.lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        }
        None => SMOKE_PROMPTS.iter().map(|s| s.to_string()).collect(),
    };
    if raw.is_empty() {
        return Err(anyhow!("no prompts to run — pass --prompts <file> with content"));
    }
    let limit = max_prompts.min(raw.len());
    Ok(raw.into_iter().take(limit).collect())
}

/// Wrap each offload-bound weight tensor in `Arc<Array2<half::bf16>>` once
/// at startup, returning a `(WeightHandle, Arc)` vector. The engine
/// registration path uses `arc.view()`; the executor uses
/// `Arc::clone` via `provision_weight_shared`. Halves the TEE-side
/// memory footprint vs cloning the bytes a second time.
fn build_weight_arcs(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
) -> Vec<(WeightHandle, Arc<Array2<half::bf16>>)> {
    let mut arcs = Vec::with_capacity(weights.layers.len() * 7);
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        // As of 2026-05-21 `DecoderLayerWeights` stores `Arc<Array2<half::bf16>>`
        // directly; clone the Arc to get a second handle on the same
        // backing buffer — no f32 byte clone.
        for (kind, tensor) in [
            (WeightKind::Q, &layer.wq),
            (WeightKind::K, &layer.wk),
            (WeightKind::V, &layer.wv),
            (WeightKind::O, &layer.wo),
            (WeightKind::FfnGate, &layer.w_gate),
            (WeightKind::FfnUp, &layer.w_up),
            (WeightKind::FfnDown, &layer.w_down),
        ] {
            // Snapshot binary holds an external Arc pool; opt out of
            // the embedder's take-after-upload pattern by cloning the
            // Arc handle (refcount += 1) and leaving the original.
            let arc = tensor
                .as_ref()
                .expect("offloadable weight missing — fresh DecoderWeights expected here");
            arcs.push((WeightHandle::new(li16, kind), Arc::clone(arc)));
        }
    }
    arcs
}

fn approx_arc_pool_gb(arcs: &[(WeightHandle, Arc<Array2<half::bf16>>)]) -> f32 {
    let bytes: usize = arcs
        .iter()
        .map(|(_, a)| a.nrows() * a.ncols() * std::mem::size_of::<half::bf16>())
        .sum();
    bytes as f32 / 1e9
}

#[allow(clippy::too_many_arguments)]
fn run_condition<E>(
    cond: Condition,
    cfg: &DecoderConfig,
    weights: &Arc<DecoderWeights>,
    weight_arcs: &[(WeightHandle, Arc<Array2<half::bf16>>)],
    rope: &Arc<RopeTables>,
    prompt_token_ids: &[Vec<u32>],
    prompts: &[String],
    snapshot_cfg: SnapshotConfig,
    seed_byte: u8,
    max_tokens: usize,
    out_dir: &Path,
    engine: E,
) -> Result<ExportArtifacts>
where
    E: GpuOffloadEngine,
{
    // ── Build the executor ONCE per condition. ──────────────────────
    // The big win vs the old code: we register weights with the engine
    // and provision them in the TEE-side cache exactly once, instead
    // of per-prompt. With wgpu on a shared-memory iGPU that's the
    // difference between ~3.4 GB resident and ~3.4 × N GB resident
    // across N prompts.

    let (shield_k, shield_energy_scale, per_forward_mask) = match cond {
        Condition::C0Plain => (0_usize, 0.0_f32, false),
        Condition::C1MaskOnly => (0_usize, 0.0_f32, false),
        Condition::C2Default => (8_usize, 4.0_f32, true),
        // C3 mirrors C2 on shield + per-forward-pass cadence; the mask
        // family is the only variable. Same numbers go into meta.json
        // so the Python loader treats C2 and C3 as comparable rows.
        Condition::C3Hd3 => (8_usize, 4.0_f32, true),
        // C6 mirrors C2 on every parameter the meta.json records. The
        // only delta is that the LM-head projection rides the masked
        // offload path; the capture sees `(1+k, vocab)` masked outputs
        // alongside the existing QKV / O / gate / up / down shapes.
        Condition::C6LmHeadOffload => (8_usize, 4.0_f32, true),
    };

    let mut executor: ExecVariant<E> = match cond {
        Condition::C0Plain => ExecVariant::Plain(
            CapturingPlaintextExecutor::new(engine).with_snapshot_capture(snapshot_cfg),
        ),
        Condition::C1MaskOnly => ExecVariant::InProc(
            InProcessTrustedExecutor::with_seed(engine, MaskSeed::from_bytes([seed_byte; 32]))
                .with_per_offload_mask()
                .with_snapshot_capture(snapshot_cfg),
        ),
        Condition::C2Default => {
            let exec = InProcessTrustedExecutor::with_seed(
                engine,
                MaskSeed::from_bytes([seed_byte; 32]),
            )
            .with_snapshot_capture(snapshot_cfg);
            debug_assert_eq!(exec.shield_config().k, 8);
            ExecVariant::InProc(exec)
        }
        Condition::C3Hd3 => {
            // Same builder as C2 except `.with_hd3_mask()` swaps the
            // mask family. Per-forward-pass cadence + shield(8, 4.0)
            // are inherited from `with_seed` defaults.
            let exec = InProcessTrustedExecutor::with_seed(
                engine,
                MaskSeed::from_bytes([seed_byte; 32]),
            )
            .with_hd3_mask()
            .with_snapshot_capture(snapshot_cfg);
            debug_assert_eq!(exec.shield_config().k, 8);
            debug_assert_eq!(exec.mask_kind(), gelo_protocol::MaskKind::Hd3);
            ExecVariant::InProc(exec)
        }
        Condition::C6LmHeadOffload => {
            // C6 — same builder as C2 (paper-parity Haar + shield(8,
            // 4.0)); the variable is that the LM-head projection
            // rides the masked offload path. The LM-head weight
            // `WeightKind::LmHead` is registered separately below
            // (after the standard 7-projection provisioning loop) so
            // the substrate has it available when
            // `compute_logits_gpu` opens its per-token
            // `begin_forward_pass(1)` bracket.
            let exec = InProcessTrustedExecutor::with_seed(
                engine,
                MaskSeed::from_bytes([seed_byte; 32]),
            )
            .with_snapshot_capture(snapshot_cfg);
            debug_assert_eq!(exec.shield_config().k, 8);
            ExecVariant::InProc(exec)
        }
    };

    // Provision weights once. For the InProc branches we use
    // `provision_weight_shared` so the TEE-side cache holds an Arc
    // rather than cloning the f32 bytes (safeguard #3). The Plain
    // wrapper doesn't keep a TEE-side cache anyway, so it falls back
    // to the standard register_weight via the inner PlaintextExecutor.
    match &mut executor {
        ExecVariant::Plain(e) => {
            for (handle, arc) in weight_arcs {
                e.provision_weight_bf16(*handle, arc.view())?;
            }
        }
        ExecVariant::InProc(e) => {
            for (handle, arc) in weight_arcs {
                e.provision_weight_bf16_shared(*handle, Arc::clone(arc))?;
            }
        }
    }

    // M1.12 R3 — LM head is the production default-and-only path
    // for `generation::generate`. Provision the tied-embedding
    // transpose unconditionally when `max_tokens > 0` so the
    // generate() loop has somewhere to dispatch the per-token
    // offload. C6 is the condition the gate actually scrutinises;
    // C0–C3 just need the registration so prefill+decode runs at all.
    // Skipped at `max_tokens == 0` (`run_prefill` direct) — saves the
    // transient bf16 transpose.
    if max_tokens > 0 {
        use gelo_embedder::decoder::weights::provision_lm_head_into;
        match &mut executor {
            ExecVariant::InProc(e) => provision_lm_head_into(weights, e)?,
            ExecVariant::Plain(e) => {
                let lm_head_t = weights
                    .token_embedding
                    .t()
                    .as_standard_layout()
                    .to_owned();
                e.provision_weight_bf16(
                    WeightHandle::new(0, WeightKind::LmHead),
                    lm_head_t.view(),
                )?;
            }
        }
    } else if matches!(cond, Condition::C6LmHeadOffload) {
        return Err(anyhow!(
            "c6 captures require --max-tokens >= 1 so the LM-head shape \
             appears in the capture (prefill-only never calls compute_logits)"
        ));
    }

    // Accumulators for the per-condition export.
    let mut all_snapshots: Vec<PcieSnapshot> = Vec::new();
    let mut prompt_indices: Vec<usize> = Vec::new();

    for (prompt_idx, (prompt_ids, prompt_text)) in prompt_token_ids
        .iter()
        .zip(prompts.iter())
        .enumerate()
    {
        // Make sure capture starts at zero rows for this prompt.
        let _ = executor.drain();

        let snaps = run_one_prompt(
            cfg,
            weights,
            rope,
            &mut executor,
            prompt_ids,
            max_tokens,
        )?;

        eprintln!(
            "  prompt[{prompt_idx:03}] (n_tok={:>4}) → {} snapshots — {:?}…",
            prompt_ids.len(),
            snaps.len(),
            &prompt_text.chars().take(48).collect::<String>(),
        );
        if snaps.is_empty() {
            return Err(anyhow!(
                "prompt {prompt_idx} produced 0 snapshots — check that the prefill code path \
                 is wired to the executor's offload_* hooks. With --max-tokens 0 we expect \
                 28 × 7 = 196 snapshots per prompt on Qwen3-1.7B."
            ));
        }
        prompt_indices.extend(std::iter::repeat_n(prompt_idx, snaps.len()));
        all_snapshots.extend(snaps);
    }

    // Re-stamp seq_idx contiguously across the whole dump so the
    // safetensors keys are unique and ordered.
    for (i, snap) in all_snapshots.iter_mut().enumerate() {
        snap.seq_idx = i;
    }

    let basename = cond.slug();
    export_snapshots(
        &all_snapshots,
        &prompt_indices,
        "Qwen/Qwen3-1.7B",
        cond,
        prompt_token_ids.to_vec(),
        shield_k,
        shield_energy_scale,
        per_forward_mask,
        0,
        out_dir,
        basename,
    )
}

/// Run one prompt's forward pass under the given executor, draining
/// the captured snapshots.
///
/// When `max_tokens == 0` we call `forward::run_prefill` directly
/// with a freshly-allocated `KvCache`. Going through
/// `generate(max_tokens=0)` would short-circuit (see
/// `generation.rs:144`) and produce zero snapshots — the bug that
/// originally turned a 132 s C0 sweep into a "0 snapshots / prompt"
/// result.
fn run_one_prompt<E: GpuOffloadEngine>(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    executor: &mut ExecVariant<E>,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Result<Vec<PcieSnapshot>> {
    if max_tokens == 0 {
        let mut kv = KvCache::new(weights.layers.len(), prompt_ids.len(), cfg.kv_dim());
        match executor {
            ExecVariant::Plain(e) => {
                let _ = forward::run_prefill(cfg, weights, rope, e, prompt_ids, &mut kv)?;
            }
            ExecVariant::InProc(e) => {
                let _ = forward::run_prefill(cfg, weights, rope, e, prompt_ids, &mut kv)?;
            }
        }
    } else {
        let gen_cfg = GenerationConfig {
            max_tokens,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        };
        match executor {
            ExecVariant::Plain(e) => {
                let _ = generate(cfg, weights, rope, e, prompt_ids, &gen_cfg)?;
            }
            ExecVariant::InProc(e) => {
                let _ = generate(cfg, weights, rope, e, prompt_ids, &gen_cfg)?;
            }
        }
    }
    Ok(executor.drain())
}

/// Owns either branch of the per-condition executor. C0 uses
/// `CapturingPlaintextExecutor` (no mask, no shield); C1/C2 share
/// `InProcessTrustedExecutor` with different builder chains. We
/// erase the difference through this enum so `run_one_prompt` is
/// one function rather than three.
enum ExecVariant<E: GpuOffloadEngine> {
    Plain(CapturingPlaintextExecutor<E>),
    InProc(InProcessTrustedExecutor<E>),
}

impl<E: GpuOffloadEngine> ExecVariant<E> {
    fn drain(&mut self) -> Vec<PcieSnapshot> {
        match self {
            ExecVariant::Plain(e) => e.drain_pcie_snapshots(),
            ExecVariant::InProc(e) => e.drain_pcie_snapshots(),
        }
    }
}

fn load_qwen3_pretrained()
-> Result<(DecoderConfig, HfTokenizer, Arc<DecoderWeights>, Arc<RopeTables>)> {
    let variant = Qwen3Variant::Q1_7B;
    let cfg = variant.config();
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("building HF hub API client")?;
    let repo = api.model(variant.hf_model_id().to_string());

    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("downloading tokenizer.json")?;
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

/// Resolve either the single-file or sharded HF safetensors layout.
/// Lifted from `qwen3_generation_e2e.rs` — the embedder's
/// `find_safetensors_shards` helper is module-private.
fn find_safetensors_shards(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    if let Ok(p) = repo.get("model.safetensors") {
        return Ok(vec![p]);
    }
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("model has neither model.safetensors nor model.safetensors.index.json")?;
    let index_bytes = std::fs::read(&index_path)?;
    let index: serde_json::Value = serde_json::from_slice(&index_bytes)?;
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
        paths.push(repo.get(&name)?);
    }
    Ok(paths)
}
