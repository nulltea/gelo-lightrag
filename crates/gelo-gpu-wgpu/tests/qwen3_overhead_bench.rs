//! Qwen3-Embedding decoder overhead benchmark.
//!
//! Reports per-text latency across the four cells that span the M3
//! protocol-overhead matrix:
//!
//!   1. cpu_plain — no GPU, no mask (CPU upper bound)
//!   2. gpu_plain — Vulkan, no mask, no OutAttnMult (unprotected baseline)
//!   3. gpu_gelo+outattn — GELO mask + TwinShield OutAttnMult (privacy-only)
//!   4. gpu_full_stack — GELO mask + shield + OutAttnMult + U-Verify (full privacy + integrity)
//!
//! Why only four cells?  Qwen3-Embedding-0.6B's weights are ~2.4 GB in
//! f32 once loaded. The earlier 6-cell version of this bench OOM-killed
//! because each embedder re-allocated the weights (CPU side), each
//! Vulkan engine re-uploaded them (GPU side, integrated GPU sharing
//! system RAM), and each verify-enabled executor cloned them yet again
//! into the TEE-side U-Verify cache. The peak working-set hit ~33 GB
//! on a 64 GB box (with the iGPU competing for the same RAM) and
//! triggered the kernel OOM killer.
//!
//! The fix here:
//!   - load the model **once** via `from_pretrained`,
//!   - share its `Arc<DecoderWeights>` and `Arc<RopeTables>` across all
//!     four embedders,
//!   - share a **single** `WgpuVulkanEngine` (via `clone_shared`) across
//!     every GPU-side executor so weights are uploaded once,
//!   - enable U-Verify on a single executor so the TEE-side weight cache
//!     is allocated only once.
//!
//! Peak working set is now ~7-8 GB (1× CPU weights, 1× GPU weights,
//! 1× TEE verify cache).
//!
//! Downloads ~1.2 GB on first run (`Qwen/Qwen3-Embedding-0.6B`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_embedder::GeloQwenEmbedder;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, PlaintextExecutor, RayonCpuEngine, ShieldConfig};
use rag_core::Embedder;

const MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";
const WARMUP_ITERS: usize = 1;
const MEASURE_ITERS: usize = 2;

fn corpus() -> Vec<String> {
    vec![
        "Confidential computing protects user data inside attested enclaves.".into(),
        "Rotary position embeddings rotate query and key vectors per token.".into(),
        "TwinShield's OutAttnMult outsources the attention QK^T matmul to the GPU.".into(),
        "U-Verify uses Freivalds-style probes to catch a tampering accelerator.".into(),
    ]
}

struct Bench {
    name: &'static str,
    iter_times: Vec<Duration>,
    embeds_per_iter: usize,
}

impl Bench {
    fn total(&self) -> Duration {
        self.iter_times.iter().copied().sum()
    }
    fn mean_iter_ms(&self) -> f64 {
        self.total().as_secs_f64() * 1000.0 / self.iter_times.len().max(1) as f64
    }
    fn stddev_iter_ms(&self) -> f64 {
        let mean = self.mean_iter_ms();
        let var: f64 = self
            .iter_times
            .iter()
            .map(|d| {
                let v = d.as_secs_f64() * 1000.0;
                (v - mean).powi(2)
            })
            .sum::<f64>()
            / self.iter_times.len().max(1) as f64;
        var.sqrt()
    }
    fn per_text_ms(&self) -> f64 {
        self.mean_iter_ms() / self.embeds_per_iter.max(1) as f64
    }
    fn throughput(&self) -> f64 {
        let total = (self.iter_times.len() * self.embeds_per_iter) as f64;
        total / self.total().as_secs_f64()
    }
}

fn time_once(embedder: &mut dyn Embedder, texts: &[String]) -> Duration {
    let t0 = Instant::now();
    let _ = embedder.embed(texts).expect("embed");
    t0.elapsed()
}

fn run_interleaved(
    names: &[&'static str],
    embedders: &mut [&mut dyn Embedder],
    texts: &[String],
) -> Vec<Bench> {
    assert_eq!(names.len(), embedders.len());
    for _ in 0..WARMUP_ITERS {
        for e in embedders.iter_mut() {
            let _ = time_once(&mut **e, texts);
        }
    }
    let mut iter_times: Vec<Vec<Duration>> =
        (0..embedders.len()).map(|_| Vec::with_capacity(MEASURE_ITERS)).collect();
    for _ in 0..MEASURE_ITERS {
        for (idx, e) in embedders.iter_mut().enumerate() {
            iter_times[idx].push(time_once(&mut **e, texts));
        }
    }
    iter_times
        .into_iter()
        .zip(names.iter())
        .map(|(times, name)| Bench {
            name,
            iter_times: times,
            embeds_per_iter: texts.len(),
        })
        .collect()
}

fn pct_over(b: &Bench, base: &Bench) -> f64 {
    100.0 * (b.total().as_secs_f64() / base.total().as_secs_f64() - 1.0)
}

/// Resident-set-size snapshot for the current process, in bytes. Logged
/// around major allocations so the bench output records the working-set
/// curve and we don't have to guess where OOM pressure came from again.
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

fn fmt_gib(bytes: usize) -> String {
    format!("{:.2} GiB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
}

#[test]
#[ignore = "downloads ~1.2 GB Qwen3-Embedding-0.6B; requires Vulkan GPU; ~7 GB RAM"]
fn qwen3_overhead_breakdown() {
    eprintln!("RSS before any load: {}", fmt_gib(rss_bytes()));

    let texts = corpus();
    eprintln!(
        "Qwen3 overhead benchmark — model={MODEL} N={} iterations={MEASURE_ITERS} warmup={WARMUP_ITERS}",
        texts.len(),
    );

    // 1. Load weights ONCE via cpu_plain (the simplest executor). Pull out
    //    the shared Arcs to seed the other configs.
    eprintln!("[load] downloading + materialising Qwen3 weights...");
    let mut cpu_plain = GeloQwenEmbedder::from_pretrained(
        MODEL,
        PlaintextExecutor::new(RayonCpuEngine::new()),
    )
    .expect("Qwen3 from_pretrained")
    .with_out_attn_mult(false);
    let weights_arc: Arc<DecoderWeights> = cpu_plain.weights_arc();
    let rope_arc: Arc<RopeTables> = cpu_plain.rope_arc();
    let tokenizer = cpu_plain.tokenizer().clone();
    let cfg: DecoderConfig = cpu_plain.config().clone();
    eprintln!("RSS after CPU model load: {}", fmt_gib(rss_bytes()));

    // 2. One shared Vulkan engine. All GPU-side configs upload weights into
    //    THIS engine's cache once and reuse.
    let gpu_root = WgpuVulkanEngine::new().expect("Vulkan adapter");
    let adapter_line = format!(
        "{} ({:?}, driver={}, info={})",
        gpu_root.adapter_info().name,
        gpu_root.adapter_info().device_type,
        gpu_root.adapter_info().driver,
        gpu_root.adapter_info().driver_info,
    );
    assert!(gpu_root.is_real_gpu(), "bench needs real GPU hardware");

    eprintln!("[gpu_plain] building...");
    let mut gpu_plain = GeloQwenEmbedder::new(
        cfg.clone(),
        tokenizer.clone(),
        Arc::clone(&weights_arc),
        Arc::clone(&rope_arc),
        PlaintextExecutor::new(gpu_root.clone_shared()),
    )
    .expect("gpu_plain")
    .with_out_attn_mult(false);
    eprintln!("RSS after gpu_plain build: {}", fmt_gib(rss_bytes()));

    eprintln!("[gpu_gelo_outattn] building...");
    let mut gpu_gelo_outattn = {
        let mut cfg = cfg.clone();
        cfg.use_out_attn_mult = true;
        // Force OutAttnMult on at any n so the bench measures it; the
        // production auto-switch (n ≥ hidden_size) would otherwise route
        // these short prompts through in-TEE attention.
        cfg.out_attn_mult_min_seq_len = Some(0);
        GeloQwenEmbedder::new(
            cfg,
            tokenizer.clone(),
            Arc::clone(&weights_arc),
            Arc::clone(&rope_arc),
            InProcessTrustedExecutor::with_seed(
                gpu_root.clone_shared(),
                MaskSeed::from_bytes([1u8; 32]),
            ),
        )
        .expect("gpu_gelo_outattn")
    };
    eprintln!("RSS after gpu_gelo_outattn build: {}", fmt_gib(rss_bytes()));

    // U-Verify on Qwen3 with k=8 probes burns ~4 GFLOPS of CPU per text on
    // the integrity check alone (1024- and 3072-wide GEMVs against every
    // offloaded weight, 8x). Drop to k=2 for the bench — that's still
    // (1/6)^2 ≈ 2.8 % undetected-tamper soundness, plenty to demonstrate
    // the protocol cell.  Production deployment with k=8 retains the
    // ~2.4e-7 figure but pays this cost; the trade-off is exposed via the
    // `with_verify_probes` knob.
    const BENCH_VERIFY_PROBES: usize = 2;
    eprintln!(
        "[gpu_full_stack] building (shield + U-Verify k={BENCH_VERIFY_PROBES}; allocates the only TEE-side verify cache)..."
    );
    let mut gpu_full_stack = {
        let mut cfg = cfg.clone();
        cfg.use_out_attn_mult = true;
        // Same OutAttnMult force-on as gpu_gelo_outattn — measure the
        // full protocol path, not the auto-switch's short-input
        // fallback.
        cfg.out_attn_mult_min_seq_len = Some(0);
        let exec = InProcessTrustedExecutor::with_shield(
            gpu_root.clone_shared(),
            MaskSeed::from_bytes([2u8; 32]),
            ShieldConfig::new(8, 6.0),
        )
        .with_verify_probes(BENCH_VERIFY_PROBES);
        GeloQwenEmbedder::new(
            cfg,
            tokenizer.clone(),
            Arc::clone(&weights_arc),
            Arc::clone(&rope_arc),
            exec,
        )
        .expect("gpu_full_stack")
    };
    eprintln!("RSS after gpu_full_stack build: {}", fmt_gib(rss_bytes()));

    let names = [
        "cpu_plain",
        "gpu_plain (no privacy)",
        "gpu + GELO + OutAttnMult",
        "gpu + shield + OutAttn + U-Verify",
    ];
    let mut embedders: Vec<&mut dyn Embedder> = vec![
        &mut cpu_plain,
        &mut gpu_plain,
        &mut gpu_gelo_outattn,
        &mut gpu_full_stack,
    ];
    let results = run_interleaved(&names, embedders.as_mut_slice(), &texts);
    eprintln!("RSS after measurement: {}", fmt_gib(rss_bytes()));

    eprintln!();
    eprintln!("{}", "=".repeat(114));
    eprintln!("Vulkan adapter: {adapter_line}");
    eprintln!("{}", "=".repeat(114));
    eprintln!(
        "{:<40} {:>12} {:>16} {:>10} {:>16} {:>11}",
        "config", "total (s)", "per-text (ms)", "±σ ms", "throughput (1/s)", "vs gpu_plain"
    );
    eprintln!("{}", "-".repeat(114));
    let baseline = &results[1];
    for r in &results {
        let overhead = if std::ptr::eq(r, baseline) {
            "(base)".to_string()
        } else {
            format!("{:+.1}%", pct_over(r, baseline))
        };
        eprintln!(
            "{:<40} {:>12.3} {:>16.2} {:>10.2} {:>16.2} {:>11}",
            r.name,
            r.total().as_secs_f64(),
            r.per_text_ms(),
            r.stddev_iter_ms() / r.embeds_per_iter as f64,
            r.throughput(),
            overhead,
        );
    }
    eprintln!("{}", "=".repeat(114));
    eprintln!(
        "Notes: iterations={MEASURE_ITERS}, warmup={WARMUP_ITERS}, model Qwen3-Embedding-0.6B (28 layers, hidden=1024, GQA 16/8)."
    );
    eprintln!(
        "       OutAttnMult: TwinShield 4-partition Q·K^T offload (16 per-head OutAttnMult dispatches per layer)."
    );
    eprintln!("       U-Verify probes k=2 for this run (≈ 2.8 % undetected-tamper); production k=8 → 2.4e-7.");
}
