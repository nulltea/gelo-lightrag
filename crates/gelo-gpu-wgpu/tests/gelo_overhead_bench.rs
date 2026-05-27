//! Benchmark: GELO TEE-mask + Vulkan offload vs. unprotected Vulkan offload.
//!
//! Three configurations are timed against an identical text corpus and BERT
//! model (`bge-small-en-v1.5`):
//!
//! 1. **Untrusted GPU baseline** — `PlaintextExecutor<WgpuVulkanEngine>`.
//!    Activations are shipped to the GPU in cleartext. This is the "no
//!    privacy" upper bound on speed.
//! 2. **GELO masked GPU** — `InProcessTrustedExecutor<WgpuVulkanEngine>`,
//!    `ShieldConfig::NONE`. Per-batch fresh orthogonal mask on the token
//!    axis; GPU never sees plaintext activations. This is the headline
//!    GELO configuration.
//! 3. **GELO masked + shield** — adds `k=8, energy_scale=6` shield rows
//!    to defeat the Gram-matrix leak (cf. `bss_recovery.rs`). The mask
//!    matrix grows from `(n × n)` to `((n+k) × (n+k))`.
//!
//! An optional CPU-engine plaintext run is also reported as a sanity
//! reference. The benchmark is gated behind `#[ignore]` because it
//! downloads ~130 MB on first run and requires a Vulkan adapter.
//!
//! Reported metrics per configuration:
//!   - total wall time over N texts × R iterations (warmup discarded)
//!   - per-text mean latency
//!   - throughput (texts/sec)
//!
//! Overhead percentages are computed relative to the untrusted GPU baseline.

use std::time::{Duration, Instant};

use gelo_embedder::GeloBertEmbedder;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    InProcessTrustedExecutor, PlaintextExecutor, ReferenceCpuEngine, ShieldConfig,
};
use rag_core::Embedder;

const MODEL: &str = "BAAI/bge-base-en-v1.5";
const WARMUP_ITERS: usize = 5;
const MEASURE_ITERS: usize = 10;

fn corpus() -> Vec<String> {
    vec![
        "Confidential computing protects user data inside attested enclaves.".into(),
        "Vulkan exposes the GPU compute pipeline directly through a thin runtime.".into(),
        "GELO masks hidden states with a fresh orthogonal matrix per batch.".into(),
        "Retrieval-augmented generation depends on accurate embedding similarity.".into(),
        "Rust enforces memory safety through ownership, borrowing, and lifetimes.".into(),
        "TwinShield adds an integrity-verifying hash row alongside the outsourced matmul.".into(),
        "Shield vectors raise UᵀU above HᵀH so the Gram-matrix leak no longer applies.".into(),
        "BERT-base embedding models compute mean-pooled token representations followed by L2 normalization.".into(),
        "Postgres uses B-tree indexes for common equality and range lookups on ordered columns.".into(),
        "The trusted side never exposes plaintext residual streams to the offloaded accelerator.".into(),
        "Householder reflectors build a Haar-uniform orthogonal mask in O(n cubed) work.".into(),
        "Per-batch refresh of the mask reduces deobfuscation to a single-batch blind source separation problem.".into(),
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
        self.total().as_secs_f64() * 1000.0 / self.iter_times.len() as f64
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
            / self.iter_times.len() as f64;
        var.sqrt()
    }
    fn per_text_ms(&self) -> f64 {
        self.mean_iter_ms() / self.embeds_per_iter.max(1) as f64
    }
    fn throughput(&self) -> f64 {
        let total_embeds = (self.iter_times.len() * self.embeds_per_iter) as f64;
        total_embeds / self.total().as_secs_f64()
    }
}

fn time_once(embedder: &mut dyn Embedder, texts: &[String]) -> Duration {
    let t0 = Instant::now();
    let _ = embedder.embed(texts).expect("embed");
    t0.elapsed()
}

/// Interleave configs across iterations so thermal / GPU-cache state affects
/// all configs symmetrically — `[A, B, C, D] × N` rather than `A×N, B×N, …`.
fn run_interleaved(
    names: &[&'static str],
    embedders: &mut [&mut dyn Embedder],
    texts: &[String],
) -> Vec<Bench> {
    assert_eq!(names.len(), embedders.len());

    // Warmup pass — equal exposure for each config.
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

fn pct_over(b: &Bench, baseline: &Bench) -> f64 {
    100.0 * (b.total().as_secs_f64() / baseline.total().as_secs_f64() - 1.0)
}

#[test]
#[ignore = "downloads bge-base (~440 MB) on first run and requires a real Vulkan GPU"]
fn gelo_overhead_vs_untrusted_baseline() {
    let texts = corpus();
    eprintln!(
        "GELO overhead benchmark — model={} N={} texts iterations={} (warmup={})",
        MODEL,
        texts.len(),
        MEASURE_ITERS,
        WARMUP_ITERS,
    );

    // CPU reference (no GPU, no mask) — for context only.
    eprintln!("\n[cpu_plaintext] loading...");
    let cpu_engine = ReferenceCpuEngine::new();
    let mut cpu_plain =
        GeloBertEmbedder::from_pretrained(MODEL, PlaintextExecutor::new(cpu_engine))
            .expect("load bge-small (cpu)");

    // Untrusted GPU baseline — Vulkan offload, no mask.
    eprintln!("[gpu_plaintext] loading...");
    let gpu_for_plain = WgpuVulkanEngine::new().expect("Vulkan adapter");
    let gpu_info_line = format!(
        "Vulkan adapter: {} ({:?}, driver={}, info={})",
        gpu_for_plain.adapter_info().name,
        gpu_for_plain.adapter_info().device_type,
        gpu_for_plain.adapter_info().driver,
        gpu_for_plain.adapter_info().driver_info,
    );
    assert!(
        gpu_for_plain.is_real_gpu(),
        "benchmark must run on real GPU hardware, got {:?}",
        gpu_for_plain.adapter_info(),
    );
    let mut gpu_plain =
        GeloBertEmbedder::from_pretrained(MODEL, PlaintextExecutor::new(gpu_for_plain))
            .expect("load bge-small (gpu plain)");

    // GELO masked GPU (no shield).
    eprintln!("[gpu_gelo] loading...");
    let gpu_for_gelo = WgpuVulkanEngine::new().expect("Vulkan adapter (gelo)");
    let mut gpu_gelo = GeloBertEmbedder::from_pretrained(
        MODEL,
        InProcessTrustedExecutor::with_seed(gpu_for_gelo, MaskSeed::from_bytes([0xA7; 32])),
    )
    .expect("load bge-small (gpu gelo)");

    // GELO masked GPU + shield.
    eprintln!("[gpu_gelo_shield] loading...");
    let gpu_for_shield = WgpuVulkanEngine::new().expect("Vulkan adapter (shield)");
    let shield_exec = InProcessTrustedExecutor::with_shield(
        gpu_for_shield,
        MaskSeed::from_bytes([0xB1; 32]),
        ShieldConfig::new(8, 6.0),
    );
    assert!(shield_exec.shield_config().enabled());
    let mut gpu_gelo_shield = GeloBertEmbedder::from_pretrained(MODEL, shield_exec)
        .expect("load bge-small (gpu gelo+shield)");

    // Measure — interleaved per iteration.
    let names = [
        "cpu_plaintext",
        "gpu_plaintext_baseline",
        "gpu_gelo_mask",
        "gpu_gelo_mask+shield(k=8,scale=6)",
    ];
    let mut embedders: Vec<&mut dyn Embedder> = vec![
        &mut cpu_plain,
        &mut gpu_plain,
        &mut gpu_gelo,
        &mut gpu_gelo_shield,
    ];
    let results = run_interleaved(&names, embedders.as_mut_slice(), &texts);
    let r_cpu_plain = &results[0];
    let r_gpu_plain = &results[1];
    let r_gpu_gelo = &results[2];
    let r_gpu_shield = &results[3];

    // Pretty-print.
    eprintln!();
    eprintln!("{}", "=".repeat(108));
    eprintln!("{gpu_info_line}");
    eprintln!("{}", "=".repeat(108));
    eprintln!(
        "{:<42} {:>12} {:>16} {:>10} {:>16} {:>9}",
        "config", "total (s)", "per-text (ms)", "±σ ms", "throughput (1/s)", "vs gpu"
    );
    eprintln!("{}", "-".repeat(108));
    for r in [r_cpu_plain, r_gpu_plain, r_gpu_gelo, r_gpu_shield] {
        let overhead = if std::ptr::eq(r, r_gpu_plain) {
            "(base)".into()
        } else {
            format!("{:+.1}%", pct_over(r, r_gpu_plain))
        };
        eprintln!(
            "{:<42} {:>12.3} {:>16.2} {:>10.2} {:>16.2} {:>9}",
            r.name,
            r.total().as_secs_f64(),
            r.per_text_ms(),
            r.stddev_iter_ms() / r.embeds_per_iter as f64,
            r.throughput(),
            overhead,
        );
    }
    eprintln!("{}", "=".repeat(108));
    eprintln!(
        "Notes: iterations={MEASURE_ITERS}, warmup={WARMUP_ITERS}. ±σ is per-text from per-iteration variance."
    );
    eprintln!(
        "       'vs gpu' = wall-time overhead vs gpu_plaintext_baseline (the unprotected GPU path)."
    );
    eprintln!(
        "       SGEMM provided by cubecl-matmul (autotuned, workgroup-tiled). At small per-text"
    );
    eprintln!(
        "       sequence lengths the per-dispatch overhead still dominates the GEMM compute time,"
    );
    eprintln!(
        "       so the absolute GPU vs CPU gap stays narrow; the protocol overhead ratios remain"
    );
    eprintln!(
        "       comparable to the naive-kernel baseline because both are dispatch-bound."
    );

    assert_eq!(r_gpu_gelo.embeds_per_iter, r_gpu_plain.embeds_per_iter);
    assert_eq!(r_gpu_shield.embeds_per_iter, r_gpu_plain.embeds_per_iter);
}
