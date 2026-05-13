//! Step-by-step runtime breakdown for the BGE-base + GELO mask (Vulkan +
//! in-process TEE) path. Uses `gelo_protocol::profile`'s thread-local
//! aggregator to attribute wall-clock to:
//!   - `gelo:mask_sample`  — Haar QR sampling of the per-batch mask A
//!   - `gelo:mask_apply`   — U = A · H
//!   - `engine:matmul`     — offloaded U · W on the burn-cubecl engine
//!   - `gelo:mask_unapply` — Aᵀ · (U · W)
//!   - `gelo:shield_stack` / `gelo:strip_shield` — TwinShield (off in default config)
//!
//! Run:
//!   cargo test -p gelo-gpu-wgpu --release --test bge_gelo_mask_breakdown \
//!     -- --ignored --nocapture
//!
//! The corpus is 5 short docs + 5 queries to match the BEIR_DOCS=5 checkpoint.

use std::time::Instant;

use gelo_embedder::GeloBertEmbedder;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{InProcessTrustedExecutor, MaskSeed, profile};
use rag_core::Embedder;

const DOCS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog. It is a pangram used in typography.",
    "A common rheumatologic disease that primarily affects synovial joints is rheumatoid arthritis.",
    "Photosynthesis converts light energy into chemical energy stored in glucose by plants.",
    "Random forests are an ensemble learning method using many decision trees for classification.",
    "The Treaty of Westphalia in 1648 ended the Thirty Years War and established modern statecraft.",
];

const QUERIES: &[&str] = &[
    "what is rheumatoid arthritis",
    "ensemble learning algorithm",
    "plant biology energy conversion",
    "early modern european history",
    "pangram example",
];

fn phase(label: &str, texts: &[String], embedder: &mut dyn Embedder) {
    profile::reset();
    let t0 = Instant::now();
    let embeds = embedder.embed(texts).expect("embed");
    let wall = t0.elapsed();

    let prof = profile::snapshot();
    eprintln!();
    eprintln!(
        "─── {label}: {} texts, wall-clock = {:.2} ms ({:.2} ms/text)",
        texts.len(),
        wall.as_secs_f64() * 1000.0,
        wall.as_secs_f64() * 1000.0 / texts.len() as f64,
    );
    prof.dump(&format!("{label} — gelo_protocol::profile buckets"));
    eprintln!("[shape] {} embeds × {} dims", embeds.len(), embeds[0].len());
}

#[test]
#[ignore]
fn bge_gelo_mask_breakdown() {
    let _ = env_logger::builder()
        .filter_module("cubecl_runtime", log::LevelFilter::Info)
        .is_test(false)
        .try_init();
    let gpu = WgpuVulkanEngine::new().expect("Vulkan adapter");
    eprintln!(
        "[adapter] {} ({:?})",
        gpu.adapter_info().name,
        gpu.adapter_info().device_type
    );
    assert!(gpu.is_real_gpu(), "need real Vulkan GPU, not lavapipe");

    let executor = InProcessTrustedExecutor::with_seed(
        gpu.clone_shared(),
        MaskSeed::from_bytes([7u8; 32]),
    );
    let mut emb: Box<dyn Embedder> = Box::new(
        GeloBertEmbedder::from_pretrained("BAAI/bge-base-en-v1.5", executor)
            .expect("load BGE-base"),
    );

    // Warm-up call so the first ingest/query phase doesn't bake in
    // autotune + kernel-compile costs that would otherwise dominate
    // the "ingest 5 docs" numbers.
    let warmup_t = Instant::now();
    let _ = emb.embed(&[
        "warmup text used to trigger autotune + kernel compile for these shapes".to_string(),
    ]);
    eprintln!(
        "[warmup] 1 text, wall-clock = {:.2} ms (autotune + kernel compile bake-in)",
        warmup_t.elapsed().as_secs_f64() * 1000.0,
    );

    let docs: Vec<String> = DOCS.iter().map(|s| s.to_string()).collect();
    let queries: Vec<String> = QUERIES.iter().map(|s| s.to_string()).collect();

    phase("ingest (5 docs)", &docs, emb.as_mut());
    phase("query (5 queries)", &queries, emb.as_mut());

    // Second pass — confirm warm steady-state.
    phase("ingest (warm rerun)", &docs, emb.as_mut());
    phase("query (warm rerun)", &queries, emb.as_mut());
}

