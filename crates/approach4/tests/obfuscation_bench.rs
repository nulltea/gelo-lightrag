//! Two wall-clock comparisons, run on the **Vulkan GPU** (`WgpuVulkanEngine`)
//! so the numbers are directly comparable to `gelo.md`'s published
//! 371 / 395 / 460 ms per text and to the existing
//! `qwen3_overhead_breakdown.rs` bench.
//!
//! ### Bench A — GELO vs GELO + DP-Forward (apples to apples)
//!
//! Same threat model, same embedder (`GeloQwenEmbedder` on Vulkan + GELO
//! mask + shield + OutAttnMult), same CAPRISE-encrypted index. The diff
//! between the two configurations is the DP-Forward overhead — clip + a
//! single isotropic Gaussian sample at the pooled-output step.
//!
//! ### Bench B — RemoteRAG (alternative path; can also compose with GELO)
//!
//! Different threat model from Bench A. GELO defends *activations during
//! inference* against an untrusted GPU/host beneath a CVM. RemoteRAG defends
//! *query content and retrieval-time exposure* against an untrusted
//! retrieval server. The two protocols target different adversaries — a
//! deployment typically picks one based on which boundary it doesn't
//! trust, though they *can* also stack as defence-in-depth (GELO inside
//! the client CVM, RemoteRAG over the network to a remote retrieval
//! server).
//!
//! For a fair perf comparison we use the **same `GeloQwenEmbedder` on
//! Vulkan** in Bench B as in Bench A. The delta between the two benches is
//! then the pure RemoteRAG protocol overhead (planar-Laplace + Paillier
//! rerank, replacing CAPRISE-encrypted-index + cosine-over-ciphertexts),
//! isolated from any embedder cost.
//!
//! `RemoteRagService` itself uses no GPU — only the inner embedder might.
//! Stage-1 cosine ANN and Stage-2 Paillier dot products are CPU.
//!
//! Run:
//!
//! ```text
//! cargo test -p approach4 --features snp-mock --release \
//!     --test obfuscation_bench -- --ignored --nocapture
//! ```

#![cfg(feature = "snp-mock")]

use std::sync::Arc;
use std::time::Instant;

use approach4::{Approach4InMemoryService, NoopAttestationVerifier};
use dp_forward::DpForwardConfig;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::{DecoderConfig, DecoderWeights, GeloQwenEmbedder};
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::InProcessTrustedExecutor;
use rag_core::{Caprise, CapriseKey, ChunkId, DocumentChunk};
use remote_rag::{PlanarLaplaceConfig, RemoteRagService};

const MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";

fn corpus() -> Vec<DocumentChunk> {
    vec![
        DocumentChunk {
            id: ChunkId("rust-memory-safety".into()),
            text: "Rust enforces memory safety through ownership and borrowing.".into(),
        },
        DocumentChunk {
            id: ChunkId("postgres-index".into()),
            text: "Postgres uses B-tree indexes for common equality and range lookups.".into(),
        },
        DocumentChunk {
            id: ChunkId("tls-attestation".into()),
            text: "Remote attestation can bind a TEE measurement into a TLS session.".into(),
        },
        DocumentChunk {
            id: ChunkId("python-asyncio".into()),
            text: "Python's asyncio event loop schedules coroutines cooperatively.".into(),
        },
    ]
}

const QUERY: &str = "How does Rust memory safety work?";
const TOP_K: usize = 2;

/// Build a GELO + OutAttnMult-on executor on the shared Vulkan adapter.
/// Matches the **bare** configuration used in `qwen3_overhead_breakdown.rs`
/// for the headline `gpu + GELO + OutAttnMult` numbers (no shield-vector
/// padding, no U-Verify probes). Shield and probes are real production
/// features but materially shift the bench off the gelo.md baseline; they
/// are intentionally omitted here so the numbers are directly comparable.
fn make_gelo_exec(gpu: &WgpuVulkanEngine) -> InProcessTrustedExecutor<WgpuVulkanEngine> {
    InProcessTrustedExecutor::with_seed(gpu.clone_shared(), MaskSeed::from_bytes([41u8; 32]))
}

fn make_gelo_embedder(
    cfg: &DecoderConfig,
    tokenizer: &gelo_embedder::HfTokenizer,
    weights: &Arc<DecoderWeights>,
    rope: &Arc<RopeTables>,
    gpu: &WgpuVulkanEngine,
) -> GeloQwenEmbedder<InProcessTrustedExecutor<WgpuVulkanEngine>> {
    // Force OutAttnMult on at any seq length so short bench prompts still
    // exercise the full protocol path.
    let mut c = cfg.clone();
    c.use_out_attn_mult = true;
    c.out_attn_mult_min_seq_len = Some(0);
    GeloQwenEmbedder::new(
        c,
        tokenizer.clone(),
        Arc::clone(weights),
        Arc::clone(rope),
        make_gelo_exec(gpu),
    )
    .expect("build GELO embedder on shared Vulkan adapter")
}

#[test]
#[ignore = "downloads Qwen3 (~1.2 GB), requires Vulkan GPU; --release recommended"]
fn obfuscation_path_wall_clock_comparison() {
    // ──────────────────────────────────────────────────────────────────────
    // One-shot model + GPU bring-up. Done once; shared across both benches'
    // GELO-configurations via Arc-cloned weights and the shared Vulkan
    // engine.
    // ──────────────────────────────────────────────────────────────────────
    eprintln!("[load] downloading + materialising Qwen3 weights...");
    let gpu = WgpuVulkanEngine::new().expect("Vulkan adapter");
    eprintln!(
        "[load] Vulkan adapter: {} ({:?})",
        gpu.adapter_info().name,
        gpu.adapter_info().device_type,
    );
    assert!(gpu.is_real_gpu(), "bench requires a real GPU, not llvmpipe");

    // Use a throw-away GELO executor just to materialise the weights from
    // HF. We re-create the executor per measurement so each config gets a
    // freshly-seeded mask state and a clean weight-provisioning timeline.
    let seed = GeloQwenEmbedder::from_pretrained(MODEL, make_gelo_exec(&gpu))
        .expect("Qwen3 from_pretrained");
    let weights = seed.weights_arc();
    let rope = seed.rope_arc();
    let tokenizer = seed.tokenizer().clone();
    let cfg = seed.config().clone();
    drop(seed);

    let dim = cfg.hidden_size;
    eprintln!(
        "[load] hidden_size = {dim}; corpus = {} docs; query = {:?}",
        corpus().len(),
        QUERY
    );
    eprintln!();
    eprintln!("=== Bench A: GELO vs GELO + DP-Forward (same threat model) ===");

    // ──────────────────────────────────────────────────────────────────────
    // Config A1 — GELO baseline (Vulkan + mask + OutAttnMult + CAPRISE).
    //
    // Warmup pattern matches `qwen3_overhead_breakdown.rs`: same embedder
    // for warmup and measurement, so the timed pass benefits from settled
    // GPU autotune and pipeline cache state. Rebuilding the embedder
    // between warmup and measurement throws away executor-side warmup work.
    // ──────────────────────────────────────────────────────────────────────
    let (gelo_ingest_ms, gelo_query_ms, gelo_top1) = {
        let embedder = make_gelo_embedder(&cfg, &tokenizer, &weights, &rope, &gpu);
        let scheme = Caprise::new(CapriseKey::generate(32.0, 0.15));
        let mut service =
            Approach4InMemoryService::new(embedder, scheme, NoopAttestationVerifier);

        // Warmup pass through the full ingest+query flow.
        service.ingest_chunks(corpus()).expect("warmup ingest");
        let _ = service.query(QUERY, TOP_K).expect("warmup query");

        // Measurement pass on the *same* service. The index now has 4
        // extra rows from the warmup pass; that's a constant cost on top
        // of the 4 we're about to add, dominated by the per-text embed
        // cost.
        let t = Instant::now();
        service.ingest_chunks(corpus()).expect("ingest");
        let ingest_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let hits = service.query(QUERY, TOP_K).expect("query");
        let query_ms = t.elapsed().as_secs_f64() * 1000.0;

        let top1 = hits
            .first()
            .map(|h| h.id.0.clone())
            .unwrap_or_else(|| "(none)".into());
        (ingest_ms, query_ms, top1)
    };

    // ──────────────────────────────────────────────────────────────────────
    // Config A2 — GELO + DP-Forward(ε=4, δ=1e-5, C=1.0) + CAPRISE.
    // ──────────────────────────────────────────────────────────────────────
    let (gelo_dp_ingest_ms, gelo_dp_query_ms, gelo_dp_top1) = {
        let dp_cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
        let embedder = make_gelo_embedder(&cfg, &tokenizer, &weights, &rope, &gpu)
            .with_dp_forward(dp_cfg);
        let scheme = Caprise::new(CapriseKey::generate(32.0, 0.15));
        let mut service =
            Approach4InMemoryService::new(embedder, scheme, NoopAttestationVerifier);

        service.ingest_chunks(corpus()).expect("warmup ingest");
        let _ = service.query(QUERY, TOP_K).expect("warmup query");

        let t = Instant::now();
        service.ingest_chunks(corpus()).expect("ingest");
        let ingest_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let hits = service.query(QUERY, TOP_K).expect("query");
        let query_ms = t.elapsed().as_secs_f64() * 1000.0;

        let top1 = hits
            .first()
            .map(|h| h.id.0.clone())
            .unwrap_or_else(|| "(none)".into());
        (ingest_ms, query_ms, top1)
    };

    let n_docs = corpus().len() as f64;
    eprintln!(
        "{:<55} {:>10} {:>11} {:>10}",
        "config", "ingest ms", "ms/doc", "query ms"
    );
    eprintln!("{}", "-".repeat(89));
    eprintln!(
        "{:<55} {:>10.2} {:>11.2} {:>10.2}   top1={}",
        "A1: GELO + CAPRISE (baseline)",
        gelo_ingest_ms,
        gelo_ingest_ms / n_docs,
        gelo_query_ms,
        gelo_top1,
    );
    eprintln!(
        "{:<55} {:>10.2} {:>11.2} {:>10.2}   top1={}",
        "A2: GELO + DP-Forward(ε=4) + CAPRISE",
        gelo_dp_ingest_ms,
        gelo_dp_ingest_ms / n_docs,
        gelo_dp_query_ms,
        gelo_dp_top1,
    );
    eprintln!(
        "    Δ DP-Forward overhead vs GELO baseline       {:>+10.2} ms / {} docs   {:>+10.2} ms / query",
        gelo_dp_ingest_ms - gelo_ingest_ms,
        corpus().len(),
        gelo_dp_query_ms - gelo_query_ms,
    );

    // ──────────────────────────────────────────────────────────────────────
    // Bench B — RemoteRAG (alternative threat model, separate embedder)
    // ──────────────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!(
        "=== Bench B: RemoteRAG — alternative to GELO (same embedder) ==="
    );
    eprintln!(
        "    GeloQwen on Vulkan + planar-Laplace on queries, plaintext index,"
    );
    eprintln!(
        "    Paillier rerank. CAPRISE replaced by RemoteRAG two-stage protocol."
    );

    let (rrag_keygen_ms, rrag_ingest_ms, rrag_query_ms, rrag_top1) = {
        let embedder = make_gelo_embedder(&cfg, &tokenizer, &weights, &rope, &gpu);
        // ε ≈ 10·n is the paper's lower bound for usable retrieval at
        // dim ∈ [384, 1536]. Qwen3-Embedding-0.6B → dim=1024 → ε ≈ 10_240.
        let dp_cfg = PlanarLaplaceConfig::new((10 * dim) as f64, dim);

        let t = Instant::now();
        let mut service = RemoteRagService::new(embedder, dp_cfg)
            .with_paillier_bits(1024)
            .with_over_fetch_factor(3)
            .with_seed([19u8; 32]);
        let keygen_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Warmup on the same service (matches Bench A pattern).
        service.ingest_chunks(corpus()).expect("warmup ingest");
        let _ = service.query(QUERY, TOP_K).expect("warmup query");

        let t = Instant::now();
        service.ingest_chunks(corpus()).expect("ingest");
        let ingest_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let hits = service.query(QUERY, TOP_K).expect("query");
        let query_ms = t.elapsed().as_secs_f64() * 1000.0;

        let top1 = hits
            .first()
            .map(|h| h.id.0.clone())
            .unwrap_or_else(|| "(none)".into());
        (keygen_ms, ingest_ms, query_ms, top1)
    };

    eprintln!();
    eprintln!(
        "{:<55} {:>10} {:>11} {:>10}",
        "config", "ingest ms", "ms/doc", "query ms"
    );
    eprintln!("{}", "-".repeat(89));
    eprintln!(
        "{:<55} {:>10.2} {:>11.2} {:>10.2}   top1={}",
        format!("B:  GeloQwen on Vulkan + RemoteRAG (ε≈10·n, k'=3·top_k)"),
        rrag_ingest_ms,
        rrag_ingest_ms / n_docs,
        rrag_query_ms,
        rrag_top1,
    );
    eprintln!(
        "    (one-time 1024-bit Paillier keygen: {:.2} ms)",
        rrag_keygen_ms,
    );
    eprintln!(
        "    Per-candidate Paillier dot product ≈ query_ms / (over_fetch · top_k) = {:.2} ms",
        (rrag_query_ms - gelo_query_ms * (1.0 / TOP_K as f64).max(0.5))
            / (3.0 * TOP_K as f64).max(1.0),
    );
    eprintln!();
    eprintln!(
        "    Δ vs Bench A (same Qwen3 embedder cost):"
    );
    eprintln!(
        "      ingest:  {:>+10.2} ms / {} docs  (RemoteRAG saves CAPRISE work)",
        rrag_ingest_ms - gelo_ingest_ms,
        corpus().len(),
    );
    eprintln!(
        "      query:   {:>+10.2} ms          (Paillier rerank net cost)",
        rrag_query_ms - gelo_query_ms,
    );
}
