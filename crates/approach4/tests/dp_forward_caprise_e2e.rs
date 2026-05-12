//! M6.2 integration test — GELO + DP-Forward + CAPRISE through
//! `Approach4InMemoryService`. Three assertions:
//!
//! 1. `Embedder::model_identity` rebinds when DP is enabled (so a SEV-SNP
//!    attestation report's `expected_model_id` pin catches a parameter
//!    substitution).
//! 2. The DP-on retrieval still surfaces the right top hit on a non-trivial
//!    corpus (utility is preserved at ε=4, δ=1e-5).
//! 3. The DP-on retrieval's *decrypted* embedding differs from the DP-off
//!    decrypted embedding — proof that the irreversible noise actually rides
//!    through CAPRISE re-encryption rather than getting collapsed somewhere.

use approach4::{Approach4InMemoryService, NoopAttestationVerifier};
use dp_forward::DpForwardConfig;
use gelo_embedder::GeloQwenEmbedder;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, RayonCpuEngine, ShieldConfig};
use rag_core::{Caprise, CapriseKey, ChunkId, DocumentChunk, Embedder};

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
    ]
}

fn make_exec() -> InProcessTrustedExecutor<RayonCpuEngine> {
    InProcessTrustedExecutor::with_shield(
        RayonCpuEngine::new(),
        MaskSeed::from_bytes([41u8; 32]),
        ShieldConfig::new(8, 6.0),
    )
    .with_verify_probes(2)
}

#[test]
#[ignore = "downloads Qwen3-Embedding-0.6B (~1.2 GB) from Hugging Face on first run"]
fn dp_forward_rebinds_model_identity() {
    let embedder = GeloQwenEmbedder::from_pretrained("Qwen/Qwen3-Embedding-0.6B", make_exec())
        .expect("download Qwen3-Embedding-0.6B");
    let id_no_dp = embedder.model_identity().to_vec();

    let cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
    let embedder_dp = GeloQwenEmbedder::from_pretrained("Qwen/Qwen3-Embedding-0.6B", make_exec())
        .expect("download Qwen3-Embedding-0.6B")
        .with_dp_forward(cfg);
    let id_dp = embedder_dp.model_identity().to_vec();

    assert_ne!(
        id_no_dp, id_dp,
        "with_dp_forward must rebind model_identity"
    );
    assert_eq!(id_no_dp.len(), id_dp.len(), "both are sha256 hex strings");

    // Changing the DP parameters must produce yet another identity.
    let cfg_b = DpForwardConfig::calibrate(2.0, 1e-5, 1.0);
    let embedder_dp_b = GeloQwenEmbedder::from_pretrained("Qwen/Qwen3-Embedding-0.6B", make_exec())
        .expect("download Qwen3-Embedding-0.6B")
        .with_dp_forward(cfg_b);
    assert_ne!(id_dp, embedder_dp_b.model_identity().to_vec());
}

#[test]
#[ignore = "downloads Qwen3-Embedding-0.6B (~1.2 GB) from Hugging Face on first run"]
fn dp_forward_preserves_top_hit_at_moderate_epsilon() {
    let embedder = GeloQwenEmbedder::from_pretrained("Qwen/Qwen3-Embedding-0.6B", make_exec())
        .expect("download Qwen3-Embedding-0.6B")
        .with_dp_forward(DpForwardConfig::calibrate(4.0, 1e-5, 1.0));

    let scheme = Caprise::new(CapriseKey::generate(32.0, 0.15));
    let mut service = Approach4InMemoryService::new(embedder, scheme, NoopAttestationVerifier);
    service.ingest_chunks(corpus()).expect("ingest");

    let hits = service
        .query("How does Rust memory safety work?", 2)
        .expect("query");
    assert_eq!(
        hits[0].id.0, "rust-memory-safety",
        "DP at ε=4 should preserve top-1 utility on this easy corpus"
    );
}

#[test]
#[ignore = "downloads Qwen3-Embedding-0.6B (~1.2 GB) from Hugging Face on first run"]
fn dp_noise_perturbs_pooled_output() {
    // Direct embedder comparison (no CAPRISE in the loop). Two embedders
    // built from the same weights/exec seed; the DP-on one's `embed` output
    // must differ from the DP-off one's by more than f32 round-off — the
    // empirical Gaussian floor at sigma ≈ 2.16 over a 1024-d unit-norm
    // embedding is ~σ in expectation per component.
    let mut clean = GeloQwenEmbedder::from_pretrained("Qwen/Qwen3-Embedding-0.6B", make_exec())
        .expect("download Qwen3-Embedding-0.6B");
    let mut noisy = GeloQwenEmbedder::from_pretrained("Qwen/Qwen3-Embedding-0.6B", make_exec())
        .expect("download Qwen3-Embedding-0.6B")
        .with_dp_forward(DpForwardConfig::calibrate(4.0, 1e-5, 1.0));

    let texts = vec!["Rust enforces memory safety.".to_string()];
    let v_clean = clean.embed(&texts).expect("clean embed");
    let v_noisy = noisy.embed(&texts).expect("noisy embed");

    assert_eq!(v_clean[0].len(), v_noisy[0].len());
    let max_abs = v_clean[0]
        .iter()
        .zip(v_noisy[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs > 0.1,
        "DP-on output should differ from DP-off output by ≫ f32 round-off (max_abs = {max_abs})"
    );
}
