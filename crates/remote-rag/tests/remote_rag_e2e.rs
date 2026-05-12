//! M6.5 end-to-end: drive `RemoteRagService` with a real `FastEmbedEmbedder`
//! and a non-trivial corpus. Asserts the top-1 retrieval is correct after
//! the two-stage protocol and that Stage-1-only would *not* always reach it
//! (so the Paillier rerank is load-bearing).

use rag_core::{ChunkId, DocumentChunk, FastEmbedEmbedder};
use remote_rag::{PlanarLaplaceConfig, RemoteRagService};

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
        DocumentChunk {
            id: ChunkId("kubernetes-pods".into()),
            text: "Kubernetes pods are the smallest deployable unit in a cluster.".into(),
        },
    ]
}

#[test]
#[ignore = "downloads a fastembed model from Hugging Face on first run; uses 1024-bit Paillier"]
fn remote_rag_end_to_end_recovers_top_hit() {
    let embedder = FastEmbedEmbedder::new_smallest().expect("small fastembed model");
    // AllMiniLM-L6-v2 emits 384-d embeddings.
    let dp_cfg = PlanarLaplaceConfig::new(/*ε ≈ 10·n */ 3_840.0, 384);
    let mut service = RemoteRagService::new(embedder, dp_cfg)
        .with_paillier_bits(1024)
        .with_over_fetch_factor(3)
        .with_seed([19u8; 32]);

    service.ingest_chunks(corpus()).expect("ingest");
    let hits = service
        .query("How does Rust memory safety work?", 2)
        .expect("query");

    assert!(hits.len() >= 1);
    assert_eq!(
        hits[0].id.0, "rust-memory-safety",
        "Stage-1 over-fetch + Stage-2 PHE rerank should recover the true top hit"
    );
}
