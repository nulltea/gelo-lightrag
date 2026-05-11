use approach4::{Approach4InMemoryService, NoopAttestationVerifier};
use rag_core::{Caprise, CapriseKey, ChunkId, DocumentChunk, FastEmbedEmbedder, SapKey, SapScheme};

#[test]
#[ignore = "downloads a fastembed model from Hugging Face on first run"]
fn fastembed_sap_retrieval_smoke_test() {
    let embedder = FastEmbedEmbedder::new_smallest().expect("small fastembed model");
    let scheme = SapScheme::new(SapKey::generate(32.0, 0.15));
    let mut service = Approach4InMemoryService::new(embedder, scheme, NoopAttestationVerifier);

    service
        .ingest_chunks(vec![
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
        ])
        .expect("ingest");

    let hits = service
        .query("How does Rust memory safety work?", 2)
        .expect("query");

    assert_eq!(hits[0].id.0, "rust-memory-safety");
}

#[test]
#[ignore = "downloads a fastembed model from Hugging Face on first run"]
fn fastembed_caprise_retrieval_smoke_test() {
    let embedder = FastEmbedEmbedder::new_smallest().expect("small fastembed model");
    let scheme = Caprise::new(CapriseKey::generate(32.0, 0.15));
    let mut service = Approach4InMemoryService::new(embedder, scheme, NoopAttestationVerifier);

    service
        .ingest_chunks(vec![
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
        ])
        .expect("ingest");

    let hits = service
        .query("How does Rust memory safety work?", 2)
        .expect("query");

    assert_eq!(hits[0].id.0, "rust-memory-safety");
}
