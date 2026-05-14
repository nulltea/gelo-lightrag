//! End-to-end performance tests for SAP and CAPRISE storage encryption.
//!
//! These tests exercise the full Approach 4 pipeline against real components:
//! - Source markdown fetched from the running edgequake API
//!   (`GET /api/v1/documents/{id}` -> `pdf_id` -> `GET /api/v1/documents/pdf/{pdf_id}/content`)
//! - Chunks produced by the ported `TokenBasedChunking` strategy
//! - 1024-dim embeddings from Ollama's `qwen3-embedding:0.6b`
//! - Both `SapScheme` and `Caprise` as the `EmbeddingEncryptionScheme`
//!
//! They are ignored by default because they require:
//! - Ollama running locally with `qwen3-embedding:0.6b` pulled
//! - The edgequake API reachable (defaults to `http://localhost:8080`)
//!
//! Run with: `cargo test --test sap_caprise_perf -- --ignored --nocapture`.

mod common;

use std::time::{Duration, Instant};

use gelo_rag::{GeloRagInMemoryService, NoopAttestationVerifier};
use rag_core::{
    AesChunkCipher, Caprise, CapriseKey, ChunkCiphertext, ChunkId, DocumentChunk, Embedder,
    EmbeddingEncryptionScheme, EncryptedEmbedding, InMemoryEncryptedIndex, SapKey, SapScheme,
};

use common::{ChunkerConfig, OllamaEmbedder, TokenBasedChunker, fetch_document_markdown};

const DOCUMENT_ID: &str = "6b59cf0a-6ec7-40f7-bdd6-11c0b8c9444f";
const EMBEDDING_MODEL: &str = "qwen3-embedding:0.6b";
const EXPECTED_DIMS: usize = 1024;

const QUERIES: &[&str] = &[
    "How does CAPRISE protect stored embeddings against inversion attacks?",
    "What privacy guarantees does DistanceDP provide for the query vector?",
    "Which threat models does the paper systematize for retrieval-augmented generation?",
    "What is the role of remote attestation in a TEE-based embedding service?",
    "How do access pattern leaks threaten confidentiality in RAG pipelines?",
];

const TOP_K: usize = 5;

struct PreparedChunks {
    chunks: Vec<DocumentChunk>,
    embeddings: Vec<Vec<f32>>,
    embed_elapsed: Duration,
}

fn prepare() -> PreparedChunks {
    let markdown = fetch_document_markdown(DOCUMENT_ID)
        .expect("fetch markdown from edgequake; set EDGEQUAKE_BASE_URL if not localhost:8080");
    assert!(
        !markdown.trim().is_empty(),
        "fetched markdown is empty for {DOCUMENT_ID}"
    );

    let config = ChunkerConfig::default();
    let texts = TokenBasedChunker::chunk(&markdown, &config);
    assert!(
        !texts.is_empty(),
        "TokenBasedChunking produced zero chunks (markdown len={})",
        markdown.len()
    );

    let chunks: Vec<DocumentChunk> = texts
        .into_iter()
        .enumerate()
        .map(|(idx, text)| DocumentChunk {
            id: ChunkId(format!("{DOCUMENT_ID}-chunk-{idx:04}")),
            text,
        })
        .collect();

    let mut embedder = OllamaEmbedder::new(EMBEDDING_MODEL)
        .expect("construct OllamaEmbedder; set OLLAMA_BASE_URL if not localhost:11434");

    // Warm-up — the first call loads the model into Ollama's VRAM and would
    // otherwise swamp the embedding timing number.
    let _ = embedder
        .embed(&["warmup".to_string()])
        .expect("ollama warmup (is qwen3-embedding:0.6b pulled?)");

    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let start = Instant::now();
    let embeddings = embedder.embed(&texts).expect("batch embed");
    let embed_elapsed = start.elapsed();

    assert_eq!(embeddings.len(), chunks.len());
    assert_eq!(
        embeddings[0].len(),
        EXPECTED_DIMS,
        "expected {EXPECTED_DIMS}-dim embeddings for {EMBEDDING_MODEL}"
    );

    PreparedChunks {
        chunks,
        embeddings,
        embed_elapsed,
    }
}

struct SchemeTimings {
    scheme: &'static str,
    chunk_count: usize,
    dims: usize,
    encrypt_doc_total: Duration,
    encrypt_chunk_total: Duration,
    index_insert_total: Duration,
    query_embed_total: Duration,
    query_encrypt_total: Duration,
    search_total: Duration,
    chunk_decrypt_total: Duration,
    embedding_decrypt_total: Duration,
    query_count: usize,
    hit_count: usize,
}

impl SchemeTimings {
    fn print(&self) {
        let per_doc = |d: Duration| d / self.chunk_count.max(1) as u32;
        let throughput = |d: Duration| {
            let secs = d.as_secs_f64();
            if secs == 0.0 {
                f64::INFINITY
            } else {
                self.chunk_count as f64 / secs
            }
        };
        let per_query = |d: Duration| d / self.query_count.max(1) as u32;
        let per_hit = |d: Duration| d / self.hit_count.max(1) as u32;

        println!("\nperf[{scheme}]: chunks={chunks} dims={dims}", scheme = self.scheme, chunks = self.chunk_count, dims = self.dims);
        println!(
            "perf[{scheme}]:   encrypt_document  total={total:?}  avg={avg:?}  throughput={tput:.1} vec/s",
            scheme = self.scheme,
            total = self.encrypt_doc_total,
            avg = per_doc(self.encrypt_doc_total),
            tput = throughput(self.encrypt_doc_total),
        );
        println!(
            "perf[{scheme}]:   encrypt_chunk     total={total:?}  avg={avg:?}  throughput={tput:.1} chunk/s",
            scheme = self.scheme,
            total = self.encrypt_chunk_total,
            avg = per_doc(self.encrypt_chunk_total),
            tput = throughput(self.encrypt_chunk_total),
        );
        println!(
            "perf[{scheme}]:   index_insert      total={total:?}  avg={avg:?}",
            scheme = self.scheme,
            total = self.index_insert_total,
            avg = per_doc(self.index_insert_total),
        );
        println!(
            "perf[{scheme}]:   query_embed       total={total:?}  avg={avg:?}  (n={q})",
            scheme = self.scheme,
            total = self.query_embed_total,
            avg = per_query(self.query_embed_total),
            q = self.query_count,
        );
        println!(
            "perf[{scheme}]:   query_encrypt     total={total:?}  avg={avg:?}",
            scheme = self.scheme,
            total = self.query_encrypt_total,
            avg = per_query(self.query_encrypt_total),
        );
        println!(
            "perf[{scheme}]:   search(top_k={k}) total={total:?}  avg={avg:?}  (over {chunks} docs)",
            scheme = self.scheme,
            k = TOP_K,
            total = self.search_total,
            avg = per_query(self.search_total),
            chunks = self.chunk_count,
        );
        println!(
            "perf[{scheme}]:   decrypt_chunk     total={total:?}  avg={avg:?}  (n={hits})",
            scheme = self.scheme,
            total = self.chunk_decrypt_total,
            avg = per_hit(self.chunk_decrypt_total),
            hits = self.hit_count,
        );
        println!(
            "perf[{scheme}]:   decrypt_embedding total={total:?}  avg={avg:?}",
            scheme = self.scheme,
            total = self.embedding_decrypt_total,
            avg = per_hit(self.embedding_decrypt_total),
        );
    }
}

fn run_perf<S>(scheme_name: &'static str, mut scheme: S, prepared: &PreparedChunks) -> SchemeTimings
where
    S: EmbeddingEncryptionScheme,
{
    let dims = prepared.embeddings[0].len();
    let chunk_cipher = AesChunkCipher::generate();
    let mut index = InMemoryEncryptedIndex::default();

    // ── Document ingestion ────────────────────────────────────────────────
    let mut encrypted_embeddings: Vec<EncryptedEmbedding> = Vec::with_capacity(prepared.chunks.len());
    let mut encrypt_doc_total = Duration::ZERO;
    for embedding in &prepared.embeddings {
        let start = Instant::now();
        let encrypted = scheme
            .encrypt_document(embedding)
            .expect("encrypt_document");
        encrypt_doc_total += start.elapsed();
        encrypted_embeddings.push(encrypted);
    }

    let mut encrypted_chunks: Vec<ChunkCiphertext> = Vec::with_capacity(prepared.chunks.len());
    let mut encrypt_chunk_total = Duration::ZERO;
    for chunk in &prepared.chunks {
        let start = Instant::now();
        let ct = chunk_cipher.encrypt_chunk(chunk).expect("encrypt_chunk");
        encrypt_chunk_total += start.elapsed();
        encrypted_chunks.push(ct);
    }

    let start = Instant::now();
    for (ct, embedding) in encrypted_chunks
        .into_iter()
        .zip(encrypted_embeddings.into_iter())
    {
        index.insert(ct, embedding);
    }
    let index_insert_total = start.elapsed();
    assert_eq!(index.len(), prepared.chunks.len());

    // ── Query path ────────────────────────────────────────────────────────
    let mut embedder = OllamaEmbedder::new(EMBEDDING_MODEL).expect("ollama embedder");

    let mut query_embed_total = Duration::ZERO;
    let mut query_encrypt_total = Duration::ZERO;
    let mut search_total = Duration::ZERO;
    let mut chunk_decrypt_total = Duration::ZERO;
    let mut embedding_decrypt_total = Duration::ZERO;
    let mut hit_count = 0usize;

    for query in QUERIES {
        let start = Instant::now();
        let mut q_embeds = embedder
            .embed(&[query.to_string()])
            .expect("embed query");
        query_embed_total += start.elapsed();
        let q_embed = q_embeds.remove(0);
        assert_eq!(q_embed.len(), dims);

        let start = Instant::now();
        let q_encrypted = scheme.encrypt_query(&q_embed).expect("encrypt_query");
        query_encrypt_total += start.elapsed();

        let start = Instant::now();
        let hits = index.search(&q_encrypted, TOP_K);
        search_total += start.elapsed();
        assert!(
            !hits.is_empty(),
            "[{scheme_name}] expected at least one retrieval hit for query {query:?}"
        );
        hit_count += hits.len();

        for (ct, embedding, _score) in &hits {
            let start = Instant::now();
            let _chunk = chunk_cipher
                .decrypt_chunk(ct)
                .expect("decrypt retrieved chunk");
            chunk_decrypt_total += start.elapsed();

            // CAPRISE only supports `decrypt_document` for `caprise-db` ciphertexts;
            // `caprise-query` is not reversible from the stored side. The stored
            // embeddings in `hits` are the document-side ciphertexts, so this is safe.
            let start = Instant::now();
            let _plain = scheme
                .decrypt_document(embedding)
                .expect("decrypt retrieved embedding");
            embedding_decrypt_total += start.elapsed();
        }
    }

    SchemeTimings {
        scheme: scheme_name,
        chunk_count: prepared.chunks.len(),
        dims,
        encrypt_doc_total,
        encrypt_chunk_total,
        index_insert_total,
        query_embed_total,
        query_encrypt_total,
        search_total,
        chunk_decrypt_total,
        embedding_decrypt_total,
        query_count: QUERIES.len(),
        hit_count,
    }
}

#[test]
#[ignore = "requires local Ollama (qwen3-embedding:0.6b) and edgequake API"]
fn sap_caprise_encrypt_decrypt_retrieve_perf() {
    let prepared = prepare();

    println!(
        "\nperf[setup]: chunks={} dims={} embed_total={:?} embed_avg={:?}",
        prepared.chunks.len(),
        prepared.embeddings[0].len(),
        prepared.embed_elapsed,
        prepared.embed_elapsed / prepared.chunks.len().max(1) as u32,
    );

    let sap_timings = run_perf(
        "SAP",
        SapScheme::new(SapKey::generate(32.0, 0.15)),
        &prepared,
    );
    sap_timings.print();

    let caprise_timings = run_perf(
        "CAPRISE",
        Caprise::new(CapriseKey::generate(32.0, 0.15)),
        &prepared,
    );
    caprise_timings.print();
}

// ── Sanity wiring: quickly prove the `GeloRagInMemoryService` path also
// works end-to-end against Ollama + each scheme. Not a perf measurement,
// just a regression guard that the service wrapper composes correctly with
// the real 1024-dim model. ─────────────────────────────────────────────────

#[test]
#[ignore = "requires local Ollama (qwen3-embedding:0.6b) and edgequake API"]
fn sap_service_e2e_smoke() {
    run_service_smoke(SapScheme::new(SapKey::generate(32.0, 0.15)));
}

#[test]
#[ignore = "requires local Ollama (qwen3-embedding:0.6b) and edgequake API"]
fn caprise_service_e2e_smoke() {
    run_service_smoke(Caprise::new(CapriseKey::generate(32.0, 0.15)));
}

fn run_service_smoke<S>(scheme: S)
where
    S: EmbeddingEncryptionScheme,
{
    let prepared = prepare();
    let embedder = OllamaEmbedder::new(EMBEDDING_MODEL).expect("ollama embedder");
    let mut service = GeloRagInMemoryService::new(embedder, scheme, NoopAttestationVerifier);

    // Re-embed inside the service so we actually exercise the ingest path.
    // The `prepare()` embeddings are ignored here on purpose; we only reuse
    // the chunk texts so the corpus is identical to the perf run.
    service
        .ingest_chunks(prepared.chunks.clone())
        .expect("ingest");
    assert_eq!(service.index_len(), prepared.chunks.len());

    let hits = service.query(QUERIES[0], TOP_K).expect("query");
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.score.is_finite()));
}
