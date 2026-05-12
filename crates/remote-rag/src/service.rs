//! `RemoteRagService` — RemoteRAG's two-stage protocol in a single
//! in-memory shape. Client and server state live in the same struct for
//! ergonomics; comments mark the boundaries.
//!
//! Threat model: **plaintext embedding index server-side**, AES-GCM chunk
//! payloads (encrypted), client-held Paillier private key, Stage-1
//! planar-Laplace noise on the query, Stage-2 homomorphic dot-product
//! rerank. **Mutually exclusive with CAPRISE-encrypted-index** — see
//! `crates/remote-rag/README.md` for the why.

use anyhow::Result;
use rag_core::{
    AesChunkCipher, ChunkCiphertext, ChunkId, DocumentChunk, Embedder, RetrievalHit,
    cosine_similarity,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::paillier::{
    DEFAULT_SCALE_BITS, PaillierPrivateKey, dequantize_product, quantize,
};
use crate::planar_laplace::{PlanarLaplaceConfig, perturb};

/// One row in the (untrusted) server's index. Embedding is plaintext;
/// chunk text is AES-GCM encrypted under the client's chunk key.
struct IndexEntry {
    chunk_ct: ChunkCiphertext,
    embedding: Vec<f32>,
}

/// Per-thread RNG initializer for rayon `map_init`. Each worker thread
/// gets a fresh ChaCha20 seeded from `OsRng` — Paillier nonces must be
/// unique per ciphertext, so threads must not share an RNG nor derive
/// from a fixed seed.
fn seed_thread_rng() -> ChaCha20Rng {
    let mut seed = [0u8; 32];
    rand::rng().fill_bytes(&mut seed);
    ChaCha20Rng::from_seed(seed)
}

/// Server- and client-side state of the RemoteRAG protocol.
pub struct RemoteRagService<E> {
    // === client-side state ===
    embedder: E,
    chunk_cipher: AesChunkCipher,
    paillier_sk: PaillierPrivateKey,
    dp_cfg: PlanarLaplaceConfig,
    /// Over-fetch factor: Stage 1 returns `over_fetch_factor · top_k`
    /// candidates, which Stage 2 reranks. Paper recommends 3–5.
    over_fetch_factor: usize,
    /// Fixed-point quantization scale for Paillier dot products. Default
    /// `DEFAULT_SCALE_BITS = 16`.
    scale_bits: u32,
    /// RNG for Paillier nonces + planar-Laplace samples. Seeded from
    /// `OsRng` by default; tests can `with_seed` for determinism.
    rng: ChaCha20Rng,

    // === server-side state ===
    index: Vec<IndexEntry>,
}

impl<E: Embedder> RemoteRagService<E> {
    /// Construct with default Paillier key (1024-bit), `over_fetch=3`,
    /// and seeded `OsRng`. `dp_cfg.n` must equal the embedder's output
    /// dimensionality — checked lazily at the first `query` call.
    pub fn new(embedder: E, dp_cfg: PlanarLaplaceConfig) -> Self {
        let mut seed = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rng(), &mut seed);
        Self {
            embedder,
            chunk_cipher: AesChunkCipher::generate(),
            paillier_sk: PaillierPrivateKey::generate(),
            dp_cfg,
            over_fetch_factor: 3,
            scale_bits: DEFAULT_SCALE_BITS,
            rng: ChaCha20Rng::from_seed(seed),
            index: Vec::new(),
        }
    }

    /// Override the Paillier key size (default 1024-bit). Smaller is
    /// faster but reduces security.
    pub fn with_paillier_bits(mut self, n_bits: usize) -> Self {
        self.paillier_sk = PaillierPrivateKey::generate_with_bits(n_bits);
        self
    }

    /// Override the over-fetch factor. Paper recommends 3–5.
    pub fn with_over_fetch_factor(mut self, k: usize) -> Self {
        assert!(k >= 1);
        self.over_fetch_factor = k;
        self
    }

    /// Seed the RNG deterministically for tests.
    pub fn with_seed(mut self, seed: [u8; 32]) -> Self {
        self.rng = ChaCha20Rng::from_seed(seed);
        self
    }

    /// Override the quantization scale (default 16 bits of fractional
    /// precision). Larger = more accurate dot product but slower.
    pub fn with_scale_bits(mut self, bits: u32) -> Self {
        assert!(bits >= 1 && bits <= 30);
        self.scale_bits = bits;
        self
    }

    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Ingest a batch of chunks. The embedder is called once; chunk text
    /// is AES-GCM-encrypted; plaintext embedding lands in the server-side
    /// index.
    pub fn ingest_chunks(&mut self, chunks: Vec<DocumentChunk>) -> Result<()> {
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embedder.embed(&texts)?;
        for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            let chunk_ct = self.chunk_cipher.encrypt_chunk(&chunk)?;
            self.index.push(IndexEntry {
                chunk_ct,
                embedding,
            });
        }
        Ok(())
    }

    /// Two-stage retrieval. Returns the true top-`top_k` by clean-query
    /// cosine, computed under Paillier homomorphism over the Stage-1
    /// candidates.
    pub fn query(&mut self, text: &str, top_k: usize) -> Result<Vec<RetrievalHit>> {
        let clean_embedding = self.embedder.embed(&[text.to_owned()])?.remove(0);
        anyhow::ensure!(
            clean_embedding.len() == self.dp_cfg.n,
            "embedder dim ({}) does not match dp_cfg.n ({})",
            clean_embedding.len(),
            self.dp_cfg.n,
        );

        // === Stage 1 (client) ===
        let mut noisy = clean_embedding.clone();
        perturb(&mut noisy, &self.dp_cfg, &mut self.rng);

        // === Stage 1 (server) ===
        let stage1_size = (top_k * self.over_fetch_factor).min(self.index.len());
        let mut scored: Vec<(usize, f32)> = self
            .index
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine_similarity(&noisy, &e.embedding)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let stage1: Vec<usize> = scored.into_iter().take(stage1_size).map(|(i, _)| i).collect();

        // === Stage 2 (client encrypt) ===
        // The client holds the keypair, so `PaillierPrivateKey::encrypt_signed`
        // uses the CRT path (~2× faster than the public-key form). Per-dim
        // encryptions are independent, so rayon-parallelize across cores —
        // each thread gets a fresh OsRng-seeded ChaCha for nonce sampling
        // (Paillier randomness MUST be unique per ciphertext, not derived
        // from a shared seed).
        let q_int = quantize(&clean_embedding, self.scale_bits);
        let sk_arc = &self.paillier_sk;
        let q_ct: Vec<_> = q_int
            .par_iter()
            .map_init(seed_thread_rng, |rng, m| sk_arc.encrypt_signed(m, rng))
            .collect();

        // === Stage 2 (server homomorphic dot products + client decrypt) ===
        // Each candidate's dot product is independent ⇒ rayon-parallelize.
        // The `homomorphic_dot` itself now uses multi-exponentiation.
        let scale_bits = self.scale_bits;
        let mut rerank: Vec<(usize, f64)> = stage1
            .par_iter()
            .map_init(seed_thread_rng, |rng, &idx| {
                let e_d_int = quantize(&self.index[idx].embedding, scale_bits);
                let dot_ct = sk_arc
                    .public()
                    .homomorphic_dot(&q_ct, &e_d_int, rng)
                    .expect("homomorphic_dot");
                let dot_int = sk_arc.decrypt_signed(&dot_ct);
                let dot = dequantize_product(&dot_int, scale_bits);
                // Query + docs are L2-normalised by the embedder, so dot ≈ cosine.
                (idx, dot)
            })
            .collect();
        rerank.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rerank.truncate(top_k);

        // Build retrieval hits with decrypted chunk text.
        let mut hits = Vec::with_capacity(rerank.len());
        for (idx, score) in rerank {
            let entry = &self.index[idx];
            let chunk = self.chunk_cipher.decrypt_chunk(&entry.chunk_ct)?;
            hits.push(RetrievalHit {
                id: ChunkId(chunk.id.0),
                score: score as f32,
                text: chunk.text,
                // RemoteRAG indexes plaintext embeddings, so re-export as
                // an "identity-scheme" EncryptedEmbedding so the public
                // `RetrievalHit` shape stays uniform with the CAPRISE path.
                embedding: rag_core::EncryptedEmbedding {
                    scheme: "remote-rag-plaintext",
                    vector: entry.embedding.clone(),
                    nonce: Vec::new(),
                    original_dimension: entry.embedding.len(),
                },
            });
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rag_core::ChunkId;

    /// Stub embedder: each text maps to a fixed unit-norm vector. Used so
    /// the protocol test doesn't require a real model.
    struct StubEmbedder;

    impl Embedder for StubEmbedder {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            // dim = 4. Map keywords to canonical unit vectors so retrieval
            // is well-defined.
            let canon = |key: &str| -> Vec<f32> {
                match key {
                    k if k.contains("apple") => vec![1.0, 0.0, 0.0, 0.0],
                    k if k.contains("banana") => vec![0.0, 1.0, 0.0, 0.0],
                    k if k.contains("cherry") => vec![0.0, 0.0, 1.0, 0.0],
                    _ => vec![0.0, 0.0, 0.0, 1.0],
                }
            };
            Ok(texts.iter().map(|t| canon(t)).collect())
        }
    }

    fn corpus() -> Vec<DocumentChunk> {
        vec![
            DocumentChunk {
                id: ChunkId("apple".into()),
                text: "apple orchard".into(),
            },
            DocumentChunk {
                id: ChunkId("banana".into()),
                text: "banana bread".into(),
            },
            DocumentChunk {
                id: ChunkId("cherry".into()),
                text: "cherry blossom".into(),
            },
        ]
    }

    #[test]
    fn round_trip_with_low_noise_recovers_top_hit() {
        // Tiny modulus (256-bit) and tight ε keep this test under ~1 s.
        let dp_cfg = PlanarLaplaceConfig::new(/*ε*/ 50.0, /*n*/ 4);
        let mut service = RemoteRagService::new(StubEmbedder, dp_cfg)
            .with_paillier_bits(256)
            .with_over_fetch_factor(3)
            .with_seed([7u8; 32]);
        service.ingest_chunks(corpus()).expect("ingest");

        let hits = service.query("apple pie", 1).expect("query");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id.0, "apple");
    }

    #[test]
    fn paillier_rerank_recovers_when_stage1_is_noisy() {
        // Tighter ε perturbs Stage-1 enough that cosine over the noisy query
        // *might* miss the true top-1, but Stage-2 PHE rerank should still
        // surface it because over_fetch=3 captures the right candidate.
        let dp_cfg = PlanarLaplaceConfig::new(/*ε*/ 4.0, /*n*/ 4);
        let mut service = RemoteRagService::new(StubEmbedder, dp_cfg)
            .with_paillier_bits(256)
            .with_over_fetch_factor(3) // = corpus size, so Stage 1 returns everything
            .with_seed([11u8; 32]);
        service.ingest_chunks(corpus()).expect("ingest");

        let hits = service.query("apple pie", 1).expect("query");
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].id.0, "apple",
            "PHE rerank should restore the true top-1 even under Stage-1 noise"
        );
    }

    #[test]
    fn top_k_is_bounded_by_corpus_size() {
        let dp_cfg = PlanarLaplaceConfig::new(50.0, 4);
        let mut service = RemoteRagService::new(StubEmbedder, dp_cfg)
            .with_paillier_bits(256)
            .with_seed([3u8; 32]);
        service.ingest_chunks(corpus()).expect("ingest");

        // Asking for k=10 on a 3-doc corpus should return 3 hits.
        let hits = service.query("apple", 10).expect("query");
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn dimension_mismatch_errors() {
        // dp_cfg.n = 8 but stub embedder produces dim 4.
        let dp_cfg = PlanarLaplaceConfig::new(50.0, 8);
        let mut service = RemoteRagService::new(StubEmbedder, dp_cfg)
            .with_paillier_bits(256)
            .with_seed([5u8; 32]);
        service.ingest_chunks(corpus()).expect("ingest");
        let result = service.query("apple", 1);
        assert!(result.is_err(), "should reject query with dim mismatch");
    }
}
