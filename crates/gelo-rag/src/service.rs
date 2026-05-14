use anyhow::Result;
use rag_core::{
    AesChunkCipher, DocumentChunk, Embedder, EmbeddingEncryptionScheme, InMemoryEncryptedIndex,
    RetrievalHit,
};

use crate::attestation::{AttestationEvidence, AttestationVerifier};

pub struct GeloRagInMemoryService<E, S, V> {
    embedder: E,
    scheme: S,
    verifier: V,
    chunk_cipher: AesChunkCipher,
    index: InMemoryEncryptedIndex,
}

impl<E, S, V> GeloRagInMemoryService<E, S, V>
where
    E: Embedder,
    S: EmbeddingEncryptionScheme,
    V: AttestationVerifier,
{
    pub fn new(embedder: E, scheme: S, verifier: V) -> Self {
        Self {
            embedder,
            scheme,
            verifier,
            chunk_cipher: AesChunkCipher::generate(),
            index: InMemoryEncryptedIndex::default(),
        }
    }

    pub fn attest(&self, evidence: &AttestationEvidence) -> Result<()> {
        self.verifier.verify(evidence)
    }

    pub fn ingest_chunks(&mut self, chunks: Vec<DocumentChunk>) -> Result<()> {
        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
        let embeddings = self.embedder.embed(&texts)?;

        for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            let encrypted = self.scheme.encrypt_document(&embedding)?;
            let encrypted_chunk = self.chunk_cipher.encrypt_chunk(&chunk)?;
            self.index.insert(encrypted_chunk, encrypted);
        }

        Ok(())
    }

    pub fn query(&mut self, text: &str, top_k: usize) -> Result<Vec<RetrievalHit>> {
        let embeddings = self.embedder.embed(&[text.to_owned()])?;
        let encrypted_query = self.scheme.encrypt_query(&embeddings[0])?;
        self.index
            .search(&encrypted_query, top_k)
            .into_iter()
            .map(|(encrypted_chunk, embedding, score)| {
                let chunk = self.chunk_cipher.decrypt_chunk(&encrypted_chunk)?;
                Ok(RetrievalHit {
                    id: chunk.id,
                    score,
                    text: chunk.text,
                    embedding,
                })
            })
            .collect()
    }

    pub fn index_len(&self) -> usize {
        self.index.len()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rag_core::{
        ChunkId, DocumentChunk, Embedder, EmbeddingEncryptionScheme, EncryptedEmbedding,
    };

    use crate::{GeloRagInMemoryService, NoopAttestationVerifier};

    struct StubEmbedder;

    impl Embedder for StubEmbedder {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    if text.contains("apple") {
                        vec![1.0, 0.0]
                    } else if text.contains("banana") {
                        vec![0.0, 1.0]
                    } else {
                        vec![0.9, 0.1]
                    }
                })
                .collect())
        }
    }

    #[derive(Clone)]
    struct IdentityScheme;

    impl EmbeddingEncryptionScheme for IdentityScheme {
        fn scheme_name(&self) -> &'static str {
            "identity"
        }

        fn encrypt_document(&mut self, embedding: &[f32]) -> Result<EncryptedEmbedding> {
            Ok(EncryptedEmbedding {
                scheme: "identity",
                vector: embedding.to_vec(),
                nonce: vec![],
                original_dimension: embedding.len(),
            })
        }

        fn encrypt_query(&mut self, embedding: &[f32]) -> Result<EncryptedEmbedding> {
            self.encrypt_document(embedding)
        }

        fn decrypt_document(&mut self, ciphertext: &EncryptedEmbedding) -> Result<Vec<f32>> {
            Ok(ciphertext.vector.clone())
        }
    }

    #[test]
    fn gelo_rag_service_retrieves_expected_chunk() {
        let mut service =
            GeloRagInMemoryService::new(StubEmbedder, IdentityScheme, NoopAttestationVerifier);

        service
            .ingest_chunks(vec![
                DocumentChunk {
                    id: ChunkId("apple-doc".into()),
                    text: "apple orchard".into(),
                },
                DocumentChunk {
                    id: ChunkId("banana-doc".into()),
                    text: "banana bread".into(),
                },
            ])
            .unwrap();

        let hits = service.query("apple pie", 1).unwrap();
        assert_eq!(hits[0].id.0, "apple-doc");
    }
}
