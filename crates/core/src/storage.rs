use crate::{ChunkCiphertext, ChunkId, EncryptedEmbedding, top_k_by_similarity};

#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub chunk_id: ChunkId,
    pub encrypted_chunk: ChunkCiphertext,
    pub embedding: EncryptedEmbedding,
}

#[derive(Debug, Default, Clone)]
pub struct InMemoryEncryptedIndex {
    records: Vec<StoredChunk>,
}

impl InMemoryEncryptedIndex {
    pub fn insert(&mut self, encrypted_chunk: ChunkCiphertext, embedding: EncryptedEmbedding) {
        self.records.push(StoredChunk {
            chunk_id: encrypted_chunk.chunk_id.clone(),
            encrypted_chunk,
            embedding,
        });
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn search(
        &self,
        encrypted_query: &EncryptedEmbedding,
        top_k: usize,
    ) -> Vec<(ChunkCiphertext, EncryptedEmbedding, f32)> {
        let scored = top_k_by_similarity(
            &encrypted_query.vector,
            self.records.iter().map(|r| &r.embedding),
            top_k,
        );

        scored
            .into_iter()
            .map(|(idx, score)| {
                let record = &self.records[idx];
                (
                    record.encrypted_chunk.clone(),
                    record.embedding.clone(),
                    score,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::InMemoryEncryptedIndex;
    use crate::{ChunkCiphertext, ChunkId, EncryptedEmbedding};

    #[test]
    fn index_returns_most_similar_ciphertext() {
        let mut index = InMemoryEncryptedIndex::default();
        index.insert(
            ChunkCiphertext {
                chunk_id: ChunkId("a".into()),
                scheme: "aes-256-gcm",
                nonce: vec![1; 12],
                ciphertext: vec![7, 8, 9],
            },
            EncryptedEmbedding {
                scheme: "test",
                vector: vec![1.0, 0.0],
                nonce: vec![],
                original_dimension: 2,
            },
        );
        index.insert(
            ChunkCiphertext {
                chunk_id: ChunkId("b".into()),
                scheme: "aes-256-gcm",
                nonce: vec![2; 12],
                ciphertext: vec![1, 2, 3],
            },
            EncryptedEmbedding {
                scheme: "test",
                vector: vec![0.0, 1.0],
                nonce: vec![],
                original_dimension: 2,
            },
        );

        let query = EncryptedEmbedding {
            scheme: "test",
            vector: vec![0.9, 0.1],
            nonce: vec![],
            original_dimension: 2,
        };
        let hits = index.search(&query, 1);
        assert_eq!(hits[0].0.chunk_id.0, "a");
    }
}
