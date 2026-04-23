use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub String);

impl From<&str> for ChunkId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: ChunkId,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedEmbedding {
    pub scheme: &'static str,
    pub vector: Vec<f32>,
    pub nonce: Vec<u8>,
    pub original_dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkCiphertext {
    pub chunk_id: ChunkId,
    pub scheme: &'static str,
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub id: ChunkId,
    pub score: f32,
    pub text: String,
    pub embedding: EncryptedEmbedding,
}

pub trait Embedder {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

pub trait EmbeddingEncryptionScheme {
    fn scheme_name(&self) -> &'static str;
    fn encrypt_document(&mut self, embedding: &[f32]) -> anyhow::Result<EncryptedEmbedding>;
    fn encrypt_query(&mut self, embedding: &[f32]) -> anyhow::Result<EncryptedEmbedding>;
    fn decrypt_document(&mut self, ciphertext: &EncryptedEmbedding) -> anyhow::Result<Vec<f32>>;
}
