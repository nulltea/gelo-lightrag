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

    /// Stable identifier for the loaded model — typically the SHA-256 of the
    /// weights manifest. Used by attestation backends to bind a CVM report to
    /// "this specific publicly-known model was loaded".
    ///
    /// The default returns `b""`, which a default `NoopAttestationVerifier`
    /// will accept; production embedders that participate in attestation
    /// (e.g. `GeloBertEmbedder` / `GeloQwenEmbedder`) override with the actual
    /// hash bytes cached at model load.
    fn model_identity(&self) -> &[u8] {
        b""
    }
}

/// Forwarding impl for `Box<dyn Embedder>`. Useful when a caller wants to
/// choose between embedder variants at runtime (e.g. caching wrapper on/off)
/// without making the consumer generic over concrete embedder types.
impl Embedder for Box<dyn Embedder> {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        (**self).embed(texts)
    }
    fn model_identity(&self) -> &[u8] {
        (**self).model_identity()
    }
}

pub trait EmbeddingEncryptionScheme {
    fn scheme_name(&self) -> &'static str;
    fn encrypt_document(&mut self, embedding: &[f32]) -> anyhow::Result<EncryptedEmbedding>;
    fn encrypt_query(&mut self, embedding: &[f32]) -> anyhow::Result<EncryptedEmbedding>;
    fn decrypt_document(&mut self, ciphertext: &EncryptedEmbedding) -> anyhow::Result<Vec<f32>>;
}
