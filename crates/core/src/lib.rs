pub mod caprise;
pub mod content;
pub mod distance;
pub mod embed;
pub mod keying;
mod prf;
pub mod sap;
pub mod storage;
pub mod types;

pub use caprise::{Caprise, CapriseKey};
pub use content::AesChunkCipher;
pub use distance::{cosine_similarity, top_k_by_similarity};
pub use embed::FastEmbedEmbedder;
pub use keying::{HkdfPolicy, SchemeParams, TenantId};
pub use sap::{SapKey, SapScheme};
pub use storage::{InMemoryEncryptedIndex, StoredChunk};
pub use types::{
    ChunkCiphertext, ChunkId, DocumentChunk, Embedder, EmbeddingEncryptionScheme,
    EncryptedEmbedding, RetrievalHit,
};
