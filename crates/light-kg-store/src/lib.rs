//! LightRAG-shaped storage facade — M6 wiring.
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.4, §7 M6.
//!
//! Composes the six encrypted stores that back the LightRAG retrieval
//! surface (`kg_query`, chunk fan-out, adjacency lookup):
//!
//! - 3× [`compass_index::CompassIndex`] — entities, relations, chunks
//! - 2× [`xormm_emm::XorMmClient`] — adjacency, source_chunks
//! - 1× [`AesChunkStore`] — chunk text under AES-GCM
//!
//! Per-tenant keys are derived once via
//! [`rag_core::keying::HkdfPolicyV2`]; the eight V2 child keys split:
//! `caprise_seed` and `search_pattern_key` are consumed elsewhere
//! (embedder + retrieval respectively), the remaining six are held by
//! [`LightKgStore::keys`].
//!
//! See [`LightKgStore::build_from_kg`] for the build entry point.

mod aes_chunk_store;
mod keys;
mod store;
mod types;

pub use aes_chunk_store::{AesChunkStore, ChunkStoreError};
pub use compass_index::{CompassIndexParams, RingOramParams};
pub use keys::{derive_logical_key, label};
pub use store::{KeyBundle, LightKgError, LightKgParams, LightKgStore};
pub use types::{Chunk, Entity, ExtractedKg, Relation};
pub use xormm_emm::XorMmParams;
