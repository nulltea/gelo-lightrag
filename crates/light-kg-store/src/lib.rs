//! LightRAG-shaped storage facade — skeleton (M0).
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.4.
//!
//! Composes:
//! - 3× [`compass_index::CompassIndex`] — entities, relations, chunks
//! - 2× [`xormm_emm::XorMmClient`] — adjacency, `source_id`
//! - 2× encrypted KV stores — node props, edge props (also Ring-ORAM)
//! - 1× AES-GCM chunk store
//!
//! Per-tenant keys derived once via
//! [`rag_core::keying::HkdfPolicyV2`]; eight child keys total —
//! `caprise_seed`, `aes_chunk_key`, `oram_keys × 3`, `emm_keys × 2`,
//! `search_pattern_key`.

pub use compass_index::RingOramParams;
pub use xormm_emm::XorMmParams;
