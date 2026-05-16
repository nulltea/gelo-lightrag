//! LightRAG retrieval protocol over an encrypted graph store — M7.1
//! ships the Local mode + `search_perturb`. Subsequent M7.x milestones
//! add the remaining modes, the `_token_truncation` cap, the upstream
//! `_build_context_str` template, and the `(weight, cosine)` edge-sort.
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.5, §7 M7.
//!
//! This crate owns:
//! - `search_perturb` — per-session HMAC perturbation on each
//!   embedding (§8.6) driven by the `search_pattern_key` HKDF child.
//! - `LightRagPrivateService` — the `kg_query` orchestrator over
//!   `LightKgStore`.

mod perturb;
mod service;

pub use perturb::{
    DEFAULT_EPSILON, EmbeddingKind, SessionKey, perturb, perturb_with_epsilon,
};
pub use service::{KgContext, KgQueryParams, LightRagPrivateService, QueryShape};

/// LightRAG retrieval modes. Mirrors upstream
/// `lightrag/operate.py:QueryParam.mode`. M7.1 wires only `Local` —
/// the other modes will hang their `QueryShape` variant off of this
/// enum as they ship.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    Local,
    Global,
    Hybrid,
    Mix,
    Naive,
}
