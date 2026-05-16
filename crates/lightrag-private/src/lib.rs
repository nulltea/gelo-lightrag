//! LightRAG retrieval protocol over an encrypted graph store — skeleton (M0).
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.5.
//!
//! M7 lands a faithful port of LightRAG's `kg_query` /
//! `_perform_kg_search` / `_apply_token_truncation` / `_merge_all_chunks`
//! / `_build_context_str` against the [`light_kg_store::LightKgStore`]
//! facade.
//!
//! This crate also owns `search_perturb` — the per-session HMAC
//! perturbation on each embedding (§8.6 of the plan) — driven by the
//! `search_pattern_key` HKDF child.

/// LightRAG retrieval modes. Mirrors upstream
/// `lightrag/operate.py:QueryParam.mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    Local,
    Global,
    Hybrid,
    Mix,
    Naive,
}
