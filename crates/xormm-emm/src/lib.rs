//! XorMM volume-hiding encrypted multi-map — skeleton (M0).
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.3.
//! Reference: Patel, Persiano, Yeo — *Practical Volume-Hiding EMM*,
//! CCS 2022.
//!
//! Used as the substrate for LightRAG's adjacency-list and per-entity /
//! per-edge `source_id` multi-maps. M2 lands the static build + `get` +
//! `get_batch`. Incremental ingest is handled by a small fresh-tier
//! buffer rebuilt at threshold (≥ 1024 inserts or 5 % growth).

/// XorMM construction parameters. Pinned in the
/// [`crate::keying::XorMmParams`] canonical encoding so attestation
/// covers the volume-hiding budget chosen at build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XorMmParams {
    /// Hard upper bound on the response-set cardinality. All `get`
    /// responses are bucket-padded to this length, so any value
    /// dependence on per-key volume is hidden.
    pub volume_bound: u32,
    /// In-TEE stash size for overflow entries (paper §4).
    pub stash_size: u32,
}

impl Default for XorMmParams {
    fn default() -> Self {
        Self {
            volume_bound: 64,
            stash_size: 128,
        }
    }
}
