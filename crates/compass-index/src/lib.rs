//! Compass-style HNSW-over-Ring-ORAM — skeleton (M0).
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.2.
//! Reference: Zhu, Patel, Zaharia, Popa — *Compass: Encrypted Semantic
//! Search with High Accuracy*, OSDI 2025; artifact
//! [`Clive2312/compass`](https://github.com/Clive2312/compass).
//!
//! M3 lands the plain Ring-ORAM HNSW; M4 adds the three Compass
//! optimisations (Directional Neighbor Filtering, Speculative Neighbor
//! Prefetch, Graph-Traversal-Tailored ORAM).

pub use ring_oram::RingOramParams;
