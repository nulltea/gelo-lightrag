//! Compass-style HNSW-over-Ring-ORAM — M3 strawman baseline.
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.2.
//! Reference: Zhu, Patel, Zaharia, Popa — *Compass: Encrypted Semantic
//! Search with High Accuracy*, OSDI 2025; artifact
//! [`Clive2312/compass`](https://github.com/Clive2312/compass).
//!
//! # Status
//!
//! M3 ships the **strawman** ORAM-mediated HNSW search (paper §4.3) —
//! one ORAM read per visited node, no optimisations. Correct but slow.
//! M4 layers on the three Compass tricks:
//!
//! - *Directional Neighbor Filtering* — quantized hints prune
//!   neighbours before fetching their full embeddings.
//! - *Speculative Neighbor Prefetch* — batched `ef_spec` candidates
//!   per ORAM round.
//! - *Graph-Traversal-Tailored ORAM* — multi-hop lazy eviction +
//!   treetop caching; embeddings + neighbour lists colocated in one
//!   block (already done in this M3 layout).
//!
//! # Layout per ORAM block
//!
//! ```text
//! [embedding: D × f32-LE] ‖ [neighbor_count u32-LE] ‖ [neighbor_ids: M × u32-LE] ‖ [padding]
//! ```
//!
//! Total = `4D + 4 + 4M` bytes, padded up to `RingOramParams::block_bytes`.
//! D=128, M=16 ⇒ 580 bytes; M3 default. CompassParams's production
//! D=768, M=12 ⇒ 3124 bytes, requires a 4 KB block_bytes bump and a
//! re-attestation.

mod codec;
mod hnsw_plain;
mod index;
mod search;

pub use codec::NodeBlock;
pub use hnsw_plain::{PlainHnsw, PlainHnswParams};
pub use index::{CompassIndex, CompassIndexParams, CompassIndexError};
pub use ring_oram::RingOramParams;
