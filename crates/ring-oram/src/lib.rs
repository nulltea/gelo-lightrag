//! Ring-ORAM client + server protocol — semi-honest baseline (M1).
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.1, §7 M1.
//! Reference: Ren et al., USENIX Security 2015; port from
//! [`Clive2312/compass`](https://github.com/Clive2312/compass) (M4
//! parity gate).
//!
//! # Status
//!
//! **Experimental — not production.** M1 ships the *correct* semi-honest
//! baseline: client-held position map + stash, server-held encrypted
//! bucket tree, AES-GCM bucket encryption with `nonce = bucket_id ‖
//! write_counter`. Deferred to later milestones:
//!
//! - **M4** — XOR trick for constant online bandwidth; Merkle integrity
//!   for malicious-server resistance; multi-block batched reads;
//!   multi-hop lazy eviction; the three Compass HNSW optimisations.
//! - **M5** — networked `BlockBackend` (replacing the in-memory one).
//!
//! # Layout
//!
//! A complete binary tree with `2^L - 1` buckets at `L` levels.
//! `path_id ∈ [0, 2^(L-1))` identifies a leaf; `path_buckets(path_id)`
//! returns the `L` bucket indices from root → leaf. Every block lives
//! either in the stash or on a single path; the [`PositionMap`] tracks
//! which.
//!
//! Each bucket holds `Z + S` block slots — `Z` real, `S` dummy — exactly
//! as the paper specifies. The bucket payload is one AES-GCM frame; the
//! 12-byte nonce is structured `u64-LE bucket_id ‖ u32-LE write_counter`
//! so re-encrypting after eviction never reuses a nonce.

mod backend;
mod block;
mod client;
mod codec;
mod params;
mod path;
mod posmap;
mod stash;

pub use backend::{BlockBackend, EncryptedBucket, InMemoryBlockBackend};
pub use block::{Block, BlockId, BlockPayload};
pub use client::{OramError, RingOramClient};
pub use params::RingOramParams;
pub use path::{PathId, path_buckets, total_buckets, tree_levels};
pub use posmap::PositionMap;
pub use stash::Stash;
