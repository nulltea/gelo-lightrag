//! XorMM-shaped volume-hiding encrypted multi-map — M2 baseline.
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.3, §7 M2.
//! Reference: Patel, Persiano, Yeo — *Practical Volume-Hiding EMM*,
//! CCS 2022.
//!
//! # Scope
//!
//! The full XorMM paper uses an XOR-filter for guaranteed O(1) bucket
//! placement. M2 ships the **same public API** with a simpler
//! **cuckoo-hashed bucket-padded** internal placement. Both schemes
//! deliver:
//!
//! - **Volume hiding** — every `get(k)` returns exactly `volume_bound`
//!   values, padded with sentinel dummies. Per-key list-length leakage
//!   reduces to a per-EMM constant.
//! - **Access-pattern hiding** — each `get(k)` reads exactly two
//!   server-side buckets (the two candidate slots of `k`). The
//!   adversary cannot distinguish which of the two holds the real
//!   record; both are AES-GCM ciphertexts of equal length.
//!
//! Internal placement defers to M2.1 (XOR-filter optimisation) when
//! the cuckoo path proves either too high-overhead or insufficient
//! for security at LightRAG corpus scale.
//!
//! # Layout
//!
//! At `build` time, every logical key `k` gets two candidate bucket
//! indices `(H_seed,1(k), H_seed,2(k))`. The placement algorithm
//! (cuckoo + small stash for unplaceable keys, paper-style) commits
//! each key to exactly one of those two buckets. Each bucket holds
//! one `(key_fingerprint, value_list_padded_to_volume_bound)` entry
//! or a dummy entry of equal byte length, AES-GCM-encrypted under a
//! key derived from the master `EmmKey`.

mod backend;
mod bucket;
mod client;
mod params;

pub use backend::{ByteStoreBackend, EncryptedBucket, InMemoryByteStore};
pub use client::{XorMmClient, XorMmError};
pub use params::XorMmParams;

/// 32-byte logical key (typically `HMAC(s_key, entity_name)` or
/// `HMAC(s_key, sorted_pair)` upstream). The EMM doesn't care about
/// origin — it sees opaque bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogicalKey(pub [u8; 32]);

/// Variable-length opaque value bytes. LightRAG adjacency lists
/// serialise `(src_id, tgt_id)` pairs into these; `source_id` EMM
/// stores raw chunk-id strings. Per-bucket the value list is padded
/// to `volume_bound` entries, each padded to the per-EMM
/// `value_bytes` constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalValue(pub Vec<u8>);
