//! Ring-ORAM parameters. Mirrors `rag_core::keying::CompassParams` for
//! the subset of fields the ORAM layer reads.

use crate::path::tree_levels;

/// Ring-ORAM bucket + tree parameters. Locked at index construction time
/// and pinned into the V2 attestation `scheme_identity` via
/// `rag_core::keying::CompassParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingOramParams {
    /// Real-block slots per bucket. Plaintext blocks = `n_blocks`,
    /// total tree capacity = `Z · num_buckets` real slots.
    pub z: u32,
    /// Dummy slots per bucket. Read-target masking + early-reshuffle
    /// budget. Compass paper §2.2 / Tab. 3 default `S = 5` for `Z = 4`.
    pub s: u32,
    /// Eviction rate: `EvictPath` runs every `a` `ReadPath` ops.
    pub a: u32,
    /// AES-GCM-encrypted payload bytes per block (excluding 12-byte
    /// nonce + 16-byte tag, which the backend layers on top).
    pub block_bytes: u32,
    /// Tree leaf count = `2^(levels - 1)`. The number of distinct
    /// `path_id` values. `1` ⇒ degenerate single-bucket tree.
    pub n_leaves: u32,
    /// Number of top levels of the ORAM tree cached client-side in
    /// CVM RAM (Compass paper §4.7). Backend reads/writes for buckets
    /// in the top `treetop_levels` are mirrored into the cache; reads
    /// hit the cache; writes go to both cache and backend (for
    /// recovery). `0` ⇒ no caching, equivalent to the M1 baseline.
    /// Setting this large saves bandwidth on a networked backend at
    /// the cost of CVM RAM (each cached bucket is one AES-GCM frame).
    pub treetop_levels: u32,
}

impl Default for RingOramParams {
    /// M1 default: 64-leaf tree (= 127 buckets, 7 levels). Tiny — picked
    /// for fast tests. Production sizing is per-corpus and pinned via
    /// `CompassParams` at index build time.
    fn default() -> Self {
        Self {
            z: 4,
            s: 5,
            a: 3,
            block_bytes: 2048,
            n_leaves: 64,
            treetop_levels: 0,
        }
    }
}

impl RingOramParams {
    /// `Z + S` — total block slots per bucket.
    pub fn bucket_capacity(&self) -> u32 {
        self.z + self.s
    }

    /// `L` — number of levels in the tree. `n_leaves = 2^(L-1)`.
    pub fn levels(&self) -> u32 {
        tree_levels(self.n_leaves)
    }

    /// `2^L - 1` — total bucket count (root + every interior node + every leaf).
    pub fn num_buckets(&self) -> u32 {
        (1u32 << self.levels()) - 1
    }

    /// Number of buckets that live in the treetop cache. `2^t - 1`
    /// for `t = treetop_levels`. `0` when `treetop_levels == 0`.
    pub fn treetop_bucket_count(&self) -> u32 {
        if self.treetop_levels == 0 {
            0
        } else {
            (1u32 << self.treetop_levels) - 1
        }
    }

    /// Predicate: does `bucket_id` fall inside the cached treetop?
    pub fn bucket_in_treetop(&self, bucket_id: u32) -> bool {
        bucket_id < self.treetop_bucket_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_are_well_formed() {
        let p = RingOramParams::default();
        assert_eq!(p.bucket_capacity(), 9);
        // 64 leaves ⇒ 7 levels (1, 2, 4, …, 64) ⇒ 127 buckets.
        assert_eq!(p.levels(), 7);
        assert_eq!(p.num_buckets(), 127);
    }

    #[test]
    fn bucket_capacity_is_z_plus_s() {
        let p = RingOramParams { z: 7, s: 3, ..RingOramParams::default() };
        assert_eq!(p.bucket_capacity(), 10);
    }
}
