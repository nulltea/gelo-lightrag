//! Position map: `BlockId → PathId`. Held *inside the CVM* by the
//! Ring-ORAM client; never visible to the storage server.
//!
//! In production a position map for N blocks is N · log₂(N) bits and
//! can itself be recursively stored in another ORAM (the standard
//! Path-ORAM / Ring-ORAM recursion trick) when N grows. For the
//! Variant-A LightRAG corpora (10⁴ – 10⁶ entries) the flat map fits
//! comfortably in CVM RAM — at N = 10⁶, a `HashMap<BlockId, PathId>`
//! is ~16 MB, well under the per-tenant budget. We start flat; the
//! recursion knob is deferred to M4 alongside the Compass-tailored
//! optimisations.

use crate::block::BlockId;
use crate::path::PathId;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PositionMap {
    inner: HashMap<BlockId, PathId>,
}

impl PositionMap {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Pre-allocate for `cap` entries. Useful at construction time.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(cap),
        }
    }

    /// Look up the path currently assigned to `block_id`. Returns
    /// `None` for blocks that have never been admitted.
    pub fn get(&self, block_id: BlockId) -> Option<PathId> {
        self.inner.get(&block_id).copied()
    }

    /// Assign or reassign `block_id` to `path_id`. Ring-ORAM remaps
    /// every accessed block to a fresh uniform-random path on each
    /// access (the source of access-pattern hiding), so callers
    /// invoke this on every `ReadPath`.
    pub fn set(&mut self, block_id: BlockId, path_id: PathId) {
        debug_assert!(
            !block_id.is_dummy(),
            "dummy blocks must not enter the position map"
        );
        self.inner.insert(block_id, path_id);
    }

    /// Number of admitted blocks.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for PositionMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_map_returns_none() {
        let pm = PositionMap::new();
        assert!(pm.get(BlockId(0)).is_none());
    }

    #[test]
    fn set_then_get_round_trips() {
        let mut pm = PositionMap::new();
        pm.set(BlockId(7), PathId(3));
        assert_eq!(pm.get(BlockId(7)), Some(PathId(3)));
    }

    #[test]
    fn set_overwrites() {
        let mut pm = PositionMap::new();
        pm.set(BlockId(1), PathId(5));
        pm.set(BlockId(1), PathId(2));
        assert_eq!(pm.get(BlockId(1)), Some(PathId(2)));
    }

    #[test]
    #[should_panic(expected = "dummy")]
    #[cfg(debug_assertions)]
    fn dummy_block_in_posmap_panics_in_debug() {
        let mut pm = PositionMap::new();
        pm.set(BlockId::DUMMY, PathId(0));
    }
}
