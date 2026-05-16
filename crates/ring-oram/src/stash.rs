//! Client-side stash: real blocks read from a path that don't fit
//! back into the tree on eviction. Held *inside the CVM*.
//!
//! Paper bound: with `Z = 4`, `S = 5`, `A = 3`, the stash size is
//! O(log N) in expectation and stays small in practice. We don't
//! enforce a hard cap in this M1 baseline — instead we surface the
//! current size for tests, and M4 will add a configurable bound +
//! overflow telemetry per the Compass paper's stash-bound proof
//! sketch (§4.7).

use crate::block::{Block, BlockId};
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct Stash {
    blocks: HashMap<BlockId, Block>,
}

impl Stash {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take ownership of a block (panics on duplicate insert — that
    /// would be a protocol bug).
    pub fn insert(&mut self, block: Block) {
        debug_assert!(!block.is_dummy(), "dummy blocks must not enter the stash");
        let prev = self.blocks.insert(block.id, block);
        debug_assert!(
            prev.is_none(),
            "stash duplicate insert — protocol bug; same BlockId fetched twice without intervening write-back"
        );
    }

    /// Remove and return the block with `id`, if present.
    pub fn take(&mut self, id: BlockId) -> Option<Block> {
        self.blocks.remove(&id)
    }

    /// Borrow a block without removing it. Used during eviction
    /// candidate selection.
    pub fn peek(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(&id)
    }

    /// All resident block ids. Used by eviction to scan candidates.
    pub fn ids(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.blocks.keys().copied()
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Drain blocks matching a predicate into a Vec. Used by
    /// `evict_path` to pull stash blocks that can land on the
    /// evicted path.
    pub fn drain_matching<F>(&mut self, mut pred: F) -> Vec<Block>
    where
        F: FnMut(BlockId) -> bool,
    {
        let take_ids: Vec<BlockId> = self.blocks.keys().copied().filter(|id| pred(*id)).collect();
        take_ids
            .into_iter()
            .map(|id| self.blocks.remove(&id).expect("just listed"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockPayload;

    fn b(id: u32) -> Block {
        Block::new(BlockId(id), BlockPayload::zero(8))
    }

    #[test]
    fn insert_then_take_round_trips() {
        let mut s = Stash::new();
        s.insert(b(1));
        assert_eq!(s.len(), 1);
        let got = s.take(BlockId(1));
        assert!(got.is_some());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn drain_matching_pulls_selected_blocks() {
        let mut s = Stash::new();
        for i in 0..5 {
            s.insert(b(i));
        }
        let evens = s.drain_matching(|id| id.0 % 2 == 0);
        assert_eq!(evens.len(), 3);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn ids_iterates_all_resident() {
        let mut s = Stash::new();
        s.insert(b(7));
        s.insert(b(11));
        let mut got: Vec<u32> = s.ids().map(|i| i.0).collect();
        got.sort();
        assert_eq!(got, vec![7, 11]);
    }
}
