//! Strawman ORAM-mediated greedy HNSW search.
//!
//! Algorithm (paper §4.3, no optimisations):
//!
//! 1. Start at the entry node. Push (dist(entry, query), entry) onto
//!    a min-heap `candidates` and a max-heap `top_k`.
//! 2. While `candidates` is non-empty:
//!    a. Pop the closest unvisited `current` from `candidates`.
//!    b. If `current.dist > top_k.peek().dist` and `top_k.len() ≥ ef`,
//!       break — no closer node can improve the result.
//!    c. ORAM-read `current.block` to get its neighbours.
//!    d. For each unvisited neighbour `n`, ORAM-read its block, compute
//!       `dist(n.embedding, query)`, push to both heaps.
//!    e. Bound `top_k` to `ef` by dropping the farthest.
//! 3. Return the `k` smallest from `top_k`.
//!
//! Each visited node requires one ORAM read for its neighbour list +
//! D-vector. Each unvisited neighbour evaluated requires another. For
//! ef=64 / M=16 the typical search visits ~50 nodes ⇒ ~50 reads.
//! M4 reduces this with Directional Filter + Speculative Prefetch.

use ring_oram::{BlockBackend, BlockId};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::hnsw_plain::cosine_distance;
use crate::index::{CompassIndex, CompassIndexError};

#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Smaller dist = "less" (we want min-heap for candidates).
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

pub(crate) fn strawman_search<B: BlockBackend>(
    index: &mut CompassIndex<B>,
    query: &[f32],
    k: usize,
) -> Result<Vec<u32>, CompassIndexError> {
    assert_eq!(
        query.len(),
        index.params.hnsw.dim,
        "query dim mismatch with CompassIndex"
    );

    let ef = index.params.ef_search.max(k);
    let entry_dist = cosine_distance(query, &index.entry_embedding);
    let entry_cand = Candidate { dist: entry_dist, id: index.entry.0 };

    // candidates: min-heap by dist (use Reverse to make BinaryHeap min-style).
    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    candidates.push(Reverse(entry_cand));
    // top_k: max-heap by dist; size bounded by ef. Largest at top.
    let mut top_k: BinaryHeap<Candidate> = BinaryHeap::new();
    top_k.push(entry_cand);

    let mut visited: HashSet<u32> = HashSet::new();
    visited.insert(index.entry.0);

    while let Some(Reverse(current)) = candidates.pop() {
        // Early termination: if current is farther than the worst of
        // the current top_k AND top_k is full, we can't improve.
        if top_k.len() >= ef {
            let farthest = top_k.peek().expect("non-empty top_k").dist;
            if current.dist > farthest {
                break;
            }
        }

        // ORAM-read current's neighbour list.
        let node = index.read_node(BlockId(current.id))?;

        // Visit each neighbour: ORAM-read its embedding, score, push.
        for &nb_id in &node.neighbors {
            if !visited.insert(nb_id) {
                continue;
            }
            let nb_node = index.read_node(BlockId(nb_id))?;
            let nb_dist = cosine_distance(query, &nb_node.embedding);
            let cand = Candidate { dist: nb_dist, id: nb_id };
            // Maybe add to top_k.
            if top_k.len() < ef {
                top_k.push(cand);
                candidates.push(Reverse(cand));
            } else {
                let farthest = top_k.peek().expect("non-empty top_k").dist;
                if nb_dist < farthest {
                    top_k.pop();
                    top_k.push(cand);
                    candidates.push(Reverse(cand));
                }
            }
        }
    }

    // Extract top-k smallest from the max-heap (need to invert).
    let mut all: Vec<Candidate> = top_k.into_iter().collect();
    all.sort_by(|a, b| a.cmp(b));
    Ok(all.into_iter().take(k).map(|c| c.id).collect())
}
