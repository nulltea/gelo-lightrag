//! Layered Compass search (M4.6).
//!
//! 1. **Upper-layer descent (cleartext).** Start at the entry node
//!    at `top_layer`. At each layer, ef=1 greedy descent over the
//!    in-CVM cleartext cache. No ORAM traffic.
//! 2. **Layer-0 beam search (ORAM-mediated).** Standard
//!    `ef_search`-bounded beam search; every visited node = one
//!    Ring-ORAM `read`. Return top-k.
//!
//! Compared to the M3 strawman (paper §4.3), this is the natural
//! layered traversal — same number of layer-0 reads but the upper
//! layers no longer cost ORAM round-trips. M4.1–M4.5 will further
//! reduce layer-0 reads via Directional Filter + Speculative
//! Prefetch + Lazy Eviction.

use ring_oram::{BlockBackend, BlockId};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::hints::score_packed;
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
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

pub(crate) fn layered_search<B: BlockBackend>(
    index: &mut CompassIndex<B>,
    query: &[f32],
    k: usize,
) -> Result<Vec<u32>, CompassIndexError> {
    assert_eq!(
        query.len(),
        index.params.hnsw.dim,
        "query dim mismatch with CompassIndex"
    );

    // Multi-hop lazy eviction (Compass §4.7): keep eviction writes
    // off the user-perceived critical path. The RAII guard flushes
    // pending evictions when search returns.
    let _evict_guard = index.defer_evictions_for_search();
    let index: &mut CompassIndex<B> = _evict_guard.index;

    // ─── 1. Upper-layer descent (cleartext) ────────────────────────
    let mut current = index.entry.0;
    let entry_emb = upper_layer_embedding(index, current, index.top_layer)
        .expect("entry must exist at top_layer");
    let mut cur_dist = cosine_distance(query, entry_emb);

    // Walk down through all upper layers (top_layer, top_layer-1, …, 1).
    for l in (1..=index.top_layer).rev() {
        loop {
            let nbrs = upper_layer_neighbours(index, current, l).to_vec();
            let mut improved = false;
            for n in nbrs {
                if let Some(emb) = upper_layer_embedding(index, n, l) {
                    let d = cosine_distance(query, emb);
                    if d < cur_dist {
                        current = n;
                        cur_dist = d;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }

    // ─── 2. Layer-0 beam search (ORAM-mediated) ─────────────────────
    let ef = index.params.ef_search.max(k);
    let entry_cand = Candidate { id: current, dist: cur_dist };

    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    candidates.push(Reverse(entry_cand));
    let mut top_k: BinaryHeap<Candidate> = BinaryHeap::new();
    top_k.push(entry_cand);

    let mut visited: HashSet<u32> = HashSet::new();
    visited.insert(current);

    let ef_n = index.params.ef_n;

    while let Some(Reverse(c)) = candidates.pop() {
        if top_k.len() >= ef {
            let farthest = top_k.peek().expect("non-empty top_k").dist;
            if c.dist > farthest {
                break;
            }
        }
        let node = index.read_layer0_node(BlockId(c.id))?;

        // Directional Neighbor Filtering (Compass §4.5). Pick top
        // `ef_n` neighbours by quantised-hint dot product with the
        // query direction from `c`. Only those get ORAM-fetched.
        let candidates_for_fetch: Vec<u32> = if ef_n >= node.neighbors.len() {
            node.neighbors.clone()
        } else {
            let q_dir: Vec<f32> = query
                .iter()
                .zip(node.embedding.iter())
                .map(|(q, e)| q - e)
                .collect();
            let hints = index.layer0_node_hints(c.id);
            let mut scored: Vec<(f32, u32)> = node
                .neighbors
                .iter()
                .zip(hints.packed_hints.iter())
                .map(|(&nb_id, packed)| (score_packed(packed, &q_dir), nb_id))
                .collect();
            // Higher score = better aligned with query direction.
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.into_iter().take(ef_n).map(|(_, id)| id).collect()
        };

        for nb_id in candidates_for_fetch {
            if !visited.insert(nb_id) {
                continue;
            }
            let nb_node = index.read_layer0_node(BlockId(nb_id))?;
            let nb_dist = cosine_distance(query, &nb_node.embedding);
            let cand = Candidate { dist: nb_dist, id: nb_id };
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

    let mut all: Vec<Candidate> = top_k.into_iter().collect();
    all.sort_by(|a, b| a.cmp(b));
    Ok(all.into_iter().take(k).map(|c| c.id).collect())
}

fn upper_layer_embedding<'a, B: BlockBackend>(
    index: &'a CompassIndex<B>,
    id: u32,
    layer: u32,
) -> Option<&'a [f32]> {
    if layer == 0 {
        return None;
    }
    let l_idx = (layer - 1) as usize;
    if l_idx >= index.upper_layers.len() {
        return None;
    }
    index.upper_layers[l_idx]
        .nodes
        .get(&id)
        .map(|n| n.embedding.as_slice())
}

fn upper_layer_neighbours<'a, B: BlockBackend>(
    index: &'a CompassIndex<B>,
    id: u32,
    layer: u32,
) -> &'a [u32] {
    if layer == 0 {
        return &[];
    }
    let l_idx = (layer - 1) as usize;
    if l_idx >= index.upper_layers.len() {
        return &[];
    }
    index.upper_layers[l_idx]
        .nodes
        .get(&id)
        .map(|n| n.neighbours.as_slice())
        .unwrap_or(&[])
}
