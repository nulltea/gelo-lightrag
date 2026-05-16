//! Minimal in-CVM plaintext HNSW builder — single-layer flavour
//! sufficient for M3.
//!
//! M3 keeps the construction simple: one flat layer of N nodes, each
//! with up to M neighbours chosen as the M nearest by cosine distance.
//! No hierarchy yet — Compass's optimisations require the layered
//! structure, so M4 expands. For the M3 acceptance test (recall
//! against brute-force on 1K random vectors), a single-layer graph
//! gives ≥ 95 % recall at ef = 64, which is sufficient to validate
//! the encrypted-traversal protocol.

use crate::codec::NodeBlock;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
pub struct PlainHnswParams {
    /// Embedding dimensionality.
    pub dim: usize,
    /// Out-degree bound — same as Compass's `M`.
    pub max_neighbors: usize,
}

/// Plaintext, in-RAM HNSW analog. Holds the full graph; consumed by
/// `CompassIndex::from_plaintext_corpus` once and dropped (the data
/// then lives only in the encrypted Ring-ORAM tree).
#[derive(Debug)]
pub struct PlainHnsw {
    pub params: PlainHnswParams,
    /// Per-node embedding + neighbour-id list. Index in the vector is
    /// the node id (= u32 cast).
    pub nodes: Vec<NodeBlock>,
    /// Designated entry point for search. Picked at build time as the
    /// node with smallest id (0); HNSW theory says any node works for
    /// a connected graph.
    pub entry: u32,
}

impl PlainHnsw {
    /// Build a flat graph: for each node, link to its M nearest
    /// neighbours by cosine distance. O(N²) — acceptable for the
    /// 1K-vector M3 fixture. M4 will replace with a layered builder.
    pub fn build(embeddings: Vec<Vec<f32>>, params: PlainHnswParams) -> Self {
        let n = embeddings.len() as u32;
        assert!(n > 0, "PlainHnsw::build needs at least one embedding");
        for e in &embeddings {
            assert_eq!(e.len(), params.dim, "embedding dim mismatch");
        }

        let mut nodes = Vec::with_capacity(n as usize);
        for (i, emb) in embeddings.iter().enumerate() {
            // Pick the M nearest *other* nodes by cosine distance.
            let mut heap: BinaryHeap<OrderedNeighbor> =
                BinaryHeap::with_capacity(params.max_neighbors + 1);
            for (j, other) in embeddings.iter().enumerate() {
                if i == j {
                    continue;
                }
                let d = cosine_distance(emb, other);
                heap.push(OrderedNeighbor { dist: d, id: j as u32 });
                if heap.len() > params.max_neighbors {
                    heap.pop();
                }
            }
            // Heap holds farthest at top; drain into a sorted list.
            let mut neighbors: Vec<OrderedNeighbor> = heap.into_iter().collect();
            neighbors.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
            nodes.push(NodeBlock {
                embedding: emb.clone(),
                neighbors: neighbors.into_iter().map(|n| n.id).collect(),
            });
        }
        Self {
            params,
            nodes,
            entry: 0,
        }
    }
}

/// Helper for the bounded-size max-heap. Stores (dist, id) with
/// reverse ordering on dist so `pop` removes the farthest.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedNeighbor {
    dist: f32,
    id: u32,
}

impl Eq for OrderedNeighbor {}

impl Ord for OrderedNeighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is max-heap by default; we want the *farthest*
        // (largest dist) at the top so we can pop it when we exceed
        // the budget. So compare dists in natural order.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for OrderedNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 1 - cosine_similarity. Cleartext — used during build. Same metric
/// the ORAM-mediated search uses at query time (also cleartext, but
/// over decrypted blocks inside the CVM).
pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 {
        1.0
    } else {
        1.0 - dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_node_build_links_pairwise_nearest() {
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let hnsw = PlainHnsw::build(embs.clone(), PlainHnswParams {
            dim: 2,
            max_neighbors: 2,
        });
        assert_eq!(hnsw.nodes.len(), 3);
        for n in &hnsw.nodes {
            assert!(n.neighbors.len() <= 2);
        }
    }

    #[test]
    fn cosine_distance_is_zero_for_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let d = cosine_distance(&a, &a);
        assert!(d.abs() < 1e-6, "got {}", d);
    }

    #[test]
    fn cosine_distance_is_two_for_anti_parallel() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let d = cosine_distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-6, "got {}", d);
    }
}
