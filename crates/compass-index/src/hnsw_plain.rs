//! Plaintext, in-CVM layered HNSW builder — paper-faithful enough to
//! drive Compass's encrypted search (M4.6).
//!
//! Algorithm: Malkov & Yashunin (TPAMI 2018), "Efficient and robust
//! approximate nearest neighbor search using Hierarchical Navigable
//! Small World graphs."
//!
//! - Per-node layer drawn from `l = ⌊-ln(U(0,1)) · mL⌋` with
//!   `mL = 1 / ln(M)`.
//! - Insertion sorted by decreasing level so the entry point exists
//!   when each node is inserted.
//! - At each layer above the new node's level, ef=1 greedy descent.
//! - From the new node's level down to 0, ef_construction-bounded
//!   beam search, then Algorithm 4 neighbour selection bounded by
//!   `M_l0` at layer 0 / `M_upper` elsewhere.
//!
//! After build, `PlainHnsw::upper_graph[l]` is a sparse adjacency
//! map (only nodes assigned to layer ≥ l have entries); `layer0[id]`
//! is a dense vector indexed by node id. `CompassIndex` keeps the
//! upper graphs + their embeddings cleartext inside the CVM and
//! pushes only `layer0` nodes through the ORAM.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::codec::NodeBlock;
use crate::hints::{NodeHints, pack_direction};

/// HNSW build configuration. Derived from
/// `rag_core::keying::CompassParams` defaults at the
/// `CompassIndexParams::default()` boundary.
#[derive(Debug, Clone, Copy)]
pub struct PlainHnswParams {
    pub dim: usize,
    /// Out-degree at layer 0 (densest layer). Paper default `2·M`.
    pub max_neighbors_l0: usize,
    /// Out-degree at layers ≥ 1. Paper default `M`.
    pub max_neighbors_upper: usize,
    /// Beam width during insertion at each layer. Paper default 200.
    pub ef_construction: usize,
    /// Layer-assignment scale `mL = 1/ln(M)`. Larger ⇒ taller tree.
    pub ml: f32,
    /// PRNG seed for layer assignment. Same `(corpus, seed)` ⇒
    /// byte-identical graph.
    pub build_seed: [u8; 32],
    /// Neighbour selection rule. Algorithm 4 is the paper default
    /// and the only path verified against `hnsw_rs`'s output.
    pub neighbour_heuristic: NeighbourHeuristic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighbourHeuristic {
    /// Malkov-Yashunin Algorithm 4 — diverse-set rule.
    Algorithm4,
    /// Naive top-M by distance.
    Simple,
}

impl PlainHnswParams {
    /// Paper-default `PlainHnswParams` for the given `M`.
    pub fn paper_defaults(dim: usize, m: usize) -> Self {
        let ml = 1.0_f32 / (m as f32).ln();
        Self {
            dim,
            max_neighbors_l0: 2 * m,
            max_neighbors_upper: m,
            ef_construction: 200,
            ml,
            build_seed: [0u8; 32], // tenants override; tests use this default
            neighbour_heuristic: NeighbourHeuristic::Algorithm4,
        }
    }
}

/// In-CVM plaintext HNSW. Held during `compass_init`; the upper
/// layers + their embeddings persist in `CompassIndex` cleartext; the
/// bottom layer becomes encrypted ORAM blocks and is dropped.
#[derive(Debug)]
pub struct PlainHnsw {
    pub params: PlainHnswParams,
    /// Embeddings indexed by node id.
    pub embeddings: Vec<Vec<f32>>,
    /// Per-node maximum assigned layer (`l(x)` in the paper).
    pub levels: Vec<u32>,
    /// Adjacency at layer 0 — dense (every node lives at layer 0).
    pub layer0: Vec<Vec<u32>>,
    /// Adjacency at layers ≥ 1. `upper[l - 1][&id]` is `id`'s
    /// neighbour list at layer `l`. Sparse — only nodes with `level ≥ l`
    /// appear.
    pub upper: Vec<HashMap<u32, Vec<u32>>>,
    /// Entry point — the node with the highest assigned layer (ties
    /// broken by lowest id, deterministically).
    pub entry: u32,
    /// `top_layer = upper.len()`. May be 0 (degenerate single-layer
    /// graph) when no node sampled a level > 0.
    pub top_layer: u32,
}

impl PlainHnsw {
    /// Build a layered HNSW over `embeddings`. O(N · ef_construction · M)
    /// time roughly — acceptable for the M4 fixture corpora (up to ~10K
    /// vectors); a single-thread build runs in seconds.
    pub fn build(embeddings: Vec<Vec<f32>>, params: PlainHnswParams) -> Self {
        let n = embeddings.len() as u32;
        assert!(n > 0, "PlainHnsw::build needs at least one embedding");
        for e in &embeddings {
            assert_eq!(e.len(), params.dim, "embedding dim mismatch");
        }

        // 1. Sample levels.
        let mut rng = ChaCha20Rng::from_seed(params.build_seed);
        let mut levels = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let u: f32 = rng.random_range(f32::MIN_POSITIVE..1.0);
            let l = ((-u.ln()) * params.ml).floor() as u32;
            levels.push(l);
        }
        let top_layer = *levels.iter().max().unwrap_or(&0);

        // 2. Order nodes by decreasing level so the entry point is
        //    always already inserted. Stable tie-break by id.
        let mut order: Vec<u32> = (0..n).collect();
        order.sort_by(|&a, &b| {
            levels[b as usize]
                .cmp(&levels[a as usize])
                .then(a.cmp(&b))
        });
        let entry = order[0];

        // 3. Initialise data structures.
        let mut layer0: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
        let mut upper: Vec<HashMap<u32, Vec<u32>>> =
            (0..top_layer).map(|_| HashMap::new()).collect();

        // 4. Insert nodes one at a time.
        let mut inserted = HashSet::new();
        for &x in &order {
            let lx = levels[x as usize];

            if inserted.is_empty() {
                // First (highest-level) node — entry point. No
                // neighbours yet.
                for l in 1..=lx {
                    upper[(l - 1) as usize].insert(x, Vec::new());
                }
                inserted.insert(x);
                continue;
            }

            // 4a. Greedy descent at ef=1 from top_layer down to lx+1.
            let mut current = entry;
            let mut cur_dist = cosine_distance(&embeddings[x as usize], &embeddings[current as usize]);
            for l in ((lx + 1)..=top_layer).rev() {
                let (best, best_dist) = greedy_one_hop(
                    x,
                    current,
                    cur_dist,
                    l,
                    &embeddings,
                    &layer0,
                    &upper,
                );
                current = best;
                cur_dist = best_dist;
            }

            // 4b. From layer lx down to 0: ef_construction beam, then
            //     Algorithm 4 → bidirectional link.
            let mut entry_at_layer = vec![current];
            for l in (0..=lx).rev() {
                let max_m = if l == 0 {
                    params.max_neighbors_l0
                } else {
                    params.max_neighbors_upper
                };
                let candidates = beam_search_layer(
                    x,
                    &entry_at_layer,
                    l,
                    params.ef_construction,
                    &embeddings,
                    &layer0,
                    &upper,
                    &inserted,
                );
                let selected = match params.neighbour_heuristic {
                    NeighbourHeuristic::Algorithm4 => {
                        algorithm4_select(x, &candidates, max_m, &embeddings)
                    }
                    NeighbourHeuristic::Simple => {
                        let mut c = candidates.clone();
                        c.sort_by(|a, b| {
                            a.dist
                                .partial_cmp(&b.dist)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        c.into_iter().take(max_m).map(|c| c.id).collect()
                    }
                };

                // Write x's neighbours at layer l.
                set_neighbours(x, l, selected.clone(), &mut layer0, &mut upper);

                // Bidirectionally link: for each selected `s`, add `x`
                // to s's neighbours; if s now exceeds max_m, prune s
                // using the same heuristic.
                for s in &selected {
                    let mut s_n = neighbours_of(*s, l, &layer0, &upper).to_vec();
                    s_n.push(x);
                    let s_max = if l == 0 {
                        params.max_neighbors_l0
                    } else {
                        params.max_neighbors_upper
                    };
                    if s_n.len() > s_max {
                        // Re-select s's neighbours from the union.
                        let cand: Vec<NodeDist> = s_n
                            .iter()
                            .map(|&id| NodeDist {
                                id,
                                dist: cosine_distance(
                                    &embeddings[*s as usize],
                                    &embeddings[id as usize],
                                ),
                            })
                            .collect();
                        let pruned = match params.neighbour_heuristic {
                            NeighbourHeuristic::Algorithm4 => {
                                algorithm4_select(*s, &cand, s_max, &embeddings)
                            }
                            NeighbourHeuristic::Simple => {
                                let mut c = cand;
                                c.sort_by(|a, b| {
                                    a.dist
                                        .partial_cmp(&b.dist)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                c.into_iter().take(s_max).map(|c| c.id).collect()
                            }
                        };
                        set_neighbours(*s, l, pruned, &mut layer0, &mut upper);
                    } else {
                        set_neighbours(*s, l, s_n, &mut layer0, &mut upper);
                    }
                }

                entry_at_layer = selected;
                if entry_at_layer.is_empty() {
                    entry_at_layer.push(current);
                }
            }

            inserted.insert(x);
        }

        Self {
            params,
            embeddings,
            levels,
            layer0,
            upper,
            entry,
            top_layer,
        }
    }

    /// Materialise the bottom layer as `NodeBlock`s — one block per
    /// node, in id order, neighbour list truncated to
    /// `max_neighbors_l0`. Called by `CompassIndex::from_plaintext_corpus`
    /// when admitting blocks into the Ring-ORAM.
    pub fn layer0_blocks(&self) -> Vec<NodeBlock> {
        self.embeddings
            .iter()
            .zip(self.layer0.iter())
            .map(|(emb, nb)| NodeBlock {
                embedding: emb.clone(),
                neighbors: nb.clone(),
            })
            .collect()
    }

    /// Compute the directional hints for every layer-0 node. The
    /// hint for `(x, i)` is the quantised unit vector from `x` toward
    /// its `i`-th neighbour. Called once at index build time;
    /// `CompassIndex` then holds the result cleartext.
    pub(crate) fn layer0_directional_hints(&self) -> Vec<NodeHints> {
        self.embeddings
            .iter()
            .enumerate()
            .map(|(x_id, x_emb)| {
                let neighbours = &self.layer0[x_id];
                let packed: Vec<Vec<u8>> = neighbours
                    .iter()
                    .map(|&n_id| {
                        let n_emb = &self.embeddings[n_id as usize];
                        let dir: Vec<f32> = n_emb
                            .iter()
                            .zip(x_emb.iter())
                            .map(|(n, x)| n - x)
                            .collect();
                        pack_direction(&dir)
                    })
                    .collect();
                NodeHints { packed_hints: packed }
            })
            .collect()
    }
}

// ─── internals ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct NodeDist {
    id: u32,
    dist: f32,
}

fn neighbours_of<'a>(
    id: u32,
    layer: u32,
    layer0: &'a [Vec<u32>],
    upper: &'a [HashMap<u32, Vec<u32>>],
) -> &'a [u32] {
    if layer == 0 {
        &layer0[id as usize]
    } else {
        upper[(layer - 1) as usize]
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

fn set_neighbours(
    id: u32,
    layer: u32,
    neighbours: Vec<u32>,
    layer0: &mut Vec<Vec<u32>>,
    upper: &mut [HashMap<u32, Vec<u32>>],
) {
    if layer == 0 {
        layer0[id as usize] = neighbours;
    } else {
        upper[(layer - 1) as usize].insert(id, neighbours);
    }
}

fn greedy_one_hop(
    target: u32,
    start: u32,
    start_dist: f32,
    layer: u32,
    embeddings: &[Vec<f32>],
    layer0: &[Vec<u32>],
    upper: &[HashMap<u32, Vec<u32>>],
) -> (u32, f32) {
    let target_emb = &embeddings[target as usize];
    let mut best = start;
    let mut best_dist = start_dist;
    loop {
        let nbrs = neighbours_of(best, layer, layer0, upper).to_vec();
        let mut improved = false;
        for n in nbrs {
            let d = cosine_distance(target_emb, &embeddings[n as usize]);
            if d < best_dist {
                best = n;
                best_dist = d;
                improved = true;
            }
        }
        if !improved {
            return (best, best_dist);
        }
    }
}

fn beam_search_layer(
    target: u32,
    entry_points: &[u32],
    layer: u32,
    ef: usize,
    embeddings: &[Vec<f32>],
    layer0: &[Vec<u32>],
    upper: &[HashMap<u32, Vec<u32>>],
    inserted: &HashSet<u32>,
) -> Vec<NodeDist> {
    let target_emb = &embeddings[target as usize];
    // Candidate min-heap (by dist), result max-heap.
    let mut visited: HashSet<u32> = HashSet::new();
    let mut candidates: BinaryHeap<std::cmp::Reverse<OrderedNode>> = BinaryHeap::new();
    let mut top: BinaryHeap<OrderedNode> = BinaryHeap::new();

    for &e in entry_points {
        if !inserted.contains(&e) {
            continue;
        }
        if !visited.insert(e) {
            continue;
        }
        let d = cosine_distance(target_emb, &embeddings[e as usize]);
        let node = OrderedNode { id: e, dist: d };
        candidates.push(std::cmp::Reverse(node));
        top.push(node);
    }

    while let Some(std::cmp::Reverse(c)) = candidates.pop() {
        if top.len() >= ef {
            let farthest = top.peek().expect("non-empty top").dist;
            if c.dist > farthest {
                break;
            }
        }
        let nbrs = neighbours_of(c.id, layer, layer0, upper).to_vec();
        for n in nbrs {
            if !inserted.contains(&n) {
                continue;
            }
            if !visited.insert(n) {
                continue;
            }
            let d = cosine_distance(target_emb, &embeddings[n as usize]);
            let node = OrderedNode { id: n, dist: d };
            if top.len() < ef {
                top.push(node);
                candidates.push(std::cmp::Reverse(node));
            } else {
                let farthest = top.peek().expect("non-empty top").dist;
                if d < farthest {
                    top.pop();
                    top.push(node);
                    candidates.push(std::cmp::Reverse(node));
                }
            }
        }
    }

    let mut out: Vec<NodeDist> = top
        .into_iter()
        .map(|on| NodeDist { id: on.id, dist: on.dist })
        .collect();
    out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Malkov-Yashunin Algorithm 4. Without extendCandidates and without
/// keepPrunedConnections — those are listed as optional in the paper
/// and `hnsw_rs` defaults to "off."
fn algorithm4_select(
    q: u32,
    candidates: &[NodeDist],
    m: usize,
    embeddings: &[Vec<f32>],
) -> Vec<u32> {
    let q_emb = &embeddings[q as usize];
    // Walk candidates closest-first.
    let mut sorted: Vec<NodeDist> = candidates.to_vec();
    sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    let mut selected: Vec<u32> = Vec::with_capacity(m);
    for cand in sorted {
        if cand.id == q {
            continue;
        }
        if selected.len() >= m {
            break;
        }
        // Include `cand` only if `cand` is closer to `q` than to any
        // already-selected neighbour. (The paper writes "closer to q
        // than to any element of R"; equivalent.)
        let cand_emb = &embeddings[cand.id as usize];
        let mut diverse = true;
        for &s in &selected {
            let s_emb = &embeddings[s as usize];
            let d_cs = cosine_distance(cand_emb, s_emb);
            if d_cs < cand.dist {
                diverse = false;
                break;
            }
        }
        // `cand.dist` here is `dist(cand, q)`, computed by caller.
        let _ = q_emb;
        if diverse {
            selected.push(cand.id);
        }
    }
    selected
}

#[derive(Debug, Clone, Copy)]
struct OrderedNode {
    id: u32,
    dist: f32,
}

impl PartialEq for OrderedNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for OrderedNode {}
impl Ord for OrderedNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}
impl PartialOrd for OrderedNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 1 - cosine_similarity. Cleartext — used during both build and
/// search (over decrypted blocks inside the CVM).
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

    #[test]
    fn small_build_produces_connected_layer0() {
        let embs: Vec<Vec<f32>> = (0..16)
            .map(|i| {
                let theta = (i as f32) * 0.4;
                vec![theta.cos(), theta.sin()]
            })
            .collect();
        let p = PlainHnswParams::paper_defaults(2, 4);
        let h = PlainHnsw::build(embs, p);
        // Every node should have at least one neighbour at layer 0
        // (no disconnected nodes).
        for (i, nb) in h.layer0.iter().enumerate() {
            assert!(!nb.is_empty(), "node {i} has no neighbours at layer 0");
        }
        // top_layer is u32; just confirm we built *something*.
        assert!(!h.layer0.is_empty());
    }

    #[test]
    fn entry_point_has_max_level() {
        // Pin the build seed deterministically.
        let mut p = PlainHnswParams::paper_defaults(4, 4);
        p.build_seed = [0xab; 32];
        let embs: Vec<Vec<f32>> = (0..32).map(|i| vec![i as f32, 0.0, 0.0, 1.0]).collect();
        let h = PlainHnsw::build(embs, p);
        // Entry's level must equal top_layer (paper invariant).
        assert_eq!(h.levels[h.entry as usize], h.top_layer);
    }

    #[test]
    fn algorithm4_diverse_set_picks_spread_neighbours() {
        // Three candidates: one close, two far but in opposite
        // directions. Algorithm 4 should keep the close one + the
        // most-orthogonal far one, not the two close-together ones.
        let embs = vec![
            vec![1.0, 0.0],      // q
            vec![0.9, 0.0],      // c0 — very close to q
            vec![-0.9, 0.0],     // c1 — far, antipodal to q (but close to c0's mirror)
            vec![0.0, -1.0],     // c2 — orthogonal to q
        ];
        let cands = vec![
            NodeDist { id: 1, dist: cosine_distance(&embs[0], &embs[1]) },
            NodeDist { id: 2, dist: cosine_distance(&embs[0], &embs[2]) },
            NodeDist { id: 3, dist: cosine_distance(&embs[0], &embs[3]) },
        ];
        let selected = algorithm4_select(0, &cands, 2, &embs);
        assert_eq!(selected.len(), 2);
        // The closest (id=1) must always be selected.
        assert!(selected.contains(&1));
    }
}
