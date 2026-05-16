//! `CompassIndex` — the encrypted graph-ANN index. Wraps a
//! [`RingOramClient`] holding HNSW layer-0 node blocks; layers ≥ 1
//! live cleartext inside the CVM.

use ring_oram::{
    BlockBackend, BlockId, InMemoryBlockBackend, OramError, RingOramClient, RingOramParams,
};
use std::collections::HashMap;
use thiserror::Error;

use crate::codec::{CodecError, deserialise_node, serialise_node};
use crate::hnsw_plain::{PlainHnsw, PlainHnswParams};

#[derive(Debug, Error)]
pub enum CompassIndexError {
    #[error("ORAM error: {0}")]
    Oram(#[from] OramError),
    #[error("codec error: {0}")]
    Codec(#[from] CodecError),
    #[error("entry node missing from posmap (build error)")]
    MissingEntry,
}

/// Per-index configuration: combines the HNSW parameters and the
/// Ring-ORAM parameters. They must be self-consistent — `block_bytes`
/// must accommodate a serialised layer-0 node (D-vector + neighbour
/// list of length `max_neighbors_l0`) plus zero padding.
#[derive(Debug, Clone, Copy)]
pub struct CompassIndexParams {
    pub hnsw: PlainHnswParams,
    pub oram: RingOramParams,
    /// Greedy-search candidate-list width at layer 0. Affects recall
    /// and the number of ORAM reads per query.
    pub ef_search: usize,
}

impl Default for CompassIndexParams {
    /// M4 defaults: 128-dim embeddings, M=16 (so M_l0=32). Tree sized
    /// for a ~1K-vector test fixture.
    fn default() -> Self {
        Self {
            hnsw: PlainHnswParams::paper_defaults(128, 16),
            oram: RingOramParams {
                z: 4,
                s: 5,
                a: 3,
                // 128·4 + 4 + 32·4 = 644 — pad to 1024.
                block_bytes: 1024,
                n_leaves: 2048,
            },
            ef_search: 64,
        }
    }
}

/// Cleartext-cached upper layer: per-node `(embedding, neighbours)`.
/// Stored inside the CVM; never reaches the storage server.
#[derive(Debug)]
pub(crate) struct UpperLayer {
    pub(crate) nodes: HashMap<u32, UpperNode>,
}

#[derive(Debug)]
pub(crate) struct UpperNode {
    pub(crate) embedding: Vec<f32>,
    pub(crate) neighbours: Vec<u32>,
}

pub struct CompassIndex<B: BlockBackend> {
    pub(crate) oram: RingOramClient<B>,
    pub(crate) params: CompassIndexParams,
    pub(crate) entry: BlockId,
    pub(crate) top_layer: u32,
    /// Cleartext upper-layer cache. `upper_layers[l - 1]` is the
    /// adjacency + embeddings at layer `l`. Bottom layer (0) is in
    /// ORAM and not present here.
    pub(crate) upper_layers: Vec<UpperLayer>,
}

impl CompassIndex<InMemoryBlockBackend> {
    /// `compass_init` analog using the in-memory backend. Builds a
    /// layered HNSW from `embeddings`, caches upper layers cleartext,
    /// encodes layer-0 nodes as ORAM blocks, admits them.
    pub fn from_plaintext_corpus(
        embeddings: Vec<Vec<f32>>,
        params: CompassIndexParams,
    ) -> Result<Self, CompassIndexError> {
        let hnsw = PlainHnsw::build(embeddings, params.hnsw);
        let backend = InMemoryBlockBackend::new(params.oram.num_buckets());
        // Use the HNSW build seed to derive the ORAM key + RNG seed
        // deterministically. Production tenants will swap these for
        // the V2 HKDF children (oram_entities_key etc.).
        let key = derive_test_key(&hnsw.params.build_seed, b"oram-key");
        let rng_seed = derive_test_key(&hnsw.params.build_seed, b"oram-rng");
        let mut oram = RingOramClient::new(backend, params.oram, key, rng_seed);

        let entry = hnsw.entry;
        let top_layer = hnsw.top_layer;

        // Cache upper layers cleartext.
        let upper_layers = build_upper_layer_cache(&hnsw);

        // Push every layer-0 node into ORAM as a fixed-size block.
        let blocks = hnsw.layer0_blocks();
        for (id, node) in blocks.iter().enumerate() {
            // Truncate neighbour list to fit the ORAM block. The
            // CompassParams encoding caps at max_neighbors_l0.
            let mut node = node.clone();
            node.neighbors.truncate(params.hnsw.max_neighbors_l0);
            let bytes = serialise_node(
                &node,
                params.hnsw.dim,
                params.hnsw.max_neighbors_l0,
                params.oram.block_bytes as usize,
            )?;
            oram.admit(BlockId(id as u32), bytes)?;
        }

        Ok(Self {
            oram,
            params,
            entry: BlockId(entry),
            top_layer,
            upper_layers,
        })
    }
}

fn build_upper_layer_cache(hnsw: &PlainHnsw) -> Vec<UpperLayer> {
    hnsw.upper
        .iter()
        .map(|adj| {
            let nodes: HashMap<u32, UpperNode> = adj
                .iter()
                .map(|(&id, neighbours)| {
                    (
                        id,
                        UpperNode {
                            embedding: hnsw.embeddings[id as usize].clone(),
                            neighbours: neighbours.clone(),
                        },
                    )
                })
                .collect();
            UpperLayer { nodes }
        })
        .collect()
}

/// Trivial key derivation for tests/integration — replaced by the
/// V2 HKDF children at the `light-kg-store` boundary (M6).
fn derive_test_key(seed: &[u8; 32], label: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(seed);
    hasher.update(label);
    hasher.finalize().into()
}

impl<B: BlockBackend> CompassIndex<B> {
    /// Top-k search. Layered routing through cleartext upper layers,
    /// then ef-bounded beam search in the ORAM-resident layer 0.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<u32>, CompassIndexError> {
        crate::search::layered_search(self, query, k)
    }

    /// Internal accessor: decode block `id` from the ORAM.
    pub(crate) fn read_layer0_node(
        &mut self,
        id: BlockId,
    ) -> Result<crate::codec::NodeBlock, CompassIndexError> {
        let bytes = self.oram.read(id)?;
        let node = deserialise_node(
            &bytes,
            self.params.hnsw.dim,
            self.params.hnsw.max_neighbors_l0,
        )?;
        Ok(node)
    }

    /// Multi-hop lazy-eviction guard (Compass §4.7). RAII: enables
    /// defer on construction, flushes on drop. Use during a search
    /// to keep eviction off the user-perceived critical path.
    pub(crate) fn defer_evictions_for_search<'a>(&'a mut self) -> EvictionDeferGuard<'a, B> {
        self.oram.set_defer_evictions(true);
        EvictionDeferGuard { index: self }
    }
}

/// RAII guard returned by `defer_evictions_for_search`. On drop,
/// resumes inline eviction and flushes anything pending.
pub(crate) struct EvictionDeferGuard<'a, B: BlockBackend> {
    pub(crate) index: &'a mut CompassIndex<B>,
}

impl<'a, B: BlockBackend> Drop for EvictionDeferGuard<'a, B> {
    fn drop(&mut self) {
        // set_defer_evictions(false) also flushes any pending.
        self.index.oram.set_defer_evictions(false);
    }
}
