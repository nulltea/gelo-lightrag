//! `CompassIndex` — the encrypted graph-ANN index. Wraps a
//! [`RingOramClient`] holding HNSW layer-0 node blocks; layers ≥ 1
//! live cleartext inside the CVM.
//!
//! Async since M5 because the underlying `BlockBackend` is async — a
//! real (REST-shaped) backend will block on network I/O. The in-memory
//! backend used by tests stays effectively sync (its async methods
//! return ready futures).

use ring_oram::{
    BlockBackend, BlockId, InMemoryBlockBackend, OramError, RingOramClient, RingOramParams,
};
use std::collections::HashMap;
use thiserror::Error;

use crate::codec::{CodecError, deserialise_node, serialise_node};
use crate::hints::NodeHints;
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
    /// Directional Neighbor Filtering budget (Compass §4.5). For each
    /// visited layer-0 node with M neighbours, only fetch the top
    /// `ef_n` neighbours' full blocks via ORAM. Set to `usize::MAX`
    /// to disable filtering (M3 strawman behaviour). Paper default:
    /// 4 — see `CompassParams::hnsw_ef_n`.
    pub ef_n: usize,
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
                // Cache top 4 levels of the ORAM tree cleartext in
                // CVM RAM (Compass §4.7). 2^4 - 1 = 15 buckets.
                treetop_levels: 4,
            },
            ef_search: 64,
            ef_n: 4,
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
    /// Directional hints for layer-0 nodes (Compass §4.5).
    /// `layer0_hints[node_id]` carries one packed direction per
    /// neighbour, in the same order as the neighbour list. Cached
    /// cleartext in CVM RAM.
    pub(crate) layer0_hints: Vec<NodeHints>,
    /// Telemetry: count of ORAM block reads served via
    /// `read_layer0_node`. Used by tests and benches to measure the
    /// effect of directional filtering. Wraps after `2^64` reads.
    pub(crate) layer0_read_count: u64,
}

impl CompassIndex<InMemoryBlockBackend> {
    /// `compass_init` analog using the in-memory backend. Builds a
    /// layered HNSW from `embeddings`, caches upper layers cleartext,
    /// encodes layer-0 nodes as ORAM blocks, admits them.
    pub async fn from_plaintext_corpus(
        embeddings: Vec<Vec<f32>>,
        params: CompassIndexParams,
    ) -> Result<Self, CompassIndexError> {
        let backend = InMemoryBlockBackend::new(params.oram.num_buckets());
        Self::from_plaintext_corpus_on(embeddings, params, backend).await
    }
}

impl<B: BlockBackend> CompassIndex<B> {
    /// Build an index over the supplied backend. Used by the in-memory
    /// constructor above and (M5.2) by the REST-backed integration test.
    pub async fn from_plaintext_corpus_on(
        embeddings: Vec<Vec<f32>>,
        params: CompassIndexParams,
        backend: B,
    ) -> Result<Self, CompassIndexError> {
        let hnsw = PlainHnsw::build(embeddings, params.hnsw);
        // Use the HNSW build seed to derive the ORAM key + RNG seed
        // deterministically. Production tenants will swap these for
        // the V2 HKDF children (oram_entities_key etc.).
        let key = derive_test_key(&hnsw.params.build_seed, b"oram-key");
        let rng_seed = derive_test_key(&hnsw.params.build_seed, b"oram-rng");
        let mut oram = RingOramClient::new(backend, params.oram, key, rng_seed).await?;

        let entry = hnsw.entry;
        let top_layer = hnsw.top_layer;

        // Cache upper layers + directional hints cleartext.
        let upper_layers = build_upper_layer_cache(&hnsw);
        let layer0_hints = hnsw.layer0_directional_hints();

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
            oram.admit(BlockId(id as u32), bytes).await?;
        }

        Ok(Self {
            oram,
            params,
            entry: BlockId(entry),
            top_layer,
            upper_layers,
            layer0_hints,
            layer0_read_count: 0,
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
    ///
    /// Async because layer-0 traversal issues one ORAM round-trip per
    /// visited node.
    pub async fn search(
        &mut self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<u32>, CompassIndexError> {
        crate::search::layered_search(self, query, k).await
    }

    /// Internal accessor: decode block `id` from the ORAM. Bumps the
    /// telemetry counter used by tests to measure the directional
    /// filter's effect.
    pub(crate) async fn read_layer0_node(
        &mut self,
        id: BlockId,
    ) -> Result<crate::codec::NodeBlock, CompassIndexError> {
        let bytes = self.oram.read(id).await?;
        let node = deserialise_node(
            &bytes,
            self.params.hnsw.dim,
            self.params.hnsw.max_neighbors_l0,
        )?;
        self.layer0_read_count += 1;
        Ok(node)
    }

    /// Telemetry: total layer-0 ORAM block reads since construction.
    /// Used by the directional-filter effectiveness test.
    pub fn layer0_read_count(&self) -> u64 {
        self.layer0_read_count
    }

    /// Internal accessor: hints for layer-0 node `id`. Read-only
    /// view; the directional filter consumes this during beam search.
    pub(crate) fn layer0_node_hints(&self, id: u32) -> &NodeHints {
        &self.layer0_hints[id as usize]
    }

    /// Mutable handle on the underlying ORAM client. Used by
    /// `search.rs` to drive the deferred-eviction toggle and flush
    /// the queue at the end of a search.
    pub(crate) fn oram_mut(&mut self) -> &mut RingOramClient<B> {
        &mut self.oram
    }
}
