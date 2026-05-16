//! `CompassIndex` — the encrypted graph-ANN index. Wraps a
//! [`RingOramClient`] holding HNSW node blocks.

use ring_oram::{
    BlockBackend, BlockId, InMemoryBlockBackend, OramError, RingOramClient, RingOramParams,
};
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

/// Per-index configuration: combines the HNSW parameters (D, M) and
/// the Ring-ORAM parameters (Z, S, A, block_bytes, n_leaves). These
/// must be self-consistent — `block_bytes` must accommodate a
/// serialised node of (D, M) plus zero padding.
#[derive(Debug, Clone, Copy)]
pub struct CompassIndexParams {
    pub hnsw: PlainHnswParams,
    pub oram: RingOramParams,
    /// Greedy-search candidate-list width. Affects recall and the
    /// number of ORAM reads per query.
    pub ef_search: usize,
}

impl Default for CompassIndexParams {
    /// M3 defaults: 128-dim embeddings, 16-degree HNSW, ef=64. Tree
    /// sized for a ~1K-vector test fixture.
    fn default() -> Self {
        Self {
            hnsw: PlainHnswParams { dim: 128, max_neighbors: 16 },
            oram: RingOramParams {
                z: 4,
                s: 5,
                a: 3,
                block_bytes: 1024, // 128·4 + 4 + 16·4 = 580; pad to 1024
                n_leaves: 2048, // ≥ 1K real blocks + headroom
            },
            ef_search: 64,
        }
    }
}

pub struct CompassIndex<B: BlockBackend> {
    pub(crate) oram: RingOramClient<B>,
    pub(crate) params: CompassIndexParams,
    pub(crate) entry: BlockId,
    /// Embedding-of-the-entry-node cached cleartext inside the CVM.
    /// HNSW's greedy search needs the entry distance to bootstrap; we
    /// cache it to avoid an ORAM read in the very first hop. Tiny —
    /// `dim · 4` bytes.
    pub(crate) entry_embedding: Vec<f32>,
}

impl CompassIndex<InMemoryBlockBackend> {
    /// `compass_init` analog using the in-memory backend. Builds a
    /// plaintext HNSW from `embeddings`, encodes every node as one
    /// ORAM block, admits all blocks. Returns the ready-to-query
    /// index. Consumes the plaintext build state.
    pub fn from_plaintext_corpus(
        embeddings: Vec<Vec<f32>>,
        params: CompassIndexParams,
    ) -> Result<Self, CompassIndexError> {
        let hnsw = PlainHnsw::build(embeddings, params.hnsw);
        let backend = InMemoryBlockBackend::new(params.oram.num_buckets());
        let key = [0x33u8; 32];
        let rng_seed = [0x55u8; 32];
        let mut oram = RingOramClient::new(backend, params.oram, key, rng_seed);

        let entry = hnsw.entry;
        let entry_embedding = hnsw.nodes[entry as usize].embedding.clone();

        for (id, node) in hnsw.nodes.iter().enumerate() {
            let bytes = serialise_node(
                node,
                params.hnsw.dim,
                params.hnsw.max_neighbors,
                params.oram.block_bytes as usize,
            )?;
            oram.admit(BlockId(id as u32), bytes)?;
        }

        Ok(Self {
            oram,
            params,
            entry: BlockId(entry),
            entry_embedding,
        })
    }
}

impl<B: BlockBackend> CompassIndex<B> {
    /// Search interface: returns the top-k node ids ordered by
    /// ascending cosine distance to the query.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<u32>, CompassIndexError> {
        crate::search::strawman_search(self, query, k)
    }

    /// Internal accessor: decode block `id` from the ORAM.
    pub(crate) fn read_node(&mut self, id: BlockId) -> Result<crate::codec::NodeBlock, CompassIndexError> {
        let bytes = self.oram.read(id)?;
        let node = deserialise_node(&bytes, self.params.hnsw.dim, self.params.hnsw.max_neighbors)?;
        Ok(node)
    }
}
