//! Per-node block codec. One HNSW node packs into one Ring-ORAM
//! block.
//!
//! Layout: `[embedding D × f32-LE] ‖ [neighbor_count u32-LE] ‖
//! [neighbor_ids M × u32-LE] ‖ [zero padding to block_bytes]`.

use thiserror::Error;

/// In-memory representation of one node fetched from / about to be
/// pushed to the ORAM.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeBlock {
    pub embedding: Vec<f32>,
    /// Real neighbour ids; length ≤ `M`. The remaining `M -
    /// neighbour_count` slots in the serialised block are zero.
    pub neighbors: Vec<u32>,
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("embedding dim mismatch: got {got}, expected {expected}")]
    DimMismatch { got: usize, expected: usize },
    #[error("too many neighbours: got {got}, max {max}")]
    NeighborOverflow { got: usize, max: usize },
    #[error("block too small for layout (D={dim} M={max_neighbors} ⇒ {needed}B, got {block_bytes}B)")]
    BlockTooSmall {
        dim: usize,
        max_neighbors: usize,
        needed: usize,
        block_bytes: usize,
    },
}

pub(crate) fn serialised_size(dim: usize, max_neighbors: usize) -> usize {
    4 * dim + 4 + 4 * max_neighbors
}

/// Serialise a node into a zero-padded block of length `block_bytes`.
pub(crate) fn serialise_node(
    node: &NodeBlock,
    dim: usize,
    max_neighbors: usize,
    block_bytes: usize,
) -> Result<Vec<u8>, CodecError> {
    if node.embedding.len() != dim {
        return Err(CodecError::DimMismatch {
            got: node.embedding.len(),
            expected: dim,
        });
    }
    if node.neighbors.len() > max_neighbors {
        return Err(CodecError::NeighborOverflow {
            got: node.neighbors.len(),
            max: max_neighbors,
        });
    }
    let need = serialised_size(dim, max_neighbors);
    if block_bytes < need {
        return Err(CodecError::BlockTooSmall {
            dim,
            max_neighbors,
            needed: need,
            block_bytes,
        });
    }

    let mut out = vec![0u8; block_bytes];
    for (i, x) in node.embedding.iter().enumerate() {
        let off = i * 4;
        out[off..off + 4].copy_from_slice(&x.to_le_bytes());
    }
    let count_off = 4 * dim;
    out[count_off..count_off + 4].copy_from_slice(&(node.neighbors.len() as u32).to_le_bytes());
    let nbr_off = count_off + 4;
    for (i, n) in node.neighbors.iter().enumerate() {
        let off = nbr_off + i * 4;
        out[off..off + 4].copy_from_slice(&n.to_le_bytes());
    }
    Ok(out)
}

pub(crate) fn deserialise_node(
    bytes: &[u8],
    dim: usize,
    max_neighbors: usize,
) -> Result<NodeBlock, CodecError> {
    let need = serialised_size(dim, max_neighbors);
    if bytes.len() < need {
        return Err(CodecError::BlockTooSmall {
            dim,
            max_neighbors,
            needed: need,
            block_bytes: bytes.len(),
        });
    }
    let mut embedding = Vec::with_capacity(dim);
    for i in 0..dim {
        let off = i * 4;
        embedding.push(f32::from_le_bytes(
            bytes[off..off + 4].try_into().expect("4B"),
        ));
    }
    let count_off = 4 * dim;
    let count = u32::from_le_bytes(bytes[count_off..count_off + 4].try_into().expect("4B")) as usize;
    let count = count.min(max_neighbors);
    let nbr_off = count_off + 4;
    let mut neighbors = Vec::with_capacity(count);
    for i in 0..count {
        let off = nbr_off + i * 4;
        neighbors.push(u32::from_le_bytes(
            bytes[off..off + 4].try_into().expect("4B"),
        ));
    }
    Ok(NodeBlock { embedding, neighbors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_dense_neighbour_list() {
        let node = NodeBlock {
            embedding: (0..8).map(|i| i as f32 * 0.1).collect(),
            neighbors: vec![3, 5, 7, 11, 13],
        };
        let bytes = serialise_node(&node, 8, 6, 64).unwrap();
        assert_eq!(bytes.len(), 64);
        let back = deserialise_node(&bytes, 8, 6).unwrap();
        assert_eq!(back, node);
    }

    #[test]
    fn round_trips_partial_neighbour_list() {
        let node = NodeBlock {
            embedding: vec![1.0, 2.0, 3.0, 4.0],
            neighbors: vec![42],
        };
        let bytes = serialise_node(&node, 4, 8, 64).unwrap();
        let back = deserialise_node(&bytes, 4, 8).unwrap();
        assert_eq!(back.neighbors, vec![42]);
    }

    #[test]
    fn dim_mismatch_errors() {
        let node = NodeBlock { embedding: vec![1.0; 4], neighbors: vec![] };
        let err = serialise_node(&node, 8, 4, 64).unwrap_err();
        assert!(matches!(err, CodecError::DimMismatch { .. }));
    }

    #[test]
    fn too_small_block_errors() {
        let node = NodeBlock { embedding: vec![1.0; 4], neighbors: vec![] };
        // need 4·4 + 4 + 4·8 = 52; we give 32.
        let err = serialise_node(&node, 4, 8, 32).unwrap_err();
        assert!(matches!(err, CodecError::BlockTooSmall { .. }));
    }
}
