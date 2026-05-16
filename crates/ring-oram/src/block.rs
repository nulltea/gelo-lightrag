//! Logical block — a fixed-size payload tagged with a stable
//! `BlockId`. The Ring-ORAM tree contains exactly `n_real` real blocks
//! plus dummies; clients address blocks by `BlockId`, the position map
//! translates to `PathId`.

use zeroize::Zeroize;

/// Stable logical identifier. In the Compass setting one HNSW node ⇔
/// one `BlockId`. Dummy blocks reuse [`BlockId::DUMMY`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Sentinel for dummy blocks. Real blocks use `0..n_real`.
    pub const DUMMY: BlockId = BlockId(u32::MAX);

    pub fn is_dummy(&self) -> bool {
        *self == Self::DUMMY
    }
}

/// Block payload — opaque bytes the higher layer (compass-index) maps
/// to whatever it stores (HNSW node = `(embedding, neighbor_list)`).
/// Always exactly `RingOramParams::block_bytes` long once placed in the
/// tree; constructors pad with zeros.
#[derive(Clone)]
pub struct BlockPayload(Box<[u8]>);

impl BlockPayload {
    /// Wrap an exact-size byte buffer. Panics if `bytes.len() != size`.
    pub fn from_exact(bytes: Vec<u8>, size: usize) -> Self {
        assert_eq!(
            bytes.len(),
            size,
            "BlockPayload::from_exact size mismatch: got {}, expected {}",
            bytes.len(),
            size
        );
        Self(bytes.into_boxed_slice())
    }

    /// Build from a possibly-shorter byte buffer, zero-padded to `size`.
    /// Panics if `bytes.len() > size`.
    pub fn padded(mut bytes: Vec<u8>, size: usize) -> Self {
        assert!(
            bytes.len() <= size,
            "BlockPayload::padded overflow: {} > {}",
            bytes.len(),
            size
        );
        bytes.resize(size, 0);
        Self(bytes.into_boxed_slice())
    }

    /// All-zero payload of length `size`.
    pub fn zero(size: usize) -> Self {
        Self(vec![0u8; size].into_boxed_slice())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for BlockPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't dump bytes — keep tracing free of payload content.
        write!(f, "BlockPayload({} bytes)", self.0.len())
    }
}

impl Drop for BlockPayload {
    fn drop(&mut self) {
        // Zero on drop. Block payloads carry plaintext between the
        // ORAM controller and the higher layers; defense in depth even
        // though the buffer lives inside the CVM.
        self.0.zeroize();
    }
}

/// A real or dummy block held in the stash or in a bucket slot.
#[derive(Clone, Debug)]
pub struct Block {
    pub id: BlockId,
    pub payload: BlockPayload,
}

impl Block {
    pub fn new(id: BlockId, payload: BlockPayload) -> Self {
        Self { id, payload }
    }

    /// Dummy block of size `block_bytes`. Bucket slots not holding a
    /// real block are filled with these so the on-disk layout is
    /// uniform.
    pub fn dummy(block_bytes: usize) -> Self {
        Self {
            id: BlockId::DUMMY,
            payload: BlockPayload::zero(block_bytes),
        }
    }

    pub fn is_dummy(&self) -> bool {
        self.id.is_dummy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_block_has_dummy_id() {
        let d = Block::dummy(64);
        assert!(d.is_dummy());
        assert_eq!(d.payload.len(), 64);
    }

    #[test]
    fn padded_payload_zero_extends() {
        let p = BlockPayload::padded(vec![0xab, 0xcd], 8);
        assert_eq!(p.len(), 8);
        assert_eq!(p.as_bytes(), &[0xab, 0xcd, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "BlockPayload::from_exact size mismatch")]
    fn from_exact_panics_on_size_mismatch() {
        let _ = BlockPayload::from_exact(vec![1, 2, 3], 8);
    }
}
