//! Storage-side trait: `BlockBackend` is what the (untrusted) storage
//! server implements. The client only ever sees encrypted buckets.
//!
//! M1 ships `InMemoryBlockBackend` for tests; M5 swaps in a real
//! networked implementation (REST or S3-shaped). The trait is
//! intentionally small — three methods — so swapping is a one-file
//! change.

/// One encrypted bucket as the server sees it. The client AES-GCM-
/// encrypts every bucket before write; the server stores opaque bytes.
///
/// `nonce = bucket_id u64-LE ‖ write_counter u32-LE` (12 bytes total),
/// constructed by the client. The server holds `write_counter` and
/// returns it alongside the ciphertext so the client can re-derive
/// the nonce on read. Malicious mode (M4) protects this counter via a
/// Merkle tree; semi-honest mode (M1) trusts it.
#[derive(Clone, Debug)]
pub struct EncryptedBucket {
    pub bucket_id: u32,
    pub write_counter: u32,
    /// AES-GCM ciphertext (does NOT include nonce — that's the
    /// (`bucket_id`, `write_counter`) pair). Includes the 16-byte
    /// authenticator tag at the tail.
    pub ciphertext: Vec<u8>,
}

/// What the server-side backend must implement. Methods are
/// synchronous in M1 for simplicity; M5 promotes to async over a real
/// transport.
pub trait BlockBackend {
    /// Fetch one path's worth of buckets, in root-first order. Caller
    /// passes the result of [`crate::path::path_buckets`].
    fn read_path(&self, bucket_ids: &[u32]) -> Vec<EncryptedBucket>;

    /// Overwrite a contiguous batch of buckets. The implementation
    /// updates each bucket's `write_counter` atomically — the client
    /// passes the *new* counter values; server stores them.
    fn write_buckets(&mut self, buckets: &[EncryptedBucket]);

    /// Number of buckets in the tree. The client uses this only to
    /// sanity-check the configured `n_leaves`.
    fn num_buckets(&self) -> u32;
}

// ─── In-memory backend ────────────────────────────────────────────

/// Trivial backend storing ciphertext in a `Vec` of `EncryptedBucket`.
/// Used by every M1 test and by M3's `compass-index` integration tests.
#[derive(Debug)]
pub struct InMemoryBlockBackend {
    buckets: Vec<EncryptedBucket>,
}

impl InMemoryBlockBackend {
    /// Allocate a tree with `num_buckets` empty (zero-ciphertext)
    /// buckets. The client must write each one at least once before
    /// reading; tests typically populate via the protocol's
    /// `init_tree` helper (added in M1.3).
    pub fn new(num_buckets: u32) -> Self {
        let buckets = (0..num_buckets)
            .map(|bucket_id| EncryptedBucket {
                bucket_id,
                write_counter: 0,
                ciphertext: Vec::new(),
            })
            .collect();
        Self { buckets }
    }

    /// Borrow the raw vector for assertions in tests. Production code
    /// must not reach past `BlockBackend`.
    #[cfg(test)]
    pub fn raw(&self) -> &[EncryptedBucket] {
        &self.buckets
    }
}

impl BlockBackend for InMemoryBlockBackend {
    fn read_path(&self, bucket_ids: &[u32]) -> Vec<EncryptedBucket> {
        bucket_ids
            .iter()
            .map(|&i| self.buckets[i as usize].clone())
            .collect()
    }

    fn write_buckets(&mut self, buckets: &[EncryptedBucket]) {
        for b in buckets {
            self.buckets[b.bucket_id as usize] = b.clone();
        }
    }

    fn num_buckets(&self) -> u32 {
        self.buckets.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_backend_round_trips_writes() {
        let mut be = InMemoryBlockBackend::new(4);
        assert_eq!(be.num_buckets(), 4);
        let bucket = EncryptedBucket {
            bucket_id: 2,
            write_counter: 7,
            ciphertext: vec![0xde, 0xad, 0xbe, 0xef],
        };
        be.write_buckets(&[bucket.clone()]);
        let got = be.read_path(&[2]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].bucket_id, 2);
        assert_eq!(got[0].write_counter, 7);
        assert_eq!(got[0].ciphertext, vec![0xde, 0xad, 0xbe, 0xef]);
    }
}
