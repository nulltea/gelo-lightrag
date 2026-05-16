//! Storage-side trait: `BlockBackend` is what the (untrusted) storage
//! server implements. The client only ever sees encrypted buckets.
//!
//! M5 promotes this trait to async (`#[async_trait]`) so a single
//! `RingOramClient` can run against either the in-memory backend used
//! by unit/integration tests or the REST-shaped `compass-rest-backend`
//! that ships in M5.1. The trait surface is intentionally small —
//! three methods — so swapping is a one-file change at the call site.

use async_trait::async_trait;

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

/// Error type returned by backend operations. `anyhow::Error` so each
/// implementation can wrap whatever transport-specific error type it
/// uses (sled, reqwest, …) without needing a shared error enum.
pub type BackendError = anyhow::Error;

/// What the server-side backend must implement. Async since M5 so the
/// REST-backed implementation can stream over the network without
/// blocking; the in-memory backend simply returns ready futures.
#[async_trait]
pub trait BlockBackend: Send + Sync {
    /// Fetch one path's worth of buckets, in root-first order. Caller
    /// passes the result of [`crate::path::path_buckets`].
    async fn read_path(&self, bucket_ids: &[u32]) -> Result<Vec<EncryptedBucket>, BackendError>;

    /// Overwrite a contiguous batch of buckets. The implementation
    /// updates each bucket's `write_counter` atomically — the client
    /// passes the *new* counter values; server stores them.
    async fn write_buckets(&mut self, buckets: &[EncryptedBucket]) -> Result<(), BackendError>;

    /// Number of buckets in the tree. The client uses this only to
    /// sanity-check the configured `n_leaves`. Sync by design — a
    /// single u32 the server holds in memory.
    fn num_buckets(&self) -> u32;
}

// ─── In-memory backend ────────────────────────────────────────────

/// Trivial backend storing ciphertext in a `Vec` of `EncryptedBucket`.
/// Used by every M1 test and by the M3/M4 `compass-index` integration
/// tests. The async methods are effectively sync — they wrap the
/// in-memory operations in a ready future.
#[derive(Debug)]
pub struct InMemoryBlockBackend {
    buckets: Vec<EncryptedBucket>,
    read_count: std::sync::atomic::AtomicU64,
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
        Self {
            buckets,
            read_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Borrow the raw vector for assertions in tests. Production code
    /// must not reach past `BlockBackend`.
    #[cfg(test)]
    pub fn raw(&self) -> &[EncryptedBucket] {
        &self.buckets
    }

    /// Total individual-bucket reads served via `read_path`. Used by
    /// the treetop-caching test to confirm cached reads bypass the
    /// backend.
    pub fn read_count(&self) -> u64 {
        self.read_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl BlockBackend for InMemoryBlockBackend {
    async fn read_path(&self, bucket_ids: &[u32]) -> Result<Vec<EncryptedBucket>, BackendError> {
        self.read_count
            .fetch_add(bucket_ids.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bucket_ids
            .iter()
            .map(|&i| self.buckets[i as usize].clone())
            .collect())
    }

    async fn write_buckets(&mut self, buckets: &[EncryptedBucket]) -> Result<(), BackendError> {
        for b in buckets {
            self.buckets[b.bucket_id as usize] = b.clone();
        }
        Ok(())
    }

    fn num_buckets(&self) -> u32 {
        self.buckets.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_backend_round_trips_writes() {
        let mut be = InMemoryBlockBackend::new(4);
        assert_eq!(be.num_buckets(), 4);
        let bucket = EncryptedBucket {
            bucket_id: 2,
            write_counter: 7,
            ciphertext: vec![0xde, 0xad, 0xbe, 0xef],
        };
        be.write_buckets(&[bucket.clone()]).await.unwrap();
        let got = be.read_path(&[2]).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].bucket_id, 2);
        assert_eq!(got[0].write_counter, 7);
        assert_eq!(got[0].ciphertext, vec![0xde, 0xad, 0xbe, 0xef]);
    }
}
