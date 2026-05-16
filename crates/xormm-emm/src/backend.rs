//! Storage-side trait for XorMM: the untrusted byte store the server
//! actually runs. Reads/writes opaque AES-GCM frames keyed by bucket
//! index.

/// One encrypted bucket. AES-GCM frame; the client owns the key.
/// `nonce` is 12 bytes derived from `(bucket_id, generation)` — the
/// generation increments each rebuild so re-uploading the same EMM
/// after a fresh-tier flush never reuses an AES-GCM nonce.
#[derive(Clone, Debug)]
pub struct EncryptedBucket {
    pub bucket_id: u32,
    pub generation: u32,
    pub ciphertext: Vec<u8>,
}

pub trait ByteStoreBackend {
    /// Fetch one or more buckets in a batch. Order of the response
    /// matches the request.
    fn read_buckets(&self, bucket_ids: &[u32]) -> Vec<EncryptedBucket>;
    /// Write a contiguous range of buckets. Used at build time.
    fn write_buckets(&mut self, buckets: &[EncryptedBucket]);
    /// Server-visible bucket count.
    fn num_buckets(&self) -> u32;
}

/// In-memory backend used by tests and by `light-kg-store`
/// integration tests at M6.
#[derive(Debug)]
pub struct InMemoryByteStore {
    buckets: Vec<EncryptedBucket>,
}

impl InMemoryByteStore {
    pub fn new(num_buckets: u32) -> Self {
        let buckets = (0..num_buckets)
            .map(|bucket_id| EncryptedBucket {
                bucket_id,
                generation: 0,
                ciphertext: Vec::new(),
            })
            .collect();
        Self { buckets }
    }
}

impl ByteStoreBackend for InMemoryByteStore {
    fn read_buckets(&self, bucket_ids: &[u32]) -> Vec<EncryptedBucket> {
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
    fn in_memory_backend_round_trips() {
        let mut be = InMemoryByteStore::new(4);
        be.write_buckets(&[EncryptedBucket {
            bucket_id: 2,
            generation: 1,
            ciphertext: vec![1, 2, 3],
        }]);
        let got = be.read_buckets(&[2, 0]);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].ciphertext, vec![1, 2, 3]);
        assert_eq!(got[1].ciphertext, Vec::<u8>::new()); // untouched
    }
}
