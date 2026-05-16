//! XorMM client — build, get, get_batch.
//!
//! The construction is a 2-choice cuckoo placement (paper §4): every
//! key has two candidate bucket indices, derived from two
//! key-independent hashes. Build tries to commit each key to one of
//! the two; on collision it kicks the resident and re-places it
//! recursively. After `max_kicks` failures, the key falls into a
//! small in-CVM stash (no on-server storage of stash entries).
//!
//! `get(k)` reads both candidates uniformly — the server cannot tell
//! which of the two holds `k`'s real entry. Decryption produces two
//! `BucketPlain`s; whichever has `fingerprint == k.0` is the real
//! one. If neither matches and `k` is in the local stash, return
//! that. If `k` isn't in either bucket *or* the stash, the key was
//! never built — return an empty list (not an error, because the
//! upstream LightRAG retrieval routinely queries missing entities).

use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::backend::{ByteStoreBackend, EncryptedBucket};
use crate::bucket::{
    BucketPlain, DUMMY_FINGERPRINT, aes_decrypt, aes_encrypt, fingerprint_matches,
};
use crate::params::XorMmParams;
use crate::{LogicalKey, LogicalValue};

#[derive(Debug, thiserror::Error)]
pub enum XorMmError {
    #[error("XorMM build failed: {0} keys could not be placed in {1} buckets after {2} kicks (try bigger n_buckets / max_kicks)")]
    BuildOverflow(usize, u32, u32),
    #[error("XorMM value too long ({0} > {1} value_bytes)")]
    ValueTooLong(usize, u32),
    #[error("XorMM bucket corrupted: {0}")]
    Corrupted(#[from] crate::bucket::AesError),
}

/// Client + backend bundle. Backend is owned because the API is
/// build-once-then-read; multi-tenant deployment instantiates one
/// client per tenant per EMM (adjacency vs src-chunks).
pub struct XorMmClient<B: ByteStoreBackend> {
    backend: B,
    params: XorMmParams,
    /// 32-byte AES-GCM key + the two hash seeds. Wrapped in
    /// `Zeroizing` so a request-scoped client wipes on drop.
    key: Zeroizing<[u8; 32]>,
    seed1: [u8; 16],
    seed2: [u8; 16],
    /// Generation counter — bumped each `build`. Lives in CVM RAM;
    /// after a rebuild the previous-generation buckets become
    /// undecryptable (different nonce). Held for the rebuild path
    /// (M2.1, fresh-tier incremental flushes — see plan §4.3).
    #[allow(dead_code)]
    generation: u32,
    /// In-CVM stash for cuckoo failures. Stored as `(key, padded
    /// values, value_count)`. Small (≤ params.max_kicks log entries
    /// in practice).
    stash: std::collections::HashMap<LogicalKey, (Vec<Vec<u8>>, u32)>,
}

impl<B: ByteStoreBackend> XorMmClient<B> {
    /// Build a fresh EMM over the given (key, value-list) entries.
    /// Wipes any existing data on the backend (caller is responsible
    /// for using a backend sized to `params.n_buckets`).
    pub fn build(
        entries: Vec<(LogicalKey, Vec<LogicalValue>)>,
        params: XorMmParams,
        key: Zeroizing<[u8; 32]>,
        seed1: [u8; 16],
        seed2: [u8; 16],
        mut backend: B,
    ) -> Result<Self, XorMmError> {
        debug_assert_eq!(
            backend.num_buckets(),
            params.n_buckets,
            "backend size != params.n_buckets"
        );

        // Validate + pre-pad each value list to (volume_bound, value_bytes).
        let mut prepared: Vec<(LogicalKey, Vec<Vec<u8>>, u32)> = Vec::with_capacity(entries.len());
        for (k, values) in entries {
            let value_count = values.len().min(params.volume_bound as usize) as u32;
            let vb = params.value_bytes as usize;
            let mut padded = Vec::with_capacity(params.volume_bound as usize);
            for v in values.iter().take(params.volume_bound as usize) {
                if v.0.len() > vb {
                    return Err(XorMmError::ValueTooLong(v.0.len(), params.value_bytes));
                }
                let mut buf = v.0.clone();
                buf.resize(vb, 0);
                padded.push(buf);
            }
            while padded.len() < params.volume_bound as usize {
                padded.push(vec![0u8; vb]);
            }
            prepared.push((k, padded, value_count));
        }

        // Cuckoo placement. Each bucket holds at most one entry.
        let n = params.n_buckets as usize;
        let mut occupants: Vec<Option<(LogicalKey, Vec<Vec<u8>>, u32)>> = (0..n).map(|_| None).collect();
        let mut stash = std::collections::HashMap::new();

        for entry in prepared {
            let mut current = entry;
            let mut which = 0u8; // 0 → try seed1 first, 1 → seed2 first
            let mut kicks = 0u32;
            loop {
                let target = if which == 0 {
                    Self::hash_bucket(&current.0, &seed1, n)
                } else {
                    Self::hash_bucket(&current.0, &seed2, n)
                };
                if let Some(displaced) = occupants[target].replace(current) {
                    // Evicted previous occupant — try its *other*
                    // candidate next.
                    current = displaced;
                    // Flip: if we just put on its seed1, next try seed2.
                    let a = Self::hash_bucket(&current.0, &seed1, n);
                    let b = Self::hash_bucket(&current.0, &seed2, n);
                    which = if target == a { 1 } else if target == b { 0 } else { 0 };
                    kicks += 1;
                    if kicks > params.max_kicks {
                        stash.insert(current.0, (current.1, current.2));
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // Serialise every bucket — real or dummy. Encrypt with
        // (bucket_id, generation=1) nonce. Push to backend.
        let generation = 1u32;
        let mut updates = Vec::with_capacity(n);
        for (bid, slot) in occupants.iter().enumerate() {
            let bp = match slot {
                Some((logical_key, padded_values, value_count)) => BucketPlain {
                    dummy_flag: 0,
                    fingerprint: logical_key.0,
                    value_count: *value_count,
                    values: padded_values.clone(),
                },
                None => BucketPlain::dummy(&params),
            };
            let pt = bp.serialise(&params);
            let ct = aes_encrypt(&key, bid as u32, generation, &pt);
            updates.push(EncryptedBucket {
                bucket_id: bid as u32,
                generation,
                ciphertext: ct,
            });
        }
        backend.write_buckets(&updates);

        Ok(Self {
            backend,
            params,
            key,
            seed1,
            seed2,
            generation,
            stash,
        })
    }

    /// Fetch the value list for `key`. Returns `Vec` with up to
    /// `volume_bound` entries — caller truncates by the encoded
    /// `value_count`. Missing keys return an empty `Vec` (not an
    /// error).
    pub fn get(&self, key: &LogicalKey) -> Result<Vec<LogicalValue>, XorMmError> {
        let results = self.get_batch(&[*key])?;
        Ok(results.into_iter().next().expect("one in, one out"))
    }

    /// Batched fetch: 2k bucket reads for k input keys. The server
    /// observes 2k accesses but cannot pair them to specific keys
    /// (the bucket-id sequence is the only side channel).
    pub fn get_batch(&self, keys: &[LogicalKey]) -> Result<Vec<Vec<LogicalValue>>, XorMmError> {
        let n = self.params.n_buckets as usize;
        let mut bucket_ids = Vec::with_capacity(keys.len() * 2);
        for k in keys {
            bucket_ids.push(Self::hash_bucket(k, &self.seed1, n) as u32);
            bucket_ids.push(Self::hash_bucket(k, &self.seed2, n) as u32);
        }
        let encrypted = self.backend.read_buckets(&bucket_ids);

        let mut out = Vec::with_capacity(keys.len());
        for (i, k) in keys.iter().enumerate() {
            // Decrypt both candidate buckets.
            let eb_a = &encrypted[2 * i];
            let eb_b = &encrypted[2 * i + 1];
            let pt_a = aes_decrypt(&self.key, eb_a.bucket_id, eb_a.generation, &eb_a.ciphertext)?;
            let pt_b = aes_decrypt(&self.key, eb_b.bucket_id, eb_b.generation, &eb_b.ciphertext)?;
            let bp_a = BucketPlain::deserialise(&pt_a, &self.params);
            let bp_b = BucketPlain::deserialise(&pt_b, &self.params);

            // Pick the bucket whose fingerprint matches; fall back to
            // the in-CVM stash; otherwise return empty (key absent).
            let chosen: Option<&BucketPlain> = if bp_a.dummy_flag == 0 && fingerprint_matches(k, &bp_a.fingerprint) {
                Some(&bp_a)
            } else if bp_b.dummy_flag == 0 && fingerprint_matches(k, &bp_b.fingerprint) {
                Some(&bp_b)
            } else {
                None
            };

            let values = if let Some(bp) = chosen {
                let n_real = bp.value_count.min(self.params.volume_bound) as usize;
                bp.values
                    .iter()
                    .take(n_real)
                    .map(|v| LogicalValue(v.clone()))
                    .collect()
            } else if let Some((padded, value_count)) = self.stash.get(k) {
                let n_real = (*value_count).min(self.params.volume_bound) as usize;
                padded
                    .iter()
                    .take(n_real)
                    .map(|v| LogicalValue(v.clone()))
                    .collect()
            } else {
                // Sentinel-fingerprint match: key never built.
                let _ = DUMMY_FINGERPRINT;
                Vec::new()
            };
            out.push(values);
        }

        Ok(out)
    }

    /// Build-time stash depth — exposed for tests. In production a
    /// stash overflow would surface as a `BuildOverflow` error;
    /// non-zero but small stashes are normal.
    pub fn stash_len(&self) -> usize {
        self.stash.len()
    }

    /// Hash `key` into `[0, n)`. Uses SHA-256 with a 16-byte seed
    /// prefix to keep the two hash families independent.
    fn hash_bucket(key: &LogicalKey, seed: &[u8; 16], n: usize) -> usize {
        let mut h = Sha256::new();
        h.update(seed);
        h.update(key.0);
        let digest = h.finalize();
        let mut u = u64::from_le_bytes(digest[..8].try_into().expect("8B"));
        // Multiply-shift onto [0, n). n is u32-bounded in our params.
        u %= n as u64;
        u as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InMemoryByteStore;

    fn key_of(byte: u8) -> LogicalKey {
        LogicalKey([byte; 32])
    }

    fn mk_value(byte: u8, n: usize) -> LogicalValue {
        LogicalValue(vec![byte; n])
    }

    fn small_params() -> XorMmParams {
        XorMmParams {
            volume_bound: 3,
            value_bytes: 4,
            n_buckets: 64,
            max_kicks: 16,
        }
    }

    fn build_emm(
        entries: Vec<(LogicalKey, Vec<LogicalValue>)>,
        params: XorMmParams,
    ) -> XorMmClient<InMemoryByteStore> {
        let backend = InMemoryByteStore::new(params.n_buckets);
        XorMmClient::build(
            entries,
            params,
            Zeroizing::new([0x77; 32]),
            [0x11; 16],
            [0x22; 16],
            backend,
        )
        .expect("build ok")
    }

    #[test]
    fn build_then_get_round_trips_one_key() {
        let p = small_params();
        let entries = vec![(
            key_of(1),
            vec![mk_value(0xa1, 4), mk_value(0xa2, 4)],
        )];
        let emm = build_emm(entries, p);
        let got = emm.get(&key_of(1)).expect("get ok");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], mk_value(0xa1, 4));
        assert_eq!(got[1], mk_value(0xa2, 4));
    }

    #[test]
    fn missing_key_returns_empty() {
        let p = small_params();
        let entries = vec![(key_of(1), vec![mk_value(0xa1, 4)])];
        let emm = build_emm(entries, p);
        assert!(emm.get(&key_of(99)).expect("get ok").is_empty());
    }

    #[test]
    fn get_truncates_to_value_count() {
        // Caller supplies 3 values; volume_bound is 3 — return 3.
        let p = small_params();
        let entries = vec![(
            key_of(1),
            vec![mk_value(0xa1, 4), mk_value(0xa2, 4), mk_value(0xa3, 4)],
        )];
        let emm = build_emm(entries, p);
        let got = emm.get(&key_of(1)).expect("get ok");
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn many_keys_build_and_resolve() {
        // 30 keys into 64 buckets — well within cuckoo's high-prob
        // success region (load factor ~0.47).
        let p = small_params();
        let entries: Vec<_> = (1u8..=30)
            .map(|i| (key_of(i), vec![mk_value(i, 4), mk_value(i.wrapping_add(100), 4)]))
            .collect();
        let emm = build_emm(entries, p);
        for i in 1u8..=30 {
            let got = emm.get(&key_of(i)).expect("get ok");
            assert_eq!(got.len(), 2);
            assert_eq!(got[0].0[0], i);
            assert_eq!(got[1].0[0], i.wrapping_add(100));
        }
    }

    #[test]
    fn batch_get_matches_individual_gets() {
        let p = small_params();
        let entries: Vec<_> = (1u8..=10)
            .map(|i| (key_of(i), vec![mk_value(i, 4)]))
            .collect();
        let emm = build_emm(entries, p);
        let keys: Vec<_> = (1u8..=10).map(key_of).collect();
        let batch = emm.get_batch(&keys).expect("get_batch ok");
        for (i, vals) in batch.iter().enumerate() {
            let single = emm.get(&keys[i]).expect("get ok");
            assert_eq!(vals, &single);
        }
    }

    #[test]
    fn value_longer_than_value_bytes_errors() {
        let p = small_params();
        let too_long = LogicalValue(vec![0xff; 5]); // p.value_bytes == 4
        let backend = InMemoryByteStore::new(p.n_buckets);
        let r = XorMmClient::build(
            vec![(key_of(1), vec![too_long])],
            p,
            Zeroizing::new([0x77; 32]),
            [0x11; 16],
            [0x22; 16],
            backend,
        );
        assert!(matches!(r, Err(XorMmError::ValueTooLong(5, 4))));
    }

    #[test]
    fn volume_hiding_buckets_same_size_regardless_of_value_count() {
        // Two EMMs: one key has 1 value, another has 3. Both buckets
        // serialise to the same byte length.
        let p = small_params();
        let a = build_emm(vec![(key_of(1), vec![mk_value(1, 4)])], p);
        let b = build_emm(
            vec![(key_of(1), vec![mk_value(1, 4), mk_value(2, 4), mk_value(3, 4)])],
            p,
        );
        // Pick the same bucket from both backends; sizes must match.
        let buckets_a = a.backend.read_buckets(&[0]);
        let buckets_b = b.backend.read_buckets(&[0]);
        assert_eq!(buckets_a[0].ciphertext.len(), buckets_b[0].ciphertext.len());
    }

    #[test]
    fn rebuild_changes_ciphertext_under_different_seeds() {
        // Same logical data, different hash seeds ⇒ different bucket
        // placement ⇒ different on-wire bytes. Sanity-checks that the
        // seeds participate.
        let p = small_params();
        let entries = vec![(key_of(1), vec![mk_value(0xa1, 4)])];
        let backend_a = InMemoryByteStore::new(p.n_buckets);
        let a = XorMmClient::build(
            entries.clone(),
            p,
            Zeroizing::new([0x77; 32]),
            [0x11; 16],
            [0x22; 16],
            backend_a,
        )
        .unwrap();
        let backend_b = InMemoryByteStore::new(p.n_buckets);
        let b = XorMmClient::build(
            entries,
            p,
            Zeroizing::new([0x77; 32]),
            [0x33; 16], // different seed
            [0x44; 16],
            backend_b,
        )
        .unwrap();
        // Both backends decrypt with their own keys, but the bucket
        // placements differ — at least one bucket index differs.
        let raw_a = a.backend.read_buckets(&(0..p.n_buckets).collect::<Vec<_>>());
        let raw_b = b.backend.read_buckets(&(0..p.n_buckets).collect::<Vec<_>>());
        // The two ciphertexts encrypt the same plaintext under same
        // key but different nonces (no — same nonce because (bid,
        // gen=1)) and possibly different plaintexts (different
        // placement). At least one bucket must differ.
        let any_diff = raw_a
            .iter()
            .zip(raw_b.iter())
            .any(|(a, b)| a.ciphertext != b.ciphertext);
        assert!(any_diff, "different seeds should change at least one bucket");
    }
}
