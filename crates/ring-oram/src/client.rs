//! `RingOramClient` — the semi-honest baseline (M1).
//!
//! Held entirely inside the CVM. Owns the [`PositionMap`], the
//! [`Stash`], the 32-byte AES-GCM key, and a per-tree write-counter
//! mirror used to re-derive AES-GCM nonces. Talks to one
//! [`BlockBackend`] which represents the untrusted storage server.
//!
//! Protocol:
//!
//! 1. **read** — look up the block's current `PathId`, fetch the path
//!    from the backend, decrypt all `Z + S · levels` slots, locate the
//!    target block (or note its absence — it's in the stash from a
//!    previous read), pick a fresh uniform `PathId` for it, drop the
//!    fetched path back into the stash, and write the path back with
//!    *only dummies* — the real blocks live in the stash until
//!    `evict_path` returns them.
//! 2. **evict_path** — pick a reverse-lexicographic path, fill it
//!    greedily from the stash with blocks whose current path crosses
//!    each bucket, write the path back encrypted.
//!
//! No XOR trick, no Merkle integrity, no multi-block batching, no lazy
//! eviction. Those are M4 layered on top.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::backend::{BlockBackend, EncryptedBucket};
use crate::block::{Block, BlockId, BlockPayload};
use crate::codec::{aes_decrypt, aes_encrypt, deserialise_bucket, serialise_bucket, AesError};
use crate::params::RingOramParams;
use crate::path::{PathId, path_buckets};
use crate::posmap::PositionMap;
use crate::stash::Stash;

#[derive(Debug, thiserror::Error)]
pub enum OramError {
    #[error("block {0:?} not admitted to ORAM")]
    UnknownBlock(BlockId),
    #[error("bucket corruption: {0}")]
    Corrupted(#[from] AesError),
    #[error("payload size mismatch: expected {expected}, got {got}")]
    PayloadSize { expected: usize, got: usize },
    #[error("stash overflow: {0} blocks unable to land back on tree")]
    StashOverflow(usize),
}

pub struct RingOramClient<B: BlockBackend> {
    backend: B,
    params: RingOramParams,
    key: [u8; 32],
    posmap: PositionMap,
    stash: Stash,
    /// Mirror of each bucket's write-counter. Needed because AES-GCM
    /// nonce = (bucket_id, write_counter) and the client picks both.
    /// In semi-honest mode the server is trusted to return the same
    /// counter on read; M4 adds Merkle integrity over this value.
    counters: Vec<u32>,
    /// Accesses (admit + read + write) since the last `evict_path`.
    /// Eviction fires when this hits `A`. Both `admit` and `read` count
    /// because they both put a block into the stash that needs to
    /// drain back to the tree.
    accesses_since_evict: u32,
    /// Reverse-lexicographic eviction cursor. Increments after each
    /// eviction; wraps at `n_leaves`.
    evict_cursor: u32,
    /// PRNG for fresh `PathId` choices. Seeded from a 32-byte secret
    /// at construction; deterministic given the seed so tests can
    /// reproduce traces.
    rng: ChaCha20Rng,
}

impl<B: BlockBackend> RingOramClient<B> {
    /// Construct an empty client over a backend that has already been
    /// allocated with `params.num_buckets()` slots. The tree is filled
    /// with all-dummy buckets via an initial write pass.
    pub fn new(backend: B, params: RingOramParams, key: [u8; 32], rng_seed: [u8; 32]) -> Self {
        assert_eq!(
            backend.num_buckets(),
            params.num_buckets(),
            "backend size mismatch: backend has {} buckets, params demand {}",
            backend.num_buckets(),
            params.num_buckets()
        );
        let counters = vec![0u32; params.num_buckets() as usize];
        let mut client = Self {
            backend,
            params,
            key,
            posmap: PositionMap::with_capacity(0),
            stash: Stash::new(),
            counters,
            accesses_since_evict: 0,
            evict_cursor: 0,
            rng: ChaCha20Rng::from_seed(rng_seed),
        };
        client.initialize_tree();
        client
    }

    /// Initial write pass: every bucket gets `(Z + S)` dummies under
    /// `write_counter = 1`. After this, every read produces a valid
    /// AEAD frame.
    fn initialize_tree(&mut self) {
        let cap = self.params.bucket_capacity() as usize;
        let dummies: Vec<Block> = (0..cap)
            .map(|_| Block::dummy(self.params.block_bytes as usize))
            .collect();
        let pt = serialise_bucket(&dummies, &self.params);
        let mut updates = Vec::with_capacity(self.params.num_buckets() as usize);
        for bid in 0..self.params.num_buckets() {
            self.counters[bid as usize] = 1;
            let ct = aes_encrypt(&self.key, bid, 1, &pt);
            updates.push(EncryptedBucket {
                bucket_id: bid,
                write_counter: 1,
                ciphertext: ct,
            });
        }
        self.backend.write_buckets(&updates);
    }

    /// Admit a new block at a fresh random path. The block lands in
    /// the stash; an `evict_path` call moves it onto the tree. M1's
    /// simple put-then-evict pattern is sufficient for correctness;
    /// M4 will combine admit + write via the unified `write` op.
    pub fn admit(&mut self, id: BlockId, payload: Vec<u8>) -> Result<(), OramError> {
        if payload.len() != self.params.block_bytes as usize {
            return Err(OramError::PayloadSize {
                expected: self.params.block_bytes as usize,
                got: payload.len(),
            });
        }
        let path = self.fresh_path();
        self.posmap.set(id, path);
        let block = Block {
            id,
            payload: BlockPayload::from_exact(payload, self.params.block_bytes as usize),
        };
        self.stash.insert(block);
        self.accesses_since_evict += 1;
        self.maybe_evict();
        Ok(())
    }

    /// Fetch the current payload of `id`. Returns a fresh `Vec<u8>` —
    /// caller owns it. The block is reassigned to a new random path
    /// before return so two consecutive reads of the same id visit
    /// independent server-side paths.
    pub fn read(&mut self, id: BlockId) -> Result<Vec<u8>, OramError> {
        let path = self
            .posmap
            .get(id)
            .ok_or(OramError::UnknownBlock(id))?;

        // 1. Pull the full path off the backend.
        let bucket_ids = path_buckets(path, self.params.n_leaves);
        let encrypted = self.backend.read_path(&bucket_ids);

        // 2. Decrypt every bucket; non-dummy real blocks go to the stash.
        for (eb, &bid) in encrypted.iter().zip(bucket_ids.iter()) {
            debug_assert_eq!(eb.bucket_id, bid);
            let pt = aes_decrypt(&self.key, eb.bucket_id, eb.write_counter, &eb.ciphertext)?;
            let blocks = deserialise_bucket(&pt, &self.params);
            for block in blocks {
                if !block.is_dummy() && self.stash.peek(block.id).is_none() {
                    self.stash.insert(block);
                }
            }
        }

        // 3. Now the stash holds the target. Read its payload …
        let payload_out = self
            .stash
            .peek(id)
            .expect("block claimed in posmap but missing post-read")
            .payload
            .as_bytes()
            .to_vec();

        // 4. … assign it a fresh path; it stays in the stash until
        //    the next `evict_path` flushes it back onto the tree.
        let new_path = self.fresh_path();
        self.posmap.set(id, new_path);

        // 5. Write back the path with *only dummies* — the real
        //    blocks are now stash-resident. Refreshes write_counters.
        let cap = self.params.bucket_capacity() as usize;
        let dummies: Vec<Block> = (0..cap)
            .map(|_| Block::dummy(self.params.block_bytes as usize))
            .collect();
        let pt = serialise_bucket(&dummies, &self.params);
        let mut updates = Vec::with_capacity(bucket_ids.len());
        for &bid in &bucket_ids {
            self.counters[bid as usize] += 1;
            let counter = self.counters[bid as usize];
            let ct = aes_encrypt(&self.key, bid, counter, &pt);
            updates.push(EncryptedBucket {
                bucket_id: bid,
                write_counter: counter,
                ciphertext: ct,
            });
        }
        self.backend.write_buckets(&updates);

        self.accesses_since_evict += 1;
        self.maybe_evict();
        Ok(payload_out)
    }

    /// Overwrite a block in place. Equivalent to read + write-back,
    /// but skips the read-then-discard payload copy.
    pub fn write(&mut self, id: BlockId, payload: Vec<u8>) -> Result<(), OramError> {
        if payload.len() != self.params.block_bytes as usize {
            return Err(OramError::PayloadSize {
                expected: self.params.block_bytes as usize,
                got: payload.len(),
            });
        }
        // Bring the block into the stash via a read; discard the
        // payload that came back.
        let _ = self.read(id)?;
        // Replace the stash entry with the new payload.
        let mut entry = self
            .stash
            .take(id)
            .expect("read placed it in the stash");
        entry.payload =
            BlockPayload::from_exact(payload, self.params.block_bytes as usize);
        self.stash.insert(entry);
        Ok(())
    }

    /// Current stash depth — exposed for tests and the M4 bound proof.
    pub fn stash_len(&self) -> usize {
        self.stash.len()
    }

    /// Public size of the backend tree — useful for sizing tests.
    pub fn num_buckets(&self) -> u32 {
        self.backend.num_buckets()
    }

    // ─── internals ────────────────────────────────────────────────

    fn fresh_path(&mut self) -> PathId {
        PathId(self.rng.random_range(0..self.params.n_leaves))
    }

    fn maybe_evict(&mut self) {
        if self.accesses_since_evict >= self.params.a {
            self.evict_path();
            self.accesses_since_evict = 0;
        }
    }

    /// Reverse-lexicographic eviction. Pick the next leaf via the
    /// bit-reversal trick the paper specifies (paper §III); for M1 we
    /// approximate with a simple round-robin cursor — uniform over all
    /// leaves, which is the property the stash-bound proof actually
    /// needs. M4 swaps to true reverse-lexicographic for paper
    /// parity.
    ///
    /// Protocol (standard Path-ORAM eviction):
    ///   1. Read the eviction path. Drain every real block into the
    ///      stash. This is critical — without it, writing the eviction
    ///      path *overwrites* upper-level buckets that may already hold
    ///      real blocks placed by an earlier eviction along a
    ///      different path (root and other shared ancestors).
    ///   2. Re-pack: from leaf to root, fill each bucket greedily with
    ///      up to `Z` stash blocks whose assigned path crosses that
    ///      bucket (deepest wins — closer to leaf = less future
    ///      reshuffling).
    ///   3. Write the path back encrypted, bumping each bucket's
    ///      write_counter so AES-GCM nonces don't reuse.
    fn evict_path(&mut self) {
        let path = PathId(self.evict_cursor % self.params.n_leaves);
        self.evict_cursor = (self.evict_cursor + 1) % self.params.n_leaves;
        let bucket_ids = path_buckets(path, self.params.n_leaves);

        // 1. Read the path, drain real blocks into the stash. Mirrors
        //    the same step in `read()` — without this, we'd silently
        //    erase tree contents at shared ancestors.
        let encrypted = self.backend.read_path(&bucket_ids);
        for (eb, &bid) in encrypted.iter().zip(bucket_ids.iter()) {
            debug_assert_eq!(eb.bucket_id, bid);
            let pt = aes_decrypt(&self.key, eb.bucket_id, eb.write_counter, &eb.ciphertext)
                .expect("eviction path decrypt — tree contents must remain authentic");
            let blocks = deserialise_bucket(&pt, &self.params);
            for block in blocks {
                if !block.is_dummy() && self.stash.peek(block.id).is_none() {
                    self.stash.insert(block);
                }
            }
        }

        // 2. Re-pack: for each bucket on the eviction path, pull from
        //    the stash blocks whose assigned path also crosses that
        //    bucket. Greedy depth-first — try leaves first so blocks
        //    sink as far down as their assigned path allows.
        let cap = self.params.bucket_capacity() as usize;
        let z = self.params.z as usize;
        let mut bucket_contents: Vec<Vec<Block>> = vec![Vec::new(); bucket_ids.len()];
        let candidate_ids: Vec<BlockId> = self.stash.ids().collect();
        for block_id in candidate_ids {
            let block_path = self
                .posmap
                .get(block_id)
                .expect("stash block must be in posmap");
            let block_bucket_ids = path_buckets(block_path, self.params.n_leaves);
            for (depth_idx, &bid) in bucket_ids.iter().enumerate().rev() {
                if block_bucket_ids.contains(&bid) && bucket_contents[depth_idx].len() < z {
                    let block = self
                        .stash
                        .take(block_id)
                        .expect("just iterated, must still be present");
                    bucket_contents[depth_idx].push(block);
                    break;
                }
            }
            // If unplaceable, block stays in the stash. The paper
            // bounds the residual at O(log N) in expectation.
        }

        // 3. Write the path back: pad each bucket up to (Z + S) with
        //    dummies, serialise, encrypt under bumped counter, push.
        let mut updates = Vec::with_capacity(bucket_ids.len());
        for (i, &bid) in bucket_ids.iter().enumerate() {
            let real = std::mem::take(&mut bucket_contents[i]);
            let mut blocks: Vec<Block> = real;
            while blocks.len() < cap {
                blocks.push(Block::dummy(self.params.block_bytes as usize));
            }
            let pt = serialise_bucket(&blocks, &self.params);
            self.counters[bid as usize] += 1;
            let counter = self.counters[bid as usize];
            let ct = aes_encrypt(&self.key, bid, counter, &pt);
            updates.push(EncryptedBucket {
                bucket_id: bid,
                write_counter: counter,
                ciphertext: ct,
            });
        }
        self.backend.write_buckets(&updates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InMemoryBlockBackend;

    fn mk_client(n_leaves: u32, block_bytes: u32) -> RingOramClient<InMemoryBlockBackend> {
        let params = RingOramParams {
            z: 4,
            s: 5,
            a: 3,
            block_bytes,
            n_leaves,
        };
        let backend = InMemoryBlockBackend::new(params.num_buckets());
        RingOramClient::new(backend, params, [0x11; 32], [0x22; 32])
    }

    #[test]
    fn admit_then_read_round_trips_one_block() {
        let mut c = mk_client(16, 8);
        c.admit(BlockId(0), vec![0xab; 8]).unwrap();
        let got = c.read(BlockId(0)).unwrap();
        assert_eq!(got, vec![0xab; 8]);
    }

    #[test]
    fn read_unknown_block_errors() {
        let mut c = mk_client(16, 8);
        let r = c.read(BlockId(7));
        assert!(matches!(r, Err(OramError::UnknownBlock(_))));
    }

    #[test]
    fn write_overwrites_payload() {
        let mut c = mk_client(16, 8);
        c.admit(BlockId(3), vec![0x01; 8]).unwrap();
        c.write(BlockId(3), vec![0x02; 8]).unwrap();
        let got = c.read(BlockId(3)).unwrap();
        assert_eq!(got, vec![0x02; 8]);
    }

    #[test]
    fn many_blocks_round_trip_through_eviction() {
        // 32 leaves ⇒ 63 buckets ⇒ Z·63 = 252 real slots, enough
        // headroom for 32 random blocks plus the stash budget.
        let mut c = mk_client(32, 16);
        let n = 32u32;
        for i in 0..n {
            let mut buf = vec![0u8; 16];
            buf[..4].copy_from_slice(&i.to_le_bytes());
            c.admit(BlockId(i), buf).unwrap();
        }
        // Read every block twice in mixed order — exercises the
        // path-reassignment + stash path.
        for &i in &[5u32, 17, 0, 31, 12, 8, 8, 5] {
            let got = c.read(BlockId(i)).unwrap();
            let want_prefix = i.to_le_bytes();
            assert_eq!(&got[..4], &want_prefix);
        }
    }

    #[test]
    fn stash_stays_small_under_load() {
        // Worst-case-ish: lots of reads + writes, watch the stash.
        let mut c = mk_client(64, 8);
        let n = 64u32;
        for i in 0..n {
            c.admit(BlockId(i), vec![(i & 0xff) as u8; 8]).unwrap();
        }
        // Drive 4 × n accesses; with Z=4, S=5, A=3 the stash should
        // remain well under 4N just by the paper bound. The actual
        // sharp bound is O(log N) in expectation; we assert a
        // generous functional ceiling.
        for round in 0..4 {
            for i in 0..n {
                let _ = c.read(BlockId(i)).unwrap();
                assert!(
                    c.stash_len() < (n as usize),
                    "stash exploded at round {round} i={i}: {}",
                    c.stash_len()
                );
            }
        }
    }

    #[test]
    fn writes_are_durable_across_many_evictions() {
        let mut c = mk_client(32, 8);
        for i in 0..16u32 {
            c.admit(BlockId(i), vec![i as u8; 8]).unwrap();
        }
        // Touch every block; eviction will run multiple times.
        for _ in 0..3 {
            for i in 0..16u32 {
                assert_eq!(c.read(BlockId(i)).unwrap(), vec![i as u8; 8]);
            }
        }
    }

    #[test]
    fn aes_key_change_breaks_reads() {
        // Build a client, admit a block, then wrap a *different* key
        // around the same backend and verify the new client cannot
        // decrypt. Catches a key-handling regression.
        let params = RingOramParams {
            z: 4,
            s: 5,
            a: 3,
            block_bytes: 8,
            n_leaves: 16,
        };
        let backend = InMemoryBlockBackend::new(params.num_buckets());
        let mut c = RingOramClient::new(backend, params, [0xaa; 32], [0x01; 32]);
        c.admit(BlockId(2), vec![0xff; 8]).unwrap();

        // Drain a path through the backend and try to decrypt with a
        // wrong key.
        let bad_key = [0xbb; 32];
        let path = path_buckets(PathId(0), params.n_leaves);
        let buckets = c.backend.read_path(&path);
        let result = aes_decrypt(&bad_key, buckets[0].bucket_id, buckets[0].write_counter, &buckets[0].ciphertext);
        assert!(result.is_err());
    }
}
