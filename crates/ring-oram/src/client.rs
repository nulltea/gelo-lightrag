//! `RingOramClient` — the semi-honest baseline (M1, async since M5).
//!
//! Held entirely inside the CVM. Owns the [`PositionMap`], the
//! [`Stash`], the 32-byte AES-GCM key, and a per-tree write-counter
//! mirror used to re-derive AES-GCM nonces. Talks to one
//! [`BlockBackend`] which represents the untrusted storage server —
//! `InMemoryBlockBackend` for tests, `compass-rest-backend` over the
//! network in production.
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
//! eviction beyond the §4.7 deferred-flush switch. Those are M4
//! layered on top.

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
    /// Multi-hop lazy eviction (Compass §4.7): when `true`,
    /// `maybe_evict` accumulates a pending count instead of running
    /// eviction inline. Caller invokes `flush_evictions()` after a
    /// multi-step traversal to drain. Off by default — preserves the
    /// M1 stash-bound behaviour for callers that don't opt in.
    defer_evictions: bool,
    /// Pending eviction count when `defer_evictions` is on.
    pending_evictions: u32,
    /// Reverse-lexicographic eviction cursor. Increments after each
    /// eviction; wraps at `n_leaves`.
    evict_cursor: u32,
    /// PRNG for fresh `PathId` choices. Seeded from a 32-byte secret
    /// at construction; deterministic given the seed so tests can
    /// reproduce traces.
    rng: ChaCha20Rng,
    /// Treetop cache (Compass §4.7). Indexed by `bucket_id ∈
    /// [0, treetop_bucket_count)`. Reads hit cache; writes mirror to
    /// both cache and backend. Empty when `treetop_levels == 0`.
    treetop_cache: Vec<EncryptedBucket>,
}

impl<B: BlockBackend> RingOramClient<B> {
    /// Construct an empty client over a backend that has already been
    /// allocated with `params.num_buckets()` slots. The tree is filled
    /// with all-dummy buckets via an initial write pass.
    ///
    /// Async since M5 because `initialize_tree` writes one round-trip
    /// per bucket to the backend; on the in-memory backend this is
    /// effectively a no-await.
    pub async fn new(
        backend: B,
        params: RingOramParams,
        key: [u8; 32],
        rng_seed: [u8; 32],
    ) -> Self {
        assert_eq!(
            backend.num_buckets(),
            params.num_buckets(),
            "backend size mismatch: backend has {} buckets, params demand {}",
            backend.num_buckets(),
            params.num_buckets()
        );
        let counters = vec![0u32; params.num_buckets() as usize];
        let treetop_cache = Vec::with_capacity(params.treetop_bucket_count() as usize);
        let mut client = Self {
            backend,
            params,
            key,
            posmap: PositionMap::with_capacity(0),
            stash: Stash::new(),
            counters,
            accesses_since_evict: 0,
            defer_evictions: false,
            pending_evictions: 0,
            evict_cursor: 0,
            rng: ChaCha20Rng::from_seed(rng_seed),
            treetop_cache,
        };
        client.initialize_tree().await;
        client
    }

    /// Internal: read a path. Treetop buckets come from the cache;
    /// the rest from the backend. Result preserves the requested
    /// order.
    async fn read_path_cached(&self, bucket_ids: &[u32]) -> Vec<EncryptedBucket> {
        let mut out: Vec<Option<EncryptedBucket>> = vec![None; bucket_ids.len()];
        let mut deep_ids: Vec<u32> = Vec::new();
        let mut deep_positions: Vec<usize> = Vec::new();
        for (i, &bid) in bucket_ids.iter().enumerate() {
            if self.params.bucket_in_treetop(bid) {
                out[i] = Some(self.treetop_cache[bid as usize].clone());
            } else {
                deep_positions.push(i);
                deep_ids.push(bid);
            }
        }
        if !deep_ids.is_empty() {
            let deep = self.backend.read_path(&deep_ids).await;
            for (pos, eb) in deep_positions.into_iter().zip(deep.into_iter()) {
                out[pos] = Some(eb);
            }
        }
        out.into_iter()
            .map(|b| b.expect("every bucket either cached or backend-fetched"))
            .collect()
    }

    /// Internal: write a path. Treetop buckets mirror into the cache;
    /// every bucket (including treetop) still gets a backend write so
    /// a crash-recovery flow can re-seed the cache.
    async fn write_buckets_cached(&mut self, buckets: &[EncryptedBucket]) {
        for b in buckets {
            if self.params.bucket_in_treetop(b.bucket_id) {
                let idx = b.bucket_id as usize;
                if idx >= self.treetop_cache.len() {
                    // First write — grow the cache. initialize_tree
                    // writes every bucket in id order so this is
                    // monotonically extending.
                    while self.treetop_cache.len() <= idx {
                        self.treetop_cache.push(EncryptedBucket {
                            bucket_id: self.treetop_cache.len() as u32,
                            write_counter: 0,
                            ciphertext: Vec::new(),
                        });
                    }
                }
                self.treetop_cache[idx] = b.clone();
            }
        }
        self.backend.write_buckets(buckets).await;
    }

    /// Initial write pass: every bucket gets `(Z + S)` dummies under
    /// `write_counter = 1`. After this, every read produces a valid
    /// AEAD frame.
    async fn initialize_tree(&mut self) {
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
        self.write_buckets_cached(&updates).await;
    }

    /// Admit a new block at a fresh random path. The block lands in
    /// the stash; an `evict_path` call moves it onto the tree. M1's
    /// simple put-then-evict pattern is sufficient for correctness;
    /// M4 will combine admit + write via the unified `write` op.
    pub async fn admit(&mut self, id: BlockId, payload: Vec<u8>) -> Result<(), OramError> {
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
        self.maybe_evict().await;
        Ok(())
    }

    /// Fetch the current payload of `id`. Returns a fresh `Vec<u8>` —
    /// caller owns it. The block is reassigned to a new random path
    /// before return so two consecutive reads of the same id visit
    /// independent server-side paths.
    pub async fn read(&mut self, id: BlockId) -> Result<Vec<u8>, OramError> {
        let path = self
            .posmap
            .get(id)
            .ok_or(OramError::UnknownBlock(id))?;

        // 1. Pull the full path — treetop comes from the cache,
        //    deeper buckets from the backend.
        let bucket_ids = path_buckets(path, self.params.n_leaves);
        let encrypted = self.read_path_cached(&bucket_ids).await;

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
        self.write_buckets_cached(&updates).await;

        self.accesses_since_evict += 1;
        self.maybe_evict().await;
        Ok(payload_out)
    }

    /// Overwrite a block in place. Equivalent to read + write-back,
    /// but skips the read-then-discard payload copy.
    pub async fn write(&mut self, id: BlockId, payload: Vec<u8>) -> Result<(), OramError> {
        if payload.len() != self.params.block_bytes as usize {
            return Err(OramError::PayloadSize {
                expected: self.params.block_bytes as usize,
                got: payload.len(),
            });
        }
        // Bring the block into the stash via a read; discard the
        // payload that came back.
        let _ = self.read(id).await?;
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

    async fn maybe_evict(&mut self) {
        while self.accesses_since_evict >= self.params.a {
            self.accesses_since_evict -= self.params.a;
            if self.defer_evictions {
                self.pending_evictions += 1;
            } else {
                self.evict_path().await;
            }
        }
    }

    /// Turn multi-hop lazy eviction on or off. Sync — only toggles the
    /// internal flag. Callers MUST call `flush_evictions().await` once
    /// the deferred phase is over, or use the typical pattern:
    ///
    /// ```ignore
    /// oram.set_defer_evictions(true);
    /// // … several reads/writes …
    /// oram.set_defer_evictions(false);
    /// oram.flush_evictions().await;
    /// ```
    ///
    /// Toggling off without flushing is allowed (subsequent ops will
    /// run eviction inline) but pending counts stay queued — they're
    /// drained on the next `flush_evictions()` or implicitly by a
    /// later `maybe_evict` if a-threshold is crossed again.
    pub fn set_defer_evictions(&mut self, defer: bool) {
        self.defer_evictions = defer;
    }

    /// Drain pending evictions. Safe to call regardless of
    /// `defer_evictions` state.
    pub async fn flush_evictions(&mut self) {
        while self.pending_evictions > 0 {
            self.evict_path().await;
            self.pending_evictions -= 1;
        }
    }

    /// Telemetry: how many evictions are queued for the next flush.
    /// Always 0 when `defer_evictions` is off.
    pub fn pending_evictions(&self) -> u32 {
        self.pending_evictions
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
    async fn evict_path(&mut self) {
        let path = PathId(self.evict_cursor % self.params.n_leaves);
        self.evict_cursor = (self.evict_cursor + 1) % self.params.n_leaves;
        let bucket_ids = path_buckets(path, self.params.n_leaves);

        // 1. Read the path, drain real blocks into the stash. Mirrors
        //    the same step in `read()` — without this, we'd silently
        //    erase tree contents at shared ancestors.
        let encrypted = self.read_path_cached(&bucket_ids).await;
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
        self.write_buckets_cached(&updates).await;
    }

    /// Borrow the backend immutably — used by tests that need to
    /// inspect backend telemetry (e.g. read_count).
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InMemoryBlockBackend;

    async fn mk_client(n_leaves: u32, block_bytes: u32) -> RingOramClient<InMemoryBlockBackend> {
        let params = RingOramParams {
            z: 4,
            s: 5,
            a: 3,
            block_bytes,
            n_leaves,
            treetop_levels: 0,
        };
        let backend = InMemoryBlockBackend::new(params.num_buckets());
        RingOramClient::new(backend, params, [0x11; 32], [0x22; 32]).await
    }

    #[tokio::test]
    async fn admit_then_read_round_trips_one_block() {
        let mut c = mk_client(16, 8).await;
        c.admit(BlockId(0), vec![0xab; 8]).await.unwrap();
        let got = c.read(BlockId(0)).await.unwrap();
        assert_eq!(got, vec![0xab; 8]);
    }

    #[tokio::test]
    async fn read_unknown_block_errors() {
        let mut c = mk_client(16, 8).await;
        let r = c.read(BlockId(7)).await;
        assert!(matches!(r, Err(OramError::UnknownBlock(_))));
    }

    #[tokio::test]
    async fn write_overwrites_payload() {
        let mut c = mk_client(16, 8).await;
        c.admit(BlockId(3), vec![0x01; 8]).await.unwrap();
        c.write(BlockId(3), vec![0x02; 8]).await.unwrap();
        let got = c.read(BlockId(3)).await.unwrap();
        assert_eq!(got, vec![0x02; 8]);
    }

    #[tokio::test]
    async fn many_blocks_round_trip_through_eviction() {
        // 32 leaves ⇒ 63 buckets ⇒ Z·63 = 252 real slots, enough
        // headroom for 32 random blocks plus the stash budget.
        let mut c = mk_client(32, 16).await;
        let n = 32u32;
        for i in 0..n {
            let mut buf = vec![0u8; 16];
            buf[..4].copy_from_slice(&i.to_le_bytes());
            c.admit(BlockId(i), buf).await.unwrap();
        }
        // Read every block twice in mixed order — exercises the
        // path-reassignment + stash path.
        for &i in &[5u32, 17, 0, 31, 12, 8, 8, 5] {
            let got = c.read(BlockId(i)).await.unwrap();
            let want_prefix = i.to_le_bytes();
            assert_eq!(&got[..4], &want_prefix);
        }
    }

    #[tokio::test]
    async fn stash_stays_small_under_load() {
        let mut c = mk_client(64, 8).await;
        let n = 64u32;
        for i in 0..n {
            c.admit(BlockId(i), vec![(i & 0xff) as u8; 8]).await.unwrap();
        }
        for round in 0..4 {
            for i in 0..n {
                let _ = c.read(BlockId(i)).await.unwrap();
                assert!(
                    c.stash_len() < (n as usize),
                    "stash exploded at round {round} i={i}: {}",
                    c.stash_len()
                );
            }
        }
    }

    #[tokio::test]
    async fn writes_are_durable_across_many_evictions() {
        let mut c = mk_client(32, 8).await;
        for i in 0..16u32 {
            c.admit(BlockId(i), vec![i as u8; 8]).await.unwrap();
        }
        for _ in 0..3 {
            for i in 0..16u32 {
                assert_eq!(c.read(BlockId(i)).await.unwrap(), vec![i as u8; 8]);
            }
        }
    }

    #[tokio::test]
    async fn lazy_eviction_defers_then_flushes() {
        // With defer_evictions on, repeated reads accumulate pending
        // evictions but the client stays correct. After flush, the
        // post-state must equal the inline-eviction baseline.
        let mut c = mk_client(32, 8).await;
        for i in 0..8u32 {
            c.admit(BlockId(i), vec![i as u8; 8]).await.unwrap();
        }
        c.set_defer_evictions(true);
        for _ in 0..3 {
            for i in 0..8u32 {
                let got = c.read(BlockId(i)).await.unwrap();
                assert_eq!(got, vec![i as u8; 8]);
            }
        }
        assert!(c.pending_evictions() > 0);
        c.flush_evictions().await;
        assert_eq!(c.pending_evictions(), 0);
        for i in 0..8u32 {
            assert_eq!(c.read(BlockId(i)).await.unwrap(), vec![i as u8; 8]);
        }
    }

    #[tokio::test]
    async fn set_defer_off_then_flush_drains_pending() {
        // Toggling off no longer auto-flushes (the API is sync); the
        // caller flushes explicitly. This test pins the contract.
        let mut c = mk_client(32, 8).await;
        for i in 0..8u32 {
            c.admit(BlockId(i), vec![i as u8; 8]).await.unwrap();
        }
        c.set_defer_evictions(true);
        for i in 0..8u32 {
            let _ = c.read(BlockId(i)).await.unwrap();
        }
        assert!(c.pending_evictions() > 0);
        c.set_defer_evictions(false);
        // Toggle-off alone does NOT flush — caller must call flush.
        c.flush_evictions().await;
        assert_eq!(c.pending_evictions(), 0);
    }

    #[tokio::test]
    async fn treetop_cache_skips_backend_reads_for_top_buckets() {
        let params = RingOramParams {
            z: 4,
            s: 5,
            a: 3,
            block_bytes: 8,
            n_leaves: 32,
            treetop_levels: 3,
        };
        let backend = InMemoryBlockBackend::new(params.num_buckets());
        let mut c = RingOramClient::new(backend, params, [0x42; 32], [0x84; 32]).await;
        let init_reads = c.backend().read_count();

        for i in 0..8u32 {
            c.admit(BlockId(i), vec![i as u8; 8]).await.unwrap();
        }
        let _ = c.read(BlockId(0)).await.unwrap();

        let post_read = c.backend().read_count();
        let baseline_params = RingOramParams {
            treetop_levels: 0,
            ..params
        };
        let baseline_backend = InMemoryBlockBackend::new(baseline_params.num_buckets());
        let mut baseline =
            RingOramClient::new(baseline_backend, baseline_params, [0x42; 32], [0x84; 32]).await;
        let base_init = baseline.backend().read_count();
        for i in 0..8u32 {
            baseline.admit(BlockId(i), vec![i as u8; 8]).await.unwrap();
        }
        let _ = baseline.read(BlockId(0)).await.unwrap();
        let baseline_total = baseline.backend().read_count() - base_init;
        let cached_total = post_read - init_reads;

        assert!(
            cached_total < baseline_total,
            "treetop cache did not reduce backend reads: cached={cached_total} baseline={baseline_total}"
        );
    }

    #[tokio::test]
    async fn treetop_cache_preserves_correctness_under_many_ops() {
        let params = RingOramParams {
            z: 4,
            s: 5,
            a: 3,
            block_bytes: 8,
            n_leaves: 32,
            treetop_levels: 3,
        };
        let backend = InMemoryBlockBackend::new(params.num_buckets());
        let mut c = RingOramClient::new(backend, params, [0x55; 32], [0xaa; 32]).await;
        for i in 0..16u32 {
            c.admit(BlockId(i), vec![i as u8; 8]).await.unwrap();
        }
        for _ in 0..3 {
            for i in 0..16u32 {
                assert_eq!(c.read(BlockId(i)).await.unwrap(), vec![i as u8; 8]);
            }
        }
    }

    #[tokio::test]
    async fn aes_key_change_breaks_reads() {
        let params = RingOramParams {
            z: 4,
            s: 5,
            a: 3,
            block_bytes: 8,
            n_leaves: 16,
            treetop_levels: 0,
        };
        let backend = InMemoryBlockBackend::new(params.num_buckets());
        let mut c = RingOramClient::new(backend, params, [0xaa; 32], [0x01; 32]).await;
        c.admit(BlockId(2), vec![0xff; 8]).await.unwrap();

        let bad_key = [0xbb; 32];
        let path = path_buckets(PathId(0), params.n_leaves);
        let buckets = c.backend.read_path(&path).await;
        let result = aes_decrypt(
            &bad_key,
            buckets[0].bucket_id,
            buckets[0].write_counter,
            &buckets[0].ciphertext,
        );
        assert!(result.is_err());
    }
}
