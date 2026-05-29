//! In-CVM KV cache for autoregressive decode.
//!
//! Per the generation prototype doc (`docs/prototype/gelo-llm.html` §06):
//! the KV cache lives entirely in encrypted CVM DRAM. The GPU never
//! touches cached K/V bytes — attention's `Q · Kᵀ_cache` math at decode
//! stays in-TEE (the per-step compute is microseconds-scale at
//! `n_q = 1`). The cache is a plain `ndarray::Array3` allocated on the
//! Rust heap inside the trusted process; encryption is the SEV-SNP
//! page-level RMP, not anything this struct manages.
//!
//! **M1.11 D1.2** — layout is `(B, max_cache_len, kv_dim)` per layer
//! and a per-layer `Vec<usize>` of per-sequence valid lengths. Single-
//! sequence callers (`KvCache::new`) get the degenerate `B = 1` case
//! and the legacy `.append() / .view() / .len()` API still works.
//!
//! Layout per layer:
//!
//! - [`LayerKvStore::Separate`] — distinct `K` and `V` arrays of
//!   shape `(B, max_cache_len, kv_dim)`, the default for Qwen3 /
//!   Llama and Gemma 4 local layers.
//! - [`LayerKvStore::Shared`] — a single `KV` array used for both K
//!   and V, the M1.4 optimisation for Gemma 4 global layers where
//!   the trained model satisfies `W_k = W_v` (the "K equals V" trick).
//!   Halves the per-layer cache footprint.
//!
//! Indexing is `(layer_idx, batch_idx, position)`. The
//! `(layer_idx, head_idx)` grouping is implicit — `kv_dim =
//! num_kv_heads × head_dim` and per-head slicing happens at the
//! attention call site.

use anyhow::{Result, anyhow};
use ndarray::{Array3, ArrayView2, s};

/// Backing storage for one layer's K/V.
enum LayerKvStore {
    Separate {
        /// Shape `(B, max_cache_len, kv_dim)`.
        k: Array3<f32>,
        v: Array3<f32>,
    },
    Shared {
        /// Shape `(B, max_cache_len, kv_dim)`.
        kv: Array3<f32>,
    },
}

impl LayerKvStore {
    fn separate(batch_size: usize, max_cache_len: usize, kv_dim: usize) -> Self {
        Self::Separate {
            k: Array3::<f32>::zeros((batch_size, max_cache_len, kv_dim)),
            v: Array3::<f32>::zeros((batch_size, max_cache_len, kv_dim)),
        }
    }
    fn shared(batch_size: usize, max_cache_len: usize, kv_dim: usize) -> Self {
        Self::Shared {
            kv: Array3::<f32>::zeros((batch_size, max_cache_len, kv_dim)),
        }
    }
}

/// One transformer layer's K and V cache.
pub struct LayerKvCache {
    storage: LayerKvStore,
    /// Per-sequence valid-prefix lengths. `lens.len() == batch_size`.
    /// All sequences share the same `max_cache_len` capacity (the
    /// outer `KvCache::max_cache_len`).
    lens: Vec<usize>,
}

impl LayerKvCache {
    fn new_separate(batch_size: usize, max_cache_len: usize, kv_dim: usize) -> Self {
        Self {
            storage: LayerKvStore::separate(batch_size, max_cache_len, kv_dim),
            lens: vec![0; batch_size],
        }
    }

    fn new_shared(batch_size: usize, max_cache_len: usize, kv_dim: usize) -> Self {
        Self {
            storage: LayerKvStore::shared(batch_size, max_cache_len, kv_dim),
            lens: vec![0; batch_size],
        }
    }

    /// True iff this layer uses the K=V shared-storage optimisation.
    pub fn is_shared(&self) -> bool {
        matches!(self.storage, LayerKvStore::Shared { .. })
    }

    /// Per-sequence valid prefix lengths.
    pub fn lens(&self) -> &[usize] {
        &self.lens
    }

    /// Valid prefix length of sequence `b`.
    pub fn len_b(&self, b: usize) -> usize {
        self.lens[b]
    }

    /// Read the valid prefix of sequence `b`'s K and V as views into
    /// the backing store. For shared layers, both views alias the
    /// same buffer.
    pub fn view_b(&self, b: usize) -> (ArrayView2<'_, f32>, ArrayView2<'_, f32>) {
        let len = self.lens[b];
        match &self.storage {
            LayerKvStore::Separate { k, v } => (
                k.slice(s![b, ..len, ..]),
                v.slice(s![b, ..len, ..]),
            ),
            LayerKvStore::Shared { kv } => (
                kv.slice(s![b, ..len, ..]),
                kv.slice(s![b, ..len, ..]),
            ),
        }
    }

    /// Byte footprint of this layer's backing tensor(s). Used by the
    /// memory-savings tests to confirm shared layers are roughly half
    /// the size of separate ones.
    pub fn bytes(&self) -> usize {
        match &self.storage {
            LayerKvStore::Separate { k, v } => k.len() * 4 + v.len() * 4,
            LayerKvStore::Shared { kv } => kv.len() * 4,
        }
    }
}

/// KV cache for an entire decoder, one entry per layer.
///
/// Construct with [`KvCache::new`] for an all-separate single-sequence
/// cache, [`KvCache::new_batched`] for a B-sequence batched cache, or
/// [`KvCache::new_with_sharing`] for Gemma 4 hybrid models that tie K
/// and V on global layers.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    max_cache_len: usize,
    kv_dim: usize,
    batch_size: usize,
    /// GPU-resident attention session id per layer (perm-attn-gpu-offload
    /// Phase 4 perf wire-up). `None` until the first decode step creates
    /// the session from this layer's cache; per-generation lifecycle
    /// matches the `KvCache` (fresh `None`s per generation → no stale
    /// session reuse). Only used when the GPU-resident decode path is on.
    gpu_sessions: Vec<Option<u64>>,
}

impl KvCache {
    /// Allocate `num_layers` empty caches with separate K and V
    /// storage per layer (no K=V tying). Pre-sized to `max_cache_len`
    /// rows. `kv_dim = num_kv_heads × head_dim`.
    ///
    /// Single-sequence — equivalent to `new_batched(1, ...)`. The
    /// legacy `.len()` / `.append()` / `.view()` API is supported on
    /// the returned cache.
    pub fn new(num_layers: usize, max_cache_len: usize, kv_dim: usize) -> Self {
        Self::new_batched(1, num_layers, max_cache_len, kv_dim)
    }

    /// Allocate `num_layers` empty caches with separate K and V
    /// storage per layer, batched across `batch_size` sequences. Each
    /// sequence shares the same `max_cache_len` capacity.
    pub fn new_batched(
        batch_size: usize,
        num_layers: usize,
        max_cache_len: usize,
        kv_dim: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new_separate(batch_size, max_cache_len, kv_dim))
            .collect();
        Self {
            layers,
            max_cache_len,
            kv_dim,
            batch_size,
            gpu_sessions: vec![None; num_layers],
        }
    }

    /// Allocate `num_layers` caches, choosing per-layer storage from
    /// the `shared` mask. `shared[li] = true` builds a K=V shared
    /// store for layer `li` (Gemma 4 global layers); false builds a
    /// Separate store. `shared.len()` must equal `num_layers`.
    pub fn new_with_sharing(
        num_layers: usize,
        max_cache_len: usize,
        kv_dim: usize,
        shared: &[bool],
    ) -> Self {
        Self::new_with_sharing_batched(1, num_layers, max_cache_len, kv_dim, shared)
    }

    /// Batched variant of [`Self::new_with_sharing`].
    pub fn new_with_sharing_batched(
        batch_size: usize,
        num_layers: usize,
        max_cache_len: usize,
        kv_dim: usize,
        shared: &[bool],
    ) -> Self {
        assert_eq!(
            shared.len(),
            num_layers,
            "new_with_sharing: shared mask length {} != num_layers {}",
            shared.len(),
            num_layers,
        );
        let layers = (0..num_layers)
            .map(|li| {
                if shared[li] {
                    LayerKvCache::new_shared(batch_size, max_cache_len, kv_dim)
                } else {
                    LayerKvCache::new_separate(batch_size, max_cache_len, kv_dim)
                }
            })
            .collect();
        Self {
            layers,
            max_cache_len,
            kv_dim,
            batch_size,
            gpu_sessions: vec![None; num_layers],
        }
    }

    /// Number of decoder layers covered by this cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Borrow layer `li`'s cache (for reading out the populated K/V after
    /// a prefill — used by the attention-cover capture in
    /// `docs/plans/perm-attn-gpu-offload.md`).
    pub fn layer(&self, li: usize) -> &LayerKvCache {
        &self.layers[li]
    }

    /// GPU-resident attention session id for layer `li` (Phase-4 perf
    /// wire-up), or `None` if not yet created this generation.
    pub fn gpu_session(&self, li: usize) -> Option<u64> {
        self.gpu_sessions[li]
    }

    /// Record the GPU-resident session id created for layer `li`.
    pub fn set_gpu_session(&mut self, li: usize, id: u64) {
        self.gpu_sessions[li] = Some(id);
    }

    /// All live GPU-resident session ids (for end-of-generation cleanup).
    pub fn gpu_session_ids(&self) -> Vec<u64> {
        self.gpu_sessions.iter().flatten().copied().collect()
    }

    /// Pre-allocated capacity in positions per layer.
    pub fn capacity(&self) -> usize {
        self.max_cache_len
    }

    /// Width of one cached row.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Number of sequences in this batched cache. `1` for the
    /// single-sequence default.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// **Legacy** — single-sequence current valid length. Returns
    /// `lens[0]` and asserts batch_size == 1. Use [`Self::lens`] /
    /// [`Self::len_b`] for batched callers.
    pub fn len(&self) -> usize {
        assert_eq!(
            self.batch_size, 1,
            "KvCache::len() is single-sequence only; batched cache has batch_size={}",
            self.batch_size,
        );
        let len = self.layers.first().map(|l| l.lens[0]).unwrap_or(0);
        debug_assert!(
            self.layers.iter().all(|l| l.lens[0] == len),
            "KvCache layers got out of sync — all layers must grow together",
        );
        len
    }

    /// Per-sequence current lens of layer `li`. All layers grow in
    /// lockstep; debug-asserts cross-layer agreement.
    pub fn lens(&self, li: usize) -> &[usize] {
        debug_assert!(
            self.layers.iter().all(|l| l.lens == self.layers[0].lens),
            "KvCache: per-sequence lens diverged across layers",
        );
        self.layers[li].lens()
    }

    /// Per-sequence current len of sequence `b` at layer `li`.
    pub fn len_b(&self, li: usize, b: usize) -> usize {
        self.layers[li].len_b(b)
    }

    pub fn is_empty(&self) -> bool {
        self.layers
            .iter()
            .all(|l| l.lens.iter().all(|&n| n == 0))
    }

    pub fn is_layer_shared(&self, li: usize) -> bool {
        self.layers
            .get(li)
            .map(LayerKvCache::is_shared)
            .unwrap_or(false)
    }

    /// Total byte footprint across all layer caches.
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(LayerKvCache::bytes).sum()
    }

    /// **Legacy single-sequence append** — appends `new_k.nrows()`
    /// positions to layer `li` of sequence 0. Asserts batch_size == 1.
    /// Equivalent to `append_prefill(li, 0, new_k, new_v)`.
    pub fn append(
        &mut self,
        li: usize,
        new_k: ArrayView2<'_, f32>,
        new_v: ArrayView2<'_, f32>,
    ) -> Result<()> {
        assert_eq!(
            self.batch_size, 1,
            "KvCache::append() is single-sequence only; batched cache has batch_size={}",
            self.batch_size,
        );
        self.append_prefill(li, 0, new_k, new_v)
    }

    /// **Batched prefill append** — appends `new_k.nrows()` rows for
    /// sequence `b` at layer `li`, starting at the sequence's
    /// current `lens[b]`. Used by the prefill phase when each
    /// sequence has its own prompt length.
    pub fn append_prefill(
        &mut self,
        li: usize,
        b: usize,
        new_k: ArrayView2<'_, f32>,
        new_v: ArrayView2<'_, f32>,
    ) -> Result<()> {
        let layer = self
            .layers
            .get_mut(li)
            .ok_or_else(|| anyhow!("KvCache layer index {li} out of range"))?;
        if b >= self.batch_size {
            return Err(anyhow!(
                "KvCache append_prefill: batch_idx {b} out of range (batch_size={})",
                self.batch_size
            ));
        }
        let added = new_k.nrows();
        if new_v.nrows() != added {
            return Err(anyhow!(
                "KvCache append_prefill: K rows {} != V rows {}",
                added,
                new_v.nrows()
            ));
        }
        if new_k.ncols() != self.kv_dim {
            return Err(anyhow!(
                "KvCache append_prefill: K width {} != kv_dim {}",
                new_k.ncols(),
                self.kv_dim
            ));
        }
        if new_v.ncols() != self.kv_dim {
            return Err(anyhow!(
                "KvCache append_prefill: V width {} != kv_dim {}",
                new_v.ncols(),
                self.kv_dim
            ));
        }
        let cur = layer.lens[b];
        let new_len = cur + added;
        if new_len > self.max_cache_len {
            return Err(anyhow!(
                "KvCache overflow: layer {li} sequence {b} cur {} + {} > max_cache_len {}",
                cur,
                added,
                self.max_cache_len,
            ));
        }
        match &mut layer.storage {
            LayerKvStore::Separate { k, v } => {
                k.slice_mut(s![b, cur..new_len, ..]).assign(&new_k);
                v.slice_mut(s![b, cur..new_len, ..]).assign(&new_v);
            }
            LayerKvStore::Shared { kv } => {
                debug_assert!(
                    arrays_equal(&new_k, &new_v),
                    "KvCache: layer {li} is K=V shared but append got K != V",
                );
                kv.slice_mut(s![b, cur..new_len, ..]).assign(&new_k);
            }
        }
        layer.lens[b] = new_len;
        Ok(())
    }

    /// **Batched decode append** — appends ONE row per sequence at
    /// layer `li`. `new_k` and `new_v` have shape `(B, kv_dim)`; row
    /// `b` of each is appended at the current `lens[b]` and `lens[b]`
    /// is advanced. Used by the decode-step phase where every
    /// sequence contributes exactly one new token row.
    pub fn append_decode(
        &mut self,
        li: usize,
        new_k: ArrayView2<'_, f32>,
        new_v: ArrayView2<'_, f32>,
    ) -> Result<()> {
        let layer = self
            .layers
            .get_mut(li)
            .ok_or_else(|| anyhow!("KvCache layer index {li} out of range"))?;
        if new_k.nrows() != self.batch_size {
            return Err(anyhow!(
                "KvCache append_decode: K rows {} != batch_size {}",
                new_k.nrows(),
                self.batch_size
            ));
        }
        if new_v.nrows() != self.batch_size {
            return Err(anyhow!(
                "KvCache append_decode: V rows {} != batch_size {}",
                new_v.nrows(),
                self.batch_size
            ));
        }
        if new_k.ncols() != self.kv_dim {
            return Err(anyhow!(
                "KvCache append_decode: K width {} != kv_dim {}",
                new_k.ncols(),
                self.kv_dim
            ));
        }
        if new_v.ncols() != self.kv_dim {
            return Err(anyhow!(
                "KvCache append_decode: V width {} != kv_dim {}",
                new_v.ncols(),
                self.kv_dim
            ));
        }
        for b in 0..self.batch_size {
            let cur = layer.lens[b];
            if cur + 1 > self.max_cache_len {
                return Err(anyhow!(
                    "KvCache overflow: layer {li} sequence {b} cur {} + 1 > max_cache_len {}",
                    cur,
                    self.max_cache_len,
                ));
            }
            let new_k_row = new_k.slice(s![b, ..]);
            let new_v_row = new_v.slice(s![b, ..]);
            match &mut layer.storage {
                LayerKvStore::Separate { k, v } => {
                    k.slice_mut(s![b, cur..cur + 1, ..])
                        .assign(&new_k_row.insert_axis(ndarray::Axis(0)));
                    v.slice_mut(s![b, cur..cur + 1, ..])
                        .assign(&new_v_row.insert_axis(ndarray::Axis(0)));
                }
                LayerKvStore::Shared { kv } => {
                    debug_assert!(
                        (0..self.kv_dim).all(|d| (new_k_row[d] - new_v_row[d]).abs() < 1e-9),
                        "KvCache: layer {li} is K=V shared but decode append got K != V at b={b}",
                    );
                    kv.slice_mut(s![b, cur..cur + 1, ..])
                        .assign(&new_k_row.insert_axis(ndarray::Axis(0)));
                }
            }
            layer.lens[b] = cur + 1;
        }
        Ok(())
    }

    /// **Legacy** — view sequence 0's valid prefix at layer `li`.
    /// Asserts batch_size == 1.
    pub fn view(&self, li: usize) -> Result<(ArrayView2<'_, f32>, ArrayView2<'_, f32>)> {
        assert_eq!(
            self.batch_size, 1,
            "KvCache::view() is single-sequence only; batched cache has batch_size={}",
            self.batch_size,
        );
        self.view_b(li, 0)
    }

    /// View sequence `b`'s valid prefix at layer `li`. Shape
    /// `(lens[b], kv_dim)` for both K and V.
    pub fn view_b(
        &self,
        li: usize,
        b: usize,
    ) -> Result<(ArrayView2<'_, f32>, ArrayView2<'_, f32>)> {
        let layer = self
            .layers
            .get(li)
            .ok_or_else(|| anyhow!("KvCache layer index {li} out of range"))?;
        if b >= self.batch_size {
            return Err(anyhow!(
                "KvCache view_b: batch_idx {b} out of range (batch_size={})",
                self.batch_size,
            ));
        }
        Ok(layer.view_b(b))
    }

    /// Drop all cached state. Layers stay allocated; only `lens` reset.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            for n in layer.lens.iter_mut() {
                *n = 0;
            }
        }
    }
}

#[cfg(debug_assertions)]
fn arrays_equal(a: &ArrayView2<'_, f32>, b: &ArrayView2<'_, f32>) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

#[cfg(not(debug_assertions))]
fn arrays_equal(_a: &ArrayView2<'_, f32>, _b: &ArrayView2<'_, f32>) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn empty_cache_reports_zero_len() {
        let cache = KvCache::new(3, 16, 4);
        assert_eq!(cache.num_layers(), 3);
        assert_eq!(cache.capacity(), 16);
        assert_eq!(cache.kv_dim(), 4);
        assert_eq!(cache.batch_size(), 1);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        let (k, v) = cache.view(0).unwrap();
        assert_eq!(k.nrows(), 0);
        assert_eq!(v.nrows(), 0);
    }

    #[test]
    fn append_then_view_round_trip() {
        let mut cache = KvCache::new(2, 8, 3);
        let k0 = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let v0 = array![[10.0_f32, 20.0, 30.0], [40.0, 50.0, 60.0]];
        cache.append(0, k0.view(), v0.view()).unwrap();
        cache.append(1, k0.view(), v0.view()).unwrap();
        assert_eq!(cache.len(), 2);

        let (k_view, v_view) = cache.view(0).unwrap();
        assert_eq!(k_view, k0.view());
        assert_eq!(v_view, v0.view());

        let k1 = array![[7.0_f32, 8.0, 9.0]];
        let v1 = array![[70.0_f32, 80.0, 90.0]];
        cache.append(0, k1.view(), v1.view()).unwrap();
        cache.append(1, k1.view(), v1.view()).unwrap();
        assert_eq!(cache.len(), 3);
        let (k_view, _) = cache.view(0).unwrap();
        assert_eq!(k_view.row(2), k1.row(0));
    }

    #[test]
    fn reset_zeros_len_keeps_capacity() {
        let mut cache = KvCache::new(1, 4, 2);
        cache
            .append(0, array![[1.0_f32, 2.0]].view(), array![[3.0_f32, 4.0]].view())
            .unwrap();
        assert_eq!(cache.len(), 1);
        cache.reset();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 4);
    }

    #[test]
    fn overflow_returns_err() {
        let mut cache = KvCache::new(1, 2, 1);
        cache
            .append(0, array![[1.0_f32]].view(), array![[2.0_f32]].view())
            .unwrap();
        cache
            .append(0, array![[3.0_f32]].view(), array![[4.0_f32]].view())
            .unwrap();
        let err = cache
            .append(0, array![[5.0_f32]].view(), array![[6.0_f32]].view())
            .unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn shape_mismatch_errors() {
        let mut cache = KvCache::new(1, 4, 3);
        let err = cache
            .append(
                0,
                array![[1.0_f32, 2.0]].view(),
                array![[3.0_f32, 4.0]].view(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("kv_dim"));
    }

    #[test]
    fn layer_index_out_of_range() {
        let mut cache = KvCache::new(2, 4, 1);
        let err = cache
            .append(5, array![[1.0_f32]].view(), array![[2.0_f32]].view())
            .unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn shared_layer_halves_memory_footprint() {
        let max_cache_len = 8;
        let kv_dim = 16;
        let cache = KvCache::new_with_sharing(
            4,
            max_cache_len,
            kv_dim,
            &[false, true, true, false],
        );
        let per_separate = max_cache_len * kv_dim * 4 * 2;
        let per_shared = max_cache_len * kv_dim * 4;
        let expected = 2 * per_separate + 2 * per_shared;
        assert_eq!(cache.bytes(), expected);

        let all_sep = KvCache::new(4, max_cache_len, kv_dim);
        assert_eq!(all_sep.bytes(), 4 * per_separate);
        assert_eq!(cache.bytes() * 4, all_sep.bytes() * 3);
    }

    #[test]
    fn shared_layer_round_trip_view_aliases() {
        let mut cache = KvCache::new_with_sharing(1, 4, 2, &[true]);
        let kv = array![[1.0_f32, 2.0], [3.0_f32, 4.0]];
        cache.append(0, kv.view(), kv.view()).unwrap();
        assert!(cache.is_layer_shared(0));
        let (k_view, v_view) = cache.view(0).unwrap();
        assert_eq!(k_view, v_view);
        assert_eq!(k_view.nrows(), 2);
        assert_eq!(k_view[[1, 1]], 4.0);
    }

    /// **M1.11 D1.2** — batched cache with B=3 sequences, varying
    /// prefill lengths via `append_prefill`, then one `append_decode`
    /// step advancing all sequences by 1.
    #[test]
    fn batched_append_prefill_then_decode_round_trips() {
        let mut cache = KvCache::new_batched(3, 1, 16, 2);
        assert_eq!(cache.batch_size(), 3);

        // Per-sequence prefill: sequences with prompts of length 4, 2, 5.
        let prompts: [Vec<f32>; 3] = [
            (0..4 * 2).map(|i| i as f32).collect(),
            (100..100 + 2 * 2).map(|i| i as f32).collect(),
            (200..200 + 5 * 2).map(|i| i as f32).collect(),
        ];
        let lens = [4, 2, 5];
        for b in 0..3 {
            let arr = ndarray::Array2::from_shape_vec((lens[b], 2), prompts[b].clone()).unwrap();
            cache.append_prefill(0, b, arr.view(), arr.view()).unwrap();
        }
        assert_eq!(cache.lens(0), &[4usize, 2, 5]);

        // Each sequence's view returns its own prefix.
        for b in 0..3 {
            let (k_view, _) = cache.view_b(0, b).unwrap();
            assert_eq!(k_view.nrows(), lens[b]);
            assert_eq!(k_view[[0, 0]], prompts[b][0]);
        }

        // One decode step: shape (B, kv_dim).
        let new_k = array![[42.0_f32, 43.0], [142.0, 143.0], [242.0, 243.0]];
        let new_v = new_k.clone();
        cache.append_decode(0, new_k.view(), new_v.view()).unwrap();
        assert_eq!(cache.lens(0), &[5usize, 3, 6]);

        // Each sequence's tail row matches its decode contribution.
        for b in 0..3 {
            let (k_view, _) = cache.view_b(0, b).unwrap();
            let last = k_view.row(k_view.nrows() - 1);
            assert_eq!(last[0], new_k[[b, 0]]);
            assert_eq!(last[1], new_k[[b, 1]]);
        }
    }

    #[test]
    fn batched_overflow_per_sequence() {
        let mut cache = KvCache::new_batched(2, 1, 2, 1);
        cache
            .append_prefill(0, 0, array![[1.0_f32], [2.0]].view(), array![[1.0_f32], [2.0]].view())
            .unwrap();
        // Sequence 0 is full; decode-append should fail because b=0
        // hits the overflow.
        let nk = array![[9.0_f32], [10.0]];
        let err = cache.append_decode(0, nk.view(), nk.view()).unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn batched_view_out_of_range_rejected() {
        let cache = KvCache::new_batched(2, 1, 4, 1);
        let err = cache.view_b(0, 5).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }
}
