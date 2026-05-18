//! In-CVM KV cache for autoregressive decode.
//!
//! Per the generation prototype doc (`docs/prototype/gelo-llm.html` §06):
//! the KV cache lives entirely in encrypted CVM DRAM. The GPU never
//! touches cached K/V bytes — attention's `Q · Kᵀ_cache` math at decode
//! stays in-TEE (the per-step compute is microseconds-scale at
//! `n_q = 1`). The cache is a plain `ndarray::Array2` allocated on the
//! Rust heap inside the trusted process; encryption is the SEV-SNP
//! page-level RMP, not anything this struct manages.
//!
//! Layout: one storage slot per layer, with a `len` counter tracking
//! how many of its rows are populated. The slot is either:
//!
//! - [`LayerKvStore::Separate`] — distinct `K` and `V` arrays, the
//!   default for Qwen3 / Llama and Gemma 4 local layers.
//! - [`LayerKvStore::Shared`] — a single `KV` array used for both K
//!   and V, the M1.4 optimisation for Gemma 4 global layers where
//!   the trained model satisfies `W_k = W_v` (the "K equals V" trick).
//!   Halves the per-layer cache footprint.
//!
//! Indexing is `(layer_idx, position)`. The `(layer_idx, head_idx)`
//! grouping is implicit — `kv_dim = num_kv_heads × head_dim` and
//! per-head slicing happens at the attention call site.

use anyhow::{Result, anyhow};
use ndarray::{Array2, ArrayView2, s};

/// Backing storage for one layer's K/V.
enum LayerKvStore {
    Separate {
        k: Array2<f32>,
        v: Array2<f32>,
    },
    Shared {
        kv: Array2<f32>,
    },
}

impl LayerKvStore {
    fn separate(max_cache_len: usize, kv_dim: usize) -> Self {
        Self::Separate {
            k: Array2::<f32>::zeros((max_cache_len, kv_dim)),
            v: Array2::<f32>::zeros((max_cache_len, kv_dim)),
        }
    }
    fn shared(max_cache_len: usize, kv_dim: usize) -> Self {
        Self::Shared {
            kv: Array2::<f32>::zeros((max_cache_len, kv_dim)),
        }
    }
}

/// One transformer layer's K and V cache.
pub struct LayerKvCache {
    storage: LayerKvStore,
    /// Number of currently-populated rows. Same value for K and V
    /// regardless of storage layout.
    len: usize,
}

impl LayerKvCache {
    fn new_separate(max_cache_len: usize, kv_dim: usize) -> Self {
        Self {
            storage: LayerKvStore::separate(max_cache_len, kv_dim),
            len: 0,
        }
    }

    fn new_shared(max_cache_len: usize, kv_dim: usize) -> Self {
        Self {
            storage: LayerKvStore::shared(max_cache_len, kv_dim),
            len: 0,
        }
    }

    /// True iff this layer uses the K=V shared-storage optimisation.
    pub fn is_shared(&self) -> bool {
        matches!(self.storage, LayerKvStore::Shared { .. })
    }

    /// Valid prefix length (number of cached positions).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read the valid prefix of K and V as views into the backing
    /// store. For shared layers, both views alias the same buffer.
    pub fn view(&self) -> (ArrayView2<'_, f32>, ArrayView2<'_, f32>) {
        match &self.storage {
            LayerKvStore::Separate { k, v } => (
                k.slice(s![..self.len, ..]),
                v.slice(s![..self.len, ..]),
            ),
            LayerKvStore::Shared { kv } => (
                kv.slice(s![..self.len, ..]),
                kv.slice(s![..self.len, ..]),
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
/// Construct with [`KvCache::new`] for an all-separate cache (the
/// existing Qwen3 / Llama path) or [`KvCache::new_with_sharing`] for
/// Gemma 4 hybrid models that tie K and V on global layers.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    max_cache_len: usize,
    kv_dim: usize,
}

impl KvCache {
    /// Allocate `num_layers` empty caches with separate K and V
    /// storage per layer (no K=V tying). Pre-sized to `max_cache_len`
    /// rows. `kv_dim = num_kv_heads × head_dim`.
    pub fn new(num_layers: usize, max_cache_len: usize, kv_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new_separate(max_cache_len, kv_dim))
            .collect();
        Self {
            layers,
            max_cache_len,
            kv_dim,
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
                    LayerKvCache::new_shared(max_cache_len, kv_dim)
                } else {
                    LayerKvCache::new_separate(max_cache_len, kv_dim)
                }
            })
            .collect();
        Self {
            layers,
            max_cache_len,
            kv_dim,
        }
    }

    /// Number of decoder layers covered by this cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Pre-allocated capacity in positions per layer.
    pub fn capacity(&self) -> usize {
        self.max_cache_len
    }

    /// Width of one cached row.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Current valid length. All layers grow in lockstep — this returns
    /// layer 0's length; debug builds assert all layers agree.
    pub fn len(&self) -> usize {
        let len = self.layers.first().map(|l| l.len).unwrap_or(0);
        debug_assert!(
            self.layers.iter().all(|l| l.len == len),
            "KvCache layers got out of sync — all layers must grow together",
        );
        len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Append `new_k.nrows()` positions to layer `li`. For Separate
    /// layers `new_k` and `new_v` may differ; for Shared layers they
    /// must point to identical content (caller's responsibility — the
    /// K=V branch in `decoder_block_cached` computes one tensor and
    /// passes it for both arguments). When shared, only `new_k` is
    /// actually written.
    pub fn append(
        &mut self,
        li: usize,
        new_k: ArrayView2<'_, f32>,
        new_v: ArrayView2<'_, f32>,
    ) -> Result<()> {
        let layer = self
            .layers
            .get_mut(li)
            .ok_or_else(|| anyhow!("KvCache layer index {li} out of range"))?;
        let added = new_k.nrows();
        if new_v.nrows() != added {
            return Err(anyhow!(
                "KvCache append: K rows {} != V rows {}",
                added,
                new_v.nrows()
            ));
        }
        if new_k.ncols() != self.kv_dim {
            return Err(anyhow!(
                "KvCache append: K width {} != kv_dim {}",
                new_k.ncols(),
                self.kv_dim
            ));
        }
        if new_v.ncols() != self.kv_dim {
            return Err(anyhow!(
                "KvCache append: V width {} != kv_dim {}",
                new_v.ncols(),
                self.kv_dim
            ));
        }
        let new_len = layer.len + added;
        if new_len > self.max_cache_len {
            return Err(anyhow!(
                "KvCache overflow: layer {li} cur_len {} + {} > max_cache_len {}",
                layer.len,
                added,
                self.max_cache_len,
            ));
        }
        match &mut layer.storage {
            LayerKvStore::Separate { k, v } => {
                k.slice_mut(s![layer.len..new_len, ..]).assign(&new_k);
                v.slice_mut(s![layer.len..new_len, ..]).assign(&new_v);
            }
            LayerKvStore::Shared { kv } => {
                // Caller must guarantee K == V at the call site.
                // Debug asserts catch misuse.
                debug_assert!(
                    arrays_equal(&new_k, &new_v),
                    "KvCache: layer {li} is K=V shared but append got K != V",
                );
                kv.slice_mut(s![layer.len..new_len, ..]).assign(&new_k);
            }
        }
        layer.len = new_len;
        Ok(())
    }

    /// View the valid prefix of layer `li`'s K and V.
    pub fn view(&self, li: usize) -> Result<(ArrayView2<'_, f32>, ArrayView2<'_, f32>)> {
        let layer = self
            .layers
            .get(li)
            .ok_or_else(|| anyhow!("KvCache layer index {li} out of range"))?;
        Ok(layer.view())
    }

    /// Drop all cached state. Layers stay allocated; only `len` resets.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.len = 0;
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
        // 4 layers, 2 shared. Each layer is max_cache_len × kv_dim × 4
        // bytes; shared layers store one of those, separate store two.
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
        // 75% of all-separate is the spec.
        assert_eq!(cache.bytes() * 4, all_sep.bytes() * 3);
    }

    #[test]
    fn shared_layer_round_trip_view_aliases() {
        // Shared layer: appending K = V once must produce K view == V view.
        let mut cache = KvCache::new_with_sharing(1, 4, 2, &[true]);
        let kv = array![[1.0_f32, 2.0], [3.0_f32, 4.0]];
        cache.append(0, kv.view(), kv.view()).unwrap();
        assert!(cache.is_layer_shared(0));
        let (k_view, v_view) = cache.view(0).unwrap();
        assert_eq!(k_view, v_view);
        assert_eq!(k_view.nrows(), 2);
        assert_eq!(k_view[[1, 1]], 4.0);
    }
}
