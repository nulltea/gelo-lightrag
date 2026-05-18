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
//! Layout: one `(max_cache_len, kv_dim)` block per layer, with a `len`
//! counter tracking how many of those rows are populated. Decode appends
//! one row per step; prefill writes `n_prompt` rows at once. Reading
//! returns the valid prefix as an `ArrayView2`.
//!
//! Indexing is `(layer_idx, position)`. The `(layer_idx, head_idx)`
//! grouping called out in `path-1-gelo-gemma.md` M1.0 is implicit —
//! `kv_dim = num_kv_heads × head_dim` and per-head slicing happens at
//! the attention call site, same as the existing prefill path.

use anyhow::{Result, anyhow};
use ndarray::{Array2, ArrayView2, s};

/// One transformer layer's K and V cache. K and V are kept as separate
/// tensors here even for Gemma 4 global layers (K=V tying); the
/// `kv_shared` optimisation in M1.4 will collapse them into a single
/// backing store at that point.
pub struct LayerKvCache {
    /// `(max_cache_len, kv_dim)`. Rows `0..len` are valid.
    k: Array2<f32>,
    /// `(max_cache_len, kv_dim)`. Rows `0..len` are valid.
    v: Array2<f32>,
    /// Number of currently-populated rows.
    len: usize,
}

impl LayerKvCache {
    fn new(max_cache_len: usize, kv_dim: usize) -> Self {
        Self {
            k: Array2::<f32>::zeros((max_cache_len, kv_dim)),
            v: Array2::<f32>::zeros((max_cache_len, kv_dim)),
            len: 0,
        }
    }

    /// Valid prefix length (number of cached positions).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read the valid prefix of K and V as views into the backing store.
    pub fn view(&self) -> (ArrayView2<'_, f32>, ArrayView2<'_, f32>) {
        (
            self.k.slice(s![..self.len, ..]),
            self.v.slice(s![..self.len, ..]),
        )
    }
}

/// KV cache for an entire decoder, one entry per layer.
///
/// Construct with `KvCache::new(num_layers, max_cache_len, kv_dim)`.
/// All layers share the same `(max_cache_len, kv_dim)` shape; the same
/// number of positions is appended to every layer per forward.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    max_cache_len: usize,
    kv_dim: usize,
}

impl KvCache {
    /// Allocate `num_layers` empty caches, each pre-sized to
    /// `max_cache_len` rows. `kv_dim = num_kv_heads * head_dim`.
    pub fn new(num_layers: usize, max_cache_len: usize, kv_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new(max_cache_len, kv_dim))
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

    /// Append `new_k.nrows()` positions to layer `li`. The caller must
    /// call this in layer order during a forward pass so layers stay
    /// in sync.
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
        layer
            .k
            .slice_mut(s![layer.len..new_len, ..])
            .assign(&new_k);
        layer
            .v
            .slice_mut(s![layer.len..new_len, ..])
            .assign(&new_v);
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

        // Append one more position to both layers.
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
}
