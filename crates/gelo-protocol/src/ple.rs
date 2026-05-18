//! Per-Layer Embedding (PLE) table — TEE-resident lookup.
//!
//! Gemma 3n / Gemma 4 E-variants ship an additional embedding table
//! of shape `[vocab_size × num_layers × d_ple]` that's indexed by
//! `token_id` (NOT by position) at each layer. The token-id-keyed
//! gather is the P0 leak called out in `docs/prototype/gelo-llm.html`
//! §03: if the table sat on the untrusted GPU, an adversary observing
//! gather addresses would recover the prompt **in plaintext** — no
//! inversion needed. The fix is structural: the table lives in
//! encrypted CVM DRAM (i.e. owned by the `TrustedExecutor`), the
//! gather happens in-process, and only the gathered f32 tensor
//! crosses subsequent protocol boundaries (under the same per-batch
//! mask `A` as the rest of the activation stream).
//!
//! For E2B the table is `[262_144 × 35 × 256]` int8 = ~2.3 GB
//! (700 MB if we pack each layer slice separately); E4B is similar
//! at 42 layers. Both fit comfortably in a typical 32-64 GB SEV-SNP
//! CVM, with room left for the model weights and KV cache.
//!
//! This module is intentionally pure data — no protocol state, no
//! mask state. The substrate-level wiring (provisioning into a
//! `TrustedExecutor`, gathering during a forward pass) lives in
//! `crate::substrate`.

use anyhow::{Result, anyhow};
use ndarray::Array2;
use std::sync::Arc;

/// Per-layer embedding table.
///
/// Layout: one `Vec<i8>` per layer, length `vocab_size × d_ple`. Row
/// `t` of layer `li` lives at `data[li][t * d_ple .. (t + 1) * d_ple]`.
/// int8 storage matches Gemma's published checkpoint dtype;
/// dequantisation to f32 happens at gather time inside the TEE so the
/// downstream consumer sees a normal `(n, d_ple)` f32 tensor.
///
/// `Arc<i8>` storage lets multiple `TrustedExecutor` clones (e.g. the
/// rayon-parallel embed path) share the table without copying its
/// multi-hundred-MB backing buffers.
///
/// Quantisation scale: the table is quantised symmetrically as
/// `i8 = round(f32 * scale)`. Dequant: `f32 = i8 / scale`. The scale
/// is per-table (not per-layer or per-row) for v1 simplicity; M1.8
/// validation will check that this matches the official quantisation
/// recipe and may move to per-channel scales if accuracy requires it.
#[derive(Clone, Debug)]
pub struct PleTable {
    /// Per-layer raw int8 rows. `data[li]` has length `vocab_size * d_ple`.
    data: Vec<Arc<[i8]>>,
    vocab_size: usize,
    d_ple: usize,
    /// Symmetric dequant scale: `f32 = i8 as f32 / scale`. A scale of
    /// `127.0` maps the int8 range `[-127, 127]` to roughly `[-1, 1]`.
    scale: f32,
}

impl PleTable {
    /// Construct from a raw `[vocab × layers × d_ple]` int8 cube laid
    /// out as `Vec<Vec<i8>>` where the outer index is `layer_idx` and
    /// the inner buffer is the row-major `vocab × d_ple` slab.
    pub fn from_int8_rows(
        per_layer_data: Vec<Vec<i8>>,
        vocab_size: usize,
        d_ple: usize,
        scale: f32,
    ) -> Result<Self> {
        if per_layer_data.is_empty() {
            return Err(anyhow!("PleTable: must have at least one layer"));
        }
        if scale <= 0.0 {
            return Err(anyhow!("PleTable: scale must be > 0, got {scale}"));
        }
        let expected = vocab_size
            .checked_mul(d_ple)
            .ok_or_else(|| anyhow!("PleTable: vocab_size * d_ple overflow"))?;
        for (li, slab) in per_layer_data.iter().enumerate() {
            if slab.len() != expected {
                return Err(anyhow!(
                    "PleTable: layer {li} has {} bytes, expected vocab_size * d_ple = {}",
                    slab.len(),
                    expected,
                ));
            }
        }
        let data = per_layer_data
            .into_iter()
            .map(|v| Arc::from(v.into_boxed_slice()))
            .collect();
        Ok(Self {
            data,
            vocab_size,
            d_ple,
            scale,
        })
    }

    /// Number of layers covered by this PLE table.
    pub fn num_layers(&self) -> usize {
        self.data.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn d_ple(&self) -> usize {
        self.d_ple
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Approximate memory footprint of the table's backing buffers.
    /// Excludes the per-`Arc` overhead. Used by callers that want to
    /// log "PLE table size = X MB" at provisioning time.
    pub fn bytes(&self) -> usize {
        self.data.iter().map(|a| a.len()).sum()
    }

    /// Gather `(n, d_ple)` f32 rows from layer `layer_idx`, one row per
    /// `token_id`. int8 → f32 dequant via `scale` happens here.
    ///
    /// Out-of-vocabulary token ids and out-of-range layer indices are
    /// rejected with an error rather than silently truncating — the
    /// prototype targets attestation-pinned tokenizers so an oob token
    /// is a real bug, not normal traffic.
    pub fn gather(&self, token_ids: &[u32], layer_idx: usize) -> Result<Array2<f32>> {
        let layer = self
            .data
            .get(layer_idx)
            .ok_or_else(|| anyhow!("PleTable: layer_idx {layer_idx} out of range (have {})", self.data.len()))?;
        let inv_scale = 1.0_f32 / self.scale;
        let n = token_ids.len();
        let d = self.d_ple;
        let mut out = Array2::<f32>::zeros((n, d));
        for (i, &tid) in token_ids.iter().enumerate() {
            let t = tid as usize;
            if t >= self.vocab_size {
                return Err(anyhow!(
                    "PleTable: token_id {tid} out of vocab range (vocab_size = {})",
                    self.vocab_size,
                ));
            }
            let start = t * d;
            let row = &layer[start..start + d];
            let mut dst = out.row_mut(i);
            let dst_slice = dst
                .as_slice_mut()
                .expect("Array2 row is contiguous in row-major");
            for (j, &b) in row.iter().enumerate() {
                dst_slice[j] = b as f32 * inv_scale;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_layer_table() -> PleTable {
        // 4 tokens, 3 layers, d_ple = 2. Make the row contents
        // distinguishable: row t at layer li has values
        // (li * 10 + t * 2, li * 10 + t * 2 + 1).
        let vocab = 4;
        let d = 2;
        let mut layers = Vec::new();
        for li in 0..3 {
            let mut slab = Vec::with_capacity(vocab * d);
            for t in 0..vocab {
                slab.push((li * 10 + t * 2) as i8);
                slab.push((li * 10 + t * 2 + 1) as i8);
            }
            layers.push(slab);
        }
        PleTable::from_int8_rows(layers, vocab, d, 1.0).unwrap()
    }

    #[test]
    fn shape_metadata_round_trip() {
        let table = three_layer_table();
        assert_eq!(table.num_layers(), 3);
        assert_eq!(table.vocab_size(), 4);
        assert_eq!(table.d_ple(), 2);
        assert_eq!(table.scale(), 1.0);
        assert_eq!(table.bytes(), 3 * 4 * 2);
    }

    #[test]
    fn gather_returns_dequantised_rows() {
        let table = three_layer_table();
        // Layer 0, tokens [0, 2]:
        //   t=0 → (0, 1); t=2 → (4, 5)
        let out = table.gather(&[0, 2], 0).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out[[0, 0]], 0.0);
        assert_eq!(out[[0, 1]], 1.0);
        assert_eq!(out[[1, 0]], 4.0);
        assert_eq!(out[[1, 1]], 5.0);

        // Layer 2, tokens [1, 3]:
        //   t=1 → (22, 23); t=3 → (26, 27)
        let out = table.gather(&[1, 3], 2).unwrap();
        assert_eq!(out[[0, 0]], 22.0);
        assert_eq!(out[[1, 1]], 27.0);
    }

    #[test]
    fn nonidentity_scale_dequantises_correctly() {
        let vocab = 2;
        let d = 2;
        let layer0 = vec![10_i8, -20, 30, -40]; // rows: [10, -20], [30, -40]
        let table = PleTable::from_int8_rows(vec![layer0], vocab, d, 10.0).unwrap();
        let out = table.gather(&[0, 1], 0).unwrap();
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], -2.0);
        assert_eq!(out[[1, 0]], 3.0);
        assert_eq!(out[[1, 1]], -4.0);
    }

    #[test]
    fn layer_out_of_range_errors() {
        let table = three_layer_table();
        let err = table.gather(&[0], 99).unwrap_err();
        assert!(err.to_string().contains("layer_idx 99"));
    }

    #[test]
    fn token_out_of_vocab_errors() {
        let table = three_layer_table();
        let err = table.gather(&[99], 0).unwrap_err();
        assert!(err.to_string().contains("out of vocab"));
    }

    #[test]
    fn zero_scale_rejected() {
        let err = PleTable::from_int8_rows(vec![vec![0_i8; 4]], 2, 2, 0.0).unwrap_err();
        assert!(err.to_string().contains("scale"));
    }

    #[test]
    fn empty_layers_rejected() {
        let err = PleTable::from_int8_rows(Vec::<Vec<i8>>::new(), 2, 2, 1.0).unwrap_err();
        assert!(err.to_string().contains("at least one layer"));
    }

    #[test]
    fn malformed_slab_size_rejected() {
        let err = PleTable::from_int8_rows(vec![vec![0_i8; 3]], 2, 2, 1.0).unwrap_err();
        assert!(err.to_string().contains("expected"));
    }
}
