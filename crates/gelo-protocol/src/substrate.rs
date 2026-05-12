use std::sync::Arc;

use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};

/// Identifies a specific projection weight inside a transformer encoder.
/// The trusted side uses this both to address weights on the GPU and to
/// decide whether a given matmul should be offloaded or run locally.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct WeightHandle {
    pub layer: u16,
    pub kind: WeightKind,
}

impl WeightHandle {
    pub const fn new(layer: u16, kind: WeightKind) -> Self {
        Self { layer, kind }
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum WeightKind {
    Q,
    K,
    V,
    O,
    FfnUp,
    FfnDown,
}

/// The untrusted accelerator side of the split protocol.
///
/// All implementations must accept activations through [`Self::matmul`] in
/// whatever form the trusted side supplies them — masked, plaintext, or
/// otherwise. The engine has no notion of correctness verification; integrity
/// is the [`TrustedExecutor`]'s responsibility.
pub trait GpuOffloadEngine: Send {
    /// Provision a public weight tensor to the offload engine. Called once at
    /// model load. Shape is `(in_features, out_features)`.
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()>;

    /// Compute `input · W[handle]` and return the product.
    ///
    /// `input` has shape `(n, in_features)`; the result has shape
    /// `(n, out_features)`. The engine treats `input` as opaque bytes —
    /// masking is applied by the trusted side before the call.
    fn matmul(&self, handle: WeightHandle, input: ArrayView2<f32>) -> Result<Array2<f32>>;

    /// Two-operand dynamic matmul where neither operand is a pre-registered
    /// weight. Required by OutAttnMult (TwinShield §V-A): both `Q` and `Kᵀ`
    /// are runtime values, so neither side can be uploaded ahead of time.
    ///
    /// `lhs` has shape `(m, k)`, `rhs` has shape `(k, n)`, result is `(m, n)`.
    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<f32>,
        rhs: ArrayView2<f32>,
    ) -> Result<Array2<f32>>;

    /// Batched two-operand dynamic matmul. `lhs` is `(B, M, K)`, `rhs` is
    /// `(B, K, N)`, result is `(B, M, N)`. Each batch element is an
    /// independent GEMM — no cross-batch reduction. Used by OutAttnMult to
    /// fuse all Q-heads of a layer into one GPU dispatch.
    ///
    /// Default impl loops over the batch axis calling `matmul_dynamic`, so
    /// engines that haven't been upgraded still produce the right answer
    /// (just without the dispatch saving). Wgpu/cubecl backends override
    /// with one batched launch.
    fn matmul_dynamic_batched(
        &self,
        lhs: ArrayView3<f32>,
        rhs: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        let b = lhs.shape()[0];
        let m = lhs.shape()[1];
        let k = lhs.shape()[2];
        if rhs.shape()[0] != b || rhs.shape()[1] != k {
            return Err(anyhow::anyhow!(
                "matmul_dynamic_batched shape mismatch: lhs ({b},{m},{k}) vs rhs {:?}",
                rhs.shape()
            ));
        }
        let n = rhs.shape()[2];
        let mut out = Array3::<f32>::zeros((b, m, n));
        for i in 0..b {
            let r = self.matmul_dynamic(
                lhs.index_axis(Axis(0), i),
                rhs.index_axis(Axis(0), i),
            )?;
            out.index_axis_mut(Axis(0), i).assign(&r);
        }
        Ok(out)
    }
}

/// The trusted side of the split protocol.
///
/// Implementations own the mask RNG, perform the `A·H` / `Aᵀ·(U·W)`
/// dance, and decide which projections to offload vs. run locally
/// (e.g. for the sensitive first/last layers per GELO §3.2).
pub trait TrustedExecutor {
    /// Hand a public weight to the offload engine. Called at model load.
    fn provision_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()>;

    /// Same as [`Self::provision_weight`] but accepts an `Arc<Array2<f32>>` so
    /// the executor's TEE-side weight cache (for U-Verify probe computation)
    /// can share storage with the embedder's loaded weight bytes instead of
    /// cloning them. GELO targets openweight models — weight confidentiality
    /// is not a goal — so the only reason the executor used to keep a
    /// separate copy was to have the bytes in encrypted CVM RAM. With this
    /// API the embedder's existing `Arc<DecoderWeights>` shards are reused
    /// directly, halving the encrypted memory footprint on Qwen3-class
    /// models (−2.4 GB).
    ///
    /// Default impl falls back to the cloning path so existing callers keep
    /// working unchanged.
    fn provision_weight_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<f32>>,
    ) -> Result<()> {
        self.provision_weight(handle, weight.view())
    }

    /// Run a single offloaded linear: mask `hidden` on the token axis, ship
    /// to the engine, unmask, return `hidden · W[handle]`.
    ///
    /// `hidden` shape is `(n, in_features)`; result shape is
    /// `(n, out_features)`.
    fn offload_linear(
        &mut self,
        handle: WeightHandle,
        hidden: ArrayView2<f32>,
    ) -> Result<Array2<f32>>;

    /// Offload Q, K, V projections in one shot, sharing a single fresh mask
    /// across all three. This is the optimization called out in GELO §3:
    /// reusing `A` across the three reads of the same hidden state saves
    /// two mask samples per block without leaking additional information.
    fn offload_qkv(
        &mut self,
        layer: u16,
        hidden: ArrayView2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let q = self.offload_linear(WeightHandle::new(layer, WeightKind::Q), hidden)?;
        let k = self.offload_linear(WeightHandle::new(layer, WeightKind::K), hidden)?;
        let v = self.offload_linear(WeightHandle::new(layer, WeightKind::V), hidden)?;
        Ok((q, k, v))
    }

    /// Offload the attention `Q · Kᵀ` matmul via the TwinShield OutAttnMult
    /// 4-partition embedding (Xue et al. 2025 §V-A). Both `q` and `kt` are
    /// runtime values; the trick lets the untrusted engine compute the
    /// product without recovering either operand.
    ///
    /// `q` shape: `(n, d_head)`, `kt` shape: `(d_head, n)`, result `(n, n)`.
    ///
    /// Default impl just calls the engine's `matmul_dynamic` directly with
    /// no masking, suitable for `PlaintextExecutor` parity baselines.
    fn offload_attention_qkt(
        &mut self,
        _q: ArrayView2<f32>,
        _kt: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        unimplemented!("offload_attention_qkt not implemented for this executor")
    }

    /// Batched OutAttnMult — one GPU dispatch covering every Q head in a
    /// layer. `q` is `(num_q_heads, n, d_head)`, `kt` is
    /// `(num_q_heads, d_head, n)` (with K already repeated to match Q heads
    /// for GQA), result is `(num_q_heads, n, n)`.
    ///
    /// Each head gets independent masks, scalars, and permutations — the
    /// privacy story stays identical to the per-head form. Default impl
    /// loops over the batch axis calling `offload_attention_qkt`; engines
    /// implementing the batched engine primitive will override.
    fn offload_attention_qkt_batched(
        &mut self,
        q: ArrayView3<f32>,
        kt: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        let h = q.shape()[0];
        let n = q.shape()[1];
        let mut out = Array3::<f32>::zeros((h, n, n));
        for i in 0..h {
            let r = self.offload_attention_qkt(
                q.index_axis(Axis(0), i),
                kt.index_axis(Axis(0), i),
            )?;
            out.index_axis_mut(Axis(0), i).assign(&r);
        }
        Ok(out)
    }
}
