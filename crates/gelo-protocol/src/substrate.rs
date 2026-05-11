use anyhow::Result;
use ndarray::{Array2, ArrayView2};

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
}

/// The trusted side of the split protocol.
///
/// Implementations own the mask RNG, perform the `A·H` / `Aᵀ·(U·W)`
/// dance, and decide which projections to offload vs. run locally
/// (e.g. for the sensitive first/last layers per GELO §3.2).
pub trait TrustedExecutor {
    /// Hand a public weight to the offload engine. Called at model load.
    fn provision_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()>;

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
}
