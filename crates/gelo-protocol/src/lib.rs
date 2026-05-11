//! GELO-style TEE+untrusted-GPU split inference protocol.
//!
//! Implements the per-batch fresh invertible mask described in Belikov &
//! Fedotov (arXiv 2603.05035). The trusted side samples an orthogonal token-axis
//! matrix `A`, computes `U = A·H`, ships `(U, W_handle)` to an untrusted
//! offload engine, then recovers `H·W = Aᵀ·(U·W)` on return.
//!
//! The crate is substrate-agnostic: the trusted side is any type that
//! implements [`TrustedExecutor`] and the offload side is any type that
//! implements [`GpuOffloadEngine`]. A reference in-process simulation
//! (`InProcessTrustedExecutor` + `RayonCpuEngine`) lives in [`sim`].

pub mod integrity;
pub mod mask;
pub mod out_attn_mult;
pub mod rng;
pub mod shield;
pub mod sim;
pub mod substrate;

pub use mask::{GeloMask, MaskSeed};
pub use shield::ShieldConfig;
pub use sim::{InProcessTrustedExecutor, PlaintextExecutor, RayonCpuEngine};
pub use substrate::{GpuOffloadEngine, TrustedExecutor, WeightHandle, WeightKind};
