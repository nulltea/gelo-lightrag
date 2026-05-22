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

// Pull in `blas-src` so the linker picks up the BLIS-provided BLAS
// symbols that ndarray's `blas` fast path dispatches to. The `use _`
// idiom forces the unused dep to be linked. No-op without the
// `blas` feature.
#[cfg(feature = "blas")]
use blas_src as _;

#[cfg(feature = "blas")]
pub mod aocl_lpgemm;
pub mod attention;
pub mod dct4;
pub mod gaussian;
pub mod hd3;
pub mod integrity;
pub mod mask;
pub mod out_attn_mult;
pub mod ple;
pub mod profile;
pub mod rng;
pub mod shield;
pub mod sim;
pub mod snapshot;
pub mod substrate;

pub use attention::PermAttnConfig;
pub use dct4::Dct4Mask;
pub use hd3::Hd3Mask;
pub use mask::{
    GeloMask, HD3_AUTO_MAX_PAD_RATIO_DEN, HD3_AUTO_MAX_PAD_RATIO_NUM, MaskFamily, MaskKind,
    MaskSeed, ensure_blis_single_thread, mask_backend_description, resolve_mask_kind_for_shape,
    set_blis_num_threads, tee_matmul, tee_matmul_bf16,
};
pub use ple::PleTable;
pub use shield::ShieldConfig;
pub use sim::{InProcessTrustedExecutor, PlaintextExecutor, RayonCpuEngine};
pub use snapshot::{PcieSnapshot, SnapshotCapture, SnapshotConfig};
pub use substrate::{GpuOffloadEngine, TrustedExecutor, WeightHandle, WeightKind};
