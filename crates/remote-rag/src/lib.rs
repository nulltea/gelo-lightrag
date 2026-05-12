//! Rust implementation of [RemoteRAG](https://arxiv.org/abs/2412.12775)
//! (Cheng et al., ACL Findings 2025). See `README.md` for the threat model,
//! the CAPRISE-mutual-exclusion call-out, and the upcoming-module roadmap.

pub mod paillier;
pub mod planar_laplace;
pub mod service;

pub use paillier::{
    DEFAULT_KEY_BITS, DEFAULT_SCALE_BITS, PaillierCiphertext, PaillierPrivateKey,
    PaillierPublicKey,
};
pub use planar_laplace::PlanarLaplaceConfig;
pub use service::RemoteRagService;
