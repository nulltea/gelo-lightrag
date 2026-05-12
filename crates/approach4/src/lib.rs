pub mod attestation;
pub mod service;

#[cfg(feature = "snp")]
pub mod snp;

pub use attestation::{AttestationEvidence, AttestationVerifier, NoopAttestationVerifier};
pub use service::Approach4InMemoryService;

#[cfg(feature = "snp")]
pub use snp::SnpVerifierAdapter;
