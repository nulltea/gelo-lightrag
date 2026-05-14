pub mod attestation;
pub mod service;
pub mod two_party_service;

#[cfg(feature = "snp")]
pub mod snp;

pub use attestation::{AttestationEvidence, AttestationVerifier, NoopAttestationVerifier};
pub use service::GeloRagInMemoryService;
pub use two_party_service::{GeloRagTwoPartyService, TwoPartyError};

#[cfg(feature = "snp")]
pub use snp::SnpVerifierAdapter;
