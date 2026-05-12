//! SEV-SNP-attested trusted-executor backend for the GELO protocol.
//!
//! Provides:
//! - [`SnpTrustedExecutor`] — wraps [`gelo_protocol::sim::InProcessTrustedExecutor`]
//!   with `/dev/sev-guest`-driven attestation report generation.
//! - [`SnpAttestationVerifier`] — relying-party verifier for SEV-SNP attestation
//!   reports. Always built (verifier code is environment-agnostic).
//! - [`MockReportIssuer`] (with `mock` feature) — fake-signs reports against a
//!   bundled `ARK → ASK → VCEK` test PKI for runs on non-SEV-SNP hardware.
//!
//! The crate compiles in two configurations:
//!
//! - `--features sev-snp` (default) — production path. The `sev` crate's
//!   `/dev/sev-guest` ioctls are linked but only succeed inside a real
//!   SEV-SNP CVM.
//! - `--features mock` — for tests + CI. Replaces the hardware path with a
//!   `MockReportIssuer` that mints DCAP-byte-compatible mock reports signed
//!   by the bundled test root.
//!
//! The two paths are not mutually exclusive — enabling both is useful for
//! round-trip tests where a mock-issued report flows through the same
//! verifier code as a real one.

pub mod executor;
pub mod report;
pub mod report_data;
pub mod runtime_mode;
pub mod verify;

#[cfg(feature = "mock")]
pub mod cert_chain;
#[cfg(feature = "mock")]
pub mod mock;

pub use executor::{AttestationIssuer, SnpEvidence, SnpTrustedExecutor};
pub use report_data::ReportData;
pub use runtime_mode::{RuntimeMode, RuntimeModeError};
pub use verify::{AttestedBinding, SnpAttestationVerifier, SnpRootTrust, TimeSource};
