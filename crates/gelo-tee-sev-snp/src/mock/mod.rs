//! Mock SEV-SNP attestation: bundled test PKI + report issuer.
//!
//! Mirrors AMD's production chain (`ARK → ASK → VCEK`) using ECDSA-P-384
//! certs minted by the `mint-test-pki` example and committed under
//! `tests/fixtures/`. The [`MockReportIssuer`] signs reports with the VCEK
//! private key in the same byte shape as a real SEV-SNP report (SHA-384
//! over the first `0x2a0` bytes of the encoded report, ECDSA-P-384).

pub mod issuer;
pub mod test_pki;

pub use issuer::{IssuedAttestation, MockReportIssuer};
