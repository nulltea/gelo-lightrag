//! Bundled test PKI (PEM blobs minted by `examples/mint-test-pki.rs` and
//! committed under `tests/fixtures/`).
//!
//! These are **test-only** certificates with explicit mock semantics. The
//! production verifier only accepts the AMD-published `ARK` root; the mock
//! root is reachable only through a `#[cfg(feature = "mock")]` accessor.

/// Mock AMD Root Key (self-signed, P-384).
pub const ARK_PEM: &[u8] = include_bytes!("../../tests/fixtures/mock_ark.pem");
/// Mock AMD Signing Key (ARK-signed).
pub const ASK_PEM: &[u8] = include_bytes!("../../tests/fixtures/mock_ask.pem");
/// Mock AMD Versioned Chip Endorsement Key (ASK-signed, signs reports).
pub const VCEK_PEM: &[u8] = include_bytes!("../../tests/fixtures/mock_vcek.pem");

/// Mock VCEK private key (only used by `MockReportIssuer` to sign reports;
/// never linked into production builds).
pub const VCEK_KEY_PEM: &[u8] = include_bytes!("../../tests/fixtures/mock_vcek.key.pem");
