//! ECDSA-P-384 SEV-SNP certificate-chain validation.
//!
//! AMD's production chain (`ARK → ASK → VCEK`) uses RSA-PSS for the
//! cert-to-cert signatures and ECDSA-P-384 only for the VCEK-to-report
//! signature. The `sev` crate's `crypto_nossl` chain verifier therefore
//! requires RSA-PSS for the chain hops.
//!
//! Our test PKI (rcgen-minted, `ARK/ASK/VCEK` all P-384) intentionally
//! uses ECDSA throughout because it keeps the dependency surface
//! Apache-2.0-only and the cert generation deterministic. This module
//! provides a small hand-rolled chain validator that handles the
//! all-ECDSA case. The report-vs-VCEK signature check still goes through
//! `sev`'s production-grade `Verifiable for (&Certificate, &AttestationReport)`
//! impl, so the security-critical report validation is shared between mock
//! and real silicon.
//!
//! For real EPYC deployment (M5.9), this module is bypassed in favour of
//! `sev::certs::snp::Chain::verify()` which handles AMD's RSA-PSS chain.

use anyhow::{Context, Result, anyhow};
use p384::ecdsa::signature::DigestVerifier;
use sha2::{Digest, Sha384};
use x509_cert::Certificate;
use x509_cert::der::{DecodePem, Encode};
use x509_cert::spki::ObjectIdentifier;

/// OID for `ecdsa-with-SHA384` (RFC 5758 §3.2).
const ECDSA_WITH_SHA384_OID: ObjectIdentifier =
    ObjectIdentifier::new_unwrap("1.2.840.10045.4.3.3");

/// Parse a PEM blob into an x509 `Certificate`.
pub fn parse_cert_pem(pem: &[u8]) -> Result<Certificate> {
    let cert = Certificate::from_pem(pem).context("parsing PEM-encoded certificate")?;
    Ok(cert)
}

/// Verify that `child` is signed by `parent`, assuming both certificates use
/// `ecdsa-with-SHA384`. Performs only the cryptographic-signature check;
/// validity-window and name-constraint checks live in the higher-level
/// verifier.
pub fn verify_signed_by(parent: &Certificate, child: &Certificate) -> Result<()> {
    if child.signature_algorithm.oid != ECDSA_WITH_SHA384_OID {
        return Err(anyhow!(
            "child certificate uses unsupported signature algorithm: {:?}",
            child.signature_algorithm.oid
        ));
    }

    // Extract parent's ECDSA-P-384 verifying key from its SPKI.
    let parent_spki = &parent.tbs_certificate.subject_public_key_info;
    let pub_key_bytes = parent_spki
        .subject_public_key
        .as_bytes()
        .ok_or_else(|| anyhow!("parent SPKI has no public-key bytes"))?;
    let verifying_key = p384::ecdsa::VerifyingKey::from_sec1_bytes(pub_key_bytes)
        .map_err(|e| anyhow!("parent's SPKI is not a valid ECDSA-P-384 public key: {e}"))?;

    // Re-serialise the child's TBSCertificate and hash with SHA-384.
    let tbs_der = child
        .tbs_certificate
        .to_der()
        .context("re-serialising child's TBSCertificate as DER")?;
    let digest = Sha384::new_with_prefix(&tbs_der);

    // The signature is a DER-encoded ECDSA-Sig-Value (SEQUENCE { r, s }).
    let sig_bytes = child
        .signature
        .as_bytes()
        .ok_or_else(|| anyhow!("child certificate signature is empty"))?;
    let sig = p384::ecdsa::Signature::from_der(sig_bytes)
        .map_err(|e| anyhow!("parsing child's ECDSA signature: {e}"))?;

    verifying_key
        .verify_digest(digest, &sig)
        .map_err(|e| anyhow!("child signature does not verify against parent public key: {e}"))
}

/// Validate the full mock SEV-SNP chain: `ARK ⊢ self`, `ARK ⊢ ASK`, `ASK ⊢ VCEK`.
/// Returns the parsed VCEK certificate for downstream report validation.
pub fn verify_mock_chain(
    ark_pem: &[u8],
    ask_pem: &[u8],
    vcek_pem: &[u8],
) -> Result<Certificate> {
    let ark = parse_cert_pem(ark_pem).context("parsing ARK")?;
    let ask = parse_cert_pem(ask_pem).context("parsing ASK")?;
    let vcek = parse_cert_pem(vcek_pem).context("parsing VCEK")?;
    verify_signed_by(&ark, &ark).context("ARK self-signature")?;
    verify_signed_by(&ark, &ask).context("ARK → ASK")?;
    verify_signed_by(&ask, &vcek).context("ASK → VCEK")?;
    Ok(vcek)
}

/// Inspect the certificate's `notBefore` / `notAfter` validity window.
/// Returns the UNIX-seconds bounds for downstream clock-driven checks.
pub fn validity_window(cert: &Certificate) -> (i64, i64) {
    use x509_cert::time::Time;
    let to_unix = |t: &Time| -> i64 {
        match t {
            Time::UtcTime(u) => u.to_unix_duration().as_secs() as i64,
            Time::GeneralTime(g) => g.to_unix_duration().as_secs() as i64,
        }
    };
    let nb = to_unix(&cert.tbs_certificate.validity.not_before);
    let na = to_unix(&cert.tbs_certificate.validity.not_after);
    (nb, na)
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;
    use crate::mock::test_pki;

    #[test]
    fn mock_chain_verifies() {
        let vcek = verify_mock_chain(test_pki::ARK_PEM, test_pki::ASK_PEM, test_pki::VCEK_PEM)
            .expect("mock chain should validate end-to-end");
        let (nb, na) = validity_window(&vcek);
        assert!(na > nb, "validity window should be non-degenerate");
    }

    #[test]
    fn swapping_ark_and_ask_breaks_chain() {
        // Pretend the ASK signs itself (i.e. treat ASK as ARK).
        let r = verify_mock_chain(test_pki::ASK_PEM, test_pki::ASK_PEM, test_pki::VCEK_PEM);
        assert!(r.is_err(), "swapped chain must not validate");
    }

    #[test]
    fn tampered_vcek_breaks_chain() {
        let mut tampered = test_pki::VCEK_PEM.to_vec();
        // Flip a bit somewhere inside the DER-encoded section. PEM has a
        // base64 body between the BEGIN/END markers; find the first
        // base64-looking char after the header and flip it.
        let header_end = tampered
            .windows(b"-----\n".len())
            .position(|w| w == b"-----\n")
            .map(|p| p + b"-----\n".len())
            .unwrap_or(64);
        if let Some(b) = tampered.get_mut(header_end + 30) {
            *b ^= 0x04; // mangle a base64 char (must stay valid base64-alphabet)
        }
        let r = verify_mock_chain(test_pki::ARK_PEM, test_pki::ASK_PEM, &tampered);
        assert!(r.is_err(), "tampered VCEK must not validate");
    }
}
