//! `MockReportIssuer` — signs SEV-SNP attestation reports with the bundled
//! test VCEK so the full verifier path (chain check + ECDSA-P-384 + SHA-384
//! over `report[..0x2a0]`) is exercised end-to-end on non-EPYC hardware.

use anyhow::{Context, Result, anyhow};
use p384::ecdsa::{SigningKey, signature::DigestSigner};
use p384::pkcs8::DecodePrivateKey;
use sev::certs::snp::ecdsa::Signature as SnpSignature;
use sev::parser::ByteParser;
use sha2::{Digest, Sha384};

use super::test_pki;
use crate::report::skeleton_for_tests;
use crate::report_data::ReportData;

/// Bytes of the encoded report that are covered by the VCEK signature
/// (everything except the trailing 512-byte signature field).
const MEASURABLE_PREFIX_LEN: usize = 0x2a0;

/// Output of a `MockReportIssuer::issue` call: the signed report bytes and
/// the matching VCEK cert. The relying party feeds both into
/// `SnpAttestationVerifier::verify`.
#[derive(Clone, Debug)]
pub struct IssuedAttestation {
    /// Encoded SEV-SNP attestation report (1184 bytes per AMD ABI §7.3).
    pub report_bytes: Vec<u8>,
    /// VCEK certificate (PEM) — the leaf that signed `report_bytes`.
    pub vcek_cert_pem: Vec<u8>,
}

#[derive(Clone)]
pub struct MockReportIssuer {
    vcek_signing_key: SigningKey,
}

impl MockReportIssuer {
    /// Load the bundled mock VCEK private key for signing.
    pub fn from_bundled() -> Result<Self> {
        let key_pem = std::str::from_utf8(test_pki::VCEK_KEY_PEM)
            .context("bundled VCEK private-key PEM is not UTF-8")?;
        let vcek_signing_key = SigningKey::from_pkcs8_pem(key_pem)
            .map_err(|e| anyhow!("parsing bundled VCEK private key: {e}"))?;
        Ok(Self { vcek_signing_key })
    }

    /// Issue a signed attestation report carrying the given `report_data`.
    ///
    /// The report's other fields come from `skeleton_for_tests()` (version=2,
    /// non-zero chip_id, all other AMD-spec fields zeroed) so the encoder
    /// resolves to Genoa cleanly without inspecting host CPUID.
    pub fn issue(&self, report_data: ReportData) -> Result<IssuedAttestation> {
        let mut report = skeleton_for_tests();
        report.report_data = (*report_data.as_bytes()).into();
        report.signature = SnpSignature::default();

        // Serialize once with a zero signature to obtain the measurable bytes.
        let encoded = report
            .to_bytes()
            .map_err(|e| anyhow!("serializing report (pre-sign): {e}"))?;
        let measurable = encoded
            .as_ref()
            .get(..MEASURABLE_PREFIX_LEN)
            .ok_or_else(|| anyhow!("encoded report shorter than expected measurable prefix"))?;

        let digest = Sha384::new_with_prefix(measurable);
        let sig: p384::ecdsa::Signature = self.vcek_signing_key.sign_digest(digest);
        report.signature = pack_signature(&sig)?;

        let final_bytes = report
            .to_bytes()
            .map_err(|e| anyhow!("serializing report (post-sign): {e}"))?
            .as_ref()
            .to_vec();

        Ok(IssuedAttestation {
            report_bytes: final_bytes,
            vcek_cert_pem: test_pki::VCEK_PEM.to_vec(),
        })
    }
}

/// Pack a `p384::ecdsa::Signature` into the SEV-SNP `Signature` byte layout.
///
/// The crate's `Signature` stores `r: [u8; 72]` and `s: [u8; 72]` in
/// **little-endian** order (per the AMD ABI). p384 returns 48-byte
/// big-endian scalars; we reverse them and pad the trailing 24 bytes with
/// zero. The `TryFrom<&Signature> for p384::ecdsa::Signature` impl in the
/// `sev` crate is the inverse — it takes the first 48 bytes (LE) and
/// reverses them.
fn pack_signature(sig: &p384::ecdsa::Signature) -> Result<SnpSignature> {
    let (r, s) = sig.split_bytes();
    if r.len() != 48 || s.len() != 48 {
        return Err(anyhow!(
            "expected 48-byte ECDSA P-384 scalars, got r={} s={}",
            r.len(),
            s.len()
        ));
    }
    let mut r_le = [0u8; 72];
    let mut s_le = [0u8; 72];
    for (dst, src) in r_le[..48].iter_mut().zip(r.iter().rev()) {
        *dst = *src;
    }
    for (dst, src) in s_le[..48].iter_mut().zip(s.iter().rev()) {
        *dst = *src;
    }
    Ok(SnpSignature::new(r_le, s_le))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::parse_report;
    use sev::certs::snp::Certificate;
    use sev::certs::snp::Verifiable;

    #[test]
    fn issuer_loads_bundled_key() {
        let _ = MockReportIssuer::from_bundled().expect("load bundled VCEK key");
    }

    #[test]
    fn issued_report_round_trips_and_carries_report_data() {
        let issuer = MockReportIssuer::from_bundled().unwrap();
        let rd = ReportData::build(b"model-x", b"scheme-x", Some(b"nonce-1"));
        let issued = issuer.issue(rd).unwrap();
        let report = parse_report(&issued.report_bytes).unwrap();
        assert_eq!(&report.report_data, rd.as_bytes());
        assert!(!issued.vcek_cert_pem.is_empty());
    }

    #[test]
    fn issued_signature_verifies_against_bundled_vcek() {
        let issuer = MockReportIssuer::from_bundled().unwrap();
        let rd = ReportData::build(b"m", b"s", None);
        let issued = issuer.issue(rd).unwrap();
        let report = parse_report(&issued.report_bytes).unwrap();
        let vcek = Certificate::from_pem(test_pki::VCEK_PEM).expect("parse bundled VCEK");
        (&vcek, &report)
            .verify()
            .expect("VCEK should sign its own issued report");
    }
}
