//! Thin wrappers around `virtee/sev`'s [`AttestationReport`] type.
//!
//! The `sev` crate already implements byte-level parse + serialize for the
//! SEV-SNP attestation report (AMD SEV ABI Spec §7.3) via its `ByteParser`
//! trait. We re-export the type through a project-local alias and expose
//! convenience `parse_report` / `serialize_report` helpers that pin the
//! error type to `anyhow::Result` for consistency with the rest of the
//! workspace.

pub use sev::firmware::guest::AttestationReport;

use anyhow::{Context, Result};
use sev::parser::ByteParser;

/// Parse a `&[u8]` slice into an [`AttestationReport`].
///
/// The slice length must match the report's `EXPECTED_LEN` for this version
/// (1184 bytes for v3 + post-Genoa reports); shorter/longer inputs are
/// rejected by `sev`'s parser.
pub fn parse_report(bytes: &[u8]) -> Result<AttestationReport> {
    AttestationReport::from_bytes(bytes)
        .context("parsing SEV-SNP attestation report bytes")
}

/// Serialize an [`AttestationReport`] back into its canonical byte layout.
pub fn serialize_report(report: &AttestationReport) -> Result<Vec<u8>> {
    let bytes = report
        .to_bytes()
        .context("serializing SEV-SNP attestation report")?;
    Ok(bytes.as_ref().to_vec())
}

/// Construct a test-suitable AttestationReport with valid `version` + chip_id
/// fields so the `sev` encoder can determine a generation without inspecting
/// the host CPU (which fails on non-EPYC dev boxes).
///
/// Sets `version = 2` so the encoder uses the chip_id heuristic
/// (`chip_id_is_turin_like`) rather than CPUID. Sets a non-zero, non-Turin
/// chip_id so the heuristic resolves to Genoa.
#[cfg(any(test, feature = "mock"))]
pub fn skeleton_for_tests() -> AttestationReport {
    let mut report = AttestationReport::default();
    report.version = 2;
    // Non-zero across all 64 bytes ⇒ neither "masked" (all-zero) nor
    // "Turin-like" (bytes[8..] all zero). Resolves to Genoa.
    report.chip_id = [0xAB; 64];
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skeleton report round-trips through parse + serialize.
    #[test]
    fn skeleton_report_round_trip() {
        let report = skeleton_for_tests();
        let bytes = serialize_report(&report).unwrap();
        let parsed = parse_report(&bytes).unwrap();
        let rebytes = serialize_report(&parsed).unwrap();
        assert_eq!(bytes, rebytes, "round-trip should be byte-identical");
        assert!(!bytes.is_empty(), "encoded report must not be empty");
    }

    /// A report with a specific REPORT_DATA value round-trips losslessly.
    #[test]
    fn report_data_field_survives_round_trip() {
        let mut report = skeleton_for_tests();
        let test_data: [u8; 64] = std::array::from_fn(|i| (i as u8) ^ 0x5a);
        report.report_data = test_data;
        let bytes = serialize_report(&report).unwrap();
        let parsed = parse_report(&bytes).unwrap();
        assert_eq!(parsed.report_data, test_data);
    }

    #[test]
    fn truncated_bytes_rejected() {
        let report = skeleton_for_tests();
        let bytes = serialize_report(&report).unwrap();
        let bad = &bytes[..bytes.len() / 2];
        assert!(parse_report(bad).is_err());
    }
}
