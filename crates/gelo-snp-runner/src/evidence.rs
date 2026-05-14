//! Assembling `AttestationEvidence` from an `IssuerHandle`.
//!
//! The runner exposes attestation evidence both as a top-level `/attest`
//! endpoint and as the `attestation` field on every `/query` response, so
//! callers don't have to make two round-trips.

use anyhow::Result;
use gelo_rag::AttestationEvidence;
use gelo_tee_sev_snp::executor::AttestationIssuer;
use gelo_tee_sev_snp::report_data::ReportData;

pub use crate::issuer::IssuerHandle;

/// Issue a fresh report binding `(model_identity, scheme_identity)` and
/// wrap it in the workspace's [`AttestationEvidence`] type.
pub fn build_evidence(
    issuer: &IssuerHandle,
    model_identity: &[u8],
    scheme_identity: &[u8],
) -> Result<AttestationEvidence> {
    let rd = ReportData::build(model_identity, scheme_identity, None);
    let issued = issuer.issue(rd)?;
    // `report_data` is baked into REPORT_DATA inside the report itself;
    // the string fields here are returned to clients as a convenience so
    // they can recompute and cross-check without parsing the 1184-byte
    // blob.
    Ok(AttestationEvidence {
        tee_measurement: "snp".to_string(),
        model_identity: String::from_utf8_lossy(model_identity).into_owned(),
        scheme_identity: String::from_utf8_lossy(scheme_identity).into_owned(),
        report: Some(issued.report_bytes),
        vcek_cert: Some(issued.vcek_cert_pem),
    })
}
