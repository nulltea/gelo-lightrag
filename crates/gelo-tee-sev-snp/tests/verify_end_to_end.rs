//! End-to-end SnpAttestationVerifier tests.
//!
//! Issues a mock-signed SEV-SNP report and runs the verifier against it
//! under various conditions: honest happy path, tampered report bytes,
//! tampered signature, wrong binding, expired VCEK.

#![cfg(feature = "mock")]

use gelo_tee_sev_snp::mock::MockReportIssuer;
use gelo_tee_sev_snp::report::parse_report;
use gelo_tee_sev_snp::report_data::ReportData;
use gelo_tee_sev_snp::verify::{
    AttestedBinding, MockTimeSource, SnpAttestationVerifier, SnpRootTrust, SnpVerifyError,
};

fn issuer_and_verifier() -> (MockReportIssuer, SnpAttestationVerifier) {
    let issuer = MockReportIssuer::from_bundled().unwrap();
    let verifier = SnpAttestationVerifier::new(SnpRootTrust::with_mock_root());
    (issuer, verifier)
}

#[test]
fn happy_path_accepts_correct_binding() {
    let (issuer, verifier) = issuer_and_verifier();
    let binding = AttestedBinding {
        model_identity: b"Qwen/Qwen3-Embedding-0.6B@rev-1",
        scheme_identity: b"scheme-v1",
        nonce: Some(b"session-42"),
    };
    let rd = ReportData::build(
        binding.model_identity,
        binding.scheme_identity,
        binding.nonce,
    );
    let issued = issuer.issue(rd).unwrap();
    verifier
        .verify(&issued.report_bytes, &issued.vcek_cert_pem, binding)
        .expect("happy path should validate");
}

#[test]
fn rejects_wrong_model_identity() {
    let (issuer, verifier) = issuer_and_verifier();
    let real_binding = AttestedBinding {
        model_identity: b"model-X",
        scheme_identity: b"scheme-v1",
        nonce: None,
    };
    let rd = ReportData::build(
        real_binding.model_identity,
        real_binding.scheme_identity,
        real_binding.nonce,
    );
    let issued = issuer.issue(rd).unwrap();

    // Verifier expects model-Y instead. report_data won't match.
    let wrong_binding = AttestedBinding {
        model_identity: b"model-Y",
        scheme_identity: b"scheme-v1",
        nonce: None,
    };
    let err = verifier
        .verify(&issued.report_bytes, &issued.vcek_cert_pem, wrong_binding)
        .unwrap_err();
    assert!(matches!(err, SnpVerifyError::ReportDataMismatch { .. }));
}

#[test]
fn rejects_wrong_scheme_identity() {
    let (issuer, verifier) = issuer_and_verifier();
    let rd = ReportData::build(b"m", b"scheme-a", None);
    let issued = issuer.issue(rd).unwrap();
    let wrong = AttestedBinding {
        model_identity: b"m",
        scheme_identity: b"scheme-b",
        nonce: None,
    };
    let err = verifier
        .verify(&issued.report_bytes, &issued.vcek_cert_pem, wrong)
        .unwrap_err();
    assert!(matches!(err, SnpVerifyError::ReportDataMismatch { .. }));
}

#[test]
fn rejects_tampered_report_body() {
    let (issuer, verifier) = issuer_and_verifier();
    let binding = AttestedBinding {
        model_identity: b"m",
        scheme_identity: b"s",
        nonce: None,
    };
    let rd = ReportData::build(binding.model_identity, binding.scheme_identity, binding.nonce);
    let issued = issuer.issue(rd).unwrap();
    // Flip a byte in the measurable prefix (anywhere before offset 0x2a0).
    let mut tampered = issued.report_bytes.clone();
    tampered[0x100] ^= 0xff;
    let err = verifier
        .verify(&tampered, &issued.vcek_cert_pem, binding)
        .unwrap_err();
    // Either the report-data check trips first (if we mangled REPORT_DATA),
    // or the signature check (otherwise). Both are acceptable rejections.
    assert!(matches!(
        err,
        SnpVerifyError::ReportSignature(_) | SnpVerifyError::ReportDataMismatch { .. }
    ));
}

#[test]
fn rejects_tampered_signature() {
    let (issuer, verifier) = issuer_and_verifier();
    let binding = AttestedBinding {
        model_identity: b"m",
        scheme_identity: b"s",
        nonce: None,
    };
    let rd = ReportData::build(binding.model_identity, binding.scheme_identity, binding.nonce);
    let issued = issuer.issue(rd).unwrap();
    // The signature lives in the trailing section after the measurable prefix.
    let mut tampered = issued.report_bytes.clone();
    let sig_offset = 0x2a0; // start of signature section in SEV-SNP report
    tampered[sig_offset + 10] ^= 0x55;
    let err = verifier
        .verify(&tampered, &issued.vcek_cert_pem, binding)
        .unwrap_err();
    assert!(matches!(err, SnpVerifyError::ReportSignature(_)));
}

#[test]
fn rejects_expired_vcek() {
    let (issuer, verifier) = issuer_and_verifier();
    let binding = AttestedBinding {
        model_identity: b"m",
        scheme_identity: b"s",
        nonce: None,
    };
    let rd = ReportData::build(binding.model_identity, binding.scheme_identity, binding.nonce);
    let issued = issuer.issue(rd).unwrap();

    // Move the clock to far past the test PKI's 20-year validity window.
    let future = 60i64 * 60 * 24 * 365 * 100; // year ~2070
    let verifier = verifier.with_clock(Box::new(MockTimeSource::at(future)));
    let err = verifier
        .verify(&issued.report_bytes, &issued.vcek_cert_pem, binding)
        .unwrap_err();
    assert!(matches!(err, SnpVerifyError::VcekExpired { .. }));
}

#[test]
fn rejects_pre_validity_clock() {
    let (issuer, verifier) = issuer_and_verifier();
    let binding = AttestedBinding {
        model_identity: b"m",
        scheme_identity: b"s",
        nonce: None,
    };
    let rd = ReportData::build(binding.model_identity, binding.scheme_identity, binding.nonce);
    let issued = issuer.issue(rd).unwrap();

    // Move the clock to before VCEK's notBefore (test PKI minted at "now-rounded";
    // year 2000 is comfortably before).
    let past = 60i64 * 60 * 24 * 365 * 30; // year ~2000
    let verifier = verifier.with_clock(Box::new(MockTimeSource::at(past)));
    let err = verifier
        .verify(&issued.report_bytes, &issued.vcek_cert_pem, binding)
        .unwrap_err();
    assert!(matches!(err, SnpVerifyError::VcekExpired { .. }));
}

#[test]
fn pinned_model_id_must_match() {
    let (issuer, _) = issuer_and_verifier();
    let binding = AttestedBinding {
        model_identity: b"m",
        scheme_identity: b"s",
        nonce: None,
    };
    let rd = ReportData::build(binding.model_identity, binding.scheme_identity, binding.nonce);
    let issued = issuer.issue(rd).unwrap();

    let report = parse_report(&issued.report_bytes).unwrap();
    let correct_pin: [u8; 32] = report.report_data[..32].try_into().unwrap();
    let wrong_pin = [0u8; 32];

    let verifier_ok = SnpAttestationVerifier::new(SnpRootTrust::with_mock_root())
        .with_expected_model_id(correct_pin);
    verifier_ok
        .verify(&issued.report_bytes, &issued.vcek_cert_pem, binding)
        .expect("matching pin should validate");

    let verifier_wrong = SnpAttestationVerifier::new(SnpRootTrust::with_mock_root())
        .with_expected_model_id(wrong_pin);
    let err = verifier_wrong
        .verify(&issued.report_bytes, &issued.vcek_cert_pem, binding)
        .unwrap_err();
    assert!(matches!(err, SnpVerifyError::ReportDataMismatch { .. }));
}
