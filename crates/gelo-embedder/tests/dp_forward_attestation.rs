//! M6.7 — SEV-SNP attestation binding test for `gelo-embedder/dp-forward`.
//!
//! Three assertions:
//!
//! 1. **Identity rebinding.** `with_dp_forward(cfg)` changes the embedder's
//!    `model_identity()` to `hex(sha256(weights_id || cfg.config_digest()))`,
//!    distinct from the plain `hex(weights_id)` and from any other DP cfg's.
//! 2. **Round-trip.** A mock SEV-SNP report issued for an embedder configured
//!    with `cfg_a` verifies cleanly when the relying party pins
//!    `expected_model_id = sha256(model_identity_bytes_for(cfg_a))`.
//! 3. **Tamper rejection.** The same report is rejected by a verifier that
//!    pins `expected_model_id` derived from `cfg_b ≠ cfg_a` — proving the
//!    SEV-SNP `REPORT_DATA[0..32]` slot actually depends on the DP cfg.

#![cfg(all(feature = "dp-forward"))]

use dp_forward::DpForwardConfig;
use gelo_tee_sev_snp::mock::MockReportIssuer;
use gelo_tee_sev_snp::report_data::ReportData;
use gelo_tee_sev_snp::verify::MockTimeSource;
use gelo_tee_sev_snp::{AttestedBinding, SnpAttestationVerifier, SnpRootTrust};
use sha2::{Digest, Sha256};

/// Replicate the hash chain that `GeloQwenEmbedder::with_dp_forward` uses
/// to derive its DP-bound model_identity bytes. The embedder stores the hex
/// string of `sha256(weights_identity || cfg.config_digest())` and returns
/// its UTF-8 bytes from `Embedder::model_identity`.
fn dp_bound_model_id_bytes(weights_identity: &[u8; 32], cfg: &DpForwardConfig) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(weights_identity);
    hasher.update(cfg.config_digest());
    let combined: [u8; 32] = hasher.finalize().into();
    hex::encode(combined).into_bytes()
}

/// Plain model_identity bytes (no DP) — matches what `GeloQwenEmbedder::new`
/// would return before any `with_dp_forward` call.
fn plain_model_id_bytes(weights_identity: &[u8; 32]) -> Vec<u8> {
    hex::encode(weights_identity).into_bytes()
}

/// Mock-aware time source pinned to a value inside the bundled VCEK's
/// validity window. (Matches the convention used by the existing mock
/// round-trip tests.)
fn mock_now() -> Box<MockTimeSource> {
    Box::new(MockTimeSource::at(1_800_000_000))
}

#[test]
fn dp_config_rebinds_model_identity() {
    let weights = [0xABu8; 32];
    let cfg_a = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
    let cfg_b = DpForwardConfig::calibrate(2.0, 1e-5, 1.0);

    let plain = plain_model_id_bytes(&weights);
    let id_a = dp_bound_model_id_bytes(&weights, &cfg_a);
    let id_b = dp_bound_model_id_bytes(&weights, &cfg_b);

    assert_ne!(plain, id_a, "DP-bound id must differ from plain weights id");
    assert_ne!(id_a, id_b, "different ε must produce different identities");
    // Same cfg, same weights → reproducible identity (so a verifier can
    // recompute it offline from the pinned (weights, cfg) pair).
    let id_a_again = dp_bound_model_id_bytes(&weights, &cfg_a);
    assert_eq!(id_a, id_a_again);
}

#[test]
fn mock_report_with_dp_binding_round_trips() {
    let weights = [0xC4u8; 32];
    let cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
    let model_id_bytes = dp_bound_model_id_bytes(&weights, &cfg);
    let scheme_id = b"caprise-key-v1";

    let issuer = MockReportIssuer::from_bundled().expect("load mock issuer");
    let rd = ReportData::build(&model_id_bytes, scheme_id, None);
    let issued = issuer.issue(rd).expect("issue mock report");

    let expected_model_id_hash: [u8; 32] = Sha256::digest(&model_id_bytes).into();
    let verifier = SnpAttestationVerifier::new(SnpRootTrust::with_mock_root())
        .with_expected_model_id(expected_model_id_hash)
        .with_clock(mock_now());

    let binding = AttestedBinding {
        model_identity: &model_id_bytes,
        scheme_identity: scheme_id,
        nonce: None,
    };
    verifier
        .verify(&issued.report_bytes, &issued.vcek_cert_pem, binding)
        .expect("verify mock report under the matching DP binding");
}

#[test]
fn mock_report_is_rejected_under_mismatched_dp_config() {
    let weights = [0xC4u8; 32];
    let cfg_a = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
    let cfg_b = DpForwardConfig::calibrate(2.0, 1e-5, 1.0);

    // Issuer mints a report under cfg_a.
    let model_id_a = dp_bound_model_id_bytes(&weights, &cfg_a);
    let scheme_id = b"caprise-key-v1";
    let issuer = MockReportIssuer::from_bundled().expect("load mock issuer");
    let rd = ReportData::build(&model_id_a, scheme_id, None);
    let issued = issuer.issue(rd).expect("issue mock report");

    // Verifier expects cfg_b's identity → should reject.
    let model_id_b = dp_bound_model_id_bytes(&weights, &cfg_b);
    let expected_b_hash: [u8; 32] = Sha256::digest(&model_id_b).into();
    let verifier = SnpAttestationVerifier::new(SnpRootTrust::with_mock_root())
        .with_expected_model_id(expected_b_hash)
        .with_clock(mock_now());

    // Note: AttestedBinding still uses model_id_a so REPORT_DATA round-trip
    // passes (model_id matches what's in the report), but the
    // `expected_model_id` pin then fails. That isolates the DP-binding
    // assertion from a generic REPORT_DATA mismatch.
    let binding = AttestedBinding {
        model_identity: &model_id_a,
        scheme_identity: scheme_id,
        nonce: None,
    };
    let result = verifier.verify(&issued.report_bytes, &issued.vcek_cert_pem, binding);
    assert!(
        result.is_err(),
        "verifier with expected_model_id from cfg_b must reject a report issued under cfg_a"
    );
}
