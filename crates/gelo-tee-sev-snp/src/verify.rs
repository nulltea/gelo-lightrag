//! Relying-party SEV-SNP attestation verifier.
//!
//! Validates a full attestation bundle:
//!
//! 1. Parse the 1184-byte attestation report.
//! 2. Validate the cert chain `ARK → ASK → VCEK` against the configured root.
//!    Mock chains (ECDSA throughout) go through [`cert_chain::verify_mock_chain`];
//!    real AMD chains (RSA-PSS) will go through `sev`'s production validator
//!    when M5.9 lands.
//! 3. Verify the report's ECDSA-P-384 signature against the VCEK's public key
//!    via `sev::certs::snp::Verifiable for (&Certificate, &AttestationReport)`.
//! 4. Recompute the expected `REPORT_DATA` from the attested binding
//!    (model_identity || scheme_identity || nonce) and compare to the report.
//! 5. If pinned, check `MEASUREMENT` and `POLICY` fields.
//! 6. Check the VCEK's `notBefore` / `notAfter` against the configured clock.
//!
//! Errors are flagged with distinct `thiserror` variants so callers can
//! tell apart "wrong chain" vs "wrong report-data" vs "VCEK expired".

use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

use crate::cert_chain;
use crate::report::{AttestationReport, parse_report};
use crate::report_data::ReportData;

/// The (model_identity, scheme_identity, nonce) tuple the relying party
/// expects to find baked into `REPORT_DATA`.
#[derive(Clone, Copy, Debug)]
pub struct AttestedBinding<'a> {
    pub model_identity: &'a [u8],
    pub scheme_identity: &'a [u8],
    pub nonce: Option<&'a [u8]>,
}

/// Trust root configuration. Mock roots (the bundled test PKI) are only
/// reachable through a `#[cfg(feature = "mock")]` constructor — production
/// builds cannot link them.
pub struct SnpRootTrust {
    pub ark_pem: Vec<u8>,
    pub ask_pem: Vec<u8>,
    /// `true` ⇒ chain validated via the mock ECDSA path; `false` ⇒ AMD's
    /// RSA-PSS chain (deferred to M5.9, currently unsupported in this crate).
    pub use_mock_chain: bool,
}

impl SnpRootTrust {
    #[cfg(feature = "mock")]
    pub fn with_mock_root() -> Self {
        use crate::mock::test_pki;
        Self {
            ark_pem: test_pki::ARK_PEM.to_vec(),
            ask_pem: test_pki::ASK_PEM.to_vec(),
            use_mock_chain: true,
        }
    }
}

/// Pluggable clock for VCEK validity-window checks. `MockTimeSource` lets
/// tests advance time deterministically; production uses `SystemClock`.
pub trait TimeSource: Send + Sync {
    fn now_unix_seconds(&self) -> i64;
}

pub struct SystemClock;
impl TimeSource for SystemClock {
    fn now_unix_seconds(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }
}

/// Caller-controlled time for tests. `std::sync::atomic::AtomicI64` so it's
/// trivially `Send + Sync`.
pub struct MockTimeSource(pub std::sync::atomic::AtomicI64);

impl MockTimeSource {
    pub fn at(unix_seconds: i64) -> Self {
        Self(std::sync::atomic::AtomicI64::new(unix_seconds))
    }
    pub fn set(&self, unix_seconds: i64) {
        self.0.store(unix_seconds, std::sync::atomic::Ordering::Relaxed);
    }
}

impl TimeSource for MockTimeSource {
    fn now_unix_seconds(&self) -> i64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Debug, Error)]
pub enum SnpVerifyError {
    #[error("attestation report bytes missing from evidence")]
    MissingReport,
    #[error("VCEK certificate missing from evidence")]
    MissingVcek,
    #[error("parsing attestation report: {0}")]
    ReportParse(anyhow::Error),
    #[error("cert chain validation: {0}")]
    Chain(anyhow::Error),
    #[error("report signature does not verify against VCEK")]
    ReportSignature(#[source] std::io::Error),
    #[error("expected REPORT_DATA = {expected:x?}, got {actual:x?}")]
    ReportDataMismatch {
        expected: [u8; 64],
        actual: [u8; 64],
    },
    #[error("expected MEASUREMENT mismatch")]
    MeasurementMismatch,
    #[error("expected POLICY mismatch (expected {expected}, got {actual})")]
    PolicyMismatch { expected: u64, actual: u64 },
    #[error(
        "VCEK validity window ({not_before}..{not_after}) does not include current time {now}"
    )]
    VcekExpired {
        not_before: i64,
        not_after: i64,
        now: i64,
    },
    #[error("AMD RSA-PSS chain validation not yet implemented; use mock chain for now")]
    UnsupportedProductionChain,
}

pub struct SnpAttestationVerifier {
    pub root_trust: SnpRootTrust,
    pub expected_measurement: Option<[u8; 48]>,
    pub expected_model_id: Option<[u8; 32]>,
    pub expected_policy: Option<u64>,
    pub clock: Box<dyn TimeSource>,
}

impl SnpAttestationVerifier {
    pub fn new(root_trust: SnpRootTrust) -> Self {
        Self {
            root_trust,
            expected_measurement: None,
            expected_model_id: None,
            expected_policy: None,
            clock: Box::new(SystemClock),
        }
    }

    pub fn with_expected_measurement(mut self, m: [u8; 48]) -> Self {
        self.expected_measurement = Some(m);
        self
    }

    pub fn with_expected_model_id(mut self, m: [u8; 32]) -> Self {
        self.expected_model_id = Some(m);
        self
    }

    pub fn with_expected_policy(mut self, p: u64) -> Self {
        self.expected_policy = Some(p);
        self
    }

    pub fn with_clock(mut self, clock: Box<dyn TimeSource>) -> Self {
        self.clock = clock;
        self
    }

    pub fn verify(
        &self,
        report_bytes: &[u8],
        vcek_pem: &[u8],
        binding: AttestedBinding<'_>,
    ) -> Result<AttestationReport, SnpVerifyError> {
        // 1. Parse report.
        let report = parse_report(report_bytes).map_err(SnpVerifyError::ReportParse)?;

        // 2. Validate cert chain.
        let vcek = if self.root_trust.use_mock_chain {
            cert_chain::verify_mock_chain(
                &self.root_trust.ark_pem,
                &self.root_trust.ask_pem,
                vcek_pem,
            )
            .map_err(SnpVerifyError::Chain)?
        } else {
            return Err(SnpVerifyError::UnsupportedProductionChain);
        };

        // 3. Check VCEK validity window.
        let (nb, na) = cert_chain::validity_window(&vcek);
        let now = self.clock.now_unix_seconds();
        if now < nb || now > na {
            return Err(SnpVerifyError::VcekExpired {
                not_before: nb,
                not_after: na,
                now,
            });
        }

        // 4. Verify report signature against VCEK.
        verify_report_signature(&report, vcek_pem)?;

        // 5. Recompute expected REPORT_DATA, compare.
        let expected =
            ReportData::build(binding.model_identity, binding.scheme_identity, binding.nonce);
        if &report.report_data[..] != expected.as_bytes().as_slice() {
            return Err(SnpVerifyError::ReportDataMismatch {
                expected: *expected.as_bytes(),
                actual: report.report_data.into(),
            });
        }

        // 6. Optional MEASUREMENT pin.
        if let Some(expected_m) = &self.expected_measurement {
            if &report.measurement[..] != &expected_m[..] {
                return Err(SnpVerifyError::MeasurementMismatch);
            }
        }

        // 7. Optional POLICY pin.
        if let Some(expected_p) = self.expected_policy {
            let actual = report.policy.0;
            if actual != expected_p {
                return Err(SnpVerifyError::PolicyMismatch {
                    expected: expected_p,
                    actual,
                });
            }
        }

        // 8. Optional model_id pin (left half of REPORT_DATA).
        if let Some(expected_id) = &self.expected_model_id {
            if &report.report_data[..32] != &expected_id[..] {
                return Err(SnpVerifyError::ReportDataMismatch {
                    expected: *expected.as_bytes(),
                    actual: report.report_data.into(),
                });
            }
        }

        Ok(report)
    }
}

/// Use the `sev` crate's production-grade ECDSA-P-384 + SHA-384 report
/// verification. Works for both mock-issued and real-EPYC reports because
/// the on-the-wire signature shape is identical.
fn verify_report_signature(
    report: &AttestationReport,
    vcek_pem: &[u8],
) -> Result<(), SnpVerifyError> {
    use sev::certs::snp::{Certificate, Verifiable};
    let vcek = Certificate::from_pem(vcek_pem)
        .map_err(|e| SnpVerifyError::ReportSignature(std::io::Error::other(format!("{e:?}"))))?;
    (&vcek, report).verify().map_err(SnpVerifyError::ReportSignature)
}
