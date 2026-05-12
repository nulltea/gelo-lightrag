//! Attestation-issuer selection at startup.
//!
//! The runner picks exactly one issuer at boot based on `SNP_MODE`:
//! - `production` → [`gelo_tee_sev_snp::HardwareReportIssuer`] (opens
//!   `/dev/sev-guest`).
//! - `mock` → [`gelo_tee_sev_snp::mock::MockReportIssuer`] (signs against
//!   the bundled test PKI).
//!
//! Selection is fail-closed: if the requested mode's feature isn't compiled
//! in, startup aborts with an explicit message rather than silently
//! falling back.

#![allow(unused_imports)] // some imports only used under specific feature combos
use anyhow::{Result, anyhow};
use gelo_tee_sev_snp::executor::{AttestationIssuer, SnpEvidence};
use gelo_tee_sev_snp::report_data::ReportData;
use gelo_tee_sev_snp::runtime_mode::RuntimeMode;

#[allow(dead_code)] // some variants are gated behind features
pub enum IssuerHandle {
    #[cfg(all(feature = "snp", target_os = "linux"))]
    Hardware(gelo_tee_sev_snp::HardwareReportIssuer),
    #[cfg(feature = "mock")]
    Mock(gelo_tee_sev_snp::mock::MockReportIssuer),
}

impl IssuerHandle {
    pub fn for_mode(mode: RuntimeMode) -> Result<Self> {
        match mode {
            RuntimeMode::Production => {
                #[cfg(all(feature = "snp", target_os = "linux"))]
                {
                    let h = gelo_tee_sev_snp::HardwareReportIssuer::new()?;
                    tracing::info!("issuer: HardwareReportIssuer via /dev/sev-guest");
                    return Ok(Self::Hardware(h));
                }
                #[cfg(not(all(feature = "snp", target_os = "linux")))]
                {
                    Err(anyhow!(
                        "SNP_MODE=production but this binary was built without the `snp` feature \
                         (or for a non-Linux target); rebuild with --features snp"
                    ))
                }
            }
            RuntimeMode::Mock => {
                #[cfg(feature = "mock")]
                {
                    let h = gelo_tee_sev_snp::mock::MockReportIssuer::from_bundled()?;
                    tracing::warn!(
                        "issuer: MockReportIssuer (bundled test PKI). \
                         Reports from this issuer will NOT verify against AMD's production ARK."
                    );
                    return Ok(Self::Mock(h));
                }
                #[cfg(not(feature = "mock"))]
                {
                    Err(anyhow!(
                        "SNP_MODE=mock but this binary was built without the `mock` feature; \
                         rebuild with --features mock"
                    ))
                }
            }
        }
    }
}

impl AttestationIssuer for IssuerHandle {
    fn issue(&self, report_data: ReportData) -> Result<SnpEvidence> {
        match self {
            #[cfg(all(feature = "snp", target_os = "linux"))]
            IssuerHandle::Hardware(h) => AttestationIssuer::issue(h, report_data),
            #[cfg(feature = "mock")]
            IssuerHandle::Mock(m) => AttestationIssuer::issue(m, report_data),
        }
    }
}
