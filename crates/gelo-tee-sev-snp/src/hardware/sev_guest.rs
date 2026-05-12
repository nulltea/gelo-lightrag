//! `HardwareReportIssuer` — talks to `/dev/sev-guest` via the `virtee/sev`
//! `Firmware` API.
//!
//! Calls `SNP_GET_EXT_REPORT`, which returns both the 1184-byte attestation
//! report (signed by the per-chip VCEK) **and** the cert table containing
//! the matching VCEK PEM. That's the same shape `SnpAttestationVerifier`
//! consumes via the [`AttestationIssuer`] trait, so the production and
//! mock paths feed into identical verifier code.
//!
//! This module compiles on any x86_64 Linux box but the underlying ioctls
//! only succeed inside a real SEV-SNP guest. On a non-CVM host the device
//! file is absent and `new()` returns an `Err`; the runtime-mode dispatch
//! ([`crate::runtime_mode`]) catches that at process startup so we
//! fail-closed before any embedding work begins.

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use sev::firmware::guest::Firmware;
use sev::firmware::host::CertType;

use crate::executor::{AttestationIssuer, SnpEvidence};
use crate::report_data::ReportData;

/// Opens `/dev/sev-guest` once at construction and reuses the handle for
/// every `issue()` call. The `Firmware` handle is `!Sync` (it wraps a
/// `std::fs::File` mutated by ioctls), so we wrap it in a `Mutex` to satisfy
/// the [`AttestationIssuer: Send + Sync`] bound.
pub struct HardwareReportIssuer {
    firmware: Mutex<Firmware>,
    /// VMPL the report is requested for. SEV-SNP supports up to four VMPLs
    /// per CVM (0 = highest privilege); we default to 0, matching how a
    /// non-paravisor-using CVM is launched.
    vmpl: u32,
}

impl HardwareReportIssuer {
    /// Open `/dev/sev-guest`. Fails if the device file is absent (non-CVM
    /// host) or the calling user lacks permission to open it.
    pub fn new() -> Result<Self> {
        Self::with_vmpl(0)
    }

    /// Same as [`Self::new`] but lets the caller pin a specific VMPL. Useful
    /// when the CVM uses a paravisor at VMPL 0 and the workload runs at
    /// VMPL 1.
    pub fn with_vmpl(vmpl: u32) -> Result<Self> {
        let firmware = Firmware::open()
            .context("opening /dev/sev-guest (is this binary running inside a SEV-SNP CVM?)")?;
        Ok(Self {
            firmware: Mutex::new(firmware),
            vmpl,
        })
    }

    /// VMPL this issuer pins reports to.
    pub fn vmpl(&self) -> u32 {
        self.vmpl
    }
}

impl AttestationIssuer for HardwareReportIssuer {
    fn issue(&self, report_data: ReportData) -> Result<SnpEvidence> {
        let mut fw = self
            .firmware
            .lock()
            .map_err(|_| anyhow!("HardwareReportIssuer firmware mutex poisoned"))?;

        // The Firmware::get_ext_report ioctl returns the 1184-byte report
        // plus an optional cert table populated by the host (the EXT path
        // uses the host's `/dev/sev/cert` blob).
        let (report_bytes, certs) = fw
            .get_ext_report(None, Some(*report_data.as_bytes()), Some(self.vmpl))
            .map_err(|e| anyhow!("SNP_GET_EXT_REPORT failed: {e:?}"))?;

        let certs = certs
            .ok_or_else(|| anyhow!("SNP_GET_EXT_REPORT returned no cert table; host /dev/sev/cert may be unconfigured"))?;
        let vcek_cert_pem = certs
            .iter()
            .find(|c| matches!(c.cert_type, CertType::VCEK))
            .ok_or_else(|| anyhow!("cert table from /dev/sev-guest contains no VCEK entry"))?
            .data
            .clone();

        Ok(SnpEvidence {
            report_bytes,
            vcek_cert_pem,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On non-EPYC dev boxes the device file is absent; `new()` must fail
    /// rather than panic or silently return a broken issuer.
    #[test]
    fn open_fails_outside_cvm() {
        if std::path::Path::new("/dev/sev-guest").exists() {
            eprintln!("skipping: /dev/sev-guest is present on this host");
            return;
        }
        let r = HardwareReportIssuer::new();
        assert!(r.is_err(), "should fail without /dev/sev-guest");
    }
}
