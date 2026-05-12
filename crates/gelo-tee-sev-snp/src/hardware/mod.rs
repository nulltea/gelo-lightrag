//! Hardware-backed attestation: opens `/dev/sev-guest` to obtain a real
//! SEV-SNP attestation report from the AMD Secure Processor.
//!
//! Gated behind the `sev-snp` cargo feature. The `sev` crate's `Firmware`
//! API is the same surface AMD's `snpguest` userspace CLI uses; the
//! `SNP_GET_EXT_REPORT` ioctl additionally hands back the VCEK certificate
//! from the host's `/dev/sev/cert` blob so the relying party can validate
//! the full `ARK → ASK → VCEK → report` chain.
//!
//! This module compiles on any x86_64 Linux box but the ioctls themselves
//! only succeed when running inside a real SEV-SNP CVM. On a non-CVM host
//! `/dev/sev-guest` is absent and `HardwareReportIssuer::new()` returns an
//! error — the runtime-mode dispatch in [`crate::runtime_mode`] catches that
//! at startup and fails closed.

#[cfg(target_os = "linux")]
pub mod sev_guest;

#[cfg(target_os = "linux")]
pub use sev_guest::HardwareReportIssuer;
