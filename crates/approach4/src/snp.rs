//! SEV-SNP attestation adapter.
//!
//! Bridges [`gelo_tee_sev_snp::SnpAttestationVerifier`] into the
//! [`AttestationVerifier`] trait used by [`Approach4InMemoryService`].
//!
//! The adapter pulls `report` and `vcek_cert` from
//! [`AttestationEvidence`] (added as non-breaking `Option` fields in M5.5)
//! and feeds them through the underlying verifier together with the
//! `(model_identity, scheme_identity)` pair the relying party expects.
//!
//! Compiled only when the `snp` feature is enabled so the default
//! `approach4` build doesn't drag in the SEV-SNP dependency graph.

use anyhow::{Result, anyhow};
use gelo_tee_sev_snp::{AttestedBinding, SnpAttestationVerifier};

use crate::attestation::{AttestationEvidence, AttestationVerifier};

/// Wraps [`SnpAttestationVerifier`] and exposes it through the workspace's
/// [`AttestationVerifier`] trait.
pub struct SnpVerifierAdapter {
    inner: SnpAttestationVerifier,
}

impl SnpVerifierAdapter {
    pub fn new(inner: SnpAttestationVerifier) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &SnpAttestationVerifier {
        &self.inner
    }
}

impl From<SnpAttestationVerifier> for SnpVerifierAdapter {
    fn from(inner: SnpAttestationVerifier) -> Self {
        Self::new(inner)
    }
}

impl AttestationVerifier for SnpVerifierAdapter {
    fn verify(&self, evidence: &AttestationEvidence) -> Result<()> {
        let report = evidence
            .report
            .as_deref()
            .ok_or_else(|| anyhow!("AttestationEvidence.report is missing — SnpVerifierAdapter requires SEV-SNP report bytes"))?;
        let vcek = evidence
            .vcek_cert
            .as_deref()
            .ok_or_else(|| anyhow!("AttestationEvidence.vcek_cert is missing — SnpVerifierAdapter requires a VCEK certificate"))?;
        let binding = AttestedBinding {
            model_identity: evidence.model_identity.as_bytes(),
            scheme_identity: evidence.scheme_identity.as_bytes(),
            nonce: None,
        };
        self.inner
            .verify(report, vcek, binding)
            .map(|_| ())
            .map_err(|e| anyhow!("SEV-SNP attestation verification failed: {e}"))
    }
}
