use anyhow::Result;

#[derive(Debug, Clone, Default)]
pub struct AttestationEvidence {
    pub tee_measurement: String,
    pub model_identity: String,
    pub scheme_identity: String,
    /// Raw 1184-byte SEV-SNP attestation report (AMD ABI Spec §7.3). `None`
    /// for backends that don't ship hardware attestation (e.g.
    /// `NoopAttestationVerifier`-paired stubs). Verifiers that need the report
    /// bytes return an error when this field is missing.
    pub report: Option<Vec<u8>>,
    /// PEM-encoded VCEK certificate that signed `report`. Populated together
    /// with `report` by SEV-SNP-capable executors.
    pub vcek_cert: Option<Vec<u8>>,
}

pub trait AttestationVerifier {
    fn verify(&self, evidence: &AttestationEvidence) -> Result<()>;
}

#[derive(Debug, Default, Clone)]
pub struct NoopAttestationVerifier;

impl AttestationVerifier for NoopAttestationVerifier {
    fn verify(&self, _evidence: &AttestationEvidence) -> Result<()> {
        Ok(())
    }
}
