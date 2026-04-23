use anyhow::Result;

#[derive(Debug, Clone)]
pub struct AttestationEvidence {
    pub tee_measurement: String,
    pub model_identity: String,
    pub scheme_identity: String,
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
