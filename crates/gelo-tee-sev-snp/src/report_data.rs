//! Construction of the 64-byte `REPORT_DATA` field that the CVM stamps into
//! every SEV-SNP attestation report.
//!
//! Layout (locked-in by a deterministic test vector in `tests/`):
//!
//! ```text
//! REPORT_DATA[0..32]  = sha256(model_identity_bytes)
//! REPORT_DATA[32..64] = sha256(scheme_identity_bytes || optional_nonce)
//! ```
//!
//! Splitting model and scheme identity into separate 32-byte halves means the
//! relying party can pin either independently:
//!
//! - `model_identity` (left half) is the **publicly-known** SHA-256 of the
//!   loaded weight manifest. Openweight deployments — the only kind GELO
//!   targets — let the relying party verify "the CVM is processing requests
//!   with these specific public weights" without any private model material.
//! - `scheme_identity` (right half) covers the protocol-secret state
//!   (`MaskSeed` + `ShieldConfig`), optionally bound to a per-session
//!   challenge nonce.

use sha2::{Digest, Sha256};

/// The 64-byte field that goes into the SEV-SNP attestation report's
/// `REPORT_DATA` slot. Caller-controlled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReportData(pub [u8; 64]);

impl ReportData {
    /// Construct from a public model identity and a (possibly secret) scheme
    /// identity. An optional nonce binds the report to a session challenge.
    pub fn build(
        model_identity: &[u8],
        scheme_identity: &[u8],
        nonce: Option<&[u8]>,
    ) -> Self {
        let mut out = [0u8; 64];
        let model_hash: [u8; 32] = Sha256::digest(model_identity).into();
        out[..32].copy_from_slice(&model_hash);

        let mut scheme_hasher = Sha256::new();
        scheme_hasher.update(scheme_identity);
        if let Some(n) = nonce {
            scheme_hasher.update(n);
        }
        let scheme_hash: [u8; 32] = scheme_hasher.finalize().into();
        out[32..].copy_from_slice(&scheme_hash);
        Self(out)
    }

    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }

    pub fn model_id_hash(&self) -> &[u8] {
        &self.0[..32]
    }

    pub fn scheme_id_hash(&self) -> &[u8] {
        &self.0[32..]
    }
}

impl From<[u8; 64]> for ReportData {
    fn from(b: [u8; 64]) -> Self {
        Self(b)
    }
}

impl From<ReportData> for [u8; 64] {
    fn from(r: ReportData) -> Self {
        r.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locked-in test vector: any change to the layout breaks this.
    #[test]
    fn report_data_layout_is_stable() {
        let rd = ReportData::build(b"Qwen/Qwen3-Embedding-0.6B@main", b"scheme-v1", None);
        // sha256("Qwen/Qwen3-Embedding-0.6B@main") and
        // sha256("scheme-v1") concatenated, no nonce.
        let model_h = Sha256::digest(b"Qwen/Qwen3-Embedding-0.6B@main");
        let scheme_h = Sha256::digest(b"scheme-v1");
        assert_eq!(rd.model_id_hash(), &model_h[..]);
        assert_eq!(rd.scheme_id_hash(), &scheme_h[..]);
    }

    #[test]
    fn nonce_changes_scheme_half_only() {
        let a = ReportData::build(b"m", b"s", None);
        let b = ReportData::build(b"m", b"s", Some(b"nonce-1"));
        assert_eq!(a.model_id_hash(), b.model_id_hash());
        assert_ne!(a.scheme_id_hash(), b.scheme_id_hash());
    }

    #[test]
    fn different_model_ids_change_left_half() {
        let a = ReportData::build(b"model-A", b"s", None);
        let b = ReportData::build(b"model-B", b"s", None);
        assert_ne!(a.model_id_hash(), b.model_id_hash());
        assert_eq!(a.scheme_id_hash(), b.scheme_id_hash());
    }
}
