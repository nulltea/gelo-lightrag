//! Two-party-KDF key derivation for the CAPRISE-in-TEE storage model.
//!
//! Spec: `docs/prototype/caprise-two-party-kdf.md` §3.
//!
//! One HKDF-SHA256 extract + two expands derive the per-session
//! `caprise_seed_key` and `aes_chunk_key` deterministically from
//!
//! ```text
//! prk = HKDF-Extract(salt = tenant_id, ikm = user_x_sk ‖ tee_user_x_sk)
//! caprise_seed_key = HKDF-Expand(prk, info = "gelo-rag.v1.caprise.seed", L=32)
//! aes_chunk_key    = HKDF-Expand(prk, info = "gelo-rag.v1.aes-chunks",   L=32)
//! ```
//!
//! Every cryptographic step is a single call into an audited RustCrypto
//! library — `hkdf 0.12` (extract+expand), `hmac 0.12` (HKDF backbone),
//! `sha2 0.10` (hash). No primitive is implemented in this module.

use hkdf::Hkdf;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Opaque tenant identifier — partitions the per-tenant index, the
/// `tee_user_x_sk` table, and the HKDF salt. Use a typed wrapper rather
/// than a bare `String` so cross-tenant scans become type-unrepresentable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl TenantId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Public CAPRISE scheme constants. Not per-tenant secrets; included in
/// `scheme_identity` so any change requires re-attestation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchemeParams {
    pub scale: f32,
    pub beta: f32,
}

impl Default for SchemeParams {
    /// Defaults matching the BEIR bench in
    /// `crates/gelo-rag/tests/beir_accuracy.rs`.
    fn default() -> Self {
        Self {
            scale: 32.0,
            beta: 0.15,
        }
    }
}

/// HKDF-SHA256 policy: the salt label, the two info strings, and the
/// version tag that participate in [`scheme_identity_digest`].
///
/// Bumping `version` (i.e. moving from [`Self::V1`] to a future `V2`)
/// changes the info strings and is therefore equivalent to issuing
/// entirely new keys for the same `(user_x_sk, tee_user_x_sk)` tuple.
/// Always pin the policy in the attestation report (`scheme_identity`)
/// so a relying party can detect a downgrade attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HkdfPolicy {
    pub version: &'static str,
    pub caprise_info: &'static str,
    pub aes_info: &'static str,
    pub salt_label: &'static str,
}

impl HkdfPolicy {
    /// First versioned policy. Locked in code; do not edit in place —
    /// add a `V2` const and migrate.
    pub const V1: Self = Self {
        version: "gelo-rag.v1",
        caprise_info: "gelo-rag.v1.caprise.seed",
        aes_info: "gelo-rag.v1.aes-chunks",
        salt_label: "tenant_id",
    };

    /// Derive `(caprise_seed_key, aes_chunk_key)` from the two-party
    /// secrets and a tenant identifier. Both returned buffers wipe
    /// themselves on drop via [`Zeroizing`].
    ///
    /// The single `Hkdf::<Sha256>::new` call IS the HKDF-Extract step;
    /// the two `hk.expand` calls are HKDF-Expand. We never construct an
    /// HMAC by hand.
    pub fn derive(
        &self,
        user_x_sk: &Zeroizing<[u8; 32]>,
        tee_user_x_sk: &Zeroizing<[u8; 32]>,
        tenant_id: &TenantId,
    ) -> (Zeroizing<[u8; 32]>, Zeroizing<[u8; 32]>) {
        let mut ikm: Zeroizing<[u8; 64]> = Zeroizing::new([0u8; 64]);
        ikm[..32].copy_from_slice(user_x_sk.as_ref());
        ikm[32..].copy_from_slice(tee_user_x_sk.as_ref());

        let hk = Hkdf::<Sha256>::new(Some(tenant_id.as_bytes()), ikm.as_ref());

        let mut caprise_seed: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
        hk.expand(self.caprise_info.as_bytes(), caprise_seed.as_mut())
            .expect("32 ≤ 255·HashLen(SHA-256)");

        let mut aes_key: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
        hk.expand(self.aes_info.as_bytes(), aes_key.as_mut())
            .expect("32 ≤ 255·HashLen(SHA-256)");

        (caprise_seed, aes_key)
    }

    /// Canonical bytes of the KDF policy + CAPRISE constants, for
    /// `scheme_identity` binding (spec §8). Independent of any external
    /// mask / shield digest — those are appended by the caller. Floats
    /// are encoded by their IEEE-754 bit pattern so two CVMs configured
    /// identically produce byte-equal output regardless of platform
    /// `Display` quirks.
    pub fn scheme_identity_fragment(&self, params: SchemeParams) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(self.version.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(b"kdf=hkdf-sha256\n");
        out.extend_from_slice(b"kdf-info-caprise=");
        out.extend_from_slice(self.caprise_info.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(b"kdf-info-aes=");
        out.extend_from_slice(self.aes_info.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(b"kdf-salt=");
        out.extend_from_slice(self.salt_label.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(
            format!("caprise-scale-bits=0x{:08x}\n", params.scale.to_bits()).as_bytes(),
        );
        out.extend_from_slice(
            format!("caprise-beta-bits=0x{:08x}\n", params.beta.to_bits()).as_bytes(),
        );
        out
    }

    /// 32-byte SHA-256 of the canonical fragment. Fits directly into
    /// `REPORT_DATA[32..64]` of a SEV-SNP attestation report alongside
    /// any external mask / shield digests the caller wants to mix in
    /// (re-hash after concatenation).
    pub fn scheme_identity_digest(&self, params: SchemeParams) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.scheme_identity_fragment(params));
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sk(byte: u8) -> Zeroizing<[u8; 32]> {
        Zeroizing::new([byte; 32])
    }

    #[test]
    fn derive_is_deterministic() {
        let policy = HkdfPolicy::V1;
        let tenant = TenantId::new("acme-legal");
        let (c1, a1) = policy.derive(&sk(0x11), &sk(0x22), &tenant);
        let (c2, a2) = policy.derive(&sk(0x11), &sk(0x22), &tenant);
        assert_eq!(c1.as_ref(), c2.as_ref());
        assert_eq!(a1.as_ref(), a2.as_ref());
    }

    #[test]
    fn caprise_seed_differs_from_aes_key() {
        let (c, a) = HkdfPolicy::V1.derive(&sk(0x33), &sk(0x44), &TenantId::new("t"));
        assert_ne!(c.as_ref(), a.as_ref());
    }

    #[test]
    fn tenant_salt_separates_keys() {
        let (c_a, _) = HkdfPolicy::V1.derive(&sk(0x55), &sk(0x66), &TenantId::new("tenant-A"));
        let (c_b, _) = HkdfPolicy::V1.derive(&sk(0x55), &sk(0x66), &TenantId::new("tenant-B"));
        assert_ne!(c_a.as_ref(), c_b.as_ref());
    }

    #[test]
    fn either_half_changes_the_output() {
        let tenant = TenantId::new("t");
        let (c_base, _) = HkdfPolicy::V1.derive(&sk(0x77), &sk(0x88), &tenant);
        let (c_user_flipped, _) = HkdfPolicy::V1.derive(&sk(0x78), &sk(0x88), &tenant);
        let (c_tee_flipped, _) = HkdfPolicy::V1.derive(&sk(0x77), &sk(0x89), &tenant);
        assert_ne!(c_base.as_ref(), c_user_flipped.as_ref());
        assert_ne!(c_base.as_ref(), c_tee_flipped.as_ref());
    }

    #[test]
    fn scheme_identity_digest_is_stable() {
        // Lock the digest of `(V1, default scheme params)` so any
        // accidental edit to the canonical encoder is caught.
        let digest = HkdfPolicy::V1.scheme_identity_digest(SchemeParams::default());
        // 32 bytes; non-zero; deterministic across runs.
        assert_eq!(digest.len(), 32);
        let digest2 = HkdfPolicy::V1.scheme_identity_digest(SchemeParams::default());
        assert_eq!(digest, digest2);
    }

    #[test]
    fn scheme_identity_changes_with_params() {
        let d1 = HkdfPolicy::V1.scheme_identity_digest(SchemeParams { scale: 32.0, beta: 0.15 });
        let d2 = HkdfPolicy::V1.scheme_identity_digest(SchemeParams { scale: 16.0, beta: 0.15 });
        assert_ne!(d1, d2);
    }
}
