//! Per-session and per-query HKDF key derivation for the rerank
//! re-encryption path (`docs/research/private-reranking-research-round-2.md`
//! §3 + the follow-up conversation on TEE-internal reordering).
//!
//! Composition mirrors `rag_core::keying::HkdfPolicy`:
//!
//! ```text
//! SessionKey = HKDF-SHA256(
//!     salt = b"gelo-rerank.session.v1",
//!     ikm  = client_TEE_shared_secret,
//!     info = "gelo-rerank.session.v1",
//! )
//!
//! QueryKey   = HKDF-SHA256(
//!     salt = SessionKey,
//!     ikm  = query_id,
//!     info = "gelo-rerank.query.v1",
//! )
//! ```
//!
//! Both keys are 32 bytes and self-wipe on drop (`Zeroizing`). The
//! attestation report's `REPORT_DATA[32..64]` is the place to bind the
//! session-key context to the running CVM identity, exactly as
//! `HkdfPolicy::scheme_identity_digest` does for CAPRISE — keep both
//! contributions disjoint so any rerank-side change is detectable.

use hkdf::Hkdf;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use zeroize::Zeroizing;

/// Opaque per-query identifier. Must be unique for the lifetime of a
/// session — replays of the same `(SessionKey, QueryId)` would derive
/// the same [`QueryKey`] and reuse nonces under it, which AES-GCM does
/// not survive. Caller picks a monotonic counter or a random 128-bit
/// token.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(pub Vec<u8>);

impl QueryId {
    pub fn new(b: impl Into<Vec<u8>>) -> Self {
        Self(b.into())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl From<&[u8]> for QueryId {
    fn from(b: &[u8]) -> Self {
        Self(b.to_owned())
    }
}

impl From<&str> for QueryId {
    fn from(s: &str) -> Self {
        Self(s.as_bytes().to_owned())
    }
}

/// HKDF info-string + salt-label policy. Bumping `version` is equivalent
/// to issuing entirely new keys for the same `(shared_secret, query_id)`
/// tuple — pin it in the attestation report so relying parties can
/// detect a downgrade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionKeyPolicy {
    pub version: &'static str,
    pub session_salt: &'static [u8],
    pub session_info: &'static [u8],
    pub query_info: &'static [u8],
}

impl SessionKeyPolicy {
    /// First versioned policy. Locked in code — do not edit in place;
    /// add a `V2` const and migrate.
    pub const V1: Self = Self {
        version: "gelo-rerank.v1",
        session_salt: b"gelo-rerank.session.v1",
        session_info: b"gelo-rerank.session.v1",
        query_info: b"gelo-rerank.query.v1",
    };
}

/// 32-byte session root. Wraps the post-HKDF bytes in [`Zeroizing`] so
/// they're wiped on drop.
#[derive(Clone)]
pub struct SessionKey {
    bytes: Zeroizing<[u8; 32]>,
    policy: SessionKeyPolicy,
}

impl std::fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionKey")
            .field("bytes", &"<redacted 32B>")
            .field("policy", &self.policy.version)
            .finish()
    }
}

impl SessionKey {
    /// Derive a session root from a client/TEE-shared secret (e.g. the
    /// output of an ECDH key agreement performed during the attestation
    /// handshake). The secret is consumed via [`Zeroizing`] so callers
    /// can't accidentally retain it.
    pub fn derive(
        shared_secret: &Zeroizing<Vec<u8>>,
        policy: SessionKeyPolicy,
    ) -> Self {
        let hk = Hkdf::<Sha256>::new(Some(policy.session_salt), shared_secret.as_ref());
        let mut out: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
        hk.expand(policy.session_info, out.as_mut())
            .expect("32 ≤ 255·HashLen(SHA-256)");
        Self {
            bytes: out,
            policy,
        }
    }

    /// Test-only constructor. Marked `#[cfg(any(test, feature = "...))]`
    /// would force a feature gate; we just leave it as a regular pub
    /// item used only inside the workspace, because every caller other
    /// than the production attestation handshake is a test.
    #[doc(hidden)]
    pub fn from_bytes_for_tests(bytes: [u8; 32], policy: SessionKeyPolicy) -> Self {
        Self {
            bytes: Zeroizing::new(bytes),
            policy,
        }
    }

    pub fn policy(&self) -> SessionKeyPolicy {
        self.policy
    }

    /// Derive a per-query AEAD key. `query_id` rides as the HKDF IKM;
    /// the session bytes are the salt. Same query_id under the same
    /// session yields the same QueryKey — caller must guarantee unique
    /// query_ids per session.
    pub fn derive_query_key(&self, query_id: &QueryId) -> QueryKey {
        let hk = Hkdf::<Sha256>::new(Some(self.bytes.as_ref()), query_id.as_bytes());
        let mut out: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
        hk.expand(self.policy.query_info, out.as_mut())
            .expect("32 ≤ 255·HashLen(SHA-256)");
        QueryKey { bytes: out }
    }
}

/// 32-byte AES-256-GCM key for one rerank response. Self-wipes on drop.
#[derive(Clone)]
pub struct QueryKey {
    bytes: Zeroizing<[u8; 32]>,
}

impl std::fmt::Debug for QueryKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryKey").field("bytes", &"<redacted 32B>").finish()
    }
}

impl QueryKey {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Test-only constructor mirroring `SessionKey::from_bytes_for_tests`.
    #[doc(hidden)]
    pub fn from_bytes_for_tests(bytes: [u8; 32]) -> Self {
        Self { bytes: Zeroizing::new(bytes) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secret(byte: u8, len: usize) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(vec![byte; len])
    }

    #[test]
    fn session_derive_is_deterministic() {
        let s = secret(0x11, 32);
        let a = SessionKey::derive(&s, SessionKeyPolicy::V1);
        let b = SessionKey::derive(&s, SessionKeyPolicy::V1);
        assert_eq!(a.bytes.as_ref(), b.bytes.as_ref());
    }

    #[test]
    fn session_differs_with_secret() {
        let a = SessionKey::derive(&secret(0x11, 32), SessionKeyPolicy::V1);
        let b = SessionKey::derive(&secret(0x12, 32), SessionKeyPolicy::V1);
        assert_ne!(a.bytes.as_ref(), b.bytes.as_ref());
    }

    #[test]
    fn query_key_is_deterministic_per_session() {
        let s = SessionKey::derive(&secret(0x22, 48), SessionKeyPolicy::V1);
        let q1 = s.derive_query_key(&QueryId::from("query-1"));
        let q2 = s.derive_query_key(&QueryId::from("query-1"));
        assert_eq!(q1.as_bytes(), q2.as_bytes());
    }

    #[test]
    fn distinct_query_ids_yield_distinct_keys() {
        let s = SessionKey::derive(&secret(0x22, 48), SessionKeyPolicy::V1);
        let a = s.derive_query_key(&QueryId::from("query-1"));
        let b = s.derive_query_key(&QueryId::from("query-2"));
        assert_ne!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn distinct_sessions_yield_distinct_query_keys() {
        let s1 = SessionKey::derive(&secret(0x33, 48), SessionKeyPolicy::V1);
        let s2 = SessionKey::derive(&secret(0x34, 48), SessionKeyPolicy::V1);
        let qid = QueryId::from("same");
        assert_ne!(s1.derive_query_key(&qid).as_bytes(), s2.derive_query_key(&qid).as_bytes());
    }
}
