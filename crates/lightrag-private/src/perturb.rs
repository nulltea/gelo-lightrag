//! Per-session search-pattern perturbation. Plan §8.6.
//!
//! Closes the content-level fingerprint that Compass+Ring-ORAM leave
//! open: same query embedding ⇒ same HNSW traversal ⇒ same RPC
//! count, batch sizes, prune rates. By per-session-tweaking the
//! embedding deterministically (within a session, identical) we
//! preserve cacheability while breaking cross-session linkability.
//!
//! Construction (matches the plan §8.6 spec verbatim):
//!
//! ```text
//! s_search      = HKDF.Expand(prk, "gelo-rag.v2.search-pattern")    // 8th child key
//! session_nonce = runner.fresh_nonce_16()                            // per RATLS session
//! session_key   = HMAC(s_search, session_nonce)                      // per session
//!
//! fn perturb(e, kind):
//!     h         = HMAC(session_key, kind ‖ 0x00 ‖ quantize(e))
//!     direction = unit_vector_from_32_bytes(h, dim)
//!     normalize(e + ε · direction)                                   // ε ≈ 2 %
//! ```
//!
//! Cost: one HMAC-SHA-256 + one 32-byte → D-f32 projection per
//! embedding. Negligible against the Compass critical path.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

type HmacSha256 = Hmac<Sha256>;

/// Embedding kind tag — namespaces the perturbation across the three
/// LightRAG embeddings consumed by `kg_query` (q = full query,
/// hl = high-level keyword, ll = low-level keyword). The string is
/// part of the HMAC input so two embeddings of the same content but
/// different kind get different perturbations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingKind {
    /// Full query embedding (drives the chunks-vdb search in `mix` mode).
    Q,
    /// High-level keyword embedding (drives the relations-vdb search).
    Hl,
    /// Low-level keyword embedding (drives the entities-vdb search).
    Ll,
}

impl EmbeddingKind {
    pub fn as_bytes(&self) -> &'static [u8] {
        match self {
            EmbeddingKind::Q => b"q",
            EmbeddingKind::Hl => b"hl",
            EmbeddingKind::Ll => b"ll",
        }
    }
}

/// Default perturbation magnitude (~ 2 % of unit-norm embeddings).
/// Tuned at index-build time on the parity bench — exposed so M7's
/// linkability-test fixture can override.
pub const DEFAULT_EPSILON: f32 = 0.02;

/// Quantise an f32 vector to 16-bit signed integers, byte-encoded
/// little-endian. Bins embeddings within ~3·10⁻⁵ Euclidean radius to
/// the same quantisation — embeddings in the same HNSW neighbourhood
/// hash to the same `direction`, keeping recall stable under tiny
/// CAPRISE-DPE noise.
fn quantise(e: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(e.len() * 2);
    for &x in e {
        let clamped = x.clamp(-1.0, 1.0);
        let scaled = (clamped * 32767.0).round() as i16;
        out.extend_from_slice(&scaled.to_le_bytes());
    }
    out
}

/// Expand a 32-byte HMAC tag into a `dim`-long unit-norm f32 vector.
/// SHA-256 of `(tag ‖ counter)` for successive 32-bit counters until
/// we have `dim · 4` bytes, then read as little-endian u32s ↦
/// [-1, 1] via a uniform mapping, then L2-normalise. The result is
/// uniformly distributed on the unit sphere modulo the SHA-256
/// statistical assumption.
fn unit_vector_from_32_bytes(tag: &[u8; 32], dim: usize) -> Vec<f32> {
    let need_bytes = dim * 4;
    let mut bytes = Vec::with_capacity(need_bytes.next_multiple_of(32));
    let mut counter = 0u32;
    while bytes.len() < need_bytes {
        let mut h = Sha256::new();
        h.update(tag);
        h.update(counter.to_le_bytes());
        bytes.extend_from_slice(&h.finalize());
        counter += 1;
    }
    bytes.truncate(need_bytes);

    let mut out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| {
            let n = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            // [0, 2^32) → [-1, 1)
            (n as f64 / (u32::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect();
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::MIN_POSITIVE {
        for x in &mut out {
            *x /= norm;
        }
    }
    out
}

/// Session-scoped key used to perturb every embedding within a RATLS
/// session. Derive once at session start; zeroize on session end.
#[derive(Debug)]
pub struct SessionKey {
    key: Zeroizing<[u8; 32]>,
}

impl SessionKey {
    /// `session_key = HMAC(search_pattern_key, session_nonce)`.
    pub fn derive(
        search_pattern_key: &Zeroizing<[u8; 32]>,
        session_nonce: &[u8],
    ) -> Self {
        let mut mac = HmacSha256::new_from_slice(search_pattern_key.as_ref())
            .expect("HMAC accepts 32-byte key");
        mac.update(session_nonce);
        let tag = mac.finalize().into_bytes();
        let mut key = Zeroizing::new([0u8; 32]);
        key.copy_from_slice(&tag);
        Self { key }
    }
}

/// Perturb an embedding under a session key + kind. Returns a fresh
/// `Vec<f32>` — the caller is free to drop the input.
///
/// Determinism: identical `(session_key, kind, e)` always returns
/// the same output. Distinct sessions on the same `e` return
/// near-uncorrelated outputs (Hamming distance over the underlying
/// HMAC dominates).
pub fn perturb(session_key: &SessionKey, kind: EmbeddingKind, e: &[f32]) -> Vec<f32> {
    perturb_with_epsilon(session_key, kind, e, DEFAULT_EPSILON)
}

pub fn perturb_with_epsilon(
    session_key: &SessionKey,
    kind: EmbeddingKind,
    e: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let q = quantise(e);
    let mut mac = HmacSha256::new_from_slice(session_key.key.as_ref())
        .expect("HMAC accepts 32-byte key");
    mac.update(kind.as_bytes());
    mac.update(&[0u8]);
    mac.update(&q);
    let tag = mac.finalize().into_bytes();
    let mut tag_arr = [0u8; 32];
    tag_arr.copy_from_slice(&tag);

    let direction = unit_vector_from_32_bytes(&tag_arr, e.len());

    let mut out: Vec<f32> = e
        .iter()
        .zip(direction.iter())
        .map(|(a, b)| a + epsilon * b)
        .collect();
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::MIN_POSITIVE {
        for x in &mut out {
            *x /= norm;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(seed: u8, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((seed as u32 + i as u32) as f32).sin();
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn perturb_is_deterministic_within_session() {
        let sk = SessionKey::derive(&Zeroizing::new([0x11; 32]), b"nonce-A");
        let e = unit_vec(7, 16);
        let a = perturb(&sk, EmbeddingKind::Q, &e);
        let b = perturb(&sk, EmbeddingKind::Q, &e);
        assert_eq!(a, b);
    }

    #[test]
    fn different_sessions_diverge() {
        let sk_a = SessionKey::derive(&Zeroizing::new([0x11; 32]), b"nonce-A");
        let sk_b = SessionKey::derive(&Zeroizing::new([0x11; 32]), b"nonce-B");
        let e = unit_vec(7, 32);
        let a = perturb(&sk_a, EmbeddingKind::Q, &e);
        let b = perturb(&sk_b, EmbeddingKind::Q, &e);
        // Outputs must differ in at least one component. Stronger: the
        // cosine similarity is not exactly 1.
        let cos: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (cos - 1.0).abs() > 1e-4,
            "two sessions produced near-identical perturbations: cos={cos}"
        );
    }

    #[test]
    fn different_kinds_diverge_within_session() {
        let sk = SessionKey::derive(&Zeroizing::new([0x22; 32]), b"nonce");
        let e = unit_vec(7, 32);
        let q = perturb(&sk, EmbeddingKind::Q, &e);
        let ll = perturb(&sk, EmbeddingKind::Ll, &e);
        let hl = perturb(&sk, EmbeddingKind::Hl, &e);
        assert_ne!(q, ll);
        assert_ne!(q, hl);
        assert_ne!(ll, hl);
    }

    #[test]
    fn perturbed_vector_is_unit_norm() {
        let sk = SessionKey::derive(&Zeroizing::new([0x33; 32]), b"n");
        let e = unit_vec(3, 24);
        let p = perturb(&sk, EmbeddingKind::Q, &e);
        let norm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    #[test]
    fn perturbation_is_small_at_default_epsilon() {
        // ε = 2 % so cosine similarity to the original should be
        // roughly cos(ε) ≈ 1 - ε²/2 ≈ 0.9998. Allow 99.5 % to leave
        // headroom for the quantisation step.
        let sk = SessionKey::derive(&Zeroizing::new([0x44; 32]), b"n");
        let e = unit_vec(11, 64);
        let p = perturb(&sk, EmbeddingKind::Q, &e);
        let cos: f32 = e.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
        assert!(cos > 0.995, "perturbation too aggressive: cos={cos}");
    }
}
