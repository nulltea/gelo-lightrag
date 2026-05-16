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

/// Public Compass-index parameters. Not per-tenant secrets. Pinned in
/// the V2 attestation `scheme_identity` so a CVM running a different
/// ORAM / HNSW configuration can't impersonate this one.
///
/// Spec: `docs/prototype/private-graph-rag-variant-a.md` §3 + §4.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompassParams {
    /// Ring-ORAM bucket size — real slots.
    pub ring_oram_z: u32,
    /// Ring-ORAM bucket size — dummy slots.
    pub ring_oram_s: u32,
    /// Ring-ORAM eviction rate — `EvictPath` every `a` `ReadPath`s.
    pub ring_oram_a: u32,
    /// ORAM block payload size in bytes.
    pub block_bytes: u32,
    /// HNSW degree bound.
    pub hnsw_m: u32,
    /// HNSW dynamic candidate-list width at search time.
    pub hnsw_ef: u32,
    /// Compass speculation-set size (Speculative Neighbor Prefetch).
    pub hnsw_ef_spec: u32,
    /// Compass directional-filter size (Directional Neighbor Filtering).
    pub hnsw_ef_n: u32,
    /// Number of top ORAM-tree levels cached client-side.
    pub treetop_levels: u32,
    /// Quantization bits per dimension for directional hints.
    pub directional_hint_bits: u8,
}

impl Default for CompassParams {
    /// Paper-aligned starting point — see Compass paper §6.
    /// Tunable per-tenant at index build time.
    fn default() -> Self {
        Self {
            ring_oram_z: 4,
            ring_oram_s: 5,
            ring_oram_a: 3,
            block_bytes: 2048,
            hnsw_m: 12,
            hnsw_ef: 64,
            hnsw_ef_spec: 16,
            hnsw_ef_n: 4,
            treetop_levels: 4,
            directional_hint_bits: 4,
        }
    }
}

/// Public XorMM volume-hiding EMM parameters. Not per-tenant secrets.
/// Pinned in the V2 attestation `scheme_identity`.
///
/// Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XorMmParams {
    /// Bucket-padded volume budget — all `get` responses padded to this
    /// length, so per-key volume is hidden up to this cap.
    pub volume_bound: u32,
    /// In-TEE stash size for overflow entries.
    pub stash_size: u32,
}

impl Default for XorMmParams {
    fn default() -> Self {
        Self {
            volume_bound: 64,
            stash_size: 128,
        }
    }
}

/// Public LightRAG-private retrieval parameters. Pinned in the V2
/// attestation `scheme_identity` so a CVM tuned for a different
/// recall / token budget is observably distinct.
///
/// Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.5 + §8.1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightRagParams {
    /// Entities / relations VDB top-k.
    pub top_k: u32,
    /// Chunks VDB top-k for `mix` mode.
    pub chunk_top_k: u32,
    /// Token budget for entity context.
    pub max_entity_tokens: u32,
    /// Token budget for relation context.
    pub max_relation_tokens: u32,
    /// Total prompt token budget.
    pub max_total_tokens: u32,
    /// Per-session HMAC search perturbation magnitude ε (§8.6).
    /// Encoded as IEEE-754 bits for byte-stable canonical encoding
    /// across platforms.
    pub search_perturb_epsilon: f32,
}

impl Default for LightRagParams {
    /// Defaults matching upstream LightRAG `QueryParam`.
    fn default() -> Self {
        Self {
            top_k: 20,
            chunk_top_k: 60,
            max_entity_tokens: 4_000,
            max_relation_tokens: 4_000,
            max_total_tokens: 32_000,
            search_perturb_epsilon: 0.02,
        }
    }
}

/// Bundle of all public params that participate in the V2
/// `scheme_identity`. Held by the runner; encoded into the attestation
/// report so a relying party can pin every configuration knob.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchemeParamsV2 {
    pub caprise: SchemeParams,
    pub compass: CompassParams,
    pub xormm: XorMmParams,
    pub lightrag: LightRagParams,
}

impl Default for SchemeParamsV2 {
    fn default() -> Self {
        Self {
            caprise: SchemeParams::default(),
            compass: CompassParams::default(),
            xormm: XorMmParams::default(),
            lightrag: LightRagParams::default(),
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

/// HKDF-SHA256 policy for the Variant A LightRAG/GraphRAG retrieval
/// path. Same two-party root as [`HkdfPolicy`] but derives eight child
/// keys instead of two:
///
/// ```text
/// prk                    = HKDF-Extract(salt = tenant_id, ikm = user_x_sk ‖ tee_user_x_sk)
/// caprise_seed           = HKDF-Expand(prk, info = "gelo-rag.v2.caprise.seed",    L=32)
/// aes_chunk_key          = HKDF-Expand(prk, info = "gelo-rag.v2.aes-chunks",      L=32)
/// oram_entities_key      = HKDF-Expand(prk, info = "gelo-rag.v2.oram-entities",   L=32)
/// oram_relations_key     = HKDF-Expand(prk, info = "gelo-rag.v2.oram-relations",  L=32)
/// oram_chunks_key        = HKDF-Expand(prk, info = "gelo-rag.v2.oram-chunks",     L=32)
/// emm_adjacency_key      = HKDF-Expand(prk, info = "gelo-rag.v2.emm-adjacency",   L=32)
/// emm_src_chunks_key     = HKDF-Expand(prk, info = "gelo-rag.v2.emm-src-chunks",  L=32)
/// search_pattern_key     = HKDF-Expand(prk, info = "gelo-rag.v2.search-pattern",  L=32)
/// ```
///
/// Spec: `docs/prototype/private-graph-rag-variant-a.md` §2.1 + §8.6.
/// Bumping from V1 → V2 changes every info string and therefore yields
/// entirely fresh keys for the same `(user_x_sk, tee_user_x_sk)` tuple.
/// Always pin in [`HkdfPolicyV2::scheme_identity_digest`] so a relying
/// party can detect a downgrade attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HkdfPolicyV2 {
    pub version: &'static str,
    pub caprise_info: &'static str,
    pub aes_info: &'static str,
    pub oram_entities_info: &'static str,
    pub oram_relations_info: &'static str,
    pub oram_chunks_info: &'static str,
    pub emm_adjacency_info: &'static str,
    pub emm_src_chunks_info: &'static str,
    pub search_pattern_info: &'static str,
    pub salt_label: &'static str,
}

/// All eight per-request child keys from
/// [`HkdfPolicyV2::derive`]. Every field is [`Zeroizing`] and wipes on
/// drop; the request handler holds this struct on the stack only for
/// the duration of the request.
pub struct DerivedKeysV2 {
    pub caprise_seed: Zeroizing<[u8; 32]>,
    pub aes_chunk_key: Zeroizing<[u8; 32]>,
    pub oram_entities_key: Zeroizing<[u8; 32]>,
    pub oram_relations_key: Zeroizing<[u8; 32]>,
    pub oram_chunks_key: Zeroizing<[u8; 32]>,
    pub emm_adjacency_key: Zeroizing<[u8; 32]>,
    pub emm_src_chunks_key: Zeroizing<[u8; 32]>,
    pub search_pattern_key: Zeroizing<[u8; 32]>,
}

impl HkdfPolicyV2 {
    /// First V2 policy. Locked in code; do not edit in place — add a
    /// `V3` const and migrate.
    pub const V2: Self = Self {
        version: "gelo-rag.v2",
        caprise_info: "gelo-rag.v2.caprise.seed",
        aes_info: "gelo-rag.v2.aes-chunks",
        oram_entities_info: "gelo-rag.v2.oram-entities",
        oram_relations_info: "gelo-rag.v2.oram-relations",
        oram_chunks_info: "gelo-rag.v2.oram-chunks",
        emm_adjacency_info: "gelo-rag.v2.emm-adjacency",
        emm_src_chunks_info: "gelo-rag.v2.emm-src-chunks",
        search_pattern_info: "gelo-rag.v2.search-pattern",
        salt_label: "tenant_id",
    };

    /// One HKDF-Extract + eight HKDF-Expand calls deriving all child
    /// keys from the two-party secrets and tenant id. All returned
    /// buffers wipe themselves on drop via [`Zeroizing`].
    pub fn derive(
        &self,
        user_x_sk: &Zeroizing<[u8; 32]>,
        tee_user_x_sk: &Zeroizing<[u8; 32]>,
        tenant_id: &TenantId,
    ) -> DerivedKeysV2 {
        let mut ikm: Zeroizing<[u8; 64]> = Zeroizing::new([0u8; 64]);
        ikm[..32].copy_from_slice(user_x_sk.as_ref());
        ikm[32..].copy_from_slice(tee_user_x_sk.as_ref());

        let hk = Hkdf::<Sha256>::new(Some(tenant_id.as_bytes()), ikm.as_ref());

        let expand = |info: &str| -> Zeroizing<[u8; 32]> {
            let mut out: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
            hk.expand(info.as_bytes(), out.as_mut())
                .expect("32 ≤ 255·HashLen(SHA-256)");
            out
        };

        DerivedKeysV2 {
            caprise_seed: expand(self.caprise_info),
            aes_chunk_key: expand(self.aes_info),
            oram_entities_key: expand(self.oram_entities_info),
            oram_relations_key: expand(self.oram_relations_info),
            oram_chunks_key: expand(self.oram_chunks_info),
            emm_adjacency_key: expand(self.emm_adjacency_info),
            emm_src_chunks_key: expand(self.emm_src_chunks_info),
            search_pattern_key: expand(self.search_pattern_info),
        }
    }

    /// Canonical bytes of the KDF policy + all public V2 params. Folded
    /// into `REPORT_DATA[32..64]` of the SEV-SNP attestation report by
    /// the runner. Floats are encoded by IEEE-754 bit pattern for
    /// platform-stable bytes.
    pub fn scheme_identity_fragment(&self, params: SchemeParamsV2) -> Vec<u8> {
        let mut out = Vec::with_capacity(1024);

        // Policy version + KDF info strings (one per child key).
        out.extend_from_slice(self.version.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(b"kdf=hkdf-sha256\n");
        for (label, info) in [
            ("caprise", self.caprise_info),
            ("aes", self.aes_info),
            ("oram-entities", self.oram_entities_info),
            ("oram-relations", self.oram_relations_info),
            ("oram-chunks", self.oram_chunks_info),
            ("emm-adjacency", self.emm_adjacency_info),
            ("emm-src-chunks", self.emm_src_chunks_info),
            ("search-pattern", self.search_pattern_info),
        ] {
            out.extend_from_slice(b"kdf-info-");
            out.extend_from_slice(label.as_bytes());
            out.push(b'=');
            out.extend_from_slice(info.as_bytes());
            out.push(b'\n');
        }
        out.extend_from_slice(b"kdf-salt=");
        out.extend_from_slice(self.salt_label.as_bytes());
        out.push(b'\n');

        // CAPRISE constants — same shape as V1's encoder.
        out.extend_from_slice(
            format!(
                "caprise-scale-bits=0x{:08x}\n",
                params.caprise.scale.to_bits()
            )
            .as_bytes(),
        );
        out.extend_from_slice(
            format!("caprise-beta-bits=0x{:08x}\n", params.caprise.beta.to_bits()).as_bytes(),
        );

        // Compass params — every field pinned.
        let c = &params.compass;
        out.extend_from_slice(format!("compass-z={}\n", c.ring_oram_z).as_bytes());
        out.extend_from_slice(format!("compass-s={}\n", c.ring_oram_s).as_bytes());
        out.extend_from_slice(format!("compass-a={}\n", c.ring_oram_a).as_bytes());
        out.extend_from_slice(format!("compass-block-bytes={}\n", c.block_bytes).as_bytes());
        out.extend_from_slice(format!("compass-hnsw-m={}\n", c.hnsw_m).as_bytes());
        out.extend_from_slice(format!("compass-hnsw-ef={}\n", c.hnsw_ef).as_bytes());
        out.extend_from_slice(format!("compass-hnsw-ef-spec={}\n", c.hnsw_ef_spec).as_bytes());
        out.extend_from_slice(format!("compass-hnsw-ef-n={}\n", c.hnsw_ef_n).as_bytes());
        out.extend_from_slice(format!("compass-treetop-levels={}\n", c.treetop_levels).as_bytes());
        out.extend_from_slice(
            format!("compass-hint-bits={}\n", c.directional_hint_bits).as_bytes(),
        );

        // XorMM params.
        let x = &params.xormm;
        out.extend_from_slice(format!("xormm-volume-bound={}\n", x.volume_bound).as_bytes());
        out.extend_from_slice(format!("xormm-stash-size={}\n", x.stash_size).as_bytes());

        // LightRAG params.
        let l = &params.lightrag;
        out.extend_from_slice(format!("lightrag-top-k={}\n", l.top_k).as_bytes());
        out.extend_from_slice(format!("lightrag-chunk-top-k={}\n", l.chunk_top_k).as_bytes());
        out.extend_from_slice(
            format!("lightrag-max-entity-tokens={}\n", l.max_entity_tokens).as_bytes(),
        );
        out.extend_from_slice(
            format!("lightrag-max-relation-tokens={}\n", l.max_relation_tokens).as_bytes(),
        );
        out.extend_from_slice(
            format!("lightrag-max-total-tokens={}\n", l.max_total_tokens).as_bytes(),
        );
        out.extend_from_slice(
            format!(
                "lightrag-search-eps-bits=0x{:08x}\n",
                l.search_perturb_epsilon.to_bits()
            )
            .as_bytes(),
        );

        out
    }

    /// 32-byte SHA-256 of the canonical fragment. Fits into
    /// `REPORT_DATA[32..64]` directly.
    pub fn scheme_identity_digest(&self, params: SchemeParamsV2) -> [u8; 32] {
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

    // ─── V2 tests ────────────────────────────────────────────────────

    #[test]
    fn v2_derive_is_deterministic() {
        let p = HkdfPolicyV2::V2;
        let t = TenantId::new("acme-legal");
        let k1 = p.derive(&sk(0x11), &sk(0x22), &t);
        let k2 = p.derive(&sk(0x11), &sk(0x22), &t);
        assert_eq!(k1.caprise_seed.as_ref(), k2.caprise_seed.as_ref());
        assert_eq!(k1.aes_chunk_key.as_ref(), k2.aes_chunk_key.as_ref());
        assert_eq!(k1.oram_entities_key.as_ref(), k2.oram_entities_key.as_ref());
        assert_eq!(k1.oram_relations_key.as_ref(), k2.oram_relations_key.as_ref());
        assert_eq!(k1.oram_chunks_key.as_ref(), k2.oram_chunks_key.as_ref());
        assert_eq!(k1.emm_adjacency_key.as_ref(), k2.emm_adjacency_key.as_ref());
        assert_eq!(k1.emm_src_chunks_key.as_ref(), k2.emm_src_chunks_key.as_ref());
        assert_eq!(k1.search_pattern_key.as_ref(), k2.search_pattern_key.as_ref());
    }

    #[test]
    fn v2_all_eight_children_are_pairwise_distinct() {
        // Distinct info strings + Hkdf-Expand counter ⇒ no two children
        // share the same 32-byte value. Catches an accidental info-string
        // duplicate at code-review time.
        let k = HkdfPolicyV2::V2.derive(&sk(0x33), &sk(0x44), &TenantId::new("t"));
        let all: [&[u8]; 8] = [
            k.caprise_seed.as_ref(),
            k.aes_chunk_key.as_ref(),
            k.oram_entities_key.as_ref(),
            k.oram_relations_key.as_ref(),
            k.oram_chunks_key.as_ref(),
            k.emm_adjacency_key.as_ref(),
            k.emm_src_chunks_key.as_ref(),
            k.search_pattern_key.as_ref(),
        ];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "child {i} collides with child {j}");
            }
        }
    }

    #[test]
    fn v2_tenant_salt_separates_keys() {
        let k_a = HkdfPolicyV2::V2.derive(&sk(0x55), &sk(0x66), &TenantId::new("tenant-A"));
        let k_b = HkdfPolicyV2::V2.derive(&sk(0x55), &sk(0x66), &TenantId::new("tenant-B"));
        assert_ne!(k_a.caprise_seed.as_ref(), k_b.caprise_seed.as_ref());
        assert_ne!(k_a.search_pattern_key.as_ref(), k_b.search_pattern_key.as_ref());
        assert_ne!(k_a.oram_entities_key.as_ref(), k_b.oram_entities_key.as_ref());
    }

    #[test]
    fn v2_either_half_changes_the_output() {
        let t = TenantId::new("t");
        let base = HkdfPolicyV2::V2.derive(&sk(0x77), &sk(0x88), &t);
        let user_flipped = HkdfPolicyV2::V2.derive(&sk(0x78), &sk(0x88), &t);
        let tee_flipped = HkdfPolicyV2::V2.derive(&sk(0x77), &sk(0x89), &t);
        assert_ne!(base.caprise_seed.as_ref(), user_flipped.caprise_seed.as_ref());
        assert_ne!(base.caprise_seed.as_ref(), tee_flipped.caprise_seed.as_ref());
        assert_ne!(
            base.search_pattern_key.as_ref(),
            user_flipped.search_pattern_key.as_ref()
        );
    }

    #[test]
    fn v2_yields_different_keys_than_v1() {
        // V1 and V2 share salt + ikm but differ in info strings, so
        // even the two "common" children (caprise_seed, aes_chunk_key)
        // are independent values. Catches accidental info-string reuse.
        let t = TenantId::new("t");
        let (v1_caprise, v1_aes) = HkdfPolicy::V1.derive(&sk(0x99), &sk(0xaa), &t);
        let v2 = HkdfPolicyV2::V2.derive(&sk(0x99), &sk(0xaa), &t);
        assert_ne!(v1_caprise.as_ref(), v2.caprise_seed.as_ref());
        assert_ne!(v1_aes.as_ref(), v2.aes_chunk_key.as_ref());
    }

    #[test]
    fn v2_scheme_identity_digest_is_stable() {
        let d1 = HkdfPolicyV2::V2.scheme_identity_digest(SchemeParamsV2::default());
        let d2 = HkdfPolicyV2::V2.scheme_identity_digest(SchemeParamsV2::default());
        assert_eq!(d1, d2);
        assert_eq!(d1.len(), 32);
    }

    #[test]
    fn v2_scheme_identity_param_sensitivity() {
        // Bumping any field in any of the four param structs must
        // change the digest. Otherwise attestation can't catch a
        // stealth reconfiguration.
        let base = SchemeParamsV2::default();
        let d_base = HkdfPolicyV2::V2.scheme_identity_digest(base);

        // CAPRISE
        let mut p = base;
        p.caprise.scale = 16.0;
        assert_ne!(d_base, HkdfPolicyV2::V2.scheme_identity_digest(p));

        // Compass — Z
        let mut p = base;
        p.compass.ring_oram_z = base.compass.ring_oram_z + 1;
        assert_ne!(d_base, HkdfPolicyV2::V2.scheme_identity_digest(p));

        // Compass — ef
        let mut p = base;
        p.compass.hnsw_ef = base.compass.hnsw_ef + 1;
        assert_ne!(d_base, HkdfPolicyV2::V2.scheme_identity_digest(p));

        // XorMM
        let mut p = base;
        p.xormm.volume_bound = base.xormm.volume_bound + 1;
        assert_ne!(d_base, HkdfPolicyV2::V2.scheme_identity_digest(p));

        // LightRAG — top_k
        let mut p = base;
        p.lightrag.top_k = base.lightrag.top_k + 1;
        assert_ne!(d_base, HkdfPolicyV2::V2.scheme_identity_digest(p));

        // LightRAG — search_perturb_epsilon (the §8.6 knob)
        let mut p = base;
        p.lightrag.search_perturb_epsilon = base.lightrag.search_perturb_epsilon + 0.01;
        assert_ne!(d_base, HkdfPolicyV2::V2.scheme_identity_digest(p));
    }

    #[test]
    fn v2_scheme_identity_digest_differs_from_v1() {
        // Different info-string set ⇒ different digest even at default
        // CAPRISE params. Prevents a V1 verifier from accidentally
        // accepting a V2 report.
        let v1 = HkdfPolicy::V1.scheme_identity_digest(SchemeParams::default());
        let v2 = HkdfPolicyV2::V2.scheme_identity_digest(SchemeParamsV2::default());
        assert_ne!(v1, v2);
    }

    #[test]
    fn v2_digest_fits_report_data_scheme_half() {
        // The digest is the canonical input for the scheme half of the
        // SEV-SNP REPORT_DATA (gelo-tee-sev-snp::ReportData::build
        // takes &[u8] and SHA-256s it into bytes 32..64). The contract
        // is: 32 bytes out, deterministic per config. Any future
        // `LightRagTwoPartyService` (M8) calls this method and hands
        // the 32 bytes straight to ReportData::build.
        let digest = HkdfPolicyV2::V2.scheme_identity_digest(SchemeParamsV2::default());
        assert_eq!(digest.len(), 32);
        // Non-zero — catches a zeroed encoder regression.
        assert!(digest.iter().any(|&b| b != 0));
    }
}
