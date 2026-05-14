//! `GeloRagTwoPartyService` — multi-tenant CAPRISE-in-TEE service using
//! the two-party-KDF construction (see
//! `docs/prototype/caprise-two-party-kdf.md`).
//!
//! Holds **no per-tenant scheme instance** at idle — every request
//! derives `(caprise_seed_key, aes_chunk_key)` from
//!
//! ```text
//! (user_x_sk, tee_user_x_sk, tenant_id) ─HKDF─→ (caprise_seed, aes_key)
//! ```
//!
//! and zeroizes them at request end. A future TEE-only seal break
//! recovers `tee_user_x_sk` but not `user_x_sk` (the latter is never
//! persisted; the RATLS that carried it is forward-secret), so past
//! sessions remain undecryptable — the design's headline property.
//!
//! Today the service uses **Variant A** persistence for
//! `tee_user_x_sk`: in-memory only, lost on CVM restart. The
//! request handlers map a missing entry to [`TwoPartyError::UnknownTenant`]
//! so the runner can return HTTP 410 Gone instead of silently
//! re-encrypting under a fresh secret. Variants B (KMS-released) and C
//! (RP-escrowed) are deferred to milestones M7/M8.

use std::collections::HashMap;

use anyhow::Result;
use rag_core::{
    AesChunkCipher, Caprise, CapriseKey, DocumentChunk, Embedder, EmbeddingEncryptionScheme,
    HkdfPolicy, InMemoryEncryptedIndex, RetrievalHit, SchemeParams, TenantId,
};
use rand::RngCore;
use zeroize::Zeroizing;

use crate::attestation::{AttestationEvidence, AttestationVerifier};

/// Errors a relying-party service maps to non-200 HTTP responses.
#[derive(thiserror::Error, Debug)]
pub enum TwoPartyError {
    /// No `tee_user_x_sk` row for this tenant. Either the tenant was
    /// never bootstrapped or the CVM restarted under Variant A
    /// persistence. Map to HTTP 410 Gone — see the design doc §12.
    #[error("tenant {0} unknown — re-bootstrap the tenant (CVM may have restarted)")]
    UnknownTenant(TenantId),
    /// Anything else (embedding failure, AES auth tag mismatch, …).
    #[error(transparent)]
    Inner(#[from] anyhow::Error),
}

/// Multi-tenant service shape (spec §6). `embedder` is owned per-process;
/// every other piece of state is per-tenant.
pub struct GeloRagTwoPartyService<E, V> {
    embedder: E,
    verifier: V,
    /// Variant-A persistence for `tee_user_x_sk`: in-memory only.
    /// Lost on process restart. Each value zeroes itself on drop.
    tee_secrets: HashMap<TenantId, Zeroizing<[u8; 32]>>,
    indices: HashMap<TenantId, InMemoryEncryptedIndex>,
    scheme_params: SchemeParams,
    hkdf_policy: HkdfPolicy,
}

impl<E, V> GeloRagTwoPartyService<E, V>
where
    E: Embedder,
    V: AttestationVerifier,
{
    pub fn new(embedder: E, verifier: V) -> Self {
        Self::with_params(embedder, verifier, SchemeParams::default(), HkdfPolicy::V1)
    }

    pub fn with_params(
        embedder: E,
        verifier: V,
        scheme_params: SchemeParams,
        hkdf_policy: HkdfPolicy,
    ) -> Self {
        Self {
            embedder,
            verifier,
            tee_secrets: HashMap::new(),
            indices: HashMap::new(),
            scheme_params,
            hkdf_policy,
        }
    }

    /// Forward to the attached verifier — RP-side parity with the
    /// existing [`super::GeloRagInMemoryService::attest`].
    pub fn attest(&self, evidence: &AttestationEvidence) -> Result<()> {
        self.verifier.verify(evidence)
    }

    /// Canonical 32-byte digest of the KDF policy + CAPRISE constants —
    /// goes into `REPORT_DATA[32..64]` of the SEV-SNP attestation
    /// report (§5 / spec §8). The runner is responsible for composing
    /// this with any external mask / shield digests.
    pub fn scheme_identity(&self) -> [u8; 32] {
        self.hkdf_policy.scheme_identity_digest(self.scheme_params)
    }

    pub fn scheme_params(&self) -> SchemeParams {
        self.scheme_params
    }

    pub fn hkdf_policy(&self) -> HkdfPolicy {
        self.hkdf_policy
    }

    pub fn tenant_known(&self, tenant_id: &TenantId) -> bool {
        self.tee_secrets.contains_key(tenant_id)
    }

    pub fn known_tenants(&self) -> impl Iterator<Item = &TenantId> {
        self.tee_secrets.keys()
    }

    pub fn index_len_for(&self, tenant_id: &TenantId) -> usize {
        self.indices
            .get(tenant_id)
            .map(|i| i.len())
            .unwrap_or(0)
    }

    /// Wipe all state for `tenant_id`. The dropped `Zeroizing<[u8;32]>`
    /// zeroes the underlying bytes; ciphertexts in the index are
    /// dropped (the storage server still holds its copy if it was
    /// pushed there — out of scope).
    pub fn forget_tenant(&mut self, tenant_id: &TenantId) {
        self.tee_secrets.remove(tenant_id);
        self.indices.remove(tenant_id);
    }

    /// Get-or-create the per-tenant `tee_user_x_sk`. First contact
    /// generates 32 bytes from `OsRng` and persists in the in-memory
    /// table.
    fn or_create_tee_secret(&mut self, tenant_id: &TenantId) -> Zeroizing<[u8; 32]> {
        let entry = self.tee_secrets.entry(tenant_id.clone()).or_insert_with(|| {
            let mut buf: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
            rand::rng().fill_bytes(buf.as_mut());
            buf
        });
        clone_secret(entry)
    }

    fn require_tee_secret(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Zeroizing<[u8; 32]>, TwoPartyError> {
        self.tee_secrets
            .get(tenant_id)
            .map(clone_secret)
            .ok_or_else(|| TwoPartyError::UnknownTenant(tenant_id.clone()))
    }

    /// Ingest path (spec §7.1). `user_x_sk` is consumed and zeroized at
    /// return; the derived sub-keys are zeroized when the local
    /// `Caprise` / `AesChunkCipher` go out of scope at function end.
    pub fn ingest_chunks_for(
        &mut self,
        tenant_id: &TenantId,
        user_x_sk: Zeroizing<[u8; 32]>,
        chunks: Vec<DocumentChunk>,
    ) -> Result<(), TwoPartyError> {
        let tee_sk = self.or_create_tee_secret(tenant_id);
        let (caprise_seed, aes_key) =
            self.hkdf_policy
                .derive(&user_x_sk, &tee_sk, tenant_id);

        let mut caprise = Caprise::new(CapriseKey::from_seed(
            self.scheme_params.scale,
            self.scheme_params.beta,
            *caprise_seed,
        ));
        let chunk_cipher = AesChunkCipher::from_key(*aes_key);

        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embedder.embed(&texts).map_err(TwoPartyError::Inner)?;

        let index = self.indices.entry(tenant_id.clone()).or_default();
        for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            let encrypted = caprise.encrypt_document(&embedding)?;
            let encrypted_chunk = chunk_cipher.encrypt_chunk(&chunk)?;
            index.insert(encrypted_chunk, encrypted);
        }
        Ok(())
        // user_x_sk, tee_sk, caprise_seed, aes_key, caprise, chunk_cipher
        // all drop here and zeroize.
    }

    /// Query path (spec §7.2). Returns plaintext hits over what the
    /// caller is responsible for tunneling back as RATLS — see the
    /// "Why TEE returns plaintext hits over RATLS" note in §7.2.
    pub fn query_for(
        &mut self,
        tenant_id: &TenantId,
        user_x_sk: Zeroizing<[u8; 32]>,
        text: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalHit>, TwoPartyError> {
        let tee_sk = self.require_tee_secret(tenant_id)?;
        let (caprise_seed, aes_key) =
            self.hkdf_policy
                .derive(&user_x_sk, &tee_sk, tenant_id);

        let mut caprise = Caprise::new(CapriseKey::from_seed(
            self.scheme_params.scale,
            self.scheme_params.beta,
            *caprise_seed,
        ));
        let chunk_cipher = AesChunkCipher::from_key(*aes_key);

        let embeddings = self
            .embedder
            .embed(&[text.to_owned()])
            .map_err(TwoPartyError::Inner)?;
        let encrypted_query = caprise
            .encrypt_query(&embeddings[0])
            .map_err(TwoPartyError::Inner)?;

        // No index yet for this tenant ⇒ empty result (still a known
        // tenant, just nothing to retrieve). UnknownTenant has already
        // returned above.
        let Some(index) = self.indices.get(tenant_id) else {
            return Ok(vec![]);
        };

        let hits = index.search(&encrypted_query, top_k);
        let out: Result<Vec<RetrievalHit>, anyhow::Error> = hits
            .into_iter()
            .map(|(enc_chunk, embedding, score)| {
                let chunk = chunk_cipher.decrypt_chunk(&enc_chunk)?;
                Ok(RetrievalHit {
                    id: chunk.id,
                    score,
                    text: chunk.text,
                    embedding,
                })
            })
            .collect();
        out.map_err(TwoPartyError::Inner)
    }

    /// Rotation stub (spec §5.3 / M8). Returns
    /// `TwoPartyError::Inner(...)` with a "not implemented" error so
    /// the runner can map to HTTP 501. The full rotation flow
    /// (decrypt-under-old, re-encrypt-under-new, swap atomic) is M8.
    pub fn rotate_tenant(
        &mut self,
        tenant_id: &TenantId,
        _old_user_x_sk: Zeroizing<[u8; 32]>,
        _new_user_x_sk: Zeroizing<[u8; 32]>,
    ) -> Result<(), TwoPartyError> {
        if !self.tenant_known(tenant_id) {
            return Err(TwoPartyError::UnknownTenant(tenant_id.clone()));
        }
        Err(TwoPartyError::Inner(anyhow::anyhow!(
            "tenant rotation not implemented (deferred to milestone M8)"
        )))
    }
}

/// Clone a `Zeroizing<[u8; 32]>` while preserving zeroize-on-drop on
/// both copies. `Zeroizing` doesn't implement `Clone` on inner arrays
/// directly, so we go via a fresh allocation.
fn clone_secret(src: &Zeroizing<[u8; 32]>) -> Zeroizing<[u8; 32]> {
    let mut out: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
    out.copy_from_slice(src.as_ref());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoopAttestationVerifier;
    use rag_core::ChunkId;

    struct StubEmbedder;

    impl Embedder for StubEmbedder {
        fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    if text.contains("apple") {
                        vec![1.0, 0.0]
                    } else if text.contains("banana") {
                        vec![0.0, 1.0]
                    } else {
                        vec![0.9, 0.1]
                    }
                })
                .collect())
        }
    }

    fn sk(byte: u8) -> Zeroizing<[u8; 32]> {
        Zeroizing::new([byte; 32])
    }

    #[test]
    fn ingest_then_query_recovers_chunk() {
        let mut service = GeloRagTwoPartyService::new(StubEmbedder, NoopAttestationVerifier);
        let tenant = TenantId::new("tenant-A");
        service
            .ingest_chunks_for(
                &tenant,
                sk(0x11),
                vec![
                    DocumentChunk {
                        id: ChunkId("apple-doc".into()),
                        text: "apple orchard".into(),
                    },
                    DocumentChunk {
                        id: ChunkId("banana-doc".into()),
                        text: "banana bread".into(),
                    },
                ],
            )
            .unwrap();

        let hits = service.query_for(&tenant, sk(0x11), "apple pie", 1).unwrap();
        assert_eq!(hits[0].id.0, "apple-doc");
    }

    #[test]
    fn unknown_tenant_returns_dedicated_error() {
        let mut service = GeloRagTwoPartyService::new(StubEmbedder, NoopAttestationVerifier);
        let err = service
            .query_for(&TenantId::new("never-seen"), sk(0x22), "apple", 1)
            .unwrap_err();
        assert!(matches!(err, TwoPartyError::UnknownTenant(_)));
    }
}
