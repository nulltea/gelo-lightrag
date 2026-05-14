//! M6 — End-to-end integration test for `GeloRagTwoPartyService`.
//!
//! Covers the spec's three load-bearing claims from
//! `docs/prototype/caprise-two-party-kdf.md`:
//!
//! 1. **Happy path** — ingest under tenant A with a `user_x_sk`, query
//!    later with the same `user_x_sk`, get the expected chunk back. The
//!    `(caprise_seed, aes_key)` re-derive byte-for-byte across the two
//!    sessions.
//! 2. **Forward security against TEE-only compromise** — simulate CVM
//!    restart by dropping the service. Re-create with the same
//!    `user_x_sk` but a fresh `tee_user_x_sk`. The old index (if it
//!    were preserved by a Variant-B persistent storage layer) would be
//!    encrypted under a key the new CVM cannot reproduce. Variant A
//!    drops everything; the test asserts the loud `UnknownTenant` error
//!    fires before the client could silently re-encrypt over the gap.
//! 3. **Cross-tenant key separation** — same `user_x_sk` under two
//!    distinct `tenant_id`s produces non-equal CAPRISE ciphertexts for
//!    the same plaintext embedding. The HKDF salt is doing its job.
//!
//! The test pins the SEV-SNP layer to its no-op verifier
//! (`NoopAttestationVerifier`); the attested-mock-issuer flow is the
//! subject of `snp_attest_e2e.rs`, not this file. This file is about
//! the **KDF + service shape**, not the report-bytes plumbing.

use anyhow::Result;
use gelo_rag::{GeloRagTwoPartyService, NoopAttestationVerifier, TwoPartyError};
use rag_core::{
    ChunkId, DocumentChunk, Embedder, EncryptedEmbedding, HkdfPolicy, SchemeParams, TenantId,
};
use zeroize::Zeroizing;

/// Trivial deterministic embedder — same text ⇒ same vector. Three
/// "topics" with orthogonal vectors so cosine search is unambiguous.
struct StubEmbedder;

impl Embedder for StubEmbedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                if t.contains("apple") {
                    vec![1.0, 0.0, 0.0]
                } else if t.contains("banana") {
                    vec![0.0, 1.0, 0.0]
                } else if t.contains("cherry") {
                    vec![0.0, 0.0, 1.0]
                } else {
                    vec![0.33, 0.33, 0.33]
                }
            })
            .collect())
    }
}

fn sk(byte: u8) -> Zeroizing<[u8; 32]> {
    Zeroizing::new([byte; 32])
}

fn fresh_service() -> GeloRagTwoPartyService<StubEmbedder, NoopAttestationVerifier> {
    GeloRagTwoPartyService::new(StubEmbedder, NoopAttestationVerifier)
}

fn doc(id: &str, text: &str) -> DocumentChunk {
    DocumentChunk {
        id: ChunkId(id.into()),
        text: text.into(),
    }
}

// ─────────────────────────────────────────────────────────────────────
// 1. Happy path: ingest then query recovers the chunk.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn happy_path_ingest_then_query() {
    let mut service = fresh_service();
    let tenant = TenantId::new("acme-legal");
    let user_sk = sk(0x11);

    service
        .ingest_chunks_for(
            &tenant,
            sk(0x11), // ingest with one copy of the secret
            vec![
                doc("apple-doc", "apple orchard notes"),
                doc("banana-doc", "banana bread recipe"),
                doc("cherry-doc", "cherry harvest report"),
            ],
        )
        .unwrap();

    let hits = service
        .query_for(&tenant, user_sk, "apple pie filling", 1)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.0, "apple-doc");
    assert_eq!(hits[0].text, "apple orchard notes");
}

#[test]
fn query_re_derives_same_keys_across_sessions() {
    // Two ingests under the same tenant + same user_x_sk produce
    // ciphertexts encrypted under the same caprise_seed_key — the
    // service is stateless w.r.t. derived keys; only `tee_user_x_sk`
    // persists across requests, and the HKDF re-runs each call.
    let mut service = fresh_service();
    let tenant = TenantId::new("acme-legal");

    service
        .ingest_chunks_for(&tenant, sk(0x22), vec![doc("apple-1", "apple")])
        .unwrap();
    // Simulate a second session: a totally separate user_x_sk move
    // (the client re-sends the same 32 bytes; the CVM derives anew).
    service
        .ingest_chunks_for(&tenant, sk(0x22), vec![doc("apple-2", "apple")])
        .unwrap();

    let hits = service.query_for(&tenant, sk(0x22), "apple", 2).unwrap();
    let ids: std::collections::HashSet<_> = hits.iter().map(|h| h.id.0.as_str()).collect();
    assert!(ids.contains("apple-1"));
    assert!(ids.contains("apple-2"));
}

// ─────────────────────────────────────────────────────────────────────
// 2. Forward security: TEE restart loses `tee_user_x_sk`; the loud
//    failure contract (UnknownTenant → 410 Gone in the runner) fires.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn cvm_restart_drops_tenant_loud_failure() {
    let tenant = TenantId::new("acme-legal");
    let user_sk_v = || sk(0x33); // reproducible for the test, simulating
                                  // the same client coming back

    // Session 1 — ingest under the original CVM.
    let mut session1 = fresh_service();
    session1
        .ingest_chunks_for(&tenant, user_sk_v(), vec![doc("apple-doc", "apple")])
        .unwrap();
    assert_eq!(session1.index_len_for(&tenant), 1);

    // Simulate CVM restart: drop the entire service, including the
    // in-memory `tee_user_x_sk` table.
    drop(session1);

    // Session 2 — fresh CVM, never saw this tenant before.
    let mut session2 = fresh_service();
    let err = session2
        .query_for(&tenant, user_sk_v(), "apple", 1)
        .unwrap_err();
    assert!(
        matches!(err, TwoPartyError::UnknownTenant(ref t) if *t == tenant),
        "expected UnknownTenant, got {err:?}"
    );
}

#[test]
fn fresh_tee_secret_under_same_user_sk_yields_unrecoverable_index() {
    // This is the forward-security claim in the design doc's bold row:
    // even *with* the right `user_x_sk`, a fresh `tee_user_x_sk` makes
    // the prior session's caprise_seed_key unrecoverable. We simulate it
    // by ingesting twice under the same tenant id but in two different
    // service instances (the second's `tee_user_x_sk` is independent of
    // the first's).
    let tenant = TenantId::new("acme-legal");
    let user_sk_v = || sk(0x44);

    // Ingest in session 1, record the ciphertext.
    let mut session1 = fresh_service();
    session1
        .ingest_chunks_for(
            &tenant,
            user_sk_v(),
            vec![doc("apple-doc", "apple orchard")],
        )
        .unwrap();
    let session1_hit = session1
        .query_for(&tenant, user_sk_v(), "apple", 1)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let session1_ciphertext: EncryptedEmbedding = session1_hit.embedding.clone();
    drop(session1);

    // Ingest the *same plaintext under the same user_x_sk + tenant* in
    // session 2 — but session 2 has a fresh `tee_user_x_sk`.
    let mut session2 = fresh_service();
    session2
        .ingest_chunks_for(
            &tenant,
            user_sk_v(),
            vec![doc("apple-doc", "apple orchard")],
        )
        .unwrap();
    let session2_hit = session2
        .query_for(&tenant, user_sk_v(), "apple", 1)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let session2_ciphertext: EncryptedEmbedding = session2_hit.embedding.clone();

    // The two ciphertexts encrypt the same plaintext under different
    // caprise_seed_keys ⇒ different vectors. If they were equal, the
    // forward-security property would be broken.
    assert_ne!(
        session1_ciphertext.vector, session2_ciphertext.vector,
        "fresh tee_user_x_sk must produce different CAPRISE ciphertexts \
         for the same plaintext"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 3. Cross-tenant key separation: same user_x_sk, different tenants ⇒
//    different ciphertexts.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn cross_tenant_isolation() {
    let mut service = fresh_service();
    let user_sk_v = || sk(0x55);

    service
        .ingest_chunks_for(
            &TenantId::new("tenant-A"),
            user_sk_v(),
            vec![doc("apple-doc-A", "apple orchard")],
        )
        .unwrap();
    service
        .ingest_chunks_for(
            &TenantId::new("tenant-B"),
            user_sk_v(),
            vec![doc("apple-doc-B", "apple orchard")],
        )
        .unwrap();

    let hits_a = service
        .query_for(&TenantId::new("tenant-A"), user_sk_v(), "apple", 1)
        .unwrap();
    let hits_b = service
        .query_for(&TenantId::new("tenant-B"), user_sk_v(), "apple", 1)
        .unwrap();

    assert_eq!(hits_a[0].id.0, "apple-doc-A");
    assert_eq!(hits_b[0].id.0, "apple-doc-B");

    // The encrypted embedding vectors for the *same plaintext under the
    // same user_x_sk* must differ because the HKDF salt = tenant_id
    // splits the derived seeds. That's the §3.3 "Salt = tenant_id"
    // claim made concrete.
    assert_ne!(
        hits_a[0].embedding.vector, hits_b[0].embedding.vector,
        "cross-tenant ciphertexts under same user_x_sk must differ"
    );
}

#[test]
fn unknown_tenant_is_observable() {
    let service = fresh_service();
    assert!(!service.tenant_known(&TenantId::new("never-seen")));
    assert_eq!(service.index_len_for(&TenantId::new("never-seen")), 0);
}

#[test]
fn forget_tenant_zeroes_state() {
    let mut service = fresh_service();
    let t = TenantId::new("acme-legal");
    service
        .ingest_chunks_for(&t, sk(0x66), vec![doc("a", "apple")])
        .unwrap();
    assert!(service.tenant_known(&t));
    service.forget_tenant(&t);
    assert!(!service.tenant_known(&t));
    let err = service.query_for(&t, sk(0x66), "apple", 1).unwrap_err();
    assert!(matches!(err, TwoPartyError::UnknownTenant(_)));
}

// ─────────────────────────────────────────────────────────────────────
// 4. scheme_identity binding — the canonical digest is exposed via
//    `service.scheme_identity()` and is stable across instances with
//    matching configuration.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn scheme_identity_is_stable_and_param_sensitive() {
    let s_default = fresh_service();
    let s_default2 = fresh_service();
    assert_eq!(
        s_default.scheme_identity(),
        s_default2.scheme_identity(),
        "two services with default params must agree byte-for-byte"
    );

    let s_alt = GeloRagTwoPartyService::with_params(
        StubEmbedder,
        NoopAttestationVerifier,
        SchemeParams {
            scale: 16.0,
            beta: 0.15,
        },
        HkdfPolicy::V1,
    );
    assert_ne!(
        s_default.scheme_identity(),
        s_alt.scheme_identity(),
        "different CAPRISE params must produce a different scheme_identity"
    );
}

#[test]
fn rotate_is_stubbed_but_distinguishes_unknown_tenant() {
    let mut service = fresh_service();
    let t = TenantId::new("acme-legal");

    // Unknown tenant: should be UnknownTenant, not "not implemented".
    let err = service.rotate_tenant(&t, sk(0xAA), sk(0xBB)).unwrap_err();
    assert!(matches!(err, TwoPartyError::UnknownTenant(_)));

    // After ingest, rotation returns the explicit not-implemented error
    // so the runner can map it to 501 instead of confusing it with a
    // 500 internal-error.
    service
        .ingest_chunks_for(&t, sk(0x77), vec![doc("a", "apple")])
        .unwrap();
    let err = service.rotate_tenant(&t, sk(0xAA), sk(0xBB)).unwrap_err();
    assert!(matches!(err, TwoPartyError::Inner(_)));
}
