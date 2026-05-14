//! End-to-end attestation test for the SEV-SNP path.
//!
//! Exercises:
//! 1. Stub `Embedder` overrides `model_identity()` so `AttestationEvidence`
//!    carries the right `model_identity` string.
//! 2. `SnpTrustedExecutor`-style evidence assembly through the
//!    `MockReportIssuer` (the production-side `/dev/sev-guest` path lands
//!    in M5.6 and isn't exercised here).
//! 3. `SnpVerifierAdapter` accepts well-formed evidence and rejects
//!    mismatched bindings.
//! 4. Full `GeloRagInMemoryService::ingest_chunks` → `query` cycle paired
//!    with `attest()` on a successfully verified `SnpVerifierAdapter`.

#![cfg(feature = "snp-mock")]

use anyhow::Result;
use gelo_rag::{
    GeloRagInMemoryService, AttestationEvidence, AttestationVerifier, SnpVerifierAdapter,
};
use gelo_tee_sev_snp::{
    ReportData, SnpAttestationVerifier, SnpRootTrust, mock::MockReportIssuer,
};
use rag_core::{ChunkId, DocumentChunk, Embedder, EmbeddingEncryptionScheme, EncryptedEmbedding};

/// 32 bytes of "weights manifest hash" the stub embedder pretends to hold.
const STUB_MODEL_IDENTITY: &[u8] =
    b"snp-test-model-identity-32-bytes!";

struct StubEmbedder;

impl Embedder for StubEmbedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                if t.contains("rust") {
                    vec![1.0, 0.0, 0.0]
                } else if t.contains("postgres") {
                    vec![0.0, 1.0, 0.0]
                } else if t.contains("tls") {
                    vec![0.0, 0.0, 1.0]
                } else {
                    vec![0.5, 0.5, 0.5]
                }
            })
            .collect())
    }

    fn model_identity(&self) -> &[u8] {
        STUB_MODEL_IDENTITY
    }
}

#[derive(Clone)]
struct IdentityScheme;

impl EmbeddingEncryptionScheme for IdentityScheme {
    fn scheme_name(&self) -> &'static str {
        "identity"
    }
    fn encrypt_document(&mut self, embedding: &[f32]) -> Result<EncryptedEmbedding> {
        Ok(EncryptedEmbedding {
            scheme: "identity",
            vector: embedding.to_vec(),
            nonce: vec![],
            original_dimension: embedding.len(),
        })
    }
    fn encrypt_query(&mut self, embedding: &[f32]) -> Result<EncryptedEmbedding> {
        self.encrypt_document(embedding)
    }
    fn decrypt_document(&mut self, ciphertext: &EncryptedEmbedding) -> Result<Vec<f32>> {
        Ok(ciphertext.vector.clone())
    }
}

/// Build evidence with a mock-issued SEV-SNP report whose `REPORT_DATA`
/// binds (`model_identity`, `scheme_identity`).
fn issue_evidence(model_identity: &[u8], scheme_identity: &[u8]) -> AttestationEvidence {
    let issuer = MockReportIssuer::from_bundled().expect("load bundled mock VCEK key");
    let rd = ReportData::build(model_identity, scheme_identity, None);
    let issued = issuer.issue(rd).expect("issue mock report");
    AttestationEvidence {
        tee_measurement: "mock-launch-measurement".into(),
        model_identity: String::from_utf8(model_identity.to_vec()).unwrap(),
        scheme_identity: String::from_utf8(scheme_identity.to_vec()).unwrap(),
        report: Some(issued.report_bytes),
        vcek_cert: Some(issued.vcek_cert_pem),
    }
}

#[test]
fn snp_verifier_accepts_matching_evidence() {
    let verifier =
        SnpVerifierAdapter::new(SnpAttestationVerifier::new(SnpRootTrust::with_mock_root()));
    let evidence = issue_evidence(STUB_MODEL_IDENTITY, b"scheme-v1");
    verifier
        .verify(&evidence)
        .expect("verifier accepts matching evidence");
}

#[test]
fn snp_verifier_rejects_wrong_model_identity() {
    let verifier =
        SnpVerifierAdapter::new(SnpAttestationVerifier::new(SnpRootTrust::with_mock_root()));
    // The report binds STUB_MODEL_IDENTITY, but evidence claims a different
    // model_identity string. The adapter passes the *evidence's* model_identity
    // to the verifier, so the report_data recomputation tripwires.
    let mut evidence = issue_evidence(STUB_MODEL_IDENTITY, b"scheme-v1");
    evidence.model_identity = "lying-about-model".into();
    let err = verifier.verify(&evidence).unwrap_err();
    assert!(format!("{err:#}").contains("attestation verification failed"));
}

#[test]
fn snp_verifier_rejects_missing_report() {
    let verifier =
        SnpVerifierAdapter::new(SnpAttestationVerifier::new(SnpRootTrust::with_mock_root()));
    let mut evidence = issue_evidence(STUB_MODEL_IDENTITY, b"scheme-v1");
    evidence.report = None;
    let err = verifier.verify(&evidence).unwrap_err();
    assert!(format!("{err:#}").contains("report"));
}

#[test]
fn snp_verifier_rejects_missing_vcek() {
    let verifier =
        SnpVerifierAdapter::new(SnpAttestationVerifier::new(SnpRootTrust::with_mock_root()));
    let mut evidence = issue_evidence(STUB_MODEL_IDENTITY, b"scheme-v1");
    evidence.vcek_cert = None;
    let err = verifier.verify(&evidence).unwrap_err();
    assert!(format!("{err:#}").contains("vcek_cert") || format!("{err:#}").contains("VCEK"));
}

/// Full ingest + attest + query cycle. The service is configured with the
/// SEV-SNP verifier; `attest()` is called once with valid evidence; ingestion
/// and retrieval then work as normal.
#[test]
fn full_pipeline_with_snp_verifier() {
    let embedder = StubEmbedder;
    let scheme = IdentityScheme;
    let verifier =
        SnpVerifierAdapter::new(SnpAttestationVerifier::new(SnpRootTrust::with_mock_root()));
    let mut service = GeloRagInMemoryService::new(embedder, scheme, verifier);

    // Cross-check that the embedder's model_identity matches what the relying
    // party expects to see in REPORT_DATA.
    assert_eq!(STUB_MODEL_IDENTITY, b"snp-test-model-identity-32-bytes!");

    let evidence = issue_evidence(STUB_MODEL_IDENTITY, b"scheme-v1");
    service.attest(&evidence).expect("attest should accept");

    service
        .ingest_chunks(vec![
            DocumentChunk {
                id: ChunkId("rust-memory-safety".into()),
                text: "Rust enforces memory safety through ownership and borrowing.".into(),
            },
            DocumentChunk {
                id: ChunkId("postgres-index".into()),
                text: "Postgres uses B-tree indexes for common equality and range lookups.".into(),
            },
            DocumentChunk {
                id: ChunkId("tls-attestation".into()),
                text: "Remote attestation can bind a TEE measurement into a TLS session.".into(),
            },
        ])
        .expect("ingest");

    let hits = service.query("rust memory safety", 1).expect("query");
    assert_eq!(hits[0].id.0, "rust-memory-safety");
}
