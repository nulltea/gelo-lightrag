//! [`RerankService`] trait and request/response types shared by the
//! cross-encoder and causal-discriminator service impls.
//!
//! The trait deliberately does not surface scores or rank order in its
//! return type — the only thing that leaves a [`RerankService::rerank`]
//! call is an [`EncryptedRerankBundle`] of fixed-shape ciphertexts. That
//!'s the architectural property the design relies on
//! (`docs/research/private-reranking-research-round-2.md` §4.1).

use rag_core::ChunkId;
use thiserror::Error;

use crate::output::EncryptedRerankBundle;
use crate::session::{QueryId, SessionKey};

/// One reranking candidate handed to the service. The caller is
/// responsible for AES-decrypting chunk text inside the CVM and passing
/// it as plaintext here — the service does not touch storage keys.
#[derive(Debug, Clone)]
pub struct RerankCandidate {
    pub chunk_id: ChunkId,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct RerankRequest<'a> {
    pub query: &'a str,
    pub candidates: &'a [RerankCandidate],
    /// Final ranked window. Must be ≤ `k_max`; the service will pad up
    /// to `k_max` with random decoy ciphertexts so the host can't infer
    /// `k` from the wire shape.
    pub top_k: usize,
    /// Fixed emission count. The bundle always carries exactly `k_max`
    /// items; pinning it per deployment hides the rank-window selection.
    pub k_max: usize,
    /// Per-query identifier; participates in the HKDF query-key
    /// derivation. Must be unique within a session.
    pub query_id: QueryId,
}

#[derive(Debug, Error)]
pub enum RerankError {
    #[error("model forward failed: {0}")]
    Forward(#[source] anyhow::Error),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("AEAD failure: {0}")]
    Aead(&'static str),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Common shape for both rerank service variants.
///
/// Implementations:
/// - [`crate::cross_encoder::CrossEncoderRerankService`] —
///   BERT-class cross-encoder; suits bge-reranker-v2-m3 et al.
/// - [`crate::causal_discriminator::CausalDiscriminatorRerankService`] —
///   causal LM with yes/no discriminator head; suits Qwen3-Reranker-0.6B
///   et al.
pub trait RerankService {
    /// Stable identifier for the loaded reranker — typically the
    /// SHA-256 of the safetensors manifest, possibly extended with the
    /// SHA-256 of any head weights. Bound into REPORT_DATA[0..32] of
    /// the attestation report. Same role as
    /// `rag_core::Embedder::model_identity`.
    fn model_identity(&self) -> &[u8];

    /// Reranker family tag — `"cross-encoder"` or
    /// `"causal-discriminator"`. Forms part of `scheme_identity` so a
    /// relying party can verify what kind of reranker is running, not
    /// just which weights.
    fn family(&self) -> &'static str;

    /// Score the candidates, sort, take top-k, pad to k_max, shuffle,
    /// AEAD-re-encrypt under a per-query HKDF key. Scores and rank
    /// order never leave the service.
    fn rerank(
        &mut self,
        session: &SessionKey,
        request: &RerankRequest<'_>,
    ) -> Result<EncryptedRerankBundle, RerankError>;
}
