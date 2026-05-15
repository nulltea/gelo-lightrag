//! Private reranking on top of the GELO embedder.
//!
//! Two architecture-named services share one [`RerankService`] trait:
//! [`cross_encoder::CrossEncoderRerankService`] (BERT-class cross-encoder,
//! e.g. `BAAI/bge-reranker-v2-m3`) and
//! [`causal_discriminator::CausalDiscriminatorRerankService`] (causal-LM
//! yes/no discriminator, e.g. `Qwen/Qwen3-Reranker-0.6B`). Both run their
//! forward passes through a [`gelo_protocol::TrustedExecutor`] — the GELO
//! mask + TwinShield primitives carry over from the embedder unchanged.
//!
//! The TEE-internal flow is:
//! 1. Decrypt candidate chunk text inside the CVM (existing AES key path
//!    from `rag_core::AesChunkCipher`).
//! 2. Score each `(query, chunk)` pair under the GELO mask — scores stay
//!    inside the TEE.
//! 3. Sort, take top-k, pad to `k_max`, shuffle emission order.
//! 4. Re-encrypt each chunk under a per-query [`session::QueryKey`]
//!    derived from a per-session HKDF root, so output ciphertexts are
//!    unlinkable to the storage-time ciphertexts and the host can't
//!    correlate ordered emission back to specific chunks.
//!
//! See `docs/research/private-reranking-research-round-2.md` for the
//! design rationale.

pub mod causal_discriminator;
pub mod cross_encoder;
pub mod head;
pub mod output;
pub mod score;
pub mod service;
pub mod session;

pub use causal_discriminator::CausalDiscriminatorRerankService;
pub use cross_encoder::CrossEncoderRerankService;
pub use output::{EncryptedRerankBundle, EncryptedRerankItem};
pub use score::{RankedItem, ScoredCandidate};
pub use service::{RerankCandidate, RerankError, RerankRequest, RerankService};
pub use session::{QueryId, QueryKey, SessionKey, SessionKeyPolicy};
