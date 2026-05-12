//! DP-Forward (Yue et al., CCS 2023, [arXiv 2309.06746](https://arxiv.org/abs/2309.06746))
//! analytic Matrix Gaussian Mechanism (aMGM) primitives.
//!
//! Scope intentionally narrow: this crate exposes only the math from the
//! DP-Forward paper. It does **not** depend on [`rag-core`] and does **not**
//! wrap an [`Embedder`]. The planar-Laplace mechanism that RemoteRAG uses
//! lives in [`remote-rag`] because it is a different paper.
//!
//! Two callers consume this crate:
//! - [`gelo-embedder`] (behind its `dp-forward` Cargo feature) applies the
//!   mechanism to the pooled embedding before returning it, and folds
//!   [`DpForwardConfig::config_digest`] into `Embedder::model_identity` so
//!   SEV-SNP attestation binds to the DP parameters.
//! - [`remote-rag`] optionally applies the mechanism to document embeddings
//!   at ingestion (Recipe-B noise on the at-rest side, complementing the
//!   planar-Laplace noise on the query side).

pub mod amgm;
pub mod config;

pub use config::DpForwardConfig;
