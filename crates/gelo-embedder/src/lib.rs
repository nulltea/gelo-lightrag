//! BERT-class sentence embedder driven through a [`gelo_protocol::TrustedExecutor`].
//!
//! The encoder forward pass routes every Q/K/V/O and FFN GEMM through the
//! executor so the protocol's per-batch token-axis mask is applied
//! transparently. Non-linear ops, residual adds, embedding lookup, mean
//! pooling and L2 normalization all run inside the trusted side.

pub mod attention;
pub mod config;
pub mod embedder;
pub mod forward;
pub mod pooling;
pub mod tokenizer;
pub mod weights;

pub use config::BertConfig;
pub use embedder::GeloBertEmbedder;
pub use tokenizer::BertTokenizer;
pub use weights::{BertLayerWeights, BertWeights};
