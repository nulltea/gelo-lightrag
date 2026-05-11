//! BERT-class and decoder-LLM-class sentence embedders driven through a
//! [`gelo_protocol::TrustedExecutor`].
//!
//! The encoder forward pass routes every offloadable matmul through the
//! executor so the protocol's per-batch token-axis mask is applied
//! transparently. Non-linear ops, residual adds, embedding lookup, pooling
//! and L2 normalization all run inside the trusted side.

pub mod bert;
pub mod common;
pub mod decoder;

pub use bert::{BertConfig, BertLayerWeights, BertWeights, GeloBertEmbedder};
pub use common::HfTokenizer;
pub use decoder::{DecoderConfig, DecoderLayerWeights, DecoderWeights, GeloQwenEmbedder};
