//! Decoder-LLM-class sentence embedder (Qwen3-Embedding et al).
//!
//! Architecture differences vs. the BERT path in `bert/`:
//! - causal (lower-triangular) attention
//! - RMSNorm (pre-residual) rather than LayerNorm (post-residual)
//! - rotary position embedding (RoPE) on `Q`, `K` after projection
//! - grouped-query attention (`num_attention_heads > num_key_value_heads`)
//! - SwiGLU feed-forward (`down(silu(gate) ⊙ up)`)
//! - no bias terms anywhere
//! - last-token pooling for the embedding

pub mod attention;
pub mod config;
pub mod embedder;
pub mod forward;
pub mod generation;
pub mod kv_cache;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;
pub mod weights;

pub use config::DecoderConfig;
pub use embedder::GeloQwenEmbedder;
pub use generation::{GenerationConfig, GenerationOutput, SamplerConfig, generate};
pub use kv_cache::{KvCache, LayerKvCache};
pub use weights::{DecoderLayerWeights, DecoderWeights};
