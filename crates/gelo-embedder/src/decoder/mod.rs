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
pub mod gemma4;
pub mod generation;
pub mod kv_cache;
pub mod qwen3;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;
pub mod weights;

pub use config::{AttentionClass, DecoderConfig};
pub use embedder::GeloQwenEmbedder;
pub use gemma4::{Gemma4Variant, gemma4_attention_classes};
pub use generation::{GenerationConfig, GenerationOutput, SamplerConfig, generate};
pub use kv_cache::{KvCache, LayerKvCache};
pub use qwen3::Qwen3Variant;
pub use weights::{DecoderLayerWeights, DecoderWeights};
