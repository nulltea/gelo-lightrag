use serde::Deserialize;

/// Minimal config for a Qwen3-family decoder (Qwen2/Qwen3/LLaMA-3 share this
/// shape modulo field names). Read from a model's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: usize,
    /// Per-head dimension. Modern configs (Qwen3) make this independent of
    /// `hidden_size`. When absent (older configs) we fall back to
    /// `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    /// Sensitive-layer exclusion per GELO §3.2.
    #[serde(default)]
    pub skip_first_layers: usize,
    #[serde(default)]
    pub skip_last_layer: bool,
    /// Route per-head `Q · Kᵀ` through TwinShield OutAttnMult when the layer
    /// is offloaded. Disabling this leaves attention scores computed inside
    /// the TEE while Q/K/V/O + SwiGLU offload still applies — useful for
    /// benchmarks measuring the OutAttnMult delta in isolation.
    #[serde(default = "default_out_attn_mult")]
    pub use_out_attn_mult: bool,
}

const fn default_out_attn_mult() -> bool {
    true
}

const fn default_kv_heads() -> usize {
    1
}
const fn default_max_pos() -> usize {
    8192
}
const fn default_rms_eps() -> f32 {
    1e-6
}
const fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_hidden_act() -> String {
    "silu".into()
}
const fn default_max_seq_len() -> usize {
    2048
}

impl DecoderConfig {
    pub fn head_dim_value(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim_value()
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim_value()
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn offload_layer(&self, layer: usize) -> bool {
        if layer < self.skip_first_layers {
            return false;
        }
        if self.skip_last_layer && layer + 1 == self.num_hidden_layers {
            return false;
        }
        true
    }
}
