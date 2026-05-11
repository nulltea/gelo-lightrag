use serde::Deserialize;

/// Minimal BERT configuration sufficient to drive a sentence-transformers
/// embedding forward pass. Mirrors the fields present in HuggingFace
/// `config.json` for bge-small / MiniLM / BGE-base.
#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    /// First `n` layers that should run entirely inside the trusted side
    /// (no offload). GELO §3.2 recommends excluding the first few + last
    /// layer from offload as a defense against known-plaintext attacks.
    #[serde(default = "default_skip_first")]
    pub skip_first_layers: usize,
    /// If true, the final encoder layer also runs entirely inside the
    /// trusted side.
    #[serde(default = "default_skip_last")]
    pub skip_last_layer: bool,
}

const fn default_max_position_embeddings() -> usize {
    512
}
const fn default_type_vocab_size() -> usize {
    2
}
const fn default_layer_norm_eps() -> f32 {
    1e-12
}
fn default_hidden_act() -> String {
    "gelu".into()
}
const fn default_max_seq_len() -> usize {
    512
}
const fn default_skip_first() -> usize {
    0
}
const fn default_skip_last() -> bool {
    false
}

impl BertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether a given layer index should be offloaded under the protocol.
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
