use serde::Deserialize;

/// Per-layer attention class. Qwen3-style "all global" decoders leave
/// the config's `attention_classes` as `None`; Gemma 3n / Gemma 4
/// hybrid models supply a per-layer vector mixing `Local(W)` and
/// `Global`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AttentionClass {
    /// Causal sliding-window attention of window size `W` tokens. Each
    /// query position attends to at most `W` of the most recent K
    /// positions. Stays in-TEE unconditionally (the band-mask is
    /// permutation-incompatible — see `docs/prototype/gelo-llm.html`
    /// §02).
    Local { window: usize },
    /// Full causal attention over the entire cached prefix. Subject to
    /// the length-based auto-switch (`DecoderConfig::out_attn_mult_*`,
    /// `perm_attention_*`, M1.10 fused permuted) for prefill; stays
    /// in-TEE at decode shape always.
    Global,
}

impl AttentionClass {
    pub fn is_local(self) -> bool {
        matches!(self, AttentionClass::Local { .. })
    }

    pub fn is_global(self) -> bool {
        matches!(self, AttentionClass::Global)
    }

    pub fn window(self) -> Option<usize> {
        match self {
            AttentionClass::Local { window } => Some(window),
            AttentionClass::Global => None,
        }
    }
}

/// Minimal config for a Qwen3-family decoder (Qwen2/Qwen3/LLaMA-3 share this
/// shape modulo field names). Read from a model's `config.json`.
///
/// Gemma 3n / Gemma 4 add three orthogonal extensions that ride on
/// optional fields:
///
/// 1. **Hybrid attention** — populate `attention_classes` with a
///    per-layer `AttentionClass` vector. Length must equal
///    `num_hidden_layers`. When `None` (default for Qwen3 / Llama),
///    every layer behaves as `Global`.
/// 2. **p-RoPE** — set `partial_rope` to `Some(p)` to rotate only the
///    first `floor(p · head_dim)` dims of each Q / K head. Gemma 4
///    global layers use `p = 0.25`; local layers and other model
///    families use full rotation (`None`, equivalent to `Some(1.0)`).
/// 3. **K = V tying in global layers** — set `kv_shared_in_global` to
///    `true` for Gemma 4. M1.4 halves the global-layer mask sample
///    count and KV-cache memory when this is on. Default `false`
///    keeps the Qwen3 / Llama behaviour.
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
    /// Master kill-switch for routing Q·Kᵀ through TwinShield OutAttnMult.
    /// When `false`, attention always runs in the TEE (Q and K never cross
    /// PCIe — confidentiality is preserved by construction).
    ///
    /// When `true`, OutAttnMult engages **only when the sequence length
    /// reaches [`out_attn_mult_min_seq_len`]** — see that field for the
    /// auto-switch rationale. Set both to `true` / `Some(0)` to force
    /// OutAttnMult unconditionally (useful for parity tests).
    #[serde(default = "default_out_attn_mult")]
    pub use_out_attn_mult: bool,

    /// Length threshold at which OutAttnMult begins to pay for itself:
    /// `causal_gqa_attention_with_offload` engages when `n ≥ threshold`,
    /// otherwise attention runs in-TEE.
    ///
    /// `None` resolves to `hidden_size` at runtime via
    /// [`Self::out_attn_mult_threshold`]. This is the FLOP-balance
    /// crossover — attention is O(n²·d) and one linear projection is
    /// O(n·d²), so n ≈ d is where attention starts matching one
    /// projection's work. Below this point the 4-partition scheme's 4×
    /// FLOP widening and CPU-side operand stacking lose to in-TEE
    /// attention; above it, GPU throughput wins even at 4× FLOPs.
    ///
    /// Measured on Qwen3-Embedding-0.6B (d=1024) / RADV / 3 short texts
    /// (n ≈ 32): +24% wall-clock with OutAttnMult vs +6% with in-TEE
    /// attention.
    #[serde(default)]
    pub out_attn_mult_min_seq_len: Option<usize>,

    /// Master switch for Tier 1 permutation-shielded attention
    /// (Amulet-inspired). When enabled and the sequence length reaches
    /// [`perm_attention_min_seq_len`] (but not yet
    /// [`out_attn_mult_min_seq_len`]), `causal_gqa_attention_permuted`
    /// routes the full attention chain — Q·Kᵀ, softmax, ·V — through
    /// the GPU under a fresh per-batch row permutation.
    ///
    /// Cheaper than OutAttnMult at medium sequence lengths because the
    /// permutation doesn't widen the operand to 2n×2n; softmax lives on
    /// the GPU rather than on the TEE side.
    #[serde(default = "default_perm_attention")]
    pub use_perm_attention: bool,

    /// Length threshold at which the permutation-shielded attention
    /// path engages. Below this `n`, in-TEE attention is the default
    /// (cheap at short sequences). Above [`out_attn_mult_min_seq_len`]
    /// the permuted path yields to OutAttnMult (which has stronger
    /// privacy when Q, K are valuable runtime values at long context).
    ///
    /// `None` resolves to `64` at runtime via
    /// [`Self::perm_attention_threshold`]. Empirically tuned for the
    /// Qwen3 / NFCorpus shape: n ≈ 400 is well above the threshold so
    /// the permuted path engages.
    #[serde(default)]
    pub perm_attention_min_seq_len: Option<usize>,

    /// Per-layer hybrid-attention class vector. `None` keeps the
    /// existing Qwen3 / Llama behaviour (every layer global, no
    /// sliding window). When `Some(v)`, `v.len()` must equal
    /// `num_hidden_layers` — see `decoder::gemma4` for the per-variant
    /// builders.
    #[serde(default)]
    pub attention_classes: Option<Vec<AttentionClass>>,

    /// p-RoPE partial-rotation factor. `None` resolves to full rotation
    /// (the existing RoPE behaviour). `Some(p)` rotates only the first
    /// `floor(p · head_dim)` dimensions of each head; Gemma 4 global
    /// layers use `p = 0.25`. Wired by M1.5; M1.1 only stores it.
    #[serde(default)]
    pub partial_rope: Option<f32>,

    /// K = V tying in global layers (some Gemma 3 variants ship with
    /// this; real Gemma 4 E2B/E4B configs set `attention_k_eq_v: false`
    /// so it does not engage there). When `true`, the model
    /// materialises K and V as the same tensor in global layers and
    /// the protocol samples one mask / one permutation for both.
    /// Default `false` preserves Qwen3 / Llama and matches Gemma 4.
    /// Wired by M1.4; M1.1 only stores it.
    #[serde(default)]
    pub kv_shared_in_global: bool,

    /// Final-logit softcap. When `Some(c)`, the LM-head output is
    /// passed through `tanh(x / c) * c` before sampling — bounds the
    /// logit range and stabilises temperature sampling. Real Gemma 4
    /// uses `c = 30.0`. `None` (default) preserves the existing
    /// Qwen3 / Llama path byte-for-byte. Applied in
    /// `decoder::generation::compute_logits`.
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
}

const fn default_perm_attention() -> bool {
    false
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

    /// Effective OutAttnMult auto-switch threshold. Resolves
    /// `out_attn_mult_min_seq_len = None` to `hidden_size` (the FLOP
    /// balance point between attention and one linear projection).
    pub fn out_attn_mult_threshold(&self) -> usize {
        self.out_attn_mult_min_seq_len.unwrap_or(self.hidden_size)
    }

    /// True iff the dispatch layer should route Q·Kᵀ through
    /// OutAttnMult for a forward pass of sequence length `n`.
    /// Combines the master switch and the auto-switch threshold so
    /// callers don't have to remember the precedence.
    pub fn out_attn_mult_enabled_for(&self, n: usize) -> bool {
        self.use_out_attn_mult && n >= self.out_attn_mult_threshold()
    }

    /// Effective threshold at which permutation-shielded attention
    /// engages. `None` resolves to `64` (the empirical knee where the
    /// engine offload starts amortising the extra PCIe round-trips).
    pub fn perm_attention_threshold(&self) -> usize {
        self.perm_attention_min_seq_len.unwrap_or(64)
    }

    /// True iff the dispatch layer should route the full attention
    /// chain through permutation-shielded attention for sequence
    /// length `n`. The 3-way autoswitch precedence is:
    /// - OutAttnMult wins at very long `n` (its declared threshold)
    /// - permuted attention wins in the medium range
    /// - in-TEE attention is the fallback for short `n`
    pub fn perm_attention_enabled_for(&self, n: usize) -> bool {
        if !self.use_perm_attention {
            return false;
        }
        if self.out_attn_mult_enabled_for(n) {
            return false;
        }
        n >= self.perm_attention_threshold()
    }

    /// Effective attention class for layer `li`. Falls back to
    /// `AttentionClass::Global` when `attention_classes` is `None`
    /// (the existing Qwen3 / Llama back-compat behaviour) or when the
    /// vector is too short. Panics on shape mismatch only when the
    /// vector is present and shorter than `num_hidden_layers` — that
    /// is a configuration error.
    pub fn effective_attention_class(&self, li: usize) -> AttentionClass {
        match &self.attention_classes {
            None => AttentionClass::Global,
            Some(v) => {
                assert!(
                    v.len() == self.num_hidden_layers,
                    "DecoderConfig.attention_classes length {} != num_hidden_layers {}",
                    v.len(),
                    self.num_hidden_layers,
                );
                v[li]
            }
        }
    }

    /// True iff any layer in this config has `AttentionClass::Local`.
    /// Convenience for callers that want to know whether they need to
    /// wire the sliding-window kernel.
    pub fn is_hybrid_attention(&self) -> bool {
        self.attention_classes
            .as_ref()
            .map(|v| v.iter().any(|c| c.is_local()))
            .unwrap_or(false)
    }

    /// Effective p-RoPE rotation count: how many dims of each head get
    /// rotated. Defaults to the full `head_dim` when `partial_rope` is
    /// `None`.
    pub fn rotated_dim(&self) -> usize {
        let head = self.head_dim_value();
        match self.partial_rope {
            None => head,
            Some(p) => {
                assert!(
                    (0.0..=1.0).contains(&p),
                    "partial_rope {p} must be in [0, 1]",
                );
                let r = (p * head as f32).floor() as usize;
                // Snap to even — RoPE rotates pairs (low/high halves
                // of the half-range), so an odd count is ill-defined.
                r & !1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with(use_master: bool, threshold: Option<usize>, hidden: usize) -> DecoderConfig {
        DecoderConfig {
            vocab_size: 1,
            hidden_size: hidden,
            intermediate_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: Some(1),
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            max_seq_len: 16,
            skip_first_layers: 0,
            skip_last_layer: false,
            use_out_attn_mult: use_master,
            out_attn_mult_min_seq_len: threshold,
            use_perm_attention: false,
            perm_attention_min_seq_len: None,
            attention_classes: None,
            partial_rope: None,
            kv_shared_in_global: false,
            final_logit_softcapping: None,
        }
    }

    #[test]
    fn master_off_disables_at_every_length() {
        let c = cfg_with(false, Some(0), 1024);
        assert!(!c.out_attn_mult_enabled_for(0));
        assert!(!c.out_attn_mult_enabled_for(1));
        assert!(!c.out_attn_mult_enabled_for(99_999));
    }

    #[test]
    fn auto_threshold_defaults_to_hidden_size() {
        let c = cfg_with(true, None, 1024);
        assert_eq!(c.out_attn_mult_threshold(), 1024);
        assert!(!c.out_attn_mult_enabled_for(1023));
        assert!(c.out_attn_mult_enabled_for(1024));
    }

    #[test]
    fn explicit_threshold_overrides_hidden_size() {
        let c = cfg_with(true, Some(256), 1024);
        assert_eq!(c.out_attn_mult_threshold(), 256);
        assert!(!c.out_attn_mult_enabled_for(255));
        assert!(c.out_attn_mult_enabled_for(256));
    }

    #[test]
    fn threshold_zero_forces_on_at_any_length() {
        let c = cfg_with(true, Some(0), 1024);
        assert!(c.out_attn_mult_enabled_for(0));
        assert!(c.out_attn_mult_enabled_for(1));
    }

    #[test]
    fn attention_classes_none_defaults_to_global() {
        let c = cfg_with(true, None, 32);
        assert_eq!(c.effective_attention_class(0), AttentionClass::Global);
        assert!(!c.is_hybrid_attention());
    }

    #[test]
    fn attention_classes_some_returns_per_layer() {
        let mut c = cfg_with(true, None, 32);
        c.num_hidden_layers = 4;
        c.attention_classes = Some(vec![
            AttentionClass::Local { window: 128 },
            AttentionClass::Local { window: 128 },
            AttentionClass::Local { window: 128 },
            AttentionClass::Global,
        ]);
        assert_eq!(
            c.effective_attention_class(0),
            AttentionClass::Local { window: 128 },
        );
        assert_eq!(c.effective_attention_class(3), AttentionClass::Global);
        assert!(c.is_hybrid_attention());
    }

    #[test]
    #[should_panic(expected = "attention_classes length")]
    fn attention_classes_wrong_length_panics() {
        let mut c = cfg_with(true, None, 32);
        c.num_hidden_layers = 4;
        c.attention_classes = Some(vec![AttentionClass::Global; 3]);
        let _ = c.effective_attention_class(0);
    }

    #[test]
    fn rotated_dim_full_when_partial_rope_none() {
        let mut c = cfg_with(true, None, 64);
        c.head_dim = Some(64);
        assert_eq!(c.rotated_dim(), 64);
    }

    #[test]
    fn rotated_dim_partial_floor_and_even() {
        let mut c = cfg_with(true, None, 64);
        c.head_dim = Some(256);
        c.partial_rope = Some(0.25);
        // 0.25 * 256 = 64, already even.
        assert_eq!(c.rotated_dim(), 64);

        // 0.1 * 256 = 25.6 → floor 25 → snap to 24 (even).
        c.partial_rope = Some(0.1);
        assert_eq!(c.rotated_dim(), 24);

        // p = 0 → rotated_dim = 0 (no rotation).
        c.partial_rope = Some(0.0);
        assert_eq!(c.rotated_dim(), 0);

        // p = 1 → full head_dim.
        c.partial_rope = Some(1.0);
        assert_eq!(c.rotated_dim(), 256);
    }
}
