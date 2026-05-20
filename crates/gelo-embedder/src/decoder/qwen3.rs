//! Qwen3 family — v1 generation-target variant scaffolding.
//!
//! Constants verified against the real HuggingFace config (2026-05-18
//! session check, public/non-gated):
//! - `Qwen/Qwen3-1.7B/raw/main/config.json`
//!
//! | Variant | Layers | Hidden · Inter | d_head | Q · KV heads | rope_theta | vocab  |
//! | Q1_7B   | 28     | 2048 · 6144    | 128    | 16 · 8       | 1_000_000  | 151936 |
//!
//! Architecture is **vanilla GQA decoder**:
//!  - SwiGLU activation (`silu`)
//!  - GQA 2:1 (16 attn heads / 8 KV heads)
//!  - Full causal attention, no sliding window
//!     (`use_sliding_window: false`, `sliding_window: null`)
//!  - No hybrid attention / no per-layer attention class vector
//!  - Full RoPE (no `partial_rotary_factor`); single `rope_theta`
//!  - **QK-norm**: Qwen3-specific per-head RMSNorm on Q and K **before**
//!    RoPE (`self_attn.q_norm.weight`, `self_attn.k_norm.weight`,
//!    each `(head_dim,)`). Loaded as `Option<Array1<f32>>` on
//!    `DecoderLayerWeights` and applied by
//!    `rms_norm::apply_qk_norm` in the forward path.
//!  - No final-logit softcap
//!  - `tie_word_embeddings: true`
//!  - `attention_bias: false` (no biases anywhere — matches the
//!    existing loader's no-bias assumption)
//!
//! ## Why Qwen3 is the v1 generation target (not Qwen3.5, not Gemma 4)
//!
//! Verified at 2026-05-18:
//!  - **Gemma 4 E2B/E4B** require Phase 1.5 architectural work
//!    (hybrid attention, PLE, p-RoPE, cross-layer KV sharing, AltUp,
//!    GeGLU dispatch, use_double_wide_mlp). Parked — see
//!    `docs/prototype/gemma4-architecture-roadmap.md`.
//!  - **Qwen3.5 family** (2B / 4B / 9B / 35B-A3B) ships `Qwen3_5ForConditional`
//!    Generation` — multimodal VLM with a 24-layer vision encoder,
//!    `linear_attention` (Gated DeltaNet / SSM) layers in a 3:1
//!    hybrid with `full_attention`, MRoPE with `mrope_section`
//!    interleaving, `partial_rotary_factor: 0.25`, MTP head,
//!    `attn_output_gate`. Strictly harder than Gemma 4, not easier.
//!  - **Qwen3 1.7B / 4B** are vanilla GQA decoders with QK-norm —
//!    the only Qwen3-vs-Qwen2 change is the QK-norm step, which
//!    fits cleanly under our existing `decoder_block` / RoPE
//!    plumbing as an optional pre-RoPE per-head RMSNorm.

use super::config::DecoderConfig;

/// The v1 Qwen3 variants in scope. Q1_7B is the demonstrator target;
/// Q4B is a near-future stretch goal that drops in by changing only the
/// constants below (same architecture, just bigger).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3Variant {
    /// Qwen3-1.7B: 28 layers, 2048 hidden, 6144 intermediate, 16/8 heads.
    Q1_7B,
    /// Qwen3-4B: 36 layers, 2560 hidden, 9728 intermediate, 32/8 heads.
    /// Constants pinned from `Qwen/Qwen3-4B/raw/main/config.json`.
    Q4B,
}

impl Qwen3Variant {
    /// Number of decoder layers.
    pub const fn num_layers(self) -> usize {
        match self {
            Self::Q1_7B => 28,
            Self::Q4B => 36,
        }
    }

    /// Hidden size `d_hidden`.
    pub const fn hidden_size(self) -> usize {
        match self {
            Self::Q1_7B => 2048,
            Self::Q4B => 2560,
        }
    }

    /// FFN intermediate size.
    pub const fn intermediate_size(self) -> usize {
        match self {
            Self::Q1_7B => 6144,
            Self::Q4B => 9728,
        }
    }

    /// Number of Q heads per layer.
    pub const fn num_attention_heads(self) -> usize {
        match self {
            Self::Q1_7B => 16,
            Self::Q4B => 32,
        }
    }

    /// Number of KV heads — GQA 2:1 (Q1_7B) / 4:1 (Q4B).
    pub const fn num_key_value_heads(self) -> usize {
        8
    }

    /// Head dim is independent of `hidden_size / num_attention_heads`
    /// for Qwen3 (Q1_7B: 128 even though 2048/16 = 128; Q4B: 128 even
    /// though 2560/32 = 80). Always pin explicitly.
    pub const fn head_dim(self) -> usize {
        128
    }

    pub const fn rope_theta(self) -> f32 {
        1_000_000.0
    }

    pub const fn rms_norm_eps(self) -> f32 {
        1e-6
    }

    pub const fn vocab_size(self) -> usize {
        151_936
    }

    pub const fn max_position_embeddings(self) -> usize {
        40_960
    }

    /// Stable HuggingFace model id for this variant. Used by `from_pretrained`
    /// helpers so re-pinning a variant target is a one-line change.
    pub const fn hf_model_id(self) -> &'static str {
        match self {
            Self::Q1_7B => "Qwen/Qwen3-1.7B",
            Self::Q4B => "Qwen/Qwen3-4B",
        }
    }

    /// Build a [`DecoderConfig`] for this variant. All Gemma 4-specific
    /// extensions are explicitly disabled: no hybrid attention, no
    /// partial RoPE, no K=V sharing, no final-logit softcap.
    pub fn config(self) -> DecoderConfig {
        DecoderConfig {
            vocab_size: self.vocab_size(),
            hidden_size: self.hidden_size(),
            intermediate_size: self.intermediate_size(),
            num_hidden_layers: self.num_layers(),
            num_attention_heads: self.num_attention_heads(),
            num_key_value_heads: self.num_key_value_heads(),
            head_dim: Some(self.head_dim()),
            max_position_embeddings: self.max_position_embeddings(),
            rms_norm_eps: self.rms_norm_eps(),
            rope_theta: self.rope_theta(),
            hidden_act: "silu".into(),
            tie_word_embeddings: true,
            max_seq_len: self.max_position_embeddings(),
            // Sensitive-layer exclusion (paper §3.2 + DP-Forward §5.2) is
            // a security recommendation we would like to honour by default,
            // but on this substrate it currently costs ~234 ms TPOT per
            // decode step at n=2048 — the in-TEE direct matmul falls back
            // to `ndarray::dot()` at the decode shape `(1, d)·(d, p)` which
            // hits a slow path in matrixmultiply (~1 GFLOP/s observed at
            // m=1 vs ~125 GFLOP/s nominal). The `tee_matmul` BLIS-mt routing
            // closes the *prefill* side of the cost (saves ~1.0 s TTFT) but
            // not the decode side because BLIS at m=1 has too much
            // per-call thread overhead. See
            // `memory/tee_direct_m1_gemv_slowness.md` for the open
            // optimisation surface. Default stays off until the m=1 GEMV
            // path is fast enough that decode TPOT doesn't regress.
            skip_first_layers: 0,
            skip_last_layer: false,
            // Match existing GeloQwenEmbedder defaults: OutAttnMult on
            // (auto-switches by sequence length), permuted attention off.
            use_out_attn_mult: true,
            out_attn_mult_min_seq_len: None,
            use_perm_attention: false,
            perm_attention_min_seq_len: None,
            // All Gemma-specific fields off.
            attention_classes: None,
            partial_rope: None,
            kv_shared_in_global: false,
            final_logit_softcapping: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q1_7b_config_matches_real_hf_config() {
        // All assertions verified against Qwen/Qwen3-1.7B/config.json
        // fetched 2026-05-18.
        let cfg = Qwen3Variant::Q1_7B.config();
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim_value(), 128);
        assert_eq!(cfg.q_dim(), 16 * 128);
        assert_eq!(cfg.kv_dim(), 8 * 128);
        assert_eq!(cfg.kv_group_size(), 2);
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.max_position_embeddings, 40_960);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.hidden_act, "silu");
        assert!(cfg.tie_word_embeddings);
        // No Gemma extensions:
        assert!(!cfg.is_hybrid_attention());
        assert!(cfg.attention_classes.is_none());
        assert!(cfg.partial_rope.is_none());
        assert!(!cfg.kv_shared_in_global);
        assert!(cfg.final_logit_softcapping.is_none());
    }

    #[test]
    fn q4b_constants_pinned() {
        let cfg = Qwen3Variant::Q4B.config();
        assert_eq!(cfg.num_hidden_layers, 36);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 9728);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim_value(), 128);
    }

    #[test]
    fn hf_model_id_is_stable() {
        assert_eq!(Qwen3Variant::Q1_7B.hf_model_id(), "Qwen/Qwen3-1.7B");
        assert_eq!(Qwen3Variant::Q4B.hf_model_id(), "Qwen/Qwen3-4B");
    }
}
