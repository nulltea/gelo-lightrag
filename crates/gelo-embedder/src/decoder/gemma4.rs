//! Gemma 4 family — E2B and E4B variant scaffolding.
//!
//! The plan in `docs/plans/path-1-gelo-gemma.md` M1.1 specifies the
//! per-variant architecture pinned in the prototype doc
//! `docs/prototype/gelo-llm.html` §05:
//!
//! | Variant | Layers | Hidden · Inter · d_head | Q · KV | Ratio (L:G) | SWA W | PLE       |
//! | E2B     | 35     | 1536 · 8192 · 256       | 8 · 1  | 4 : 1       | 512   | yes (int8)|
//! | E4B     | 42     | 2560 · 16384 · 256      | 8 · 1  | 5 : 1       | 512   | yes (int8)|
//!
//! Both variants:
//!  - GQA(8:1) — already handled by the existing decoder
//!  - p-RoPE p=0.25 on global layers (wired by M1.5)
//!  - K = V tying on global layers (wired by M1.4)
//!  - Last layer always Global, regardless of the local:global ratio
//!
//! This module provides:
//!  - [`Gemma4Variant`] — `E2B` / `E4B` enum
//!  - [`gemma4_attention_classes`] — generic per-layer-class builder
//!    (used by M1.3 to drive hybrid dispatch; here only as the load-
//!    bearing helper that the variant factories call)
//!  - [`Gemma4Variant::config`] — fully-populated [`DecoderConfig`]
//!    with the variant's hybrid pattern + p-RoPE + K=V tying flags set
//!
//! It does NOT yet load real Gemma 4 weights from HuggingFace — that's
//! the M1.8 integration item (the parity test downloads safetensors
//! and feeds them into `DecoderWeights::from_safetensors`). M1.1's
//! acceptance is the synthetic-weight test in
//! `tests/generation_harness.rs` running greedy `generate()` under
//! an E2B-shaped config; the new fields don't change the forward path
//! until M1.3 wires them.

use super::config::{AttentionClass, DecoderConfig};

/// The two v1 Gemma 4 variants in scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Variant {
    /// E2B: 35 layers, 4:1 local:global, 1536 hidden, 8192 intermediate.
    /// PLE: `262_144 × 35 × 256` int8.
    E2B,
    /// E4B: 42 layers, 5:1 local:global, 2560 hidden, 16384 intermediate.
    /// PLE: `262_144 × 42 × 256` int8.
    E4B,
}

impl Gemma4Variant {
    /// Number of decoder layers.
    pub const fn num_layers(self) -> usize {
        match self {
            Self::E2B => 35,
            Self::E4B => 42,
        }
    }

    /// Hidden size `d_hidden`.
    pub const fn hidden_size(self) -> usize {
        match self {
            Self::E2B => 1536,
            Self::E4B => 2560,
        }
    }

    /// FFN intermediate size.
    pub const fn intermediate_size(self) -> usize {
        match self {
            Self::E2B => 8192,
            Self::E4B => 16384,
        }
    }

    /// Number of local-attention layers per repeating pattern unit
    /// (the "L" in the "L:G" ratio). E2B = 4, E4B = 5.
    pub const fn local_per_pattern(self) -> usize {
        match self {
            Self::E2B => 4,
            Self::E4B => 5,
        }
    }

    /// Sliding-window size used on local layers (Gemma 4 small models
    /// use W=512; the 31B dense variant — out of v1 scope — uses
    /// W=1024).
    pub const fn local_window(self) -> usize {
        512
    }

    pub const fn num_attention_heads(self) -> usize {
        // Same head count across both variants: 8 Q heads, GQA 8:1.
        8
    }

    pub const fn num_key_value_heads(self) -> usize {
        1
    }

    pub const fn head_dim(self) -> usize {
        256
    }

    /// p-RoPE rotation factor applied to global-layer Q/K. Local
    /// layers use full rotation per the Gemma 4 spec.
    pub const fn partial_rope_global(self) -> f32 {
        0.25
    }

    pub const fn rope_theta(self) -> f32 {
        // Gemma 4 family uses the same base as Gemma 3 / Qwen3.
        // M1.8 parity will pin this against the HF config; until then
        // 10_000.0 matches the existing default.
        10_000.0
    }

    pub const fn rms_norm_eps(self) -> f32 {
        1e-6
    }

    pub const fn vocab_size(self) -> usize {
        262_144
    }

    pub const fn max_position_embeddings(self) -> usize {
        // Both small variants support 32k context. Long-context
        // benchmarks (n ≥ 8k) drive the M1.10 fused permuted attention
        // workstream.
        32_768
    }

    /// Build the per-layer attention-class vector for this variant.
    pub fn attention_classes(self) -> Vec<AttentionClass> {
        gemma4_attention_classes(
            self.num_layers(),
            self.local_per_pattern(),
            self.local_window(),
        )
    }

    /// Build a fully-populated [`DecoderConfig`] for this variant. Used
    /// by tests and by the (forthcoming) Gemma 4 model loader to seed
    /// the config when no `config.json` is supplied. Does not load
    /// weights — that's `DecoderWeights::from_safetensors`'s job.
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
            skip_first_layers: 0,
            skip_last_layer: false,
            // OutAttnMult / permuted attention default off for the
            // variant baseline; M1.10 wires the fused permuted path
            // and tunes the auto-switch threshold from there.
            use_out_attn_mult: false,
            out_attn_mult_min_seq_len: None,
            use_perm_attention: false,
            perm_attention_min_seq_len: None,
            // Hybrid attention + Gemma-specific extensions.
            attention_classes: Some(self.attention_classes()),
            partial_rope: Some(self.partial_rope_global()),
            kv_shared_in_global: true,
        }
    }
}

/// Generic per-layer attention-class builder for the Gemma family.
///
/// Lays down repeats of `[Local; local_per_pattern, Global]` until the
/// layer count is filled, then forces the LAST layer to be `Global`
/// regardless of where the natural ratio would have placed it. The
/// last-layer override matches the Gemma paper's stated rule.
///
/// Panics on `num_layers = 0`, `local_per_pattern = 0`, or
/// `window = 0` — those would all encode meaningless configurations
/// (no model / all-global / zero-width SWA).
pub fn gemma4_attention_classes(
    num_layers: usize,
    local_per_pattern: usize,
    window: usize,
) -> Vec<AttentionClass> {
    assert!(num_layers > 0, "gemma4_attention_classes: num_layers must be > 0");
    assert!(
        local_per_pattern > 0,
        "gemma4_attention_classes: local_per_pattern must be > 0",
    );
    assert!(window > 0, "gemma4_attention_classes: window must be > 0");

    let pattern_unit = local_per_pattern + 1; // L locals + 1 global
    let mut classes = Vec::with_capacity(num_layers);
    for li in 0..num_layers {
        let pos_in_pattern = li % pattern_unit;
        if pos_in_pattern < local_per_pattern {
            classes.push(AttentionClass::Local { window });
        } else {
            classes.push(AttentionClass::Global);
        }
    }
    // Spec: the last layer is always Global, regardless of where the
    // natural pattern would have placed it. Overwrite unconditionally.
    *classes.last_mut().expect("num_layers > 0") = AttentionClass::Global;
    classes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2b_has_35_layers_with_4to1_pattern_and_last_global() {
        let classes = Gemma4Variant::E2B.attention_classes();
        assert_eq!(classes.len(), 35);
        // First five = 4 local + 1 global.
        for (i, c) in classes.iter().take(4).enumerate() {
            assert_eq!(
                *c,
                AttentionClass::Local { window: 512 },
                "E2B layer {i} should be Local(512)",
            );
        }
        assert_eq!(classes[4], AttentionClass::Global);
        // Last layer always Global.
        assert_eq!(classes[34], AttentionClass::Global);
        // Spot check: one of the pattern-positioned globals at li=4, 9, 14, ...
        for li in [4, 9, 14, 19, 24, 29].iter() {
            assert_eq!(classes[*li], AttentionClass::Global);
        }
    }

    #[test]
    fn e4b_has_42_layers_with_5to1_pattern_and_last_global() {
        let classes = Gemma4Variant::E4B.attention_classes();
        assert_eq!(classes.len(), 42);
        for (i, c) in classes.iter().take(5).enumerate() {
            assert_eq!(
                *c,
                AttentionClass::Local { window: 512 },
                "E4B layer {i} should be Local(512)",
            );
        }
        assert_eq!(classes[5], AttentionClass::Global);
        // Last layer always Global.
        assert_eq!(classes[41], AttentionClass::Global);
        // Spot check: pattern globals at li=5, 11, 17, 23, 29, 35.
        for li in [5, 11, 17, 23, 29, 35].iter() {
            assert_eq!(classes[*li], AttentionClass::Global);
        }
    }

    #[test]
    fn last_layer_override_engages_when_pattern_would_be_local() {
        // 4:1 ratio, 4 layers: pattern would be all-local. The
        // last-always-Global override must flip layer 3 to Global.
        let classes = gemma4_attention_classes(4, 4, 256);
        assert_eq!(classes[0], AttentionClass::Local { window: 256 });
        assert_eq!(classes[1], AttentionClass::Local { window: 256 });
        assert_eq!(classes[2], AttentionClass::Local { window: 256 });
        assert_eq!(classes[3], AttentionClass::Global, "last-layer override");
    }

    #[test]
    fn config_round_trip_populates_hybrid_fields() {
        let cfg = Gemma4Variant::E2B.config();
        assert_eq!(cfg.num_hidden_layers, 35);
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.head_dim_value(), 256);
        assert_eq!(cfg.q_dim(), 8 * 256);
        assert_eq!(cfg.kv_dim(), 1 * 256);
        assert_eq!(cfg.kv_group_size(), 8);
        assert!(cfg.is_hybrid_attention());
        assert_eq!(cfg.partial_rope, Some(0.25));
        assert!(cfg.kv_shared_in_global);
        assert_eq!(cfg.effective_attention_class(0), AttentionClass::Local { window: 512 });
        assert_eq!(cfg.effective_attention_class(34), AttentionClass::Global);
        assert_eq!(cfg.rotated_dim(), 64); // 0.25 * 256 = 64
    }

    #[test]
    fn e4b_config_consistency() {
        let cfg = Gemma4Variant::E4B.config();
        assert_eq!(cfg.num_hidden_layers, 42);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 16384);
        assert!(cfg.is_hybrid_attention());
        assert_eq!(cfg.effective_attention_class(41), AttentionClass::Global);
    }

    #[test]
    #[should_panic(expected = "num_layers")]
    fn zero_layers_panics() {
        let _ = gemma4_attention_classes(0, 4, 512);
    }

    #[test]
    #[should_panic(expected = "local_per_pattern")]
    fn zero_local_per_pattern_panics() {
        let _ = gemma4_attention_classes(35, 0, 512);
    }

    #[test]
    #[should_panic(expected = "window")]
    fn zero_window_panics() {
        let _ = gemma4_attention_classes(35, 4, 0);
    }
}
