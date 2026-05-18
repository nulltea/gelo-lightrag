//! Gemma 4 family — E2B and E4B variant scaffolding.
//!
//! Constants verified against the real HuggingFace configs (2026-05-18
//! session check, public/non-gated):
//! - `google/gemma-4-E2B/raw/main/config.json`
//! - `google/gemma-4-E4B/raw/main/config.json`
//!
//! | Variant | Layers | Hidden · Inter · d_head | Q · KV heads | Ratio (L:G) | SWA W | num_kv_shared_layers | PLE d_ple |
//! | E2B     | 35     | 1536 · 6144  · 256/512g  | 8 · 1        | 4 : 1       | 512   | 20                  | 256       |
//! | E4B     | 42     | 2560 · 10240 · 256/512g  | 8 · 2        | 5 : 1       | 512   | 18                  | 256       |
//!
//! Both variants:
//!  - Hybrid attention: layer_types alternates `sliding_attention` ×N
//!    + `full_attention` × 1; last layer is always full_attention
//!  - p-RoPE p=0.25 on FULL (global) layers only; sliding layers use
//!    full RoPE (rope_type=default)
//!  - Per-class rope_theta: sliding=10000.0, full=1000000.0
//!  - Per-class head_dim: local=256, global=512 (NOT YET WIRED — see
//!    "Architectural gaps" below)
//!  - Activation: `gelu_pytorch_tanh` (GeGLU), NOT SwiGLU
//!  - final_logit_softcapping: 30.0
//!  - tie_word_embeddings: true
//!  - hidden_size_per_layer_input (PLE d_ple): 256, vocab_size_per_layer_input: 262144
//!
//! ## Architectural gaps vs real Gemma 4 (Phase 1.5 follow-ups)
//!
//! The v1 scaffolding lands accurate constants on dimensions our
//! existing `DecoderConfig` can express. The following structural
//! features require deeper refactors and DO NOT engage on real
//! Gemma 4 weights yet:
//!
//! 1. **Per-layer-class head_dim** (256 local / 512 global): every Q
//!    head's projection has different shape per layer class. Current
//!    `cfg.head_dim_value()` returns a single number → would produce
//!    wrong shape on real global-layer Q/K/V projections.
//! 2. **Cross-layer KV sharing** (`num_kv_shared_layers: 20` / 18):
//!    20 (E2B) / 18 (E4B) of the layers REUSE an earlier layer's K
//!    and V cache instead of computing their own. **This is NOT the
//!    within-layer K=V tying that M1.4 implemented** — real Gemma 4
//!    has `attention_k_eq_v: false`. The M1.4 code stays in place
//!    for forward compatibility with K=V-tying-style models but
//!    does NOT engage on Gemma 4. Cross-layer sharing is a separate
//!    Phase 1.5 item.
//! 3. **`use_double_wide_mlp: true`**: real Gemma 4 FFN structure
//!    diverges from the standard `down(act(gate) ⊙ up)` GeGLU; exact
//!    semantics need HF transformers source verification.
//! 4. **Per-class rope_theta**: real model uses two different theta
//!    bases (sliding 10000 vs full 1_000_000). Current `DecoderConfig`
//!    has one `rope_theta`. Phase 1.5 adds a class-aware RopeTables.
//! 5. **AltUp / `altup_*` modules** (mentioned in Gemma 3n
//!    architecture): alternating-update residual stream variant. Not
//!    yet modelled.
//!
//! Until these land, `Gemma4Variant::config()` produces a config that
//! is **architecturally accurate on metadata it can express** but
//! **not yet usable for real-weight forward passes**. The
//! `gemma4_e2e.rs` test stays `#[ignore]` with this rationale
//! recorded.

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

    /// FFN intermediate size — verified against real HF configs.
    pub const fn intermediate_size(self) -> usize {
        match self {
            Self::E2B => 6144,
            Self::E4B => 10240,
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

    /// Sliding-window size used on local layers.
    pub const fn local_window(self) -> usize {
        512
    }

    /// Number of Q heads per layer. Real config: 8 across both
    /// variants.
    pub const fn num_attention_heads(self) -> usize {
        8
    }

    /// Number of KV heads. E2B = 1 (8:1 GQA), E4B = 2 (4:1 GQA).
    /// Verified against real HF configs.
    pub const fn num_key_value_heads(self) -> usize {
        match self {
            Self::E2B => 1,
            Self::E4B => 2,
        }
    }

    /// Head dim for LOCAL (sliding) layers. Global layers use
    /// [`Self::global_head_dim`] which is double this — wiring per-
    /// class head_dim is a Phase 1.5 item (see module docs).
    pub const fn head_dim(self) -> usize {
        256
    }

    /// Head dim for GLOBAL (full-attention) layers. Phase 1.5 item:
    /// not yet plumbed through the per-layer attention dispatch.
    pub const fn global_head_dim(self) -> usize {
        512
    }

    /// Number of layers that REUSE an earlier layer's K and V cache
    /// instead of computing their own (Gemma 4 cross-layer KV sharing).
    /// Phase 1.5 item — not yet implemented in `KvCache` /
    /// `decoder_block_cached`.
    pub const fn num_kv_shared_layers(self) -> usize {
        match self {
            Self::E2B => 20,
            Self::E4B => 18,
        }
    }

    /// p-RoPE rotation factor applied to global-layer Q/K. Local
    /// layers use full rotation per the Gemma 4 spec.
    pub const fn partial_rope_global(self) -> f32 {
        0.25
    }

    /// RoPE base for LOCAL (sliding) layers.
    pub const fn rope_theta_local(self) -> f32 {
        10_000.0
    }

    /// RoPE base for GLOBAL (full-attention) layers. Real config uses
    /// 1_000_000.0 (much wider rotation frequencies than local).
    /// Phase 1.5 item — currently `Gemma4Variant::config()` reports
    /// only one rope_theta; per-class RoPE tables come with the head-
    /// dim refactor.
    pub const fn rope_theta_global(self) -> f32 {
        1_000_000.0
    }

    pub const fn rms_norm_eps(self) -> f32 {
        1e-6
    }

    pub const fn vocab_size(self) -> usize {
        262_144
    }

    pub const fn max_position_embeddings(self) -> usize {
        131_072
    }

    /// Final logit softcap value applied as `tanh(x / cap) * cap`
    /// to the LM head output before sampling. Real Gemma 4 = 30.0.
    pub const fn final_logit_softcapping(self) -> f32 {
        30.0
    }

    /// PLE table embedding dim per layer (`hidden_size_per_layer_input`).
    pub const fn ple_dim(self) -> usize {
        256
    }

    /// Build the per-layer attention-class vector for this variant.
    pub fn attention_classes(self) -> Vec<AttentionClass> {
        gemma4_attention_classes(
            self.num_layers(),
            self.local_per_pattern(),
            self.local_window(),
        )
    }

    /// Build a [`DecoderConfig`] for this variant. Reflects all real-
    /// HF-config dimensions that the current `DecoderConfig` can
    /// express:
    /// - Layer count, hidden_size, intermediate_size, head_dim (local),
    ///   GQA, sliding_window, max_position_embeddings, rope_theta,
    ///   rms_norm_eps, vocab_size, hybrid attention class vector,
    ///   p-RoPE partial-rotary factor, final logit softcap,
    ///   tie_word_embeddings, GeGLU activation.
    ///
    /// Does NOT yet express (Phase 1.5):
    /// - Per-class head_dim (global = 512 vs local = 256)
    /// - Per-class rope_theta (global = 1_000_000 vs local = 10_000)
    /// - Cross-layer KV sharing (num_kv_shared_layers = 20 / 18)
    /// - `use_double_wide_mlp`
    /// - AltUp residual stream variant
    ///
    /// So a forward pass driven by this config produces a model that's
    /// architecturally correct on metadata it can express, but will
    /// NOT produce HF-parity output until the Phase 1.5 items land.
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
            // Local RoPE base; global layers use 1_000_000 but
            // per-class rope_theta is Phase 1.5.
            rope_theta: self.rope_theta_local(),
            // Real activation is `gelu_pytorch_tanh` (GeGLU). Phase 1.5
            // dispatches GeGLU vs SwiGLU in `decoder::swiglu` based on
            // this string.
            hidden_act: "gelu_pytorch_tanh".into(),
            tie_word_embeddings: true,
            max_seq_len: self.max_position_embeddings(),
            skip_first_layers: 0,
            skip_last_layer: false,
            // OutAttnMult / permuted attention default off for the
            // variant baseline; M1.10 wires the fused permuted path.
            use_out_attn_mult: false,
            out_attn_mult_min_seq_len: None,
            use_perm_attention: false,
            perm_attention_min_seq_len: None,
            // Hybrid attention vector + p-RoPE.
            attention_classes: Some(self.attention_classes()),
            partial_rope: Some(self.partial_rope_global()),
            // Real Gemma 4 sets attention_k_eq_v: false. The M1.4 K=V
            // optimisation does NOT engage on Gemma 4 — kept off here
            // so the config truthfully reflects the model. Cross-layer
            // KV sharing (the real Gemma 4 optimisation) is Phase 1.5.
            kv_shared_in_global: false,
            // Soft-cap on output logits (`tanh(x / 30) * 30`).
            final_logit_softcapping: Some(self.final_logit_softcapping()),
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
    fn e2b_config_matches_real_hf_config() {
        // All assertions verified against google/gemma-4-E2B/config.json
        // fetched 2026-05-18.
        let cfg = Gemma4Variant::E2B.config();
        assert_eq!(cfg.num_hidden_layers, 35);
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.intermediate_size, 6144); // was 8192 — corrected
        assert_eq!(cfg.max_position_embeddings, 131_072); // was 32_768 — corrected
        assert_eq!(cfg.head_dim_value(), 256);
        assert_eq!(cfg.q_dim(), 8 * 256);
        assert_eq!(cfg.kv_dim(), 1 * 256);
        assert_eq!(cfg.kv_group_size(), 8);
        assert_eq!(cfg.vocab_size, 262_144);
        assert_eq!(cfg.rope_theta, 10_000.0); // local RoPE base
        assert_eq!(cfg.hidden_act, "gelu_pytorch_tanh"); // was "silu" — corrected
        assert!(cfg.tie_word_embeddings);
        assert!(cfg.is_hybrid_attention());
        assert_eq!(cfg.partial_rope, Some(0.25));
        // Real Gemma 4 has attention_k_eq_v: false — corrected from
        // the M1.4-default-true state.
        assert!(!cfg.kv_shared_in_global);
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert_eq!(cfg.effective_attention_class(0), AttentionClass::Local { window: 512 });
        assert_eq!(cfg.effective_attention_class(34), AttentionClass::Global);
        assert_eq!(cfg.rotated_dim(), 64); // 0.25 * 256 = 64
    }

    #[test]
    fn e4b_config_matches_real_hf_config() {
        // All assertions verified against google/gemma-4-E4B/config.json
        // fetched 2026-05-18.
        let cfg = Gemma4Variant::E4B.config();
        assert_eq!(cfg.num_hidden_layers, 42);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 10240); // was 16384 — corrected
        assert_eq!(cfg.max_position_embeddings, 131_072);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2); // was 1 — corrected
        assert_eq!(cfg.kv_group_size(), 4); // 8 Q heads / 2 KV heads
        assert!(cfg.is_hybrid_attention());
        assert_eq!(cfg.effective_attention_class(41), AttentionClass::Global);
        assert_eq!(cfg.final_logit_softcapping, Some(30.0));
        assert!(!cfg.kv_shared_in_global);
    }

    #[test]
    fn variant_methods_report_phase_15_dimensions() {
        // Sanity check on the metadata the v1 DecoderConfig can't
        // express but Gemma4Variant exposes for Phase 1.5 work.
        let e2b = Gemma4Variant::E2B;
        assert_eq!(e2b.global_head_dim(), 512);
        assert_eq!(e2b.rope_theta_global(), 1_000_000.0);
        assert_eq!(e2b.num_kv_shared_layers(), 20);
        assert_eq!(e2b.ple_dim(), 256);

        let e4b = Gemma4Variant::E4B;
        assert_eq!(e4b.global_head_dim(), 512);
        assert_eq!(e4b.num_kv_shared_layers(), 18);
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
