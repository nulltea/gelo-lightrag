---
type: dev-log
status: current
created: 2026-05-18
updated: 2026-05-21
tags: [gemma, path-1, gemma4, aloepri]
companion: [2026-05-18-gemma4-architecture-support, 2026-05-21-aloepri-gemma-deferred]
---

# Gemma + path-1 — distilled architecture log

> Knowledge layer for Gemma 3/4 architecture support readiness (path-1 GELO)
> and AloePri-on-Gemma workstream. Parked infrastructure, pivot rationale,
> and entry point for future Gemma support after Qwen3 validation completes.

## Gemma 4 architecture support state

**Status:** Deferred. Phase 1.5 milestone blocked; v1 pivoted to Qwen3-class models.

**Dormant infrastructure in tree (shipped, tested, reusable):**
- `AttentionClass::{Local, Global}` enum + per-layer-class dispatch.
- `PleTable` data structure + `TrustedExecutor::provision_ple_table` + PCIe-leak verification (P0 fix).
- `causal_gqa_attention_swa_cached(window, q_pos_offset)` sliding-window kernel.
- `RopeTables::apply_partial_at(rotated_dim)` p-RoPE kernel (partial rotation for global layers).
- `LayerKvCache` Separate/Shared enum (forward-compatible for Gemma 3 variants; doesn't engage on Gemma 4).
- `Gemma4Variant::{E2B, E4B}` config builder with HF-verified constants.
- `final_logit_softcapping: Option<f32>` on `DecoderConfig` (Gemma 4 logit soft-cap, tanh(x/30)/30, fully done).
- `gemma4_e2e.rs` integration test + `gemma4_hf_parity.rs` parity stub (both `#[ignore]`d, Phase 1.5-blocked).
- `GpuOffloadEngine::fused_attention_batched` trait method (seam for M1.10 fused permuted FlashAttention).

**Phase 1.5 critical path (4–5 weeks to greedy-generate parity on real E2B weights):**

| § | Item | Effort |
|---|---|---|
| §8.1 | GeGLU activation dispatch | 1–2 days |
| §8.2 | Per-class head_dim refactor | 2 weeks (most invasive) |
| §8.3 | Per-class rope_theta | 3 days |
| §8.4 | Cross-layer KV sharing (corrects M1.4) | 2 weeks |
| §8.5 | use_double_wide_mlp semantics | 1–2 days |
| §8.6 | AltUp residual stream | 1 week |

---

## AloePri on Gemma (deferred)

**Status:** Deferred. Active work pivoted to Qwen3 on 2026-05-18.

**Blocker:** Gemma 4's **post-norms** (3 per block: `post_attention_norm`, `post_ffw_norm`, `post_norm`) cannot be made covariant by paper §5.2.5 fuse-and-scale construction alone.

**Why:** paper's fusion works for pre-norms where `y = (x · γ) / RMS(x) == (x / RMS(x)) @ (Diag(γ) · W)`. For post-norms where `y = (out · γ) / RMS(out)`, the fusion becomes `y' = (out · γ) / RMS(out · γ)`, and the ratio `RMS(out) / RMS(out · γ) ≈ 1/√d` per site. Over 35 layers × 3 post-norms, the error compounds catastrophically (measured: produces gibberish).

**Paper baseline:** Qwen / Llama / DeepSeek dense use only pre-norms (2 per block). Gemma 4 with 5 residual sites (3 post + 2 pre per block) is the outlier.

**Findings to carry forward:**
- llama.cpp Gemma 4 support (`LLM_ARCH_GEMMA4`) is shipped; no fork needed for architecture itself.
- `hidden_size = d + 2h` expansion propagates cleanly (no assertion requiring `hidden_size == n_heads · head_dim`).
- K=V tying is runtime cache sharing only; GGUF stores separate `attn_k`, `attn_v` at every layer (no weight untying needed).
- PLE is one fused `[262144 × 8960]` tensor (τ permutation is one numpy operation).
- Algorithm 1 KeyMat math verified to ≤3·10⁻⁷ max-absolute-error at fp64 for d=1536 and d=2560.
- κ_correct ≈ 7.42 for E2B (d=1536, h=128, λ=0.3).

**Path forward after Qwen3 validation completes:**
- **Option A (recommended):** add `ggml_rms_norm_then_scale(x, γ_per_dim_obf, κ)` op to llama.cpp (2–3 weeks, guarded by metadata flag, ~30 lines diff).
- **Option B (research):** find a novel §5.2.5-style construction handling post-norms offline (algebraic reformulation; unknown feasibility).

---

## What's blocked, what's available

**Blocked for path-1 v1 (Qwen3 demonstrator):**
- Phase 1.5 work — Gemma 4 hybrid attention, PLE, p-RoPE wiring is future-work.
- MoE generation (Qwen3-MoE / Gemma 26B-A4B) — routing-histogram leak is round-2 P1; separate CryptoMoE protocol surface (plan §7.2 deferred).
- Speculative decoding under per-batch fresh mask — security unexamined; gated on per-forward-pass mask story.
- Gemma 4 31B dense — no 64 GB CVM SKU committed.

**Available (shipped, tested, dormant):**
- All Phase 1.5 infrastructure listed above — validated by synthetic tests, ready to wire into `DecoderConfig` once milestones start.
- `gelo-llm.html` design + threat model (round-2 research backing in current `private-llm-inference.md`).
- PLE address-bus leak fix (P0, verified) and PCIe-leak test.
- Sliding-window + p-RoPE kernels ready for hybrid-attention dispatch.
