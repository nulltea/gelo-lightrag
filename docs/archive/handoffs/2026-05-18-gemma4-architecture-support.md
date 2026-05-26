---
type: handoff
status: current
created: 2026-05-18
updated: 2026-05-18
tags: [gemma, gemma4]
---

# Gemma 3 / 4 architecture support roadmap

> **Status.** Deferred while v1 pivots to Qwen3-class models. All the
> Gemma-family-specific code is in the tree as **dormant
> infrastructure** — validated by synthetic tests, not yet exercised
> against real Gemma weights. This document is the handoff for the
> worker who picks up Gemma 3/4 architecture support later.
>
> **Audience.** Future implementer of the Phase 1.5 milestone block in
> [`../../plans/path-1-gelo-gemma.md`](../../plans/path-1-gelo-gemma.md) §8.
> Read that section first — it has the milestone breakdown, effort
> estimates, and dependency graph. This document covers the *context*
> a fresh agent needs to pick up the work without re-deriving the
> 2026-05-18 audit conclusions.
>
> **Not duplicated here:**
> - Per-milestone scope, files, acceptance criteria → plan §8
> - Public-facing design (threat model, components, compute flow) →
>   [`gelo-llm.html`](gelo-llm.html)
> - LLM serving primitives (fused attention, decode KV cache) →
>   [`../../dev/prototype/gelo-llm.md`](../../dev/prototype/gelo-llm.md)
> - Round-2 research backing → [`../../research/private-llm-inference-round-2.md`](../../research/private-llm-inference-round-2.md)

---

## 1. Why this is parked

The 2026-05-18 architecture audit (plan §0 status table entry of the
same date) fetched the real `google/gemma-4-E2B` and `-E4B` configs
and discovered that real Gemma 4 inference needs six structural
extensions our `DecoderConfig` doesn't yet express. Estimated 4-5
weeks of refactor work to make any real-weight forward pass produce
HF-parity output.

In the same session the user decided to pivot the v1 demonstration
target to Qwen3-class small models (1.7B / 4B), which use the
vanilla-decoder architecture our existing `GeloQwenEmbedder` /
`decoder` stack already handles. Gemma 3/4 work becomes a Phase 2
milestone instead of a v1 blocker. Phase 1.5 stays as the path back
to Gemma support.

---

## 2. What's already in the tree (dormant infrastructure)

Each item below is **shipped, tested, and reusable** — but does NOT
engage on Qwen3 weights or on real Gemma weights until the Phase 1.5
refactors land.

| Item | Commit | What it gives Gemma support | What's missing |
|---|---|---|---|
| `AttentionClass::{Local, Global}` enum + `attention_classes: Option<Vec<…>>` on `DecoderConfig` | `060f053` | Per-layer-class dispatch surface for hybrid attention | Per-class `head_dim` + `rope_theta` (§8.2, §8.3) |
| `PleTable` data structure + `TrustedExecutor::provision_ple_table` / `ple_gather` extensions + PCIe-leak verification test (P0) | `4d59d81` | The protocol-level fix for the round-2 P0 PLE address-bus leak | Loader bridge from Gemma 4 safetensors PLE blob (§8.7) |
| `causal_gqa_attention_swa_cached(window, q_pos_offset)` sliding-window kernel | `43f4c7c` | In-TEE SWA for local layers; band-mask + decode-replay verified | Wiring needs per-class `head_dim` (§8.2) |
| `RopeTables::apply_partial_at(rotated_dim)` p-RoPE kernel | `2655edd` | Partial rotation for Gemma 4 global layers (0.25 × head_dim) | Per-class `rope_theta` (§8.3) |
| `LayerKvCache` Separate/Shared enum + `KvCache::new_with_sharing` | `66bba90` | Within-layer K=V tying — forward-compatible with Gemma 3 variants where `attention_k_eq_v: true`. **Does NOT engage on real Gemma 4** (which sets the flag false). | Cross-layer KV sharing — the actually-used Gemma 4 mechanism (§8.4) |
| `Gemma4Variant::{E2B, E4B}` config builder with HF-verified constants | `416fed7` | Single source of truth for variant dimensions; exposes Phase 1.5 metadata accessors (`global_head_dim`, `rope_theta_global`, `num_kv_shared_layers`, `ple_dim`) | Phase 1.5 refactors to wire these into `DecoderConfig` |
| `final_logit_softcapping: Option<f32>` on `DecoderConfig` + `compute_logits` integration | `416fed7` | Gemma 4 logit soft-cap (`tanh(x/c) * c`, c=30.0) | None — this is fully done |
| `gemma4_e2e.rs` integration test (`#[ignore]`d) + `gemma4_hf_parity.rs` parity stub | `dc9074d` (scaffolding) + `416fed7` (corrected blocker rationale) | Test bodies pinned to `google/gemma-4-E2B`; un-ignore prerequisites documented at the top of each file | Phase 1.5 + fixture JSON |
| `GpuOffloadEngine::fused_attention_batched` trait method with default-impl 3-dispatch composition | `dc9074d` | Engine seam for the M1.10 fused permuted FlashAttention kernel | The actual GPU kernel (cubek / upstream PR / custom WGSL — gated on burn-cubecl maturity per plan §7.2) |
| `decoder/gemma4.rs` module docs listing Phase 1.5 gaps inline | `416fed7` | Authoritative in-source pointer to the deferred-work list | n/a |

---

## 3. The Phase 1.5 milestone list

Reference: [`../../plans/path-1-gelo-gemma.md`](../../plans/path-1-gelo-gemma.md) §8.

In brief, in the critical-path order:

```
§8.1 GeGLU activation dispatch    (1-2 days)  ┐
                                              ├─→ §8.7 loader (1w) → flip M1.6 / M1.8 #[ignore]
§8.2 Per-class head_dim refactor  (2 weeks)   ┤
§8.3 Per-class rope_theta          (3 days)    ┤
§8.4 Cross-layer KV sharing        (2 weeks)   ┘
§8.5 use_double_wide_mlp semantics (1-2 days)  ─ (post-flip refinement, independent)
§8.6 AltUp residual stream         (~1 week)   ─ (post-flip refinement, independent)
```

Total realistic timeline from kickoff to a successful greedy
`generate()` against `google/gemma-4-E2B` real weights: **4-5 weeks**.

---

## 4. Open design questions the next worker has to resolve

These weren't settled in the audit session — needs HF transformers
source check or further research before §8 work starts.

1. **Cross-layer KV sharing producer-layer rule (§8.4).** The audit
   inferred that "shared layers reference the most recent same-class
   prior layer's KV". Confirm against `transformers/src/transformers/models/gemma4`
   on github — the exact rule may differ (e.g. paired indices, fixed
   step). The `Gemma4Variant::num_kv_shared_layers` accessor reports
   only the count, not the topology.
2. **Per-class head_dim placement (§8.2).** Should the per-class
   dim live on `DecoderConfig` (a `head_dim_local` + `head_dim_global`
   pair, plus a `head_dim_for(layer_class)` accessor) or on
   `DecoderLayerWeights` (each layer reports its own head_dim derived
   from its `wq.ncols() / num_attention_heads`)? Config-side is
   cleaner; per-layer is more flexible.
3. **`use_double_wide_mlp` semantics (§8.5).** Real Gemma 4 sets
   this to `true`. Likely either the reported `intermediate_size`
   is half the actual gate/up width OR a different MLP topology is
   used. Read HF transformers source before sizing the work.
4. **AltUp residual stream (§8.6).** Gemma 3n architecture detail.
   Inferred to be inherited by Gemma 4 but not yet confirmed against
   the live HF implementation.
5. **PLE quantisation scale layout.** `PleTable::from_int8_rows`
   currently takes one scale per table. The HF safetensors blob may
   ship per-channel scales — verify and extend `PleTable` if so.
   Per-block scales are also possible.

---

## 5. Important correction vs the original design plan

The M1.4 milestone in the original plan was titled "K=V tying on
global layers" and assumed the within-layer K=V trick. The audit
revealed real Gemma 4 has `attention_k_eq_v: false`, so M1.4's
implementation **does not engage on Gemma 4**. The real Gemma 4
optimisation is **cross-layer KV sharing** (`num_kv_shared_layers`)
— a different mechanism entirely, scoped as Phase 1.5 §8.4.

The M1.4 code stays in the tree as forward-compatible infrastructure
for Gemma 3 / 3n variants that DO set `attention_k_eq_v: true`. Its
synthetic tests still validate it. Don't delete it during the §8.4
work — they coexist.

---

## 6. Cross-references to live code

| Concern | File |
|---|---|
| Per-layer attention class enum | `crates/gelo-embedder/src/decoder/config.rs` (search `AttentionClass`) |
| Variant constants + Phase 1.5 metadata | `crates/gelo-embedder/src/decoder/gemma4.rs` |
| PLE data structure + gather | `crates/gelo-protocol/src/ple.rs` |
| PCIe-leak verification | `crates/gelo-protocol/tests/ple_pcie_leak.rs` |
| Sliding-window attention | `crates/gelo-embedder/src/decoder/attention.rs::causal_gqa_attention_swa_cached` |
| p-RoPE partial rotation | `crates/gelo-embedder/src/decoder/rope.rs::apply_partial_at` |
| KvCache Separate/Shared | `crates/gelo-embedder/src/decoder/kv_cache.rs::LayerKvStore` |
| Hybrid dispatch site | `crates/gelo-embedder/src/decoder/forward.rs::decoder_block_cached` |
| Final-logit softcap | `crates/gelo-embedder/src/decoder/generation.rs::compute_logits` |
| E2E integration test (ignored, Phase 1.5-blocked) | `crates/gelo-embedder/tests/gemma4_e2e.rs` |
| HF parity test (ignored, Phase 1.5-blocked) | `crates/gelo-embedder/tests/gemma4_hf_parity.rs` |

---

## 7. Pre-flight checks before starting Phase 1.5

Before writing any §8 code, the next session should:

1. **Read `crates/gelo-embedder/src/decoder/gemma4.rs` module
   docstring in full** — it's the canonical record of every Phase 1.5
   gap with line-of-sight to the metadata accessors.
2. **Fetch `google/gemma-4-E2B/config.json` again** — confirm no
   field changes since the 2026-05-18 audit.
3. **Read `transformers/src/transformers/models/gemma4/modeling_gemma4.py`**
   (or equivalent in the current HF transformers tree) — this is the
   ground truth for the cross-layer KV sharing rule, `use_double_wide_mlp`
   semantics, and AltUp wiring.
4. **Decide §8.2 head_dim placement** before §8.4 begins — the
   `LayerKvCache` refactor in §8.4 depends on knowing the per-layer
   head_dim resolution path.

---

## 8. v1 pivot context (why this is parked)

The v1 demonstration target is now Qwen3 small models
(`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`), which are pure-decoder
architectures matching the existing `GeloQwenEmbedder` stack
byte-for-byte. None of the Phase 1.5 work blocks Qwen3 v1 —
hybrid attention, PLE, p-RoPE, K=V, GeGLU, AltUp, soft-cap, per-class
head_dim are all dormant on Qwen3 (`sliding_window: null`,
`use_sliding_window: false`, full GQA, single rope_theta, SwiGLU
activation).

**Unresolved at handoff time:** the user mentioned "Qwen 3.5" as the
pivot target, but search results suggested Qwen3.5 may use a "GDN
hybrid architecture" (Gated DeltaNet, ~Feb-March 2026 release). The
audit session was interrupted before this could be verified. If the
next Qwen3 pivot session targets Qwen3.5 specifically, the very first
check should be: fetch a Qwen3.5 small config (e.g.
`Qwen/Qwen3.5-4B-Base`) and confirm whether `model_type` is plain
`qwen3` (vanilla path our code handles) or something hybrid. If
hybrid, the Gemma-style Phase 1.5 work effectively re-applies with
different naming.

---

## 9. Suggested next-session skills

When the time comes to land Phase 1.5:

- **`Plan`** subagent to scope §8.2 (the per-class head_dim refactor) —
  it's the largest and most invasive item, and benefits from
  upfront architectural deliberation before code starts.
- **`Explore`** subagent for the HF transformers source-reading phase —
  cross-layer KV sharing, `use_double_wide_mlp`, AltUp all need
  source-of-truth verification before writing.
- **`improve-codebase-architecture`** skill could be useful for §8.2's
  `head_dim_value()` → `head_dim_for(layer_class)` migration since
  it touches many call sites uniformly.
- **`diagnose`** skill once `gemma4_hf_parity.rs` runs and (probably)
  diverges from HF reference — the divergence-bisection workflow
  is exactly what that skill is built for.

---

## 10. Out-of-scope but worth flagging

These are NOT Phase 1.5 items (orthogonal workstreams) but might
land alongside or be triggered by Gemma support:

- **MoE generation** (Qwen3-MoE / Gemma 4 26B-A4B): routing-histogram
  leak is the round-2 P1 finding. Separate protocol surface
  (CryptoMoE balanced dispatch). Plan §7.2 has the deferred reference.
- **Gemma 4 31B dense**: dropped from v1 per the 2026-05-18 decision
  (no 64 GB CVM SKU committed). Phase 2+ if hardware lands.
- **Speculative decoding under per-batch fresh `A`**: completely
  unexplored security-wise. Probably out of scope until the
  protocol's per-forward-pass mask story can survive a draft +
  verify-step pattern.
