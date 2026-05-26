---
type: plan
status: current
created: 2026-05-18
updated: 2026-05-18
tags: [path-1, gelo, gemma, qwen3]
---

# Path 1 — GELO TEE-GPU Split Inference for Gemma E2B/E4B

> **Worktree:** original (this one). Branch: `path-1-gelo-gemma`
> (to be created from `master`).
>
> **Sibling plan:** [`aloepri-gemma.md`](aloepri-gemma.md)
> develops in `../private-rag-path-2`.
>
> **Shared framework:** [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md).
>
> **Goal:** Extend the existing GELO+TwinShield prototype (currently
> validated on Qwen3-Embedding-0.6B) to support **autoregressive text
> generation on Gemma E2B/E4B** with the openweight threat model
> preserved. Produce performance, accuracy, and attack-resistance
> numbers comparable to AloePri on the same models.

---

## 0. Status

| Date | Note |
|---|---|
| 2026-05-18 | Plan written. Pending kickoff. |
| 2026-05-18 | Design choices locked: HF `transformers` is the M1.8 accuracy baseline; decode global attention uses the embedding-stack length-based auto-switch; fused permuted attention promoted from §7.2 deferred into v1 scope as M1.10; Gemma 4 31B stretch dropped from scope (revisit if a 64 GB SEV-SNP SKU becomes available). |
| 2026-05-18 | M1.0 LLM-serving harness landed (commit `54d2c12`): `KvCache`, `causal_gqa_attention_cached`, `RopeTables::apply_at`, `forward::run_prefill` + `run_decode_step`, `generation::generate` (greedy). Decode-replay invariant test passes; full gelo-embedder + gelo-reranker suites still 100% green. |
| 2026-05-18 | M1.1 Gemma 4 scaffolding landed (commit `060f053`): `AttentionClass` enum, `attention_classes` / `partial_rope` / `kv_shared_in_global` on `DecoderConfig`, `Gemma4Variant::{E2B, E4B}` factories, `gemma4_attention_classes` builder with last-layer-always-Global override. 7 gemma4 + 5 config tests green. |
| 2026-05-18 | M1.2 PLE in TEE DRAM landed (commit `4d59d81`, **P0**): `gelo_protocol::PleTable` (int8, dequant on gather), `TrustedExecutor::provision_ple_table` + `ple_gather` extensions, default-impl fail-loud, `tests/ple_pcie_leak.rs` spy-engine confirms zero PLE-keyed activity reaches the offload engine across both `InProcessTrustedExecutor` and `PlaintextExecutor`. |
| 2026-05-18 | M1.3 hybrid attention dispatch landed (commit `43f4c7c`): `causal_gqa_attention_swa_cached(window, q_pos_offset)` band-mask kernel; `decoder_block_cached` consults `effective_attention_class(li)` to pick SWA vs dense-causal. Tight-window divergence + max-window-equals-all-global + decode-replay invariants green. |
| 2026-05-18 | M1.5 p-RoPE landed (commit `2655edd`): `RopeTables::apply_partial_at(rotated_dim)` rotates first `rotated_dim` (even-snap floor) of each head, identity-pass-through on the rest. `decoder_block_cached` routes Global+`partial_rope=Some` to partial rotation, Local-or-None to full rotation. Divergence test against full rotation green. |
| 2026-05-18 | M1.4 K=V tying landed (commit `66bba90`): `LayerKvCache` becomes a Separate/Shared enum; `KvCache::new_with_sharing(shared: &[bool])` halves global-layer cache memory; `decoder_block_cached` skips the V matmul when `kv_shared_in_global` + layer is Global. Parity (wk==wv → identical output) + memory (75% of all-separate at 4-layer 2:1) tests green. |
| 2026-05-18 | M1.10a + M1.6 + M1.8 scaffolding landed (commit `dc9074d`): `GpuOffloadEngine::fused_attention_batched` default-impl composition (no kernel yet); `gemma4_e2e.rs` + `gemma4_hf_parity.rs` `#[ignore]`-gated integration tests with documented un-ignore prerequisites. |
| 2026-05-18 | **Gemma 4 weight source + architecture audit.** Fetched real `google/gemma-4-E2B` and `-E4B` config.json from HuggingFace (public/non-gated; safetensors bf16 at ~4 GB / ~8 GB). Confirmed `google/gemma-4-*` is the canonical v1 source — GGUF detour dropped per user decision. The audit revealed several architectural inaccuracies in M1.0–M1.5: E2B intermediate_size was 8192 (real: 6144); E4B was 16384 (real: 10240); E4B num_key_value_heads was 1 (real: 2); max_position was 32768 (real: 131072); hidden_activation was `silu` (real: `gelu_pytorch_tanh`, i.e. GeGLU); M1.4 implemented within-layer K=V tying but real Gemma 4 has `attention_k_eq_v: false` and uses cross-layer KV sharing via `num_kv_shared_layers: 20` (E2B) / 18 (E4B); per-class head_dim differs (local 256 / global 512); per-class rope_theta differs (10_000 local / 1_000_000 global); final_logit_softcapping = 30.0 was missing. **Phase 1 fixes landed** (commit `<TBD>`): all numeric constants corrected, `DecoderConfig::final_logit_softcapping` added + wired through `compute_logits`, `Gemma4Variant` now exposes the Phase 1.5 metadata (global_head_dim, rope_theta_global, num_kv_shared_layers, ple_dim) for the follow-up workstream. **Real-weight inference still blocked on Phase 1.5** (~3-4 weeks): per-class head_dim refactor, per-class RoPE, cross-layer KV sharing, GeGLU dispatch, AltUp. See plan §8 (new) for Phase 1.5 milestone list. |
| 2026-05-18 | **v1 demonstrator pivot to Qwen3-1.7B.** All Phase 1.5 items are Gemma-only architecture work; pivoting v1 to a vanilla GQA decoder unblocks real-weight end-to-end generation today while Gemma support continues as a separate workstream (see [`../archive/handoffs/2026-05-18-gemma4-architecture-support.md`](../archive/handoffs/2026-05-18-gemma4-architecture-support.md)). **Investigated and rejected: Qwen3.5** (2B/4B/9B/35B-A3B) — `Qwen3_5ForConditionalGeneration` is a multimodal VLM with GDN-style `linear_attention` (Mamba/SSM) layers in 3:1 hybrid with `full_attention`, MRoPE, `partial_rotary_factor: 0.25`, MTP head, `attn_output_gate`. Strictly harder than Gemma 4, not easier. **Qwen3-1.7B** confirmed vanilla `Qwen3ForCausalLM` (28 layers, 16/8 GQA, head_dim 128, full RoPE θ=1M, SwiGLU, tied embeddings, no sliding window, no softcap) with one Qwen3-vs-Qwen2 addition: per-head RMSNorm on Q and K **before** RoPE (`self_attn.{q,k}_norm.weight`). Landed (commit `<TBD>`): `Qwen3Variant::{Q1_7B, Q4B}` config builder, `DecoderLayerWeights.{q_norm, k_norm}: Option<Array1<f32>>` (back-compat — `None` for Qwen2/LLaMA/Mistral), `rms_norm::apply_qk_norm` helper, both `decoder_block` and `decoder_block_cached` apply QK-norm when populated. Latent bug fixed: Qwen3-Embedding-0.6B (existing parity-test target) was silently dropping the same QK-norm step → embedder numbers will shift; parity property preserved (both branches change identically). New `tests/qwen3_generation_e2e.rs` runs greedy `generate()` against `Qwen/Qwen3-1.7B` bf16 safetensors under both `PlaintextExecutor` and `InProcessTrustedExecutor` and asserts identical token sequences. **PASS 2026-05-18** — emitted `" jumps over the lazy dog. 1"` (8 tokens, bit-identical on both branches; 377 s wall-clock on CPU). See plan §10. |

(Update this table at every weekly sync.)

---

## 1. Baseline state of the prototype

What already exists (verified against current `master`):

- **Protocol**: GELO mask + TwinShield shield rows + U-Verify
  (`crates/gelo-protocol/`). Per-batch fresh Haar-uniform A.
- **Engines**: `RayonCpuEngine` (sim), `WgpuVulkanEngine`
  (production GPU via wgpu+cubecl-matmul).
- **Trusted executor**: `InProcessTrustedExecutor` (sim) and
  `SnpTrustedExecutor` (SEV-SNP wrapper).
- **Model harness**: `gelo-embedder` runs Qwen3-Embedding-0.6B
  forward pass with 28 decoder layers, GQA attention, RoPE, SwiGLU
  FFN, RMSNorm. Last-token pool for embedding output.
- **Attention paths**: in-TEE attention (default at embedding
  shape), OutAttnMult (4-partition Q·K^T), permutation-shielded
  attention (Amulet-style, default off — regresses at embedding
  shape).
- **Decode/generation harness**: **does not exist yet**. The
  embedder is single-forward-pass. Per `gelo-llm.md`, the LLM
  serving harness is a 1-2 week pure-engineering item that gates
  all decode-phase work.

What needs to be added for Path 1 (in order):

1. LLM-serving harness (sampling loop, KV cache management)
2. Gemma 4 model loader (architecture-specific weight layout)
3. PLE-in-TEE-DRAM machinery (P0 leak fix from round 2)
4. Hybrid attention placement (in-TEE local + offload global)
5. K=V tensor handling in global layers
6. p-RoPE support
7. Benchmark + accuracy harness on E2B then E4B
8. Attack-resistance suite integration

---

## 2. Milestones

### M1.0 — LLM-serving harness (precondition)

**Scope:** `gelo-llm.md` §6 step 1. Load a Gemma decoder for
**generation** (not embedding pooling); implement greedy / top-p
sampling loop; manage growing KV cache across decode steps.

**Files to add/modify:**
- `crates/gelo-embedder/src/decoder/generation.rs` (new) —
  generation loop, sampler, KV-cache management
- `crates/gelo-embedder/src/decoder/forward.rs` — refactor to
  separate prefill vs decode dispatch
- `crates/gelo-embedder/src/decoder/kv_cache.rs` (new) — KV
  cache storage in CVM DRAM, indexed by `(layer_idx, head_idx)`

**Acceptance:**
- Plain (non-protected) generation works on a Qwen3-0.6B *decoder*
  model (existing weights, just not the embedding variant) — sanity
  check the harness.
- KV cache grows correctly across decode steps.
- Sampler produces the same output as `transformers` reference at
  `temperature=0, top_p=1`.

**Effort:** 1.5 weeks.

**Dependencies:** None.

**Risk:** Moderate. New code surface but well-understood pattern.

---

### M1.1 — Gemma 4 model loader

**Scope:** Read Gemma 4 safetensors, populate the existing
`DecoderWeights` structure with Gemma-specific layout. Handle:
- Layer count: 35 (E2B) or 42 (E4B)
- Per-layer attention-class metadata (local-512 vs global-8K)
- Hidden size: 1536 (E2B) or 2560 (E4B)
- KV head sharing (8-to-1 GQA)
- Embedding table + lm_head + PLE table

**Files to add/modify:**
- `crates/gelo-embedder/src/decoder/gemma4.rs` (new) — model
  variant trait impl
- `crates/gelo-embedder/src/decoder/config.rs` — add
  `AttentionClass::{Local(usize), Global}` enum, layer-wise
  vector of classes
- `crates/gelo-embedder/src/loader/safetensors.rs` — Gemma 4
  weight-key mapping

**Acceptance:**
- E2B and E4B safetensors load cleanly into memory.
- Per-layer attention class vector matches paper (E2B: 4:1
  pattern repeating; E4B: 5:1 pattern repeating; last layer
  always global).
- Sanity check: forward pass against a known prompt produces the
  same logits as HuggingFace transformers reference (no GELO
  involvement — just model loading parity).

**Effort:** 2 weeks.

**Dependencies:** M1.0.

**Risk:** Low. Mostly schema work.

---

### M1.2 — PLE table in TEE DRAM + gather kernel

**Scope:** Per round 2 §D.5 (P0). The PLE table
`[262144 × 256 × N_layers]` must live in the CVM's encrypted
memory; its gather operations must happen in-TEE, never on the
GPU. Without this, prompt token IDs leak via the memory access
pattern even under GELO masking.

**Construction:**
1. At model load, allocate the PLE table inside the
   `InProcessTrustedExecutor`'s state (not in the engine's GPU
   buffer).
2. Per-token gather: TEE selects rows `PLE[token_id, layer_idx, :]`
   into a (n, d_ple=256) tensor.
3. Project up to (n, d_hidden) via the per-layer PLE projection
   matrix (which IS a normal weight — public, can go to GPU).
4. Mask the projected vector with the current batch's `A` (or
   leave unmasked if it's added as a residual inside the TEE).

**Files to add/modify:**
- `crates/gelo-protocol/src/ple.rs` (new) — PLE table type, gather
- `crates/gelo-embedder/src/decoder/gemma4.rs` — wire PLE into
  per-layer forward
- `crates/gelo-protocol/src/substrate.rs` — extend
  `TrustedExecutor` trait with `provision_ple_table`,
  `ple_gather`

**Acceptance:**
- E4B int8 PLE table (~1.3 GB) loads into CVM encrypted memory.
- Per-forward gather cost measured; expected ~bandwidth-bound,
  no compute.
- **Verification test:** an attacker simulator that watches PCIe
  traffic between TEE and GPU sees NO PLE-table-side gathers.
  (Implement as `tests/ple_pcie_leak.rs` against the sim
  executor.)
- End-to-end forward parity with HF reference (Gemma 4 plain).

**Effort:** 2 weeks.

**Dependencies:** M1.1.

**Risk:** Moderate. New protocol surface and memory budget tight
on small CVM SKUs.

---

### M1.3 — Hybrid attention placement

**Scope:** Wire per-layer attention dispatch based on
`AttentionClass::{Local(W), Global}`:
- Local: in-TEE causal sliding-window attention with W=512.
  Cheap; round 2 §D.2 shows 4.57× speedup vs dense at n=8K.
- Global: same dispatch as current Qwen3 path — in-TEE for short,
  OutAttnMult or fused permuted (M1.10) for long. Length-based
  auto-switch (`gelo.md` §3.5) applies per-layer-class **and per-phase**:
  decode steps stay in-TEE for global attention at any
  realistic n_cache because the per-step attention math is
  microsecond-scale; the auto-switch threshold engages on
  prefill at long context, not on decode. Decode KV-cache
  *bandwidth* is the orthogonal axis — addressed by the
  SCX-class primitive in §7.1, not by attention dispatch.

**Files to add/modify:**
- `crates/gelo-embedder/src/decoder/attention.rs` —
  `causal_gqa_attention_local_window(window=W)` kernel
- `crates/gelo-embedder/src/decoder/forward.rs` — per-layer
  dispatch reading attention class from config
- `crates/gelo-protocol/src/attention.rs` — sliding-window mask
  support in in-TEE attention

**Acceptance:**
- Local-attention output matches HF reference at W=512, n=8K, 16K.
- Auto-switch correctly engages OutAttnMult on global layers past
  threshold; stays in-TEE on local layers regardless of n.
- Profiled wall-clock at E4B n=8K: local layers should aggregate
  <100 ms; global layers should aggregate <50 ms with OutAttnMult.

**Effort:** 3 weeks.

**Dependencies:** M1.1.

**Risk:** Moderate-high. Sliding-window kernel is new code in
multiple places (in-TEE math, mask handling, KV-cache slicing).

---

### M1.4 — K=V global-layer handling

**Scope:** Gemma 4 global layers store K and V as the same tensor.
GELO benefits from this:
- In-TEE attention: don't duplicate in memory.
- OutAttnMult: sample one mask, use for both K and V positions.
- Permuted attention: single π for K/V tensor.

**Files to add/modify:**
- `crates/gelo-protocol/src/out_attn_mult.rs` — `kv_shared: bool`
  flag, halve mask sampling when true
- `crates/gelo-protocol/src/attention.rs` — permuted-attention
  variant for K=V
- `crates/gelo-embedder/src/decoder/attention.rs` — call sites
  updated to pass `kv_shared` for global layers

**Acceptance:**
- Global-layer forward pass produces identical output to a
  separate-K-V reference (proves K=V doesn't change semantics).
- Mask GEMM count for global-layer attention drops measurably
  (specifically: OutAttnMult mask-sample count goes from 2 to 1
  per global layer).
- Memory: KV cache for global layers in CVM RAM is ~½ the
  separate-K/V baseline.

**Effort:** 1 week.

**Dependencies:** M1.3.

**Risk:** Low.

---

### M1.5 — p-RoPE support

**Scope:** Gemma 4 global layers use p-RoPE with p=0.25 — rotation
applied to the first p·d_head dims only, identity on the rest.

**Files to add/modify:**
- `crates/gelo-embedder/src/decoder/rope.rs` — add `partial: Option<f32>`
  param; when set, apply rotation to `floor(p·d_head)` dims only

**Acceptance:**
- p-RoPE output matches HF Gemma 4 reference.
- Existing Qwen3 / Llama RoPE paths unaffected (p=None defaults to
  full rotation).

**Effort:** 0.5 weeks.

**Dependencies:** M1.1.

**Risk:** Trivial.

---

### M1.6 — E2B end-to-end benchmark

**Scope:** Run the shared M0.1 corpus on E2B with three
configurations:
- Plain Gemma E2B (no GELO; baseline)
- GELO + in-TEE attention (default for short context)
- GELO + OutAttnMult on global layers (long context)

**Measurements:**
- TTFT @ 512-token prompt
- TPOT @ 256-token continuation
- Peak CVM RAM
- Throughput @ batch=1, batch=32

**Files to add:**
- `crates/gelo-embedder/benches/gemma_e2b_e2e.rs` (or shared
  `evals/run-eval.py` from M0.2)
- Results to `results/path-1-e2b.json`

**Acceptance:**
- E2B + GELO + in-TEE attention runs end-to-end without panic.
- TPOT overhead vs plain Gemma E2B is within the predicted
  10-30% range.
- Output text is coherent (sanity-check by inspection on 10
  prompts).

**Effort:** 1 week.

**Dependencies:** M1.0–M1.5.

**Risk:** Low.

---

### M1.7 — E4B scaling benchmark

**Scope:** Same bench as M1.6 on E4B. Compare scaling: E2B → E4B
overhead delta. Document where GELO scales linearly, where it
scales worse than linear (Householder sample at d²).

**Files to add:**
- `results/path-1-e4b.json`
- `docs/plans/path-1-status.md` (update with scaling table)

**Acceptance:**
- E4B + GELO runs end-to-end.
- Overhead delta from E2B → E4B is measured and documented.
- If overhead degrades dramatically (>2× the E2B overhead), flag
  R2 from comparison framework and apply HKDF-derived mask
  optimization if needed.

**Effort:** 0.5 weeks.

**Dependencies:** M1.6.

**Risk:** Low (engineering); moderate (if scaling regresses).

---

### M1.8 — Accuracy validation

**Scope:** Run M0.2 eval harness against GELO E2B and E4B.
- MMLU 0-shot (Tier 2: 500 prompts)
- IFEval pass-rate (500 prompts)
- PIQA accuracy (200 prompts)
- HumanEval pass@1 (200 prompts)
- Top-1 token match vs HuggingFace
  `transformers.AutoModelForCausalLM` reference at
  `temperature=0` (greedy)
- Final hidden-state cosine similarity vs HF reference

**Reference baseline.** HuggingFace `transformers` is the
canonical Gemma 4 implementation — what the model cards target
and what the open-source community treats as ground truth. Pin
the reference build by `transformers` package version SHA-256
and record both versions in `results/path-1-accuracy.json` so
re-runs are reproducible. llama.cpp and vLLM are explicitly
*not* the baseline (they're production runtimes with their own
quantisation and sampler quirks); their numbers could be added
to the report as informational rows but the accept gate is
HF-transformers parity only.

**Acceptance:**
- Top-1 token match ≥ 0.99 (GELO should be ~bit-exact in fp32).
- Accuracy delta vs plain on each benchmark within ±0.5pp.
- Hidden-state cosine similarity ≥ 0.999.

**Effort:** 1 week.

**Dependencies:** M1.6 + M0.2.

**Risk:** Low. GELO has not been shown to degrade accuracy on any
tested config (`gelo.md` Appendix).

---

### M1.9 — Attack-resistance integration (M0.3 wiring)

**Scope:** Capture PCIe-side activations from
`InProcessTrustedExecutor` and run the AloePri attack suite
(VMA / IA / ISA / IMA / TFMA / SDA) against GELO-protected
Qwen3-1.7B (v1 demonstrator) and, once Phase 1.5 lands, Gemma E2B/E4B.

**Phase 1 — Rust-side snapshot capture (✅ done 2026-05-18).** Landed
the entire `gelo_protocol::snapshot` module:
- `PcieSnapshot { seq_idx, layer, kind, masked_operand,
  masked_output }` records every PCIe-crossing matmul.
- `SnapshotCapture` aggregator + `InProcessTrustedExecutor`
  builder/accessor surface (`with_snapshot_capture`,
  `enable/disable_snapshot_capture`, `pcie_snapshots`,
  `drain_pcie_snapshots`); capture is `None` by default so the
  production embedder / reranker path pays zero overhead.
- Hook sites: `offload_linear`, `offload_qkv`, `offload_linear_many`.
- 11 tests (5 unit + 6 integration in
  `crates/gelo-protocol/tests/snapshot_capture.rs`).

**Phase 2 — Python attack harness (⏳ pending).** Handoff in
[`../dev/prototype/aloepri-attack-harness.md`](../dev/prototype/aloepri-attack-harness.md):
safetensors serialisation contract, AloePri commit pin, three-condition
control matrix (plain / mask-only / mask+shield), per-attack drivers
at `evals/aloepri-attacks/run_{vma,ima,isa,tfma,sda,ia}.py`.

**Phase 3 — CI release-gate (⏳ pending Phase 2).** Fast-variant
runner under 5 minutes; threshold: fail if IMA or ISA TTRSR ≥ 10%.

**Files to add (Phases 2-3):**
- `crates/gelo-embedder/src/attack_export.rs` — safetensors
  serialiser for captured snapshots (kept in embedder, not protocol)
- `evals/aloepri-attacks/{conftest.py, snapshots_loader.py, run_*.py}`
- `.github/workflows/aloepri-gate.yml` — release-gate CI

**Acceptance (Phase 2):**
- C2 (default GELO config) reports IMA + ISA TTRSR < 10% on Qwen3-1.7B.
- C0 (plain) reports IMA + ISA + VMA TTRSR ≥ 95% (sanity: attacks
  themselves work).
- Gap between C1 (mask only) and C2 (mask + shield) is measurable
  (shield rows demonstrably add defence).
- Results JSON committed to `results/path-1-attacks.json`.

**Effort:** Phase 1 done (~6 hours actual vs ~1 week estimated).
Phase 2 estimated 1 week, Phase 3 estimated 0.5 week (was 2 weeks
combined in the original plan).

**Dependencies:** M1.6 (Qwen3-1.7B real-weight smoke test ✅).
Gemma E2B/E4B coverage gated on Phase 1.5 + M1.6 Gemma re-pin.

**Risk:** Moderate on Phase 2 — if C2 IMA/ISA TTRSR exceeds 10%,
investigate against shield-row config (`gelo.md` §3.3) and the
per-forward-vs-per-offload mask trade-off (`gelo.md` §3.2).

---

### M1.10 — Fused permuted attention for long-context prefill

> **Detailed plan:** [`m1-10-fused-permuted-attention.md`](m1-10-fused-permuted-attention.md)
> (2026-05-18). Surfaces a pre-implementation
> security gap (causal-mask leak in the existing `permuted_attention`)
> that adds Phase 0 (~1 week) before kernel work begins.
> Refined effort: ~5 weeks (Option A); ~7 weeks (Option C forced).
> Refined acceptance: gpu_gelo_fused TTFT ≤ 20 s at n=2048 on
> Qwen3-1.7B / RADV iGPU; attention-only ≤ 1 s isolated.

**Scope:** Close the upstream `burn-cubecl` gap (hardcoded
`causal: true` in `burn_cubecl::kernel::attention::flash_attention`)
and wire a fused FlashAttention-style permuted-attention kernel
into the engine. Promotes the §7.2 deferred item into v1 scope
because long-context (n ≥ 1024) prefill global-layer attention
is bandwidth-bound on the 3-dispatch path's materialised
`(heads, n, n)` score tensor (~3.2 GB/layer at n=4k; ~51 GB/layer
at n=16k), making the 3-dispatch fallback unusable for any
non-trivial RAG context. Fused path drops per-layer traffic to
`O(n·d_total)` (~130 MB/layer at n=4k) — lands long-context
prefill within ~2× of unprotected baseline **for the attention
slice** (the GELO mask round-trip on linear projections is a
separate cost — see plan §10).

**Approach options** (decision deferred to time-of-implementation
based on cubek/burn maturity at that moment):

- **Option A** — Fork the burn-cubecl wrapper into
  `gelo-gpu-wgpu` and call `cubek::attention::launch::launch_ref`
  directly with `causal: false` plus our permuted causal mask
  as the sole mask in the `Materialized` slot. ~150 LOC. Gated
  on `cubek-attention` v0.1.1 API stability (currently young,
  likely API-unstable).
- **Option B** — Upstream PR to parameterize
  `burn_cubecl::flash_attention(causal: bool)`. Lowest
  maintenance long-term; blocks on tracel-ai merge cycle.
- **Option C** — Custom WGSL fused-attention kernel
  (~500 LOC, FlashAttention-style with FLASH-D online softmax).
  Highest implementation risk; lowest dependency surface; only
  pursued if A and B are both unworkable at start of M1.10.

**Files to add/modify:**
- `crates/gelo-gpu-wgpu/src/lib.rs` — `fused_attention_batched`
  engine method (override of the default 3-dispatch composition)
- `crates/gelo-protocol/src/attention.rs` —
  `permuted_attention` checks for engine capability via the
  `TrustedEngine` trait, prefers fused when available, falls
  back to composed 3-dispatch otherwise
- `crates/gelo-protocol/src/substrate.rs` — engine trait
  extension if needed
- (Option A) `Cargo.toml` adds `cubek` + `cubek-attention`
  direct deps
- (Option C) `crates/gelo-gpu-wgpu/src/kernels/flash.wgsl`

**Acceptance:**
- Long-context prefill global-layer wall-clock drops from
  ~500 ms (3-dispatch at n=4k) to ~150-200 ms (fused), per
  `gelo-llm.md` §3.7 projection. Within ~2× of unprotected
  baseline.
- Parity test vs 3-dispatch path on permuted causal mask at
  n ∈ {256, 1024, 4096}: outputs agree within 1e-4.
- Autoswitch in `decoder::forward` engages fused path on
  global layers past the auto-switch threshold; falls back to
  3-dispatch when engine reports no fused capability.

**Effort:** 3 weeks (Option A baseline) · +2 weeks if Option C
needed.

**Dependencies:** M1.3 (hybrid attention placement defines
where global-layer attention happens; fused kernel slots into
the global-attention dispatch).

**Risk:** Moderate. `cubek-attention` v0.1.1 API may not be
stable enough for Option A — fallback chain documented above.
Option B is unbounded on upstream merge cycle and cannot be
relied on for v1.

---

## 3. Aggregate effort

| Milestone | Effort (weeks) | Cumulative |
|---|---:|---:|
| M1.0 | 1.5 | 1.5 |
| M1.1 | 2.0 | 3.5 |
| M1.2 | 2.0 | 5.5 |
| M1.5 (interleaved) | 0.5 | 6.0 |
| M1.3 | 3.0 | 9.0 |
| M1.4 | 1.0 | 10.0 |
| M1.10 (fused permuted) | 3.0 | 13.0 |
| M1.6 | 1.0 | 14.0 |
| M1.7 | 0.5 | 14.5 |
| M1.8 | 1.0 | 15.5 |
| M1.9 | 2.0 (after M0.3) | 17.5 |

**Total: ~15.5 weeks v1 (E2B + E4B + fused permuted prefill).
~17.5 weeks including the attack-resistance integration.**

The 31B stretch (previously M1.10) is dropped from v1 scope per
2026-05-18 design decision — revisit only if a 64 GB SEV-SNP
SKU becomes available.

Plus shared work:
- M0.1 + M0.2 inline with M1.0–M1.2 (~1.5 weeks of dual effort)
- M0.3 inline with M1.9 (~3 weeks shared with AloePri)
- M0.4 after both paths: ~1 week

---

## 4. Critical path

```
M1.0 → M1.1 → M1.2 → M1.3 → M1.10 → M1.6 → M1.7 → M1.8 → M0.4
                       ↓ ↘
                     M1.4  M1.5      (off critical path)
              M0.3 ───────────────→ M1.9 ─────────────────→ M0.4
```

Longest chain: M1.0 + M1.1 + M1.2 + M1.3 + M1.10 + M1.6 + M1.7 +
M1.8 + M0.4 = 15.5 weeks.

M1.4 (K=V handling) and M1.5 (p-RoPE) are small enough
(1 week / 0.5 weeks) to slot in parallel with M1.10's fused
permuted attention work — same author or split if a worker
joins.

---

## 5. Disjoint-directory contract with AloePri

To minimize merge pain between worktrees, Path 1 only writes to:

- `crates/gelo-embedder/**`
- `crates/gelo-protocol/**`
- `crates/gelo-gpu-wgpu/**`
- `crates/gelo-snp-runner/**`
- `evals/private-inference-corpus/**` (M0.1, shared, written by Path 1)
- `evals/run-eval.py` + `evals/lib/**` (M0.2, shared, written by Path 1)
- `evals/attack-harness/**` (M0.3, shared, written by Path 1)
- `docs/plans/path-1-*.md`
- `results/path-1-*.json`

AloePri only writes to:

- `vendor/aloepri-py/**` (vendored Python)
- `scripts/path-2-*.py`
- `docs/plans/path-2-*.md`
- `results/path-2-*.json`

If Path 1 needs changes to AloePri-owned files (e.g., to read
AloePri snapshots for attack harness), file a PR back to master.

---

## 6. Open questions / decisions deferred

- **Sampler choice**: greedy for v1 acceptance gates (necessary
  for deterministic HF-transformers parity at temperature=0).
  Top-p / top-k / temperature support lands alongside M1.6 once
  the harness exists; not on the M1.8 accept gate.
- **PLE table fp16 vs int8**: fp16 is 2× memory but bit-exact;
  int8 saves ~700 MB at small quality loss. Default to int8;
  M1.8 accuracy validation flips the default to fp16 if the
  int8 quantisation moves any benchmark by more than 0.5pp.
- **Fused permuted attention option choice (M1.10)**: A vs B vs
  C decided at start of M1.10 based on cubek-attention API
  stability and burn-cubecl upstream state at that moment.

**Resolved 2026-05-18:**

- ~~MatFormer slice handling~~ — two separate loaders. The 4:1
  vs 5:1 attention ratio difference between E2B and E4B makes
  a shared blob more painful than it's worth.
- ~~Stretch 31B in 32 GB CVM~~ — dropped from scope.
- ~~Reference baseline for accuracy~~ — HF `transformers` at
  `temperature=0` (greedy). See M1.8.
- ~~Decode global attention dispatch~~ — length-based
  auto-switch from `gelo.md` §3.5 applies per-phase; decode
  stays in-TEE for global attention at realistic n_cache. KV-cache
  bandwidth (the orthogonal axis) addressed by §7.1 SCX.

---

## 7. Post-v1 future work

Items intentionally out of scope for v1 (M1.0–M1.10) but
expected to land as follow-ups once the bench harness exists
and the decode-phase cost breakdown is measured.

### 7.1 SCX-style KV-cache encoding for decode

**Reference:** Yuan et al., "SCX: Stateless KV-Cache Encoding
for Cloud-Scale Confidential Transformer Serving," SIGCOMM
2025. Code: `yuanmu97/scx`. Discussed in
[`../dev/prototype/gelo-llm.md`](../dev/prototype/gelo-llm.md) §4.3.

**Problem it solves.** Decode-phase π under our protocol is
structurally awkward: fresh π per step is incompatible with KV
cache written under previous steps' π. Naive fixes (carry π
forward across the whole generation session, or re-permute the
cache every step) trade either security (one π reused across
N decode steps for one session) or wall-clock (~12 GB cache
rewrite per token on Qwen3-class, multi-GB on Gemma E4B).

**SCX's approach.** Stateless per-position encoding: at write
time, K and V are encoded with a key derived from
`(session_id, layer_id, position)` — not from a per-step mask.
Each cache entry stays in its own frame forever. Decode-step
attention reads encoded K, V directly without per-step
re-permutation; per-token overhead is one fresh per-position
encoding-key derivation. Claimed ~36 ms LLaMA-7B decode
latency in their threat model.

**Gating items before adoption.**

1. **Threat-model alignment.** SCX's setting is generic
   "confidential transformer serving"; ours is SEV-SNP CVM +
   commodity GPU under the openweight assumption. Need to
   verify the position-key derivation survives a TEE-co-located
   GPU adversary, not just a curious cloud operator.
2. **Composition with KV-Cloak / Shadow-in-the-Cache.** Wu et
   al., arXiv 2508.09442 (Aug 2025) — KV-cache inversion
   attacks + per-block-permutation defense. Need a security
   analysis showing SCX either survives the Shadow-in-the-Cache
   adversary directly, or composes cleanly with KV-Cloak.
3. **Empirical port.** SCX reference code is Python; our stack
   is Rust + wgpu. Estimate ~2 weeks port + ~1-2 weeks security
   review.

**Landing condition.** M1.6 / M1.7 benches confirm decode-step
mask-sample or cache-handling cost is the dominant per-token
overhead at E2B / E4B. If decode-step cost is dominated by
linear projections instead, SCX is a nice-to-have rather than
on the critical path.

**Effort estimate (post-v1).** ~4 weeks: 2 weeks port +
1-2 weeks security analysis + accuracy + bench validation.

### 7.2 Other deferred items

Briefly, for completeness — full discussion in
[`../dev/prototype/gelo-llm.md`](../dev/prototype/gelo-llm.md) §08:

- **HKDF-derived mask material for amortised decode-step QR.**
  Lever; lands if M1.7 shows mask-sample > ~10% of TPOT on E4B.
- **Speculative decoding under the protocol.** Completely
  unexplored security-wise.
- **MoE generation.** Routing-histogram leak is a separate
  protocol surface (CryptoMoE balanced dispatch). Out of scope
  for the dense+hybrid Gemma 4 family.
- **Token-DP / score-DP accountant.** Only relevant if a
  deployment needs to export per-token probabilities for
  downstream calibration.
- **ECDH-bound session-key handshake.** Drop-in replacement for
  the current per-request `session_secret`; API surface
  unchanged.
- **Gemma 4 31B dense.** Dropped from v1 scope; revisit only if
  a 64 GB SEV-SNP SKU becomes available. Architecture is
  5:1 hybrid, W=1024, no PLE, ~5120 hidden — same protocol
  primitives apply but the memory budget rules it out on
  32 GB CVMs.

---

## 8. Phase 1.5 — Real Gemma 4 architecture extensions

Surfaced by the 2026-05-18 config audit. None of these were anticipated
in the original M1.0–M1.5 plan; they are gating items between
"correct synthetic test surface" and "matches HF reference on real
Gemma 4 weights". Roughly ordered by dependency / effort.

### 8.1 — GeGLU activation dispatch

**Scope.** Real Gemma 4 uses `gelu_pytorch_tanh` (approximate GELU
via tanh, the same activation Gemma 1/2/3 use). Our `decoder::swiglu`
module is SiLU-only.

**Files.** `decoder/swiglu.rs` (rename → `glu.rs` or keep, add a
GeGLU branch keyed on `cfg.hidden_act`). `decoder/forward.rs`
unchanged dispatch shape.

**Effort.** ~1-2 days. Pure activation function swap on a known
shape. Test: synthetic-decoder divergence between `silu` and
`gelu_pytorch_tanh` for the same weights.

### 8.2 — Per-class head_dim (local 256 / global 512)

**Scope.** Real Gemma 4 global layers have `global_head_dim: 512`,
twice the local-layer `head_dim: 256`. This changes the Q/K/V
projection matrix shape per layer class: local layer `wq` is
`(hidden, n_q_heads · 256)`, global layer `wq` is `(hidden, n_q_heads
· 512)`. The cache shape, RoPE table shape, and attention kernel
input shape all become per-layer-class.

**Files.** `decoder/config.rs` — extend `DecoderConfig::head_dim` to
per-class. `decoder/weights.rs` — `DecoderLayerWeights` likely needs
a per-layer-class head_dim field (or we trust the safetensors
loader to populate the right shape per layer based on
`attention_classes[li]`). `decoder/forward.rs` + `attention.rs` —
all `cfg.head_dim_value()` call sites become `cfg.head_dim_for(li)`.
`decoder/rope.rs` — two `RopeTables` (one per head_dim) or a single
table sized for `max(local, global)` with per-call slicing.

**Effort.** ~1-2 weeks. Largest single Phase 1.5 item. Touches every
layer-shape assumption in the code.

**Risk.** Moderate. The KvCache and RoPE refactors are
straightforward; the attention kernels may need per-class
codepaths if head_dim varies the inner-loop unroll.

### 8.3 — Per-class rope_theta (local 10K / global 1M)

**Scope.** Real Gemma 4 global layers use `rope_theta = 1_000_000`
(rope_type `proportional`) vs local layers' `rope_theta = 10_000`
(rope_type `default`). Two precomputed cos/sin tables per model.

**Files.** `decoder/rope.rs` — host both tables in a wrapper, expose
`RopeTables::for_layer_class(class)` accessor. `decoder/config.rs` —
extend with `rope_theta_local` + `rope_theta_global` fields.
`decoder_block_cached` selects the right table at the `apply_at` call.

**Effort.** ~2-3 days. Independent of §8.2 head_dim refactor;
either order works.

### 8.4 — Cross-layer KV sharing (`num_kv_shared_layers`)

**Scope.** 20 of 35 E2B layers (18 of 42 E4B) REUSE an earlier
layer's K and V cache instead of computing their own K/V
projections. This is structurally different from the within-layer
K=V tying that M1.4 implemented (and that real Gemma 4 has off).

**The actual rule** (per HF transformers Gemma4 implementation):
shared layers reference the most recent SAME-class layer's K, V.
A sliding-attention layer at index `i` that's marked "shared"
reads K, V from the most recent prior sliding-attention layer; same
for full-attention. The model is trained with this constraint
baked in — there are simply no K, V projection weights for the
shared layers' QKV slot.

**Files.** `decoder/kv_cache.rs` — `LayerKvCache` gains a `Reference`
variant pointing to a producer-layer index. `KvCache::append(li, …)`
becomes a no-op for shared layers; `view(li)` resolves to the
producer's view. `decoder/forward.rs::decoder_block_cached` — when
the layer is shared, skip the K/V matmul entirely (the weights
literally don't exist in safetensors) and read the previous
producer's K, V from the cache.

**Memory savings.** Halves the global-layer KV cache footprint AND
saves the K/V projection matmul per shared layer. Compounds with
the GQA savings already in the protocol.

**Effort.** ~1-2 weeks. The producer-layer resolution at construction
time is the conceptually tricky part; the storage refactor is similar
to the M1.4 Separate/Shared enum.

**Honest note about M1.4.** The within-layer K=V optimisation in
M1.4 is forward-compatible with Gemma 3 variants that set
`attention_k_eq_v: true`, but does not engage on real Gemma 4. Kept
in the code as dormant infrastructure; tests still validate it.

### 8.5 — `use_double_wide_mlp` semantics

**Scope.** Real Gemma 4 sets `use_double_wide_mlp: true`. Exact
semantics need HF transformers source code verification — likely
either the intermediate_size is reported as half the actual gate/up
width, OR a different MLP variant is used.

**Files.** `decoder/swiglu.rs` once GeGLU lands (§8.1). May not
require any code change if intermediate_size is already accounted
for in the doubled count; otherwise extend the FFN shape.

**Effort.** ~1-2 days (mostly research).

### 8.6 — AltUp (alternating updates) residual stream

**Scope.** Gemma 3n architecture introduced "AltUp" — an alternating
residual stream variant where intermediate hidden states from a
subset of layers participate in the residual differently. Likely
inherited by Gemma 4.

**Files.** Probably extends `decoder_block_cached` with a residual-
write predicate based on layer index; may need additional weight
tensors (`altup_correction_*`).

**Effort.** ~1 week. Specifics need HF transformers source check
before estimate is firm.

### 8.7 — Gemma 4 safetensors loader extensions

**Scope.** Once §8.1–§8.6 land, the loader extension is small:

- Detect Gemma 4 variant from `model_type: "gemma4"` in config.json
  AND `text_config.*` for nested architecture fields
- Map per-layer `attention_classes` from `layer_types: ["sliding_attention", …, "full_attention", …]`
- Extract `model.embed_tokens_per_layer.weight` → `PleTable`
- Extract per-layer PLE input projections (key name TBD —
  `model.layers.N.input_per_layer_projection.weight` or similar)
- Skip K/V projection tensors for layers marked as shared per §8.4
- Apply the correct head_dim per layer for shape validation per §8.2

**Effort.** ~1 week.

### Phase 1.5 critical path

```
§8.1 GeGLU (1-2d) ─┐
                   ├─→ §8.7 loader (1w) → flip M1.6/M1.8 #[ignore]
§8.2 head_dim (2w) ┤
§8.3 rope (3d)     ┤
§8.4 KV share (2w) ┘
§8.5/§8.6 (independent, post-flip refinement)
```

**Realistic total: 4-5 weeks** from the start of Phase 1.5 to the
moment a successful greedy `generate()` runs against
`google/gemma-4-E2B` real weights.

---

## 10. v1 demonstrator — Qwen3-1.7B (current target)

Status: in progress 2026-05-18.

### 10.1 Why Qwen3-1.7B replaces Gemma 4 as the v1 target

Phase 1.5 ([§8](#8-phase-15--real-gemma-4-architecture-extensions))
estimates 4-5 weeks of Gemma-only architecture work before any real
Gemma 4 weights can run forward (per-class head_dim, per-class
rope_theta, cross-layer KV sharing, GeGLU dispatch, AltUp,
`use_double_wide_mlp`). None of that work is reusable on a different
model family — it's a giant up-front bet on Gemma specifically.

**Qwen3-1.7B** is a vanilla `Qwen3ForCausalLM`: 28 layers, 16/8 GQA,
head_dim 128, full RoPE θ=1M, SwiGLU, tied embeddings, no sliding
window, no softcap, no PLE, no AltUp. The single Qwen3-vs-Qwen2
addition is per-head RMSNorm on Q and K **before** RoPE
(`self_attn.{q,k}_norm.weight`, shape `(head_dim,)`). That fits under
our existing `decoder_block` / RoPE plumbing as one new optional step
between QKV projection and RoPE — no structural refactor.

**Qwen3.5 was investigated and rejected** in the same session. The
family ships as `Qwen3_5ForConditionalGeneration` — a multimodal VLM
with a 24-layer ViT vision encoder, image / video tokens, and a text
backbone using GDN-style `linear_attention` (Mamba/SSM) layers in a
3:1 hybrid with `full_attention`, MRoPE (`mrope_section` interleaved),
`partial_rotary_factor: 0.25`, MTP head, and `attn_output_gate`.
Strictly harder than Gemma 4, not easier — the "Qwen3.5 is simpler"
premise we briefly explored is empirically false.

### 10.2 Scope landed for v1 demonstrator

| Item | Where |
|---|---|
| `Qwen3Variant::{Q1_7B, Q4B}` config builder | `crates/gelo-embedder/src/decoder/qwen3.rs` |
| `DecoderLayerWeights.q_norm/k_norm: Option<Array1<f32>>` | `crates/gelo-embedder/src/decoder/weights.rs` |
| `DecoderWeights::from_safetensors` reads `q_norm` / `k_norm` when present | `weights.rs` |
| Per-head `apply_qk_norm` helper | `crates/gelo-embedder/src/decoder/rms_norm.rs` |
| QK-norm wired into both `decoder_block` and `decoder_block_cached` | `crates/gelo-embedder/src/decoder/forward.rs` |
| `HfTokenizer::decode` | `crates/gelo-embedder/src/common/tokenizer.rs` |
| `tests/qwen3_generation_e2e.rs` — greedy `generate()` on `Qwen/Qwen3-1.7B`, asserts plaintext-vs-masked token-sequence parity | `crates/gelo-embedder/tests/` |
| 5 sibling tests adjusted for new `DecoderLayerWeights` fields | `tests/{generation_harness,decoder_parity,bundle_round_trip,comparative_bench,causal_discriminator_parity}.rs` |

### 10.3 Latent bug fixed by the QK-norm wiring

`Qwen/Qwen3-Embedding-0.6B` (the existing `tests/qwen3_e2e.rs`
target) has 56 `q_norm` + `k_norm` tensors that our loader was
silently dropping. Pre-fix: both `PlaintextExecutor` and
`InProcessTrustedExecutor` produced **wrong but identical**
embeddings — the parity test compared two equally-buggy values.
Post-fix: both branches now compute correctly. **Absolute embedding
values will shift**; the parity property (both executors agree
within 1e-2) holds because the fix is symmetric on both branches.
Any downstream comparison against a fixed baseline (FastEmbed,
BEIR scores) needs re-baselining.

### 10.4 Acceptance gate for v1 demonstrator

`tests/qwen3_generation_e2e.rs::qwen3_1_7b_greedy_generates_under_both_executors`
gates the v1 cut:

1. Greedy `generate(max_tokens = 8)` produces 8 tokens on prompt
   "The quick brown fox" under both executors.
2. The two token sequences are **bit-identical** — argmax-stable
   under the float drift the Haar mask introduces.

HF-transformers parity (the M1.8-equivalent fixture comparison) is
the next step after this lands.

### 10.5 Path back to Gemma 4

This v1 demonstrator is **additive** — it adds a new vanilla-GQA
target without touching `Gemma4Variant`, the hybrid attention
dispatch in `decoder_block_cached`, PLE infrastructure, p-RoPE
support, or the cross-layer KV sharing scaffolding. All Phase 1.5
items in §8 stay valid; the
[`gemma4-architecture-roadmap.md`](../archive/handoffs/2026-05-18-gemma4-architecture-support.md)
handoff doc remains the entry point for that workstream. Once
Phase 1.5 lands, `Gemma4Variant::E2B` becomes a drop-in next-target
in this same plan structure.

---

## 9. References

- [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md)
  (shared)
- [`aloepri-gemma.md`](aloepri-gemma.md) (sibling)
- [`../dev/prototype/gelo.md`](../dev/prototype/gelo.md) — protocol baseline
- [`../dev/prototype/gelo-llm.md`](../dev/prototype/gelo-llm.md) — LLM
  generation forward plan
- [`../archive/research/private-llm-inference-R2-2026-05-18.md`](../archive/research/private-llm-inference-R2-2026-05-18.md)
  §D — Gemma 4 architecture analysis
- [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
  — technique comparison
