# Path 1 — GELO TEE-GPU Split Inference for Gemma E2B/E4B

> **Worktree:** original (this one). Branch: `path-1-gelo-gemma`
> (to be created from `master`).
>
> **Sibling plan:** [`path-2-aloepri-gemma.md`](path-2-aloepri-gemma.md)
> develops in `../private-rag-path-2`.
>
> **Shared framework:** [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md).
>
> **Goal:** Extend the existing GELO+TwinShield prototype (currently
> validated on Qwen3-Embedding-0.6B) to support **autoregressive text
> generation on Gemma E2B/E4B** with the openweight threat model
> preserved. Produce performance, accuracy, and attack-resistance
> numbers comparable to Path 2 on the same models.

---

## 0. Status

| Date | Note |
|---|---|
| 2026-05-18 | Plan written. Pending kickoff. |

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
  OutAttnMult or permuted for long. Length-based auto-switch
  (`gelo.md` §3.5) applies per-layer-class.

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
- Top-1 token match vs plain reference
- Final hidden-state cosine similarity vs plain

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

**Scope:** Wire the M0.3 attack harness to capture activations
from `InProcessTrustedExecutor` at the PCIe boundary and run
VMA / IA / ISA / IMA / NN / TFMA / SDA against GELO-protected
Gemma E2B and E4B.

**Files to add:**
- `crates/gelo-embedder/src/instrumentation.rs` — feature-flagged
  snapshot capture
- `evals/attack-harness/run-against-gelo.py` — calls the harness
  with GELO snapshots

**Acceptance:**
- TTRSR < 5% under each of VMA, IA, ISA, IMA, NN, TFMA, SDA on
  E2B and E4B.
- Documented in `results/path-1-attacks.json`.

**Effort:** 2 weeks (after M0.3 lands).

**Dependencies:** M0.3, M1.6, M1.7.

**Risk:** Moderate. If any attack exceeds 5% TTRSR, that's
unexpected and warrants investigation against shield-row config
(`gelo.md` §3.3).

---

### M1.10 — (Stretch) Gemma 4 31B dense

**Scope:** Run M1.6–M1.9 on Gemma 4 31B (5:1 hybrid, W=1024, no
PLE, dense, ~5120 hidden).

**Acceptance:**
- 31B fits in CVM RAM with current Arc-share refactor
  (`gelo.md` §5.2). Budget: ~31 GB weights + ~5 GB working set,
  fits in 64 GB SKU.
- Per-batch mask sample cost remains <50 ms (at d=5120, the
  Householder QR is ~30 ms on Genoa per round-2 estimates).
- Output coherent; accuracy delta within ±1pp.

**Effort:** 1 week.

**Dependencies:** M1.8.

**Risk:** Memory budget; mask sample cost; CPU bottleneck on
in-TEE attention for the global layers (1/6 × 42 layers at
n=128k context = 7 layers × 128k² attention compute is a lot).

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
| M1.6 | 1.0 | 11.0 |
| M1.7 | 0.5 | 11.5 |
| M1.8 | 1.0 | 12.5 |
| M1.9 | 2.0 (after M0.3) | 14.5 |
| M1.10 stretch | +1.0 | 15.5 |

**Total: ~12.5 weeks v1 (E2B + E4B), ~15.5 weeks with 31B stretch.**

Plus shared work:
- M0.1 + M0.2 inline with M1.0–M1.2 (~1.5 weeks of dual effort)
- M0.3 inline with M1.9 (~3 weeks shared with Path 2)
- M0.4 after both paths: ~1 week

---

## 4. Critical path

```
M1.0 → M1.1 → M1.2 → M1.3 → M1.4 ┐
                ↓                  ├→ M1.6 → M1.7 → M1.8 → M0.4
              M1.5 ──────────────┘
              M0.3 ───────────────→ M1.9 ─────────────────→ M0.4
```

Longest chain: M1.0 + M1.1 + M1.2 + M1.3 + M1.4 + M1.6 + M1.7 +
M1.8 + M0.4 = 12.5 weeks.

---

## 5. Disjoint-directory contract with Path 2

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

Path 2 only writes to:

- `vendor/aloepri-py/**` (vendored Python)
- `scripts/path-2-*.py`
- `docs/plans/path-2-*.md`
- `results/path-2-*.json`

If Path 1 needs changes to Path-2-owned files (e.g., to read
AloePri snapshots for attack harness), file a PR back to master.

---

## 6. Open questions / decisions deferred

- **MatFormer slice handling**: should we load E4B weights once
  and use the E2B slice when needed, or maintain two separate
  loaders? Likely the latter due to the 4:1 vs 5:1 attention
  pattern difference.
- **Sampler choice**: greedy only for v1; top-p / top-k as
  M1.10 prerequisite.
- **Stretch 31B in 32 GB CVM**: only attempt if we have a 64 GB
  CVM available. Don't fight the memory budget on the small SKU.
- **PLE table fp16 vs int8**: fp16 is 2× memory but bit-exact;
  int8 saves ~700 MB at small quality loss. Default to fp16 unless
  memory budget forces int8.

---

## 7. References

- [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md)
  (shared)
- [`path-2-aloepri-gemma.md`](path-2-aloepri-gemma.md) (sibling)
- [`../prototype/gelo.md`](../prototype/gelo.md) — protocol baseline
- [`../prototype/gelo-llm.md`](../prototype/gelo-llm.md) — LLM
  generation forward plan
- [`../research/private-llm-inference-round-2.md`](../research/private-llm-inference-round-2.md)
  §D — Gemma 4 architecture analysis
- [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
  — technique comparison
