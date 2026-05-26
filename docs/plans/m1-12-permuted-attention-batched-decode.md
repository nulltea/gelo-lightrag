---
type: plan
status: stale
created: 2026-05-22
updated: 2026-05-22
tags: [m1.12, attention]
archive_reason: "R1.4 Phase A aborted; failed gate by 16x on iGPU. Bucket 2 deferred indefinitely."
---

# M1.12 — Permuted-attention batched-decode kernel (perf-bucket 2)

> **Parent context:**
> - Handoff: [`2026-05-22-perf-bucket-roadmap-r3-default.md`](../handoffs/2026-05-22-perf-bucket-roadmap-r3-default.md) — perf-bucket roadmap; this plan is bucket 2.
> - Plan: [`m1-12-tee-gpu-throughput.md`](m1-12-tee-gpu-throughput.md) — M1.12 spec; bucket 2 was scoped but deferred to its own plan.
> - Plan: [`m1-11-batched-decode.md`](m1-11-batched-decode.md) — the batched substrate this rides on (D1.8 stopgap that this lever replaces).
> - Plan: [`m1-10-fused-permuted-attention.md`](m1-10-fused-permuted-attention.md) — Amulet softmax-equivariance + Hidden-No-More + F1+ origin.
> - Plan: [`m1-10-security-review.md`](m1-10-security-review.md) — F1–F8 option survey, F1+ chosen.
> - Handoff: [`2026-05-21-attn-offload-spike.md`](../archive/handoffs/2026-05-21-attn-offload-spike.md) — prior B=1 spike showing why batching is the unlock.
>
> **Status:** **R1.4 aborted at Phase A — gate failed by 16×.** Plan retained for the spike methodology + abort rationale. Bucket 2 deferred indefinitely on iGPU.
> **Author date:** 2026-05-22.
> **Phase A run date:** 2026-05-22.
> **Scope:** decode-only. Prefill attention bucket (11.6 % at B=8) stays in-TEE; addressing it requires F1+ chain (+1 TEE↔GPU round-trip per call) which conflicts with the round-trip-count clause of the acceptance gate.

---

## ⚠ Phase A result (2026-05-22) — ABORT

The crossover spike measurement at the Qwen3-4B GQA shape (B=8,
num_heads=32, num_kv_heads=8, d_head=128) on Strix Halo iGPU
(Radeon 8060S / RADV gfx1151 via wgpu Vulkan f16):

| Shape (B=8) | `in_tee_rayon_b8` | `gpu_batched_b8_no_mask` | `gpu_batched_b8_with_mask` | GPU vs in-TEE |
|---|---:|---:|---:|---:|
| n_kv = 256  | **1.06 ms**  |  48.5 ms |  47.9 ms | 45.9× slower |
| n_kv = 1024 | **7.13 ms**  | 186.2 ms | 185.8 ms | 26.1× slower |
| n_kv = 2048 | **22.3 ms**  | 364.8 ms | 369.5 ms | **16.4× slower** |

**Per plan §1 acceptance gate:** decode-wall ≥ 30 % reduction
required; result is **16× slower in the wrong direction**. Gate
failed by ~50×. R1.4 engineering (Phases B–E) abandoned.

**Q11 answered as a side-effect:** with_mask vs no_mask delta is
< 2 % at every shape. **burn-cubecl-fusion folds the `+ mask` add
into adjacent kernels** at our shape. A custom-WGSL FlashAttention-D
kernel would save < 2 % on the mask elision path.

### Why M1.11's crossover hypothesis was wrong

`m1-11-batched-decode.md` §6 assumed:
- in-TEE ≈ 2 ms × B (linear in B)
- GPU ≈ 22 ms + ε × B (**launch-dominated** until B ≈ 16)
- Crossover ≈ B 11–16

Measured reality: GPU compute scales **linearly with B**, like CPU.
The 22 ms at B=1 (prior attn-offload-spike) was already
compute-bound on RADV gfx1151, not launch-dominated. The slowdown
ratio decreasing from 45.9× (n_kv=256) → 16.4× (n_kv=2048) confirms
this — compute dominates at larger shapes, the per-element
throughput gap shrinks but doesn't invert.

### What this rules out

- **burn-chain on cubecl-wgpu** at decode-m=1 — fundamentally
  non-viable on this hardware regardless of B. Not a missing-fusion
  problem.
- **Custom WGSL FlashAttention-D for iGPU decode** — score-tensor
  materialisation HBM saving is at most a few percent (scores
  tensor is ~1 MB at decode shape; not memory-bound). The remaining
  16× gap is compute throughput, not memory bandwidth.
- **cubek-attention Strategy::Unit at decode-m=1** — already
  measured worse than burn-chain in the attn-offload-spike (lanes
  wasted at `n_q < plane_dim`).

### What might still work, but **not on this plan's critical path**

- **`cubecl-hip` backend swap** — direct ROCm/HIP path bypasses
  Vulkan and could access MFMA hardware. Plausibly 2-3× speedup at
  best; wouldn't close a 16× gap. Spike-able as a side-track when
  cubecl-hip's API stabilises.
- **dGPU substrate (M5.9 bring-up)** — PCIe DMA dominates at
  decode-m=1; the bottleneck mix is different. **Re-measure
  bucket-2 on dGPU before reviving it** — many "iGPU-blocked"
  attention levers may become genuinely large on dGPU per the
  perf-bucket roadmap's §7 hand-off.

### Bench artefacts retained

The R1.4 spike cells stay in
`crates/gelo-gpu-wgpu/benches/amulet_attention.rs` (group
`amulet_attention_r1_4/`) as a re-runnable comparison harness. Any
future bucket-2 revival (custom WGSL, cubecl-hip, dGPU) must beat
the same `in_tee_rayon_b8` baseline at the gate threshold.

```bash
cargo bench -p gelo-gpu-wgpu --bench amulet_attention -- amulet_attention_r1_4
```

### Pivot from M1.12 perf-bucket roadmap

Per `docs/handoffs/2026-05-22-perf-bucket-roadmap-r3-default.md`,
the next-largest accessible buckets after R3 are:

- **bucket 3** — bf16 mask GEMM on GPU (prefill 39 % bandwidth-
  contention lever). ~1-2 weeks engineering.
- **bucket 4** — Async pipelining (M1.12 R4). Needs Q#2 RADV-
  serialisation spike first; ~5-8 days if it survives.
- **bucket 5** — Varlen / chunked batching (production lever, no
  bench win).

Recommended next: pre-spike bucket 4's RADV-async question before
committing to bucket 3's engineering.

---

## 0. TL;DR

At Qwen3-4B B=8 n=2048 K=64, post-R3 decode wall is 112.99 s. The
single largest remaining bucket is `tee:attn_cached_inplace_many` at
**49.7 % (58.3 s)** — the in-TEE rayon-parallel-over-B attention loop
in `decoder_block_cached_batched`. This plan moves that attention
through `engine.fused_attention_batched` as a single batched GPU
dispatch per layer per decode step, under permuted-attention Phase 1b
(Amulet softmax-equivariance + Hidden-No-More noise, GPU softmax — no
mask round-trip violation because the causal mask is a no-op at
`n_q = 1`).

| Lever | Engineering | Win |
|---|---|---|
| **R1.4** — Wire `decoder_block_cached_batched` → `permuted_attention_cached_batched` → `engine.fused_attention_batched` | ~2-3 days | Aim: ≥ 30 % decode-wall reduction at Qwen3-4B B=8 (target ≤ 79 s vs 112.99 s post-R3) |

Plus de-risk + gate items:

| Item | Wall | Decides |
|---|---|---|
| **Phase A** — Crossover spike (`amulet_attention.rs` B=8 cell) | ~0.5 day | Go / no-go for R1.4 engineering |
| **Phase D** — `c5_perm_attn_pad` AloePri gate + σ sweep | ~3-4 days | Default-flip ladder advance |

Total: **~7-8 days end-to-end.**

Sequencing: **Phase A → Phase B (engineering) → Phase C (bench) → Phase D (c5 gate) → Phase E (default flip).**

---

## 1. Acceptance gate (Q1)

Hard abort gate, applied at the end of Phase C:

- Decode wall at Qwen3-4B B=8 n=2048 K=64 drops **≥ 30 %** on top of
  R3 (112.99 s → ≤ 79 s).
- `tee:attn_cached_inplace_many` bucket falls below **15 %** of decode
  wall (today 49.7 %).
- New `engine:fused_attention_batched` bucket appears at plausible
  scale (not zero, not the whole wall).
- Prefill wall at B=8 n=2048 within **±5 %** of post-R3 baseline (no
  prefill regression).
- **No growth in mask offload count** per forward (each
  `permuted_attention_cached_batched` call counts as exactly one
  offload — same as today's per-sequence in-TEE attention, which
  counts as zero, but the constraint is interpreted as "≤ 1 offload
  added per attention call" — single batched dispatch per call is
  allowed; the F1+ chain at decode would be ≥ 2 and is rejected).
- **No growth in TEE↔GPU round-trips per call beyond 1** (Phase 1b at
  decode is single-dispatch — one upload + one download per call —
  honours this).

If the gate fails: **revert R1.4 engineering**, file the bucket as
"deferred pending custom WGSL FlashAttention-D" alongside the
parked bf16-mask bucket.

---

## 2. Threat model — what changes under R1.4

GELO §3 baseline unchanged. Per-batch fresh `A`, per-forward fresh
shield, etc. R1.4 introduces one new GPU observation shape: the
permuted-attention dispatch at decode.

| Item | Today | R1.4 |
|---|---|---|
| QKV / O / gate-up / down masked offloads | GPU sees `(stacked_n, d_in)` operand + `(stacked_n, d_out)` output | unchanged |
| LM-head | masked offload (R3 default) | unchanged |
| **Attention at decode** | in-TEE rayon over B sequences, `causal_gqa_attention_cached` per b | **GPU dispatch via `permuted_attention_cached_batched`: independent `π_b` per sequence; HNM noise `η_Q_b, η_K_b`; combined right-padding mask sent as additive tensor** |
| Attention at prefill | in-TEE serial over B, `causal_gqa_attention` per b | unchanged (scope §0) |
| Weights | VRAM-stationary | unchanged |

The new GPU observation per attention call:

- `(B·num_heads, n_q=1, d_head)` permuted-noised Q
- `(B·num_heads, n_kv, d_head)` permuted-noised K
- `(B·num_heads, n_kv, d_head)` permuted V
- `(B, 1, n_kv)` soft-`-30` right-padding mask

This is two new shape regimes layered on the GELO baseline:
**permuted-attention shape under HNM** AND **right-padding mask
pattern revealing `lens[b]`**. Both are validated empirically in
Phase D's combined `c5_perm_attn_pad` gate.

---

## 3. R1.4 design

### 3.1 New TEE-side primitive: `permuted_attention_cached_batched`

New function in `crates/gelo-protocol/src/attention.rs`:

```rust
pub fn permuted_attention_cached_batched(
    engine: &dyn GpuOffloadEngine,
    q: ArrayView4<'_, f32>,   // (B, num_heads, n_q=1, d_head)
    k: ArrayView4<'_, f32>,   // (B, num_kv_heads, n_kv_max, d_head)
    v: ArrayView4<'_, f32>,   // (B, num_kv_heads, n_kv_max, d_head)
    q_pos_offsets: &[usize],  // length B
    seq_lens: &[usize],       // length B; per-sequence valid n_kv
    scale: f32,
    cfg: &PermAttnConfig,
    rng: &mut impl Rng,
) -> Result<Array4<f32>>      // (B, num_heads, n_q=1, d_head)
```

Internally:

1. **Independent permutations.** Sample `(π_q_b, π_kv_b)` independently
   per sequence b (Q6). Permute K_b and V_b along the `n_kv` axis with
   `π_kv_b`; permute Q_b along `n_q` axis with `π_q_b` (trivial at
   `n_q = 1`).
2. **HNM noise.** Add independent `η_Q_b, η_K_b ~ N(0, σ²)` to permuted
   Q and K per sequence. σ is `cfg.noise_sigma` (Q10 sweep determines
   default).
3. **GQA expansion.** Replicate each `num_kv_heads` K and V across
   `num_attention_heads // num_kv_heads` group members per sequence.
   Stack into `(B·num_heads, ·, d_head)`.
4. **Right-padding mask.** Build `(B, n_q=1, n_kv_max)` additive
   mask: `mask[b, 0, j] = -cfg.causal_mask_neg` for
   `j ≥ seq_lens[b]`, else `0`. Broadcast across `num_heads` to
   match engine input shape `(B·num_heads, n_q, n_kv)`. (Causal mask
   itself is no-op at `n_q = 1, q_pos_offset = seq_lens[b] - 1`.)
5. **Engine dispatch.** Single call to
   `engine.fused_attention_batched(q_stacked, k_stacked, v_stacked,
   scale, Some(mask))`. Engine returns
   `(B·num_heads, n_q, d_head)`.
6. **Per-sequence inverse permutation.** Apply `π_q_b⁻¹` per b to
   recover canonical row order — trivial at `n_q = 1`.
7. **Reshape back to `(B, num_heads, n_q, d_head)`** for the caller.

The existing `permuted_attention_cached` (single-sequence) stays as
the non-batched path. Implementation note: factor the per-b permute +
noise into a private helper that both functions call, so the
single-sequence path doesn't drift.

### 3.2 Wiring: `decoder_block_cached_batched`

Replace the rayon-over-B in-TEE attention loop at
`crates/gelo-embedder/src/decoder/forward.rs:429-460` with a single
call to `permuted_attention_cached_batched`. Conditional on the
env-var gate (§7).

```rust
// Replace:
profile::time("tee:attn_cached_inplace_many", || {
    use ndarray::parallel::prelude::*;
    ctx.axis_chunks_iter_mut(...).into_par_iter()...
});

// With:
let use_gpu_attn = std::env::var("R1_4_BATCHED_ATTN_GPU")
    .map(|v| v == "1")
    .unwrap_or(false);
if use_gpu_attn && batch_size >= cfg.batched_attn_min_batch {
    profile::time("engine:fused_attention_batched", || {
        permuted_attention_cached_batched(
            engine, q_view4, k_view4, v_view4,
            q_pos_offsets, &lens, scale, &cfg.perm_attn, &mut rng,
        )
    })?
} else {
    // existing rayon-over-B in-TEE path stays as fallback
    ...
}
```

`cfg.batched_attn_min_batch` defaults to `usize::MAX` until Phase A
measures the crossover B; gets set to the measured value after Q4
spike clears.

KV cache layout stays as-is (`(B, layers, max_total_len, kv_dim)` per
M1.11 D1.4). At dispatch time we pad each sequence's K/V slice to
common `n_kv_max = max(lens[b])` along the `n_kv` axis; right-padding
mask suppresses contribution from padded positions (Q7).

### 3.3 Engine kernel: burn-cubecl matmul chain via cubecl-wgpu

The engine implementation in
`crates/gelo-gpu-wgpu/src/lib.rs:555-608` already provides
`fused_attention_batched` via a burn-tensor chain (matmul → mul_scalar
→ add(mask) → softmax → matmul). The chain runs on
**cubecl-wgpu → wgpu → Vulkan → RADV → amdgpu** on the dev iGPU.

**Routing inside the engine (Q3):** keep current burn-chain for the
decode-m=1 shape (cubek-attention's `Strategy::Unit` wastes lanes
when `n_q < plane_dim`; `Strategy::BlackboxAccelerated` segfaults on
RDNA3.5 — both confirmed in the attn-offload spike). The
`with_cubek_min_n_q(usize)` builder knob is preserved so cubek can
fire on future prefill workloads; today it stays at the default
(decode-only call sites won't trip it).

**Kernel-optimality honesty (Q11):** the burn-chain via
cubecl-wgpu is the best feasible kernel without multi-week custom
WGSL work. It is NOT optimal on gfx1151 — ROCm/HIP native (via
`cubecl-hip`, already in Cargo) would access MFMA instructions and AMD
Composable Kernel FlashAttention; a hand-rolled WGSL FlashAttention-D
would avoid the `(B·H, n_q, n_kv)` scores tensor materialisation. Both
filed as M1.13+ follow-ups (§8). Crossover spike instruments actual
GPU dispatch count (§4) so we know whether cubecl-fusion is firing
on our chain.

### 3.4 KV padding topology (Q7) — lockstep + ragged + soft `-30` mask

Each decode step pads K/V to `n_kv_max = max(lens[b])` along the
`n_kv` axis. Mask suppresses padded positions via soft `-C` (where
`C = cfg.causal_mask_neg`, default `30`) — `exp(-30) ≈ 1e-13` at
softmax, structurally below f32 noise floor.

Trade: the mask pattern reveals each sequence's valid prefix length
to the GPU. This is the `c5_pad_mask` half of the combined gate (§6).

Option F (FlashAttention-2 varlen / vLLM-style block-table single
kernel that handles ragged `n_kv` natively without padding or mask)
is deferred to M1.13+ — requires cubek-attention API extension or
custom WGSL kernel.

### 3.5 PermAttnConfig changes

Three new fields on `PermAttnConfig` in `gelo-protocol::attention`:

```rust
pub struct PermAttnConfig {
    // existing:
    pub noise_sigma: f32,
    pub causal_mask_neg: f32,
    pub decode_softmax_on_gpu: bool,
    // new:
    /// Enable batched decode dispatch via fused_attention_batched.
    /// Routed from env-var R1_4_BATCHED_ATTN_GPU at runtime construction.
    pub batched_decode_attn: bool,
    /// Crossover batch-size threshold below which in-TEE rayon-over-B
    /// stays the path. Default usize::MAX until Phase A spike.
    pub batched_decode_min_batch: usize,
    /// Sampled per call; if true (production), the seed advances per call.
    /// Test fixtures pin to deterministic.
    pub deterministic_for_test: bool,
}
```

Pre-baked variants: `HIDDEN_NO_MORE_BATCHED_DECODE` mirroring the
`HIDDEN_NO_MORE_DECODE_GPU` predecessor with the new fields populated.

---

## 4. Phase A — Crossover spike (~0.5 day)

**Goal:** measure burn-chain GPU attention at B=8 vs the existing
in-TEE rayon-over-B path at the Qwen3-4B decode shape. **Plus
measure actual GPU dispatch count** to characterise cubecl-fusion
behaviour at our shape (Q11).

### 4.1 Bench extension

Extend `crates/gelo-gpu-wgpu/benches/amulet_attention.rs` with a new
cell:

```rust
fn decode_m1_qwen3_4b_b8(c: &mut Criterion) {
    // Shape: B·H = 8 · 32 = 256, n_q = 1, n_kv ∈ {256, 1024, 2048},
    // d_head = 128.  All three of:
    //   1. in_tee_rayon:       8-way rayon over B, causal_gqa_attention_cached
    //   2. burn_chain_batched: engine.fused_attention_batched
    //   3. (informational)     cubek_attention::launch::launch(Strategy::Unit)
    //                          at the same shape — expected to underperform
    //                          burn_chain at n_q = 1, confirms the routing
    //                          decision still holds.
}
```

### 4.2 GPU dispatch counting

Add timestamp-query instrumentation per-call to count how many GPU
dispatches the burn-chain actually emits:

- `wgpu::QueryType::Timestamp` queries before and after each
  `engine.fused_attention_batched` call
- Compare query-count to expected (5 if no fusion, 1-2 if full fusion)
- Equivalent: enable wgpu validation layer with command-buffer
  inspection

Outcome documented in spike result table.

### 4.3 Go/no-go decision

| Result | Action |
|---|---|
| burn-chain ≥ 1.5× faster than in-TEE-rayon at B=8 n_kv=2048 | Commit to Phase B; set `batched_decode_min_batch = 8` |
| burn-chain 1.0–1.5× faster | Borderline; commit to Phase B but expect tight perf gate at Phase C |
| burn-chain ≤ 1.0× | **Abort R1.4.** File bucket 2 as "deferred pending custom WGSL FlashAttention-D" (§8). |
| fusion fires (≤ 2 dispatches/call) | burn-chain is good as-shipped; no follow-up |
| fusion doesn't fire (≥ 4 dispatches/call) | File custom WGSL FlashAttention-D as M1.13+ priority follow-up |

---

## 5. Phase B — Engineering (~2-3 days)

Tasks, in order:

1. **B.1** (0.5 day) — `permuted_attention_cached_batched` in
   `gelo-protocol::attention` (§3.1). Factor per-b permute+noise into
   a private helper that the single-sequence path also uses to prevent
   drift.
2. **B.2** (0.5 day) — `PermAttnConfig` extension + variant
   `HIDDEN_NO_MORE_BATCHED_DECODE` (§3.5).
3. **B.3** (0.5 day) — `decoder_block_cached_batched` wire-up with
   env-var + batch-threshold gating (§3.2). Existing rayon-over-B
   path stays as fallback.
4. **B.4** (0.5 day) — Per-sequence parity test in
   `crates/gelo-protocol/tests/permutation_attention.rs`: assert
   `permuted_attention_cached_batched(B sequences)` outputs match B
   separate `permuted_attention_cached(single)` calls to f32 mask
   round-trip floor.
5. **B.5** (0.5 day) — Greedy-generation parity test in
   `crates/gelo-embedder/tests/qwen3_generation_e2e.rs`: assert
   `generate_batched` with `R1_4_BATCHED_ATTN_GPU=1` produces
   semantically-equivalent output per b to the rayon-fallback path
   (per M1.11 D3's "f32-floor parity, not byte-identical" contract).
6. **B.6** (0.25 day) — Add `engine:fused_attention_batched` profile
   bucket; wire into the existing `profile::time` taxonomy so
   measurement Phase C can attribute the new bucket.

---

## 6. Phase C — Bench validation (~1 day)

### 6.1 Extend microbench

Extend `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`
with a bucket-2 cell:

```bash
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 \
GELO_BENCH_MAX_TOKENS=64 R1_4_BATCHED_ATTN_GPU=1 \
  cargo test -p gelo-gpu-wgpu --release \
  --test qwen3_m1_12_r1_q1_microbench -- --ignored --nocapture \
  m1_12_per_op_breakdown_prefill_decode
```

### 6.2 Acceptance gate (§1)

Capture per-op breakdown for two conditions:

1. **post-R3** (existing baseline) — `R1_4_BATCHED_ATTN_GPU=0`
2. **post-R1.4** — `R1_4_BATCHED_ATTN_GPU=1`

Assert all five clauses of §1's gate. Document in the per-op
breakdown table.

---

## 7. Phase D — `c5_perm_attn_pad` AloePri gate (~3-4 days)

### 7.1 Combined-gate scope (Q8)

One capture campaign covering both new shape regimes:

- Permuted-attention dispatch shape (Q+η, K+η, V, all at
  `(B·H, n, d_head)`)
- Right-padding mask pattern `(B, 1, n_kv)`

Both regimes are observed in the same forward pass. Single capture +
single driver-run sequence.

### 7.2 σ sweep (Q10)

Sweep `cfg.noise_sigma ∈ {0.01, 0.03, 0.1, 0.3, 1.0}`. For each σ:

1. Capture PCIe-side snapshots of the bucket-2 dispatch on a
   Qwen3-4B fixture at B=8 (use existing AloePri snapshot capture
   plus a new `c5_perm_attn_pad` condition variant).
2. Run anchor_ica / jade / jd / gram_error attack drivers against
   the captured snapshots.
3. Score attack accuracy vs c2 baseline (default mask topology, no
   batched-attention dispatch).
4. Score greedy-generation correctness via existing
   `qwen3_generation_e2e` parity test at this σ.

### 7.3 Selection rule

Pick the **smallest σ** such that:

- Attack accuracy is within sample-noise of c2 (per
  `aloepri_attack_keymat_cuda_philox_bias.md`, ~5pp single-seed std at
  d=2560; use 5-seed sweep for confidence).
- Greedy-generation output is byte-stable vs σ=0 baseline (no model
  degradation).

If no σ in the range clears: extend the upper bound or revert R1.4
(model is too sensitive to HNM noise at our shapes).

### 7.4 Methodology references

Mirrors C3 (HD₃) and C5 (Phase 1b precedent) gates per
`aloepri_hd3_gate_phase_a_b.md`. Capture infrastructure extension:
add `c5_perm_attn_pad` to the `--condition` enum in
`aloepri-attack-snapshot-runner`; snapshot the new tensors (mask
included, since the mask leak is half the gate).

---

## 8. Phase E — Default-flip ladder (~0.5 day)

3-step ladder (Q9):

1. **At Phase B commit:** `R1_4_BATCHED_ATTN_GPU=0` default.
   `batched_decode_min_batch = usize::MAX` in the default
   `PermAttnConfig`. Engineering shipped but inactive.
2. **After Phase C clears the perf gate:** flip
   `R1_4_BATCHED_ATTN_GPU=1` as the runtime default for the
   gelo-snp-runner / gelo-embedder paths, with
   `batched_decode_min_batch = 8` (or whichever the spike
   measured). Lower B stays on the in-TEE-rayon fallback.
3. **After Phase D clears `c5_perm_attn_pad` at the chosen σ:**
   remove the env-var override path; bucket-2 dispatch becomes
   the only path at B ≥ threshold. Update memory + handoff to
   reflect new default. Mirrors LM-head precedent in this session's
   handoff.

---

## 9. Out of scope / follow-up spikes

Obfuscated-attention re-analysis filed three candidates with non-zero
EV that don't ship in v1:

- **F7 — fused-out matmul under F1+** (custom WGSL kernel that keeps
  probs resident GPU-side between TEE-softmax-writeback and final
  matmul, saving HBM bandwidth proportional to `n²·heads`). Only
  relevant if the prefill bucket grows enough to feel the
  materialisation cost. M1.13+ research; ~2-3 weeks engineering.
- **TwinShield-Xue softmax blinding** (arXiv 2507.03278, the Xue
  citation — disambiguated from the Liu citation in our docs) —
  `e^(X+R)` blinding pushes softmax to GPU under causal mask without
  the Amulet identity. If it survives HNM-class attacks, eliminates
  the TEE round-trip on prefill F1+ chain. ~1-2 week security spike;
  P2 in `private_llm_inference_round_2` memo.
- **Ragged-aware single-dispatch kernel** (FlashAttention-2 varlen /
  vLLM block-table). Production LLM-serving standard; eliminates
  padding + mask entirely. Multi-week engineering. Either cubek-
  attention API extension or custom WGSL. M1.13+ research item.

Plus two perf-only follow-ups not blocked by security:

- **`cubecl-hip` backend swap** — already in our Cargo tree, unused
  for WgpuVulkanEngine. Direct ROCm/HIP path on gfx1151; potentially
  material perf delta if MFMA fires. ~1-2 days for backend swap +
  microbench. Side-track once Phase C settles burn-chain numbers.
- **Custom WGSL FlashAttention-D for decode m=1** — single-pass
  fused kernel avoiding the `(B·H, n_q, n_kv)` scores tensor
  materialisation. ~1-2 weeks engineering. Priority elevated if
  Phase A's GPU-dispatch-count instrumentation shows
  cubecl-fusion not firing on our chain.

Rejected (in the re-analysis):

- F2/F3/F4 (causal-mask leak workarounds — covered in
  `m1-10-security-review.md`)
- F6 (block-randomised mask — privacy weakening, future-rnd)
- F8 (OutAttnMult + F1+ — strictly worse at long context)
- TwinShield-Liu HE softmax (heavy crypto deps; not needed)
- SCX (different problem — KV-cache integrity, not attention dispatch)
- Confidential GPU substrate (deployment fork; commodity-GPU bet
  remains correct through 2026 per
  `private_llm_inference_round_2`)

---

## 10. Open questions / risks

1. **cubecl-fusion firing** — burn-cubecl-fusion's behaviour on our
   chain at decode shapes is unmeasured. Phase A's GPU-dispatch-count
   instrumentation answers this; if it doesn't fire we file the
   custom WGSL kernel as M1.13+ priority.

2. **σ sweep ceiling** — if σ=1.0 doesn't clear the c5 gate, model
   is too sensitive to HNM noise at our shapes. Mitigation: try
   per-layer σ (currently rejected as premature in Q10).

3. **B-threshold robustness** — `batched_decode_min_batch = 8` is
   set from the Phase A measurement. If production workloads at
   B < 8 trip the threshold and fall back to in-TEE-rayon, the win
   doesn't generalise. Reranker path (`CausalDiscriminatorRerank`)
   is naturally B ≫ 8; single-stream extraction is not. Document
   that bucket-2 is a **batched-workload lever**, not a single-
   stream lever.

4. **wgpu UMA budget at B=8 K=64** — extra `(B·H, n_kv_max, d_head)`
   K/V tensors stack the GPU operand-buffer pressure. At B=8 H=32
   n_kv_max=2048 d_head=128 fp16: ~16 MB per tensor × 2 (K+V) =
   32 MB per call. Sustainable on Strix Halo iGPU UMA budget (~3 GB
   wgpu default); needs verification at Phase C if compounded with
   QKV / O / gate-up / down offload tensors.

5. **Determinism contract** — production calls advance the RNG per
   forward; test fixtures pin a seed for reproducibility. Per-call π
   sampling adds entropy consumption proportional to B per layer per
   step. Phase B.5's parity test asserts test-fixture stability; no
   contract change for production.

---

## 11. Reproducing the baseline before Phase A starts

```bash
# Existing post-R3 baseline (the 112.99 s decode wall the gate is
# measured against)
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 \
GELO_BENCH_MAX_TOKENS=64 \
  cargo test -p gelo-gpu-wgpu --release \
  --test qwen3_m1_12_r1_q1_microbench -- --ignored --nocapture \
  m1_12_per_op_breakdown_prefill_decode

# Existing amulet_attention microbench (the substrate Phase A extends)
cargo bench -p gelo-gpu-wgpu --bench amulet_attention

# Existing rerank baseline (the natural-B ≫ 8 user of bucket 2 once shipped)
cargo test -p gelo-reranker --release --test comparative_bench \
  -- --nocapture causal_discriminator
```

---

## 12. References

- Amulet softmax-equivariance: arXiv 2512.07495 (the protocol that
  motivates permuted attention)
- Hidden No More: arXiv 2505.18332 (Gaussian noise mitigation against
  sequential-vocabulary-matching attacks on fixed permutations)
- AloePri: arXiv 2603.01499 (attack-suite that the c5 gate runs)
- `crates/gelo-protocol/src/substrate.rs:253` — engine trait surface
  (`fused_attention_batched`)
- `crates/gelo-protocol/src/attention.rs:393` — existing
  `permuted_attention_cached` (Phase 1b branch + F1+ legacy branch)
- `crates/gelo-gpu-wgpu/src/lib.rs:555` —
  `WgpuVulkanEngine::fused_attention_batched` (burn-chain override)
- `crates/gelo-embedder/src/decoder/forward.rs:309` —
  `decoder_block_cached_batched` (the call site this lever rewires)
- `docs/archive/handoffs/2026-05-21-attn-offload-spike.md` — prior B=1 spike
  showing why batching is the unlock
- `docs/plans/m1-10-security-review.md` — F1+ origin (and F7 / F8
  / TwinShield-Xue follow-up references)
