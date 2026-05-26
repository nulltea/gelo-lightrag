---
type: plan
status: current
created: 2026-05-22
updated: 2026-05-22
tags: [m1.12, gpu, perf]
---

# M1.12 — TEE↔GPU throughput: memory efficiency, round-trip count, bandwidth

> **Parent context:**
> - Handoff: [`2026-05-22-q3-4b-b8-mask-sweep.md`](../handoffs/2026-05-22-q3-4b-b8-mask-sweep.md) — bottleneck triage from real Qwen3-4B B=8 n=2040 measurement.
> - Plan: [`m1-11-batched-decode.md`](m1-11-batched-decode.md) — the batched substrate this rides on.
> - Threat-model parking note: [`docs/dev/prototype/future-rnd.md`](../dev/prototype/future-rnd.md) §5 (encrypted-KV-on-GPU — explicitly out of scope).
>
> **Status:** plan, post-grilling-session revisions baked in.
> **Author date:** 2026-05-22.
> **Supersedes** an earlier draft titled "UMA residency overhaul" — UMA-mapped operand work was dropped because it's iGPU-only and doesn't carry to production SEV-SNP + dGPU.

---

## 0. TL;DR — the right axes

The Q3-4B benches and the UMA-vs-dGPU representativeness analysis
in the handoff identify the levers that **carry over to production
SEV-SNP + VFIO dGPU**:

| Lever | Production-relevant? |
|---|---|
| GELO mask family at pow2 (HD₃ + shield-to-pow2) | Yes — landed in M1.11 |
| Batched substrate (per-sequence A_b) | Yes — landed in M1.11 |
| UMA-mapped masked-operand buffer | **No** — iGPU-only artefact; dropped |
| **LM-head GPU offload** (move ~120 s/forward compute_logits) | **Yes** |
| **Async pipelining** (overlap CPU mask + GPU matmul) | **Yes** |
| **VRAM-stationary weight bookkeeping** (factor take()-on-provision) | Yes (consolidation) |
| wgpu UMA budget knob | No — dropped (dev-velocity only) |

Three items survive:

| Item | Engineering | Win |
|---|---|---|
| **R1.** Factor shared `provision_decoder_into` helper | ~1 day | Consolidate take()-on-provision across 6 callers |
| **R3.** LM-head GPU masked offload (`token_embedding` × hidden → vocab) | ~3 days + c6 gate | Retire ~120 s/forward `compute_logits` bucket |
| **R4.** Async pipelining — decouple matmul submit/wait | 5-8 days | Overlap CPU mask (~25 % share) and GPU matmul (~50 % share) → ~25-30 % wall reduction |

Plus one gate task:

| Gate | Wall | Decides |
|---|---|---|
| **c6 AloePri spot-check** on LM-head offload shape | ~3 days | Flip `LM_HEAD_GPU_OFFLOAD=1` default-on |

Sequence: **R1 → R3 (behind flag) → c6 → R4**. Total ~12-15 days end-to-end.

---

## 1. Threat model — what changes under M1.12

GELO §3 baseline unchanged. M1.12 changes which masked-operand
shapes the GPU observes per forward:

| Item | Today | M1.12 |
|---|---|---|
| QKV / O / gate-up / down masked offloads | GPU sees `(stacked_n, d_in)` operand + `(stacked_n, d_out)` output | unchanged |
| Attention | in-TEE (default) or permuted-GPU (opt-in) | unchanged (gated under M1.11 D3 separately) |
| LM-head (compute_logits) | in-TEE, single-threaded vocab × hidden loop | **R3: masked offload at `(1+k, hidden) × (hidden, vocab)`** |
| Weights | VRAM-stationary (`take()` in production) | unchanged; R1 consolidates the helper |

R3's new observation shape is the load-bearing security delta: the
GPU sees a `(1+k, 152 064)` masked output per decoded token under
the same per-forward `A`. The c6 gate captures whether the
~37× wider output dim opens an attacker-recovery surface the
existing c1–c5 conditions don't cover.

KV cache stays host-resident (parked per future-rnd §5).

---

## 2. R1 — Factor shared `provision_decoder_into` helper

### Problem

Three production callers already do `take()`-on-provision (the
pattern that drops the host bf16 weight Arc once the wgpu engine
consumes the upload):

- `gelo-embedder/src/decoder/embedder.rs::GeloQwenEmbedder::new`
- `gelo-reranker/src/causal_discriminator.rs::CausalDiscriminatorRerankService::new`
- `gelo-snp-runner/src/extraction.rs::DecoderRuntime::from_config_and_dir`

Each has a near-duplicate 7-projection take() loop. R3 will add an
8th projection (LM-head), and we'd otherwise duplicate the new
weight handle across three call sites.

### Fix

Single helper at `gelo-embedder/src/decoder/weights.rs::provision_into`:

```rust
pub fn provision_into<X: TrustedExecutor>(
    weights: &mut DecoderWeights,
    cfg: &DecoderConfig,
    exec: &mut X,
) -> Result<()>;
```

Three production callers + three bench helpers in
`crates/gelo-gpu-wgpu/tests/qwen3_*bench.rs` all switch to this
single function. Bert variant in `bert::weights::provision_into`
for symmetry.

### Acceptance

- Bench `qwen3_4b_batched_mask_sweep` RSS-after-provision drops by
  ~7 GiB on Qwen3-4B (today 9.2 GiB → target ~2.2 GiB).
- Three production callers produce byte-identical generate output
  pre/post-refactor (functional regression check via existing
  parity tests).

---

## 3. R3 — LM-head GPU masked offload

### Problem

`generation::compute_logits` runs single-threaded over the 152 064
× 2 560 token_embedding matrix per decoded token, with bf16 → f32
widening per element. Estimated **~120 s/forward** on the v7
extraction fixture (action item: confirm via
`profile::time("tee:compute_logits", …)` instrumentation before
R3 engineering starts — see §6 open questions).

This is potentially the single largest in-TEE bucket left after
M1.11 + R1.4.

### Design

Register `token_embedding` as a new VRAM weight handle
(`WeightKind::LmHead`). At each call to `compute_logits` (post-
prefill last row + per decode step), invoke the existing
`offload_linear` path:

- Substrate masks the last hidden state under the current
  per-forward `A_b` per sequence at `(1, hidden)`.
- Stacks with shield rows → `(1+k, hidden)` per sequence.
- Engine matmul: `(1+k, hidden) × (hidden, vocab=152 064)` →
  `(1+k, vocab)`.
- Substrate unmasks via `Aᵀ` → `(1+k, vocab)` plaintext-equivalent
  (mask round-tripped).
- Strips shield rows → `(1, vocab)` logits.
- TEE samples per the existing `SamplerConfig::Greedy` path.

Tied-embedding handling: `token_embedding` stays host-resident for
`embedding_lookup` (input row gather, ~tens of ms/forward, not a
bottleneck); `Arc::clone(&token_embedding)` registers as VRAM
W_lm_head. 778 MB × 2 residency cost — tractable on this hardware.
If duplication becomes binding on a future smaller-VRAM
deployment, untying via the offline conversion script at
`private-rag-path-2/python/aloepri-llm/obfuscate_qwen3_gguf.py` is
the escape hatch.

### Cost — bandwidth and dispatch count

Per decoded token at B=8:

- Upload: `(1+k=16, 2 560) f16` per sequence × 8 = ~640 KB.
- Download: `(1+k=16, 152 064) f16` per sequence × 8 = ~78 MB.
- Total per decode step: ~78.6 MB extra.

Over a 500-token decode: ~39 GiB additional transfer. Compared to
M1.11's ~48 GiB per Qwen3-4B forward (linear offloads), this is a
~80 % increase in bytes-on-wire. **Acceptable on UMA (memcpy);
material on production dGPU (PCIe DMA)**. M1.13 should evaluate
batched LM-head dispatch (accumulate K decode steps' hiddens
before dispatching one big matmul) as a follow-up.

### Feature flag

`LM_HEAD_GPU_OFFLOAD=1` env var routes through the new path. Default
off until c6 clears. Mirrors the M1.11 `BATCHED_DECODE_SHARED_A=1`
pattern.

### Acceptance

Functional: synthetic-weights parity test (LM-head GPU offload
output matches in-TEE `compute_logits` to mask round-trip floor;
greedy argmax stable). Real-weight: v7 extraction byte-identical
entity/relation output across both paths.

Performance gate: on Qwen3-4B at B=8 with `LM_HEAD_GPU_OFFLOAD=1`,
the new `tee:compute_logits` bucket (instrumented in §6) drops to
near-zero; the new `engine:matmul` LM-head sub-bucket appears at
plausible scale.

---

## 4. c6 — AloePri spot-check on LM-head offload shape

### Why this needs its own gate

The c1–c3 AloePri baseline covers QKV / O / gate-up / down offload
shapes. c4 / c5 cover M1.11 batched mask topologies. The LM-head
offload introduces a **new shape regime**: `(1+k=16, vocab=152 064)`
masked output — 37× wider than `(1+k, q_dim=4 096)` for QKV, on the
same per-forward `A`. Known AloePri attacks (VMA / IMA / JADE / JD /
anchor_ica / gram_error) scale with output dim — more samples per
mask = stronger inverse-recovery surface.

The protocol's per-forward-pass-mask argument still applies in
principle, but the proof's quantitative bounds were written for
narrower outputs. Empirical attack-suite re-measurement at the
LM-head shape is the safest signal.

### Methodology

Mirrors c4 / c5 precedent (handoff
`aloepri_hd3_gate_phase_a_b.md`):

- Capture PCIe-side snapshots at the LM-head dispatch shape on a
  Qwen3-4B fixture with `with_snapshot_capture()`.
- Run anchor_ica / jade / jd / gram_error attack drivers at this
  shape, B=8.
- Compare attack effectiveness against c2 baseline (default mask
  topology, no LM-head offload). c6 clears if attack accuracy is
  within sample noise of c2; flags if higher.

### Outcome → default flip

- **c6 clears** → `LM_HEAD_GPU_OFFLOAD=1` becomes default; ~120 s/forward
  saved on every generate.
- **c6 flags** → flag stays off in production; M1.13 considers
  per-decode-step mask refresh OR per-output-token-batched dispatch
  as mitigations.

---

## 5. R4 — Async pipelining of CPU mask + GPU matmul

### Problem

Per-layer flow today alternates CPU mask work and GPU matmul:

```
mask_apply (CPU) → upload → matmul (GPU) → download → mask_unapply (CPU) → next offload …
```

CPU and GPU are mostly serial. At B=8 n=2040 Qwen3-4B prefill the
share split is ~50 % GPU matmul, ~26 % CPU mask apply+unapply. Full
overlap could reclaim ~25-30 % of prefill wall.

### Design — async engine API

Engine `matmul` and `matmul_many` return a `SubmissionFuture`
instead of blocking on completion. Substrate dispatches offload N's
matmul, then immediately starts mask_apply for offload N+1 on the
CPU while the GPU runs N. When the substrate needs N's output (for
unmask + next layer's input), it awaits the future.

Requirements:

- **Async API on `GpuOffloadEngine`** — new methods
  `matmul_async` / `matmul_many_async` returning
  `SubmissionFuture<Array2<f32>>`. Legacy blocking methods stay as
  default impls calling `.await` on the async path.
- **Per-handle operand buffer pool** — two slots so layer N's
  masked operand isn't clobbered by N+1's mask_apply while the GPU
  is still reading N. Mirrors R1.6's `stacked_scratch` reuse
  pattern at the engine layer.
- **Command-buffer ordering** — wgpu's queue submit returns a
  `SubmissionIndex`; the substrate awaits per-submission rather
  than per-operation.

### Scope

R4's substrate refactor is the heaviest item in M1.12. The wgpu
engine override needs to match the new async signature; the legacy
default path stays for non-wgpu backends. Substrate's
`offload_linear_per_sequence` / `offload_qkv_per_sequence` /
`offload_linear_many_per_sequence` all become async-shaped.

5-8 days of engineering, including:

- Async engine API design + wgpu override.
- Substrate refactor (per-call → submit + later-await).
- Per-handle operand pool with read-write hazard tracking.
- Functional parity tests (B=2/B=8 sweeps: outputs match pre-R4 to
  f32 floor).
- Perf measurement gate (see acceptance).

### Acceptance

Profile bucket re-measurement on the same fixture as the M1.11
D1.6 / D1.8 benches. R4 ships when:

- `engine:matmul` + `engine:matmul_many` wall remains within ±5 %
  of pre-R4 (the GPU work itself doesn't change).
- `gelo:mask_apply` + `gelo:mask_unapply` wall **overlaps with**
  the GPU bucket — measured via wall-clock delta vs. summed
  per-bucket time. Concretely: `(sum of buckets) > total wall`
  by ≥ 20 % of CPU mask wall.
- End-to-end prefill TTFT drops ≥ 20 % on Qwen3-4B B=8 n=2048.

If R4 only delivers 5-10 % (the limited-overlap floor), the
substrate refactor isn't paying for itself; revert + accept that
async pipelining wasn't justified at our shape.

---

## 6. Open questions for the implementation session

1. **Confirm `compute_logits` is actually ~120 s/forward on v7.**
   Add `profile::time("tee:compute_logits", …)` instrumentation at
   `gelo-embedder/src/decoder/generation.rs::compute_logits` BEFORE
   committing R3 engineering. If the bucket measures < 30 s/forward,
   R3's priority drops — async pipelining (R4) wins better.

2. **wgpu's `SubmissionFuture` semantics on RADV.** Some Vulkan
   drivers serialise submissions despite the async API. Spike-
   measure that R4's async dispatch actually overlaps CPU and GPU
   work on this hardware. If RADV serialises, R4 is dead on this
   substrate (still relevant for the dGPU path with proper async
   support).

3. **c6 capture infrastructure.** The existing AloePri snapshot
   capture in `evals/aloepri-attacks/` knows the M1.11 condition
   shapes (c1–c5). c6 adds a new shape. May need a new condition
   variant in the harness; ~half-day to wire.

---

## 7. Cross-backlog sequencing

```
1. M1.11 D2   — orchestrator rewire (1 day, headline 5× extraction)
2. M1.11 R1.4 — batched-attention GPU kernel (2-3 days, largest perf bucket)
3. M1.12 R1   — provision_decoder_into helper (1 day, cleanup + sets up R3)
4. M1.12 R3   — LM-head GPU offload behind flag (~3 days)
5. M1.12 c6   — AloePri spot-check (~3 days, gates LM-head default-flip)
6. M1.12 R4   — async pipelining (5-8 days, biggest remaining structural lever)
7. M1.11 D3   — c5 AloePri gate + crossover bench (3 days, default-flip)
```

Total: ~17-22 days end-to-end. M1.11 D2 + R1.4 land first because
they're outside M1.12 scope but smaller engineering with comparable
impact. R1.4 specifically retires the in-TEE attention bucket,
re-shapes the M1.12 profile baseline, and reduces the value of R4
slightly (less CPU work left to overlap) — but the structural
async-pipelining lever still matters at B=8+ shapes.

---

## 8. Out of scope

- **UMA-mapped masked-operand buffer.** iGPU-only artefact (the
  earlier R3 in the superseded "UMA residency overhaul" draft). On
  production dGPU the host↔VRAM transit is genuine PCIe DMA; UMA
  zero-copy doesn't translate. Dropped.
- **wgpu UMA allocator budget spike.** Dev-velocity only on this
  iGPU; zero production value. Dropped.
- **Encrypted-KV-on-GPU.** Parked in `future-rnd.md` §5.
- **Embedding_lookup on GPU** (input-side gather). The host-side
  row gather is already memory-bound at a few ms/forward; not a
  bottleneck. Could land in M1.13 as a tidy-up if the host-side
  778 MB `token_embedding` duplicate ever becomes binding.
- **Discrete-GPU bandwidth optimisations** (pinned host memory,
  CUDA-stream-equivalent async upload) — different substrate; needs
  the §M5.9 production-hardware bring-up first.
