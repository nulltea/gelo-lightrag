---
type: handoff
status: current
created: 2026-05-22
updated: 2026-05-27
tags: [m1.12, perf]
---

# Handoff — 2026-05-22 — Perf-bucket roadmap after R3 + LM-head default-flip

Focus area for the next session: pick up the M1.12 perf roadmap at
**bucket B (batched GPU attention kernel)**. R1 + Q#1 + R3 landed
this session; LM-head GPU offload becomes the **default and only
path** (the optional `LM_HEAD_GPU_OFFLOAD` env knob is gone). A
**c6 AloePri condition variant** is wired structurally — the
attack-driver run is the security gate that retroactively validates
the default-flip and remains the only blocker between the current
state and "shipped".

## State on disk

Commits this session (master, ahead of origin/master by 26):

| Commit | What |
|---|---|
| `4686b8f` | R1 — `provision_decoder_into` helper across 3 production callers + 3 benches |
| `5f85f14` | Q#1 — `tee:compute_logits` profile bucket instrumentation |
| `b257042` | R3 — LM-head GPU masked offload behind `LM_HEAD_GPU_OFFLOAD=1` |
| `caf1601` | docs(gelo-llm): LM-head GPU offload § + alias scrub + plan / handoff |
| **(this session, post-handoff)** | LM-head offload default-and-only + c6 condition stub |

Plan + prior handoff (don't duplicate context — read these first):

- [`docs/plans/m1-12-tee-gpu-throughput.md`](../plans/m1-12-tee-gpu-throughput.md) — the M1.12 spec (R1 / R3 / R4 / c6 / Q#1 / Q#2)
- [`docs/handoffs/2026-05-22-q3-4b-b8-mask-sweep.md`](2026-05-22-q3-4b-b8-mask-sweep.md) — Qwen3-4B B=8 bottleneck triage that drove M1.12

Microbench reference (real numbers reproducible from these):

- `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`
  - default knobs: variant=Q4B, B=8, n=2048, K=64 (B=8 set this session per request)
  - override via `GELO_BENCH_VARIANT`, `GELO_BENCH_B`, `GELO_BENCH_N`, `GELO_BENCH_MAX_TOKENS`
  - two test fns: `m1_12_r1_q1_microbench` (R1 + Q#1 + R3 verdict at B=1) and `m1_12_per_op_breakdown_prefill_decode` (prefill/decode split, baseline vs R3)

Reproduce the headline B=8 numbers:

```bash
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 GELO_BENCH_MAX_TOKENS=64 \
  cargo test -p gelo-gpu-wgpu --release \
  --test qwen3_m1_12_r1_q1_microbench -- --ignored --nocapture \
  m1_12_per_op_breakdown_prefill_decode
```

## Headline measurement (Qwen3-4B, B=8, n=2048, K=64, this session)

```
Phase    Variant    wall (s)    aggregate tok/s   per-tok-per-seq (ms)
─────────────────────────────────────────────────────────────────────
Prefill  baseline   192.13       85.3              —
Prefill  R3         179.59       91.4              —
Decode   baseline   304.90        1.68            595
Decode   R3         112.99        4.53            221      (= 2.70× speedup vs baseline)

Δ decode wall under R3:  −62.9 %
Δ tee:compute_logits:    −97.6 %     (195 831 ms → 4 726 ms; residual is the profile::time wrapper)
token-prefix match:      64 / 64     on real Qwen3-4B weights
```

R3 multiplier grows with batch (1.82× at B=1 K=32, 2.0× at B=1 K=64,
**2.70× at B=8 K=64**) because the in-TEE LM-head loop scales linearly
in B while the GPU LM-head matmul amortises GPU dispatch across the
`1+k=16` shield-aligned rows for free.

Memory entry with full numbers: `m1_12_r1_q1_microbench_findings.md`
(auto-memory MEMORY.md index).

## Refined perf-bucket roadmap — execution order

The roadmap was reordered this session based on the measured per-bucket
shares at B=8. Skip the ones already done; pick up at item 2.

### 1. (DONE in this session) c6 attack-suite scaffolding + R3 default-on

c6 condition variant is wired in `evals/aloepri-attacks/`. **The
attack-driver run is the remaining gate.** Engineering complete; need
to capture snapshots at the new LM-head shape and run the recovery
attacks against c2 baseline:

```bash
cargo run --release -p aloepri-attack-snapshot-runner -- \
    --condition c6 --max-prompts 64 --max-tokens 1 --engine gpu_fp16 \
    --output snapshots/qwen3-1.7b
# then the Python attack-driver run (anchor_ica / jade / jd / gram_error)
# against the c6 condition output vs the c2 baseline. Same comparison
# methodology as the c5 gate. ~3 days.
```

Acceptance: attack accuracy on c6 within sample-noise of c2. If c6
flags, revert the default and re-design (LM-head shape is 37× wider
than QKV — known to scale recovery surface).

### 2. ~~Batched GPU attention kernel (R1.4 lever)~~ — **ABORTED 2026-05-22 at Phase A spike**

Plan was written at `docs/plans/m1-12-permuted-attention-batched-decode.md`
(11-question grilled design); Phase A crossover spike measurement
landed and **failed the gate by 16×**.

**Measured** at Qwen3-4B GQA shape (B=8, num_heads=32, num_kv_heads=8,
d_head=128) on Strix Halo iGPU (Radeon 8060S / RADV gfx1151 via wgpu
Vulkan f16):

| Shape (B=8) | in_tee_rayon_b8 | gpu_batched_b8 (burn-chain f16) | GPU vs in-TEE |
|---|---:|---:|---:|
| n_kv = 256  |  1.06 ms |  48.5 ms | 45.9× slower |
| n_kv = 1024 |  7.13 ms | 186.2 ms | 26.1× slower |
| n_kv = 2048 | 22.3 ms  | 364.8 ms | **16.4× slower** |

The acceptance gate required ≥ 1.5× faster GPU; result was 0.06×.

**Why the M1.11 "crossover at B 11–16" hypothesis was wrong**: the
22 ms at B=1 from the prior attn-offload-spike was already
compute-bound, not launch-dominated. Batching scales GPU compute
linearly too, so the gap doesn't close. Side-finding (Q11):
burn-cubecl-fusion folds the `+ mask` add — `with_mask` vs
`no_mask` delta is <2 % — so a custom WGSL FlashAttention-D kernel
wouldn't help either (mask-elision is already free; remaining gap
is compute throughput, not memory bandwidth: scores tensor is only
~1 MB at decode-m=1 shape).

**Don't re-spike on iGPU.** The bench cells stay in
`crates/gelo-gpu-wgpu/benches/amulet_attention.rs` (group
`amulet_attention_r1_4/`) as a re-runnable comparison harness — any
future bucket-2 revival (custom WGSL, cubecl-hip backend swap,
dGPU substrate) must beat the same `in_tee_rayon_b8` baseline.

Full retro in plan §"Phase A result" + memory
`bucket_2_batched_gpu_attention_aborted.md`.

**Next priority shifts down**: bucket 3 (bf16/f16-native activation
pipeline — 3a narrow OpenBLAS sbgemm mask path for fast prefill
win, 3b broader end-to-end activation-storage rework that closes
the upload-pipeline tax and unlocks any future dGPU revival of
bucket 2) or bucket 4 (R4 async pipelining, blocked on Q#2
RADV-async spike). Recommended order: bucket 3a (1 day) → Q#2
spike (½ day) → decide between bucket 3b and bucket 4 based on
the Q#2 result.

### 3. bf16/f16-native activation pipeline (mask GEMM + storage) — bandwidth-contention lever

Two layers under one bucket: the **narrow** prefill mask-GEMM win
plus the **broader** end-to-end activation-storage rework that
composes with it (and would have been bucket 2's enabler had bucket
2 not aborted on iGPU per §2 above). Both share the same root cause
— f32 activations + per-boundary f32↔f16/bf16 conversions hammer
DDR5 on the same UMA bus the GPU matmul uses — and both rely on the
same Zen 5 AVX-512_FP16 / `_BF16` instruction surface to land.

#### 3a — bf16 mask GEMM (the narrow, ship-fast variant)

Current bottleneck (B=8 prefill, n=2048): `gelo:mask_unapply` 24.5 %
(45 s) + `gelo:mask_apply` 14.9 % (27 s) — **39 % of prefill wall on
CPU DDR5**, contending with GPU matmul on the same UMA bus.

**Engineering**: 1–2 weeks. Two viable CPU-side paths — **mask
GEMM cannot move to GPU without violating the GELO threat model**
(would expose `A` on the device, defeating the protocol; same
argument as on-GPU unmask in `inference-optimization.md` §2.2.1).
The choice is between OpenBLAS `cblas_sbgemm` (1-day, pulls in the
BLAS dep alongside AOCL-BLIS) and a hand-rolled AVX-512_BF16
kernel (1–2 weeks, no new dep, matches the existing
`tee_matmul_bf16` precedent in the crate). The
`bf16_mask_gemm_skipped` memory's "~10 % TTFT, gain shrinks after
HD₃" estimate was at B=1 — at B=8 the bucket scales linearly so the
share **and** the win are larger (~25–30 %).

**Impact estimate**: ~25–30 % prefill wall reduction on iGPU UMA via
unbussing the CPU-side FWHT bandwidth. On dGPU PCIe the win is
structural in a different way — frees CPU thread occupancy and
removes the FWHT memory-bandwidth ceiling on the host.

**Scope:** only the mask path (`gelo:mask_apply` /
`gelo:mask_unapply`). The masked-operand output of mask_apply is
still f32; the GPU upload still pays the f32→f16 host-side
conversion. That residual cost is what 3b addresses.

#### 3b — bf16/f16-native activation storage end-to-end

The wider rework: keep activations as `Array2<bf16>` (or `f16`)
throughout `gelo-embedder/src/decoder/forward.rs` rather than f32.
Weights are already bf16-native on host (post-loader work); the
activations are the remaining f32 occupant of the forward-pass
working set.

**What it changes:**
- Activation tensors in the forward pass become bf16/f16. Every
  per-layer `Array2<f32>` (`h`, `h_norm`, residuals, attention
  context, FFN gate/up/down outputs, etc.) downsized 2×.
- `mask_apply` / `mask_unapply` consume bf16 in, produce bf16 out
  (natural fit with 3a's bf16 GEMM kernel — they compose; no
  intermediate widening).
- GPU upload path is bf16/f16-native — eliminates the host-side
  f32→f16 conversion + one DDR5 traverse per offload (per the
  bucket-2 spike post-mortem, the upload was paying
  ~1.5 GB DDR5 traffic per call at decode-attention shape; ~½ GB
  of that was the conversion itself).
- TEE matmul path (`tee_matmul_bf16`) is already bf16-aware; no
  change needed there.
- `apply_qk_norm`, RMSNorm, RoPE, residual adds — each needs a
  bf16-aware variant or a temporary widening at the kernel
  boundary. AVX-512_FP16 (Zen 5) handles f16 fmla natively; bf16
  needs explicit widening on the FMA loop (Zen 5 also has
  AVX-512_BF16 for the GEMM cores but not for arbitrary ops).

**Engineering**: ~2-3 weeks. Larger than 3a because it touches
every forward-pass tensor, every elementwise kernel, the precision
contract on the `GpuOffloadEngine` boundary, and the parity tests
that assert f32-floor agreement (which need re-baselined to bf16
precision).

**Impact estimate**:
- Prefill: composes with 3a — adds another ~5-10 % on top by
  eliminating the upload-side conversion (small per call but
  many calls).
- Decode: was bucket-2's enabler on iGPU. With bucket 2 deferred,
  the decode-side win is bounded by the residual `gelo:mask_apply`
  / `_unapply` cost at decode shapes (~5 % of decode wall today).
- **dGPU revival of bucket 2 requires 3b** as a prerequisite — the
  ~10× upload-pipeline tax in the bucket-2 abort post-mortem
  doesn't shrink on PCIe, it gets worse. Any future dGPU-side
  attention offload must consume bf16/f16-native activations end-
  to-end or it'll repeat the iGPU failure mode on a different
  bottleneck mix.
- Satisfies the `feedback_memory_efficiency_priority` "never
  upcast bf16 → f32" rule more thoroughly than today's path,
  which holds f32 activations transiently between offloads.

**Order interaction with 3a:** start with 3a as the 1-day OpenBLAS
spike to confirm the bf16 GEMM kernel exists and the mask path
parity holds at bf16. Then 3b builds on it incrementally —
forward-pass tensor conversions are most of the work, and 3a
proves the precision contract before we commit to it across the
whole forward pass.

#### Order interaction with bucket 4 (R4)

Shipping 3a first collapses R4's payoff (no CPU mask bucket to
overlap with). Decide order based on the dGPU timeline — see
bucket 4.

3b is largely orthogonal to R4 — it reduces the bytes the CPU mask
path moves rather than overlapping the moves with GPU work. If R4
turns out dead on iGPU (Q#2 says RADV serialises), 3b becomes the
only path to recover the prefill bucket structurally.

### 4. Async pipelining (M1.12 R4) — DECIDE BEFORE STARTING

> **Resolved 2026-05-27.** R4 implemented end-to-end on
> `feat/r4-async-overlap`, measured at production-like shape, and
> failed: flat-to-+3 % regression on iGPU. The Q#2 spike's 58 %
> overlap was a synthetic-harness artefact — the real forward
> pass has a strict apply→matmul→unapply→apply serial dependency
> with nothing to hide behind matmul-in-flight; the async
> dispatch path adds ~2 ms × ~56 matmuls/forward of pure
> bookkeeping. Cutover cancelled, master holds no R4 artefacts
> (commit `2381fd4`). Substrate retained on the feat branch as
> a dGPU-revival precondition (PCIe DMA + GPU compute are
> physically separate, so the overlap reappears) and for
> cross-prompt batching. Full retro: memory
> `r4-async-igpu-outcome`; roadmap §4.D in
> `docs/plans/gelo-llm-perf-roadmap.md`.
>
> The discussion below is preserved for historical context on
> the decision rule that led into the test.

Plan-estimated 25–30 % prefill wall via overlapping CPU mask (layer
N+1) with GPU matmul (layer N).

**On iGPU UMA**: best case ~15 %. CPU and GPU share the same DDR5
bus; overlap doesn't reduce total bytes moved.

**On production dGPU PCIe**: genuinely valuable — PCIe DMA + GPU
matmul are physically separate from CPU mask FWHT.

**Open Q#2 (gate before starting)**: does RADV actually overlap wgpu
submissions, or serialise them under the queue? Spike-measure first
(half day) — if it serialises, R4 is dead on this iGPU. Plan calls
this out explicitly.

**Order decision rule (preserve in the plan)**:

- If production dGPU is < 3 months away → **skip R4 on iGPU**; revisit
  on dGPU. Save 5–8 days of substrate refactor that returns 15 %
  marginal at best on iGPU.
- If dGPU is the long tail (≥ 6 months) → **ship R4 on iGPU** for
  dev-velocity. Q#2 spike first; commit to R4 only if RADV overlaps.

### 5. Varlen / chunked / continuous batching

Per-sequence orchestration improvements. **Zero win in the current
bench** (identical-length prompts). In production extraction with
variable chunk lengths: ~10–30 % per-prompt wall.

**Components**:
- **Varlen mask** — skip pad rows in mask FWHT + GPU matmul when
  prompts have different lengths. Substrate API change (`(B, n_b, d)`
  ragged instead of `(B, n_max, d)` stacked). ~1–2 weeks.
- **Chunked prefill** — interleave chunks of one prompt with decode of
  others. Orchestrator-level. Latency-shaping, not throughput.
- **Continuous batching** — replace finished sequences without
  draining. Throughput-oriented; helps end-to-end serving latency.

**Order**: parallel track, ideally picked up by whichever crate-team
owns LightRAG orchestrator (`crates/lightrag-private/`). Doesn't move
the bench but is the right production lever.

### 6. Slalom-additive hybrid for linear projections — R&D milestone

Per `private_llm_inference_round_3` research memo. Splits each masked
offload into a plaintext-public matmul + an additive secret
correction; the public matmul rides any commodity-optimised inference
kernel.

**Potential**: 40–60 % wall reduction at prefill AND decode if it works.

**Engineering**: multi-week protocol-level redesign + AloePri-class
attack-suite re-validation (the additive correction shape isn't
covered by existing GELO §3.2 bounds).

**Order**: separate milestone, attack-validation gated. Highest
ceiling, lowest confidence. Don't put on the M1.12-extension critical
path; pre-spike via Python sim first.

### 7. Production dGPU substrate bring-up (M5.9) — see separate handoff

The dGPU-specific attention follow-ups (persistent K/V on GPU,
GQA-aware custom WGSL kernel, single-pass FlashAttention) moved
into their own handoff so they don't share a critical path with
the in-flight iGPU buckets 3/4. The substrate-bring-up overview +
bandwidth model + bucket-2 revival plan + recommended sequencing
on M5.9 boot all live in:

→ [`2026-05-22-dgpu-attention-revival.md`](2026-05-22-dgpu-attention-revival.md)

**TL;DR rationale:** Strix Halo iGPU UMA is the architectural
ceiling for buckets 2 / 3b / 4. SEV-SNP + VFIO discrete GPU lifts
it — HBM ~3 TB/s kernel-read vs PCIe ~30 GB/s upload (**100× ratio
vs iGPU UMA's 4×**), making "persistent K/V on GPU" go from
modest-win on iGPU to primary-lever on dGPU. The bucket-2 abort
on iGPU was a "right answer, wrong hardware" — re-measure on dGPU
when M5.9 lands. Step 0 of the new handoff is a half-day re-run
of the `amulet_attention_r1_4/` bench cells that gates everything
downstream.

**Order**: hand off to M5.9 hardware bring-up. Re-run the microbench
+ the per-op breakdown bench there before re-prioritising — many
"iGPU-only" wins will look different, and several "iGPU-blocked"
levers (textbook batched-prefill scaling, async pipelining) become
genuinely large.

## Suggested skills for the next session

- **`code-review`** before the bucket-2 (batched GPU attention) PR
  lands — substrate parity is load-bearing for the M1.12 win to ride
  to production.
- **`verify`** after each bucket lands — confirm the
  per-op-breakdown microbench shows the expected bucket reduction
  before claiming the win.
- **`grill-with-docs`** before flipping default-on after bucket 2
  (the in-TEE attention is the last "GELO §3.2 sensitive-layer
  exclusion" defence; lifting it to GPU under permutation is a
  threat-model change worth grilling).
- **`diagnose`** if Q#2 (RADV serialisation spike) goes against R4.

## Memory tail (load-bearing for the next session)

Auto-memory entries created or updated this session:

- `m1_12_r1_q1_microbench_findings.md` — R1 needs `malloc_trim(0)` to
  show RSS drop; `tee:compute_logits` is 46–58 % of decode wall on
  real weights; R3 lands −45 % wall at B=1 / −63 % wall at B=8.
- (existing) `hd3_mask_landed.md`, `qwen3_4b_perf_2026_05_20.md`,
  `private_llm_inference_round_3.md` — round-3 perf research.

Pinned conventions worth re-reading:

- `CLAUDE.md` §"HTML docs" — public docs (gelo-llm.html etc.) avoid
  plan-time aliases (M1.11, Option A, v1, etc.). Re-baseline any new
  HTML section through that lens.
- `feedback_html_docs_design_not_code.md` — HTML lead with design +
  measured result, not dev-log.

## What I'd start with on day 1

1. Read `docs/plans/m1-12-tee-gpu-throughput.md` (the M1.12 spec).
2. Run the c6 capture + attack-driver (item 1's remaining gate).
   Either passes and the R3 default is retroactively validated, or it
   flags and the engineering is reverted.
3. Concurrently / next: bucket 2 (batched GPU attention kernel). Look
   at `gelo_protocol::engine::fused_attention_batched` for the
   existing entry point; check what's missing in
   `decoder_block_batched` / `decoder_block_cached_batched`.
