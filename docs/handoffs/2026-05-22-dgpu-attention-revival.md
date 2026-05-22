# Handoff — 2026-05-22 — dGPU attention revival (M5.9 follow-ups)

Focus area for the M5.9 production-dGPU substrate bring-up. This
handoff collects the **attention-specific follow-ups that don't work
on iGPU but become primary levers on dGPU**, with the bandwidth
rationale tying each lever to the hardware shift.

Previously these items lived under §7 of the perf-bucket roadmap
handoff (`2026-05-22-perf-bucket-roadmap-r3-default.md`). Moved here
in their own document because the bucket-2 abort on iGPU
(2026-05-22) sharpened what's actually needed to make GPU attention
competitive, and the dGPU-specific items don't share a critical
path with the in-flight iGPU buckets 3/4.

## Why dGPU changes the picture (the binding number)

Strix Halo iGPU UMA vs production dGPU SEV-SNP + VFIO:

| | iGPU UMA (Strix Halo) | dGPU SEV-SNP + VFIO |
|---|---:|---:|
| Per-call upload bandwidth | DDR5 memcpy ~10 GB/s effective (shared bus) | PCIe 4.0 DMA ~30 GB/s realised |
| Kernel-side K/V read | DDR5 ~40 GB/s (shared with CPU) | HBM ~3 TB/s |
| **Ratio kernel-read / upload** | **4×** | **100×** |
| Effective DDR5 peak | 80 GB/s (one bus) | irrelevant — two separate buses |

The 16.4× iGPU bucket-2 abort (see
`docs/plans/m1-12-permuted-attention-batched-decode.md` §"Phase A
result") decomposed almost entirely into the **host-to-device upload
pipeline cost**: ~180 ms of 365 ms per call was f32→f16 conversion
+ memcpy of K, V tensors that hadn't changed since the previous
decode step. On UMA both the upload and the kernel-side read hit
the same DDR5 bus so persistent K/V saves bandwidth but the kernel
is still bandwidth-bound on the same bus (modest net win).

On dGPU the kernel-side read is 100× faster than the upload — the
upload pipeline becomes the structural bottleneck, and removing it
is no longer "modest" but "the whole game." That's why every item
in this handoff is a dGPU revival, not an iGPU fix.

## Item 1 — Persistent K/V on GPU

The core dGPU optimisation. Each decode call uploads only the new
`(K_t, V_t)` row instead of the full cache (32 KB vs 256 MB at
Qwen3-4B B=8 n_kv=2048 — **8000× redundancy ratio** today). The
catch: fixed-permutation K/V across many decode calls breaks Hidden
No More (ICML '25 paper recovers fixed permutations at 99 %+).
**Two cover designs** keep the bandwidth win while restoring the
freshness property:

### 1A — Block-level fresh π (refresh every N decode steps)

K cache persists on GPU under permutation π_block for N consecutive
decode steps; π refreshes every N. Within a block, fixed π. Across
blocks, fresh π → block-boundary re-upload of the K cache.

**Bandwidth win:** N× upload reduction. At N=16 → ~50 % of the
bucket-2 gap closed; at N=64 → ~95 %.

**Security gate:** σ analysis at fixed π for N consecutive
observations. HNM's σ=0.01 is calibrated for per-call fresh π; with
N correlated observations under fixed π the adversary's signal
grows √N, so σ must scale ≈ √N to preserve the same
attack-resistance threshold:

| N | σ needed | Note |
|---|---:|---|
| 8  | 0.028 | Likely well within model tolerance |
| 16 | 0.040 | Probably fine |
| 32 | 0.057 | Needs accuracy spike |
| 64 | 0.080 | At the edge — needs accuracy spike + AloePri gate |

**Engineering:** ~2-3 weeks substrate refactor. New
`GpuOffloadEngine` method for session-resident K/V (replaces the
per-call tensor argument with a session-handle that owns growing
K/V on device). Plus a new AloePri condition `c5_block_fresh_pi`
mirroring `c5_perm_attn_pad` methodology (see plan §7).

**Security spike:** ~1 week. Snapshot the dispatch shape at fixed π
for N ∈ {8, 16, 32, 64} consecutive calls; run anchor_ica / JADE /
JD / Gram-error against the snapshots; fit σ vs N curve. Same
methodology as bucket-2's c5_perm_attn_pad gate. Mirrors the C3
HD₃ gate precedent (`aloepri_hd3_gate_phase_a_b.md`).

**Status:** novel design (not in public literature). Security
argument needs to be written and gated empirically.

### 1B — TwinShield-Xue additive softmax-blinding cover

Published 2025 scheme (arXiv 2507.03278, Xue et al.). Uses
`e^(X+R)` blinding to push softmax to GPU under causal mask without
the Amulet identity. Disambiguated from "TwinShield (Liu '25)"
which our docs cite separately — different team, different
construction (Liu uses HE-softmax).

**Why considered:** the only published 2025 scheme that pushes
softmax to GPU under fresh per-row additive cover **without HE**.
If the security argument extends to HNM-class adversaries, would
let us persist K/V on GPU AND keep fresh per-row R per call.
Currently filed as P2 in `private_llm_inference_round_2` research
memo.

**Open questions (Q for the security spike):**

- Does Xue's adversary model match ours (GELO baseline + commodity
  GPU + per-batch fresh mask)?
- What rank of R works? Full-rank R has correction cost equal to
  the original attention (no perf win on the TEE side). Structured
  / low-rank R sacrifices security strength for correction speed.
- Does it compose with the GELO mask A on linear projections, or
  does it conflict with the per-batch fresh A topology?
- HNM-class adversary survival: TwinShield-Xue's threat model
  predates HNM; needs independent analysis.

**Engineering:** unknown until security spike characterises R's
rank requirements. Best case: ~1-2 weeks port from the paper.
Worst case: rank requirements make the TEE-side correction more
expensive than the attention being protected → no perf win.

**Security spike:** ~1-2 weeks. Read paper, audit threat model
against ours, run paper's attack suite + HNM attack at our shapes.

**Status:** published scheme; security needs validation specific
to our threat model. Higher engineering certainty than 1A (no
novel security argument needed), but unknown perf win.

### 1A vs 1B trade

| | 1A (block-level fresh π) | 1B (TwinShield-Xue additive) |
|---|---|---|
| Security argument | Novel — bespoke σ-vs-N analysis | Published — read + validate |
| Bandwidth win | Scales with N (predictable) | Unknown until R's rank settled |
| Engineering surface | Session-resident K/V API | Same + correction-tensor pipeline |
| Compose with M1.11 batched substrate | Direct | Needs analysis |
| Failure mode | σ exceeds model tolerance | R rank makes correction cost > original |

Recommended: **run both security spikes in parallel** (~1-2 weeks
each, independent). Whichever clears first defines the path.

## Item 2 — GQA-aware custom WGSL kernel

The current `engine.fused_attention_batched` takes
already-GQA-replicated K/V at shape `(B·num_q_heads, n_kv, d_head)`.
At Qwen3-4B with group=4 this is 4× more data than necessary —
each kv-head row is duplicated 4 times before upload.

**The win:** kernel takes K, V at the un-replicated shape `(B,
num_kv_heads, n_kv, d_head)` and broadcasts kv-head rows across q-
heads inside the shader. At Qwen3-4B group=4: **4× reduction in K/V
data motion** per call (256 MB → 64 MB at full upload, 32 KB → 8 KB
on the delta).

**Why this matters on dGPU specifically:**
- iGPU: K/V upload + kernel read both hit DDR5; the 4× reduction
  helps both proportionally but kernel was never the binding cost.
- **dGPU: upload is 100× slower than HBM read.** Eliminating 4× of
  the upload payload is a 4× saving on the binding bottleneck.

Composes cleanly with Item 1 (persistent K/V) — the kernel reads
GQA-expanded views from the un-replicated session-resident cache.

**Engineering:** ~2-3 weeks hand-rolled WGSL kernel with autotune
entries per shape. The kernel should fold the entire
`Q·Kᵀ → scale → mask → softmax → ·V` chain into a single dispatch
(see Item 3 below — it's the same kernel).

**Maintenance burden:** custom shader has to track burn-cubecl /
cubecl-wgpu API changes. Currently we ride burn's portability
guarantees; a hand-rolled kernel is wgpu-and-Vulkan-specific.
Acceptable on dGPU where the production substrate is fixed; less
acceptable on dev hardware.

**Risk:** custom kernel-tuning effort at scale not matched by
existing in-tree expertise. May need 1-2 weeks of perf-tuning
iterations after the functional version lands.

## Item 3 — Single-pass FlashAttention (FLASH-D)

Folds the matmul → scale → mask → softmax → matmul chain into one
GPU dispatch with online softmax (Tri Dao's FLASH-2/-3 pattern,
adapted to WGSL). Avoids materialising the `(B·H, n_q, n_kv)`
scores tensor in HBM.

**iGPU view (already measured 2026-05-22):** scores tensor at
decode-m=1 shape is only ~1 MB. Memory-bandwidth saving is
marginal. burn-cubecl-fusion already folds the `+ mask` add on our
chain (Q11 answered: <2 % delta with vs without mask). FLASH-D
would save kernel-launch overhead (5 dispatches → 1) but launch
isn't binding on iGPU. **Marginal win on iGPU.**

**dGPU view:**
- At decode-m=1: ~1 MB scores → marginal HBM saving. Win is mostly
  in fewer dispatches (PCIe round-trip per submission matters more
  on dGPU than iGPU; PCIe latency adds μs-scale overhead per
  submission).
- **At prefill (n_q=2048):** scores tensor is `(B·H, 2048, 2048)`
  f16 = ~4 GB. Materialising 4 GB in HBM and re-reading it for the
  second matmul is a structural cost. FLASH-D avoids it entirely.
  This is where the win is genuinely large — prefill attention on
  dGPU.

**Engineering:** subsumed in Item 2 above. The GQA-aware WGSL
kernel IS the FLASH-D kernel. Designing them separately would
double the implementation work; the right move is one kernel that
handles both.

**Reference implementations:**
- [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) — CUDA only; reference for the algorithm
- [`tracel-ai/cubecl`](https://github.com/tracel-ai/cubecl) `kernel::attention::flash_attention` — burn's CubeCL implementation, hardcoded `causal=true`. Could potentially upstream-PR to parameterise (see `m1-10-fused-permuted-attention.md` §6 "Option B")
- [`cubek-attention`](https://crates.io/crates/cubek-attention) v0.1.1 `Strategy::Unit` — works at prefill `n_q ≥ 32` (already verified on RDNA3.5), wastes lanes at decode-m=1 (irrelevant under Item 1's persistent K/V which keeps the bandwidth bottleneck away from the kernel-read side anyway)

## Recommended sequencing on M5.9 bring-up

### Step 0 — Bench triage (½ day)

**First thing to run when M5.9 dGPU substrate boots.** Re-run the
existing `amulet_attention_r1_4` bench cells from
`crates/gelo-gpu-wgpu/benches/amulet_attention.rs`:

```bash
cargo bench -p gelo-gpu-wgpu --bench amulet_attention -- amulet_attention_r1_4
```

Compare `gpu_batched_b8_no_mask` to `in_tee_rayon_b8` at
n_kv=2048. Three branch points:

| dGPU result | Action |
|---|---|
| `gpu_batched_b8` ≥ `in_tee_rayon_b8` even with current upload-heavy path | dGPU HBM advantage already swamps the upload tax. **Skip all substrate work — ship bucket 2 on dGPU as-is.** |
| `gpu_batched_b8` 2-4× slower than `in_tee_rayon_b8` | Upload pipeline is binding. **Invest in Item 1A first** (security spike) — most direct path to recovering the bandwidth. |
| `gpu_batched_b8` ≥ 5× slower than `in_tee_rayon_b8` | dGPU is also compute-bound. Less likely given HBM bandwidth, but if so → investigate `cubecl-hip` vs Vulkan, custom WGSL. |

### Step 1 — Item 1A security spike (parallel: ~1 week)

c5_block_fresh_pi gate methodology. σ-vs-N curve. New AloePri
condition. Independent of any engineering — pure security
analysis spike that can run in parallel with substrate work.

### Step 2 — Item 1B security spike (parallel: ~1-2 weeks)

TwinShield-Xue paper analysis + R-rank characterisation. Decides
whether 1B is a viable alternative to 1A.

### Step 3 — Item 1 engineering (whichever cover wins)

~2-3 weeks substrate refactor. Session-resident K/V API on
`GpuOffloadEngine`. New session handle, new bucket-2 wire-up in
`decoder_block_cached_batched`.

### Step 4 — Item 2 + 3 (combined): GQA-aware FlashAttention WGSL kernel

~2-3 weeks. Single kernel handles GQA broadcasting AND single-pass
flash attention. Lands after Item 1 because it consumes the
session-resident K/V interface.

### Step 5 — Bucket-2 re-acceptance bench

Re-run the M1.12 microbench (`qwen3_m1_12_r1_q1_microbench`) on
dGPU with all the above wired up. Apply the original §1
acceptance gate from
`docs/plans/m1-12-permuted-attention-batched-decode.md`:

- Decode wall ≥ 30 % reduction on top of R3
- No growth in mask offload count
- No growth in TEE↔GPU round-trip count beyond 1 per call

If passes → flip default; bucket 2 ships on dGPU.

## What about prefill attention?

Currently 11.6 % of prefill wall at B=8 — scoped out of bucket 2
(iGPU plan) because the F1+ chain would add a TEE↔GPU round-trip.
On dGPU:

- F1+ chain's round-trip cost is structurally different — PCIe DMA
  + HBM round-trip is faster than UMA DDR5 round-trip on iGPU.
- FlashAttention WGSL kernel (Item 2+3) handles prefill shapes
  cleanly — `n_q ≥ 32` is where cubek-attention `Strategy::Unit`
  already performs well on iGPU (1.24 ms at n_q=64 measured prior).
- Prefill's scores tensor materialisation (~4 GB at n_q=2048) makes
  FLASH-D's HBM win actually material on dGPU.

**On dGPU, prefill attention should be in scope** for the bucket-2
revival. Adjust the plan's §0 "scope" clause when M5.9 lands.

## Bandwidth math summary table

Per-call data motion at Qwen3-4B B=8 n_kv=2048 with each
optimisation layered:

| Configuration | K/V upload per call | Kernel reads K/V | Notes |
|---|---:|---:|---|
| Today's bucket-2 (aborted) | 256 MB | 128 MB (after f32→f16) | iGPU 16.4× slower than in-TEE |
| + Item 1 (persistent K/V) | 32 KB (delta) + 256 MB per block | same | Block-amortised |
| + Item 2 (GQA-aware kernel) | 8 KB (delta) + 64 MB per block | 32 MB | 4× reduction |
| + Item 3 (FLASH-D fused) | same | 32 MB | scores tensor stays in shared memory, not HBM |
| dGPU end-state at N=64 | 1 MB per step (amortised) | 32 MB at 3 TB/s = 0.01 ms | Compute-bound on HBM, not bandwidth-bound |

On dGPU, the end-state per-call cost is dominated by the **HBM
kernel reads** at ~0.01 ms — vs current 365 ms. Per call: ~36500×
faster than today's iGPU GPU path. Per call: ~2200× faster than
today's iGPU in-TEE path.

## Related artifacts

- `docs/plans/m1-12-permuted-attention-batched-decode.md` — full
  bucket-2 plan; §"Phase A result" has the iGPU abort retro that
  motivates this dGPU revival
- `docs/handoffs/2026-05-22-perf-bucket-roadmap-r3-default.md` —
  parent perf roadmap; §7 (production dGPU substrate bring-up M5.9)
  now points back to this handoff
- `docs/handoffs/2026-05-21-attn-offload-spike.md` — prior B=1
  spike on Strix Halo iGPU; provides the substrate-level shape data
- `~/.claude/projects/.../memory/bucket_2_batched_gpu_attention_aborted.md` —
  abort findings memory (load-bearing; explains why iGPU is dead)
- `~/.claude/projects/.../memory/private_llm_inference_round_2.md` —
  P2 entry for TwinShield-Xue softmax security analysis
- `~/.claude/projects/.../memory/gelo_research_round_2.md` — broader
  research context, Amulet + Hidden No More + SCX positioning
- `docs/plans/m1-10-security-review.md` — F1-F8 option survey;
  TwinShield-Xue is "F9-class" successor (not in that doc; needs to
  be added when 1B spike lands)
- `crates/gelo-gpu-wgpu/benches/amulet_attention.rs` — the bench
  cells (group `amulet_attention_r1_4/`) for Step 0 triage

## Suggested skills for the next session

- **`grill-me`** before committing to either Item 1A or 1B as the
  cover design — the σ-vs-N curve in 1A and the R-rank trade in 1B
  are both load-bearing security choices that deserve grilling
  before substrate engineering starts.
- **`diagnose`** if Step 0's bench result is unexpectedly bad on
  dGPU (e.g. PCIe DMA worse than projected, HBM utilisation low) —
  investigate substrate before assuming the bandwidth model.
- **`code-review`** before the session-resident K/V API change
  lands in `GpuOffloadEngine`. This is a load-bearing trait
  refactor that every downstream engine must implement.
- **`verify`** at the bucket-2 re-acceptance bench step — confirm
  the M1.12 microbench shows the projected decode-wall reduction
  before claiming the win.
