# GELO-LLM performance roadmap

> **Workload anchor.** Qwen3-4B GELO inference on Strix Halo iGPU
> (Radeon 8060S, fp16 wgpu Vulkan engine, R3 LM-head GPU offload
> default-on). dGPU track points to
> [`2026-05-22-dgpu-attention-revival.md`](../handoffs/2026-05-22-dgpu-attention-revival.md)
> for substrate details; this doc only orders dGPU levers by
> bucket and EV.
>
> **Status.** Post-measurement (2026-05-26 sweep), post-Auto-tune.
> Authored 2026-05-26; reorganized 2026-05-26 around bucket-by-
> bottleneck attribution after the prior rewrite over-elevated
> DCT-IV column-locality based on a single measured shape.
>
> **Convention.** Levers are grouped by the bottleneck they
> reduce (§4.A–§4.F). Each bucket has an iGPU subsection and a
> dGPU subsection; cross-cutting levers (R4, bf16) get their own
> bucket. Engineering items that don't reduce a bucket (variance
> sweeps, memory residency, instrumentation gaps) live in §3
> ahead of the buckets — they gate or unlock the bucket work.

## TL;DR

1. **Current state at Qwen3-4B B=8 n=2048 K=32** (production
   long-n extraction shape, post-DCT-IV-cascade 2026-05-26):
   **121.3 tok/s prefill aggregate** (was 93.7 pre-cascade,
   **+29 %**), 4.62 tok/s decode aggregate (0.58 per-seq).
   Post-cascade prefill wall is CPU mask 20.4 % (down from
   38.0 %) / GPU matmul ~52 % / in-TEE attention ~16 %. Decode
   wall is unchanged: in-TEE attention 53.9 % / GPU matmul
   ~38 % / CPU mask ~4 %.

2. **iGPU ceiling is ~5-7 tok/s decode (B=1).** In-TEE attention
   is DDR5-bandwidth-bound and structurally untouchable on iGPU
   — bucket-2 spike confirmed 16× regression at the production
   decode shape (§4.C.1). 40+ tok/s decode is dGPU-only and lives
   under §4.B / §4.C dGPU subsections.

3. **Top-lever priorities updated 2026-05-26.** §4.A.1 DCT-IV
   column-locality cascade ✅ **shipped** — measured **−22 %
   prefill wall** at production shape (2.3× the original
   estimate). §3.2 #1 R1 weight Arc drop ✅ **shipped** (5.28 GiB
   measured RSS reclaim). §3.2 #2 UMA allocator unblock ✅
   **resolved as a non-issue** — B=16 runs clean but doesn't
   amortise prefill at long-n. §4.E bf16 activation pipeline ❌
   **deprioritised 2026-05-26** — standalone cascade microbench
   showed DCT-IV bf16 wins only +8 % (projects ~1.6 % wall, below
   variance) and HD₃ bf16 regresses 2× (current bulk widen-narrow
   impl). The post-cascade-refactor L2-resident tile design
   already captured the bandwidth gains bf16 was meant to deliver.
   Remaining iGPU work in EV order:
   §3.1 #2 variance sweep (calibrates everything below),
   Q#2 RADV-async spike → §4.D R4 async overlap (the next
   ~15 % wall lever, if Q#2 clears), and dGPU substrate prep
   (§4.B.2 / §4.C.2). Phase 1-3a infrastructure remains useful
   for dGPU revival where bf16-native compute kernels change
   the math.

---

## §1 Current state

Numbers below are from
`bench-results/m1-12-hd3-perf-sweep-2026-05-26_07-04-58.{log,tsv}`
(initial 14-cell sweep), the post-tune verify sweep
`bench-results/m1-12-auto-tune-verify-2026-05-26_08-42-00.{log,tsv}`,
and the measurement-gap sweep
`bench-results/measurement-gaps-2026-05-26_10-34-30.{log,tsv}` /
`bench-results/measurement-gaps-cell4-rerun2-2026-05-26_11-00-58.{log,tsv}`
(closed §3.1 #1, #3, #4 below). All runs: Qwen3-4B, fp16 wgpu
Vulkan engine, R3 LM-head GPU offload on, K=32 decode tokens
per cell.

### §1.1 Headline throughput (Qwen3-4B, K=32)

Five shapes spanning the Auto-family decision space, ordered
by pad ratio:

| B | n | pad ratio | Auto family | prefill wall (s) | prefill tps agg | decode wall (s) | decode tps agg | decode tps/seq |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 3500 | 1.17 | HD₃ | 287.87 | 97.3 | 82.11 | 3.12 | 0.39 |
| 8 | 320 | 1.56 | HD₃ | 24.22 | 105.7 | 26.82 | 9.55 | 1.19 |
| 1 | 2561 | 1.59 | HD₃ | 31.45 | 81.4 | 22.28 | 1.44 | 1.44 |
| 8 | 2400 | 1.70 | DCT-IV | 216.15 | 88.8 | 61.40 | 4.17 | 0.52 |
| 8 | 2048 | 1.99 | DCT-IV | 174.92 | 93.7 | 55.08 | 4.65 | 0.58 |

Single sample per cell; read against the §1.5 variance floor
(≥ ~7 % at long-n).

For HD₃-forced (counterfactual at DCT-IV-picking shapes):

| B | n | pad ratio | mask=hd3 | prefill wall (s) | decode wall (s) | Δ prefill vs Auto |
|---:|---:|---:|---|---:|---:|---:|
| 8 | 2400 | 1.70 | HD₃ (pad 4096) | 264.46 | 132.96 | **+22.4 %** |
| 8 | 2048 | 1.99 | HD₃ (pad 4096) | 222.35 | 55.80 | +27.1 % |
| 1 | 2561 | 1.59 | HD₃ (pad 4096) | 32.19 | 21.77 | +0.9 % |
| 8 | 320 | 1.56 | HD₃ (pad 512) | 24.42 | 27.11 | +0.8 % |

### §1.2 Prefill bucket breakdown (% of wall)

Per-op shares for the five §1.1 cells, ordered by pad ratio.
`mask_unapply` is ~1.75× `mask_apply` by call count (252:144 per
prefill — every QKV / many-output offload pays one apply +
multiple unapplies).

| B | n | pad | family | wall (s) | mask_apply % | mask_unapply % | mask total % | engine: matmul % | engine: matmul_many % | matmul total % | tee:attn % | shield+strip % |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3500 | 1.17 | HD₃ | 287.87 | 5.2 | 10.7 | **15.9** | 19.1 | 29.1 | **48.2** | **26.5** | 1.2 |
| 8 | 320 | 1.56 | HD₃ | 24.22 | 4.1 | 9.2 | 13.3 | 28.5 | 44.4 | 72.9 | 2.9 | 1.5 |
| 1 | 2561 | 1.59 | HD₃ | 31.45 | 6.6 | 11.5 | 18.1 | 21.0 | 33.5 | **54.5** | **12.5** | 7.0 |
| 8 | 2400 | 1.70 | **DCT-IV** | 216.15 | **14.7** | **26.6** | **41.3** | 15.0 | 22.9 | 37.9 | 12.9 | 1.1 |
| 8 | 2048 | 1.99 | **DCT-IV** | **174.92** | **13.5** | **24.5** | **38.0** | 16.0 | 24.5 | 40.5 | 12.6 | 1.1 |

**Two structural pictures emerge with the family split:**

- **DCT-IV shapes (B=8 n=2048-2400, pad 1.7-1.99)**: mask
  bucket is 38-41 % of prefill, GPU matmul 38-41 %, in-TEE
  attention ~13 %. **CPU mask is the single biggest lever
  here.**
- **HD₃ shapes at long n (B=8 n=3500, pad 1.17)**: mask
  shrinks to 16 %, GPU matmul rises to 48 %, **in-TEE attention
  jumps to 27 %.** Mask is no longer dominant — GPU matmul
  takes over (bandwidth-bound) and attention grows quadratically.
- **HD₃ shapes at short n / B=1**: mask is 13-18 %, dominated
  by GPU matmul (54-73 %).

For reference (HD₃-forced at shapes Auto picks DCT-IV — confirms
why Auto's choice is right at pad ratio > 1.6):

| B | n | pad | family | wall (s) | mask total % | engine matmul total % |
|---:|---:|---:|---|---:|---:|---:|
| 8 | 2048 | 1.99 | HD₃ pad 4096 | 222.35 | 19.1 | **62.8** |
| 8 | 2400 | 1.70 | HD₃ pad 4096 | 264.46 | 18.9 | **61.2** |

Forcing HD₃ at pad > 1.6 shifts wall from mask into GPU matmul
(pad penalty) and pays +22-27 % prefill. Cell 3 at pad 1.70
confirms the 1.6 Auto threshold is well-calibrated — the gap
between HD₃ and DCT-IV is meaningful 100 bps below pad 1.99.

### §1.3 Decode bucket breakdown (% of wall)

Decode runs HD₃ at stacked_n=16 in every cell (k=15 overlay
forces pow2). Mask is small at decode; the differentiator is
in-TEE attention scaling with context length. B=1 cells use the
singular `tee:attn_cached` bucket; B≥2 cells use
`tee:attn_cached_inplace_many` (§1.5 caveat).

| B | n | wall (s) | mask total % | engine matmul total % | tee:attn (cached / cached_inplace_many) % | compute_logits % ⁽ᶜ⁾ | shield+strip % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 320 | 26.82 | 9.1 | 72.7 | 10.2 | 8.4 | 5.9 |
| 1 | 2561 | 22.28 | 7.9 | 25.3 | **63.0** | 1.5 | 3.3 |
| 8 | 2400 | 61.40 | 3.7 | 33.9 | **58.7** | 3.8 | 2.7 |
| 8 | 3500 | 82.11 | 3.0 | 27.5 | **66.4** | 2.8 | 2.2 |
| 8 | 2048 | 55.08 | 4.2 | 37.7 | **53.9** | 4.2 | 3.0 |

In-TEE attention dominates decode at every long-context cell:
**53.9 % (n=2048) → 58.7 % (n=2400) → 63.0 % (B=1 n=2561) →
66.4 % (n=3500)**. The §2.2 ceiling thesis confirmed at every
shape we measured — and B=1 decode (63 %) is **even more
attention-dominated than B=8** (no batch amortisation of the
per-step in-TEE GQA kernel). At B=8 short-n (n=320) the
attention bucket is only 10.2 % because GPU matmul takes over
(72.7 %).

### §1.4 Auto family resolution (post-tune 2026-05-26)

| pad ratio | Auto picks | Why |
|---:|---|---|
| ≤ 1.6 | HD₃ | confirmed faster up to 1.59 in sweep |
| > 1.6 | DCT-IV | HD₃ pays GPU-pad penalty; at 1.99 the penalty is 19 % |
| (1.59, 1.99) | crossover region | empirical bound; one-sample cells either side |

Sweep evidence behind the boundary (post measurement-gap sweep):

| sweep cell | pad ratio | family that wins (lower wall) | margin |
|---|---:|---|---:|
| B=8 n=3500 | 1.17 | HD₃ (Auto-picked) | n/a (only HD₃ tested) |
| B=8 n=320 | 1.56 | HD₃ | 0.8 % |
| B=1 n=2561 | 1.59 | HD₃ | 0.9 % |
| B=8 n=2400 | 1.70 | **DCT-IV** | **22.4 %** ← §3.1 #4 resolved |
| B=8 n=2048 | 1.99 | DCT-IV | 27.1 % over HD₃-pad |

Pre-tune Auto threshold was 7/5 = 1.4 — under-picked HD₃ at
the 1.56-1.59 band. Post-tune (8/5 = 1.6) covers that band, and
the n=2400 cell confirms DCT-IV wins decisively at pad 1.70 —
the threshold is well-calibrated. No remaining crossover gap.

### §1.5 Measurement caveats

- **B=1 attention bucket — resolved.** The B=1 attention path is
  `decoder_block_cached` (not `decoder_block`), so B=1 emits
  `tee:attn_cached` / `tee:attn_permuted_cached` /
  `tee:attn_swa_cached` — **not** `tee:attn_inplace` / `_cached`
  as the prior caveat claimed. `dump_sweep_buckets` now lists
  all three B=1 variants alongside the `_many` variants for B≥2.
  The §1.3 B=1 n=2561 decode share of 63.0 % is the captured
  `tee:attn_cached` bucket.
- **⁽ᵇ⁾ Other** = `wall − (sum of listed buckets)`. Captures
  unlabeled time (qk_norm, rmsnorm, swiglu, kv scatter, layer
  dispatch overhead).
- **⁽ᶜ⁾ `tee:compute_logits` overlaps `engine:matmul`** because
  R3 routes the LM-head through `offload_linear`, whose
  `engine:matmul` span nests inside the `compute_logits` span.
  Sum % can exceed 100 by this margin (~70 ms per decode step).
  Decode-only; not present on prefill rows.
- **Single-cell run-to-run variance at long-n is ≥ ~7 %.** A B=8
  n=2048 cell run ~90 min apart on the same box with identical
  code measured **187.31 s vs 174.92 s** wall — a 6.6 % spread
  with no code change. Any single-cell ms claim should be read
  against this band. The variance-sweep run is §3.1 #2.

---

## §2 Ambition

### §2.1 Target

| Workload | Current | Target | Headroom |
|---|---:|---:|---:|
| **B=1 decode** (user-perceived t/s) | **3.6 tok/s** ⁽ᵈ⁾ | **40+ tok/s** | **11×** |
| **B=8 aggregate decode** | 4.58 tok/s (§1.1) | 40+ tok/s | 9× |
| **B=1 prefill** | ~100 tok/s | — | not the binding metric |

⁽ᵈ⁾ B=1 decode number from the pre-sweep R3-default baseline
(`m1_12_r1_q1_microbench_findings.md`); the 2026-05-26 sweep
was B=8-focused, so B=1 decode wasn't re-measured. Per-seq
decode at B=8 n=2048 is 0.58 tok/s — substantially worse than
B=1 because B=8 dispatches B parallel chunks of in-TEE attention
that share the DDR5 bus.

**The iGPU all-levers stack ceiling is ~5-7 tok/s at B=1
decode** (~10-15 tok/s aggregate at B=8). 40+ tok/s requires
**dGPU substrate + persistent K/V on GPU + Q4 weight
quantization + iGPU-track substrate prerequisites first.**

### §2.2 Why iGPU ceilings out at ~5-7 tok/s

The bucket-2 iGPU spike on 2026-05-22 measured GPU 16.4× slower
than in-TEE-rayon at the Qwen3-4B decode shape. Decomposition:

- Strix Halo iGPU compute is **DDR5-bandwidth-bound** at decode
  shapes (the same bus the CPU mask GEMM uses).
- 4-touch upload pipeline (host f32 → host f16 → wgpu staging
  → GPU read) costs ~180 ms of 365 ms per call.
- The kernel itself isn't slow; the bus is saturated and shared.

Consequence: at decode, **~50 % of B=8 decode wall is in-TEE
attention** (54.8 % confirmed at B=8 n=2048 in §1.3) and it is
structurally untouchable on iGPU. Bucket-2 proved that putting
attention on iGPU just adds 16× cost to already-DDR5-bound work.

At B=1 the in-TEE attention is roughly the same per-sequence
cost (no batch amortization), so the same ceiling applies.
Killing the other ~45 % of decode wall (mask, GPU matmul, other
TEE ops) takes us from 3.6 → ~7 tok/s, then stops.

### §2.3 Why dGPU lifts the ceiling 5-10×

From `2026-05-22-dgpu-attention-revival.md` §0:

| | iGPU UMA (Strix Halo) | dGPU SEV-SNP + VFIO |
|---|---:|---:|
| Per-call upload bandwidth | DDR5 memcpy ~10 GB/s | PCIe 4.0 DMA ~30 GB/s |
| **Kernel-side K/V read** | DDR5 ~40 GB/s (shared) | **HBM ~3 TB/s** |
| Ratio kernel/upload | 4× | **100×** |

On dGPU the kernel itself is ~75× faster (HBM vs DDR5 for the
read-bound attention kernel) AND persistent K/V becomes
worthwhile (only the new `K_t, V_t` row uploaded per step =
32 KB instead of 256 MB). Combined with persistent K/V and
bf16-native activations, all the GPU buckets compress
dramatically.

Compute-wise, dGPU lifts GPU matmul from DDR5-bound (~80 GB/s)
to HBM-bound (~3 TB/s) — **~40× bandwidth ceiling** for QKV / O /
gate-up / down dispatches.

---

## §3 Measurement & substrate support

These items don't reduce a bucket on their own — they unblock
or gate the bucket work below. The 2026-05-26 measurement-gaps
sweep resolved §3.1 #1, #3, #4 (see below); the variance sweep
(#2) remains pending and gates the absolute-EV claims in §4.

### §3.1 Measurement gaps

| # | Item | Status | Finding |
|---:|---|---|---|
| 1 | **Long-n HD₃ sweep cell** | ✅ **resolved** 2026-05-26 (`measurement-gaps-2026-05-26_10-34-30`) | B=8 n=3500 pad 1.17: mask 16 % / matmul 48 % / attn 27 % prefill, attn 66 % decode. **At long-n HD₃ shapes the mask bucket is less than half the DCT-IV-shape size** — DCT-IV-locality (§4.A.1) is no longer the universal top lever; substrate prereqs become co-dominant. See §4.A reframe. |
| 2 | **Within-day variance sweep** | pending | Establishes the noise floor (~7 % at long-n confirmed §1.5). Every single-cell ms claim — including the §4.A.1 ~10 % refactor target — is read against this band. ~80 min |
| 3 | **B=1 attention bucket capture** | ✅ **resolved** 2026-05-26 (`measurement-gaps-cell4-rerun2-2026-05-26_11-00-58`) | Prior caveat had wrong bucket names. B=1 emits `tee:attn_cached` / `tee:attn_permuted_cached` / `tee:attn_swa_cached`. B=1 n=2561 attention is **12.5 % prefill / 63.0 % decode** — even more attention-dominated than B=8 (no batch amortisation of in-TEE GQA). |
| 4 | **Pad-ratio (1.59, 1.99) probe** | ✅ **resolved** 2026-05-26 (same sweep) | B=8 n=2400 pad 1.70: DCT-IV wins by 22.4 % prefill wall (216 s vs 264 s HD₃-forced). The 1.6 Auto threshold is well-calibrated — no remaining crossover gap. |

### §3.2 Substrate prereqs

| # | Item | Status | Impact | Engineering |
|---:|---|---|---|---|
| 1 | **R1 weight Arc drop** | ✅ **shipped** 2026-05-22 (commit 4686b8f) | **5.28 GiB measured host RSS reclaim** post-VRAM upload (7.67 → 2.39 GiB at Qwen3-4B, confirmed by `dct4-cascade-microbench-2026-05-26`). `provision_into` `.take()`s per-layer Arcs; default `register_weight_bf16_shared` consumes the Arc after VRAM upload. Residual 2.39 GiB ≈ `token_embedding` (778 MB, still needed for `embedding_lookup`) + layer norms + config + allocator slack. | — |
| 2 | **UMA allocator unblock** | ✅ **resolved as a non-issue** 2026-05-26 (`bench-results/uma-spike-2026-05-26`) | B=16 n=2048 K=8 runs clean — no OOM, no `VK_ERROR_OUT_OF_DEVICE_MEMORY`. Cubecl's `tasks_max=32` default already chunks the forward into safe submissions; the 2026-05-22 cap was a three-executors-alive squeeze that the current sequential-executor pattern doesn't reproduce. **But also: no aggregate-tok/s lift at this shape.** B=8→16 prefill aggregate moves 121.3 → 114.8 tok/s (−5 % — slightly worse; GPU compute is already saturated). Decode aggregate moves 4.62 → 5.66 tok/s (+22 %, the surviving launch-overhead-amortised gain at `n_q=1`). The "B≥16 unlocks aggregate throughput" thesis was wrong — at long-n shapes there's no compute headroom to amortise. | — |

---

## §4 Optimization buckets

Each bucket lists its measured share, then iGPU and dGPU
levers. dGPU levers are pointers into
[`2026-05-22-dgpu-attention-revival.md`](../handoffs/2026-05-22-dgpu-attention-revival.md) —
this doc gives EV + engineering at a glance, not the full
plan.

### §4.A CPU mask — DCT-IV + HD₃ inner kernels

Profile shares vary by shape and Auto pick. Post-tune Auto:
pad ratio ≤ 1.6 → HD₃, > 1.6 → DCT-IV.

| Shape (B=8 K=32 prefill) | pad | Family | Mask % of wall |
|---|---:|---|---:|
| n=3500 (long-n HD₃) | 1.17 | HD₃ | **15.9** |
| n=320 (short-n HD₃) | 1.56 | HD₃ | 13.3 |
| n=2561, B=1 | 1.59 | HD₃ | 18.1 |
| n=2400 (crossover) | 1.70 | **DCT-IV** | **41.3** |
| n=2048 (production) | 1.99 | **DCT-IV** | **38.0** |

Decode mask ≤ 4 % across all shapes (always HD₃ at
stacked_n=16); decode is not a CPU-mask-driven bucket.

**Two structural pictures (now confirmed by §3.1 #1):**

- **DCT-IV shapes (pad 1.7-1.99, B=8 chunks ~1500-2500
  tokens)**: mask is 38-41 % of prefill — the dominant bucket.
  **§4.A.1 DCT-IV column-locality is the top iGPU lever for
  this workload class** (~10 % prefill wall).
- **HD₃ shapes at long n (pad < 1.6, B=8 chunks ≥ ~3000
  tokens)**: mask shrinks to 16 % because the rest of prefill
  (GPU matmul 48 %, attention 27 %) grows. CPU mask is no
  longer dominant. **§4.A.1 / §4.A.2 levers move at most
  ~3-5 % wall at these shapes** — substrate prereqs (§3.2)
  and dGPU work matter more here.

Cross-cutting bf16 inner-kernel levers (which compose with
both families) live in §4.E.

#### §4.A.1 DCT-IV (iGPU lever)

**DCT-IV column-locality refactor.** `dct4_cols_inplace`
(`crates/gelo-protocol/src/dct4.rs:259`) operates on a row-major
`(n, d)` buffer with column-strided access — every column of n
reads is n separate stride-`d` memory accesses (10 KB stride at
d=2560). The existing per-column copy-out / DCT / copy-back via
thread-local `COL_SCRATCH` mitigates the inner-DCT cost but
still pays three full stride-`d` passes per `apply_in_place`.

Two designs to evaluate in a 3-day spike:

1. **Transpose-once-per-call** — stage to `(d, n)` row-major
   before the cascade (one full read-write pass), run DCT-IV
   row-wise (cache-friendly), transpose back. Adds 2× transpose
   passes; saves 3× stride-`d` passes. Net win iff saved-stride
   cost exceeds transpose cost — likely yes at long-n.
2. **Block-strided DCT-IV** — process columns in `T`-column
   tiles (T ≈ 16-32) so each tile fits in L2. Avoids the
   explicit transpose; needs a small rustdct extension or
   custom inner kernel.

**Scope**: only helps shapes where Auto picks DCT-IV (pad ratio
> 1.6). Doesn't help decode (HD₃ only), short-n prefill (HD₃),
or long-n HD₃ prefill (pad < 1.6, n ≥ ~3000 at B=8).

**Status — SHIPPED 2026-05-26.** Measured impact via
`bench-results/dct4-cascade-microbench-2026-05-26_*.{log,tsv}`:

| Shape (B=8 K=32) | pad | Prefill wall before | Prefill wall after | Δ wall | Mask bucket Δ |
|---|---:|---:|---:|---:|---:|
| n=2048 (production) | 1.99 | 174.92 s | **135.13 s** | **−22.7 %** | **−58.5 %** |
| n=2400 (crossover) | 1.70 | 216.15 s | **169.76 s** | **−21.5 %** | **−53.1 %** |

Both data points clear the ≥ 20 % bucket-reduction gate by ~2.9×
(measured ~55 %). The win is **larger than the original ~10 %
prefill-wall estimate** because the tile-fused cascade
eliminates **inter-stage RAM round-trips** entirely — prior
3-stage code paid full-buffer DDR5 traffic between every DCT
and every diag (~6× buffer traffic); the cascade pays one
copy-in + one copy-out per `T=16`-column tile with all six
stages resident in L2.

**Workload-mix sensitivity** (§3.1 #1 resolved): at long-n HD₃
shapes (B=8 n=3500, pad 1.17) the cascade is no-op — prefill
wall 287.87 s → 289.09 s within variance. So the lever still
delivers 0 % wall at long-n HD₃ shapes, but at DCT-IV-dominant
shapes (the production extraction shape mix) the win is 2× the
estimate. EV stays workload-weighted.

#### §4.A.2 HD₃ (iGPU lever)

HD₃ FWHT runs at every prefill shape with pad ratio < 1.6 and
at every decode step (stacked_n=16 overlay).

**Long-n HD₃ measurement (§3.1 #1 resolved)**: at B=8 n=3500
(pad 1.17) the HD₃ mask bucket is 15.9 % of prefill — about
**40 % the size of DCT-IV's bucket at production shapes**. HD₃
prefill apply+unapply combined is 45.8 s out of 287.9 s wall.
Even a 50 % bucket reduction would yield only ~8 % wall at this
shape. So HD₃-prefill is a real bucket but not large enough to
warrant a dedicated optimization on its own.

Levers in scope:

- **Column-axis FWHT parallelism**. At decode (stacked_n=16,
  3 % wall) the absolute gain is < 2 % decode. At prefill
  long-n HD₃ (15.9 % wall) a 50 % bucket reduction gives ~8 %
  wall — below the §1.5 7 % variance floor without compounding.
  Not worth a dedicated effort; subsumed by §4.E.1 bf16 inner
  kernel which covers the same arithmetic-bandwidth budget.
- **Radix-8 FWHT scratch reuse** is already shipped
  ([[hd3_radix8_and_scratch_reuse]]); no further parallelism
  work landed inside the FWHT kernel since.

#### §4.A.3 dGPU note

Mask path cannot move to GPU without violating the GELO threat
model (would expose `A` on the device). iGPU optimizations here
carry over unchanged to dGPU — the absolute CPU mask cost is
identical; only its share of wall shifts as other buckets shrink
on dGPU.

### §4.B GPU matmul (engine: matmul + matmul_many)

| Shape | Prefill share | Decode share |
|---|---:|---:|
| B=8 n=2048 | 40.5 % | 37.7 % |
| B=8 n=320 | 72.9 % | 72.7 % |
| B=1 n=2561 | 54.5 % | 25.9 % |

#### §4.B.1 iGPU

DDR5-bandwidth-bound; **no direct kernel lever exists on iGPU**.
The GPU matmul kernels themselves aren't the cost — the bus is.

**B≥16 amortisation does not help at production shape**
(§3.2 #2 spike resolved 2026-05-26): B=8 → B=16 at n=2048
measured prefill aggregate 121.3 → 114.8 tok/s — slightly
worse, not better. GPU compute is already saturated at B=8 on
this shape; doubling B doubles the work without amortising
the dominant matmul bucket. Decode aggregate amortises +22 %
at B=16 (4.62 → 5.66 tok/s — the surviving launch-overhead
gain at `n_q=1`).

Where B-scaling DOES help: short-n shapes where dispatch
launch overhead dominates compute. At those shapes B≥16 has
real amortisation headroom — but they're not the binding
production workload.

#### §4.B.2 dGPU

HBM ~3 TB/s vs DDR5 ~80 GB/s → **~40× bandwidth headroom** for
QKV / O / gate-up / down dispatches. Concrete levers
(`2026-05-22-dgpu-attention-revival.md` §1-§3):

| Lever | Impact (vs iGPU baseline) | Engineering |
|---|---|---|
| dGPU substrate baseline (no other change) | ~3-5× GPU matmul wall | hardware-gated (M5.9) |
| Q4 weight quantization on dGPU | GPU matmul memory ÷4 + ~2-3× extra throughput | 4 weeks ([`q4-gpu-weights.md`](q4-gpu-weights.md)) |

### §4.C In-TEE attention (per-sequence in-TEE causal attention)

| Shape | Prefill share | Decode share |
|---|---:|---:|
| B=8 n=2048 | 12.6 % | **53.9 %** |
| B=8 n=320 | 2.9 % | 10.2 % |
| B=1 n=2561 | – ⁽ᵃ⁾ | – ⁽ᵃ⁾ |

Decode at long-n is the binding bucket for the 40+ tok/s target.

#### §4.C.1 iGPU — ABORTED

Bucket-2 spike on 2026-05-22 measured GPU 16.4× slower than
in-TEE-rayon at Qwen3-4B B=8 n_kv=2048
([[bucket_2_batched_gpu_attention_aborted]]). RADV gfx1151
compute is the binding factor — not launch overhead.
burn-cubecl-fusion folds mask add (< 2 % delta), so a custom
WGSL FlashAttention-D kernel wouldn't help either: scores tensor
is only ~1 MB at decode-m=1 shape; the gap is compute throughput
on a shared bus.

**No iGPU lever exists for this bucket.** Don't re-spike on iGPU
without a different substrate (cubecl-hip API stable, or dGPU).

#### §4.C.2 dGPU

HBM ~3 TB/s kernel-read vs DDR5 ~40 GB/s shared → ~75× faster
attention kernel; persistent K/V eliminates the 256 MB-per-step
upload pipeline (only the new `K_t, V_t` row = 32 KB). Levers
from the dGPU revival handoff:

| Lever | Impact | Engineering |
|---|---|---|
| Persistent K/V security spike — Option I (block-fresh-π) | enables persistent K/V if σ-vs-N curve permits | 1 week |
| Persistent K/V security spike — Option G (TwinShield-Xue) | published-scheme alternative to I | 1-2 weeks |
| Persistent K/V substrate refactor (post-spike) | session-resident K/V; eliminates upload pipeline | 2-3 weeks |
| GQA-aware custom WGSL FlashAttention kernel | 4× K/V data motion reduction (group=4) | 2-3 weeks |

Compounded, dGPU compresses bucket C from 53.9 % decode wall to
~10-20 % at the same shape. Details in
`2026-05-22-dgpu-attention-revival.md` §1-§3.

### §4.D Compute pipelining (R4 async overlap)

Cross-cuts buckets A and B by overlapping CPU mask (layer N+1)
with GPU matmul (layer N). Doesn't fit any single bottleneck,
so it gets its own bucket.

**Gate — Q#2 RESOLVED 2026-05-26** (`bench-results/q2-radv-async-spike-2026-05-26_14-14-23`):
PARTIAL OVERLAP measured. burn-tensor exposes `into_data_async`
under the hood so wgpu submit is non-blocking by design — the
question reduced to bus contention on Strix Halo UMA. Result:

| Regime | Wall (n=2056, d=2560, d_out=2560) |
|---|---:|
| T_gpu (engine.matmul alone) | 19.08 ms ± 0.42 |
| T_cpu (DCT-IV cascade alone) | 10.02 ms ± 0.23 |
| T_concurrent (both via std::thread) | 23.26 ms ± 0.84 |
| Speedup vs serial | **1.25×** |
| Wall saved | **5.84 ms = 58.3 % of min(T_cpu, T_gpu)** |

CPU runs at ~58 % efficiency under concurrent GPU load. Well
above the "weak" threshold; in the "partial overlap, R4 viable"
band. **R4 green-lit.**

**Impact estimate (refreshed by measurement)**:

| Substrate | Estimate | Reason |
|---|---|---|
| iGPU (UMA) | **~12 % wall** at production prefill | Mask bucket 20 % × 58 % overlap × applicable shapes |
| dGPU (PCIe) | ~25-30 % wall | PCIe DMA + GPU matmul are physically separate; no shared-bus contention |

**Engineering**: 5-8 days substrate refactor — add
`engine.matmul_async` + `engine.read_result` to the trait,
expose async path through substrate `offload_linear_async`,
pipeline forward.rs to issue layer N's matmul before computing
layer N+1's mask cascade.

**Order interaction** (revised): §4.E bf16 activation pipeline
was deprioritised 2026-05-26 (microbench-disconfirmed). R4 is
now the next legitimate iGPU lever. The 5-8 day investment is
justified by the 12 % wall projection — at production shape
135 s → 119 s prefill.

### §4.E bf16 / activation precision (cross-cutting)

Touches buckets A, B, C, F. Three composable scopes — all in
this bucket per the grilling decision (bf16 work shipped as a
single coordinated effort rather than fragmented across the
perf buckets it benefits).

**Status — DEPRIORITISED 2026-05-26 by standalone cascade
microbench** (`bench-results/bf16-cascade-microbench-2026-05-26_13-47-56`).
Engine-side Path β (shipped 7abb9f1) + bf16 elementwise kernels
(shipped 98271f0) + bf16 mask cascade variants (shipped a05eb8a)
delivered the precision-contract infrastructure cleanly, but
the validation microbench shows the cascade DRAM savings don't
translate to a wall lever at production scale on iGPU:

| Cascade | Shape | f32 wall | bf16 wall | Speedup | Notes |
|---|---|---:|---:|---:|---|
| DCT-IV | n=2056 d=2560 (prod prefill) | 11.16 ms | 10.28 ms | 1.085× | Tile-fused widen-narrow; +8 % standalone |
| HD₃ | n=4096 d=2560 (prod HD₃ shape) | 20.35 ms | 39.06 ms | 0.521× | Bulk widen-narrow + per-call alloc; 2× slower |

Projecting to production prefill (B=8 n=2048, cascade = 20 % of
wall): DCT-IV cascade gain × cascade share = **~1.6 % wall**
reduction — below the §1.5 ~7 % variance floor. HD₃ at decode
mask = 4 % wall × 2× slowdown = **~4 % decode regression** at
HD₃ shapes.

The hypothesis from `m1-12-bf16-activation-pipeline.md` ("3-10 %
iGPU direct wall reduction") does not survive standalone
measurement. The post-cascade-refactor design already ensures
L2-resident tiles, so the bf16 DRAM saving at the tile boundary
is small in absolute terms — and the widen-narrow at the boundary
eats some of it back.

Multi-week phase 3b (substrate offload_linear_bf16) and phase 3c
(forward.rs wire-up) are NOT WORTH PURSUING on iGPU. The
infrastructure landed across phases 1-3a remains useful for the
dGPU substrate revival (dGPU has bf16-native compute kernels via
cuBLAS, so the precision story flips). On iGPU, the next iGPU
lever is **Q#2 RADV-async spike → R4 async overlap** (§4.D).

#### §4.E.1 Inner-kernel rewrite (FWHT + DCT-IV in bf16)

FWHT butterflies in bf16, DCT-IV inner steps in bf16. Saves
arithmetic bandwidth inside the cascade. Independent of the
boundary work and the end-to-end activation rework — can ship
standalone.

- **HD₃ inner kernel**: AVX-512_BF16 (Zen 5) has native FMA
  support; the existing radix-8 butterfly maps cleanly.
- **DCT-IV inner kernel**: harder — uses rustdct as the inner
  primitive. Either rustdct fork or hand-rolled bf16 cascade
  (1-2 weeks).

**Estimated impact**: 5-10 % of mask-bucket arithmetic bandwidth
on iGPU; bigger lever on dGPU once C bucket compresses and
mask's relative share grows.

**Why this was missing from prior roadmap revisions**: the
2026-05-22 plan had a Haar-flavoured bf16 LPGEMM lever ("bucket
3a"). Production turned out to use DCT-IV/HD₃ (not Haar) per
[[bucket_3a_inert_in_production]]. Inner-kernel bf16 for the
families Auto actually picks went missing in the simplification;
this entry restores it.

#### §4.E.2 Boundary conversion (operand in/out bf16, cascade f32)

Operand arrives bf16 at the mask kernel, expand to f32 inside
`apply_in_place_slice`, run cascade at f32, narrow on exit.
Saves only the f32→f16 host-side upload conversion (~½ GiB DDR5
traffic per call at decode-attention shape per the bucket-2
post-mortem).

Subset of §4.E.3 — once end-to-end bf16 lands, the mask kernel
just consumes bf16 inputs natively and this boundary work
collapses out.

#### §4.E.3 End-to-end bf16 activations (3b rework)

Every forward-pass `Array2<f32>` (`h`, `h_norm`, residuals,
attention context, FFN gate/up/down outputs, etc.) downsized
to bf16. Touches RMSNorm / qk_norm / RoPE / SwiGLU / residual
kernels + parity tests + GPU offload precision contract.

Detailed plan: [`m1-12-bf16-activation-pipeline.md`](m1-12-bf16-activation-pipeline.md).

**Estimated impact**:
- iGPU: 3-10 % direct wall reduction via removed boundary
  conversions and ½ DDR5 traffic on every per-layer activation
  pass.
- dGPU: **prerequisite for any bucket-2 revival**. The upload-
  pipeline tax that aborted iGPU bucket-2 scales worse on PCIe,
  not better — any dGPU attention offload must consume bf16/f16
  activations end-to-end or repeats the iGPU failure mode.

**Engineering**: 2-3 weeks (per the existing pipeline plan).

#### §4.E sequencing

| Order option | Pro | Con |
|---|---|---|
| §4.E.1 first | Independent; lands in 1-2 weeks; validates bf16 precision contract on a narrow surface before committing forward-pass-wide | Most of E.3's structural value still pending |
| §4.E.3 first | Unblocks dGPU bucket-2 revival; compounds with E.1 automatically | Multi-week commit before any measured wall win |

Recommended: §4.E.1 → §4.E.3 (§4.E.2 collapses into E.3
naturally). E.1 proves the precision contract on a small surface
before committing to E.3's forward-pass-wide rework.

### §4.F Misc TEE ops

| Op | B=8 n=2048 prefill % | B=8 n=2048 decode % |
|---|---:|---:|
| `tee:residual` | 2.0 | small |
| `tee:qk_norm` | 1.2 | small |
| `tee:swiglu_activate` | 0.7 | small |
| `tee:rope` | 0.4 | small |
| `tee:rmsnorm` | 0.3 | small |
| **combined** | **~5 %** | **~5-8 %** |

**Parked.** No individual op exceeds the §1.5 variance floor.
§4.E.3 end-to-end bf16 activations incidentally halves these
buckets' DDR5 traffic without per-op rework.

**LM-head** (`compute_logits`) was the only formerly-misc bucket
that warranted promotion — R3 lifted it to GPU under masked
offload and the bucket dropped 97.6 % at B=8 K=64
([[m1_12_r1_q1_microbench_findings]]). Done.

---

## §5 Per-bucket EV summary

iGPU and dGPU at a glance. EV reads against the §1.5 variance
floor (~7 % at long-n); single-cell estimates are gated on
§3.1 #2.

| Bucket | Top iGPU lever | iGPU EV | Top dGPU lever | dGPU EV |
|---|---|---|---|---|
| **§4.A.1 DCT-IV mask** | column-locality cascade ✅ | **~22 % prefill at DCT-IV shapes (measured)** | (inherited from iGPU) | (no separate dGPU lever) |
| **§4.A.2 HD₃ mask** | column-axis FWHT | < 2 % decode; < 8 % long-n prefill (subsumed by §4.E.1) | (inherited from iGPU) | (no separate dGPU lever) |
| **§4.B GPU matmul** | substrate prereqs (B≥16) | aggregate-tok/s only | persistent K/V upload elision; Q4 weights | ~40× ceiling |
| **§4.C In-TEE attention** | none — aborted | 0 % | bucket-2 revival + GQA WGSL + persistent K/V | 53.9 % → ~10-20 % decode |
| **§4.D Compute pipelining (R4)** | async overlap (gated Q#2) | ~15 % wall iGPU best | async overlap | ~25-30 % wall |
| **§4.E bf16 / activation precision** | E.1 inner kernel → E.3 end-to-end | 3-10 % direct + structural | prerequisite for §4.C.2 | unblocks dGPU C |
| **§4.F Misc TEE ops** | parked | < 2 % | parked | < 2 % |

iGPU cumulative ceiling (post-cascade baseline, with remaining
levers landing): ~5-7 tok/s B=1 decode, ~10-15 tok/s B=8
aggregate, ~150-180 tok/s B=8 prefill aggregate (production
shape now at **121 tok/s** post-cascade, was 94 tok/s pre).
dGPU cumulative target: 40-60+ tok/s B=1 decode once §4.C.2
levers compound with §4.B.2 and §4.E.3.

---

## §6 Open decisions / risks

### §6.1 Gating decisions

| Decision | Gated by | Outcome |
|---|---|---|
| Top mask-bucket lever | §3.1 #1 long-n HD₃ cells | Picks §4.A.1 DCT-IV vs §4.A.2 HD₃ work |
| Whether R4 ships at all (§4.D) | Q#2 RADV-async spike | If RADV serialises, R4 dead on iGPU — skip to §4.E + dGPU prerequisites |
| dGPU persistent K/V cover design | Option I vs Option G security spikes | Whichever clears first defines the dGPU revival path |
| Bucket 3a / threads-dispatch — keep or delete | Production decision on Haar | Currently inert; safe to leave in tree but don't invest further |
| M5.9 hardware procurement timeline | Out-of-band (business) | Sets dGPU calendar |
| §4.E.1 vs §4.E.3 first | Variance-floor confidence after §3.1 #2 | If variance is wider than expected, prefer E.1 (narrower surface, cleaner attribution) |
| §4.A.1 spike design (transpose vs block-strided) | spike result | Pick the simpler one if both clear the 20 % gate |

### §6.2 Engineering risks

| Risk | Mitigation |
|---|---|
| §4.A.1 DCT-IV refactor shows < 20 % bucket reduction in spike | Drop the refactor; try the alternative design (transpose vs block-strided) or escalate to a custom DCT-IV kernel (out of v1 scope). |
| §4.E.3 bf16 activations introduces numerical regression | `m1-12-bf16-activation-pipeline.md` §1.3 specifies parity contract (bf16-floor 1e-3 abs, greedy token stability). Re-baseline tests are part of the deliverable. |
| §4.D R4 async pipelining shows no win on iGPU | Q#2 spike (½ day) decides before engineering commits. iGPU-specific; dGPU has its own async story. |
| §4.C.2 Option I σ-vs-N curve exceeds model tolerance | Option G parallel spike. If neither clears, persistent K/V is impossible; full §4.C.2 falls to bucket-2-equivalent without K/V persistence, ceiling ~25 tok/s instead of 40+. |

### §6.3 Hardware risks

| Risk | Mitigation |
|---|---|
| dGPU substrate doesn't behave (PCIe topology, driver, RPC overhead) | §4.B.2 / §4.C.2 includes "as-is baseline" before further engineering. If broken, fall back to iGPU ceiling and reassess. |
| SEV-SNP attestation flow blocks bring-up | Hardware track is independent of the engineering plan. iGPU work ships regardless. |
| dGPU compute throughput is lower than projected | Re-baseline at substrate bring-up step. Persistent K/V engineering still helps; ceiling may drop to ~25 tok/s. |

### §6.4 Security risks

| Risk | Mitigation |
|---|---|
| AloePri c6 (R3 LM-head default) fails | Revert R3 default. Cuts decode wall by ~45 % regression at B=1, ~63 % at B=8. |
| §4.E.3 activation precision regression on extraction quality | Phased migration (W2-W4 in `m1-12-bf16-activation-pipeline.md`) lets us bisect failures per-kernel. |
| Persistent K/V security spike on Option I/G uncovers new attack class | Filed as research outcome; v1 falls back to per-call fresh K/V on dGPU; ceiling ~25 tok/s. |

---

## §7 Post-optimization follow-ups

Not Gelo-LLM (`gelo-*` crates) scope; lives in adjacent crates /
orchestrator / paper-research. Captured here to keep the
roadmap honest about what's a perf lever vs what's a separate
workstream.

- **D2 orchestrator rewire** — substrate landed in
  `gelo-embedder::DecoderRuntime::generate_extraction_batched`;
  remaining edit in `lightrag-private::extract_kg_from_chunks`.
  Realises 5× end-to-end extraction wall on the v7 fixture. Not
  Gelo-LLM scope. Plan: [`m1-11-batched-decode.md`](m1-11-batched-decode.md);
  handoff: [`2026-05-22-q3-4b-b8-mask-sweep.md`](../handoffs/2026-05-22-q3-4b-b8-mask-sweep.md) §"Status on M1.11".
- **Varlen / chunked batching** — per-sequence orchestration for
  ragged sequences. Zero win in identical-length bench; ~10-30 %
  per-prompt wall in production extraction with variable chunks.
  Substrate API change. Owner: orchestrator (`lightrag-private`).
- **Continuous batching / PagedAttention** — throughput-oriented
  scheduler that replaces finished sequences without draining.
  Orchestrator-level. iGPU doesn't need it; dGPU evaluates at
  M5.9+.

---

## §8 Out of scope / future-rnd

- **Speculative decoding** — breaks greedy parity contract
- **Encrypted KV on GPU (SCX-class)** — multi-month research;
  `future-rnd.md` §5
- **bf16-native DCT-IV inner kernel via rustdct upstream** —
  §4.E.1 covers the hand-rolled version; upstream change is
  separate and not on critical path
- **Bucket 3a + per-shape BLIS thread dispatch** — Haar-only;
  inert because Auto never picks Haar
  ([[bucket_3a_inert_in_production]])
- **HD₃ Phase 0 spike / 4a / 4b / 4c laundry list from prior
  rewrites** — superseded by §4.A.2 + §4.E.1; the prior
  per-numeric-tag organisation was a session alias and the
  underlying levers live in their bucket homes now
- **Slalom-additive hybrid for linear projections** —
  multi-week protocol-level redesign + AloePri-class attack-suite
  re-validation; highest ceiling, lowest confidence; pre-spike
  via Python sim first ([[private_llm_inference_round_3]])
- **Confidential GPU (H100 CC / B200 TEE-I/O)** — deployment
  fork; out of v1 scope per [[private_llm_inference_round_2]]

---

## §9 References

- `bench-results/m1-12-hd3-perf-sweep-2026-05-26_07-04-58.{log,tsv}` — the §1 sweep (14 cells)
- `bench-results/m1-12-auto-tune-verify-2026-05-26_08-42-00.{log,tsv}` — post-tune verification (3 cells)
- `scripts/m1-12-hd3-perf-sweep.sh` — full sweep driver (script name retains the historical milestone tag)
- `scripts/m1-12-auto-tune-verify.sh` — Auto-tune verify driver
- `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs` — `m1_12_sweep_cell` test (env-driven)
- `crates/gelo-protocol/src/{hd3.rs, dct4.rs, mask.rs, sim.rs}` — mask kernel + Auto resolver
- [`2026-05-22-perf-bucket-roadmap-r3-default.md`](../handoffs/2026-05-22-perf-bucket-roadmap-r3-default.md) — pre-rewrite baseline this roadmap supersedes
- [`2026-05-22-q3-4b-b8-mask-sweep.md`](../handoffs/2026-05-22-q3-4b-b8-mask-sweep.md) — per-op breakdown predecessor
- [`2026-05-22-dgpu-attention-revival.md`](../handoffs/2026-05-22-dgpu-attention-revival.md) — dGPU bandwidth model + §4.B.2 / §4.C.2 detail
- [`m1-12-bf16-activation-pipeline.md`](m1-12-bf16-activation-pipeline.md) — §4.E.3 detail
- [`m1-12-permuted-attention-batched-decode.md`](m1-12-permuted-attention-batched-decode.md) — §4.C.1 iGPU abort retro
- [`m1-12-blis-thread-dispatch.md`](m1-12-blis-thread-dispatch.md) — Haar-only follow-up (inert)
- [`m1-12-tee-gpu-throughput.md`](m1-12-tee-gpu-throughput.md) — original M1.12 spec
- [`q4-gpu-weights.md`](q4-gpu-weights.md) — §4.B.2 weight quant plan
- [`2026-05-26-mask-instrumentation-and-auto-tune.md`](../handoffs/2026-05-26-mask-instrumentation-and-auto-tune.md) — 2026-05-26 patch round retro (instrumentation + Auto threshold tune)
- [[bucket_3a_inert_in_production]] — Haar-vs-Auto discovery memory
- [[bucket_2_batched_gpu_attention_aborted]] — iGPU bucket-2 abort retro
- [[m1_12_r1_q1_microbench_findings]] — B=1 decode baseline (3.6 tok/s figure)
- [[m1_12_production_mask_is_dct4]] — 2026-05-26 sweep finding memory
