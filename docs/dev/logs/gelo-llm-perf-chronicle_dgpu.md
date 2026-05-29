---
type: dev-log
status: current
created: 2026-05-29
updated: 2026-05-29
tags: [gelo, perf, dgpu, nvidia, rtx5090, vulkan, attention, mask, chronicle]
companion: [gelo-llm-perf-chronicle]
---

# GELO-LLM perf chronicle — dGPU (Nvidia) track

> Companion to [`gelo-llm-perf-chronicle.md`](gelo-llm-perf-chronicle.md)
> (the iGPU / Strix Halo track). That document is the spine for the
> protocol cost model, mask-family history, and the dated optimisation
> chronicle. **This document holds the discrete-GPU measurements** —
> the hardware the iGPU roadmap repeatedly deferred to as the
> "dGPU substrate (M5.9), hardware-gated."
>
> First entry: 2026-05-29, RTX 5090 bring-up. The headline finding is
> that **raw dGPU hardware does not lift throughput on the current
> code** — prefill is materially *slower* than the Strix Halo iGPU, and
> the win the dGPU was supposed to deliver is gated on the per-call
> upload-pipeline replacement the iGPU chronicle's dGPU-revival design
> already specified (persistent K/V, end-to-end bf16 activations,
> GQA-aware kernel). The measurements below pin exactly which buckets
> regress and why.

## 1. Hardware substrate

| | iGPU track (reference) | dGPU track (this doc) |
|---|---|---|
| Box | Strix Halo (Ryzen AI Max+ 395) | Ryzen 9 7900X + RTX 5090 |
| CPU | 16 Zen5 cores | **12 Zen4 cores** (24 threads), 5.7 GHz max |
| System RAM | quad-channel LPDDR5X, **~256 GB/s** (UMA, shared with iGPU) | dual-channel DDR5, **~80–90 GB/s** (not shared) |
| GPU compute | Radeon 8060S iGPU (RDNA 3.5, gfx1151), fp16 wgpu Vulkan | **RTX 5090** (Blackwell, 32 GB GDDR7 ~1.8 TB/s), fp16 wgpu Vulkan |
| GPU memory bus | shared DDR5 (UMA) | dedicated GDDR7 over **PCIe** |
| BLAS | AOCL-BLIS, `GELO_BLIS_THREADS=16` | AOCL-BLIS, threads unset (default) |

Both run the same engine: fp16 wgpu **Vulkan**, R3 LM-head GPU offload
on, mask = Auto. The wgpu adapter auto-selects
`NVIDIA GeForce RTX 5090 (DiscreteGpu)` here — confirmed in the bench
banner and by GPU-memory movement under load.

The two boxes differ on **three** axes at once — GPU (5090 vs iGPU),
CPU core count (12 vs 16), and **system-memory bandwidth (~3× in the
iGPU box's favour)**. The last one matters most: GELO runs attention
and the mask GEMM on the CPU, and both are memory-bandwidth-bound (see
the iGPU chronicle §2 / §7). So this is not a clean GPU-vs-GPU
comparison — it is a whole-machine comparison, and the bandwidth gap is
the dominant confound to keep in mind throughout.

## 2. Benchmark

`gelo_llm_prefill_decode_breakdown` (the Gelo-LLM main bench, see
`CLAUDE.md`), Qwen3-4B, fp16, R3 on, mask=Auto, K=32 decode tokens,
single sample per cell. Release build. First-run cubecl autotune is
included in the GPU buckets (no warm-up pass) — see §5 confound 1.

## 3. Results — RTX 5090

### 3.1 B=1 n=2048 (single-stream)

Prefill 25.3 s (80.9 tok/s); decode 17.5 s (1.8 tok/s, 546 ms/step).

| PREFILL op | ms | share | DECODE op | ms | share |
|---|--:|--:|---|--:|--:|
| `engine:matmul_many` ◆ | 6760 | 28.5% | `tee:attn_cached` | 9179 | 51.6% |
| `tee:attn_cached` | 5015 | 21.2% | `engine:matmul_many` ◆ | 2236 | 12.6% |
| `engine:matmul` ◆ | 4094 | 17.3% | `engine:matmul` ◆ | 2233 | 12.6% |
| `gelo:mask_unapply:dct4` | 2567 | 10.8% | `gelo:mask_unapply:hd3` | 1837 | 10.3% |
| `gelo:strip_shield` | 2384 | 10.1% | `gelo:mask_apply:hd3` | 1041 | 5.9% |
| `gelo:mask_apply:dct4` | 1603 | 6.8% | `gelo:shield_stack` | 862 | 4.8% |
| (rest) | <400 ea | | `tee:compute_logits` | 340 | 1.9% |

◆ = GPU (Vulkan). All other buckets are CPU, in-TEE.

### 3.2 B=8 n=2048 (production shape — the iGPU comparison anchor)

Prefill 216.9 s (75.5 tok/s agg, 9.44 tok/s/seq); decode 40.6 s
(6.30 tok/s agg, 0.79 tok/s/seq, 1269 ms/step).

| PREFILL op | ms | share | DECODE op | ms | share |
|---|--:|--:|---|--:|--:|
| `engine:matmul_many` ◆ | 68790 | 33.4% | `tee:attn_cached_inplace_many` | 14574 | 34.0% |
| `tee:attn_inplace_many` | 43774 | 21.2% | `engine:matmul_many` ◆ | 9941 | 23.2% |
| `engine:matmul` ◆ | 39986 | 19.4% | `engine:matmul` ◆ | 8956 | 20.9% |
| `gelo:mask_unapply:dct4` | 27367 | 13.3% | `gelo:mask_unapply:hd3` | 3445 | 8.0% |
| `gelo:mask_apply:dct4` | 11759 | 5.7% | `tee:compute_logits` | 2526 | 5.9% |
| `tee:residual` | 4648 | 2.3% | `gelo:shield_stack` | 1564 | 3.6% |
| `gelo:shield_stack` | 4151 | 2.0% | `gelo:mask_apply:hd3` | 1533 | 3.6% |
| (rest) | <2200 ea | | (rest) | <200 ea | |

**GPU matmul total**: prefill 108.8 s (52.8%), decode 18.9 s (44.1%).

## 4. Comparison vs iGPU (B=8 n=2048, the documented production cell)

iGPU reference: post-DCT-cascade prefill **135.13 s** (121.3 tok/s agg),
decode **55.08 s** (0.58 tok/s/seq); decode shares attn 53.9% /
matmul 37.7% / mask 4.2% (iGPU chronicle §4 + roadmap §1.3). iGPU
prefill bucket absolutes are *derived* from the pre-cascade per-op
shares (matmul and attention are untouched by the cascade; mask is the
post-cascade figure).

### 4.1 Wall

| Phase | RTX 5090 | iGPU (Strix Halo) | 5090 ÷ iGPU |
|---|--:|--:|--:|
| **Prefill** | 216.9 s | 135.1 s | **1.61× (slower)** |
| **Decode** | 40.6 s | 55.1 s | **0.74× (faster)** |

So we are **slower on prefill, faster on decode**. The "we're slower"
impression is real and lives entirely in prefill.

### 4.2 Where prefill regresses (absolute bucket time)

| Prefill bucket | RTX 5090 | iGPU (approx) | 5090 ÷ iGPU |
|---|--:|--:|--:|
| GPU matmul (both) ◆ | 108.8 s | ~70.8 s | ~1.5× slower |
| in-TEE attention | 43.8 s | ~22.0 s | ~2.0× slower |
| mask DCT-IV (apply+unapply) | 39.1 s | ~27.6 s | ~1.4× slower |

**Every prefill bucket is slower on the 5090 box** — including the GPU
matmul. Two distinct causes:

- **GPU matmul slower (◆) — the PCIe upload tax.** GELO uploads the
  *masked* f32 activation `U = A·H` to the GPU on every projection (it
  is fresh per forward and cannot be cached). At B=8 n=2048 that is
  ~80 MB/dispatch × 72 dispatches, plus the f32→f16 conversion and the
  output read-back. On the iGPU this rides a **mapped UMA buffer**
  (DDR5, no transit); on the dGPU it is a **PCIe DMA round-trip per
  call**. The 5090's HBM compute finishes in microseconds but the
  per-call upload dominates, so the matmul bucket *grows* rather than
  shrinks. This is precisely the failure mode the iGPU chronicle's
  dGPU-revival design flagged — "the upload-pipeline tax scales worse
  on PCIe, not better" (roadmap §4.E.3) — now measured directly.
- **CPU buckets slower — memory bandwidth.** Attention and the mask
  GEMM are memory-bandwidth-bound. Strix Halo's quad-channel LPDDR5X
  (~256 GB/s) is ~3× this box's dual-channel DDR5 (~80–90 GB/s), and it
  also has 16 cores vs 12. The prefill attention (`tee:attn_inplace_many`,
  a streaming O(n²) pass) is the most bandwidth-sensitive and regresses
  the hardest (~2×).

### 4.3 Why decode is faster despite all that

| Decode bucket | RTX 5090 | iGPU | 5090 ÷ iGPU |
|---|--:|--:|--:|
| in-TEE attention | 14.6 s | ~29.7 s | **0.49× (2× faster)** |
| GPU matmul (both) ◆ | 18.9 s | ~20.8 s | ~0.91× |
| mask HD₃ | 5.0 s | ~2.3 s | ~2.2× slower |

Decode flips because its working set is tiny — `n_q=1`, so each masked
activation upload is a handful of rows, and the **PCIe upload tax
vanishes** (GPU matmul becomes competitive). The big swing is
attention: decode attention is **2× faster** on the 5090 box. The
likely reason is **UMA bus contention** — on Strix Halo the CPU
attention and the GPU matmul fight over the *same* DDR5 bus every step;
here the CPU attention has system DDR5 to itself (the GPU lives on
GDDR7), and the 7900X's higher single-thread clocks help the
latency-bound per-step kernel. The decode working set is too small to
be bandwidth-bound, so the iGPU's bandwidth edge doesn't apply — only
its contention penalty does.

## 5. Confounds (read the deltas against these)

1. **Cold autotune.** Single run, no warm-up; first-touch cubecl
   autotune is counted inside `engine:matmul*`. Part of the prefill
   GPU-matmul regression is one-time. A warm re-run is needed to split
   autotune from the genuine PCIe upload tax.
2. **BLIS threads unset** here vs `GELO_BLIS_THREADS=16` on the iGPU
   reference — affects the mask-bucket comparison.
3. **Whole-machine, not GPU-isolated.** CPU core count and memory
   bandwidth both differ; the CPU buckets are a different-machine
   comparison, not a GPU one.
4. **iGPU prefill bucket absolutes are derived** from pre-cascade
   shares; treat ±10%.
5. **Single sample**, ~7% variance floor (iGPU chronicle §7).

## 6. Takeaways + next levers

- **The dGPU does not help the current code at the production shape.**
  Net prefill is 1.6× slower; decode is 1.35× faster; the binding
  prefill bottleneck moved from CPU mask (iGPU) to **PCIe activation
  upload** (dGPU GPU-matmul bucket) stacked on **lower CPU memory
  bandwidth**.
- **This validates the dGPU-revival sequencing.** The card's HBM is
  wasted until the per-call upload pipeline is replaced. The relevant
  levers, all already scoped on the iGPU track, are now the critical
  path here:
  - **End-to-end bf16 activations** (roadmap §4.E.3) — halves the
    bytes-on-wire per upload; prerequisite for any dGPU attention move.
  - **Persistent K/V on GPU** (§4.C.2 / dGPU-revival Item 1) — kills
    the per-step K/V upload; the decode attention bucket is the target.
  - **GQA-aware single-pass WGSL kernel** (Item 2+3) — 4× less K/V
    motion + fused FlashAttention.
  - **R4 async overlap** — on PCIe the CPU mask / GPU matmul overlap
    that vanished on UMA reappears (roadmap §4.D disposition).
- **Immediate measurement debt:** (a) warm re-run to isolate autotune
  from the upload tax in the GPU-matmul bucket; (b) sweep
  `GELO_BLIS_THREADS` (12 physical cores here); (c) re-profile after
  bf16 activations land — that is the first lever expected to move the
  prefill GPU bucket on this hardware.

## 7. Artefacts

- Bench logs: `/tmp/gelo_b8_n2048.log`, `/tmp/gelo_perop_split.log`
  (B=1) — capture to `bench-results/` on the next run.
- Bench: `gelo_llm_prefill_decode_breakdown` in
  `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`.
- Trace fix this session: GPU offload now emits `engine:matmul` (single
  weight: O, FfnDown, R3 LM-head) vs `engine:matmul_many` (fused QKV,
  gate∥up) instead of the merged `engine:registered_linear` bucket
  (`crates/gelo-protocol/src/substrate.rs::run_registered_linear`).

## 8. CUDA backend A/B (2026-05-29) — modest, not transformative

A compile-time `cuda` feature on `gelo-gpu-wgpu` swaps the cubecl runtime
(`WgpuRuntime`→`CudaRuntime`) behind `Rt`/`Dev` aliases; Vulkan stays the
default. cubecl-cuda 0.9.0 + cudarc 0.18.2 build against CUDA 13.0.3 and
**execute correctly on the Blackwell RTX 5090 (sm_120)** — provision,
prefill, decode, token generation all complete.

**Protocol:** pure runtime swap, **warm** (one discarded forward via
`GELO_BENCH_WARMUP=1` populates the autotune cache, then the measured
forward), per-call readback held identical. Vulkan warm ≈ Vulkan cold
(SPIR-V autotune is cheap); CUDA cold was 2–4× *slower* (nvrtc autotune)
— so only the warm numbers are meaningful.

Warm A/B, Qwen3-4B B=1 n=2048 K=32, R3:

| Bucket | Vulkan warm | CUDA warm | CUDA advantage |
|---|--:|--:|--:|
| prefill `engine:matmul_many` | 6865 ms | 5042 ms | 1.36× |
| prefill `engine:matmul` | 4369 ms | 2574 ms | 1.70× |
| **prefill GPU total** | 11.2 s | 7.6 s | **1.47×** |
| **decode GPU total** | 4.3 s | 2.4 s | **1.84×** |
| CPU attn / mask | ~5.0 / ~4.2 s | ~5.1 / ~4.2 s | backend-invariant |
| **prefill wall** | 25.8 s | 22.5 s | **1.15×** |
| **decode wall** | 17.4 s | 15.1 s | **1.15×** (1.84→2.12 tok/s) |

**Finding — and a correction to §4.2.** §4.2 attributed the slow GPU
matmul to a "PCIe upload tax." That was wrong: the per-call data is ~1 s
of transfer over prefill, and CUDA (which would not change PCIe
bandwidth) makes the matmul only ~1.5× faster. The real picture:

- GPU matmul on the Vulkan/SPIR-V path is **not catastrophically
  tensor-core-starved** — CUDA's kernels are only ~1.5× (prefill) to
  ~1.8× (decode) faster, not the 50–300× an absent-vs-present tensor-core
  gap implies. Effective throughput is ~1.3 TFLOP/s (Vulkan) → ~2 TFLOP/s
  (CUDA) at B=1, both far below the card.
- The dominant cost is **backend-invariant**: the per-call blocking
  readback (every masked matmul round-trips to the TEE to unmask,
  serialising the dispatches) plus the CPU attention/mask buckets. GPU
  matmul is only ~45 % of prefill / ~25 % of decode, so a 1.5× kernel
  win caps at ~13 % wall.

**Disposition (research phase).** CUDA delivers a real but modest ~13 %
wall win at B=1 — far below the bar that would justify a production
backend fork. The high-value lever is the **per-call readback/sync**
(batched/streamed unmask, R4 async, persistent K/V) — backend-agnostic,
helps Vulkan too. The `cuda` feature is retained as a measurement tool +
opt-in Nvidia path; it is not promoted. Next probes: (a) same warm A/B at
B=8; (b) whether enabling cubecl-wgpu SPIR-V cooperative-matrix closes
even the ~1.5× matmul gap on the portable path.

**Artefacts:** `/tmp/warm_vulkan_b1.log`, `/tmp/warm_cuda_b1.log`,
`/tmp/gelo_cuda_b1.log` (cold) — capture to `bench-results/` next run.
Engine `cuda` feature: `crates/gelo-gpu-wgpu/{Cargo.toml,src/lib.rs}`;
warmup knob `GELO_BENCH_WARMUP` in the main bench.

## 9. Per-op A/B tables: CUDA vs Vulkan (B=1 and B=8)

Qwen3-4B n=2048 K=32, R3, RTX 5090. ◆ = GPU-executed (`engine:*`); all
other buckets run on the CPU in-TEE and are backend-invariant. B=1 is
warm-vs-warm. B=8 is **Vulkan cold vs CUDA warm** — fair because Vulkan
warm≈cold (SPIR-V autotune is cheap; verified at B=1: 25.8 vs 25.3 s),
whereas CUDA must be warm (nvrtc autotune is heavy: the B=8 CUDA warmup
prefill was 225 s vs the measured 197.6 s — a ~27 s one-time cost, vs
~1 s at B=1). Single sample each; ~7 % variance floor.

### B=1 — PREFILL (wall 25.8 s → 22.5 s, 0.87×)

| op | Vulkan ms | CUDA ms | Δ ms | CUDA/Vk |
|---|--:|--:|--:|--:|
| ◆ `engine:matmul_many` | 6865.0 | 5041.9 | −1823.0 | 0.73× |
| `tee:attn_cached` | 5036.0 | 5061.6 | +25.6 | 1.01× |
| ◆ `engine:matmul` | 4368.6 | 2574.4 | −1794.2 | 0.59× |
| `gelo:mask_unapply:dct4` | 2562.9 | 2606.4 | +43.6 | 1.02× |
| `gelo:strip_shield` | 2423.8 | 2589.8 | +166.0 | 1.07× |
| `gelo:mask_apply:dct4` | 1605.3 | 1602.1 | −3.1 | 1.00× |
| `gelo:shield_stack` | 380.0 | 417.1 | +37.2 | 1.10× |
| `tee:swiglu_activate` | 305.6 | 307.1 | +1.5 | 1.00× |
| `tee:qk_norm` | 254.3 | 254.2 | −0.1 | 1.00× |
| `tee:residual` | 158.1 | 159.8 | +1.6 | 1.01× |
| `tee:rmsnorm` | 120.9 | 127.8 | +6.8 | 1.06× |
| `tee:rope` | 67.7 | 65.3 | −2.4 | 0.96× |
| **TOTAL (Σ)** | **24151.7** | **20811.0** | **−3340.7** | **0.86×** |

### B=1 — DECODE (wall 17.4 s → 15.1 s, 0.87×; 1.84→2.12 tok/s)

| op | Vulkan ms | CUDA ms | Δ ms | CUDA/Vk |
|---|--:|--:|--:|--:|
| `tee:attn_cached` | 9148.3 | 8943.7 | −204.6 | 0.98× |
| ◆ `engine:matmul` | 2205.6 | 1247.3 | −958.3 | 0.57× |
| ◆ `engine:matmul_many` | 2115.2 | 1106.2 | −1009.1 | 0.52× |
| `gelo:mask_unapply:hd3` | 1886.0 | 1797.1 | −88.9 | 0.95× |
| `gelo:mask_apply:hd3` | 1067.9 | 1016.8 | −51.1 | 0.95× |
| `gelo:shield_stack` | 866.8 | 854.1 | −12.6 | 0.99× |
| `tee:compute_logits` (LM-head◆) | 327.1 | 214.1 | −113.1 | 0.65× |
| **TOTAL (Σ)** | **17686.3** | **15243.4** | **−2442.8** | **0.86×** |

### B=8 — PREFILL (wall 216.9 s → 197.6 s, 0.91×)

| op | Vulkan ms | CUDA ms | Δ ms | CUDA/Vk |
|---|--:|--:|--:|--:|
| ◆ `engine:matmul_many` | 68789.7 | 57617.2 | −11172.5 | 0.84× |
| `tee:attn_inplace_many` | 43773.7 | 43475.2 | −298.6 | 0.99× |
| ◆ `engine:matmul` | 39986.1 | 32031.7 | −7954.4 | 0.80× |
| `gelo:mask_unapply:dct4` | 27366.6 | 27312.2 | −54.4 | 1.00× |
| `gelo:mask_apply:dct4` | 11758.6 | 11589.1 | −169.5 | 0.99× |
| `tee:residual` | 4647.5 | 4812.4 | +164.9 | 1.04× |
| `gelo:shield_stack` | 4151.3 | 4112.7 | −38.5 | 0.99× |
| `tee:swiglu_activate` | 2196.7 | 2186.6 | −10.1 | 1.00× |
| `tee:qk_norm` | 1997.8 | 1998.5 | +0.7 | 1.00× |
| `tee:rmsnorm` | 894.9 | 900.5 | +5.6 | 1.01× |
| `tee:rope` | 547.2 | 543.3 | −3.9 | 0.99× |
| **TOTAL (Σ)** | **206181.7** | **186654.1** | **−19527.6** | **0.91×** |

### B=8 — DECODE (wall 40.6 s → 34.8 s, 0.86×; 0.79→0.92 tok/s/seq, agg 6.3→7.4)

| op | Vulkan ms | CUDA ms | Δ ms | CUDA/Vk |
|---|--:|--:|--:|--:|
| `tee:attn_cached_inplace_many` | 14574.4 | 14769.4 | +195.0 | 1.01× |
| ◆ `engine:matmul_many` | 9941.1 | 6377.6 | −3563.6 | 0.64× |
| ◆ `engine:matmul` | 8955.9 | 6860.6 | −2095.3 | 0.77× |
| `gelo:mask_unapply:hd3` | 3445.1 | 3173.3 | −271.8 | 0.92× |
| `tee:compute_logits` (LM-head◆) | 2525.5 | 1599.8 | −925.7 | 0.63× |
| `gelo:shield_stack` | 1563.9 | 1521.4 | −42.5 | 0.97× |
| `gelo:mask_apply:hd3` | 1533.3 | 1509.7 | −23.6 | 0.98× |
| **TOTAL (Σ)** | **42925.5** | **36184.0** | **−6741.4** | **0.84×** |

**Reading.** Identical conclusion at both batch sizes: CUDA moves only the
`engine:*` rows (prefill matmul ~0.73–0.84×, decode matmul ~0.52–0.77×,
i.e. ~1.2–1.9× faster kernels) and the R3 LM-head (`compute_logits`
~0.63–0.65×). **Every CPU bucket is ~1.00×** — `tee:attn*` is literally
within 1 % across backends at both B. Net wall lands at **0.84–0.91×**
(CUDA ~10–16 % faster), Amdahl-capped by the backend-invariant CPU
attention + mask buckets. The matmul advantage is slightly *smaller* at
B=8 (0.80–0.84× vs 0.59–0.73× prefill) — larger matmuls are more
compute-bound, so the Vulkan/SPIR-V kernels close some of the gap. None
of this is the step-change a tensor-core-absent Vulkan path would show;
the lever remains the backend-invariant in-TEE attention + per-call
round-trip, not the GPU backend.

**Artefacts (B=8):** `/tmp/gelo_b8_n2048.log` (Vulkan cold),
`/tmp/warm_cuda_b8.log` (CUDA warm).

## 10. Gate-1 persistent-K/V microbench (2026-05-29) — the upload tax is ~100% of it

Per the persistent-attention plan
([`perm-attn-gpu-offload.md`](../../plans/perm-attn-gpu-offload.md)),
gate 1 asks: does **device-resident K/V** (upload once; per-step upload
only Q) beat the in-TEE baseline? Added a `gpu_resident_b8` cell to
`amulet_attention_r1_4` (engine `upload_resident_kv` / `attend_resident`,
fp16, Vulkan) alongside the existing in-TEE and full-upload cells.

| n_kv (decode, B=8) | in-TEE rayon | full-upload (no_mask) | **resident** | resident vs in-TEE | resident vs full-upload |
|---:|---:|---:|---:|---:|---:|
| 256  | 1.08 ms | 69.8 ms  | **0.30 ms**  | 3.6× faster | 233× |
| 1024 | 5.28 ms | 271.8 ms | **0.373 ms** | 14× faster  | 728× |
| 2048 | 11.15 ms| 489 ms   | **0.465 ms** | **24× faster** | 1052× |

**Gate 1 passes by 24× at the production shape.** Two measured findings:

- **The "fixed-overhead-bound" hypothesis is confirmed, not inferred.** At
  n_kv=2048, full-upload 489 ms − resident 0.465 ms ≈ **99.9%** of the
  no-mask cost is the per-call K/V upload + f32→f16 convert + staging +
  blocking sync — the exact term persistent K/V deletes. The 5090's HBM
  read + kernel is ~0.5 ms; the upload pipeline was ~100% of the §3 triage
  cost, ~0% the compute. (Corrects §4.2's bandwidth framing definitively.)
- **The win grows with context** (3.6× → 14× → 24× as n_kv 256 → 2048):
  resident reads at HBM ~1.8 TB/s while in-TEE scales with DDR5 ~85 GB/s.
  Resident is sub-linear in n_kv (0.30 → 0.47 ms for 8× the keys) — per-step
  cost is fixed dispatch / Q-upload / readback, not the HBM read — so it
  stays cheap into the long-context regime the design targets.

### 10.1 Representative decode — per-step append, prefill-only re-permute

The `gpu_resident_append_b8` cell adds the realistic growing-cache cost:
the cover is applied **once at prefill** (`create_kv_session`); each step
*appends* the new token's K/V row (`append_kv`, O(1) `slice_assign`) and
attends over `[0..len]` (`attend_session`) — **no per-block re-permute**.
This is the **optimistic prefill-only case** (N = ∞) the security gate
will later test: we benchmark it first to decide if the approach is worth
building at all.

| n_kv (decode, B=8) | in-TEE | resident (attend only) | **append + attend** | append overhead | vs in-TEE |
|---:|---:|---:|---:|---:|---:|
| 256  | 1.35 ms  | 0.298 ms | **0.500 ms** | +0.20 ms | 2.7× |
| 1024 | 4.44 ms  | 0.379 ms | **0.572 ms** | +0.19 ms | 7.8× |
| 2048 | 10.88 ms | 0.463 ms | **0.662 ms** | +0.20 ms | **16.4×** |

- **Append is O(1):** ~0.20 ms *constant* across n_kv → `slice_assign`
  mutates the resident buffer in place (no O(n) recopy). The growing-cache
  per-step cost is flat.
- **Worth it, decisively:** the optimistic prefill-only case, *with* the
  per-step append, is **16.4× faster than in-TEE at production n=2048**,
  scaling 2.7× → 16.4× with context. Even the simplest persistent design
  (zero decode re-permute) is a large win → the approach is worth building.
  The open question collapses to a security one: does the gate permit
  prefill-only (this best case), or force periodic re-permute (still a
  likely win per the §10 upload-optimization model)?

**The re-permute half is conditional.** `no_mask − resident` isolates the
K/V convert+upload exactly = **~488 ms @ n=2048** on the *current* pipeline
(GQA-expanded, f32→f16). That is the per-block re-permute *upload* tax;
amortized over N=16 it is ~30 ms/step — it would **lose** to the 11 ms
in-TEE baseline. So persistence wins the per-step read unconditionally
(0.465 ms) but wins the re-permute half **only after** the upload is
optimized: un-replicated storage (4× less) + bf16-native K/V (no convert)
→ modeled ~5 ms ÷16 ≈ 0.3 ms/step. That optimization is part of the
substrate refactor.

**Remaining gaps (after §10.1).** Per-step append is now measured (O(1),
~0.20 ms). What's left unmodelled is **by design**: the §10.1 cell is the
optimistic *prefill-only* case (no decode re-permute) — whether that's
security-achievable is the gate's job, and the re-permute fallback cost is
the §10 upload-optimization model. Also: no permutation / σ-noise / `O_v`
in the per-step path (they're prefill-one-time, trivial for Q); full
softmax over the whole active slice (no prefix/tail merge — a wash);
GQA-**expanded** K/V (un-replicated is 4× less → faster); fp16 Vulkan
(CUDA prod kernel ~1.5–1.9× faster, §9).

**Artefacts:** `bench-results/amulet-attn-resident-5090-2026-05-29.log`
(§10) + `bench-results/amulet-attn-append-5090-2026-05-29.log` (§10.1);
cells `gpu_resident_b8` / `gpu_resident_append_b8` in
`crates/gelo-gpu-wgpu/benches/amulet_attention.rs`; engine seams
`ResidentKvF16` (`upload_resident_kv` / `attend_resident`) and
`ResidentKvSession` (`create_kv_session` / `append_kv` / `attend_session`)
in `crates/gelo-gpu-wgpu/src/lib.rs`.

## 11. Wired decode path (2026-05-29) — the microbench-to-forward gap is dispatch, not attend

The §10.1 cell is an *isolated* attend. §11 is the same engine seam
threaded through the real forward (`decoder_block_cached_batched`,
GLOBAL layers, behind `GELO_GPU_RESIDENT_ATTN`; in-TEE default). Same
production bench as §9 (`gelo_llm_prefill_decode_breakdown`, 4b, B=8,
N=2048, **K=32**, Vulkan). Greedy-parity holds byte-for-byte off-vs-on.

| B=8 decode, Vulkan | in-TEE (§9 baseline) | GPU-resident | Δ |
|---|--:|--:|--:|
| **attn bucket** | 14 574 ms (`tee:attn_cached_inplace_many`) | **8 724 ms** (`tee:attn_resident_gpu`, 1152 calls) | **0.60×** |
| per-call attn | 12.65 ms | 7.57 ms | 1.67× |
| decode wall | 40.6 s | 28.81 s | 0.71× **(confounded — see below)** |
| tok/s/seq | 0.79 | 1.11 | 1.40× |

**Read the bucket, not the wall.** The attention bucket — the only one
the flag touches — is the clean number: **0.60× (−5.85 s)**. The decode
wall fell 40.6→28.81 s, but `engine:matmul` (8956→4382) and
`matmul_many` (9941→8670) also moved between these *separate-session*
runs, and the flag cannot touch the projection/LM-head matmuls. That is
cross-run autotune/thermal variance (§5 confounds), not the wire-up.
**Do not attribute the −11.8 s wall to this change** without a
same-session off/on A/B.

**Why 1.67×, not the §10.1 16.4×.** §10.1 measured a lone `attend`
(0.662 ms/step incl. append). The wired path is 7.57 ms per *(layer,
step)* — the gap is integration overhead the microbench omits, paid
×36 layers ×32 steps: `stack_heads` (f32→f16 of the full Q block per
step), GQA-broadcast attend over the growing slice, `unstack_heads`
(f16→f32 of ctx), and a TEE↔GPU round-trip *serialized per layer*. The
decode loop is **dispatch-bound**, exactly the Phase-3 concern. The
next lever is the fused FlashAttention-D kernel + collapsing per-step
dispatches — not raw attend throughput, which §10.1 already showed is
ample.

**Artefact:** `bench-results/phase4-resident-decode-5090-2026-05-29.log`;
wire-up in `crates/gelo-embedder/src/decoder/forward.rs`
(`decoder_block_cached_batched`, `stack_heads`/`unstack_heads`),
`KvCache::gpu_sessions`, `TrustedExecutor::resident_kv_*`.
