---
type: dev-log
status: current
created: 2026-05-18
updated: 2026-05-26
tags: [gelo, perf, mask, bf16, hd3, dct4, blis, attention, batched, chronicle]
companion: [gelo-mask-perf-log, 2026-05-18-m1-10-perf, 2026-05-19-bf16-mask-deferred, 2026-05-19-hd3-followups, 2026-05-21-attn-offload-spike, 2026-05-21-gelo-perf-shield-attn-batched, 2026-05-22-dgpu-attention-revival, 2026-05-22-perf-bucket-roadmap-r3-default, 2026-05-22-q3-4b-b8-mask-sweep, 2026-05-26-mask-instrumentation-and-auto-tune, 2026-05-26-r4-greenlight-bf16-aborted, m1-10-fused-permuted-attention, m1-10-security-review, m1-10-phase4-findings, m1-11-batched-decode, m1-12-tee-gpu-throughput, m1-12-blis-thread-dispatch, m1-12-bf16-activation-pipeline, m1-12-permuted-attention-batched-decode, m1-12-r4-async-overlap, gelo-llm-perf-roadmap]
---

# GELO-LLM perf chronicle — comprehensive

> One-stop reference for everything GELO-LLM performance/optimization-related:
> the protocol primitive cost model, the mask-family hierarchy and decision
> history, R-series perf-bucket outcomes, the dated chronicle of every
> optimization tried, dGPU revival design, methodology discipline, and open
> levers. Distilled from ~22 handoffs/plans/prototype docs + bench-results
> raw artefacts.
>
> The dated chronicle in §4 is the spine; everything else is reference
> material for interpreting those entries.

## Contents

1. [Background](#1-background)
2. [Protocol primitives — the cost model](#2-protocol-primitives--the-cost-model)
3. [Mask family hierarchy + Auto-dispatch decision](#3-mask-family-hierarchy--auto-dispatch-decision)
4. [Dated chronicle](#4-dated-chronicle)
5. [R-series perf-bucket outcomes](#5-r-series-perf-bucket-outcomes)
6. [dGPU attention revival design](#6-dgpu-attention-revival-design)
7. [Methodology + variance discipline](#7-methodology--variance-discipline)
8. [Open levers + next priorities](#8-open-levers--next-priorities)
9. [Cross-references + artefacts](#9-cross-references--artefacts)

---

## 1. Background

### Workload anchor

The production extraction shape is **Qwen3-4B GELO inference on Strix Halo iGPU (Radeon 8060S, fp16 wgpu Vulkan + AOCL-BLIS) at (B=8, n=2048, K=32–64 decode tokens)**. Smaller models (Qwen3-1.7B) and embedding workloads (Qwen3-Embedding-0.6B) appear in earlier baselines.

### Hardware substrate

- **iGPU UMA (development):** AMD RADV gfx1151 (Strix Halo), Mesa 25.2.8; AMD Zen 5 CPU; DDR5 shared with GPU; AOCL-BLIS multi-threaded for CPU mask GEMM (~1.25 TFLOP/s f32 at 16 threads).
- **dGPU (production, hardware-gated):** SEV-SNP CVM + VFIO dGPU (PCIe Gen4 ~30 GB/s realised; HBM ~3 TB/s kernel-read). 100× kernel/upload ratio vs iGPU's 4×.

### Perf glossary

| Term | Meaning |
|---|---|
| TTFT | Time-to-first-token (prefill wall) |
| TPOT | Time-per-output-token (decode wall) |
| `n` | Sequence length |
| `B` | Batch size |
| `K` | Decode tokens generated |
| `k` (shield-k) | Shield rows appended to data (k=8 baseline; k=15 at decode m=1 with shape-adaptive shield) |
| `s = n + k` | Total operand rows after shield-stack |
| `s_pad` | Pow2-padded operand for HD₃ (`s.next_power_of_two()`) |
| `d` | Model hidden size (Qwen3-1.7B: 2048; -4B: 2560; -8B: 4096) |
| `d_h` | Per-head dim (Qwen3-1.7B: 128) |
| `Auto` | Dispatcher: HD₃ at pow2 / pad ratio ≤ 1.6, DCT-IV elsewhere, Haar at pow2 fallback |
| `F1+` | TEE-resident softmax with `-C = 30` replacing `-∞` to close causal-mask leak |

### Variance floor

Single-cell variance at production shape (B=8 n=2048 long-context): **~7 %** on Strix Halo UMA. Three clean runs of `tee:attn_cached` landed at 77.5 / 69.9 / 89.7 s on identical fixture (characteristic ±15 % band). Any single-cell EV claim ≥7 % requires variance-sweep validation (~80 min).

---

## 2. Protocol primitives — the cost model

### Mask round-trip cost (the load-bearing overhead)

The GELO protocol applies a fresh orthogonal mask `A ∈ O(s)` per batch to hide activations crossing the PCIe boundary. At each layer:

1. **`mask_sample`** — sample fresh `A` (Haar QR / HD₃ ±1 / DCT-IV cascade). 1 call per forward (paper-parity default per `gelo.md` §3.2).
2. **`mask_apply`** — compute `U = A · H` via CPU GEMM (`2·s²·d_in` FLOPs). 4 calls/layer × 28 layers = 112 calls per forward (QKV bundle, O, gate-up bundle, FfnDown).
3. **`mask_unapply`** — recover `H' = Aᵀ · V` from masked GEMM output (`2·s²·d_out` FLOPs). 7 calls/layer × 28 layers = 196 calls per forward (Q/K/V × 3, O × 1, gate × 2, down × 1).

**Per-layer FLOP cost (Qwen3-1.7B, n=2048):**
| Offload | apply (d_in FLOP) | unapply (d_out FLOP) | FLOPs/layer |
|---|---|---|---|
| QKV (one mask) | 2048 d | 3 × {2048, 1024, 1024} d | ~51 GFLOPs |
| O (d=2048) | 2048 d | 1 × 2048 d | ~34 GFLOPs |
| gate∥up (d=6144) | 2048 d | 2 × 6144 d | ~119 GFLOPs |
| FfnDown (d=2048) | 6144 d | 1 × 2048 d | ~68 GFLOPs |
| **Per-layer total** | | | **~272 GFLOPs** |

× 28 layers = **7.6 TFLOPs of CPU BLIS matmul per forward**. At baseline single-thread (~50–75 GFLOPs/sec) → ~50–75 s. Observed M1.10 baseline overhead matches exactly.

### F1+ in-TEE softmax (causal-mask leak fix)

**Problem:** Original `permuted_attention` leaked the causal pattern π via the `-∞` entries in the masked score tensor sent to GPU for softmax. GPU adversary counts exact-zero post-softmax entries per row to recover π (exact-zero recovery attack).

**Solution (landed 2026-05-18, M1.10 Phase 0):**
- Move softmax in-TEE.
- Replace `-∞` with `-C = 30` so blocked positions softmax to ~exp(−30) ≈ 1e-14 (non-zero at f32 precision).

**Cost:** One PCIe round-trip on the score tensor per call (~64 MB at n=2048 prefill). Softmax CPU work negligible.

**Residual risk:** Threshold-count attack still recovers π (count probs < 1e-12 per row). Documented as F1++ (not implemented); would require adding small Gaussian noise on probs before GPU return.

**Test anchors:** three `f1plus_*` regression tests in `permutation_attention.rs`.

### Permuted attention + `use_perm_attention` opt-in

Exploits softmax-permutation equivariance: `softmax(π·Q·Kᵀ·πᵀ / √d) · π·V = π · softmax(Q·Kᵀ / √d) · V`. Allows Q·Kᵀ matmul + softmax + Attn·V on GPU under permutation masking instead of full orthogonal GELO masking.

**Production default:** Off (`cfg.use_perm_attention = false`).
**Cost when engaged:** K/V perm copies dominate (200–300 ms at n=2048); Gaussian noise (rayon-parallel) secondary.
**TPOT win post-rayon (commit cbea549):** 1693 → 1063 ms = 1.59× at n=2048 decode. Not 10–30× because rayon overhead ~6 ms fixed cost across 56 per-decode calls; per-element Ziggurat still scalar.

### Shield rows (TwinShield)

- **Default production config:** k=8 shield rows at energy σ=4.0 (TwinShield, Xue et al. 2025).
- **Shape-adaptive shield (landed 2026-05-21):** k=15 at small n ≤ 1 (decode); k=8 elsewhere.
- **Security purpose:** Gram leak `UᵀU = HᵀH + SᵀS` confounds anchor-recovery attacks (BSS-style); FastICA cannot recover `HᵀH` alone under shielding (verified in `tests/bss_recovery.rs`).
- **Separate RNG stream:** Xoshiro256++ for shield (3× faster than ChaCha20; security-safe — shield rows stripped post-forward, never propagate to logits). Mask `A` itself still from ChaCha20.

### SCX-style decode KV-cache encoding (forward-looking)

Per-batch fresh π for permuted-attention doesn't smoothly extend to autoregressive decode (KV cache would need per-step re-permutation, hundreds of MB/token). Forward-looking alternative: **SCX (Yuan et al., SIGCOMM 2025)** — stateless KV-cache encoding with per-user keys derived from `(session_id, layer_id, position)`. Identified as natural complement to permuted-attention prefill for LLM serving. Not yet implemented.

### Struck designs

- **On-GPU unmask `Aᵀ` (struck 2026-05-18, commit `68cd468`):** Putting `A` on GPU as a weight lets hostile engine compute `H = Aᵀ · masked_h` and recover plaintext H within one forward. Threat-model-incompatible; only valid under confidential-GPU (H100 CC), which GELO does not target.
- **Fused-attention kernel Phase 2 (deprecated 2026-05-18):** Under F1+ causal mask must not reach GPU; four work-arounds all fail. Phase 4 bench confirmed attention compute is NOT the bottleneck; fused-flash would not move long-context numbers.

---

## 3. Mask family hierarchy + Auto-dispatch decision

### Haar (baseline, dense)

- **Cost:** O(s² · d) FLOPs per apply/unapply.
- **Sampling:** Per-forward QR via Householder reflection — O(s³) scalar ops.
- **Overhead at M1.10 n=2048:** ~50–75 s per 28-layer forward.
- **Security:** κ=1 orthogonality exact at f32; pair with shield rows σ=4.0.
- **Status:** Baseline; Auto-dispatcher falls back to Haar at pow2 when neither HD₃ nor DCT-IV is preferred.

### HD₃ (Hadamard cascade, QuIP#/QuaRot primitive)

- **Form:** `A = D₃·H·D₂·H·D₁·H` (3·s ±1 fresh sign bits per mask; each `Dᵢ` a ±1 diagonal).
- **Cost:** Three FWHTs — O(s·d·log s) vs O(s²·d) dense.
- **SIMD:** AVX-512F 16 f32/inst, AVX-2 8 f32/inst fallback, rayon-parallel above 65 K elements.
- **At pow2 alignment:** −28 % TTFT at n=2040 (s=2048 exactly pow2).
- **Non-pow2 cost:** +51 % TTFT at n=2048 (s=2056 → pad to 4096 → 2× matmul & FWHT cost).
- **Orthogonality:** Exact at f32 (same 10⁻⁶ round-trip error as dense Haar).
- **Per-call timing at production B=8:**
  - apply: 192 µs/call (Haar 358 µs) = −46 %
  - unapply: 191 µs/call (Haar 290 µs) = −34 %
- **Security status:** Research-grade, opt-in. AloePri attack-suite re-validation (c3_hd3) pending before default-flip. QuIP# incoherence proof carries over (≥ 2^(3s) orbit size); empirical BSS-hardness under GELO threat model needs formal gate.

### DCT-IV (tile-fused cascade, production workhorse)

- **Form:** Replace each Hadamard `H` in HD₃ with DCT-IV — real-valued, orthogonal, non-pow2-capable.
- **Cost:** O(s·d·log s) via FFT, orthogonal at any N (no padding needed).
- **Material advantage:** tile-fused cascade landed 2026-05-26 (`cd1a008`) wins **−22.7 % prefill wall** (174.9 s → 135.1 s at B=8 n=2048).
- **Why it wins at production shape:** Column-locality fuse captures L2-cache tile reuse; HD₃ FWHT is more memory-bandwidth dominated.
- **Security:** Row entries are `√(2/n)·cos(...)` — same O(1/√n) incoherence bound as Hadamard. No row-sum leak (unlike DCT-II). Depth-k mixing identical to HD₃; inherits same multi-anchor resistance.
- **Status:** Current production default at non-pow2 (2026-05-26).

### Block-diagonal HD₃ (struck on security grounds)

Block-diagonal HD₃ appears structurally sound but falls to a **multi-anchor attack** when adversary observes multiple anchored data rows per block.

**Threat model (realistic for GELO):** system prompt is boilerplate (20–50 tokens) prepended to every user query. Those tokens project to specific known hidden-state rows after embedding + first-layer normalization. For k anchored rows in block i with `3·b_i` sign-bit unknowns and output dimensionality d ≈ b_i, the attacker has k·d scalar constraints — over-determined at k ≥ 2. Solvable via linearization attacks (Albrecht-Cid-Faugère 2009) in polynomial time. At 1000+ queries with one anchored token per query, thousands of over-determined rows per block → trivial to solve.

**Verdict:** Block-diagonal HD₃ at any block size satisfying `d > 1.5·b_i` is cryptanalytically broken. Removed from Phase 2 plan.

### Auto-dispatch strategy

Rules (`HD3_AUTO_MAX_PAD_RATIO_NUM = 8`, threshold 1.6, tuned 2026-05-26):

- At pow2 operand shape: use Haar (trivial, no padding).
- At non-pow2 with pad ratio `(n+k).next_pow2() / (n+k) ≤ 1.6`: HD₃ (FWHT slightly better than dense at modest padding).
- At higher pad ratio: DCT-IV (no padding, asymptotically better).

Sweep confirmed HD₃ wins measurably up to pad ratio 1.59; DCT-IV preferable above. Threshold empirically validated across {B=1, B=8} × {n=512, 1024, 2048, 4096}.

**Per-family profile categories (2026-05-26 instrumentation):** distinct `gelo:mask_apply:{haar,hd3,dct4}` + `gelo:mask_unapply:*` buckets emit so Auto-resolution visibility is complete.

---

## 4. Dated chronicle

### 2026-05-18 — M1.10 Phase 4 baseline + F1+ landing

**Source:** `2026-05-18-m1-10-perf.md`, `m1-10-fused-permuted-attention.md`, `m1-10-security-review.md`

**Config:** Qwen3-1.7B fp32, n ∈ {64, 512, 2048}, B=16 (per-head), Haar mask, shield k=8 σ=4.0, Strix Halo iGPU + single-thread BLIS (pre-step-1).

**Bench — `qwen3_long_context_bench.rs` (cached path):**

| Cell | n | TTFT | TPOT | Wall | vs gpu_plain |
|---|---:|---:|---:|---:|---:|
| gpu_plain | 64 | 231 ms | 271 ms | 2.2 s | (baseline) |
| gpu_gelo | 64 | 372 ms | 271 ms | 3.6 s | +44.5 % |
| gpu_plain | 512 | 4 064 ms | 154 ms | 5.4 s | (baseline) |
| gpu_gelo | 512 | 7 074 ms | 206 ms | 9.3 s | +55.9 % |
| gpu_plain | 2048 | 6 443 ms | 271 ms | 10.8 s | (baseline) |
| **gpu_gelo** | **2048** | **72 904 ms** | **371 ms** | **78.8 s** | **+631 %** |
| **gpu_gelo_permuted** | **2048** | **87 528 ms** | **1 063 ms** | **104.5 s** | **+870 %** |

**Optimizations landed:**
- **F1+ (in-TEE softmax + `-C = 30`)** — closes causal-mask leak; ~6× speedup on attention slice (~7s in-TEE BLIS → ~1.1s F1+).
- **Phase 1 `permuted_attention_cached` wiring** — opt-in via `cfg.use_perm_attention = true`; default off; parity test green at σ=0.

**Optimizations struck:**
- **On-GPU unmask `Aᵀ`** — threat-model-incompatible (commit `68cd468`).
- **Fused-attention kernel Phase 2** — F1+ incompatible with GPU softmax; attention not the bottleneck.

**Bottleneck diagnosis:** +631 % overhead is **not attention** — it's mask round-trip on linear projections (272 GFLOPs/layer × 28 layers = 7.6 TFLOPs CPU BLIS at ~100–150 GFLOPs/sec → ~50–75 s). Matches observed Δ exactly.

**Handoff target:** rayon-parallel Gaussian noise (task #68), F1+ end-to-end validation, decide on Phase 2 timing.

---

### 2026-05-19 (morning) — Step 1: BLIS-mt 5.04×

**Source:** `2026-05-18-m1-10-perf.md` §2.1, `2026-05-19-bf16-mask-deferred.md` §1

**Config:** Qwen3-1.7B n=2048 prefill; default-flip `blis-src` single-thread → multi-thread (16 cores via `GELO_BLIS_THREADS=16`).

**Bench:**
- Pre-step-1: TTFT 73.0 s (single-thread BLIS).
- Post-step-1: TTFT **14.5 s** = **5.04× speedup**.

**Sweep across `GELO_BLIS_THREADS ∈ {1, 4, 8, 16}` at n=2048 with `blas` feature + long-context cell:**

| threads | gpu_gelo TTFT | overhead vs plain | speedup vs matrixmultiply |
|---|---:|---:|---:|
| matrixmultiply (no blas) | 73.0 s | +1 020 % | 1.00× (baseline) |
| BLIS 1 | 39.8 s | +477 % | 1.83× |
| BLIS 4 | 19.5 s | +193 % | 3.74× |
| BLIS 8 | 16.4 s | +147 % | 4.45× |
| BLIS 16 | **14.5 s** | **+119 %** | **5.04×** |

Diminishing returns past 8 threads. Even BLIS-1 is 1.83× faster than matrixmultiply at same shape, so embedder regime (threads=1 default) is strictly improved by enabling `blas`. Combined matrixmultiply → BLIS-mt-16 drops headline overhead from +1 020 % to +119 %.

**Per-bucket profile post-step-1 (n=2048 prefill):**

| Bucket | Share |
|---|---:|
| `gelo:mask_sample` (Haar QR, single-thread) | 22.8 % |
| `engine:matmul_many` (GPU) | 18.6 % |
| `gelo:mask_unapply` (BLIS-mt-16) | 16.3 % |
| `engine:matmul` (GPU) | 13.1 % |
| `gelo:mask_apply` (BLIS-mt-16) | 9.8 % |
| `tee:attn_cached` (in-TEE GQA) | 9.5 % |
| `gelo:strip_shield` | 5.8 % |

**Embedder cliff confirmed:** at threads=8 the embedder bench regresses +42.6 %, at threads=16 by +1 540 %. Defaults stay at threads=1 for embedder; long-context opts in via env var.

**Layer-skip side-experiment regressed +6.6 s net** when attempted. Root cause: in-TEE `tee:*_direct` matmuls do NOT benefit from BLIS-mt — route through single-threaded `ndarray.dot()` → `matrixmultiply`. Blocked further layer-skip optimization until `tee:*_direct` routed through `cblas_sgemm`.

---

### 2026-05-19 (mid-morning) — bf16 mask GEMM deferred

**Source:** `2026-05-19-bf16-mask-deferred.md`

**Theoretical potential:** 1.6–1.8× GEMM throughput via AVX-512_BF16 (`VDPBF16PS`).
**Measured saving at M1.10 narrow scope:** ~10 % TTFT at n=2048 (−1.3 s on 14.5 s baseline).

**Parity simulation (`crates/gelo-protocol/tests/bf16_mask_parity.rs`):**
- Scale-invariant (7-bit mantissa floor).
- Mean rel error bf16 mask vs f32: **2.65 × 10⁻³**.
- Mean rel error bf16 mask vs bf16 everywhere: 1.87 × 10⁻³.
- Within paper Table 1 band (≥98.8 % top-1 token equality).

**Blocker (vendor AOCL):** `libblis-mt.so.5.2.2` has 3 168 symbols; zero `lpgemm/bf16/aocl_gemm` matches. AOCL-BLIS 5.2.2 detects AVX-512_BF16 at runtime but has no kernel behind it.

**Path comparison:**

| Approach | Effort | Expected gain | Risk |
|---|---|---|---|
| **OpenBLAS `cblas_sbgemm`** | ~1 day | 1.6–1.8× | Low (Zen5 → COOPERLAKE / AVX512_BF16 since v0.3.13) |
| AOCL-DLP separate lib | 1 day–1 week | ≥ OpenBLAS | Medium (availability) |
| Rebuild AOCL with lpgemm | 1 week | same | Medium (build-system surgery) |
| Hand-rolled AVX-512_BF16 | 1–2 weeks | matches OpenBLAS | Medium |
| Intel MKL | 1 day | matches OpenBLAS | **High on AMD** (cpuid downclocking) |

**Why deferred:** HD₃ subsumes the gain. HD₃ replaces dense O(s²·d) GEMM with O(s·d·log s) FWHT; 25× FLOP reduction dwarfs bf16's 1.6× throughput. Compound HD₃ + Q4-weight stack reaches paper-target ~5–6 s TTFT (QuIP#/QuaRot primitive). **bf16 on the mask is dead-end w.r.t. weight quantization.**

---

### 2026-05-19 (afternoon) — HD₃ landed (opt-in, non-pow2 regression at n=2048)

**Source:** `2026-05-19-hd3-followups.md`

**Config:** Qwen3-1.7B n ∈ {2040, 2048} prefill at B=16, 4-token greedy decode. Haar (baseline) vs HD₃ Hadamard cascade.

**Bench:**

| Shape | Haar TTFT | HD₃ TTFT | Δ | Overhead vs gpu_plain |
|---|---:|---:|---:|---:|
| **n=2040** (s=2048 exact pow2) | 14.97 s | **10.72 s** | **−28 %** | Haar +138 %, HD₃ **+78 %** |
| **n=2048** (s=2056 → pad to 4096) | 15.40 s | 23.30 s | **+51 %** | Haar +120 %, HD₃ +255 % |

**Per-bucket diagnosis at n=2048 (problem case):**
```
gelo:mask_sample    3 134 ms → 0.01 ms   (−3.13 s ✓ Haar QR gone)
gelo:mask_apply     1 492 ms → 3 001 ms  (+1.51 s ↑ FWHT at 2× padded)
gelo:mask_unapply   2 626 ms → 5 860 ms  (+3.23 s ↑ ditto)
engine:matmul_many  2 600 ms → 6 378 ms  (+3.78 s ↑ GPU does 2× rows)
engine:matmul       1 849 ms → 4 038 ms  (+2.19 s ↑ same)
                                          +7.6 s net
```

**Implementation:** `Hd3Mask` primitive in `crates/gelo-protocol/src/hd3.rs`; `MaskFamily` enum wrap; bench knob `GELO_BENCH_MASK_KIND=hd3`. 7 tests green.

**Phased plan (revised):**
1. **Auto-dispatch hybrid** (priority 1, ~2 hours): HD₃ at pow2, Haar elsewhere. Eliminates regression risk.
2. **DCT-IV cascade** (priority 2, 5–7 weeks research + impl): replaces Haar fallback at non-pow2. ~10–15 % additional win.
3. **Cascade-depth tuning** (priority 3): audit k=3 vs k=4/5 multi-anchor attack resistance.

**Security gate (B.3):** AloePri attack-suite + GELO §4.3 attacks (anchor-ICA, JADE, JD, Gram error) required before default-flip.

---

### 2026-05-19 (evening) — Rayon-parallel Gaussian (TPOT 1.59×)

**Source:** `2026-05-18-m1-10-perf.md` §2.3, `m1-10-phase4-findings.md`

**Optimization:** `add_gaussian_3d_inplace` rayon-parallel (commit `cbea549`). Split heads axis above 32 K-element threshold; each head gets independent ChaCha20 stream from pre-derived seed; parent RNG advanced deterministically.

**Bench — `gpu_gelo_permuted` TPOT:**

| Shape | Pre-opt | Post-opt | Speedup |
|---|---:|---:|---:|
| n=64 | — | — | 1.17× |
| n=512 | — | — | 1.47× |
| **n=2048** | **1 693 ms** | **1 063 ms** | **1.59×** |

**Why not 10–30× projected:** rayon work-stealing ~6 ms fixed cost across 56 per-decode calls; effective parallelism ~8 cores (not 16); per-element Ziggurat still scalar.

**Next bottleneck identified:** K/V permutation copies (single-threaded scalar memcpy, ~200–300 ms at n=2048). At n_kv=2048: 16 heads × 2048 positions × 128 head_dim × 4 bytes × 2 tensors × 28 layers = ~900 MB memory traffic/decode step, all single-core. Queued as task #69 (~½ day, expected 1.3–1.5× TPOT gain).

---

### 2026-05-21 — Attention offload spike (decode m=1 aborted)

**Source:** `2026-05-21-attn-offload-spike.md`

**Microbench: in_tee vs perm_softmax_tee vs perm_softmax_gpu (A1+A2):**

| Shape | in_tee | perm_softmax_tee | perm_softmax_gpu (A1+A2) |
|---|---:|---:|---:|
| n_kv=256 | 0.66 ms | 4.35 ms | 3.98 ms |
| n_kv=1000 | 2.08 ms | 24.49 ms | 22.24 ms |
| n_kv=2000 | 4.28 ms | 48.47 ms | 43.92 ms |

**cubek-attention Unit strategy spike (decode shapes):**

| Shape | n_q | n_kv | cubek steady-state | accuracy |
|---|---:|---:|---:|---|
| decode | 1 | 1000 | 17.9 ms | max_abs 3.3e-3 ✓ |
| prefill | 64 | 64 | 1.24 ms | max_abs 1.1e-2 ✓ |
| prefill_long | 745 | 745 | 15.1 ms | max_abs 2e-6 ✓ |

**E2E Phase 1b enabled (PHASE_1B_DECODE_AMULET=1) — 2.6× regression:**

| Metric | Baseline (bc47d04) | Phase 1b enabled |
|---|---:|---:|
| `tee:attn_cached` (s) | 89.7 (39 %) | 0 |
| `tee:attn_permuted_cached` (s) | 0 | 629.8 (45 %) |
| Per-call attention (ms) | 4.85 | 32 (6.6× slower) |
| **Generate wall** | **343 s** | **903 s (+163 %)** |

**Verdict:** GPU dispatch-latency floor ~10–20 ms (in-TEE GEMV: 2 ms). At decode m=1, no GPU strategy beats in-TEE. **Batched decode is the unlock, not single-sequence GPU offload.**

---

### 2026-05-21 — Shield SIMD wins (3.08× cumulative)

**Source:** `2026-05-21-gelo-perf-shield-attn-batched.md`

**v7 baseline (single-stream extraction, Qwen3-4B + Qwen3-Embedding-0.6B, 7-chunk doc):** wall 361.2 s, `tee:attn_cached` 32 % (89.7 s).

**Commit `144d764` (SIMD Box-Muller via `wide::f32x8`):**

| shape | legacy_scalar | fill_gaussian | speedup |
|---|---:|---:|---:|
| d=2560 / k=15 (decode) | 551.74 µs | **345.00 µs** | **1.60×** |
| d=2560 / k=8 (prefill) | 306.40 µs | **196.09 µs** | **1.56×** |

E2E: shield_stack 486 → 307 µs/call (−37 %); wall 361.2 → 341.6 s (−5.4 %).

**Commit `3eca59e` (polar rejection + Xoshiro256++):**

| variant | µs/call |
|---|---:|
| legacy_scalar (Ziggurat + ChaCha20) | 214 |
| fill_gaussian Box-Muller + ChaCha20 | 148 |
| fill_gaussian polar + ChaCha20 | 154 (tied) |
| **fill_gaussian_xoshiro polar + Xoshiro** | **61** |

E2E: shield_stack 307 → 158 µs/call (−48.5 %); 3-run mean wall ~330 s.

**Cumulative (v7 → post-3eca59e):**

| metric | v7 | post-3eca59e | factor |
|---|---:|---:|---:|
| shield_stack µs/call | 486 | 158 | **3.08×** |
| shield_stack bucket | 35.9 s (14.8 %) | 11.67 s (5.1 %) | **−67 %** |
| bucket rank | #4 | **#6** | — |

**Synergy requirement:** polar method needs 1.4× RNG bytes (rejection rate 21.5 %); only wins paired with fast RNG. Landing both methods in one commit ensures synergy.

**Xoshiro security:** shield rows are post-stripped, never propagate to logits; Xoshiro256++ passes BigCrush; AloePri c2_default re-run pending before production attestation.

**Production-shape attn bucket** became dominant decode bucket at 39 % of wall (after shield optimizations).

---

### 2026-05-21 — M1.11 batched-decode design (plan, not implementation)

**Source:** `m1-11-batched-decode.md`

**Designed substrate:**
- KV cache layout `(B, layers, max_cache_len, kv_dim)` with per-sequence `len: Vec<usize>`.
- Per-sequence `A_b` at prefill (default) or shared dense `A` at decode (opt-in, gated).
- Shape: `(B + k_base).next_power_of_two().saturating_sub(B).max(k_base)` shield rows ensure `stacked_n` hits pow2 (HD₃ everywhere). Worst case k = 2·k_base − 1 = 15.
- New `SessionKind::{Single, PerSequence}` API with `begin_prefill_pass(batch_size, n_max)`, `begin_decode_pass(batch_size)`, `end_pass()`.

**Projected wins (extrapolated, NOT measured):**

| Config | Sequences | Projected wall |
|---|---:|---:|
| Sequential | 16 pairs | 16 × 155 ms = 2.48 s |
| Rayon (today) | 16 | ~620 ms (4 cores) |
| **Batched (R1–R3)** | 16 | **~280 ms** (8.9×) |

**Decoder generate (D1–D3, projected):**

| Config | 7 chunks |
|---|---:|
| Sequential v7 | 7 × ~343 s = 2 401 s |
| Batched B=7, per-seq A_b (default) | ~700–900 s |
| Batched B=7, shared-A (opt-in, post-c5 gate) | ~480 s |

**Crossover hypothesis (B ≈ 11–16) — falsified by 2026-05-22 spike below.**

**Rollout sequencing:** R1 (~4 days, `run_batched`), R2 (~1 day, rerank uses it), R3 (~2 days, parity + AloePri c4 gate). Then D1 (~5 days, KV cache + generate_batched), D2 (~2 days, extraction batched), D3 (~3 days, crossover measurement + c5 gate).

---

### 2026-05-22 — Bucket-2 spike: GPU batched attention aborted (16.4× slower)

**Source:** `m1-12-permuted-attention-batched-decode.md`, `2026-05-22-perf-bucket-roadmap-r3-default.md` §2

**Crossover spike (Qwen3-4B, B=8, n_kv ∈ {256, 1024, 2048}):**

| Shape (B=8) | in_tee_rayon_b8 | gpu_batched_b8 (burn f16) | GPU vs in-TEE |
|---|---:|---:|---:|
| n_kv = 256 | **1.06 ms** | 48.5 ms | 45.9× slower |
| n_kv = 1024 | **7.13 ms** | 186.2 ms | 26.1× slower |
| n_kv = 2048 | **22.3 ms** | 364.8 ms | **16.4× slower** |

**Mask delta:** no_mask vs with_mask <2 % (burn-cubecl-fusion already folds `+ mask` add). Q11 answered: fusion firing (≤2 dispatches/call vs 5 without).

**Decomposition:** ~180 ms of 365 ms per call was host-to-device f32→f16 conversion + memcpy of K/V tensors. On iGPU UMA (DDR5-shared bus), upload + kernel-read contend for same bandwidth.

**Acceptance gate failed:** required ≥30 % wall reduction on top of R3 baseline (112.99 s post-R3 → ≤79 s); measured 16.4× regression. **Abort R1.4 engineering. Bucket-2 deferred indefinitely on iGPU.**

**Why M1.11 crossover hypothesis was wrong:** GPU compute scales linearly with B on RADV gfx1151 — parallel to in-TEE, not launch-dominated. The 22 ms baseline at B=1 was already compute-bound. Slowdown ratio decreasing from 45.9× → 16.4× (smaller compute → larger) confirms compute-bound, not dispatch-bound.

**Ruled out for iGPU:** burn-chain at decode m=1, cubek-attention Unit at decode m=1, custom WGSL FlashAttention-D (score-tensor HBM saving marginal). **dGPU substrate (M5.9) re-measurement gated on hardware availability.**

---

### 2026-05-22 — R3 LM-head GPU offload landed (2.70× decode)

**Source:** `2026-05-22-perf-bucket-roadmap-r3-default.md`

**Config:** Qwen3-4B, B=8, n=2048, K=64.

**Measurement:**

| Phase | Variant | Wall (s) | Aggregate tok/s | per-tok-per-seq (ms) |
|---|---|---:|---:|---:|
| Prefill | baseline | 192.13 | 85.3 | — |
| Prefill | R3 | 179.59 | 91.4 | — |
| **Decode** | **baseline** | **304.90** | **1.68** | **595** |
| **Decode** | **R3** | **112.99** | **4.53** | **221** |

**Δ decode wall under R3: −62.9 % (2.70× speedup).**
**Δ `tee:compute_logits`: −97.6 % (195 831 → 4 726 ms; residual is profile wrapper).**
**Token parity:** 64/64 on real Qwen3-4B weights.

**Scaling with batch:** 1.82× at B=1 K=32, 2.0× at B=1 K=64, **2.70× at B=8 K=64**. GPU LM-head matmul amortises a single dispatch across `1+k=16` shield-aligned rows; in-TEE compute scales linearly.

**Commits landed:** R1 `provision_decoder_into` helper, Q#1 `tee:compute_logits` instrumentation, R3 LM-head GPU masked offload.

**Security gate:** c6 AloePri attack-suite condition (LM-head shape 37× wider than QKV; recovery surface known to scale). Gate run pending before default-flip.

**Production-representativeness caveat:** all B=8 numbers are on **Strix Halo iGPU UMA**. Production target is **SEV-SNP CVM + VFIO dGPU** (PCIe Gen4 ~30 GB/s). The masked-operand round-trip cost differs — UMA fast-path (mapped buffer) vs PCIe DMA (unavoidable transit). R3's win is **iGPU UMA only**; on dGPU LM-head offload adds ~80 % more bytes-on-wire (~39 GiB over 500-token decode). Material but acceptable on UMA; production dGPU leverage is different.

---

### 2026-05-22 — Q3-4B B=8 n=2040 mask sweep (batching win confirmed)

**Source:** `2026-05-22-q3-4b-b8-mask-sweep.md`

**Headline (vs B=2 baseline):**

| Config | B | TTFT | Per-seq wall | Δ per-seq vs B=2 |
|---|---:|---:|---:|---:|
| Auto n=2048 | 2 | 49.1 s | 26.5 s | (base) |
| **Auto n=2040** | **8** | **127.7 s** | **16.8 s** | **−37 %** |
| HD₃ n=2048 | 2 | 65.1 s | 34.5 s | (base) |
| **HD₃ n=2040** | **8** | **126.8 s** | **16.7 s** | **−52 %** |

TTFT grows ~2.5× absolute (vs 4× more sequences) — **sub-linear scaling via GPU dispatch amortisation**.

**Prefill per-op breakdown (TTFT 127.7 s, Auto):**

| Bucket | Time | Share |
|---|---:|---:|
| `engine:matmul_many` (QKV-fused + gate/up-fused) | 35.3 s | 29.5 % |
| `engine:matmul` (O + down) | 24.5 s | 20.5 % |
| **GPU subtotal** | **59.8 s** | **50.0 %** |
| `tee:attn_inplace_many` | 21.1 s | **17.6 %** ← R1.4 trigger |
| `gelo:mask_unapply` | 19.4 s | 16.2 % |
| `gelo:mask_apply` | 11.4 s | 9.5 % |
| `gelo:shield_stack` | 2.5 s | 2.1 % |

GPU 50 % / CPU mask 25.7 % / in-TEE attention 17.6 % / other 6.7 %. **In-TEE attention crossed the ~10 % threshold**, triggering R1.4 acceptance gate (which subsequently failed — see bucket-2 abort above).

**Decode per-op breakdown (TPOT 1.61 s × 4 steps, Auto):**

| Bucket | Time | Share |
|---|---:|---:|
| `tee:attn_cached_inplace_many` | 3.54 s | **55.0 %** |
| `engine:matmul` | 1.14 s | 17.7 % |
| `engine:matmul_many` | 0.92 s | 14.2 % |
| `gelo:mask_unapply` | 0.35 s | 5.5 % |

At decode, in-TEE attention explodes to **55 % of decode wall** at B=8 n_kv≈2044.

---

### 2026-05-26 — Mask instrumentation + Auto threshold tune

**Source:** `2026-05-26-mask-instrumentation-and-auto-tune.md`

**Five patches landed; prefill wall delta:** +4.3 % vs 4-day-old baseline (179.6 → 187.3 s, within day-to-day variance ~7 %).

**Patch 1 — Per-family profile categories:** replaced flat `gelo:mask_apply/unapply` with `:hd3 / :dct4 / :haar` family-specific buckets. **Load-bearing for all roadmap measurements** — without it Auto-resolution is invisible.

**Patches 3–4 — Bandwidth cleanup:** `scale_inplace` fused into final D₃ diagonal; batched scratch reuse + slice mask kernels eliminate per-block `to_owned + assign`. Theoretical ~5 s / ~3 % prefill win at long-n, below 7 % variance floor.

**Auto threshold tune (`HD3_AUTO_MAX_PAD_RATIO_NUM = 7/5 = 1.4` → `8/5 = 1.6`):**

Sweep confirmed HD₃ wins measurably up to pad ratio 1.59. Post-tune verification (3 cells):

| B | n | pad ratio | Auto family | prefill wall (s) | decode wall (s) |
|---:|---:|---:|---|---:|---:|
| 1 | 2561 | 1.59 | HD₃ | 31.92 | 21.43 |
| 8 | 320 | 1.56 | HD₃ | 24.22 | 26.82 |
| 8 | 2048 | 1.99 | DCT-IV | 174.92 | 55.08 |

Auto resolved correctly in all three. No regression at DCT-IV shape.

---

### 2026-05-26 — DCT-IV tile-fused cascade landed (−22.7 % prefill wall)

**Source:** `2026-05-26-r4-greenlight-bf16-aborted.md` (commit `cd1a008`)

**Measured at production shape (B=8 n=2048):**

| | Wall (s) | Aggregate tok/s |
|---|---:|---:|
| Pre-cascade | 225.55 | 91.1 |
| **Post-cascade** | **174.92** | **121.3** |

**Δ prefill = −50.6 s (−22.4 %).** Climbing aggregate from 93.7 to 121.3 tok/s — **single largest lever this session**.

**Cascade design:** column-locality refactor that tiles the DCT-IV inverse transform to keep intermediate results in L2 cache rather than spilling to main memory. Win compounds with the earlier HD₃ threshold tuning — DCT-IV dominates at production shape (pad 1.99), and cascaded tile-fusion delivers the headline prefill win.

---

### 2026-05-26 — bf16 cascade microbench disconfirmed (aborted for iGPU)

**Source:** `2026-05-26-r4-greenlight-bf16-aborted.md` (commit `909b0a3`)

**Hypothesis:** bf16 storage in mask GEMM saves bandwidth → ~20 % prefill reduction.

**bf16 cascade microbench (standalone, production prefill mask shape s=2056, d=2560):**

| Config | Mask wall | Δ vs baseline | Projected prefill wall reduction |
|---|---:|---:|---:|
| threads=1, f32 (baseline) | ~72 s | — | — |
| **threads=1, bf16** | ~32 s | −40 s | **−20.6 %** |
| **threads=16, f32** (no bf16, no protocol change) | ~7.6 s | −64 s | **−33.5 %** |
| **threads=16, bf16** (compound) | ~4.6 s | −67 s | **−35.1 %** |

**Disconfirmation:**
1. **DCT-IV bf16 standalone +8 %** → ~1.6 % wall, **below 7 % variance floor**.
2. **HD₃ bf16 regresses 2×** standalone (bulk widen-narrow in FWHT butterflies + per-call allocation overhead).
3. **Zen 5 has no native bf16 add/sub SIMD.** AVX-512_BF16 only has `VDPBF16PS` (dot-product to f32). bf16 arithmetic means widen → f32 compute → narrow.

**On dGPU this inverts:** CUDA Tensor Cores have native bf16 compute. bf16 infrastructure (phases 1/2/3a) remains useful for dGPU revival.

**Roadmap action:** bucket 3 (bf16 activation pipeline) deprioritised. Pivot to **per-shape BLIS thread dispatch** — ~33.5 % prefill win, 1.6× the bf16 lever, ~3 days engineering post-spikes vs 3–4 weeks for 3a+3b.

---

### 2026-05-26 — Q#2 RADV-async spike (R4 green-lit at 58 % overlap)

**Source:** `2026-05-26-r4-greenlight-bf16-aborted.md` (commit `ea29602`), `m1-12-r4-async-overlap.md`

**Spike measurement (d_out=2560 O projection, production shape B=8 n=2048):**
- `engine.matmul` (sync: upload + matmul + download): **19 ms**
- `Dct4Mask` apply+unapply on separate buffer: **10 ms**
- Total concurrently measured: **23 ms** = 1.25× speedup vs 29 ms serial.
- **58 % CPU/GPU overlap** on Strix Halo UMA.

**Verdict:** RADV does not serialise submissions — wgpu's async API actually overlaps. R4 viable.

**R4 implementation plan (~10 days):**

| Step | What | Effort |
|---|---|---|
| 0 | Profile prep: cross-thread aggregator; split matmul → submit/wait | 1 day |
| 1 | Engine async API: `matmul_async` + `read_result` + opaque `MatmulToken` | 2 days |
| 2 | Substrate async API: `offload_linear_async` / `offload_qkv_async` / `offload_linear_many_async`; RAII `OffloadHandle`; shield-hoist | 3 days |
| 3 | Forward.rs wiring: `decoder_block_batched` + `decoder_block` use async | 2 days |
| 4 | Parity + bench: validation gate | 1 day |
| 5 | AloePri re-run (out-of-band): timing side-channel validation | — |
| 6 | Cutover: remove env var, delete sync `offload_*`, relocate verify | 1 day |

**Honest projection:** spike validates capability, not real-pipeline win. With `per_forward_mask=true` (paper-parity default), strict data dependencies limit real overlap to **1–3 % wall** unless shield-hoist + engine bus-pipeline deliver as designed. Target is 5–8 % if shield-hoist works; below 2 % means reassess scope before cutover.

---

### 2026-05-26 — AloePri pow2-UVO attack/accuracy gate recorded (security companion)

**Source:** `docs/prototype/aloepri-pow2-uvo-findings-2026-05-26.md`, `docs/prototype/aloepri-llm.html`, `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-*`.

**Scope note:** not a GELO throughput optimization. This result belongs in the chronicle because the GELO performance roadmap keeps an AloePri attack-suite gate as a default-flip/security companion: perf wins are not deployable if the observable surfaces regress.

**Current best 4B obfuscation cell:**

```text
untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-pow2e1-bf16-native.gguf
```

Config: Qwen3-4B Q8_0 untied base, h=128, β=8, α_e=1.0, α_h=0.2, Alg2 matrix-Γ/Hadamard, bf16-native GGUF, `--alg2-u-vo-mode pow2-monomial --alg2-u-vo-pow2-exp 1`.

**Accuracy gate:**

| Cell | Quality probe | HumanEval n=20 | Reading |
|---|---:|---:|---|
| Dense/default UVO | pass | 3/20 = 15 % | bf16 dense UVO hurts utility |
| Non-UVO canonical | pass | 6/20 = 30 % | reference utility for current obfuscation family |
| **pow2-monomial UVO** | **pass** | **6/20 = 30 %** | recovers dense-UVO utility loss |
| h256 pow2 UVO | fail | skipped | larger h is not a current accuracy fix |

**Attack gate highlights, pow2-UVO h128 β8:**

| Surface / attack | Result | Gate reading |
|---|---:|---|
| Static VMA | 8.98 % top-1 / 23.44 % top-10 | pass on top-1; top-10 nontrivial |
| Static IA | 0.05 % top-1 / 0.20 % top-10 | pass |
| IMA EmbedRow transformer | 0.00 % top-1 / 0.00 % top-10 | pass |
| IMA EmbedRow ridge | 55.47 % top-1 / 60.16 % top-10 | fail / out-of-paper side-probe |
| Token TFMA | 0.78 % top-1 / 4.69 % top-10 | pass |
| Token SDA | BLEU-4 1.28e-5 | pass |
| ISA row-split `kq` L0 | plain 48.63 % → obf 47.20 % | fail / invariant QK surface |
| ISA row-split `kqv_out` L0 | plain 97.46 % → obf 82.45 % | partial; L0 output-surface defense preserved |
| ISA row-split `kqv_out` L17 | plain 16.68 % → obf 16.68 % | no later-layer gain |

**Engineering result:** `evals/aloepri-attacks/m2_7/diagnose_isa/gpu_sweep.py` now has transparent long-run logging via `--progress-jsonl PATH` and flushed progress lines. The completed kqv_out run wrote:

- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-attn-and-output-512-20260526/logs/gpu_sweep_kqv_out.progress.jsonl`
- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-attn-and-output-512-20260526/logs/gpu_sweep_kqv_out.summary.json`

**Roadmap implication:** pow2-monomial UVO is the current utility-preserving UVO form for bf16 deployment. It does **not** solve the strongest row-split Q/K attack; future defenses need Q/K-side changes or TEE/path-1 coverage for raw `kq`.

---

## 5. R-series perf-bucket outcomes

### R1 — Weight Arc drop & `provision_decoder_into` helper

- **Status:** ✅ Landed (commit `4686b8f`).
- **Outcome:** No perf change (infrastructure); enables R3 host-memory recovery (~7 GB at Qwen3-4B).

### R3 — LM-head GPU offload (masked)

- **Measurement (B=8 K=64, Qwen3-4B):** decode wall 304.9 → 113.0 s = **2.70× speedup**.
- **Bucket:** `tee:compute_logits` 195.8 → 4.7 s (−97.6 %).
- **Token parity:** 64/64 on real Qwen3-4B weights.
- **Scaling:** 1.82× at B=1 K=32 → 2.70× at B=8 K=64.
- **Security gate:** c6 AloePri attack-suite pending (LM-head shape 37× wider than QKV).
- **Status:** ✅ Engineering complete; attack-validation in-flight.

### R4 — Async pipelining

- **Plan estimate:** 25–30 % prefill wall (iGPU UMA best-case).
- **Q#2 spike validated:** 58 % CPU/GPU overlap on Strix Halo UMA.
- **Projected wall:** ~12 % prefill reduction (accounting for DDR5 bus contention).
- **Status:** ✅ Green-lit for implementation (~10 days).

### Bucket 2 — Batched GPU attention (aborted 2026-05-22)

- **Measurement:** `gpu_batched_b8` 364.8 ms vs `in_tee_rayon_b8` 22.3 ms = **16.4× slower** at n_kv=2048.
- **Acceptance gate:** ≥1.5× faster GPU; result 0.06×.
- **Root cause:** upload pipeline + GQA replication exceed compute savings on iGPU UMA.
- **Status:** ❌ Aborted on iGPU; revival scoped for dGPU (M5.9).

### Bucket 3a / 3b — bf16 pipeline (deferred / aborted)

- **3a narrow variant (bf16 mask GEMM):** measured 10 % TTFT at M1.10; subsumed into HD₃ + Q4 compound stack.
- **3b broader rework (bf16-native activation):** ~2–3 weeks; prerequisite for dGPU bucket-2 revival.
- **2026-05-26 cascade microbench:** disconfirmed for iGPU on both HD₃ and DCT-IV (Zen 5 lacks native bf16 SIMD arithmetic).

### Q-series spikes

- **Q#1 — `tee:compute_logits` instrumentation** ✅ (enabled R3 win).
- **Q#2 — RADV-async spike** ✅ (validated R4).
- **Variance sweep at production shape** — pending (~80 min; gates every single-cell EV claim ≥7 %).

---

## 6. dGPU attention revival design

**Source:** `2026-05-22-dgpu-attention-revival.md`

The iGPU bucket-2 abort revealed that **the binding bottleneck is upload bandwidth, not GPU kernel compute**. On dGPU the ratio changes dramatically:

| | iGPU UMA (Strix Halo) | dGPU SEV-SNP + VFIO |
|---|---:|---:|
| Per-call upload bandwidth | DDR5 memcpy ~10 GB/s | PCIe 4.0 DMA ~30 GB/s |
| Kernel-side K/V read | DDR5 ~40 GB/s (shared) | **HBM ~3 TB/s** |
| Ratio kernel/upload | 4× | **100×** |

### Item 1 — Persistent K/V on GPU

**1A — Block-level fresh π (refresh every N decode steps):**
K cache persists under permutation π_block for N consecutive steps.

| N | σ needed | Status |
|---|---:|---|
| 8 | 0.028 | Likely within model tolerance |
| 16 | 0.040 | Probably fine |
| 32 | 0.057 | Needs accuracy spike |
| 64 | 0.080 | At edge — needs spike + AloePri gate |

Bandwidth win: N× upload reduction. At N=16 → ~50 % of bucket-2 gap closed; at N=64 → ~95 %.

**1B — TwinShield-Xue additive softmax-blinding** (arXiv 2507.03278). Open Q: does Xue's threat model match GELO's? What rank of R works? Full-rank R has correction cost equal to original attention (no perf win).

**Security spike:** ~1–2 weeks. Both 1A and 1B should run in parallel.

### Item 2 — GQA-aware custom WGSL kernel

Current `engine.fused_attention_batched` takes already-replicated K/V at shape `(B·num_q_heads, n_kv, d_head)`. At Qwen3-4B group=4 this is **4× redundancy**.

**Win:** kernel takes un-replicated K/V `(B, num_kv_heads, n_kv, d_head)` and broadcasts kv-head rows inside the shader. **4× reduction in K/V data motion** (256 MB → 64 MB full upload, 32 KB → 8 KB on delta). **On dGPU: upload is 100× slower than HBM read.** Eliminating 4× of upload payload is 4× saving on the binding bottleneck.

### Item 3 — Single-Pass FlashAttention (FLASH-D)

Folds matmul → scale → mask → softmax → matmul into one GPU dispatch with online softmax.

**iGPU view:** scores tensor at decode-m=1 is ~1 MB; memory saving marginal. burn-cubecl-fusion already folds `+ mask`. Marginal win.

**dGPU view at prefill (n_q=2048):** scores tensor is `(B·H, 2048, 2048)` f16 = ~4 GB. **FLASH-D avoids materialising this entirely — genuinely large win on dGPU.**

### Per-call data motion (Qwen3-4B B=8 n_kv=2048, layered)

| Configuration | K/V upload per call | Kernel reads K/V | Notes |
|---|---:|---:|---|
| Today's bucket-2 (aborted iGPU) | 256 MB | 128 MB (after f32→f16) | iGPU 16.4× slower than in-TEE |
| + Item 1 (persistent K/V) | 32 KB delta + 256 MB per block | same | Block-amortised |
| + Item 2 (GQA-aware kernel) | 8 KB delta + 64 MB per block | 32 MB | 4× reduction |
| + Item 3 (FLASH-D fused) | same | 32 MB | scores stay in shared memory |
| **dGPU end-state at N=64** | **1 MB per step (amortised)** | **32 MB at 3 TB/s = 0.01 ms** | **Compute-bound on HBM** |

**dGPU end-state per-call cost ~0.01 ms** vs current 365 ms iGPU = ~36 500× faster per call; ~2 200× faster than iGPU in-TEE path.

---

## 7. Methodology + variance discipline

### Variance floor

- **Single-cell variance at production shape (B=8 n=2048 long-context):** ~7 % on Strix Halo UMA.
- **Three clean runs of `tee:attn_cached` on identical fixture:** 77.5 / 69.9 / 89.7 s (±15 % band).
- **Gate:** any single-cell EV claim ≥7 % treated as ground truth only after variance-sweep validation (~80 min on production shape).

### Bench infrastructure

- **Harness:** `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs` (real Qwen3-4B weights, B=8, n=2048, K=64 defaults).
- **Variants:** `GELO_BENCH_VARIANT`, `GELO_BENCH_B`, `GELO_BENCH_N`, `GELO_BENCH_MAX_TOKENS`, `GELO_BLIS_THREADS`, `GELO_BENCH_MASK_KIND` env knobs.
- **Scope:** per-layer prefill/decode breakdown + headroom analysis.

### Validation checklist (for any optimization PR)

1. **Functional parity:** `cargo test -p gelo-embedder --release` green. Specifically `masked_and_plaintext_executors_agree`, `qkv_shares_one_mask`, `mock_report_is_rejected_under_mismatched_dp_config`.
2. **Protocol fidelity:** `cargo test -p gelo-rag --release --test beir_accuracy -- --ignored` at 1k-doc subset: `top1_vs_plain ≥ 0.95`, `rec@10_vs_plain ≥ 0.95`.
3. **U-Verify probes:** Freivalds-style integrity checks must pass at k=8 (`(2L)⁻⁸ ≈ 2.4·10⁻⁷` undetected-tamper rate).
4. **Attestation rebinding:** if protocol surface changes, `config_digest` in attestation `REPORT_DATA` must update so relying parties can pin expected scheme identity.

### Anti-patterns (evergreen from archived `inference-optimization.md`)

1. **Don't replace cubecl hand-rolled WGSL kernels.** Bottleneck is around the GEMM (dispatch, sync, autotune, buffer allocation), not in the kernel.
2. **Don't upload the mask to the GPU naively.** Mask `A` must come from TEE-side CSPRNG for privacy argument to hold.
3. **Don't push dispatch-async without understanding the unmask dependency.** Round-trip `Aᵀ·(U·W)` requires GPU result before next layer's input.
4. **Don't switch backends to CUDA-only or Metal-only.** GELO targets OEM-agnostic operation (consumer GPU passthrough via VFIO).
5. **Don't cache the mask across forward passes.** Security relies on fresh-per-batch sampling.
6. **Don't add MoE or sparse-expert routing.** Models in scope are dense.

### Threat-model anchors

- **M1.3 design lock:** Global attention stays in-TEE (softmax non-linearity blocks masked offload under F1+).
- **F1+ residual:** threshold-count attack still recovers π (count probs < 1e-12 per row). Documented as F1++.
- **GELO §4.3 attacks** (anchor-based recovery, JADE, JD, Gram-error) — not fully covered by AloePri harness; phase 2 of HD₃ gate will add.
- **dGPU K/V persistence:** requires σ-vs-N analysis (item 1A) or TwinShield-Xue validation (item 1B) before adoption.

---

## 8. Open levers + next priorities

### Critical (gates load-bearing claims)

1. **R4 async pipelining implementation** (~10 days, green-lit 2026-05-26). Projected ~12 % prefill wall reduction. Q#2 validation complete.
2. **AloePri c6 attack-suite gate run** for R3 LM-head default-flip. Acceptance: metrics within sample-noise of c2_default baseline. ~3 days.
3. **Variance sweep at production shape** (~80 min real-weight bench). Establishes confidence floor for single-cell EVs.
4. **Per-shape BLIS thread dispatch** (~3 days post-spikes; ~33.5 % prefill win at long-n). Detection signal: large `m ≥ 1024` AND no outer rayon parallelism. Acceptance gates: ≥30 % single-stream long-n prefill, ≤−5 % regression on embedder / reranker / decode-m=1 / batched-prefill.

### Medium term (research/security-gated)

5. **HD₃ attack-suite re-validation** (B.3 gate, ~1–2 weeks). Phase 1: AloePri 6-attack matrix vs c3_hd3. Phase 2: GELO §4.3 attacks (anchor-ICA, JADE, JD, Gram error). If passed: flip `MaskKind::Hd3` to default.
6. **dGPU substrate bring-up** (M5.9, hardware-gated). Re-measure bucket-2 with >100× bandwidth ratio; persistent K/V; GQA-aware WGSL kernel + FlashAttention-D fused dispatch.
7. **Q4 weight quantization** (`docs/plans/q4-gpu-weights.md`). Compound stack with HD₃ mask reaches paper-target ~5–6 s TTFT.
8. **DCT-IV cascade depth tuning** — audit k=3 vs k=4/5 multi-anchor attack resistance.

### Architectural future

9. **Amulet softmax-equivariance** (~1–2 week security spike). Candidate to replace the current F1+ TEE-softmax bottleneck.
10. **Slalom-additive hybrid for linear projections** (R&D milestone, multi-week). Potential 40–60 % wall reduction if security analysis passes; AloePri-class validation required.
11. **SCX-style stateless KV-cache encoding** for decode-phase (not yet explored in implementation).
12. **Fused permuted-attention kernels** — burn-cubecl upstream gap on `causal: bool` parameter.

### iGPU ceiling

- **Current:** ~5–7 tok/s per-sequence decode at B=1. In-TEE attention structurally untouchable on iGPU (bucket-2 disconfirmed GPU offload). Fifty percent of B=8 decode wall is in-TEE attention; killing the remaining 45 % takes us to ~7 tok/s.
- **dGPU ceiling (roadmap speculative):** 40+ tok/s requires dGPU substrate + persistent K/V + Q4 weight quantization + iGPU-track substrate prerequisites (R4 async, per-shape thread dispatch).

---

## 9. Cross-references + artefacts

### Companion docs (same workstream, narrower scope)

- [`gelo-mask-perf-log.md`](../../archive/dev/logs/gelo-mask-perf-log.md) (archived) — earlier distillation focused on mask-family + R-series outcomes (subset of §3–5 here).

### Static reference (protocol substrate)

- [`../prototype/gelo.md`](../prototype/gelo.md) — protocol design (§3 protocol, §5 design choices, §7 tradeoffs).
- [`../prototype/gelo-llm.md`](../prototype/gelo-llm.md) — LLM-serving extension (§3 prefill fused permuted attention, §4 decode KV-cache, §6 engineering plan).
- [`../prototype/hd3-non-pow2-fix.md`](../prototype/hd3-non-pow2-fix.md) — mask-family decision doc (Option A auto-dispatch + Option K DCT-IV cascade; §6 multi-anchor attack ruling out block-diagonal).
- [`../../plans/gelo-llm-perf-roadmap.md`](../../plans/gelo-llm-perf-roadmap.md) — current roadmap state.
- [`../../plans/m1-12-tee-gpu-throughput.md`](../../plans/m1-12-tee-gpu-throughput.md) — UMA-residency overhaul (R1/R3/R4 survivors).
- [`../../plans/m1-12-blis-thread-dispatch.md`](../../plans/m1-12-blis-thread-dispatch.md) — per-shape thread dispatch (9.5× at mask GEMM).
- [`../../plans/m1-12-bf16-activation-pipeline.md`](../../plans/m1-12-bf16-activation-pipeline.md) — 3a + 3b plan (deferred for iGPU).
- [`../../plans/m1-12-r4-async-overlap.md`](../../plans/m1-12-r4-async-overlap.md) — green-lit implementation plan.

### Archived (historical baselines)

- [`../../archive/prototype/gelo-complexity-analysis.md`](../../archive/prototype/gelo-complexity-analysis.md) — May 19 Qwen3-1.7B Haar baseline measured bottleneck breakdown (methodology evergreen; numbers stale).
- [`../../archive/prototype/inference-optimization.md`](../../archive/prototype/inference-optimization.md) — May 13–18 speculative tier'd plan (anti-patterns + validation checklist evergreen).
- [`../../plans/m1-10-phase4-findings.md`](../../plans/m1-10-phase4-findings.md) (`status: stale`) — M1.10 Phase 4 findings (Phase 2 deprecated post-F1+).
- [`../../plans/m1-12-permuted-attention-batched-decode.md`](../../plans/m1-12-permuted-attention-batched-decode.md) (`status: stale`) — R1.4 Phase A aborted; bucket 2 deferred indefinitely on iGPU.

### Code anchors

- `crates/gelo-protocol/src/mask.rs` — Haar baseline, `MaskFamily` enum, `MaskKind`, `mask_backend_description()`.
- `crates/gelo-protocol/src/hd3.rs` — HD₃ Hadamard cascade primitive.
- `crates/gelo-protocol/src/dct4.rs` — DCT-IV cascade (tile-fused as of 2026-05-26).
- `crates/gelo-protocol/src/sim.rs` — `InProcessTrustedExecutor`, `with_snapshot_capture`, `with_per_offload_mask`.
- `crates/gelo-protocol/src/attention.rs` — `permuted_attention_cached`, F1+ TEE-softmax.
- `crates/gelo-protocol/tests/bf16_mask_parity.rs` — parity regression anchor.

### Bench-results raw artefacts (2026-05-26)

All at `bench-results/`:

| File | Bench | Headline |
|---|---|---|
| `bf16-cascade-microbench-2026-05-26_13-47-56.log` | bf16 vs f32 cascade at n=2056, d=2560 | bf16 1.078× faster (10.13 ms vs 10.92 ms) — but cascade-level only +8 % DCT-IV / regression HD₃; below variance floor at wall scale |
| `dct4-cascade-microbench-2026-05-26_11-35-01.{log,tsv}` | DCT-IV tile-fused cascade validation | n=2056 projects ~20.3 s TTFT at Qwen3-4B (vs 32 s HD₃-padded, 25.9 s Haar) |
| `m1-12-auto-tune-verify-2026-05-26_08-42-00.{log,tsv}` | Auto-dispatch verification on Qwen3-4B | Threshold 8/5 = 1.6 picks correctly across cells |
| `m1-12-hd3-perf-sweep-2026-05-26_07-04-58.{log,tsv}` | 14-cell parametric sweep of HD₃ roadmap | Auto-dispatch boundary points validated |
| `measurement-gaps-2026-05-26_10-35-15.{log,tsv}` + `…cell4-rerun2…` | Diagnostic variance sweep, 4 cells × 25 iters | Confidence interval establishment; SWIOTLB spikes characterised |
| `q2-radv-async-spike-2026-05-26_14-14-23.log` | Q#2 RADV-async overlap measurement | 58 % CPU/GPU overlap; T_gpu=19 ms, T_cpu=10 ms (ratio 1.9×); R4 green-lit |
| `uma-spike-2026-05-26_12-34-26.log[.summary]` | UMA anomaly reproduction at B=16 n=2048 | Memory-pressure characterisation (RSS: 0.01 → 7.67 → 2.39 GiB) |

### Source handoffs distilled into §4

| Date | Handoff |
|---|---|
| 2026-05-18 | `2026-05-18-m1-10-perf.md` (archived) |
| 2026-05-19 | `2026-05-19-bf16-mask-deferred.md` (archived) |
| 2026-05-19 | `2026-05-19-hd3-followups.md` (archived) |
| 2026-05-21 | `2026-05-21-attn-offload-spike.md` (archived) |
| 2026-05-21 | `2026-05-21-gelo-perf-shield-attn-batched.md` (archived) |
| 2026-05-22 | `2026-05-22-dgpu-attention-revival.md` (active) |
| 2026-05-22 | `2026-05-22-perf-bucket-roadmap-r3-default.md` (active) |
| 2026-05-22 | `2026-05-22-q3-4b-b8-mask-sweep.md` (active) |
| 2026-05-26 | `2026-05-26-mask-instrumentation-and-auto-tune.md` (active) |
| 2026-05-26 | `2026-05-26-r4-greenlight-bf16-aborted.md` (active) |
