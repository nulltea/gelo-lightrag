---
type: dev-log
status: stale
created: 2026-05-18
updated: 2026-05-26
tags: [gelo, perf, mask, bf16, hd3, dct4, bench, auto-dispatch]
companion: [2026-05-18-m1-10-perf, 2026-05-19-bf16-mask-deferred, 2026-05-19-hd3-followups, 2026-05-21-attn-offload-spike, 2026-05-21-gelo-perf-shield-attn-batched, 2026-05-22-dgpu-attention-revival, 2026-05-22-perf-bucket-roadmap-r3-default, 2026-05-22-q3-4b-b8-mask-sweep, 2026-05-26-mask-instrumentation-and-auto-tune, 2026-05-26-r4-greenlight-bf16-aborted]
superseded_by: gelo-llm-perf-chronicle
archive_reason: "Earlier (2026-05-26) distillation of mask-family hierarchy + R-series outcomes + signature numbers table. Fully absorbed into the comprehensive perf chronicle, which adds protocol primitives cost model, dGPU revival design, methodology discipline, and additional dated entries."
---

# GELO mask + performance — distilled bench log

> Knowledge layer aggregating bench results, design decisions, and analysis
> conclusions from the GELO perf workstream handoffs (2026-05-18 through
> 2026-05-26). The handoffs themselves remain canonical for chronological
> context and full reasoning; this doc is what you open to look up a number,
> a design decision, or understand why a path was chosen or rejected.

## Headline performance numbers

### Long-context generation (Qwen3-1.7B, n=2048, 16 greedy tokens)

| Path | TTFT | TPOT | Wall | vs plain | Status |
|---|---:|---:|---:|---:|---|
| `gpu_plain` | 6.4 s | 271 ms | 10.8 s | (baseline) | — |
| `gpu_gelo` (Haar mask) | 72.9 s | 371 ms | 78.8 s | +631 % | Bottleneck: mask GEMM round-trip |
| `gpu_gelo_permuted` (fused permuted attention) | 87.5 s | 1063 ms | 104.5 s | +870 % | Adds K/V perm cost; sec-gated F1+ |

The +631 % overhead is **not** attention — it's the GELO mask round-trip on
linear projections (QKV, O, gate∥up, FfnDown) repeated 28 layers × 6 batches
per layer. The +870 % includes permutation-cache overhead and is the
research-grade path that trades perf for security under F1+. (2026-05-18,
M1.10 Phase 4 findings.)

### Batched prefill (Qwen3-4B, B=8, n=2048)

| Config | TTFT | Aggregate tok/s | Notes |
|---|---:|---:|---|
| B=1 baseline (DCT-IV) | 179.6 s | 91.4 | M1.12 R3 baseline |
| **B=8 Auto (HD₃ at pow2)** | **187.3 s** | **121.3** | Production shape; real weights, post-instrument |
| B=8 HD₃ shield-to-pow2 (n=2040 aligned) | 126.8 s | — | −52 % per-seq wall vs B=2 |

Prefill scales sub-linearly with batch: GPU dispatch amortises across B
sequences. Per-sequence wall drops 37–52 % going B=2 → B=8 at long context.
(2026-05-26, M1.12 instrumentation.)

### Single-stream extraction (Qwen3-4B + Qwen3-Embedding-0.6B, 7-chunk doc)

| Variant | Generate wall | vs prev | Bottleneck after |
|---|---:|---:|---|
| v6 (DCT-IV, k=8) | 359.5 s | — | `tee:attn_cached` 32 % |
| v7 (HD₃, shape-adaptive k=8/k=15) | 361.2 s | flat | Shield cost +89 % |
| Post-SIMD-shield (Box-Muller) | 341.6 s | −5.4 % | `tee:attn_cached` 31.7 % |
| **Post-polar-Xoshiro (full shield)** | **~330 s** | **−8.8 %** | `tee:attn_cached` 39.3 % |

Shield-stack optimisations total **3.08× cumulative** from v7: SIMD
Box-Muller (−37 %), then polar rejection + Xoshiro256++ (−48.5 %). The attn
bucket became the clear next target at 39 % of wall.
(2026-05-21 two shield-optimisation sessions.)

---

## Mask family hierarchy

### Haar (baseline, dense)

- **Cost:** O((n+k)² · d) FLOPs per apply/unapply.
- **Sampling:** Per-forward QR via Householder reflection — O((n+k)³) scalar ops.
- **Overhead at M1.10 n=2048:** ~50–75 s per 28-layer forward.
- **Security:** κ=1 orthogonality exact at f32; pair with shield rows σ=4.0.
- **Status:** Baseline; the Auto-dispatcher falls back to Haar at non-pow2 when neither HD₃ nor DCT-IV is preferred.

### HD₃ (Hadamard cascade, QuIP#/QuaRot primitive)

- **Cost:** Three FWHTs `D₃·H·D₂·H·D₁·H` — **O(s·d·log s)** vs O(s²·d) dense.
- **Material:** 3·s ±1 bits per mask, fresh per forward.
- **SIMD:** AVX-512F 16 f32/inst, AVX-2 8 f32/inst fallback, rayon-parallel above 65 K elements.
- **At pow2 alignment:** −28 % TTFT at n=2040 (s=2048, exactly pow2).
- **Non-pow2 cost:** +51 % TTFT at n=2048 (s=2056 → pad to 4096 → 2× matmul & FWHT cost).
- **Orthogonality:** Exact at f32 (same 10⁻⁶ round-trip error as dense Haar).
- **Per-call timing at production B=8:**
  - apply: 192 µs/call (Haar 358 µs) = −46 %
  - unapply: 191 µs/call (Haar 290 µs) = −34 %
- **Security status:** Research-grade, opt-in. Awaiting AloePri attack-suite re-validation (c3_hd3 condition) before default-flip. QuIP# incoherence proof carries over (≥ 2^(3s) orbit size); empirical BSS-hardness under GELO threat model still needs the formal gate (phase 1: AloePri 6-attack matrix; phase 2: GELO §4.3 anchor-ICA + JADE + JD + Gram-error).

### DCT-IV (tile-fused, production workhorse)

- **Cost:** O(s·d·log s) via FFT, orthogonal at any N (no padding needed).
- **Material advantage at production shape:** tile-fused cascade landed 2026-05-26 (`cd1a008`) wins **−22.7 % prefill wall** (174.9 s → 135.1 s at B=8 n=2048).
- **Why it wins at production shape:** Column-locality fuse captures L2-cache tile reuse; HD₃ FWHT is more memory-bandwidth dominated.
- **Auto-dispatch threshold:** At pad ratio ≤ 1.6 (8/5), Auto picks HD₃; above 1.6, DCT-IV.
- **Security:** Paper-equivalent orthogonality; no research-grade questions.
- **Status:** Current production default (2026-05-26 evaluation).

### Auto-dispatch strategy

Rules (`HD3_AUTO_MAX_PAD_RATIO_NUM = 8`, threshold 1.6, tuned 2026-05-26):

- At pow2 operand shape: use Haar (trivial, no padding).
- At non-pow2 with pad ratio `(n+k).next_pow2() / (n+k) ≤ 1.6`: HD₃ (FWHT slightly better than dense at modest padding).
- At higher pad ratio: DCT-IV (no padding, asymptotically better).

Sweep confirmed HD₃ wins measurably up to pad ratio 1.59; DCT-IV preferable above. Threshold empirically validated across {B=1, B=8} × {n=512, 1024, 2048, 4096}.

**Per-family profile categories (2026-05-26 instrumentation):** Distinct `gelo:mask_apply:{haar,hd3,dct4}` + `gelo:mask_unapply:*` buckets emit so Auto-resolution visibility is complete.

---

## BLIS threading and mask GEMM

### Baseline (M1.10, single-thread matrixmultiply)

- **Speed:** ~50–75 GFLOPs/sec (scalar CPU, limited by model f32 throughput).
- **Per-layer mask cost:** ~272 GFLOPs × 28 layers = 7.6 TFLOPs per 28-layer forward → ~50–75 s at n=2048.

### Multi-threaded AOCL-BLIS (M1.10 Step 1, landed 2026-05-19)

- **Speed:** ~1.25 TFLOP/s on Zen/Zen+ multi-threaded (16 cores, tuned env).
- **Speedup over scalar:** **5.04× TTFT** (73 s → 14.5 s at n=2048 prefill, M1.10 scale).
- **Implementation:** `GELO_BLIS_THREADS=16` env knob; auto-detected via `AOCL_NUM_THREADS` fallback.
- **Regression test:** `bf16_mask_parity.rs` verifies precision contract at Qwen3-1.7B shapes.
- **Parity (f32 target):** 2.65 × 10⁻³ mean relative error vs f32 reference; 1.87 × 10⁻³ vs bf16 everywhere (within paper's ≥98.8 % token-equality band).

### bf16 mask GEMM (deferred 2026-05-19, microbench-disconfirmed 2026-05-26)

- **Potential at M1.10:** 1.6–1.8× GEMM throughput via AVX-512_BF16 (VDPBF16PS microkernel).
- **Measured gain at narrow scope:** ~10 % TTFT at n=2048 (−1.3 s on 14.5 s baseline).
- **Why deferred (2026-05-19):** HD₃ subsumes the gain — replaces dense O(s²·d) GEMM with O(s·d·log s) FWHT; 25× FLOP reduction dwarfs bf16's 1.6×.
- **bf16 cascade microbench (2026-05-26, disconfirmed):**
  - DCT-IV cascade bf16: +8 % standalone → ~1.6 % wall (below 7 % variance floor).
  - HD₃ cascade bf16: regresses 2× standalone (bulk widen-narrow + per-call allocation).
  - Root cause: Zen 5 has no bf16 add/sub SIMD (only VDPBF16PS dot-product to f32); bf16 arithmetic means widen → f32 compute → narrow.
- **dGPU insight:** CUDA Tensor Cores invert this; bf16 compute is native, rewrites the math entirely.
- **Status:** Aborted for iGPU; infrastructure shipped (phase 1/2/3a) remains useful for dGPU future.

---

## Design decisions and trade-offs

### F1+ causal-mask security resolution (landed 2026-05-18)

- **Problem:** Original `permuted_attention` leaked the causal pattern π via the `-∞` entries in the masked score tensor sent to GPU for softmax (exact-zero recovery attack).
- **Solution:** Move softmax in-TEE, replace `-∞` with `-C = 30` so blocked positions softmax to ~exp(−30) ≈ 1e-14 (non-zero at f32 precision).
- **Cost:** One PCIe round-trip on the score tensor per call (~64 MB at n=2048 prefill). Softmax CPU work negligible.
- **Residual risk:** Threshold-count attack still recovers π (count probs < 1e-12 per row). Documented as F1++ (not implemented); would require adding small Gaussian noise on probs before GPU return.
- **Test anchors:** three `f1plus_*` regression tests in `permutation_attention.rs`.

### Block-diagonal mask A (deferred)

`A = diag(A₁, …, A_B)` reduces mask matmul O(n²·d) → O((n/B)·n·d), ~4–8× speedup at B=4–8. Cross-block correlations leak O(n/B) linear constraints per token; acceptable for some threat models but needs written security analysis. Scoped as `docs/dev/prototype/future-rnd.md` §5; not v1.

### HKDF-derived per-step mask material (deferred)

At decode, derive each step's A deterministically from `HKDF(SessionKey, "gelo-llm.mask", step_idx)`. Eliminates per-decode-step Haar QR sampling (~17 GFLOPs/forward → ~0.5 s at n=2048). Freshness-argument write-up required before adoption. (`docs/plans/m1-10-fused-permuted-attention.md` §10.)

### On-GPU unmask `Aᵀ` (struck 2026-05-18)

Originally proposed in archived `inference-optimization.md` Tier 3.4. Threat-model-incompatible: putting `A` on GPU lets a hostile engine compute `H = Aᵀ · masked_h` and recover plaintext H within one forward. Only valid under confidential-GPU threat model (H100 CC), which GELO does not target. Struck in commit `68cd468`.

### Fused-attention kernel (Phase 2, deprecated 2026-05-18)

Original plan: FlashAttention-style fused kernel taking `(q, k, v, scale, mask)`. Under F1+ the causal mask must not reach GPU. Four work-arounds fail:
- Infer causality from `q_pos_offset` — needs π, defeats hiding.
- Pre-noise scores in kernel — research-level, no published recipe.
- HE-mask under encrypted softmax (TwinShield Liu '25) — no HE in stack.
- Pattern-invariant mask shapes — changes model semantics.

Phase 4 bench (M1.10) confirmed attention compute is **not** the bottleneck; fused-flash would not move long-context numbers. Deprecated in `docs/plans/m1-10-fused-permuted-attention.md` §5 Phase 2 & §6.

---

## Attention paths and threat-model decisions

### In-TEE attention (causal GQA, cached decode)

- **Current state:** Stays in-TEE per M1.3 design lock (`crates/gelo-embedder/src/decoder/forward.rs:340–371`).
- **Why:** Softmax non-linearity prevents masked offload under F1+.
- **Cost at decode:** 55 % of wall at B=8 n_kv=2048 (dominant bucket post-shield optimisations).
- **Research direction:** Amulet softmax-equivariance (`gelo_research_round_2.md`) — lets masked Q·Kᵀ be offloaded if softmax rearranges to commute with orthogonal action.

### Permuted attention (fused permuted_cached, optional)

- **Path:** Two independent permutations π_q, π_kv at asymmetric n_q ≤ n_kv.
- **New protocol primitive:** `permuted_attention_cached(q, k, v, scale, q_pos_offset, mask, cfg, rng)`.
- **Trait method:** `TrustedExecutor::offload_attention_permuted_cached`; `InProcessTrustedExecutor` override.
- **Dispatcher:** `causal_gqa_attention_permuted_cached` in decoder; opt-in via `cfg.use_perm_attention`.
- **Production default:** Off (`use_perm_attention = false`).
- **Cost when engaged:** K/V perm copies dominate (200–300 ms at n=2048); Gaussian noise (rayon-parallel) secondary.
- **TPOT win (rayon-parallel Gaussian, commit cbea549):** 1693 → 1063 ms = **1.59× on TPOT** at n=2048 decode. Why not 10–30×? Rayon overhead ~6 ms fixed cost across 56 per-decode calls; per-element Ziggurat still scalar (no SIMD batching).

### dGPU attention revival (M5.9 future)

Three candidate items to restore GPU offload on discrete-GPU hardware (different bandwidth model than iGPU):

1. **Persistent K/V on GPU** (Item 1A block-fresh π or 1B additive softmax-blinding) — eliminates 8000× redundancy (full cache re-upload vs new row) at cost of σ scaling with fixed π duration.
2. **GQA-aware custom WGSL kernel** (Item 2) — un-replicated K/V shape broadcasts inside shader, 4× data reduction at Qwen3-4B group=4.
3. **Single-pass FlashAttention** (Item 3) — folds Q·Kᵀ → softmax → ·V into one dispatch, avoids materialising scores tensor in HBM.

**dGPU bandwidth math:** PCIe 4.0 ~30 GB/s upload vs HBM ~3 TB/s kernel-read (100× ratio vs iGPU UMA's 4×) — makes Item 1 a primary lever.

**iGPU outcome (2026-05-22 abort retro):** Batched-attention kernel (bucket 2) measured 16.4× slower than in-TEE at n_kv=2048 B=8. Upload pipeline + GQA replication penalty exceed compute savings. **Deferred to dGPU hardware** per `2026-05-22-dgpu-attention-revival.md`.

---

## R-series perf-bucket roadmap outcomes

### R1: Weight Arc drop & `provision_decoder_into` helper

- **Status:** ✅ Landed (commit `4686b8f`).
- **Outcome:** No perf change (infrastructure); enables R3 host-memory recovery (~7 GB at Qwen3-4B).

### R3: LM-head GPU offload (masked)

- **Measurement (B=8 K=64, Qwen3-4B):** Decode wall 304.9 → 113.0 s = **2.70× speedup**.
- **Decode bucket:** `tee:compute_logits` 195.8 → 4.7 s (−97.6 %, residual is profile wrapper).
- **Token parity:** 64/64 on real Qwen3-4B weights.
- **Scaling:** Multiplier grows with batch (1.82× at B=1 K=32 → 2.70× at B=8 K=64).
- **Security gate:** c6 AloePri attack-suite condition (LM-head shape is 37× wider than QKV; recovery surface known to scale). Gate run pending.
- **Status:** ✅ Engineering complete; attack-validation in-flight.

### R4: Async pipelining (CPU mask layer N+1 ∥ GPU matmul layer N)

- **Plan estimate:** 25–30 % prefill wall (iGPU UMA best-case).
- **Q#2 spike (2026-05-26):** RADV does support async; 58 % CPU/GPU overlap measured on Strix Halo UMA.
- **Projected wall:** ~12 % prefill reduction (accounting for shared DDR5 bus contention).
- **Status:** ✅ Green-lit for implementation (~5–8 days).

### Bucket 2: Batched GPU attention (aborted 2026-05-22)

- **Measurement:** `gpu_batched_b8` at n_kv=2048 measured 364.8 ms vs `in_tee_rayon_b8` 22.3 ms = **16.4× slower**.
- **Acceptance gate:** ≥1.5× faster GPU; result 0.06×.
- **Root cause:** Upload pipeline + GQA replication redundancy exceed compute savings on iGPU UMA.
- **Bench cells retained:** `crates/gelo-gpu-wgpu/benches/amulet_attention.rs` group `amulet_attention_r1_4/` as comparison harness for future dGPU revival.
- **Status:** ❌ Aborted on iGPU; revival scoped for dGPU hardware (M5.9 separate handoff).

### Bucket 3a / 3b: bf16 pipeline (deferred / aborted)

- **3a narrow variant (bf16 mask GEMM):** measured 10 % TTFT at M1.10; subsumed into HD₃ + Q4 compound stack. Deferred indefinitely.
- **3b broader rework (bf16-native activation pipeline):** ~2–3 weeks; eliminates f32↔f16 conversion + one DDR5 traverse per offload. Prerequisite for dGPU bucket-2 revival; blocked until M5.9 hardware.
- **2026-05-26 cascade microbench:** disconfirmed for iGPU on both HD₃ and DCT-IV (Zen 5 lacks native bf16 SIMD arithmetic).

---

## Variability and measurement discipline

### Variance floor (established 2026-05-26)

- **Single-cell variance at production shape (B=8 n=2048 long-context):** ~7 % on Strix Halo UMA.
- **Root cause:** Shared iGPU/CPU memory subsystem; coherent thermal/power-state variation; RADV driver scheduling noise.
- **Bench observation:** Three clean runs of `tee:attn_cached` landed at 77.5 / 69.9 / 89.7 s on identical fixture (characteristic ±15 % band).
- **Gate:** Any single-cell EV claim ≥7 % treated as ground truth only after variance-sweep validation (estimated ~80 min on production shape).

### Bench infrastructure

- **Harness:** `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs` (real Qwen3-4B weights, B=8, n=2048, K=64 defaults).
- **Variants:** `GELO_BENCH_VARIANT`, `GELO_BENCH_B`, `GELO_BENCH_N`, `GELO_BENCH_MAX_TOKENS` env knobs.
- **Scope:** Per-layer prefill/decode breakdown + headroom analysis.

---

## Open levers and next priorities

### Short term (documented in current plans)

1. **R4 async pipelining implementation** (~5–8 days, green-lit 2026-05-26). Projected ~12 % prefill wall reduction. Q#2 validation complete.
2. **AloePri c6 attack-suite gate run** for R3 LM-head default-flip. Acceptance: metrics within sample-noise of c2_default baseline. ~3 days.
3. **Variance sweep at production shape** (~80 min real-weight bench). Establishes confidence floor for single-cell EVs.

### Medium term (research/security-gated)

4. **HD₃ attack-suite re-validation** (B.3 gate, ~1–2 weeks). Phase 1: AloePri 6-attack matrix vs c3_hd3 (≤1 day). Phase 2: GELO §4.3 attacks (anchor-ICA, JADE, JD, Gram-error; ~1 week). If passed: flip `MaskKind::Hd3` to default.
5. **dGPU substrate bring-up** (M5.9, hardware-gated). Re-measure bucket-2 with >100× bandwidth ratio; persistent K/V; GQA-aware WGSL kernel + FlashAttention-D fused dispatch.
6. **Q4 weight quantization** (`docs/plans/q4-gpu-weights.md`). Compound stack with HD₃ mask reaches paper-target ~5–6 s TTFT.

### Architectural future

7. **Amulet softmax-equivariance** (~1–2 week security spike). Candidate to replace the current F1+ TEE-softmax bottleneck.
8. **Slalom-additive hybrid for linear projections** (R&D milestone, multi-week). Potential 40–60 % wall reduction if security analysis passes; AloePri-class validation required.

---

## Signature numbers (lookup table)

| Measurement | Value | Context | Source |
|---|---:|---|---|
| BLIS-mt-16 over matrixmultiply | 5.04× | Qwen3-1.7B n=2048 prefill | 2026-05-18 |
| bf16 mask GEMM saving (M1.10 narrow) | 1.6–1.8× | Dwarfed by HD₃ 25× | 2026-05-19 |
| HD₃ TTFT win at pow2 (n=2040 s=2048) | −28 % | −51 % regression at non-pow2 | 2026-05-19 |
| Shield SIMD Box-Muller speedup | 1.60× | d=2560 k=15 decode shape | 2026-05-21 |
| Shield polar-Xoshiro total | 3.08× | Cumulative since v7 | 2026-05-21 |
| In-TEE attention rejection factor | 16.4× | iGPU bucket-2 abort | 2026-05-22 |
| Batched prefill sub-linear scaling | 2.5× TTFT absolute | B=1 → B=8 at n=2048 | 2026-05-22 |
| Per-seq wall drop B=2 → B=8 | −37 to −52 % | Depends on mask family | 2026-05-22 |
| R3 LM-head offload decode speedup | 2.70× | B=8 K=64; compute_logits −97.6 % | 2026-05-22 |
| DCT-IV tile-fused cascade win | −22.7 % | Prefill wall at B=8 n=2048 | 2026-05-26 |
| Auto-dispatch threshold | 1.6 | Max pad ratio for HD₃ (8/5) | 2026-05-26 |
| R4 RADV-async overlap | 58 % | CPU mask ∥ GPU matmul on Strix Halo UMA | 2026-05-26 |
| R4 projected prefill wall reduction | ~12 % | Accounting for DDR5 bus contention | 2026-05-26 |
| Attn_cached dominant bucket (decode, post-shield) | 39 % | Production B=8 n_kv=2k | 2026-05-21 |

---

## Threat-model notes

- **M1.3 design lock:** Global attention stays in-TEE (softmax non-linearity blocks masked offload under F1+).
- **F1+ causal-mask leak:** Closed via TEE-resident softmax + `-C` replacement for `-∞`; residual threshold-count attack documented as F1++.
- **GELO §4.3 attacks:** Anchor-based recovery, JADE, JD, Gram-error — not fully covered by AloePri harness; phase 2 of HD₃ gate will add.
- **Shield ≠ key material:** Xoshiro256++ substitution for ChaCha20 on shield RNG is theoretically safe (shield rows post-stripped, never propagate to logits); empirical re-validation via AloePri gate pending (c2_default re-run required before production attestation).
- **dGPU K/V persistence:** Requires σ-vs-N analysis (item 1A) or published-scheme validation (item 1B TwinShield-Xue) before adoption.
