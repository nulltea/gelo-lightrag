---
type: research
status: current
created: 2026-05-19
updated: 2026-05-26
tags: [inference, llm, perf]
supersedes: [private-llm-inference-R2-2026-05-18, private-inference-R1-2026-04-21]
---

# Private LLM Inference

> Canonical research doc for private LLM inference in this project.
> Originated as "Research Round 3" (2026-05-19); promoted to canonical
> 2026-05-26 and absorbs the R1 systems survey + R2 deltas. The two
> predecessors live in `docs/archive/research/` as historical baselines.
> Future rounds should append to this file rather than spawn a new
> round-N doc.

## R2 research-spike status (G.1–G.8)

The 2026-05-18 R2 doc proposed eight research spikes. Their status as
of 2026-05-26:

| Item | Status | Resolution / pointer |
|---|---|---|
| G.1 — PLE table in TEE DRAM (Gemma 3n) | pending | Awaiting Gemma 3n model decision (see `../plans/path-1-gelo-gemma.md`). |
| G.2 — All-CPU-TEE benchmark | pending | Same gate as G.1. |
| G.3 — MoEcho replay | not started | No active owner. |
| G.4 — CryptoMoE port | blocked | Depends on G.3. |
| G.5 — TwinShield-Xue security analysis | not started | Independent; ~1–2 weeks effort. |
| G.6 — AloePri rotation cadence | **done 2026-05-18** | Verdict: static per-deployment; does not apply under openweight. |
| G.6b — Port AloePri attack suite | partially done | See `aloepri-attacks.md` + `aloepri-vs-gelo.md`. |
| G.7 — BLIS multithreading | **done 2026-05-19** | +5.04× wall on mask GEMM. See R3 §3.1 below. |
| G.8 — bf16 mask GEMM | blocked | AOCL bf16 kernel absent on Strix Halo; R4 path forward documented in `../plans/m1-12-bf16-activation-pipeline.md`. |

## R3 measured outcomes — section preserved from 2026-05-19 onward

> **Research date:** 2026-05-19. Follow-up to
> [`../archive/research/private-llm-inference-R2-2026-05-18.md`](../archive/research/private-llm-inference-R2-2026-05-18.md)
> (2026-05-18). Round 2 covered the *what to support* axis
> (MoE / hybrid-attention / PLE). This round covers the *make it fast*
> axis: candidate primitives, threat-model relaxations, and code-side
> levers that can move the 90 % CPU-mask wall measured in
> [`../archive/prototype/gelo-complexity-analysis.md`](../archive/prototype/gelo-complexity-analysis.md)
> at n=2048 prefill on Qwen3-1.7B.
>
> **Hardware scope (unchanged from round 2):** SEV-SNP CVM + commodity
> Vulkan GPU passthrough — the "replicate H100-CC-class workflows on a
> $50/mo Hetzner box" deployment. Confidential-GPU substrate is
> catalogued here for completeness but is treated as a deployment fork,
> not the primary path.
>
> **Companion docs**:
> [`../archive/prototype/gelo-complexity-analysis.md`](../archive/prototype/gelo-complexity-analysis.md)
> — the bottleneck breakdown driving this round;
> [`../dev/prototype/gelo.md`](../dev/prototype/gelo.md),
> [`../dev/prototype/gelo-llm.md`](../dev/prototype/gelo-llm.md),
> [`../archive/research/private-llm-inference-R2-2026-05-18.md`](../archive/research/private-llm-inference-R2-2026-05-18.md).

---

## Definitions

| Term | Meaning |
|---|---|
| Haar mask | The dense `(s × s)` orthogonal `A` sampled uniformly on `O(s)` via Householder QR with Mezzadri sign correction; what GELO uses today. |
| HD₃ cascade | Three independent randomised Hadamard transforms `D₃·H·D₂·H·D₁·H` (each `Dᵢ` a ±1 diagonal). QuIP# / QuaRot terminology. |
| SRHT | Subsampled Randomised Hadamard Transform — Ailon-Chazelle 2006, Tropp 2011. |
| Slalom-style additive | Per-batch one-time-pad blinding: `U = X + R`, GPU computes `(X+R)·W`, TEE subtracts precomputed `R·W`. Tramèr & Boneh 2019. |
| OTP | One-time-pad (information-theoretic privacy under fresh, uniform `R`). |
| BSS-hard | The security argument GELO uses today: distinguisher faces a single-batch Blind Source Separation problem. |
| Confidential GPU / CC | Hardware-isolated GPU TEE: NVIDIA H100/H200 CC, Blackwell TEE-I/O, AMD MI300X CC (announced). |
| PRG mask | Pseudorandom generator (HKDF / AES-CTR) expanded into the bytes of `A`; replaces true randomness with computational pseudorandomness. |
| HNM | "Hidden No More" (Thomas et al., ICML '25, arXiv:2505.18332) — the recent prompt-reconstruction attack that broke STIP and PermLLM-style 3rd-party schemes. |

---

## Measured outcomes — steps 1/2/3 (added 2026-05-19 evening)

Three near-term levers from §7 were executed and benched. Results
update the predictions in this doc:

| step | what changed | TTFT at n=2048 (gpu_gelo) | overhead vs plain | verdict |
|---|---|---:|---:|---|
| (baseline) | matrixmultiply, single-thread | 73.0 s | +1 020 % | predates step 1 |
| **step 1** | BLIS-mt with `GELO_BLIS_THREADS=16` | **14.5 s** | **+119 %** | ✅ **5.04× speedup landed** |
| **step 2** | bf16 mask GEMM | n/a (impl blocked) | n/a | ⏸ ~0.2 % round-trip rel error (parity sim) — implementation blocked on AOCL `sbgemm` upgrade |
| **step 3** | skip first 4 + last 1 layers | 20.8 s | +47.3 % | ❌ **regressed wall time** — see below |

### Step 1 (BLIS-mt) — the headline win

Sweep across `GELO_BLIS_THREADS ∈ {1, 4, 8, 16}` at n=2048 with the
`blas` feature enabled and the long-context cell only:

| threads | gpu_gelo TTFT | overhead vs plain | speedup vs matrixmultiply |
|---|---:|---:|---:|
| matrixmultiply (no blas) | 73.0 s | +1 020 % | 1.00× (baseline) |
| BLIS 1 | 39.8 s | +477 % | 1.83× |
| BLIS 4 | 19.5 s | +193 % | 3.74× |
| BLIS 8 | 16.4 s | +147 % | 4.45× |
| BLIS 16 | 14.5 s | +119 % | **5.04×** |

Diminishing returns past 8 threads (1.19× at 4→8, 1.14× at 8→16) —
sweet spot is 8-16. Even BLIS-1 is 1.83× faster than matrixmultiply
at the same shape, so the embedder regime (which keeps threads=1 by
default) is **strictly improved** by enabling `blas`. The combined
matrixmultiply→BLIS-mt-16 speedup of 5.04× drops the headline
overhead from +1 020 % to +119 %, with `gelo:mask_sample` (the
Haar QR) now the top hotspot at 22.8 % of TTFT — exactly as the
round-3 plan predicted.

**Embedder cliff confirmed**: at threads=8 the embedder bench
regresses +42.6 %, and at threads=16 by +1 540 %. Defaults must stay
at threads=1 to protect the embedder regime; long-context workloads
opt in via env var.

### Step 2 (bf16) — parity OK, but no kernel

`crates/gelo-protocol/tests/bf16_mask_parity.rs` simulates the bf16
truncation of mask + operands and measures the round-trip error vs
f32 and vs a bf16-everywhere target. Results are scale-invariant
(7-bit mantissa floor):

| shape | mean/rms error vs f32 | mean/rms error vs bf16 |
|---|---:|---:|
| `n=64, d=64, p=64` | 2.63 × 10⁻³ | 1.93 × 10⁻³ |
| `n=2056, d=2048, p=2048` (apply QKV/O/gate_up) | 2.65 × 10⁻³ | 1.87 × 10⁻³ |
| `n=2056, d=6144, p=2048` (apply FfnDown) | 2.65 × 10⁻³ | 1.87 × 10⁻³ |
| `n=2056, d=2048, p=6144` (unapply gate/up) | 2.65 × 10⁻³ | 1.87 × 10⁻³ |

The bf16 round-trip injects ~0.19 % mean relative error when both
mask and model are bf16. That sits **inside** the bf16 model's own
quantization noise band (paper Table 1: ≥98.8 % top-1 token equality
at bf16). Parity is acceptable in principle.

**Implementation blocker**: vendored AOCL-BLIS in
`vendor/aocl-install/lib/libblis-mt.so.5.2.2` exports no `sbgemm_` /
`bli_gemm_bf16bf16f32` symbols (verified via `nm | grep -i bf16`
returning empty). A real bf16 mask GEMM is gated on either upgrading
AOCL to a newer build with bf16 support, or hand-rolling AVX-512 BF16
intrinsics. Both are multi-day, beyond the 1-2 day budget. Test is
kept as a regression anchor for when AOCL bf16 lands.

### Step 3 (layer-skip) — regressed, with diagnosis

The prediction in this doc (§4.3) said skip-first-4-and-last-1 would
save ~14 % of mask volume. Measured: TTFT went **from 14.5 s to
20.8 s** at n=2048 (threads=16). The overhead-vs-plain ratio dropped
from +119 % to +47 %, but only because the plain baseline got slower
faster than gpu_gelo did.

Per-bucket diagnosis at the skip-4+1 config:

| op | baseline (28 offloaded) | skip 4+1 (23 offloaded, 5 in-TEE) | Δ |
|---|---:|---:|---:|
| `gelo:mask_apply` | 1 324 ms | 1 037 ms | −287 ms |
| `gelo:mask_unapply` | 2 331 ms | 1 850 ms | −481 ms |
| `engine:matmul*` | 4 263 ms | 3 618 ms | −645 ms |
| **mask + engine saved** | | | **−1.4 s** |
| `tee:qkv_direct` | 0 | 1 320 ms | +1 320 |
| `tee:o_direct` | 0 | 667 ms | +667 |
| `tee:swiglu_proj_direct` | 0 | **4 008 ms** | +4 008 |
| `tee:swiglu_down_direct` | 0 | 2 002 ms | +2 002 |
| **direct in-TEE added** | | | **+8.0 s** |
| net | | | **+6.6 s** |

**Root cause**: `ndarray.dot()` used by the `tee:*_direct` paths does
**not** route through BLIS — only `mask::sgemm_blis` does (the
`blas-ndarray` feature would change this but is off by default
because it would re-introduce the embedder small-shape cliff
catalogued in step 1). The in-TEE direct path runs at matrixmultiply
single-thread speed (~125 GFLOP/s), while the masked path runs at
BLIS-mt-16 speed (~1 200 GFLOP/s). So each skipped layer costs ~5×
more wall-clock than each protected layer.

This is a direct consequence of step 1 working so well: BLIS-mt-16
on the mask is now faster than ndarray-dot on the bare matmul.
Step 3 will only pay off after the in-TEE direct path also uses
BLIS-mt. Two paths to fix:

- Route `tee:*_direct` through `cblas_sgemm` with explicit thread
  control (the embedder regime would still want threads=1; the
  long-context decoder path can request threads=16). Touch points:
  `crates/gelo-embedder/src/decoder/forward.rs:271, 399, 419, 427,
  456, 540, 562, 570` (the `tee:*_direct` profile labels).
- Or enable `blas-ndarray` only for crates that are confirmed
  long-context (gelo-gpu-wgpu has the long-context bench; but
  routing all `ndarray.dot()` through BLIS at embedder shapes is the
  cliff we just measured).

**Until the in-TEE direct path is fixed, the layer-skip security
recommendation should NOT be turned into a perf claim.** The
security argument from GELO §3.2 still stands — sensitive-layer
exclusion is recommended on its own merits — but on this substrate,
turning it on increases wall time. Defaults stay
`skip_first_layers = 0`, `skip_last_layer = false`.

### What landed in main (build robustness)

- `default = ["blas"]` in `crates/gelo-protocol/Cargo.toml` and
  `crates/gelo-gpu-wgpu/Cargo.toml`. AOCL-BLIS must be installed via
  `scripts/install-aocl-blis.sh`; without it the build fails loud at
  link time rather than silently falling back to 5×-slower
  matrixmultiply (which is the bug we just caught).
- `GELO_BLIS_THREADS=N` env var honoured in
  `mask::blis_init_single_thread`. Default 1 (safe for embedder).
  Long-context bench scripts set `=16` explicitly.
- `mask::mask_backend_description()` exported; long-context bench
  prints it at startup so silent-fallback misfires are visible.
- bf16 parity test at
  `crates/gelo-protocol/tests/bf16_mask_parity.rs` runs by default
  (small shape, <1 s); realistic-shape variant is `#[ignore]`.
- Bench env knobs added: `GELO_BENCH_LENGTHS`,
  `GELO_BENCH_MAX_TOKENS`, `GELO_BENCH_SKIP_PERMUTED`.
  (`GELO_BENCH_SKIP_FIRST_LAYERS` / `GELO_BENCH_SKIP_LAST_LAYER` were
  added during the step-3 experiment and **reverted** after the
  regression diagnosis — re-add only after `tee:*_direct` routes
  through BLIS-mt.)

### Lever A revisited (2026-05-19 evening) — `tee_matmul` infra landed, defaults NOT flipped

A second attempt at the layer-skip recommendation — this time with
the `tee:*_direct` path routed through BLIS-mt via a new
`tee_matmul` dispatch (`crates/gelo-protocol/src/mask.rs` —
shape-threshold `n_rows >= 64`, BLIS above, `ndarray::dot()` below) —
was implemented and benched. Mixed result:

| metric | step-1 baseline (no skip, threads=16) | A (skip 4+1 + tee_matmul, threads=16) | delta |
|---|---:|---:|---:|
| gpu_gelo TTFT at n=2048 | 14 459 ms | **13 440 ms** | **−1.0 s** (−7 %) ✓ |
| gpu_gelo TPOT (4 decode steps) | 359 ms | **593 ms** | **+234 ms/step** ✗ |
| gpu_plain TPOT | 273 ms | 508 ms | +235 ms/step ✗ |

**The decode regression is real and dominates for any realistic
generation length.** Per skipped layer per decode step the cost
decomposition is:

```
tee:swiglu_proj_direct       28 ms / call (2 matmuls)  →  14 ms/matmul
tee:swiglu_down_direct       14 ms / call (1 matmul)
tee:qkv_direct                7 ms / call (3 matmuls)
tee:o_direct                  3 ms / call
                                                          ~52 ms / skipped layer
                                                          × 5 skipped layers = 260 ms / step
```

This is the `tee_matmul` fallback path (n_q=1 falls below the
threshold, routes to `ndarray::dot()`). The matmul itself is only
~25 MFLOPs for the swiglu projection at decode — should take ~250 µs
at 100 GFLOP/s nominal, but observed ~14 ms per matmul, i.e. ~1
GFLOP/s. The GPU at the same shape (autotuned cubecl) does it in
~1.4 ms.

**Defaults left at `skip_first_layers = 0, skip_last_layer = false`**.
The infrastructure (`tee_matmul`, `matmul_blis`, standard-layout
weight load fix in `decoder/weights.rs:read2_t`, parity test in
`mask.rs::tests::tee_matmul_parity_with_ndarray_dot`) is committed
and ready for future use. **Layer-skip cannot ship as a perf-positive
default until the m=1 GEMV path is fast.** Tracked as a separate
optimisation surface — see `memory/tee_direct_m1_gemv_slowness.md`
for the four ranked fix options (hand-rolled AVX-512 GEMV is the
direct fix at ~2-3 days effort).

For prefill-only or 0-decode-token workloads, A is a clean win and
can be enabled per call site by overriding the config. For mixed or
decode-heavy workloads it's a net regression. **Net recommendation:
defer A's default-flip; proceed to B (HD₃).**

### Next levers after steps 1-3

With the BLIS-mt-16 baseline at 14.5 s TTFT, the remaining shares of
prefill wall time (from the §3.1 profile of the round-3 doc, updated
2026-05-19 evening):

```
22.8 %  gelo:mask_sample        (Haar QR, single-thread O(s³))
18.6 %  engine:matmul_many       (GPU, won't shrink without GPU change)
16.3 %  gelo:mask_unapply        (BLIS-mt-16 GEMM)
13.1 %  engine:matmul            (GPU)
 9.8 %  gelo:mask_apply          (BLIS-mt-16 GEMM)
 9.5 %  tee:attn_cached          (in-TEE GQA)
 5.8 %  gelo:strip_shield        (memcpy)
remainder (small ops)
```

So the next-biggest single lever is `gelo:mask_sample` (Haar QR) at
22.8 %. This was item #6 (PRG-derived A) in the original §7 list but
won't fully help — the QR itself is the cost, and replacing the true
Gaussian seed with PRG output doesn't change the O(s³) cost of QR.
The right move is item #4: **HD₃ Hadamard cascade**, which removes
both the QR *and* makes the apply/unapply O(s·d·log s) instead of
O(s²·d), eliminating the top two and the #3 and #5 buckets in one
move. After HD₃ the prefill should be dominated by the GPU engine
matmuls and the in-TEE attention — the actual model compute, not the
protocol overhead.

---

## TL;DR — prioritised shortlist

Ranked by **payoff × feasibility × security defensibility** for our
SEV-SNP-CPU + commodity-GPU substrate:

| # | Lever | Expected speedup on `mask_apply + mask_unapply` at n=2048 | Security cost | Effort |
|---|---|---|---|---|
| 1 | **Multi-thread BLIS at long-n shapes** | 6–8× wall-time on the mask GEMMs | none | hours |
| 2 | **bf16 mask GEMM** (AVX-512 BF16 + AOCL-BLIS `bli_gemm_bf16bf16f32`) | 2× on top of (1) | tiny round-trip error; gated on bf16 round-trip test | days |
| 3 | **HD₃ Hadamard-cascade mask** (replace Haar) | 25–75× FLOPs; per-call cost moves from O(s²·d) to O(s·d·log s) | preserves κ=1 orthogonality; per-batch sign vectors give 3·s fresh bits; **requires running GELO attack suite (anchor / ICA / BSS) at s=2056 to confirm parity with dense Haar** before adoption | 1-2 weeks impl + security spike |
| 4 | **Slalom-style additive blinding** on the four MLP-side linear projections (`O`, `gate`, `up`, `FfnDown`), keep GELO permuted attention for Q·Kᵀ | ~2000× online compute (23 s → ~10 ms on those calls); **but** moves ~one full forward-pass-worth of FLOPs into a precompute phase | switches the security argument from BSS-hard to OTP (arguably stronger for the protected quantities); requires R·W precompute storage ~1.5 GB at fp16 for Qwen3-1.7B | 2-3 weeks impl + security write-up |
| 5 | **Aggressive sensitive-layer exclusion** (skip first 4 + last 1 of 28 layers; offload only middle 23) | 15–20 % | well-supported by DP-Forward §5, SecureInfer; deeper layers are less invertible | hours (flip two config defaults) |
| 6 | **PRG-derived A** (HKDF/AES-CTR seed → Haar-orthogonal) | only the 4 % `mask_sample` cost; ~0.5 s shaved at n=2048 | none under PPT-bounded adversary (our existing model) | days |
| — | **Confidential GPU (H100 CC / B200 TEE-I/O)** | 10-15× by moving the whole mask to GPU at multi-TFLOP/s | strong; same threat model as GELO paper; but **deployment fork — not the round-2 hardware scope** | hardware change |

Compound estimate, no threat-model change, no deployment fork:
**(1) + (2) + (5)** alone yields ~15–20× on the 62 s mask wall ⇒ TTFT at
n=2048 drops from 73 s to ~10–13 s, ~+50 % vs plaintext (compare paper's
"20-30 %" microbench target). Adding (3) or (4) gets within paper-range
or below.

**What NOT to pursue** (each broken or strictly weaker than the
above): bounded-depth Householder/Givens, banded orthogonal,
block-diagonal `A` (security degrades in proportion to speedup),
STIP-style 3-party permutation (HNM ICML '25), full FHE inference
(~8 min/token for LLaMA-7B per [BumbleBee]), IPFE per inner product
(infeasible at LLM scale), per-session `A` reuse (HNM-class attacks
break precomputed-basis schemes).

---

## 1. The problem (recap)

[`../archive/prototype/gelo-complexity-analysis.md`](../archive/prototype/gelo-complexity-analysis.md)
established that at n=2048 on Qwen3-1.7B:

- mask round-trip = 90 % of TTFT (62 s of 73 s wall)
- mask_unapply 38.7 s (196 calls/forward)
- mask_apply 23.5 s (112 calls/forward)
- mask_sample 2.9 s (1 call/forward, O(s³) Haar QR)
- CPU/BLIS at 125 GFLOP/s vs GPU at 1.4 TFLOP/s — **11× substrate gap**
- Headline overhead: +1 020 % vs plaintext-executor baseline

Asymptotic scaling is `O(s²·d)` per mask GEMM call (from `cblas_sgemm`
in `crates/gelo-protocol/src/mask.rs:192-208`); `O(s³)` for the Haar QR
(`mask.rs:233 sample_haar_orthogonal`); `O(n)` for the GPU engine
matmul. All three exponents fall out of the code, not a model.

Three orthogonal axes can move this: replace the primitive (§2),
relax the threat model (§3), or squeeze the existing substrate (§4).
A hybrid stack (§5) combines them.

---

## 2. Replace the primitive — structured orthogonal & sub-orthogonal masks

The GELO paper §6 future-work list explicitly flags this: *"Explore
faster constructions for fresh, well-conditioned mixing (e.g.,
structured orthogonal transforms)."* Survey covers Hadamard families,
butterfly/Monarch, Householder/Givens chains, banded/block-diagonal,
Kac walks, circulant/Toeplitz, and Liberty Lean Walsh.

### 2.1 Top candidate: **HD₃ cascade** (QuIP#-style)

`A = D₃ · H · D₂ · H · D₁ · H` where each `H` is the orthonormal Walsh-
Hadamard transform (no sampling — fixed matrix; `H·Hᵀ = I` exactly)
and each `Dᵢ` is a fresh `(s × s)` diagonal of ±1 entries.

| dimension | value |
|---|---|
| Sample cost | `3·s` random bits per forward (essentially free) |
| Apply cost | `3 · (5·s·d·log₂s)` FLOPs (three FWHTs interleaved with sign flips) — at s=2056, d=2048 that is **~6.9 × 10⁸** vs Haar's `1.73 × 10¹⁰` ⇒ **25× FLOP reduction** |
| Numerical conditioning | κ=1 exactly (each factor is orthonormal); safe at fp16/bf16/fp8 |
| Security: orthogonality | preserved exactly; Gram-leak surface identical to Haar (shield rows still required) |
| Security: BSS hardness | Tseng et al. ([QuIP#, arXiv:2402.04396](https://arxiv.org/abs/2402.04396)) prove incoherence bounds matching Haar to constants; a single HD is too low-entropy (Mohaisen-Hong 2008 ICA attack) but the 3-fold cascade has 3·s = 6 168 fresh bits/forward at our shape and no published BSS attack |
| Production code | [`fast-hadamard-transform`](https://github.com/Dao-AILab/fast-hadamard-transform) (Tri Dao); QuaRot CUDA kernels ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)); SpinQuant ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) |
| LLM track record | QuIP#, QuaRot, SpinQuant all in production for LLM **quantization** (not privacy); the orthogonal-mixing argument is the same primitive being used differently |

The standout property is the κ=1 orthogonality — every other "fast"
candidate either sacrifices orthogonality (SRHT, circulant) or covers
a thin sub-manifold of `O(s)` (banded, block-diagonal, bounded-depth
Householder). HD₃ alone among the cheap options preserves all of
GELO's existing security infrastructure: shield rows still work,
Gram-leak mitigation still works, `Aᵀ·A = I` exactly, the BSS
distinguishing game is unchanged in form (only the support of `A`
shrinks from `O(s)` to the HD₃ orbit).

The honest gap: **no published BSS-resistance proof specific to HD₃**.
The proxy evidence is QuIP#'s incoherence bound — same property the
BSS attacker exploits — but that bound was proved for the quantization
setting, not against an ICA/BSS adversary. **Before adoption, the GELO
paper's attack suite (anchor-based recovery, FastICA, JADE, JD; §4.3
of the paper) must be re-run against HD₃-with-shield at s=2056 and
non-anchor recovery cosine similarity compared against the dense-Haar
baseline.** If parity holds within the published noise band, HD₃ is
ready.

### 2.2 Other candidates — and why they don't beat HD₃

| Primitive | Speedup vs Haar | Security verdict |
|---|---|---|
| SRHT (Ailon-Chazelle, Tropp) | ~70× FLOPs | Not orthogonal — JL-isometry only; Gram leak is different from Haar's, may be exploitable |
| Butterfly / Monarch (Dao) | ~45× FLOPs | Sparsity pattern is public; reduces BSS to identifying ≤2·s log s bits — feasible cross-batch if `A` is reused |
| Block-diagonal `A` (B blocks of s/B) | B× FLOPs (8× at B=8) | Cross-block zeros in `Uᵀ·U` are observable; BSS reduces to B sub-problems of size s/B — ICA succeeds faster on smaller sub-problems |
| Banded orthogonal (bandwidth b) | (s/b)× | Banding visible directly in `U`; **trivially broken** |
| Householder / Givens depth k ≪ s | (s/k)× | Only a k-dim sub-space is mixed; complement passes through unmodified — trivial recovery against any anchor in the complement |
| Kac random walk | none asymptotic | Mixing time is Θ(s² log s) (Pillai-Smith 2017); to reach Haar from a depth-k Kac walk we need k = Ω(s²) — more expensive than Haar |
| Circulant / Toeplitz × signs (FFT) | ~70× | Not orthogonal (κ can be O(√log s)); 2s−1 independent entries — low-entropy mask |
| Liberty Lean Walsh (4-wise independent) | ~340× | Insufficient entropy for BSS resistance |

The pattern: every primitive cheaper than HD₃ either drops orthogonality
or covers a strict sub-manifold of `O(s)` that an attacker with any
side-information (anchors, partial plaintexts) can exploit. **HD₃ is
the Pareto frontier** at the "preserves all GELO security
infrastructure" point.

Sources: [Ailon-Chazelle FJLT](https://www.cs.princeton.edu/~chazelle/pubs/FJLT-sicomp09.pdf),
[Tropp SRHT analysis](https://arxiv.org/abs/1011.1595),
[Dao Monarch ICML'22](https://proceedings.mlr.press/v162/dao22a.html),
[Pillai-Smith Kac walk](https://arxiv.org/abs/1605.08122),
[Mhammedi depth-k Householder](https://arxiv.org/abs/1612.00188),
[Mezzadri Haar QR](https://arxiv.org/abs/math-ph/0609050),
[QuIP# (Tseng et al.)](https://arxiv.org/abs/2402.04396),
[QuaRot (Ashkboos et al.)](https://arxiv.org/abs/2404.00456).

---

## 3. Crypto-hybrid — Slalom-style additive blinding

The single largest perf lever in this survey, but **changes the
security argument**. Slalom-style additive blinding ([Tramèr & Boneh,
ICLR'19, arXiv:1806.03287](https://arxiv.org/abs/1806.03287)) does:

```
offline (per fresh mask):
    R ← uniform random in R^(s × d)
    R·W ← precomputed by TEE                      (one full GEMM per linear layer)
online (per batch):
    U ← X + R                                      (s · d additions)
    V ← U · W                                      (untrusted GPU matmul)
    H·W ← V − R·W                                  (s · d_out subtractions)
```

Online TEE cost drops from `O(s²·d)` to `O(s·d)` — for our shape
**8 GFLOP → 4 MFLOP per linear layer**, mapping our 23 s `mask_apply`
to roughly **10 ms**.

### 3.1 The catch: offline R·W precompute

The offline phase is one full GEMM per linear layer per fresh mask.
For Qwen3-1.7B at our shapes that's ~270 GFLOPs of TEE-side precompute
per fresh-masked prefill — about **2 s of CPU/BLIS work** to set up
each fresh forward.

Two ways this becomes acceptable:

- **Per-session, not per-batch.** Reuse `R` across multiple forward
  passes in the same session (e.g., one prefill plus N decode steps).
  Slalom-as-published reuses `R` across many inferences;
  TwinShield ([arXiv:2507.03278](https://arxiv.org/abs/2507.03278))
  and PermLLM ([NeurIPS '24,
  arXiv:2405.18744](https://arxiv.org/abs/2405.18744)) both do this in
  the LLM setting. **But:** mask reuse across batches is exactly the
  pattern broken by [Hidden No More (ICML
  '25)](https://arxiv.org/abs/2505.18332) — under fresh per-batch `R`
  the OTP argument is information-theoretic; under reuse, the
  attacker accumulates `U_t = X_t + R` across t and can compute
  `U_t − U_{t'} = X_t − X_{t'}`, revealing pairwise differences. For
  Qwen3-1.7B-class inference where consecutive batches in a session
  process different token positions, this is enough to recover the
  prompt under most known activation priors.

- **Per-batch fresh R, but precompute pipelined with KV-cache
  warm-up.** During the user's first prefill, the TEE can pipeline
  `R·W` precompute for the *next* expected batch — works at high
  per-user QPS, doesn't work for one-shot RAG queries.

### 3.2 The security trade — OTP vs BSS

Slalom-style additive is **information-theoretic** under fresh
uniform `R`: the GPU sees `U = X + R` which reveals nothing about
`X` to an unbounded adversary. By contrast GELO's BSS argument is
identifiability-up-to-unknown-invertible (per-batch
non-identifiability); strong empirical security but not information-
theoretic. So for the protected quantities — the activations themselves
— additive is **arguably stronger** than GELO's mask.

What you lose:

- **The Q·Kᵀ attention is not protected by Slalom.** Both factors are
  private activations; Slalom assumes one factor is a public weight
  `W`. For Q·Kᵀ the construction degrades to two-OTP secret-shared
  multiplication, which costs an extra GEMM and a TEE-side
  reconstruction (the protocol TwinShield documents). This is exactly
  the regime GELO's permuted attention already handles, suggesting
  the natural hybrid below.

- **Storage**: `R·W` for Qwen3-1.7B at fp16 ≈ 1.5 GB encrypted in
  untrusted memory per fresh mask. Large but not prohibitive.

### 3.3 The natural hybrid: Slalom-projections + GELO-attention

| operation | what it processes | recommended primitive |
|---|---|---|
| `offload_qkv` (Q, K, V projections of `h_norm`) | one private factor `H`, one public factor `W` | **Slalom additive** |
| `offload_attention_permuted_cached` (Q·Kᵀ, probs·V) | both factors private | **GELO permuted (current)** |
| `offload_linear(O)` (attention output projection) | one private, one public | **Slalom additive** |
| `offload_linear_many([gate, up])` (FFN gate + up) | one private, one public | **Slalom additive** |
| `offload_linear(FfnDown)` (FFN down projection) | one private, one public | **Slalom additive** |

This hybrid keeps GELO's BSS argument for the genuinely-private
attention compute (where Slalom doesn't apply) and replaces the
five-out-of-six linear projection sites with Slalom-style additive
blinding. Online cost on the projection sites collapses to ~0; the
remaining cost is GELO permuted attention (1.4 s/forward at n=2048
per the §3.2 of the complexity-analysis doc) plus the offline R·W
precompute (~2 s pipelined).

### 3.4 What else is in this family (catalogued but not recommended for our setting)

- **DarKnight (MICRO'21,
  [arXiv:2207.00083](https://arxiv.org/pdf/2207.00083))**: K-input
  coding-matrix generalisation of Slalom. K³ TEE-side decode cost;
  better when K large batches available — our regime is single-stream.
- **Goten (AAAI'21)**: requires 2-3 non-colluding TEEs, doesn't fit
  our deployment.
- **ShadowNet (S&P'23), SOTER (USENIX ATC'22)**: mask the **weights**
  with a fixed transform. Falls into the [HNM-class precomputed-basis
  attacks](https://arxiv.org/abs/2602.11088) — don't use.
- **AsymML / 3LegRace
  ([PoPETs'22](https://petsymposium.org/popets/2022/popets-2022-0105.pdf))**:
  TEE keeps low-rank-r component of `W`, GPU sees residual + DP noise.
  ~32× cheaper at r=64 — but the cost is paid via accuracy degradation
  + DP-noise budget. Worth a small pilot.
- **PermLLM ([NeurIPS'24,
  arXiv:2405.18744](https://arxiv.org/abs/2405.18744)),
  Fission ([eprint 2025/653](https://eprint.iacr.org/2025/653.pdf))**:
  A-SS-based MPC + permutation triples. Strictly stronger trust
  requirements (non-colluding party) — out of scope.
- **Euston / NEXUS RNS-CKKS FHE**:
  ~8 min/token for LLaMA-7B per
  [survey 2412.08145](https://arxiv.org/pdf/2412.08145). Not in our
  latency regime.
- **IPFE (Abdalla, Agrawal)**: ms-scale per inner product; would need
  s·d inner products per linear layer — **hours per layer** at our
  shape. Not viable.
- **SCX ([SIGCOMM'25](https://doi.org/10.1145/3718958.3750509))**:
  closest "near-zero online cost" competitor, but uses (ε,0)-DP
  rather than OTP — weaker privacy posture; track as a benchmark, not
  a baseline.

### 3.5 Closely related published systems

**TwinShield ([arXiv:2507.03278](https://arxiv.org/abs/2507.03278),
2025)** is the closest deployed analogue of the proposed hybrid:
additive secret sharing on the linear projections + permuted Q·Kᵀ for
attention, reports 87 % of FLOPs offloaded and 4-6× speedup over
prior TEE-only. Worth a deep read before designing our hybrid.

---

## 4. Threat-model relaxations

Each row states an assumption and a perf gain. The literature evidence
column links to the strongest published support; don't adopt anything
in this section without (a) a written security note and (b) a re-run
of the paper's attack suite at our shapes.

### 4.1 Confidential GPU substrate (paper's intended target)

| dimension | value |
|---|---|
| Assumption | trust the GPU's on-die security processor + encrypted HBM access-control + (Blackwell) encrypted NVLink fabric |
| Perf gain | TTFT overhead drops to **~5-8 %** vs plaintext GPU per [ETH benchmark study](https://arxiv.org/abs/2509.18886); B200 TEE-I/O claims ~0 % per [Corvex](https://www.corvex.ai/blog/confidential-computing-meets-nvidia-hgxtm-b200-secure-ai-without-the-performance-trade-off) (vendor-only, unverified) |
| Attack surface delta | adds GPU firmware/BAR0 paths, HBM probe (out of scope per [NVIDIA WP-11459 §threat model](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/HCC-Whitepaper-v1.0.pdf)); removes BLIS CPU side-channel surface |
| Evidence quality | strong — multiple independent benchmarks (ETH, Phala, ACM Queue 2024) |
| Hardware availability | H100/H200 CC available since 2023; B200 TEE-I/O 2025; AMD MI300X CC announced |
| Effort | hardware migration; protocol-side change is nil |

This is the paper's actual deployment target. **Treated here as a
deployment fork**, not the primary path, per the round-2 hardware
scope. Worth listing because (a) some users will have access to
confidential GPUs and the protocol should still work there, and
(b) the perf delta quantifies the cost of *not* using one.

### 4.2 PRG-derived `A` (computational pseudorandomness)

Replace Haar-random `A` with HKDF or AES-CTR expansion of a session
seed, then orthogonalise via QR. Security under a PPT-bounded
adversary is the standard Maurer indistinguishability
result ([EUROCRYPT '02](https://crypto.ethz.ch/publications/files/Maurer02.pdf)).

| dimension | value |
|---|---|
| Assumption | adversary is polynomial-time-bounded |
| Perf gain | only the 4 % `mask_sample` cost (2.9 s at n=2048 ⇒ ~0.5 s after) |
| Attack surface delta | none under our existing threat model |
| Evidence quality | strong — standard cryptographic argument |
| Effort | days; the QR step is still O(s³) so the win is small unless we also adopt §2.1 HD₃ which removes the QR entirely |

Free under our threat model — implement it as a hygiene improvement
regardless of the bigger choice in §2-3.

### 4.3 Aggressive sensitive-layer exclusion

GELO §3.2 already recommends "do not apply GELO to the first few
layers nor the final layer". Our code supports this (`config.rs:224
offload_layer` with `skip_first_layers` and `skip_last_layer`), but
the defaults are `0` and `false` — i.e., we offload everything.

| dimension | value |
|---|---|
| Assumption | embedding-inversion success drops with depth; deeper hidden states encode less raw-token information |
| Perf gain | skip 4 of 28 = ~14 % of mask volume |
| Attack surface delta | first-few-layer activations are sent to GPU plaintext, so the leak is *direct* but limited to whatever embedding-layer arithmetic the GPU sees; HNM-class inversion attacks on raw embedding-layer outputs do not yet show working recovery without DP noise |
| Evidence quality | medium — DP-Forward §5/§6 measures 20pp accuracy retention when perturbing deeper layers; SecureInfer ([arXiv:2510.19979](https://arxiv.org/abs/2510.19979)) gets 3.7× over TEE-only via a similar split; TEESlice ([arXiv:2411.09945](https://arxiv.org/html/2411.09945v1)) provides a formal framework |
| Effort | flip two config defaults; verify with attack suite |

If we drop the most-sensitive-layer constraint to "skip first 4, skip
last 1", we save ~18 % at almost no engineering cost. Worth a small
empirical attack run before adoption.

### 4.4 What NOT to adopt (with attack references)

- **Per-session `A` reuse** — [Hidden No More
  (HNM)](https://arxiv.org/abs/2505.18332) breaks every precomputed-
  basis scheme (ArrowMatch, glide-reflection, STIP, PermLLM-static);
  the moment `A` is reused across batches the BSS argument inverts.
- **STIP-style 3-party permutation** — directly broken by HNM with
  near-perfect prompt reconstruction.
- **DP-noise injection as a *replacement* for the mask** — DP-Forward
  ([CCS'23, arXiv:2309.06746](https://arxiv.org/html/2309.06746))
  achieves 88pp drop in embedding-inversion success at ε=8 *for
  classification*. For generation, ε composes across 28 layers × N
  decoded tokens — effective ε explodes. Useful only as **defence in
  depth** on top of GELO, never as a replacement.
- **Hypervisor in TCB** — undoes SEV-SNP's protection. [TEE.Fail](https://tee.fail/)
  and [WeSee (arXiv:2404.03526)](https://arxiv.org/pdf/2404.03526)
  already actively erode CC trust on Intel/AMD; weakening further is
  the wrong direction.
- **Token-sharding (Cascade,
  [arXiv:2507.05228](https://arxiv.org/abs/2507.05228))** — works
  against HNM by construction but requires non-colluding multi-party
  hosting. Different deployment model; useful as a comparison.

---

## 5. Code-side perf levers — no protocol change

Sourced from the inline audit at the start of this round. Independent
of §2-4; combine freely.

### 5.1 Multi-thread BLIS at long-n shapes

`crates/gelo-protocol/src/mask.rs:101 blis_init_single_thread` pins
BLIS to 1 thread per call. The comment justifies this for embedder
shapes (`n≈400, d∈{1024,3072}`) where per-call thread-barrier cost
dominates. At n=2048 the mask GEMM is 2.6 TFLOPs per call — easily
above the regime where multi-thread BLIS pays off.

Hardware here is **AMD Ryzen AI MAX+ 395 (Strix Halo)**, 16 cores / 32
threads, AVX-512 + `avx512_bf16` + `avx512_vnni`. AOCL-BLIS scaling
benchmarks on Zen 5 sgemm at (2048×2048)·(2048×2048) show roughly
linear up to 8 cores, ~6× wall-time at 8 threads.

Recommended path: pin to 1 thread below `s = 768` (existing embedder
regime), switch to 8 threads above. Single config flag. Expected
prefill speedup on the mask wall: ~5-7× at n=2048 ⇒ TTFT drops from
73 s to ~25 s on this lever alone. **Needs a one-shot benchmark to
confirm** before landing.

### 5.2 bf16 mask GEMM

AOCL-BLIS exposes `bli_gemm_bf16bf16f32` (bf16 inputs, f32 accumulate)
which uses AVX-512 BF16 dotprod (2× FLOP/cycle vs f32). Combined with
5.1 (multi-thread): **theoretical 12-16× over current**.

The protocol-level concern: bf16 truncation of `A` propagates to
`Aᵀ·A` round-trip error. At fp16 (10-bit mantissa) the accumulated
relative error over a 2048-dim inner product is ~2 × 10⁻³ which is
already at the edge of model-output drift. bf16 has 7-bit mantissa —
worse — but with f32 accumulate the inner-product error is bounded by
the input rounding, not the accumulation. Empirical question; needs a
parity test against the existing f32 path. GELO paper Table 1 reports
≥98.8 % top-1 token equality at bf16 *for the entire model*; the mask
contribution is a small fraction of that error budget.

If bf16 mask round-trip stays within the existing parity tolerance,
this is a multiplicative win on top of 5.1.

### 5.3 Other minor levers

- **CPU/GPU overlap** — GPU is only 6 % of TTFT at n=2048, so even
  perfect overlap saves ≤6 %. Not worth the pipeline engineering.
- **Three parallel mask_unapply for QKV** — competes with multi-thread
  BLIS for cores; subsumed by 5.1.
- **Fused sample + apply** (apply Householder reflections directly to
  `H` without materialising `A`) — same FLOP count, worse cache
  pattern. Skip.

---

## 6. Creative angles / synthesis

### 6.1 Mask precomputation pool

If we adopt HD₃ (§2.1), mask sampling becomes essentially free (3·s
bits = 6 KB). Pre-sample a pool of N=16 fresh HD₃ masks at session
start. Each forward pass consumes one mask deterministically (FIFO).
Refresh the pool asynchronously. Decouples sampling latency from
forward-pass critical path entirely.

### 6.2 Stacked primitives

Nothing in §2 / §3 / §4 / §5 is mutually exclusive (with one exception
called out below). The natural stack:

```
substrate:         AMD Strix Halo, 16 cores, AVX-512_BF16, Vulkan iGPU (current)
threading:         multi-thread BLIS above s=768                    (§5.1)
precision:         bf16 inputs, f32 accumulate for mask GEMM         (§5.2)
mask primitive:    HD₃ cascade with fresh ±1 sign vectors            (§2.1)
linear projections: Slalom-style additive (QKV, O, gate, up, FfnDown) (§3.3)
attention compute:  GELO permuted (Q·Kᵀ, probs·V) — unchanged
layer policy:       skip first 4 + last 1 of 28                      (§4.3)
sample generation:  PRG-derived signs from HKDF over session key     (§4.2)
mask reuse:         per-forward-pass (current) or per-session (§3.1)
```

Conservative compound estimate keeping current `A`-reuse policy
(per-forward fresh):
- (§5.1) × 6 on the mask wall ⇒ 62 s → ~10 s
- (§5.2) × 2 on top ⇒ ~5 s
- (§4.3) × 0.85 ⇒ ~4 s
- Net TTFT at n=2048 ≈ 4 s (mask) + 4 s (engine + in-TEE attn) ≈ **8 s** (vs current 73 s, vs plaintext baseline 6.5 s — ~+25 % overhead, close to paper target)

Aggressive compound estimate adding §2.1 (HD₃) — replaces dense mask:
- mask wall at n=2048: 7.7 TFLOPs/forward → 7.7/25 = 0.3 TFLOPs/forward at HD₃
- at 1 TFLOP/s BLIS (multi-thread + bf16): **~0.3 s on the entire mask round-trip**
- TTFT ≈ 0.3 s (mask) + 4 s (engine + attn) ≈ **4.5 s** vs plaintext 6.5 s — **negative overhead** (mask is now cheaper than the rest)

Most aggressive — additionally §3.3 (Slalom for projections, GELO for
attention only) — the mask wall on projections vanishes; only attention
permutation remains. Estimated TTFT ≈ 4 s, essentially plaintext.

The big mutual-exclusion: §2.1 HD₃ and §3.3 Slalom-additive *can*
stack — apply HD₃ as the GELO mask on the Q·Kᵀ attention compute, and
Slalom-additive on the linear projections — but the security analysis
of stacking is new. Single-primitive paths are safer to validate first.

### 6.3 Empirical attack-suite validation as a release gate

Every proposed move in §2-3 changes the security posture from "exactly
what the paper measured" to "extension of the paper's argument."
Before any of HD₃, Slalom, layer exclusion, or PRG-derived `A` lands
in main, **the GELO paper's published attack pipeline (§4.3 of the
paper: anchor-based recovery, FastICA, JADE, Joint Diagonalization,
constrained ICA) must be re-run at our shapes with the new primitive
and compared against the dense-Haar baseline.** The Lin et al.
inversion-attack paper ([arXiv:2411.05034](https://arxiv.org/abs/2411.05034))
provides additional attack code for the obfuscated-embedding setting.

The acceptance criterion: non-anchor cosine similarity p95 within
GELO paper Table 6's noise band of the dense-Haar baseline, and
Frobenius Gram-error per Table 7 within ±20 % of dense-Haar at
matched shield density.

This is a 1-2 week security-spike line item; treat it as gating on
each of HD₃, Slalom, and layer exclusion.

### 6.4 Direction not covered here but worth flagging

- **Sparse attention masks for long-context** — paper-orthogonal,
  doesn't reduce mask cost, but reduces the *baseline* compute the
  mask is proportional to. If we adopt sliding-window attention (SWA)
  for n>2k, the engine GEMM cost drops and the mask amortisation
  improves.
- **Speculative decoding** — multiple-token-per-step decode dilutes
  the per-token mask cost. Currently single-token greedy in our bench.
- **Quantization (Q8) of weights** combined with HD₃ — QuIP# / QuaRot
  already implement this combination for compression; we'd get the
  privacy mask "for free" on top of quantization. The bf16/Q8
  precision considerations of §5.2 would need full re-examination but
  the upstream quantization pipelines have already done that work.

---

## 7. Recommended next steps — concrete experiments

In priority order:

1. **One-shot multi-thread-BLIS benchmark at our shapes.** Single
   `cargo bench` or modified `qwen3_long_context_bench` run with
   `BLIS_NUM_THREADS` swept over {1, 2, 4, 8, 16}. Confirms the §5.1
   prediction. ½ day.

2. **bf16 mask GEMM parity test.** Add a `bf16` feature flag to
   `gelo-protocol` that swaps `cblas_sgemm` for
   `bli_gemm_bf16bf16f32`. Run existing parity tests
   (`mask::tests::mask_round_trip_preserves_matmul`,
   `crates/gelo-embedder/tests/generation_harness::*`). If parity
   tolerance widens unacceptably, fall back to mixed-precision (f32
   `A`, bf16 hidden). 1-2 days.

3. **Flip sensitive-layer-exclusion defaults.** Set
   `skip_first_layers = 4, skip_last_layer = true` in Qwen3-1.7B
   config; re-run `qwen3_generation_e2e.rs` and `qwen3_long_context_bench`.
   Validate top-1 token parity. ¼ day.

4. **HD₃ implementation spike.** Add an `HD3Mask` alongside `GeloMask`
   in `gelo-protocol/src/mask.rs`. Use `rustfft` or a hand-rolled FWHT
   for the Hadamard step. Wire as an opt-in
   `InProcessTrustedExecutor::with_hd3_mask()`. Run parity tests, then
   the long-context bench. 1 week.

5. **HD₃ attack-suite re-run.** Port the GELO paper's attack pipeline
   (FastICA via the `linfa-ica` crate, JADE / JD via reference
   implementations) into `crates/gelo-attacks` (new crate).
   Baseline against dense Haar; measure non-anchor cosine similarity
   and Gram error per paper §4.3.3 / §4.3.4. Gate HD₃ adoption on
   parity. 2 weeks.

6. **Slalom-additive spike on a single linear-projection site.**
   Replace `offload_linear(FfnDown)` (the largest single mask GEMM)
   with Slalom additive. Wire a precompute-pool of `R, R·W` triples.
   Bench at n=2048 with and without R·W precompute pipelined.
   Confirm parity with end-to-end generation. 2-3 weeks.

7. **TwinShield deep-read.** Before designing the full Slalom-hybrid,
   read [arXiv:2507.03278](https://arxiv.org/abs/2507.03278) section
   by section; their `OutAttnMult` is already in our codebase under
   the OutAttnMult name. Identify primitives reusable from there.
   1 day.

8. **Long-context bench at n ∈ {4 096, 8 192}.** Both for the current
   path (to confirm asymptotics extrapolate as the complexity-analysis
   predicts) and for each post-optimisation revision. 1 day per
   iteration.

Items 1, 2, 3 are nearly-free engineering with strong expected
returns; items 4-6 are research-grade and each comes with a security
spike attached. Items 7-8 are supporting.

---

## 8. Citations

The three research agents that produced the underlying surveys are
internal to this session; sources cited are public artefacts.

### Primary GELO paper and threat-model background
- [Belikov & Fedotov, *Good-Enough LLM Obfuscation*, arXiv:2603.05035](https://arxiv.org/abs/2603.05035)
- [`../archive/prototype/gelo-complexity-analysis.md`](../archive/prototype/gelo-complexity-analysis.md) — bottleneck numbers driving this round
- [`../archive/research/private-llm-inference-R2-2026-05-18.md`](../archive/research/private-llm-inference-R2-2026-05-18.md) — predecessor

### Hadamard / structured orthogonal family (§2)
- [Ailon & Chazelle, *Fast Johnson-Lindenstrauss Transform*, STOC'06](https://www.cs.princeton.edu/~chazelle/pubs/FJLT-sicomp09.pdf)
- [Tropp, *Improved Analysis of the SRHT*, SIMAX'11](https://arxiv.org/abs/1011.1595)
- [Mezzadri, *How to generate random matrices from the classical compact groups*](https://arxiv.org/abs/math-ph/0609050)
- [Tseng et al., *QuIP#: Even Better LLM Quantization with Hadamard Incoherence*, ICML'24](https://arxiv.org/abs/2402.04396)
- [Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, 2024](https://arxiv.org/abs/2404.00456)
- [Liu et al., *SpinQuant*, 2024](https://arxiv.org/abs/2405.16406)
- [Dao et al., *Monarch: Expressive Structured Matrices*, ICML'22](https://proceedings.mlr.press/v162/dao22a.html)
- [Chen, Dao et al., *Pixelated Butterfly*, ICLR'22](https://arxiv.org/abs/2112.00029)
- [`fast-hadamard-transform` (Tri Dao)](https://github.com/Dao-AILab/fast-hadamard-transform)
- [Pillai & Smith, *Mixing of Kac walk on SO(n)*](https://arxiv.org/abs/1605.08122)
- [Mhammedi et al., *Householder RNN*, ICML'17](https://arxiv.org/abs/1612.00188)
- [Mohaisen & Hong, *ICA attack on rotation masking*](https://arxiv.org/abs/0906.0202)
- [ButterflyQuant](https://arxiv.org/abs/2509.09679)

### Slalom / additive-blinding family (§3)
- [Tramèr & Boneh, *Slalom*, ICLR'19](https://arxiv.org/abs/1806.03287) — [code](https://github.com/ftramer/slalom)
- [DarKnight, MICRO'21](https://arxiv.org/pdf/2207.00083)
- [Goten, AAAI'21](https://lucieno.github.io/files/goten.pdf)
- [ShadowNet, S&P'23](https://arxiv.org/abs/2011.05905)
- [SOTER, USENIX ATC'22](https://www.usenix.org/conference/atc22/presentation/shen) — [code](https://github.com/hku-systems/SOTER)
- [AsymML / 3LegRace, PoPETs'22](https://petsymposium.org/popets/2022/popets-2022-0105.pdf)
- [TwinShield, 2025](https://arxiv.org/abs/2507.03278) — closest deployed analogue of the §3.3 hybrid
- [PermLLM, NeurIPS'24](https://arxiv.org/abs/2405.18744)
- [Fission, eprint 2025/653](https://eprint.iacr.org/2025/653.pdf)
- [Slalom at the Carnival, CiC'24](https://cic.iacr.org/p/1/3/40)
- [Euston (NEXUS-successor), 2025](https://github.com/FLL-Lab/Euston)
- [SCX, SIGCOMM'25](https://doi.org/10.1145/3718958.3750509)
- [IPFE SoK (Abdalla et al., 2204.05136)](https://arxiv.org/pdf/2204.05136)
- [Saini, Jiang, Liu, *Vulnerabilities in Precomputed-Noise TEE Inference*, 2602.11088](https://arxiv.org/pdf/2602.11088) — broken-precomputed-basis reference

### Threat-model / confidential GPU (§4)
- [ETH benchmark study, *Confidential LLM Inference Across CPU and GPU TEEs*, arXiv:2509.18886](https://arxiv.org/abs/2509.18886)
- [Phala/NVIDIA H100 CC benchmark, arXiv:2409.03992](https://arxiv.org/abs/2409.03992)
- [*Performance of Confidential Computing GPUs*, arXiv:2505.16501](https://arxiv.org/abs/2505.16501)
- [NVIDIA Hopper CC Whitepaper WP-11459](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/HCC-Whitepaper-v1.0.pdf)
- [Confidential Computing on B200, Corvex blog](https://www.corvex.ai/blog/confidential-computing-meets-nvidia-hgxtm-b200-secure-ai-without-the-performance-trade-off)
- [Creating the First Confidential GPUs, ACM Queue 2024](https://queue.acm.org/detail.cfm?id=3623391)
- [GPU CC Demystified, arXiv:2507.02770](https://arxiv.org/html/2507.02770v1)
- [TEE.Fail (Intel-2025-10-28-001)](https://tee.fail/)
- [WeSee SEV-SNP break, arXiv:2404.03526](https://arxiv.org/pdf/2404.03526)
- [Maurer, *Indistinguishability of Random Systems*, EUROCRYPT'02](https://crypto.ethz.ch/publications/files/Maurer02.pdf)
- [DP-Forward, CCS'23 (arXiv:2309.06746)](https://arxiv.org/html/2309.06746)
- [SecureInfer, arXiv:2510.19979](https://arxiv.org/abs/2510.19979)
- [TEESlice, arXiv:2411.09945](https://arxiv.org/html/2411.09945v1)

### Recent attacks against private-LLM-inference schemes (§4.4)
- [Thomas et al., *Hidden No More*, ICML'25 (arXiv:2505.18332)](https://arxiv.org/abs/2505.18332)
- [Cascade (token-sharding, post-HNM-resistant), arXiv:2507.05228](https://arxiv.org/abs/2507.05228)
- [Lin et al., *Inversion Attack Against Obfuscated Embedding*, arXiv:2411.05034](https://arxiv.org/abs/2411.05034)

### Hidden-state geometry / concentration evidence (§4 sub-Gaussian)
- [Dimensional Collapse in Transformer Attention, arXiv:2508.16929](https://arxiv.org/pdf/2508.16929)
- [Shape of Learning: Anisotropy & Intrinsic Dim, arXiv:2311.05928](https://arxiv.org/pdf/2311.05928)

### Background surveys
- [Private Transformer Inference Survey 2024, arXiv:2412.08145](https://arxiv.org/pdf/2412.08145)
- [PipeLLM (ASPLOS'25, arXiv:2411.03357)](https://arxiv.org/abs/2411.03357)
- [Confidential GPU Computing 2026 Guide (Spheron)](https://www.spheron.network/blog/confidential-gpu-computing-nvidia-tee-encrypted-vram/)
