---
type: prototype-note
status: stale
created: 2026-05-19
updated: 2026-05-19
tags: [gelo, perf]
archive_reason: "2026-05-19 Qwen3-1.7B Haar baseline; predates HD₃ adoption (2026-05-19+) and the M1.12 perf work. Methodology (per-bucket accounting, FLOPs × throughput cross-check) is referenceable via git history; numbers are stale. Current perf truth: docs/plans/m1-12-*.md + docs/handoffs/2026-05-22+, plus docs/dev/prototype/hd3-non-pow2-fix.md for mask-family choice."
---

# GELO Complexity Analysis — Qwen3-1.7B Long-Context Forward

> **Scope.** Code-grounded, measured bottleneck decomposition of the
> paper-parity GELO offload path as implemented in `gelo-protocol` +
> `gelo-embedder` + `gelo-gpu-wgpu`. Per-bucket asymptotic class,
> per-forward invocation counts, measured wall-time across `n ∈
> {256, 512, 1024, 2048}`, FLOPs × throughput, and a reconciliation
> against the paper's headline overhead numbers (Belikov & Fedotov,
> arXiv 2603.05035, §4.2). Bench source:
> `crates/gelo-gpu-wgpu/tests/qwen3_long_context_bench.rs`; raw run
> kept in `/tmp/qwen3_long_bench.log` (commit cbea549 / 2026-05-19).

---

## Definitions

| symbol | meaning | Qwen3-1.7B value |
|---|---|---|
| `L` | decoder layer count | 28 |
| `n` | prompt token count (prefill) / current step token count (decode = 1) | sweep |
| `k` | shield row count appended to hidden state before masking | 8 |
| `s` | mask side length = `n + k` | n + 8 |
| `d` | hidden size | 2048 |
| `c` | KV projection width = `num_kv_heads · head_dim` | 1024 (8 × 128) |
| `f` | FFN intermediate size | 6144 |
| `h_q` | num query heads | 16 |
| `d_h` | head dim | 128 |
| TTFT | time to first token (prefill wall) | measured |
| TPOT | time per output token (mean decode-step wall) | measured |

The bench uses paper-parity defaults: per-forward Haar `A` + shield(k=8,
σ_scale=4.0). The mask is sampled **once per forward pass** (one per
prefill, one per decode step) and **reused across every `offload_*`
call in that forward**.

---

## 1. Bucket inventory — what each op is, where it lives, and its asymptotic

Each row below names a `profile::time` bucket that appears in the bench
output. The asymptotic column is what the **code actually executes**, not
a textbook prediction.

| bucket | source (file:line) | operation | per-call FLOPs | asymptotic in `n` |
|---|---|---|---|---|
| `gelo:mask_sample` | `gelo-protocol/src/mask.rs:233` `sample_haar_orthogonal` — outer `for k in 0..s-1` loop, each iteration calls `rank1_householder_update_rows` (two passes over an `(s-k)²` submatrix, mask.rs:310) plus `rank1_householder_update_cols` (two passes over an `s · (s-k)` block of Q, mask.rs:354) | Householder QR with Mezzadri-2007 sign correction on a fresh `s × s` Gaussian | `Σₖ [4(s-k)² + 4·s·(s-k)] ≈ 10·s³/3` | **O(s³)** |
| `gelo:mask_apply` | `mask.rs:48` `GeloMask::apply` → `sgemm_blis(A, H, transpose_a=false)` → `cblas_sgemm` M=s, N=d_in, K=s (mask.rs:192) | dense `(s × s) · (s × d_in)` GEMM via AOCL-BLIS, pinned single-thread (see `blis_init_single_thread`, mask.rs:101) | `2·s²·d_in` | **O(s²·d_in)**, quadratic in `n` |
| `gelo:mask_unapply` | `mask.rs:65` `GeloMask::unapply` → `sgemm_blis(A, masked, transpose_a=true)` | dense `(s × s)ᵀ · (s × d_out)` GEMM via BLIS | `2·s²·d_out` | **O(s²·d_out)** |
| `gelo:shield_stack` | `gelo-protocol/src/sim.rs:478, 490` — fill last `k` rows of the stacked-scratch buffer with `N(0, σ²)` samples via `rand_distr::StandardNormal::sample` | k row × d_in element-wise normal draws | `O(k·d_in)` | **O(1) in n** |
| `gelo:strip_shield` | `sim.rs:613, 685, 748` — slice off the trailing `k` rows from `(s, d_out)` and `.to_owned()` | `n · d_out` f32 copy | `O(n·d_out)` | **O(n)** |
| `engine:matmul` | `gelo-gpu-wgpu/src/lib.rs:281` — cubecl `lhs.matmul(weight)` where lhs is `(s, d_in)` and weight is `(d_in, d_out)`, executed on Vulkan via burn-cubecl | dense GPU GEMM | `2·s·d_in·d_out` | **O(n)** (constant per-launch overhead amortises) |
| `engine:matmul_many` | `lib.rs:322` — same kernel as `matmul`, looped over a `Vec<weight>` sharing one lhs | k dense GPU GEMMs with shared input | `2·s·d_in·Σd_out` | **O(n)** |
| `tee:attn_cached` | `gelo-embedder/src/decoder/attention.rs:239` `causal_gqa_attention_cached` — per-head `qh.dot(kh.t())` shape `(n, n)`, asymmetric causal mask, in-place softmax, `scores.dot(vh)`, rayon-parallelised over heads above `n ≥ 64` | per-head 2× `(n_q × d_h) · (d_h × n_kv)` GEMM + softmax | `h_q · (4·n_q·n_kv·d_h + n_q·n_kv)` | **O(n²)** at prefill, **O(n)** at decode (n_q=1) |
| `tee:attn_permuted_cached` | `attention.rs:312` `causal_gqa_attention_permuted_cached` → `exec.offload_attention_permuted_cached` → `gelo-protocol/src/attention.rs:284 permuted_attention_cached` | Two random permutations, optional Gaussian σ-noise on Q/K, GPU `Q·Kᵀ` + GPU `probs·V` matmuls, in-TEE permuted-causal mask + softmax + π_q⁻¹ unpermute | mix of `O(n²)` perm copies + `O(h_q·n_q·n_kv·d_h)` matmuls | **O(n²)** |
| `tee:rmsnorm` | `forward.rs:242, 404, 447` — `rms_norm(view, gamma_slice, eps)` | linear sweep | `O(n·d)` | O(n) |
| `tee:rope` | `forward.rs:315, 490` — applies RoPE tables to Q and K | element-wise rotation | `O(n·d)` | O(n) |
| `tee:residual`, `tee:swiglu_activate`, `tee:qk_norm` | various | element-wise / shape-preserving | `O(n·d)` | O(n) |

The `cblas_sgemm` calls at `mask.rs:192-208` are the literal source of the
`2·s²·d` FLOP accounting: `M = s` (mask rows), `N = d_in` (operand cols),
`K = s` (inner dim), `α=1, β=0`. Both apply and unapply are the same C
call with `op(A)` toggled — there is no algorithmic asymmetry between
them.

---

## 2. Per-forward invocation counts

Each row gives the number of times a bucket fires **per forward pass**.
Prefill = one `run_prefill` (`forward.rs:124`); decode = one
`run_decode_step` (`forward.rs:183`). Symbolic in `L` and the per-layer
decomposition of `decoder_block_cached` (`forward.rs:~440-575`); concrete
column is Qwen3-1.7B (`L = 28`).

| bucket | symbolic (prefill) | symbolic (decode step) | Qwen3-1.7B prefill | Qwen3-1.7B decode step | Bench × 4 decode steps |
|---|---:|---:|---:|---:|---:|
| `gelo:mask_sample` | 1 (per forward) | 1 | **1** | **1** | 4 |
| `gelo:mask_apply` | `4·L` (offload_qkv + O + gate_up + FfnDown) | `4·L` | **112** | **112** | 448 |
| `gelo:mask_unapply` | `(3+1+2+1)·L = 7·L` (Q,K,V from QKV; O; gate, up from gate_up; FfnDown) | `7·L` | **196** | **196** | 784 |
| `gelo:shield_stack` | `4·L` (one per `offload_*` call) | `4·L` | **112** | **112** | 448 |
| `gelo:strip_shield` | `4·L` | `4·L` | **112** | **112** | 448 |
| `engine:matmul` (GPU) | `2·L` (O + FfnDown) | `2·L` | **56** | **56** | 224 |
| `engine:matmul_many` (GPU) | `2·L` (QKV bundle + gate_up bundle) | `2·L` | **56** | **56** | 224 |
| `tee:attn_cached` | `L` (Global layers; all 28 in Qwen3-1.7B) | `L` | **28** | **28** | 112 |
| `tee:rmsnorm` | `2·L + 1` (pre-attn, pre-FFN, final) | `2·L + 1` | **57** | **57** | 228 |
| `tee:residual` | `2·L` | `2·L` | **56** | **56** | 224 |
| `tee:qk_norm` | `L` | `L` | **28** | **28** | 112 |
| `tee:rope` | `L` | `L` | **28** | **28** | 112 |
| `tee:swiglu_activate` | `L` | `L` | **28** | **28** | 112 |
| `tee:embed_lookup` | 1 | 1 | **1** | **1** | 4 |

The "Bench × 4 decode steps" column is verified against the call-count
field in every profile bucket dump from the measured run — counts match
exactly, which confirms the symbolic formulas above (and that the
per-forward-pass mask reuse is actually engaged: 1 `mask_sample` per
prefill, 1 per decode step, not per `offload_*`).

### Per-call shape catalogue

For the four `mask_apply` calls and seven `mask_unapply` calls within
one layer (these widths are the load-bearing constants in the FLOP math):

| call site | apply input width (`d_in`) | unapply output width (`d_out`) | unapply count |
|---|---:|---:|---:|
| `offload_qkv` (sim.rs:619) | `d = 2048` | `{d=2048, c=1024, c=1024}` | 3 |
| `offload_linear(O)` (forward.rs:399) | `d = 2048` | `d = 2048` | 1 |
| `offload_linear_many(gate, up)` (forward.rs:414) | `d = 2048` | `{f=6144, f=6144}` | 2 |
| `offload_linear(FfnDown)` (forward.rs:427) | `f = 6144` | `d = 2048` | 1 |
| **per-layer Σd_in (apply)** | **3·d + f = 12 288** | | |
| **per-layer Σd_out (unapply)** | | **3·d + 2·c + 2·f = 20 480** | 7 |

The unapply width sum `3d + 2c + 2f` — equivalent to `10·d` under
Qwen3-1.7B's particular ratios `c = d/2` and `f = 3·d` — is what makes
`mask_unapply` ~1.7× more expensive in wall-clock than `mask_apply`
(apply width sum is `3·d + f = 6·d`, ratio
unapply/apply = 20480 / 12288 = **1.67×**).

---

## 3. Measured wall-time across `n ∈ {256, 512, 1024, 2048}`

From `qwen3_1_7b_long_context_breakdown` (release build, AMD Radeon
GFX1151 iGPU, RADV/Mesa 25.2.8). Numbers are sums across the prefill
forward (all 28 layers, all calls).

### 3.1 Prefill (TTFT)

| bucket | n=256 (ms) | n=512 (ms) | n=1024 (ms) | n=2048 (ms) | calls (prefill) |
|---|---:|---:|---:|---:|---:|
| `gelo:mask_unapply` | 825.9 | 2 859.3 | 10 266.5 | **38 677.7** | 196 |
| `gelo:mask_apply` | 490.9 | 1 649.4 | 6 111.3 | **23 471.1** | 112 |
| `gelo:mask_sample` | 4.4 | 36.0 | 298.3 | **2 925.7** | 1 |
| `engine:matmul_many` (GPU) | 368.8 | 541.5 | 1 320.3 | 2 518.5 | 56 |
| `engine:matmul` (GPU) | 212.1 | 345.6 | 850.2 | 1 649.8 | 56 |
| `tee:attn_cached` | 45.4 | 123.9 | 371.0 | 1 445.4 | 28 |
| `gelo:strip_shield` | 29.4 | 114.2 | 387.6 | 785.1 | 112 |
| `gelo:shield_stack` | 30.2 | 44.7 | 76.8 | 152.3 | 112 |
| TTFT (wall) | 2 117.7 | 5 985.5 | 20 198.5 | **72 852.7** | — |
| baseline plaintext-executor TTFT | 568.7 | 2 825.5 | 2 636.2 | 6 501.9 | — |
| **overhead vs plaintext baseline** | +272 % | +112 % | **+666 %** | **+1 020 %** | — |

Mask round-trip share of prefill: 64 % (n=256) → 78 % (n=512) → 84 %
(n=1024) → **90 % (n=2048)**.

### 3.2 Decode (mean per-step over 4 steps; bucket ms = bucket total / 4)

| bucket | n=256 (ms/step) | n=512 | n=1024 | n=2048 | calls (decode step) |
|---|---:|---:|---:|---:|---:|
| `engine:matmul_many` (GPU) | 86.8 | 105.7 | 101.3 | **111.7** | 56 |
| `engine:matmul` (GPU) | 61.5 | 84.7 | 100.2 | **112.1** | 56 |
| `tee:attn_cached` | 15.7 | 30.4 | 64.7 | **130.0** | 28 |
| `gelo:shield_stack` | 16.4 | 17.4 | 19.7 | 24.6 | 112 |
| `gelo:mask_unapply` | 2.0 | 2.4 | 2.7 | 3.7 | 196 |
| `gelo:mask_apply` | 1.1 | 1.2 | 1.2 | 1.5 | 112 |
| `gelo:mask_sample` | 0.015 | 0.013 | 0.013 | 0.013 | 1 |
| TPOT mean (wall) | 184.3 | 242.8 | 291.2 | **385.4** | — |
| baseline plaintext-executor TPOT | 136.4 | 150.4 | 200.8 | 274.0 | — |
| overhead vs plaintext baseline | +35 % | +61 % | +45 % | +41 % | — |

At decode `n_q = 1` so mask costs collapse from `O(n²·d)` to `O(d)` per
call. The decode hot path is in-TEE attention scaling with the cached
`n_kv` (the prompt length + tokens-so-far) and GPU matmul — protocol
overhead is now in the noise.

---

## 4. Empirical scaling exponents vs code-derived asymptotics

Slope between consecutive `n` rows in §3.1 = `log(t(n)/t(n/2)) / log(2)`.

| bucket | 256→512 | 512→1024 | 1024→2048 | code-derived | match? |
|---|---:|---:|---:|:---:|:---:|
| `gelo:mask_apply` | 1.75 | 1.89 | **1.94** | O(s²·d_in) → s² → n² | ✓ |
| `gelo:mask_unapply` | 1.79 | 1.84 | **1.91** | O(s²·d_out) → s² → n² | ✓ |
| `gelo:mask_sample` | 3.02 | 3.05 | **3.29** | O(s³) → n³ | ✓ |
| `engine:matmul` | 0.70 | 1.30 | 0.96 | O(n) | ✓ (launch overhead at small n) |
| `engine:matmul_many` | 0.55 | 1.28 | 0.93 | O(n) | ✓ |
| `tee:attn_cached` | 1.45 | 1.58 | **1.96** | O(n²) prefill | ✓ |

The empirical slopes converge to the code-derived exponents from below
as `n` grows — the gap at small `n` is dominated by per-call fixed costs
(BLIS thread barrier, GPU launch, ndarray slice allocation, rayon
work-stealing init).

---

## 5. FLOPs × throughput per bucket at `n = 2048` (`s = 2056`)

| bucket | code-derived FLOPs per prefill | measured ms | implied throughput | device |
|---|---:|---:|---:|---|
| `gelo:mask_apply` | `2·L·s²·(3d+f) = 2·28·s²·12288` ≈ **2.91 TFLOPs** | 23 471 | **124 GFLOP/s** | CPU/BLIS (1 thread/call) |
| `gelo:mask_unapply` | `2·L·s²·(3d+2c+2f) = 2·28·s²·20480` ≈ **4.85 TFLOPs** | 38 678 | **125 GFLOP/s** | CPU/BLIS (1 thread/call) |
| `gelo:mask_sample` | `~10·s³/3` ≈ **28.9 GFLOPs** | 2 926 | **9.9 GFLOP/s** | CPU (rank-1 GEMV inner loops) |
| `engine:matmul` (O + FfnDown) | `2·L·s·d·(d+f) ≈` **1.93 TFLOPs** | 1 650 | **1.17 TFLOP/s** | Vulkan GPU |
| `engine:matmul_many` (QKV + gate_up) | `2·L·s·d·(4d+2f) ≈` **3.86 TFLOPs** | 2 519 | **1.53 TFLOP/s** | Vulkan GPU |
| **mask round-trip total** | ~7.76 TFLOPs CPU | 62 149 | **125 GFLOP/s** combined | CPU |
| **engine GPU total** | ~5.79 TFLOPs GPU | 4 168 | **1.39 TFLOP/s** combined | GPU |

**The headline ratio**: mask round-trip has **1.34×** the FLOP volume of
the engine GEMM it's protecting, but runs on a substrate that is **11×
slower**. That product (1.34 × 11 = **15×**) is the wall-time ratio
between "protocol overhead" and "actual model compute" at n=2048 — and
matches the measured 62 149 / 4 168 = 14.9×.

The Haar QR throughput (`9.9 GFLOP/s`) is ~12× slower than BLIS dense
GEMM (`125 GFLOP/s`) on the same CPU. That's the fundamental reason
the n³ Haar cost doesn't scale-out: the rank-1 sub-matrix update has
a serial dependency chain (each Householder reflection depends on the
previous one's result), and the inner `rank1_householder_update_rows`
is GEMV-shaped, not GEMM-shaped — memory-bandwidth-bound, not
compute-bound.

---

## 6. Cross-check with the paper

### 6.0 What experiment the paper's perf numbers come from

Important framing — the paper has **two** Llama 2 7B experiments and they
should not be conflated:

| paper section | what runs | what is measured |
|---|---|---|
| §4.1 (Table 1) | end-to-end Llama 2 7B, 1000 OpenWebText2 samples, all of Q/K/V/O obfuscated | **functional equality only** — top-1 token equality, MSE on logits |
| §4.2 (Tables 2, 3) | **synthetic random `(n × d)` tensors** between two processes on the same machine | **latency only** — A-gen + Mix + matmul + Un-mix + socket IPC, one offload per measurement |

The "20–30% overhead" headline comes from §4.2 and is **not an
end-to-end Llama 2 7B latency number**. From §4.2 verbatim:

> Our synthetic microbenchmark uses a same-machine logical split to
> model the trusted/untrusted components. We obfuscate and transmit
> random batches between two processes running on different GPUs,
> **rather than running end-to-end LLM inference**.

And §6:

> Integrating GELO into [an inference engine] would require substantial
> software engineering effort beyond the scope of this study. We
> therefore report a controlled microbenchmark that isolates GELO-
> specific costs and their scaling, and **leave full engine integration
> and end-to-end throughput evaluation as future work**.

So the paper measures one (A-gen, Mix, matmul, Un-mix, copy) cycle on
a synthetic `(n × d)` tensor. Layer count, attention compute, FFN,
residuals, KV cache — none of it is in the latency number.

### 6.1 What the paper claims

From Belikov & Fedotov (arXiv 2603.05035), §4.2:

> The results reveal a U-shaped overhead curve:
> - For small batches (n<128), overhead is high (∼29%) because
>   GELO-specific costs (A-generation, mixing) are large relative to
>   the very fast main GEMM.
> - At n ∈ {256, 512}, overhead is minimized (∼20%). Here, the
>   O(n·d²) GEMM dominates, making GELO's costs a smaller fraction of
>   total time.
> - **For large batches (n > 2048), overhead rises as the O(n³) cost
>   of generating the n×n orthogonal matrix A becomes the bottleneck.**

§4.2 latency breakdown at n=512:

> The computational overhead of GELO is A-gen + Mix + Un-mix =
> 2.793 ms, representing the true cost of security, which is modest.
> The majority of time (~81%) in both GELO and the baseline is spent
> on Copy (socket+I/O).

### 6.2 Asymptotic match (per-call)

The paper's asymptotic claims **match what our code implements**, per
single offload call:

| op | paper's claim | our code | match |
|---|---|---|---|
| Mix (A·H) | not stated explicitly; implied O(n²·d) | `cblas_sgemm` M=N=s, K=d → `2·s²·d`, O(n²·d) | ✓ |
| Un-mix (Aᵀ · M) | implied O(n²·d) | `cblas_sgemm` M=s, K=s, N=p → `2·s²·p`, O(n²·d) | ✓ |
| A-generation | O(n³) | Householder QR with rank-1 updates summing to `~10·s³/3` | ✓ |

So the **per-offload** asymptotic analysis is the same. The disagreement
is about **which op dominates wall time**, not about the FLOP scaling
of any single op.

### 6.3 Why our headline bottleneck is mask_apply / mask_unapply, not mask_sample

Paper's "n > 2048 ⇒ Haar QR dominates" prediction is a **per-offload**
statement, derived from the single-offload microbench setup of §4.2.
In that setup the only cost terms are A-gen + Mix + GEMM + Un-mix + IPC
on one `(n × d)` tensor. The crossover at which Haar QR starts to
dominate Mix + Un-mix in that per-offload accounting is:

```
Haar             : (10/3)·n³
Mix+UnMix (1 offload)  : 4·n²·d    (assuming d_in = d_out = d for symmetry)
crossover (per-offload) : n  >  1.2·d
```

For Llama 2 7B (`d = 4096`), the per-offload crossover lands around
`n ≈ 4900`; the paper reports the rising trend already at `n > 2048`,
likely because Mix on confidential GPU saturates above some `n` while
Haar QR keeps growing as O(n³). Either way, **the prediction is
"per single offload"**, not "per forward pass through 32 decoder layers
of Llama 2 7B".

Our setup differs from the per-offload paper microbench in two
implementation choices and one substrate fact:

| dimension | paper per-offload microbench (§4.2) | our end-to-end forward |
|---|---|---|
| ops counted per measurement | 1 offload | 1 forward pass = `7·L = 196` `offload_*` calls |
| Scope of offload sites | Q, K, V, O (paper §6) — FFN listed as future work | Q, K, V, O **+ gate, up, FfnDown** (7 sites/layer) |
| A reuse policy | per offload (most natural read of §3.2 "for a single projection… sample fresh A") | one A per forward pass, **shared across all 196 offloads** in that forward (`sim.rs:450-475` `Session` reuse; per-offload mode is opt-in via `with_per_offload_mask`) |
| Substrate for Mix and Haar | confidential GPU (target deployment) | CPU/BLIS for both Mix (`sgemm_blis`) and Haar (`sample_haar_orthogonal`) |

Our per-forward FLOP balance at `n = 2048`:

```
Haar QR / forward         :    28.9 GFLOPs   (× 1   call — per-forward A reuse)
mask_apply  / forward     : 2 909   GFLOPs   (× 112 calls — 28 layers × 4 sites)
mask_unapply / forward    : 4 849   GFLOPs   (× 196 calls — 28 layers × 7 sites)
total mask GEMM / forward : 7 758   GFLOPs

Haar / total mask FLOP ratio = 28.9 / 7758 = 0.37 %
```

If we ran the paper's protocol exactly — Q/K/V/O only (4 sites/layer)
and per-offload A — at `n = 2048` on Qwen3-1.7B with `L = 28`:

```
Haar QR / forward (paper-scope) : 112 × 28.9 = 3 237 GFLOPs (4 sites × 28 layers, fresh A each)
mask GEMM / forward (paper-scope, no FFN) :
  apply  Σ d_in   per layer = 4·d = 8 192  ⇒  2·s²·8192·L  ≈  1.94 TFLOPs
  unapply Σ d_out per layer = 3·d + 2·c = 8 192 ⇒  ≈ 1.94 TFLOPs
  total ≈ 3.88 TFLOPs
Haar / mask ratio (paper-scope) = 3 237 / 3 880 ≈ 83 %  — comparable, paper's prediction holds
```

So the paper's "Haar is the bottleneck at n>2048" is consistent with a
per-projection, attention-only scope on these shapes. Our reality is:

1. **Per-forward A reuse** instead of per-projection ⇒ Haar QR is paid
   1× instead of `4·L = 112×`. This is **a security choice**
   (sharing one A across multiple H observations in one batch deviates
   from the strict per-projection reading of §3.2 and weakens the
   per-projection BSS argument — flagged in
   `memory/paper_parity_default.md`). Per-offload mode is implemented
   (`with_per_offload_mask`) but defaults off for performance reasons;
   this is the most divergent design choice from the paper.

2. **Adding FFN to the offload scope** (gate, up, FfnDown) brings the
   per-layer mask volume from `2·s²·(4d + 3d+2c) = 2·s²·14d` to
   `2·s²·(3d+f + 3d+2c+2f) = 2·s²·(6d+2c+3f)`. For Qwen3-1.7B this is
   the difference between `2·s²·28 672` (paper-scope) and
   `2·s²·32 768` (ours) — only ~14 % more mask FLOPs from FFN, but
   FFN's `d_in = f = 6144` apply also forces the highest per-call cost
   in the prefill, contributing to wall-clock asymmetry. FFN offload
   is what the paper §6 future work explicitly calls out as not yet
   covered.

3. **CPU vs GPU substrate**: paper's "trusted side" is a confidential
   GPU (H200) where both Mix GEMM and Haar QR run on the same
   accelerator at multi-TFLOP/s. Our trusted side is a CPU running
   AOCL-BLIS at ~125 GFLOP/s for Mix and ~10 GFLOP/s for Haar
   (Householder chain has serial dependencies, maps poorly to BLIS).
   This widens the wall-clock gap between Mix and Haar in our setup
   (12× throughput gap on the *same* CPU), but the dominant effect at
   `n = 2048` is the **per-forward A amortisation** (choice 1
   above) — Haar would still be the minority cost in our deployment
   even on a confidential GPU.

Crossover where Haar would equal total mask GEMM in our deployment
(per-forward A, 7-site offload, Qwen3-1.7B shapes):

```
(10/3)·n³  =  2·L·n²·[(3d+f) + (3d+2c+2f)]
           =  2·L·n²·(6d + 2c + 3f)
n          =  0.6·L·(6d + 2c + 3f)
For Qwen3-1.7B:  n  =  0.6·28·32 768  ≈  550 500 tokens
```

In our setup the per-forward Haar would only become headline-dominant
at `n ≈ 550 k tokens` — three orders of magnitude past any realistic
sequence length. **Under per-offload A (paper-scope) the crossover
moves back to roughly `n ≈ 1.2·d ≈ 2500` tokens**, which is where the
paper's prediction lands.

### 6.4 What the paper does flag that **does** apply to us

The paper's communication / I/O observation is the inverse of our
problem: in their microbench, 81 % of wall time is socket Copy I/O,
which makes the 19 % compute look modest. In our setup there is no
socket — TEE and "offload" share the same process — but the substrate
asymmetry (CPU mask vs GPU engine) plays the equivalent role of
making the mask side look outsized.

The paper also flags, in §6 Future Work:

> Explore faster constructions for fresh, well-conditioned mixing
> (e.g., structured orthogonal transforms) and system optimizations
> that reduce the communication bottleneck observed in our prototype.

Block-diagonal `A` and HKDF-amortised Haar QR (filed in
`docs/plans/m1-10-fused-permuted-attention.md` §10 and
`docs/dev/prototype/future-rnd.md` §5) are exactly the structured-transform
direction the paper anticipates.

---

## 7. Implications for optimisation

| target | what it would shave | how much of the 90 % mask wall does it touch? | gating |
|---|---|---|---|
| Confidential-GPU threat model | move `mask_apply` / `mask_unapply` / Haar QR onto GPU (paper's intended deployment) | all of it — single biggest lever | threat-model change, not a code change |
| Block-diagonal `A` (B blocks) | `O(n²·d)` → `O((n/B)·n·d)`, factor B speedup on Mix/Un-mix | up to B × on the 90 % | security spike on cross-block leakage; filed `future-rnd.md` §5 |
| HKDF-amortised per-step Haar at decode | eliminates the `mask_sample` cost at every decode step (1 QR/step → 1 QR/session) | 4 % of prefill, ~0 % of decode (mask_sample is already tiny at n=1) | freshness-argument write-up; filed `m1-10-*.md` §10 |
| Multi-thread BLIS for very large mask GEMMs | possibly +1.5× on Mix/Un-mix once each call individually saturates a single core | up to 30 % of the 90 % | needs auto-switch threshold; current single-thread pin is right at smaller shapes |
| K/V permutation copy rayon, SIMD Gaussian (queued #69) | reduces the `gpu_gelo_permuted` delta (the extra ~15 % at n=2048) | does **not** touch the 90 % mask wall | already on tasks list |

Without a threat-model change, the long-context regime cannot be made
cheap: the mask GEMM substrate gap (CPU/BLIS vs GPU) is structural, and
block-diagonal `A` is the only privacy-compatible lever that survives.

---

## 8. Reproducibility

```bash
# Per-bucket sweep used to populate §3, §4, §5:
GELO_BENCH_LENGTHS="256,512,1024,2048" GELO_BENCH_MAX_TOKENS=4 \
    cargo test -p gelo-gpu-wgpu --test qwen3_long_context_bench \
    --release -- --ignored --nocapture
```

Env vars added in 2026-05-19 instrumentation pass; default lengths are
`{64, 512, 2048}` and default `MAX_TOKENS = 16` if unset. The bench
calls `profile::reset()` / `profile::snapshot()` around the prefill and
around the decode-step loop, then dumps every populated bucket per
`(cell, n)` pair. Bench source:
`crates/gelo-gpu-wgpu/tests/qwen3_long_context_bench.rs`.
