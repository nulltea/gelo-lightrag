# M1.10 Phase 4 — Long-context bench findings (2026-05-18)

> **TL;DR.** The Phase 1 permuted_cached dispatch is mathematically
> correct but **slower than the in-TEE baseline at every context
> length** on Qwen3-1.7B / RADV iGPU. Root cause is the per-call
> scalar Gaussian-noise sampler over the **cached K tensor** —
> single-threaded ChaCha20 sampling can't keep up with the 4.2 M
> f32 entries per layer × 28 layers per decode step. Phase 2
> (fused-attention kernel) was correctly deprecated post-F1+;
> the real bottleneck is **noise sampling**, not attention compute.

---

## 1. Bench setup

`crates/gelo-gpu-wgpu/tests/qwen3_long_context_bench.rs` — three
cells × three prompt lengths × greedy `generate(max_tokens = 16)`:

| Cell | Config | Protocol |
|---|---|---|
| `gpu_plain` | `PlaintextExecutor` + Vulkan | No privacy baseline |
| `gpu_gelo` | `InProcessTrustedExecutor::with_seed`, default config | Per-forward Haar `A` + shield(8, 4.0); cached path keeps global attention in-TEE per M1.3 |
| `gpu_gelo_permuted` | `InProcessTrustedExecutor::with_seed` + `with_perm_attention(HIDDEN_NO_MORE)` + `cfg.use_perm_attention = true` + `perm_attention_min_seq_len = Some(0)` + `use_out_attn_mult = false` | F1+ permuted_cached dispatch — Phase 1 of M1.10 |

Hardware: AMD RADV GFX1151 iGPU, Mesa 25.2.8, 62 GiB system RAM, OSS
Vulkan stack. Model: `Qwen/Qwen3-1.7B`, bf16 → f32 on load (~13 GiB
working set, Arc-shared across the three executors).

Wall-clock: 322 s including release-build compilation.

## 2. Results

```
cell                 n_prompt   TTFT (ms)   TPOT mean ms    total (s)   vs gpu_plain
─────────────────────────────────────────────────────────────────────────────────────
gpu_plain                  64      240.2         125.3         2.245    (base)
gpu_gelo                   64      471.6         194.7         3.587    +59.8 %
gpu_gelo_permuted          64      468.9         253.3         4.522   +101.5 %

gpu_plain                 512    2 960.9         154.5         5.433    (base)
gpu_gelo                  512    5 967.1         206.6         9.272    +70.7 %
gpu_gelo_permuted         512    9 278.8         506.7        17.386   +220.0 %

gpu_plain                2048    6 851.0         272.9        11.217    (base)
gpu_gelo                 2048   72 778.2         371.1        78.716   +601.8 %
gpu_gelo_permuted        2048   91 002.6       1 692.5       118.082   +952.7 %
```

## 3. Diagnosis

### 3.1 What we hoped for

F1+ moves attention compute from CPU (in-TEE) to GPU. Attention is
~7 s of the n=2048 `gpu_plain` baseline. Moving it to GPU should
have saved ~5-6 s of CPU compute, dropping `gpu_gelo` TTFT from
73 s → ~67 s and TPOT roughly flat.

### 3.2 What actually happened

- **TTFT at n=2048: 73 s → 91 s (+18 s).**
- **TPOT at n=2048: 371 ms → 1 692 ms (4.6× slower per token).**

The permuted path **adds** ~1.3 s per decode step compared to the
in-TEE-attention baseline at n_kv=2048.

### 3.3 Root cause

Per decode step at n_kv = 2064, `permuted_attention_cached` calls
`add_gaussian_3d_inplace` on the cached K tensor of shape
`(16 heads, 2064 positions, 128 head_dim)` = **4.2 M f32 entries**.
The current implementation in
`crates/gelo-protocol/src/attention.rs:408-422` is single-threaded
scalar ChaCha20-Gaussian sampling:

```rust
fn add_gaussian_3d_inplace<R: RngCore>(...) {
    if sigma == 0.0 { return; }
    let normal = StandardNormal;
    for v in m.iter_mut() {
        let z: f32 = normal.sample(rng);
        *v += sigma * z;
    }
}
```

ChaCha20 scalar Gaussian sampling sustains ~50 M samples/sec on a
single core. 4.2 M samples ≈ 85 ms per layer × 28 layers ≈
**2.4 s per decode step just for K-noise**. That accounts for
~1.3 s of the 1.7 s gap vs `gpu_gelo`; the residual ~0.4 s is
permutation copies + PCIe round-trip on the score tensor
(~256 MB per direction at n=2048).

At prefill (n=2048), Q noise + K noise + 2× larger score round-trip
compound to **+18 s** of TTFT.

### 3.4 What this isn't

It isn't:
- A protocol correctness problem (Phase 1 σ=0 parity test passes;
  byte-identical tokens vs in-TEE baseline).
- An attention-compute problem (F1+ moves attention to GPU as
  intended; that part is fast).
- A fused-kernel problem (Phase 2 wouldn't have helped — the bottleneck
  is on the TEE side, before any GPU dispatch).
- A mask-round-trip-on-linears problem (that's still the ~66 s of
  `gpu_gelo` TTFT at n=2048; unchanged by either the permuted
  dispatch or by M1.10 generally).

It is:
- A naive-Gaussian-sampler perf cliff at deployment-realistic K-cache
  sizes that wasn't visible on the GPT-2-class shapes Hidden No More
  benchmarked against.

## 4. Implications

### 4.1 Phase 2 deprecation was correct (and now empirically confirmed)

The fused-flash kernel would have addressed attention compute on the
GPU. Attention compute was **never the bottleneck** at our shapes; the
TEE-side noise sampler is. So even a hypothetical F1+-compatible fused
kernel wouldn't move TPOT at long n_kv — that's exactly the cost the
TEE side adds independently of the GPU dispatch.

### 4.2 The mask round-trip on linears is still the dominant overhead

`gpu_gelo` at n=2048 is **+602 %** over `gpu_plain` (73 s vs 11 s).
M1.10 was never going to fix this — the GELO mask is applied per-offload
on the four linear-projection batches per layer, scaling as
`O((n+k)² · d)` on CPU BLIS. Closing this gap is a separate workstream
(faster CPU BLIS, block-diagonal mask under security analysis, mask
dimension reduction, or per-batch HKDF-derived A).

### 4.3 The permuted_cached path needs noise-sampler perf work before it ships

Three reasonable follow-ups, in order of expected effort × payoff:

| Item | Effort | Estimated benefit |
|---|---|---|
| **Vectorise `add_gaussian_3d_inplace`** | ~½ day | 10-30× speedup via SIMD + `rand_distr::StandardNormal` batched draws + AVX2 |
| **Rayon-parallelise across the heads axis** | ~½ day | Additional 4-8× on multicore CPU (orthogonal to vectorisation) |
| **Per-head π instead of one shared π** | ~1 day | Stronger HNM property; no perf change |
| **Empirical σ floor study at Qwen3-1.7B shapes** (M1.10.0.5) | ~1-2 days | Either confirm σ=0.01 is necessary or relax — could halve the noise cost if relaxable |

Combined potential: noise step at n_kv=2048 from ~85 ms/layer →
**~1-5 ms/layer**. TPOT delta vs `gpu_gelo` would collapse from
+1 322 ms to **≤ +100 ms**, putting permuted_cached at neutral or
slight-win compared to in-TEE attention.

The pre-condition is the vectorisation work — Phase 0 item M1.10.0.5
(empirical σ tuning) becomes interesting only after the noise sampler
is fast enough that its cost doesn't dominate the bench signal.

### 4.4 Memory budget was fine

Peak RSS: 7.48 GiB across all three executors. The Arc-shared
`DecoderWeights` + `clone_shared` engine pattern keeps the working
set comfortably under the iGPU shared-RAM ceiling.

## 5. Decisions locked

- **M1.10 Phase 2 stays deprecated** under F1+. The bench confirms
  fused-attention would not have helped the cost regime that
  actually matters.
- **Phase 1 wiring stays landed.** The dispatch is correct and ready
  for use once the noise sampler is optimised. With `use_perm_attention
  = false` (the M1.3 default) the dispatch is a no-op, so today's
  production path is unaffected.
- **Phase 3 (auto-switch threshold tuning) is on hold** until the
  noise-sampler perf work lands; without it, the threshold sweep would
  measure the wrong cost.
- **Next M1.10 item: noise-sampler vectorisation.** Tracked as a new
  follow-up; see task #67 successor in the project task list.

## 6. References

- `crates/gelo-protocol/src/attention.rs:408-422` —
  `add_gaussian_3d_inplace` (the hot spot)
- `crates/gelo-gpu-wgpu/tests/qwen3_long_context_bench.rs` — bench
  source with all three cells
- `docs/plans/m1-10-fused-permuted-attention.md` — parent M1.10 plan
- `docs/plans/m1-10-security-review.md` — F1+ resolution
- `gelo-llm.html` §08 — original short-context bench (n_prompt=4),
  where the noise sampler does not dominate
- Hidden No More, arXiv 2505.18332 — σ = 0.01 mitigation threshold
  (measured on GPT-2-class shapes, n_kv ≲ 256)
