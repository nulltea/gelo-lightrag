# Handoff — 2026-05-21 — `tee:attn_cached` GPU-offload spike + cubek-attention findings + batched-decode next

This session investigated whether the `tee:attn_cached` bucket (39 % of
wall, 89.7 s on the v7 fixture) can be eliminated by moving attention
to the GPU. **Conclusion: not on this hardware at decode m=1 with any
of the kernel options we tried.** The bottleneck is GPU dispatch
overhead at m=1, not compute. Path forward is batched decode, which
restructures the workload into a regime where GPU offload actually
wins.

The session also landed two real perf wins on the shield path that
got committed cleanly (see commits below).

## What's on `master` (committed)

Five commits past the session start; the first three (shield-related)
are clean wins, the last two (handoff doc updates) are documentation:

| Commit | Title |
|---|---|
| `144d764` | perf(gelo-protocol): SIMD Box-Muller shield-row generator (-37% shield_stack, -5.4% wall) |
| `3eca59e` | perf(gelo-protocol): polar Box-Muller + Xoshiro256++ shield-RNG (2.42× per-call) |
| `bc47d04` | docs(handoff): append §A polar+Xoshiro follow-up + post-3eca59e profile |

Cumulative since v7: `gelo:shield_stack` 486 → 158 µs/call = **3.08×**;
bucket 35.9 s → 11.67 s (−67 %); rank #4 → #6.

See memory `shield_simd_gaussian_landed.md` for the full per-stage
breakdown; see the prior handoff
`docs/handoffs/2026-05-21-gelo-perf-shield-attn-batched.md` (now with a
post-3eca59e §A follow-up section).

## What's uncommitted on `master` (the attention-offload work)

**All gated behind `PHASE_1B_DECODE_AMULET=1` env var. Default off.**

| File | What |
|---|---|
| `Cargo.toml` | + `cubek-attention = "=0.1.1"` workspace dep |
| `crates/gelo-gpu-wgpu/Cargo.toml` | + `criterion` dev-dep; + `cubek-attention` dep; `[[bench]] amulet_attention` |
| `crates/gelo-gpu-wgpu/benches/amulet_attention.rs` | **NEW** — microbench: `in_tee` vs `perm_softmax_tee` vs `perm_softmax_gpu` at decode shapes (n_q=1, n_kv ∈ {256, 1000, 2000}) |
| `crates/gelo-gpu-wgpu/tests/cubek_attention_spike.rs` | **NEW** — isolated `cubek_attention::launch::launch()` spike with warm-up + steady-state timing at decode / prefill / prefill_long shapes |
| `crates/gelo-gpu-wgpu/src/lib.rs` | **A1** — `fused_attention_batched` override via burn-tensor chain (matmul → mul_scalar → add(?) → softmax → matmul on GPU; one CPU↔GPU round-trip). **A2** — accepts `Option<mask>`; skip upload + add when None |
| `crates/gelo-protocol/src/substrate.rs` | Trait change: `fused_attention_batched(mask: Option<ArrayView3>)`. Default impl + test updated. New `default_impl_none_mask_equivalent_to_zero_mask` test. |
| `crates/gelo-protocol/src/attention.rs` | `PermAttnConfig.decode_softmax_on_gpu: bool` + `HIDDEN_NO_MORE_DECODE_GPU` const. `permuted_attention_cached` branches: Phase 1b ⇒ `engine.fused_attention_batched(... None)`; legacy ⇒ matmul+TEE-softmax+matmul (F1+) |
| `crates/gelo-protocol/tests/permutation_attention.rs` | `phase_1b_decode_softmax_on_gpu_matches_in_tee` + `phase_1b_prefill_falls_back_to_in_tee_softmax` — both pass to f32 floor |
| `crates/gelo-snp-runner/src/extraction.rs` | `PHASE_1B_DECODE_AMULET=1` env var ⇒ `cfg.use_perm_attention = true; perm_attention_min_seq_len = Some(1)` + `with_perm_attention(HIDDEN_NO_MORE_DECODE_GPU)` |

107 unit tests pass across `gelo-protocol` (2 new). All parity tests
hold to f32 floor between in-TEE and GPU paths at σ=0.

## Microbench results (post-A1+A2)

`cargo bench -p gelo-gpu-wgpu --bench amulet_attention`:

| Shape | in_tee | perm_softmax_tee | perm_softmax_gpu (A1+A2) |
|---|---:|---:|---:|
| n_kv=256 | 0.66 ms | 4.35 ms | 3.98 ms |
| n_kv=1000 | 2.08 ms | 24.49 ms | 22.24 ms |
| n_kv=2000 | 4.28 ms | 48.47 ms | 43.92 ms |

A1 closed ~30 % of the gap vs the prior 3-explicit-dispatch path
(burn-cubecl-fusion is firing on the chain). **A2 was perf-neutral**
(burn-cubecl-fusion was already absorbing the mask-add into adjacent
kernels). **Decode-m=1 is still 10× slower than in-TEE.**

## E2E bench (Phase 1b enabled at decode)

`PHASE_1B_DECODE_AMULET=1 BENCH_MAX_CHUNKS=1 cargo run ... extract_and_query_bench`:

| Metric | Baseline (`bc47d04`) | Phase 1b enabled |
|---|---:|---:|
| `tee:attn_cached` | 89.7 s (39 %) | 0 |
| `tee:attn_permuted_cached` | 0 | **629.8 s (45 %)** |
| `gelo:perm_attention_cached` | 0 | **593.7 s (42 %)** |
| Per-call attention | 4.85 ms | **32 ms** (6.6× slower) |
| Generate wall | 343 s | **903 s** (**+163 %**) |

Phase 1b at decode is a **2.6× wall-time regression**. Confirmed
across two runs (one with the original 3-explicit-dispatch
implementation, one with the A1 fused-API switch — both ~903 s).

## cubek-attention spike — separate eval

Ran `cubek_attention::launch::launch()` directly via cubecl wgpu
runtime, bypassing burn-tensor, with three shape variants:

| Shape | n_q | n_kv | cubek (Unit) steady-state | accuracy |
|---|---:|---:|---:|---|
| **decode** | 1 | 1000 | **17.9 ms** | max_abs 3.3e-3 ✓ |
| **prefill** | 64 | 64 | **1.24 ms** | max_abs 1.1e-2 ✓ |
| **prefill_long** | 745 | 745 | **15.1 ms** | max_abs 2e-6 ✓ |

JIT compile cost ~4.8 s on first launch; one-time, amortised per
process.

**`Strategy::BlackboxAccelerated` segfaults on Radeon RDNA3.5**
(`CUBEK_STRATEGY=blackbox` env var, SIGSEGV via cubecl-wgpu). Requires
NVIDIA-style cooperative matmul that doesn't exist on this GPU via
Vulkan. **`Strategy::Unit` is our only cubek-attention path.**

### Key finding

The Unit routine's stage-parallelism size is `plane_dim ≈ 32-64`. At
`n_q = 64` it fills exactly one stage; at `n_q = 1` it wastes 31-63
lanes per stage, **paying full kernel launch overhead for a
single-lane query**. This is architectural — the Unit kernel design
assumes `seq_q ≥ stage_size`. At decode m=1, cubek-attention is
essentially the same speed as the burn-tensor chain (~18 ms) and
falls into the same 9× gap vs in-TEE that all other GPU paths hit.

**Conclusion**: on Strix Halo / Radeon 8060S via wgpu/Vulkan, **no
current GPU dispatch strategy beats in-TEE attention at decode m=1**.
The launch-latency floor is ~10-20 ms per call regardless of kernel
choice; in-TEE GEMV runs in 2 ms.

cubek-attention IS spectacular at prefill (n_q ≥ 32) — **1.24 ms at
n_q=64** beats in-TEE by ~3×, and prefill_long at 15 ms is also very
fast given the workload.

## What the next session inherits

**Substrate-level state:**

- `GpuOffloadEngine::fused_attention_batched` trait now takes
  `mask: Option<ArrayView3>` instead of `ArrayView3`. Default impl in
  `substrate.rs:237`. WgpuVulkanEngine override at `lib.rs:535+`
  using `burn::Tensor::matmul` chain.
- `PermAttnConfig.decode_softmax_on_gpu: bool` defaults `false`. Three
  pre-baked variants: `DISABLED_NOISE`, `HIDDEN_NO_MORE`,
  `HIDDEN_NO_MORE_DECODE_GPU`.
- `permuted_attention_cached` branches on
  `decode_softmax_on_gpu && mask_is_noop` ⇒ goes through
  `fused_attention_batched(..., None)`. Otherwise the legacy
  matmul + TEE-softmax + matmul chain (F1+ preserved).
- `DecoderRuntime::from_config_and_dir` honours `PHASE_1B_DECODE_AMULET=1`
  to enable perm-attention at every n_q + GPU softmax at decode.
- New microbench `crates/gelo-gpu-wgpu/benches/amulet_attention.rs` and
  spike test `crates/gelo-gpu-wgpu/tests/cubek_attention_spike.rs`
  (gated `#[ignore]` for opt-in).

**Conclusions to act on:**

1. The Phase 1b code path is **correct but not deployable** at decode
   on this hardware. Don't flip the env-var default.
2. cubek-attention is **infrastructure for batched decode**, not a
   decode-m=1 fix.
3. The 89 s `tee:attn_cached` bucket cannot be addressed by GPU
   offload alone — needs batching first.

## Next-session focus: batched decode (handoff §C of the prior handoff)

Scoped in `docs/handoffs/2026-05-21-gelo-perf-shield-attn-batched.md`
§C. Estimated 3-5 days. Key changes:

1. Convert the KV cache layout in
   `gelo-embedder/src/decoder/generation.rs` to
   `(B, layers, max_cache_len, kv_dim)`.
2. Change `run_decode_step` to accept `&[u32]` (one token per
   sequence) and return `Array2<f32>` (B × hidden).
3. Mask at `stacked_n = B + shield_k` instead of `1 + shield_k` —
   amortises GELO mask/shield cost across B sequences.
4. Auto-resolves to HD₃ much more readily once `B ≥ 7`
   (stacked_n = B+8 ≥ 15, pad to 16, ratio ≤ 1.07).
5. Sampling loop diverges per sequence — early-stop on EOS per
   sequence; others continue. Standard right-padding-with-EOS trick.

**Why batched decode unlocks Phase 1b**: at B=64 sequences, the
attention block sees Q with shape (B·num_heads, 1, d_head) which (per
the cubek-attention prefill data) drops the per-call cost from 18 ms
to ~1-2 ms — **finally competitive with in-TEE** which scales
linearly with B from 2 ms × B = 128 ms.

Crossover B for cubek-attention vs in-TEE at decode is probably
B ≈ 8-16; needs measurement.

**Where it'd apply**:

- **Rerank workloads** — `CausalDiscriminatorRerankService` already
  has N (query, candidate) pairs per call. Direct fit.
- **Inference serving** — multiple user queries concurrent. Direct
  fit.
- **Single-stream extraction** (the v7 fixture) — only benefits if
  we can batch across chunks (prefills + decodes of independent
  chunks interleaved). Non-trivial scheduler change.

The extract_and_query_bench fixture won't benefit much without the
cross-chunk scheduler. But rerank perf could be transformed (~14×
batch already noted in `gelo-reranker` round 2 work). The right place
to land batched decode first is the rerank service.

## Decision tree for the next session

```
batched_decode_pivot
├── target = rerank service (recommended) — biggest existing user of N>1 workloads
│   └── reuse PHASE_1B_DECODE_AMULET wiring; batch dim absorbs to n_q in cubek
│
├── target = single-stream extraction
│   └── needs cross-chunk scheduler; ~1-2 weeks; lower priority
│
└── decision: defer GPU attention offload for single-stream decode entirely
    └── shield_stack (3.08×) was the right win; tee:attn_cached stays for now
```

The third option is the honest read of the data. If the user wants to
prioritise *headline single-stream wall*, the next perf lever is the
GPU buckets (`engine:matmul`, `engine:matmul_many` — combined 87 s),
not attention. Suggested approach there: bf16-native weight provision
with Q4 quantisation (handoff comments mention this as the long-term
target).

## Open follow-ups

- `c5_perm_attn` AloePri attack-suite condition is still owed for the
  P3 Xoshiro shield RNG (memory `shield_simd_gaussian_landed.md`
  flags this) — independent of the Phase 1b work.
- `target-cpu=native` workspace-wide — extends every SIMD codepath.
  Schedule as a deliberate experiment so attribution per crate stays
  clean.
- AVX-512 hand-rolled inner loop for shield Gaussian — `wide` only
  emits AVX2. Zen 5 has AVX-512F. Estimated ~1.5-2× on the SIMD body.
  Diminishing returns vs the remaining 5 % share.

## Suggested skills for the next session

- **`diagnose`** — for the batched-decode KV cache layout change.
- **`grill-with-docs`** — before flipping the rerank service to
  batched decode, the existing `gelo-reranker` round 2 design (memory
  `private_reranking_round_2.md`) should be cross-walked against the
  batched-decode requirements.
- **`verify`** — for any rerank-side change, confirm score-export
  correctness on a small fixture.

## Tasks state at end of session

All session tasks closed. Microbench + spike artifacts are tracked
under `crates/gelo-gpu-wgpu/{benches,tests}/`; the integration is
fully reversible via the env-var gate.

## Reproducing the spike

```bash
# Microbench (≤ 30 s)
cargo bench -p gelo-gpu-wgpu --bench amulet_attention

# cubek-attention spike (≤ 30 s, all 3 shapes)
cargo test -p gelo-gpu-wgpu --release --test cubek_attention_spike \
  -- --ignored --nocapture --test-threads=1

# E2E with Phase 1b on (NOT RECOMMENDED — known regression to ~903 s)
PHASE_1B_DECODE_AMULET=1 BENCH_MAX_CHUNKS=1 RUST_LOG=info \
  cargo run -p gelo-snp-runner --release --example extract_and_query_bench

# E2E without Phase 1b (the production path, ~343 s)
BENCH_MAX_CHUNKS=1 RUST_LOG=info \
  cargo run -p gelo-snp-runner --release --example extract_and_query_bench
```
