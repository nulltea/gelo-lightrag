# Handoff — bf16 mask GEMM: blocker analysis, path comparison, HD₃ interaction

> **Subject:** Why we explored bf16 on the GELO mask GEMM, what we
> found about library availability, what the four implementation
> paths would cost, and why HD₃ Hadamard cascade subsumes most of the
> bf16 gain. This doc closes out the bf16 question so the next agent
> doesn't relitigate.
>
> **Reference artifacts** (read these first, don't re-derive):
> - `docs/research/private-llm-inference-round-3.md` — full research
>   round including the measured outcomes section that landed steps
>   1-3 of the perf plan (§7). bf16 is step 2 there.
> - `docs/prototype/gelo-complexity-analysis.md` — the bottleneck
>   breakdown that motivates this whole round.
> - `crates/gelo-protocol/tests/bf16_mask_parity.rs` — the parity
>   simulation test that's checked in (regression anchor).
> - `memory/bf16_mask_gemm_skipped.md` — short-form summary of this
>   handoff in agent memory.
> - `memory/private_llm_inference_round_3.md` — the round-3 doc's
>   memory index entry.
> - Commit range: the round-3 doc, BLIS-default-on flip, layer-skip
>   experiment + revert, and bf16 parity test are all 2026-05-19
>   work and not yet committed at this writing — check `git log` and
>   `git status` before assuming anything has shipped.

---

## 1. Why we asked the question

After step 1 of the round-3 plan landed (multi-thread BLIS, 5.04×
speedup at n=2048 prefill, TTFT 73 s → 14.5 s), the bottleneck
re-decomposes:

```
22.8 %  gelo:mask_sample        (Haar QR, single-thread O(s³))
18.6 %  engine:matmul_many       (GPU)
16.3 %  gelo:mask_unapply        (BLIS-mt-16 GEMM)
13.1 %  engine:matmul            (GPU)
 9.8 %  gelo:mask_apply          (BLIS-mt-16 GEMM)
 9.5 %  tee:attn_cached          (in-TEE GQA)
 5.8 %  gelo:strip_shield        (memcpy)
remainder (small ops)
```

The §7 plan listed bf16 mask GEMM as step 2 because AVX-512_BF16
gives 2× FLOPs/cycle vs f32 on `gelo:mask_apply` / `gelo:mask_unapply`
— a clean 1-2 day win if a library exposes the kernel. The question
this handoff closes: is that 1-2 day win actually available?

## 2. The blocker — current vendored AOCL has no bf16 kernel

`vendor/aocl-install/lib/libblis-mt.so.5.2.2` is the "pure BLAS"
build of AOCL-BLIS. Symbol scan:

```bash
nm -D vendor/aocl-install/lib/libblis-mt.so.5.2.2 | wc -l   # 3168 T symbols
nm -D vendor/aocl-install/lib/libblis-mt.so.5.2.2 \
  | grep -iE 'lpgemm|bf16|aocl_gemm'                         # zero matches
nm -D vendor/aocl-install/lib/libblis-mt.so.5.2.2 \
  | grep -i 'bli_cpuid_is_avx512bf16_supported'              # present (helper only)
```

So the library can *detect* AVX-512_BF16 at runtime but has no
kernels behind it. The bf16 path exists in the AMD ecosystem in two
forms we don't currently have:

- **AOCL-BLIS 4.0+ built with `-DENABLE_AOCL_KERNELS=lpgemm`** —
  exposes `aocl_gemm_bf16bf16f32of32`. Our 5.2.2 was not built with
  that flag; rebuilding from source is multi-day surgery.
- **AOCL-DLP** (separate library `libaocldlp.so`, released Jan 2026,
  [github.com/amd/aocl-dlp](https://github.com/amd/aocl-dlp)) — same
  entry point. Not in our `vendor/aocl-install/lib/`.

A parity sim is checked in at
`crates/gelo-protocol/tests/bf16_mask_parity.rs`. Results
(scale-invariant, 7-bit mantissa floor): **2.65 × 10⁻³ mean relative
error vs f32 target** at all Qwen3-1.7B shapes;
**1.87 × 10⁻³ vs a bf16-everywhere target**. Both sit inside the
paper's measured bf16 model band (≥98.8 % top-1 token equality, paper
Table 1). bf16 mask wouldn't break correctness — it's purely an
implementation question.

## 3. Path comparison — researched options

From the AOCL-bf16 survey agent run on 2026-05-19
(also captured in `docs/research/private-llm-inference-round-3.md`
"Measured outcomes" §"Step 2 (bf16)"):

| approach | effort | expected gain on mask GEMM | risk | references |
|---|---|---|---|---|
| **OpenBLAS `cblas_sbgemm`** (BSD-3, stable since 0.3.13, Zen5 dispatches to COOPERLAKE / AVX512_BF16) | **~1 day** (swap `blis-src` → `openblas-src`, route mask via `cblas_sbgemm`, build with `BUILD_BFLOAT16=1 DYNAMIC_ARCH=1`) | **1.6–1.8×** on the two mask GEMMs | low | [discussion #5205](https://github.com/OpenMathLib/OpenBLAS/discussions/5205) (Zen5 dispatch), [issue #4558](https://github.com/OpenMathLib/OpenBLAS/issues/4558) (sbgemm history) |
| AOCL-DLP (separate `libaocldlp.so`) | 1 day if AMD ships prebuilt, else 1 week | ≥ OpenBLAS (AMD-tuned for Zen4/Zen5) | medium | [AMD AOCL-DLP](https://www.amd.com/en/developer/aocl/dlp.html), [github.com/amd/aocl-dlp](https://github.com/amd/aocl-dlp), [LPGEMM design slides](https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Bhaskar_BLIS_Retreat_2024_AMD_LPGEMM_0.pdf) |
| Rebuild our vendored AOCL with `-DENABLE_AOCL_KERNELS=lpgemm` | 1 week | same as DLP | medium (build-system surgery) | [github.com/amd/blis](https://github.com/amd/blis) |
| Hand-rolled AVX-512_BF16 microkernel | 1-2 weeks | matches OpenBLAS-threaded | medium | ~600-900 lines using `_mm512_dpbf16_ps`. References: [libxsmm issue #877](https://github.com/libxsmm/libxsmm/issues/877), BLIS sandybridge 6×16 template, [VDPBF16PS spec](https://www.felixcloutier.com/x86/vdpbf16ps), [WikiChip AVX512_BF16](https://en.wikichip.org/wiki/x86/avx512_bf16) |
| Intel MKL `cblas_gemm_bf16bf16f32` | 1 day | matches OpenBLAS in best case | **high on AMD** — notorious cpuid downclocking; needs `MKL_DEBUG_CPU_TYPE=5` workaround | [oneMKL 2025.2 reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/overview.html) |
| Hand-rolled via `gemm` / `faer` / candle CPU | multi-week | no native AVX512_BF16 microkernel in any of these | high (we'd be writing the kernel anyway) | [docs.rs/faer](https://docs.rs/faer/latest/faer/), [candle issue #2805](https://github.com/huggingface/candle/issues/2805) |

If we ever change our mind, **OpenBLAS `cblas_sbgemm` is the
correct path**: 1-day effort, production-quality, no AMD-specific
risk. Hand-rolling is in-scope (well-documented templates exist) but
costs 10× more for no throughput gain over OpenBLAS-threaded.

## 4. What we leave on the table

Quantified at n=2048 prefill (post-step-1, BLIS-mt-16 baseline):

| metric | f32 mask (current) | bf16-on-the-mask | delta |
|---|---:|---:|---:|
| Peak GEMM throughput on Strix Halo | 1.25 TFLOP/s | ~2.0–2.3 TFLOP/s (1.6–1.8× from VDPBF16PS, derated for memory bandwidth) | +0.8–1.0 TFLOP/s |
| `gelo:mask_apply` + `gelo:mask_unapply` | 2 887 ms | ~1 604 ms | **−1.3 s** |
| `gelo:mask_sample` (Haar QR) | 3 089 ms | 3 089 ms (unchanged — Householder is scalar, not GEMM) | 0 |
| TTFT at n=2048 | 14.5 s | ~13.0 s | **−10 %** |

So bf16 saves about **10 % TTFT** — the smallest of the remaining
levers in the round-3 plan, and it doesn't touch the new top hotspot
(`mask_sample`). That's the gain on the table.

## 5. Interaction with HD₃ Hadamard cascade (why we deferred)

HD₃ Hadamard cascade (round-3 doc §2.1 — top mask-replacement
candidate, QuIP#/QuaRot primitive) **subsumes bf16's gain**:

- Replaces `mask_sample` Haar QR with a 3·s = 6 168-bit fresh sign
  vector — eliminates the 3.1 s (22.8 %) `mask_sample` cost.
- Replaces dense `(s × s) · (s × d)` GEMM with three FWHTs
  interleaved with sign flips — `O(s · d · log s)` instead of
  `O(s² · d)`, dropping mask-apply/unapply FLOP cost ~25× at our
  shape.
- Preserves κ=1 orthogonality exactly (each Hadamard factor + sign
  flip is exactly orthonormal), so the parity story is *better* than
  bf16's 1.9 × 10⁻³ round-trip error — HD₃ at f32 has the same
  10⁻⁶ round-trip error as dense Haar at f32.

Estimated stack after HD₃ lands:
- TTFT at n=2048: drops from 14.5 s to ~10 s on its own.
- bf16-on-the-mask after HD₃: the mask GEMM is now `O(s · d · log s)`,
  so the absolute time bf16 would save shrinks from ~1.3 s to perhaps
  ~0.3 s. The 10 % relative gain is now closer to ~3 % of TTFT.

**Compounding with weight quantization** is where HD₃ really pays
off: the HD₃ rotation *is* the QuIP#/QuaRot/SpinQuant primitive that
enables Q4 weight quantization without accuracy loss. So adopting
HD₃ for the mask is **simultaneously** the keystone for moving the
GPU side from bf16 to Q4. The compound HD₃ + Q4-weight stack reaches
~5-6 s TTFT (paper-target performance) without ever touching bf16 on
the mask.

By contrast, bf16-on-the-mask is a dead end with respect to weight
quantization — it speeds up the CPU mask side but doesn't unlock
anything else.

## 6. What's checked in (so the next agent can verify)

These landed during the same session as this handoff:

- `crates/gelo-protocol/Cargo.toml` — `default = ["blas"]` (was
  `default = []`). The `blas` feature pulls in BLIS via `blis-src`
  `system` feature; binary now NEEDED-links `libblis-mt.so.5`
  dynamically.
- `crates/gelo-gpu-wgpu/Cargo.toml` — `default = ["blas"]` (was
  absent), `gelo-protocol = { …, default-features = false }` so the
  flag passes through cleanly.
- `crates/gelo-protocol/src/mask.rs:101 blis_init_single_thread` —
  honours `GELO_BLIS_THREADS=N` env var (default 1, safe for
  embedder regime).
- `crates/gelo-protocol/src/mask.rs:138 mask_backend_description()`
  — runtime visibility helper, exported from `lib.rs`; the
  long-context bench prints it at startup so silent-fallback to
  matrixmultiply is immediately visible.
- `crates/gelo-protocol/tests/bf16_mask_parity.rs` — new test file,
  parity simulation, two `#[test]`s (small shape runs by default,
  Qwen3 shapes are `#[ignore]`).
- `crates/gelo-gpu-wgpu/tests/qwen3_long_context_bench.rs` — env
  knobs `GELO_BENCH_LENGTHS`, `GELO_BENCH_MAX_TOKENS`,
  `GELO_BENCH_SKIP_PERMUTED`. The layer-skip env knobs were added
  during the step-3 experiment and **reverted after the regression
  diagnosis** (see `memory/blis_default_on_and_layer_skip_regression.md`).

Verify with:
```bash
git diff --stat
cargo check --workspace
cargo test -p gelo-protocol --test bf16_mask_parity --release  # ~1 s
```

Live bench commands (sanity, ~5-6 min each):
```bash
# Default mask backend (BLIS-mt at the env-var setting)
GELO_BLIS_THREADS=16 GELO_BENCH_LENGTHS=2048 GELO_BENCH_MAX_TOKENS=4 \
    GELO_BENCH_SKIP_PERMUTED=1 \
    cargo test -p gelo-gpu-wgpu --test qwen3_long_context_bench \
    --release -- --ignored --nocapture
```

Should print: `Mask GEMM backend: AOCL-BLIS (cblas_sgemm), threads=16`.

## 7. Decisions captured (don't re-decide)

- **bf16-on-the-mask deferred indefinitely.** Re-evaluate only if
  HD₃ is blocked for non-perf reasons (e.g., the security spike
  comes back with an attack-suite failure against HD₃-with-shield)
  AND we still need the 10 % win. Until then, the engineering budget
  goes to HD₃ + Q4-weight stack.
- **OpenBLAS sbgemm is the chosen path if we ever revisit.** Not
  AOCL-DLP (depends on AMD prebuilt availability), not hand-roll
  (10× more effort), not MKL (AMD CPU downclocking risk). The 1-day
  swap is in `openblas-src` Rust crate territory.
- **HD₃ is now the top single-lever item.** See round-3 doc §2.1 +
  the "Next levers after steps 1-3" subsection added 2026-05-19
  evening.

## 8. Suggested skills for next session

- `diagnose` — if the next agent picks up the HD₃ implementation
  spike and the parity tests fail in unexpected ways, use the
  diagnose skill to systematically reproduce-minimise-instrument the
  divergence. The signal is `crates/gelo-protocol/src/mask.rs`'s
  `mask_round_trip_preserves_matmul` test + the
  `crates/gelo-embedder/tests/generation_harness.rs` end-to-end
  parity.
- `improve-codebase-architecture` — if pursuing the HD₃ + Q4-weights
  compound stack, the integration with `burn-cubecl`'s GPU pipeline
  is an architectural decision worth the deeper analysis the skill
  provides.
- No skill needed for the OpenBLAS-swap fallback if it ever comes
  back to the queue — that's a mechanical Cargo.toml + mask.rs
  change.

## 9. One non-obvious gotcha

The `tee:*_direct` path (used by skipped layers in the layer-skip
experiment, also used at every shape when `offload_layer(li) ==
false`) goes through `ndarray.dot()` → matrixmultiply
single-thread. It does **not** benefit from BLIS-mt-16. We measured
this empirically during step 3 of the round-3 plan: skipping 5 of 28
layers added ~8 s of in-TEE direct matmul cost while saving only
~1.4 s of mask+engine cost — net **+6.6 s regression**.

This is *not* a bf16 issue but worth flagging for the next agent:
**before any layer-skip recommendation can be turned into a perf
claim, the `tee:*_direct` paths in `crates/gelo-embedder/src/decoder/forward.rs`
need to route through `cblas_sgemm` with explicit thread control.**
Touch points: lines 271, 399, 419, 427, 456, 540, 562, 570 (the
`tee:*_direct` profile labels). The fix is small but coupled to the
BLIS-thread-count-by-workload story we already worked out in step 1.
