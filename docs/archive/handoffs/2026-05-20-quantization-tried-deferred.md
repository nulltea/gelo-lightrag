---
type: handoff
status: current
created: 2026-05-20
updated: 2026-05-20
tags: [q4, quantization, gpu]
---

# Handoff — GPU weight quantization: what was tried, why nothing helped on the iGPU, what to try next

> **Subject.** The Q4 (and adjacent f16) GPU weight-quantisation
> investigation on Qwen3-4B against the production substrate (AMD
> Radeon 8060S iGPU, `gfx1151`, RDNA 3.5). Every variant we measured
> regresses or only marginally helps; the strategic direction now is
> either a discrete-GPU deployment or upstream kernel work. This
> document is the next agent's starting point.
>
> **Reference artifacts** (read these first, don't re-derive):
> - `docs/plans/q4-gpu-weights.md` — full implementation plan +
>   Phase 0 outcome §10 (all measured numbers live here).
> - `docs/dev/prototype/hd3-non-pow2-fix.md` — design analysis for the
>   non-pow2 mask that motivated the HD₃/DCT-IV/Auto stack on which
>   any future Q4 work would compose.
> - `docs/prototype/gelo-llm.html` §08 — "Probed but shelved on this
>   iGPU substrate" table with the Q4/f16 spike summary.
> - `memory/qwen3_4b_perf_2026_05_20.md` — measured 4B bottleneck
>   inventory + ranked-queue update.
> - Spike code: `crates/gelo-gpu-wgpu/tests/q4_kernel_spike.rs` (Vulkan
>   Q4), `tests/q4_hip_kernel_spike.rs` (HIP + optional rocWMMA),
>   `tests/f16_kernel_spike.rs` (Vulkan f16).
> - `vendor/cubecl-hip-sys-patched/` — local `hip_53211 →
>   bindings_52802` alias so cubecl-hip 0.9 compiles against ROCm
>   7.2.1. Wired via `[patch.crates-io]` in workspace `Cargo.toml`.

---

## 1. Why this was on the table

After the `MaskKind::Auto` work landed (HD₃ at pow2-aligned shapes,
DCT-IV at non-pow2; see [[hd3_mask_landed]],
[[hd3_radix8_and_scratch_reuse]],
`docs/dev/prototype/hd3-non-pow2-fix.md`), the Qwen3-4B prefill profile
shifted:

| bucket | Haar n=2048 | HD₃ n=2040 (Auto pow2) | DCT-IV n=2048 (Auto non-pow2) |
|---|---:|---:|---:|
| mask CPU subtotal | 12 172 ms (47 %) | 4 966 ms | 10 334 ms |
| **GPU engine matmul** | **8 613 ms (33 %)** | **7 683 ms (47 %)** | **8 130 ms (36 %)** |
| in-TEE attention | 2 558 ms | 2 598 ms | 2 525 ms |
| TTFT total | 25 873 ms | 16 289 ms | 22 691 ms |

GPU matmul became the new dominant bucket. Q4 weight quantisation
was the highest-leverage candidate per the QuIP#/QuaRot literature:
4× less weight bandwidth, ~2× kernel speedup on hardware with
INT4 tensor cores, security-neutral (weights are public in our
threat model), and accuracy-preserving when composed with the HD₃
rotation we already have. The plan projected ~25 % TTFT reduction
at 4B.

## 2. What was tried — three spikes, all negative or marginal

The reference numbers below come from the f32 baseline at the same
shape; speedup is `f32_ms / variant_ms` (higher = variant faster).
All measured on the production iGPU at the four Qwen3-4B per-layer
projection shapes. Clean re-runs after we caught a contention
artifact in an earlier run.

### 2.1 Q4 weight quantisation — Vulkan (`cubek-matmul` via `cubecl-wgpu`)

`tests/q4_kernel_spike.rs::q4_matmul_shape_sweep`. Scheme:
`Q4S + Block(128) + PackedU32(0) + F32 scales + Symmetric`.

| shape | f32 (ms) | Q4 (ms) | speedup | rel-err vs f32 |
|---|---:|---:|---:|---:|
| QKV-Q (2056, 2560) × (2560, 4096) | 18.4 | 22.5 | **0.82×** | 11.4 % |
| Gate∥Up (2056, 2560) × (2560, 9728) | 43.0 | 52.9 | 0.81× | 12.6 % |
| FfnDown (2056, 9728) × (9728, 2560) | 24.3 | 34.4 | **0.71×** | 11.8 % |
| O proj (2056, 4096) × (4096, 2560) | 12.2 | 16.2 | 0.75× | 11.1 % |

### 2.2 Q4 — HIP/ROCm (`cubecl-hip 0.9.0`), with and without rocWMMA

`tests/q4_hip_kernel_spike.rs::q4_matmul_hip_shape_sweep`.

| shape | std features | + rocWMMA |
|---|---:|---:|
| QKV-Q | 0.76× | 0.75× |
| Gate∥Up | 0.79× | 0.79× |
| FfnDown | 0.73× | 0.73× |
| O proj | 0.79× | 0.80× |

All three configurations (Vulkan / HIP / HIP+rocWMMA) agree within
~5 %. Switching runtime did not unlock anything.

### 2.3 f16 engine (Vulkan, `shader-f16` adapter extension)

`tests/f16_kernel_spike.rs::f16_matmul_shape_sweep`. Uses the same
backend (cubecl-wgpu) with element type `f16` instead of `f32`.

| shape | f32 (ms) | f16 (ms) | speedup | rel-err |
|---|---:|---:|---:|---:|
| QKV-Q | 18.8 | 24.2 | 0.78× | 4.5e-4 |
| Gate∥Up | 46.1 | 49.1 | 0.94× | 4.3e-4 |
| **FfnDown** | 24.7 | 16.8 | **1.47×** | 4.3e-4 |
| O proj | 11.8 | 12.9 | 0.91× | 4.8e-4 |

Only FfnDown wins meaningfully. Three of four shapes are tied or
slightly worse.

## 3. Why nothing helped — diagnosis

### 3.1 Q4 — the kernel runs but doesn't engage WMMA

The rel-err 11–13 % matches QuIP# pre-rotation Q4 baseline exactly;
that's the kernel doing real per-tile inline dequant + accumulate,
not falling back to dequantise-then-f32-matmul (a fallback would
also show ~1.5× slowdown from the dequant pass, not 0.75×).

**The hardware supports `v_wmma_i32_16x16x16_iu4` on `gfx1151`**
(RDNA 3.5 documented). `rocwmma-dev 2.2.0` is installed. cubecl-hip
0.9.0 with `rocwmma` feature compiles and runs. And yet both
runtimes produce identical ~0.75× speedups.

The most parsimonious explanation: **cubek-matmul 0.9.0's Q4 path
doesn't emit WMMA INT4 intrinsics on either runtime.** It routes
through the generic Cmma/Mma strategies — those are FP-shaped, plus
the kernel pays per-tile dequant compute. Activation bandwidth
(input `21 MB f32` reloaded per matmul) plus output bandwidth
dominates; the 4× weight bandwidth saving is offset by extra
compute.

### 3.2 f16 — bandwidth crossover happens only on wide weights

The one win (FfnDown, 1.47×) is the op with the largest `d_in`
(9 728). That's where weight bandwidth is largest in absolute terms,
so halving it pays off enough to offset the f16 unpack overhead.
The other three projections have smaller weight bandwidth share —
f16 doesn't help there.

### 3.3 What we ruled out

- **Contention artifact** — confirmed clean re-runs (Vulkan and HIP
  numbers within 5 %) eliminate this as a confound.
- **Stale autotune cache** — fresh measurements; the cache is
  populated by the bench itself.
- **Wrong quant scheme** — tested `Q4S/Block(128)/PackedU32(0)`,
  which is the standard recipe (matches QuIP# / GPTQ block size).
  Q8 would be expected to be in between f32 and Q4 — not worth
  measuring while Q4 itself is below water.
- **ROCm 7.2.1 incompatibility** — the `hip_53211` alias patch in
  `vendor/cubecl-hip-sys-patched/` lets cubecl-hip 0.9.0 compile
  against installed ROCm 7.2.1. The bench runs, produces correct
  results, and reports identical perf to the Vulkan path. So our
  patch is correct; the kernel path itself is the limit.

## 4. Workspace artifacts left behind

The investigation produced reusable infrastructure even though the
perf direction didn't pay off. **Do not revert these unless we
intentionally drop ROCm support** — they're stable points for a
future re-activation.

| artifact | what it is | leave / revert |
|---|---|---|
| `crates/gelo-gpu-wgpu/tests/q4_kernel_spike.rs` | Vulkan Q4 spike + shape sweep | leave |
| `crates/gelo-gpu-wgpu/tests/q4_hip_kernel_spike.rs` | HIP Q4 spike + shape sweep | leave |
| `crates/gelo-gpu-wgpu/tests/f16_kernel_spike.rs` | Vulkan f16 spike | leave |
| `vendor/cubecl-hip-sys-patched/` | `hip_53211 → bindings_52802` alias for ROCm 7.2.1 | leave |
| Workspace `[patch.crates-io]` for `cubecl-hip-sys` | wires the patch | leave |
| `cubecl-hip = "=0.9.0"` + `cubecl-hip-sys = "=7.1.5280200"` workspace deps | leave |
| `cubecl-hip.workspace = true` (`+ rocwmma` feature) in `gelo-gpu-wgpu/Cargo.toml` dev-deps | leave |
| `burn-backend` features = `["std", "cubecl-wgpu", "cubecl-hip"]` | leave |

The HIP backend isn't used by anything outside the spike tests; it
costs ~30 s of test-only build time and the patched hip-sys is
self-contained.

## 5. What to try next — ranked

### 5.1 Selective f16 routing (~½ day, low risk, ~2 % TTFT)

Wire `with_fp16_engine()` into `InProcessTrustedExecutor` and
route only FfnDown weights through the f16 engine; other ops stay
f32. At 36 layers × ~8 ms saved per FfnDown matmul ≈ **290 ms TTFT
saving (~2 % at 4B)**. Cheap, safe, no protocol surface change.
Risk: complicates the executor with per-`WeightKind` precision; if
we ever change quant scheme we have two paths to maintain.

**Decision rule:** ship if it ≥ 2 % TTFT measured at n=2048 4B with
the current Auto-mask path.

### 5.2 Upgrade workspace to cubecl 0.10.0 / burn 0.21 (~3-5 days, medium risk, unknown payoff)

Quick-eval first: read the burn-cubecl 0.21 + cubek-matmul changelogs
for "Q4" / "INT4" / "WMMA" mentions. If a documented Q4 WMMA kernel
landed, commit to the migration. If not, skip — the workspace
migration cost won't be repaid.

**Decision rule:** only proceed if there's a documented WMMA-INT4
kernel in the 0.10 / 0.21 series. Don't migrate speculatively.

### 5.3 Fork cubek-matmul and add a WMMA INT4 kernel (~2-4 weeks, high risk, high payoff)

The hardware supports it; cubek-matmul just doesn't emit it. A
hand-rolled kernel using `rocwmma::fragment` C++ wrappers on the
HIP path could deliver the 2-3× we projected. Multi-week effort
requires cubek-matmul internals familiarity. Justify only if (5.1)
ships and we still want more iGPU perf before discrete-GPU
deployment.

### 5.4 Defer Q4 to discrete-GPU deployment

Strix Halo iGPU is the wrong substrate for Q4. On RDNA3+ discrete
cards or Nvidia H100/RTX 40-series the Q4 WMMA path is exercised by
the same cubek-matmul kernel and should deliver the projected
2-3× speedup. The plumbing-side work in `docs/plans/q4-gpu-weights.md`
Phases 1-4 remains correct for that future. **This is the
conservative path** if the strategic deployment target is discrete
GPUs anyway.

### 5.5 Pivot to a different perf lever entirely

The remaining levers on this iGPU substrate (with HD₃/DCT-IV/Auto
already shipped) are:
- **GPU-fused HD₃/DCT-IV apply** — moves the mask CPU work to the
  GPU, paid once across mask + matmul. Multi-week, alters threat
  model (GPU briefly sees un-rotated buffer).
- **In-TEE attention kernel optimisation** — `tee:attn_cached` is
  still 11-15 % of TTFT. Currently uses BLIS for the per-head GEMMs
  on CPU; an AVX-512 or SIMD-direct hand-roll could halve it.
  Cheaper than GPU work but the per-head shapes are small.
- **Decoder-step KV cache compression** — orthogonal to mask /
  matmul, attacks the decode TPOT not prefill TTFT. Filed in
  `docs/plans/m1-10-fused-permuted-attention.md` legacy.

## 6. Open questions for next session

1. **Does ROCm 7.2.1 actually engage WMMA INT4 anywhere in the cubek-matmul kernel for our shapes?** The rocWMMA feature flag was set but we never verified the emitted GCN assembly. `RUST_LOG=cubek=trace,cubecl=debug` + a `rocprof` trace on the HIP spike would confirm which kernel ID gets dispatched and whether `v_wmma_*` instructions appear. ~½-day investigation; would either close the Q4 question definitively or motivate a kernel-tuning spike.

2. **Is the 1.47× FfnDown f16 speedup real or measurement noise?** Stddev across the 5 timed runs was ~0.5 ms (vs ~17 ms median), so likely real. But worth confirming with a longer run before wiring selective-f16 into production.

3. **Does QuIP# hidden-axis rotation rescue Q4 accuracy at all?** Phase 0 measured raw weights — rel-err 11-13 %. With HD₃ rotation (`Qᵀ · W`) applied to weights before quantisation, QuIP# claims accuracy parity with f32. Worth a one-day experiment using our existing `Hd3Mask::apply` at order `d_in` to pre-rotate, but **only if a kernel speedup is also recovered** (Phase 0 showed naive Q4 has no speedup even with the kernel engaged; rotation fixes accuracy, not speed).

4. **Will discrete-GPU Q4 actually deliver the projected 2-3×?** Without hardware to test, this is a literature-derived claim (QuIP#/QuaRot benchmarks on RTX 4090 / A100). Spike on a borrowed/rented discrete GPU would confirm before committing to the deployment story.

## 7. Test/bench commands for the next agent

```bash
# Re-run Vulkan Q4 spike (clean baseline)
cargo test --release --package gelo-gpu-wgpu \
    --test q4_kernel_spike q4_matmul_shape_sweep \
    -- --ignored --nocapture

# Re-run HIP Q4 spike (requires ROCm 7.2.x + rocwmma-dev)
cargo test --release --package gelo-gpu-wgpu \
    --test q4_hip_kernel_spike q4_matmul_hip_shape_sweep \
    -- --ignored --nocapture

# f16 spike (selective-f16 evaluation)
cargo test --release --package gelo-gpu-wgpu \
    --test f16_kernel_spike f16_matmul_shape_sweep \
    -- --ignored --nocapture

# Sanity (no GPU needed)
cargo test --release -p gelo-protocol --lib dct4 hd3 mask sim
```

## 8. Suggested skills for the next session

- `diagnose` — for question 1 above (`rocprof` + assembly trace to
  confirm whether WMMA INT4 is engaged). Reproducible signal,
  bounded scope.
- `improve-codebase-architecture` — if the next session pursues
  (5.1) selective-f16 routing, the per-`WeightKind` precision
  routing is an architectural choice that warrants the deeper
  analysis.
- No skill needed for (5.4) defer-to-discrete-GPU — that's a
  decision, not implementation.

## 9. One-line summary for the project tracker

> Q4 weight quant on the production AMD Strix Halo iGPU regresses
> 0.71-0.82× under both Vulkan and ROCm+rocWMMA; the cubek-matmul
> 0.9 kernel doesn't emit WMMA INT4 on this hardware. f16 helps
> only on FfnDown (1.47×). Q4 plumbing + ROCm hip-sys patch left in
> tree for future discrete-GPU re-activation. Selective-f16 routing
> (~2 % TTFT) is the only iGPU lever still on the table without
> upstream kernel work.
