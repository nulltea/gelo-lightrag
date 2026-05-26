# Q4 GPU weight quantization — implementation plan

> **Status:** 2026-05-20, drafted then **Phase 0 ran with negative
> result on AMD Strix Halo iGPU**. See §10 Phase 0 outcome for
> measured numbers and decision tree. **Do not start Phase 1 on this
> hardware** — the kernel runs correctly but is 0.78–0.95× the speed
> of f32 matmul. Reactivate on discrete-GPU deployment or after a
> successful kernel-tuning spike.
>
> Strategic next perf lever was supposed to be Q4 after `MaskKind::Auto`
> landed (Haar baseline TTFT 25.9 s → Auto 22.7 s at 4B n=2048;
> HD₃ branch 16.6 s at n=2040). GPU matmul is the dominant bucket
> (~42 % of TTFT, ~46 % of TPOT at HD₃ baseline). With Q4 not
> delivering on this iGPU, the next levers to consider are f16
> (existing, already-wired engine path) or a GPU-fused mask kernel.
>
> **Reference artifacts:**
> - `crates/gelo-gpu-wgpu/src/lib.rs` — current f32/f16 `GpuOffloadEngine`
> - `crates/gelo-protocol/src/substrate.rs` — `GpuOffloadEngine` trait + `WeightHandle`
> - `crates/gelo-protocol/src/dct4.rs`, `hd3.rs` — existing fast orthogonal kernels (reused for hidden-axis rotation)
> - `docs/research/hd3-non-pow2-fix.md` §6 — the design that ties HD₃/DCT-IV to Q4
> - `memory/qwen3_4b_perf_2026_05_20.md` — measured bottlenecks
> - `memory/bf16_mask_gemm_skipped.md` — bf16 was ruled out in favour of this Q4 path

---

## Definitions

| symbol / term | meaning |
|---|---|
| Q4 (here) | 4-bit weight quantization with per-block scale (and optional zero-point). Common variants: Q4_0 (symmetric, no zp), Q4_K_M (mixed precision, asymmetric). |
| QuIP# | "Quantization with Incoherence Processing", arXiv:2402.04396 — uses random Hadamard rotation to make weights "incoherent" before Q2/Q4 quantization. |
| QuaRot | arXiv:2404.00456 — production-grade Hadamard-rotation + Q4 weight stack, CUDA kernels open-source. |
| Incoherence | Property that all entries of a matrix are bounded by `O(1/√n)` after random rotation. Required for Q4 weights to preserve accuracy (otherwise outlier weights blow the quant budget). |
| Hidden-axis rotation `Q` | Orthogonal matrix at order `d_in` (the contraction dimension), pre-applied to weights at quantization time and to activations at runtime. **Different from the token-axis mask `A` we already use for privacy.** |
| weight-only quant | Only the weight matrix `W` is quantized; activations stay at f32/bf16. The matmul kernel does inline dequant per tile during GEMM. |

---

## 1. Goals and non-goals

### Goal

Reduce TTFT and TPOT at Qwen3-4B by **~25 % each** by replacing f32
weight storage on the GPU with Q4 weight storage + on-the-fly
dequantization in the matmul kernel. Activations stay at f32, the
mask round-trip is unchanged.

Concretely, projected at n=2040 (HD₃ branch, current TTFT 16.6 s):
- GPU matmul cost drops from 7.7 s → ~3.0–3.5 s (2.2–2.5× from Q4 vs f32 on memory-bound GEMM at our shapes).
- TTFT drops from 16.6 s → ~12.0 s (−27 %).
- TPOT drops from 800 ms → ~580 ms (−27 %).

At n=2048 (DCT-IV branch, TTFT 22.7 s):
- GPU matmul drops from 8.1 s → ~3.5 s.
- TTFT drops 22.7 s → ~18 s (−21 %).
- TPOT comparable improvement.

### Non-goals

- **Activation quantization.** The masked activation `A·X` is the load-bearing privacy primitive; quantizing it would compose with the orthogonal mask in subtle ways and break the round-trip identity at low precision. Out of scope for this phase.
- **Q8 as a separate target.** Q8 is dominated by Q4 at equal engineering cost (cubek-matmul supports both), and HD₃-rotated weights make Q4 accuracy parity with Q8 (per QuIP#/QuaRot results). Ship Q4 directly; Q8 falls out for free if needed as a debug knob.
- **Custom WGSL kernel.** Burn-cubecl 0.20.1's `q_matmul` already routes through cubek-matmul's quantized kernels with inline-dequant tile path. The old M1.10 §3.6 deferral note about "no stable quant matmul on Vulkan" is **stale** — confirmed by reading `burn-cubecl::ops::qtensor::q_matmul` + `cubek-matmul::launch::handle::MatmulInputHandleRef::quantized`.
- **Weight integrity verification under Q4.** The U-Verify probes assume f32 weights for the in-TEE reference check. Under Q4 the engine output isn't bit-equal to a TEE-side f32 reference; probes must be widened or disabled (same caveat as fp16 — see `WgpuVulkanEngine::is_fp16`). This is a known shape of the trade.

---

## 2. Architecture

### 2.1 What gets quantized

Weights `W ∈ R^(d_in × d_out)` for the six per-layer projections:
QKV (one each, sharing input), O, gate∥up (one each), FfnDown. At
Qwen3-4B that's `6 ops × 36 layers = 216 weight tensors`, totalling
~4 B params × 2 bytes (bf16 native) → **~8 GB f32 on the GPU today**
→ **~1 GB Q4 on the GPU** (4× compression).

Memory bandwidth on the iGPU is the dominant matmul-cost factor at
our shapes (see `engine:matmul` 8.1 s vs ~3 TFLOPs predicted compute
cost: bandwidth-bound). 4× less weight-bandwidth maps roughly
linearly to wall-clock improvement → **~2–2.5× speedup on
`engine:matmul`/`engine:matmul_many`** (sub-linear because activations
still need to move).

### 2.2 What does NOT get quantized

- **The masked activations sent to the GPU.** They are f32 round-trip
  carriers; Q4 quant would inject noise that doesn't compose with the
  orthogonal unmask.
- **The mask `A` itself.** It's CPU-only and small (`O(n)` for HD₃/DCT-IV
  cascade).
- **In-TEE attention compute.** `tee:attn_cached` runs on CPU under
  BLIS, doesn't touch the engine.
- **OutAttnMult dynamic Q·Kᵀ.** Both operands are runtime
  activations; quantizing them would inject error into the attention
  scores. Stays f32.

### 2.3 The accuracy story — why QuIP#-style rotation is needed

Naive per-tensor Q4 quantization on raw weights loses ~1–3 perplexity
points on common LLM benchmarks (outlier weights blow the
quantization budget; a single large weight forces a scale that makes
most of the rest round to zero).

**QuIP#/QuaRot fix:** apply a random orthogonal rotation `Q` (order
`d_in`) to weights before quantization: `W' = Qᵀ · W`. Rotated
weights have a much tighter entry distribution (the JL transform
makes them approximately Gaussian-bounded), so Q4 covers them with
negligible accuracy loss.

To keep the math right at inference, activations must be rotated by
the same `Q` (right-multiply) **before** the GPU matmul:

```
X' = X · Q             # right-multiply activations by Q
GPU computes:  X' · dequant(W'_q4)  ≈  X' · W'  =  X · Q · Qᵀ · W  =  X · W
```

Since `Q` is orthogonal `Qᵀ · Q = I`. **No mask interaction**: the
token-axis mask `A` is a left-multiply on `X`; the hidden-axis
rotation `Q` is a right-multiply. They commute:
`A · (X · Q) = (A · X) · Q`.

### 2.4 Choice of rotation `Q`

`Q` operates on the hidden dim `d_in`, which varies per op:

| op | d_in | d_out | pow2? |
|---|---:|---:|---|
| QKV | 2 560 | 4 096 (Q) / 1 024 (K, V) | no |
| O | 4 096 | 2 560 | yes |
| gate, up | 2 560 | 9 728 | no |
| FfnDown | 9 728 | 2 560 | no |

So we need a fast orthogonal at non-pow2 `d`. **Reuse the
`MaskKind::Auto` dispatch we just shipped**:
- HD₃ at pow2 `d_in` (FWHT-cascade) — 4 096 only
- DCT-IV at non-pow2 `d_in` — 2 560, 9 728

The rotation `Q` is **fixed and public** (sampled once at
quantization time with a deterministic seed). It is NOT the per-batch
fresh privacy mask — it's a one-time accuracy-preserving rotation.
Per-layer different `Q_l` is the standard recipe (defeats correlation
attacks across layers, and helps with quantization).

At inference, the per-call CPU cost of applying `Q` to activations is
`O(s · d_in · log d_in)` via FWHT/DCT-IV cascade. For Qwen3-4B at
n=2056, the rotation costs ~30–80 ms per call × 144 calls/forward
≈ 4–11 s total. **This is a non-trivial CPU cost** — comparable to
the GPU savings — and motivates an alternative: applying `Q` on the
GPU side as a fused preamble to the matmul kernel.

We'll cross that bridge in Phase 3 if needed. Phase 0–2 can ship with
NO rotation (vanilla Q4 weight quant) and confirm the kernel path
works; Phase 3 adds rotation if accuracy is unacceptable.

---

## 3. Component pieces

### 3.1 Trait surface extension

Add to `GpuOffloadEngine` (in `crates/gelo-protocol/src/substrate.rs`):

```rust
/// Register a quantized weight. Same shape contract as `register_weight`
/// but with a `QuantScheme` describing the encoding. Engines that don't
/// support quantization can fall back to dequantize-then-store-as-f32.
fn register_weight_quantized(
    &mut self,
    handle: WeightHandle,
    weight: ArrayView2<f32>,
    scheme: cubecl_common::quant::scheme::QuantScheme,
) -> Result<()> {
    // Default impl: dequant on the trusted side (i.e., do nothing here)
    // and store as f32. Backends with native quant matmul override.
    self.register_weight(handle, weight)
}

/// Whether this engine has native quantized matmul support — used by
/// the trusted side to decide whether to take the rotation path or
/// fall back to f32 weights.
fn supports_native_quantization(&self) -> bool {
    false
}
```

`matmul()` and `matmul_many()` dispatch internally: if the registered
weight for `handle` is a `QuantizedTensor`, route through `q_matmul`;
else use the existing `f32/f16` path.

### 3.2 `WgpuVulkanEngine` extension

Add a third `WeightStore` variant:

```rust
enum WeightStore {
    F32(HashMap<WeightHandle, Tensor<CubeWgpu32, 2>>),
    F16(HashMap<WeightHandle, Tensor<CubeWgpu16, 2>>),
    Quantized {
        scheme: QuantScheme,
        // burn-cubecl's QuantizedTensorPrimitive.
        weights: HashMap<WeightHandle, QuantizedTensor<CubeWgpu32>>,
    },
}
```

`register_weight_quantized` quantizes via
`burn_tensor::Tensor::from_data(...).quantize_dynamic(scheme)`.
`matmul` and `matmul_many` add a Quantized arm that calls
`Tensor::q_matmul(masked_input, weight)`.

### 3.3 `InProcessTrustedExecutor` integration

The executor calls `provision_weight(handle, weight)` once per
weight at model load. Add a parallel `provision_weight_quantized`
that takes a `QuantScheme`. Default behaviour: store as f32 (no quant)
to preserve current parity. Opt-in via builder:

```rust
let mut exec = InProcessTrustedExecutor::with_seed(engine, seed)
    .with_auto_mask()
    .with_quantized_weights(QuantScheme::default()
        .with_value(QuantValue::Q4S)
        .with_level(QuantLevel::block(&[128])));
```

The executor stores the active scheme; `provision_weight` then routes
through `register_weight_quantized` instead of `register_weight`.

### 3.4 Hidden-axis rotation (Phase 3, gated on accuracy bench)

New module `crates/gelo-protocol/src/rotation.rs`:

```rust
pub struct WeightRotation {
    /// Per-layer fixed rotation `Q_l ∈ R^(d_in × d_in)`. Stored as
    /// a [`MaskFamily`] (Hd3 at pow2 d_in, Dct4 otherwise), built
    /// once at quantization time with a deterministic seed.
    rotations: HashMap<u16 /* layer */, MaskFamily>,
}

impl WeightRotation {
    /// Pre-rotate the activation along axis 1 (hidden axis) using
    /// the layer's `Q_l`. Output has the same shape as input.
    pub fn apply_to_activation(&self, layer: u16, h: &mut Array2<f32>) {
        // Reuse Hd3Mask::apply_in_place / Dct4Mask::apply_in_place
        // but operating on the transposed view (axis 1 of the operand
        // == axis 0 of the mask's required input layout).
    }
}
```

The rotation is applied to `h_norm` before each linear projection,
and the corresponding inverse is pre-baked into the quantized weight
at provision time:

```rust
// At quantization time:
let q_l = MaskFamily::sample_for_dim(layer, d_in, &mut deterministic_rng);
let w_rotated = q_l.unapply(weight.view());  // Qᵀ · W
let w_quant = quantize(w_rotated, scheme);
engine.register_weight_quantized(handle, w_quant, scheme);

// At inference time, in offload_linear:
let masked = mask.apply(stacked_hidden);     // A · X — privacy mask, token axis
rotation.apply_to_activation(layer, &mut masked);  // (A·X) · Q — rotation, hidden axis
let out = engine.matmul(handle, masked.view());  // GPU does (A·X·Q) · (Qᵀ·W_quant) ≈ A·X·W
let unmasked = mask.unapply_take(out);       // Aᵀ · ... — back to X·W
```

This adds one extra `apply_in_place`-equivalent step per offload
(operating on the d-axis, not the s-axis — needs a transposed
variant of the existing mask kernels).

### 3.5 Quantization scheme — recommended initial pick

Per the cubecl-common `QuantScheme` API:

```rust
QuantScheme {
    value: QuantValue::Q4S,        // 4-bit symmetric (no zero-point overhead)
    param: QuantParam::F32,        // f32 scales (precise enough; small overhead)
    store: QuantStore::U32,        // 8× Q4 values packed per u32
    level: QuantLevel::block(&[128]),  // group-size 128 along the hidden axis
    mode: QuantMode::Symmetric,
}
```

Why these choices:
- **Q4S** (symmetric) over **Q4F** (full): symmetric saves the zero-point and is a closer match for HD₃-rotated weights (which are zero-mean by construction).
- **Block size 128**: standard for Q4 LLM inference (Q4_K_M uses 256, GPTQ uses 128). Smaller blocks = better accuracy at slightly more memory overhead.
- **F32 scales**: ~0.8 % memory overhead at block 128; cheap insurance against scale-quantization error.

---

## 4. Phased plan

### Phase 0 — Validate burn-cubecl `q_matmul` on Vulkan (½–1 day)

Write a stand-alone benchmark in `crates/gelo-gpu-wgpu/tests/q4_kernel_spike.rs`:
- Build a `(2056, 2560)` f32 weight tensor and a `(2056, 2560)` f32 input.
- Quantize the weight to Q4S/Block(128) via `quantize_dynamic`.
- Call `q_matmul` and compare timing + accuracy against the f32 `matmul`.
- Acceptance: ≥ 1.5× speedup on the Vulkan iGPU at this shape with mean-relative-error < 1e-2 vs the f32 reference.

**If this fails** (kernel falls back to dequant-then-f32-matmul on Vulkan, or crashes): the whole plan stalls. We'd need to upstream a cubek-matmul Vulkan quant kernel fix (likely small) or write a custom WGSL kernel (3-4 weeks). **Run this first.**

### Phase 1 — `register_weight_quantized` + native dispatch (~2 days)

If Phase 0 clears:
- Extend the `GpuOffloadEngine` trait with `register_weight_quantized` and `supports_native_quantization`.
- Implement on `WgpuVulkanEngine`: add the `Quantized` `WeightStore` variant; `matmul`/`matmul_many` dispatch on it.
- Update `PlaintextExecutor` (default impl: dequant-then-store-as-f32 to preserve parity).
- Unit tests:
  - `matmul_quantized_round_trip_matches_f32_to_1e-2` at the Qwen3 shapes.
  - `engine_supports_native_quantization` reports true for Vulkan.

### Phase 2 — Wire into `InProcessTrustedExecutor` (~1-2 days)

- Add `with_quantized_weights(scheme)` builder + `quant_scheme: Option<QuantScheme>` field.
- `provision_weight` checks the scheme and routes accordingly.
- Add bench cell `gpu_gelo_q4` to `qwen3_long_context_bench.rs`:
  - Same Auto-mask path as `gpu_gelo`, but with Q4 weight registration.
  - Skip if `!engine.supports_native_quantization()`.
- Run at Qwen3-1.7B first (faster sanity), then 4B.

### Phase 3 — Hidden-axis rotation (~3-4 days, gated on accuracy)

Only enter this phase if Phase 2 shows:
1. Q4 speedup is at least 1.7× on the engine buckets, AND
2. Greedy-token parity with f32 weights degrades > 0.5 % top-1 across a small calibration set.

If condition 2 holds:
- Implement `WeightRotation` (`crates/gelo-protocol/src/rotation.rs`).
- Add a transposed apply path to `MaskFamily` (apply along axis 1 of operand instead of axis 0).
- Quantize-time: store `Qᵀ · W` instead of raw `W`.
- Inference-time: insert `apply_to_activation` between mask and engine matmul.
- Re-run accuracy + perf benchmarks.

### Phase 4 — Bench + sign-off (~2 days)

- Run `qwen3_long_context_bench` at Qwen3-1.7B and 4B with all four mask variants × {f32 weights, Q4 weights}.
- Confirm:
  - TTFT savings at HD₃ pow2 branch: ~25–30 %
  - TTFT savings at DCT-IV non-pow2 branch: ~20–25 %
  - TPOT savings comparable.
  - Greedy-token parity with the in-TEE reference within tolerance.
- Update `docs/handoffs/` with measured numbers, decide whether to default `with_quantized_weights` on.

**Total estimated effort: 7–11 days** assuming Phase 0 clears. Multi-week
worst case only if we fall through to a custom WGSL kernel.

---

## 5. Security analysis

### 5.1 Threat model

GELO's threat model already treats weights as **public** — they're shipped
to an untrusted GPU engine in the clear. Quantizing public weights does
not change the security posture:

- **No new information leak.** The Q4-encoded weight reveals only the
  same information as the f32 weight (the model itself is public).
- **No mask interaction.** Q4 dequantization happens on the GPU side,
  inside the matmul kernel. The masked activation `A·X` is dequantized
  against `W` to produce `(A·X)·W`. The unmask `Aᵀ · ((A·X)·W) = X·W`
  is unchanged.
- **Numerical drift only.** Q4 introduces ~1e-2 relative error per matmul
  call. After 6 ops × 36 layers ≈ 216 matmuls the error budget is roughly
  `√216 · 1e-2 ≈ 0.15` — comparable to bf16's drift. Token-output parity
  with f32 reference will not be exact; the in-TEE reference must be
  recomputed at the same precision for parity checks (same caveat as the
  existing fp16 mode).

### 5.2 With hidden-axis rotation (Phase 3)

Adding `Q` doesn't change the threat surface:

- **`Q` is public.** It's part of the weight encoding (the engine needs
  to know which `Q` was used at quantization time so its `q_matmul`
  kernel can be configured correctly, OR the engine doesn't see `Q` at
  all if rotation is applied to activations on the trusted side).
- **Trusted side stores `Q`** (per-layer rotation seeds, ~32 bytes ×
  36 layers ≈ 1 KB).
- **`Q` is not the privacy mask.** Privacy is still `A` (per-forward-pass
  HD₃/DCT-IV cascade on the token axis). `Q` is a fixed orthogonal on
  the hidden axis, sampled once at model load.

### 5.3 What the attacker still cannot do under Q4

The same as before Q4: recover plaintext activations or weights. The
weight is public anyway; the activation is masked by `A`. Q4 quantizes
public data, so it adds no leak.

The **only** new gap is the same as the existing fp16 caveat:
**U-Verify probes** (integrity verification via Freivalds-style
`B·r = (W·r)·...` checks) assume f32 bit-equality between the engine
output and the in-TEE reference. Under Q4 the engine output isn't
bit-equal; probes must widen their tolerance or be disabled. We
already document this for fp16; same applies for Q4.

---

## 6. Open questions

1. **Phase 0 outcome.** Does `q_matmul` actually accelerate on Vulkan?
   Cubek-matmul has tile-quantized kernel paths but I haven't confirmed
   the WGPU runtime exercises them (vs falling back to dequant-then-f32
   matmul on adapter feature gaps). Phase 0 is the load-bearing
   experimental step.

2. **Accuracy without rotation.** Naive Q4 (no `Q`) on Qwen3-1.7B and
   Qwen3-4B: how much greedy-token degradation? QuIP# claims ~1–3 PPL
   without rotation, then negligible with. We need numbers on our
   specific Qwen3 checkpoints; running greedy parity against the
   in-TEE f32 reference on a 50-prompt calibration set should suffice.

3. **TPOT win.** Decode is n=1 → s=9 (HD₃-aligned at decode no matter
   what). Quant matmul speedup might be smaller at this shape (fixed
   kernel-launch overhead dominates). Need to confirm the projected
   ~27 % TPOT savings holds at n=1.

4. **Per-layer different `Q_l` vs single shared `Q`.** QuIP# uses
   per-layer rotations; QuaRot can share. Per-layer is safer for
   accuracy but adds 36× the rotation state. Default to per-layer;
   collapse to shared if memory becomes an issue.

5. **Activation rotation cost.** If Phase 3's `apply_to_activation`
   step costs more than the Q4 GPU savings, the whole stack regresses.
   Mitigation paths: (a) GPU-side fused rotation+matmul kernel,
   (b) apply rotation lazily (only when accuracy needs it — e.g., only
   on the FfnDown's d_in=9728 path where quant error is largest).

6. **Memory layout in `WeightStore::Quantized`.** burn-cubecl uses
   its own `QuantizedTensor` representation. Confirm we can pull
   weight handles by `WeightHandle` key without losing the scheme
   metadata across `clone_shared()`.

---

## 7. Test/bench commands for the next agent

```bash
# Phase 0 sanity (write this test first)
cargo test --release -p gelo-gpu-wgpu --test q4_kernel_spike \
    -- --ignored --nocapture

# Phase 1 unit test (after register_weight_quantized lands)
cargo test --release -p gelo-gpu-wgpu --lib quantized

# Phase 2 long-context bench at 4B
GELO_BLIS_THREADS=16 GELO_BENCH_LENGTHS=2040,2048 \
    GELO_BENCH_MAX_TOKENS=4 GELO_BENCH_SKIP_PERMUTED=1 \
    GELO_BENCH_MASK_KIND=auto GELO_BENCH_QUANTIZED_WEIGHTS=1 \
    cargo test --release -p gelo-gpu-wgpu --test qwen3_long_context_bench \
    qwen3_1_7b_long_context_breakdown -- --ignored --nocapture

# Phase 4 token parity check (after Phase 3 lands or if Phase 2 accuracy is OK)
cargo test --release -p gelo-embedder --test qwen3_generation_e2e \
    -- --ignored --nocapture
```

---

## 8. Suggested skills for next session

- **`diagnose`** — for Phase 0 if `q_matmul` falls back to f32 silently.
  Need to identify whether the cubek-matmul tile-quantized kernel is
  being launched or if dequant happens upstream. `RUST_LOG=cubecl=trace`
  + a profiler trace will surface the kernel dispatch path.
- **`improve-codebase-architecture`** — if Phase 3's rotation kernel
  needs a major refactor (e.g., transposed `apply_in_place` semantics
  on `MaskFamily`), the architectural choices around `WeightRotation`
  storage warrant the deeper analysis.

---

## 10. Phase 0 outcome (2026-05-20) — Q4 does NOT accelerate on Strix Halo iGPU

Spike code: `crates/gelo-gpu-wgpu/tests/q4_kernel_spike.rs` (Vulkan) and
`crates/gelo-gpu-wgpu/tests/q4_hip_kernel_spike.rs` (HIP/ROCm).

Tested at all four Qwen3-4B per-layer projection shapes with
Q4S/Block(128) scheme, across both runtimes and with rocWMMA toggled.
**The first Vulkan sweep below was contention-tainted (concurrent
worktree process inflating absolute times). The re-run rows are
clean.**

### Vulkan (cubecl-wgpu) — first run, with contention

| shape | f32 median | Q4 median | speedup | rel-err vs f32 |
|---|---:|---:|---:|---:|
| QKV-Q (2056, 2560) × (2560, 4096) | 26.8 ms | 30.5 ms | 0.88× | 11.4 % |
| Gate∥Up (2056, 2560) × (2560, 9728) | 128.0 ms | 134.4 ms | 0.95× | 12.6 % |
| FfnDown (2056, 9728) × (9728, 2560) | 35.3 ms | 45.0 ms | 0.78× | 11.8 % |
| O proj (2056, 4096) × (4096, 2560) | 21.8 ms | 25.8 ms | 0.85× | 11.1 % |
| matmul_many (3 weights, shared input) | 144 ms | 158 ms | 0.91× | — |

### Vulkan — clean re-run (authoritative)

| shape | f32 (ms) | Q4 (ms) | speedup |
|---|---:|---:|---:|
| QKV-Q | 18.4 | 22.5 | **0.82×** |
| Gate∥Up | 43.0 | 52.9 | 0.81× |
| FfnDown | 24.3 | 34.4 | **0.71×** |
| O proj | 12.2 | 16.2 | 0.75× |

### HIP (cubecl-hip), std features only

| shape | f32 (ms) | Q4 (ms) | speedup |
|---|---:|---:|---:|
| QKV-Q | 18.1 | 23.9 | 0.76× |
| Gate∥Up | 44.9 | 57.1 | 0.79× |
| FfnDown | 25.3 | 34.6 | 0.73× |
| O proj | 12.6 | 15.9 | 0.79× |

### HIP + rocWMMA (cubecl-hip `+rocwmma` feature, rocwmma-dev 2.2.0 installed)

| shape | f32 (ms) | Q4 (ms) | speedup |
|---|---:|---:|---:|
| QKV-Q | 18.0 | 24.0 | 0.75× |
| Gate∥Up | 42.9 | 54.1 | 0.79× |
| FfnDown | 25.2 | 34.7 | 0.73× |
| O proj | 12.6 | 15.8 | 0.80× |

**All three configurations agree within ~5 %.** Q4 regresses 0.71–0.82×
of f32 at every shape on every backend. ROCm + rocWMMA does not unlock
the WMMA INT4 path on this hardware.

### Diagnosis

- **The Q4 kernel is engaging.** rel-err 11–13 % is exactly the
  expected naive-Q4 (no HD₃ rotation) accuracy band per QuIP# pre-
  rotation baselines. If it were silently dequantizing-then-f32-
  matmuling we'd see f32-level accuracy plus a ~1.5× slowdown
  (dequant cost) — we see Q4-level accuracy with mild slowdown,
  consistent with a working tile-quant kernel.
- **WMMA INT4 is not being engaged on either runtime.** Vulkan + HIP +
  rocWMMA all produce the same ~0.75× speedup. The hardware
  (`gfx1151` does support `v_wmma_i32_16x16x16_iu4`) and the C++ wrapper
  (`rocwmma-dev 2.2.0`) are both installed, but cubek-matmul 0.9.0's
  Q4 path doesn't emit those intrinsics — it routes through generic
  Cmma/Mma strategies plus dequant compute.
- **f32 matmul is well-tuned** on both runtimes (Vulkan and HIP f32
  median differ by <3 %); the Q4 kernel hasn't received the same
  hardware-specific tuning.

### Decision

**Q4 is shelved for the Strix Halo deployment.** It remains the right
direction for discrete-GPU targets where the kernel actually wins.

Next-lever options for the current iGPU substrate:

1. **f16 engine (already exists).** `WgpuVulkanEngine::new_fp16()` is
   already wired but unused in the executor builder. ½ day to bench
   vs f32 at Qwen3 shapes. If it gives ≥1.5× speedup on shader-f16
   AMD adapters, wire `with_fp16_engine()` into the executor.
2. **Upgrade workspace to cubecl 0.10.0 / burn 0.21.** Multi-day
   workspace migration. Check the burn-cubecl 0.21 changelog for
   "Q4" / "INT4" / "WMMA" mentions before committing — only worth it
   if cubek-matmul 0.10+ ships a WMMA-INT4 kernel.
3. **Fork cubek-matmul + add WMMA INT4 kernel.** 2-4 weeks. Requires
   deep cubek-matmul/cubecl-core internals plus rocWMMA C++ wrapping.
   High leverage if successful, risky.
4. **Defer Q4 to discrete-GPU deployment.** Accept that consumer APU
   silicon (Strix Halo) doesn't benefit from Q4 with the current
   kernel stack. The Phase 1-4 plumbing-side work in this doc remains
   correct for that future.

When this re-activates (e.g., for discrete-GPU deployment or a future
cubek-matmul release with WMMA-INT4), the plumbing-side work
(Phases 1–4 of this doc) is still correct — only the Phase 0
hardware/kernel decision changes.

### Workspace artifacts (keep for re-use)

- `vendor/cubecl-hip-sys-patched/` — local cubecl-hip-sys with
  `hip_53211` aliased to `bindings_52802` so cubecl-hip 0.9.0
  compiles against ROCm 7.2.1. Excluded from workspace; wired via
  `[patch.crates-io]`. Delete + revert workspace patch once upstream
  publishes ROCm 7.2 bindings.
- `crates/gelo-gpu-wgpu/tests/q4_kernel_spike.rs` — Vulkan spike,
  reference for future Q4 measurement on this hardware.
- `crates/gelo-gpu-wgpu/tests/q4_hip_kernel_spike.rs` — HIP spike,
  reference for discrete-GPU follow-up.

---

## 9. Out of scope / future follow-ups

- **GPU-side fused rotation+matmul.** Removes the CPU cost of
  `apply_to_activation`. Would require either a custom WGSL kernel
  or upstreaming a "pre-rotation" hook into cubek-matmul. Not v1.
- **E2M1 (FP4) instead of Q4S.** Newer than Q4S, comparable accuracy,
  may have better kernel paths. Worth revisiting after Phase 4 if
  cubek-matmul's E2M1 path turns out to be faster than Q4S.
- **Cross-layer rotation sharing.** Reduces the rotation state from
  36× per-layer to 1× shared. Slightly worse accuracy in theory; in
  practice QuaRot ships with per-layer.
- **Q2 weights.** 8× compression vs f32. Would need stronger
  rotation (QuIP# uses E8 lattice quantization for Q2 — non-trivial).
  Filed for future R&D after Q4 ships.
