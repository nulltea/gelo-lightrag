# M1.12 — bf16/f16-native activation pipeline (perf-bucket 3)

> **Parent context:**
> - Handoff: [`2026-05-22-perf-bucket-roadmap-r3-default.md`](../handoffs/2026-05-22-perf-bucket-roadmap-r3-default.md) §3 — bucket 3 spec (the 3a + 3b split).
> - Plan: [`m1-12-permuted-attention-batched-decode.md`](m1-12-permuted-attention-batched-decode.md) — sibling bucket-2 plan (aborted at Phase A; bucket 3 picks up the next perf lever per the roadmap).
> - Memory: `bf16_mask_gemm_skipped.md` (2026-05-19) — original "deferred" rationale (AOCL-BLIS lacks LPGEMM; ~10 % TTFT gain at B=1 shrinks after HD₃ lands). Bucket 3 revisits because the share at B=8 is ~39 % of prefill wall, not 10 %, and HD₃ is opt-in (not default), so the shrinkage hasn't materialised.
>
> **Status:** plan, post-minimal-grilling decisions baked in.
> **Author date:** 2026-05-22.
> **Scope:** 3a + 3b together. 3a lands first to establish precision contract at low cost; 3b builds on 3a's bf16 GEMM kernel.

---

## 0. TL;DR

`gelo:mask_apply` (14.9 %) + `gelo:mask_unapply` (24.5 %) is **39 % of
prefill wall at B=8** on CPU DDR5, contending with GPU matmul on the
same UMA bus. Today the mask GEMM is `f32 · f32 → f32`; bucket 3 ships
bf16 throughout.

| Sub-bucket | What | Wall | Effort |
|---|---|---|---|
| **3a** | bf16 mask GEMM via **AOCL-BLIS LPGEMM addon** `aocl_gemm_bf16bf16f32of32` (second pivot from OpenBLAS → hand-roll → AOCL LPGEMM — see §3.0). Mask `A` still f32-sampled in TEE, downcast to bf16 only for the GEMM call. Mask never leaves TEE. | Ship-fast | ~1 day: enable addon in install script, rebuild AOCL, add FFI bindings, parity test |
| **3b** | bf16 activation storage end-to-end in the forward pass. `Array2<f32>` activations → `Array2<bf16>`. Matmul accumulates in f32; elementwise kernels widen to f32 internally. GPU upload boundary: bf16 → f16 conversion. | Structural rework | ~2-3 weeks (every forward-pass tensor touched) |

Combined estimate: **3-4 weeks end-to-end**.

Sequencing: **3a → 3b incremental**. 3a's bf16 GEMM kernel is the
precision contract that 3b consumes; ship 3a first, validate at the
mask boundary, then propagate bf16 outward through the forward pass.

---

## 1. Acceptance gates

### 1.1 Performance — sub-bucket 3a (mask GEMM)

- Qwen3-4B prefill wall at B=8 n=2048 drops ≥ **20 %** (the `gelo:mask_apply` + `gelo:mask_unapply` buckets together fall from 39 % share to ≤ 20 %).
- Decode wall within ±5 % of post-R3 baseline (no decode regression; bucket 3a is prefill-focused).
- bf16 mask GEMM parity vs f32: per-element abs tolerance ≤ **1e-3** at the masked operand. Greedy generation token parity unchanged on the v7 extraction bench.

### 1.2 Performance — sub-bucket 3b (activation storage)

- Combined 3a + 3b prefill wall drops ≥ **30 %** total (3a's 20 % + 3b's marginal ~10 % from eliminating the f32→f16 conversion at the GPU offload boundary).
- Memory footprint: forward-pass working set roughly halved (activation tensors are the dominant f32 occupant after the bf16-native weight loader landed).
- Decode wall within ±5 % of post-R3 baseline.
- bf16 forward pass produces semantically-equivalent extraction output: entity/relation sets match the f32 baseline to **f32-floor + bf16 quantisation noise** (≤ 1 % entity-set delta).

### 1.3 Precision contract (parity testing)

- Existing per-kernel parity tests in `crates/gelo-protocol/tests/`, `crates/gelo-embedder/tests/`, and `crates/gelo-reranker/tests/` re-baseline to **bf16-floor tolerance ≈ 1e-3 abs** at the activation tensor.
- Matmul kernel parity (`tee_matmul_bf16`, `engine.matmul`) stays at ~1e-5 because accumulators are f32 internally — bf16 input loss only.
- Greedy generation parity on `qwen3_generation_e2e` test: token sequences stay byte-stable on the canonical "The quick brown fox" prompt + the v7 fixture chunks. Argmax flips at tied logits are acceptable per M1.11 D3 contract.

---

## 2. Threat model — what changes

**Nothing changes structurally.** Mask `A` is still sampled in TEE
from f32 standard-normals via the existing ChaCha20 → Householder QR
path (Haar) or ChaCha20 → ±1 sign vectors (HD₃ / DCT-IV); A never
crosses PCIe. Only the precision of the multiplication `A · H` and
the activation tensors changes from f32 to bf16.

GPU observation:

| Item | Today | Bucket 3 |
|---|---|---|
| Masked operand sent to GPU | `(n+k, d)` f16 (engine converts f32 → f16 on upload) | `(n+k, d)` f16 (engine converts bf16 → f16 on upload — cheaper conversion path) |
| Mask `A` location | TEE-only | TEE-only (unchanged) |
| Adversary visibility on masked operand precision | f32 mantissa quantisation at the GEMM, then f16 at upload | **bf16 mantissa quantisation at the GEMM, then f16 at upload** |
| Effective noise floor on adversary observations | ~1e-5 (f32→f16 quantisation at upload) | ~1e-3 (bf16 → f16 quantisation at upload) — **higher** noise floor |

**Net security delta:** higher quantisation noise on adversary
observations. Numerical analysis: bf16 mantissa truncation adds
~2 bits of random rounding per element to the masked operand. The
GELO security argument is information-theoretic on `A` (random
orthogonal); quantisation noise on the output `A · H` strictly
increases the noise floor against AloePri-class inversion attacks
(anchor_ica / JADE / JD / gram_error). No new attack surface is
introduced.

**AloePri gate: SKIPPED (Q4 decision).** Math-only argument: bf16
quantisation is a noise-floor increase on the same shape regime.
Existing c1-c5 conditions cover the topology; the precision change
is monotonically safer. (Departing from c1-c5 empirical-validation
precedent, but the c1-c5 gates were for *topology* changes
[batched, shared-A, batched-decode shape, attack-driver methodology]
not for precision. Risk accepted.)

---

## 3. Sub-bucket 3a — bf16 mask GEMM via AOCL LPGEMM addon

### 3.0 Decision history (2026-05-22)

Three pivots over the course of one session, each driven by a
disconfirmation of the prior assumption:

1. **Q2 original (OpenBLAS `cblas_sbgemm`)** — rejected when
   inspection showed (a) no system OpenBLAS on the dev host,
   (b) `openblas-src` source build adds 30 min + gfortran dep,
   (c) OpenBLAS and AOCL-BLIS both define `cblas_sgemm`, creating
   link-order risk against the BLIS-mt prefill win.
2. **Hand-rolled AVX-512_BF16** — picked as the no-dep
   alternative. Rejected when web search showed OpenBLAS HAS a
   well-tuned Cooperlake AVX-512_BF16 kernel (`sbgemm_kernel_16x4_cooperlake.c`,
   auto-selected for AMD Zen 5 in OpenBLAS 0.3.29+), making the
   hand-roll a reimplementation of work AMD/OpenBLAS already
   tuned. The dep-complexity reason for avoiding OpenBLAS still
   held, but the engineering reason (no kernel) didn't.
3. **AOCL LPGEMM addon (chosen)** — local inspection of
   `vendor/aocl-blis/addon/aocl_gemm/` found the kernel we need
   IS in the AOCL-BLIS 5.2.2 source we already vendor:
   `aocl_gemm_bf16bf16f32of32.c` (bf16 × bf16 → f32 output),
   with runtime `AVX512_BF16 ISA` check. The vendored library
   doesn't expose the symbols because the install script
   doesn't pass `--enable-addon=aocl_gemm`. Just enable the
   addon and rebuild.

The 2026-05-19 `bf16_mask_gemm_skipped` memory entry asserted
"vendored AOCL-BLIS 5.2.2 has no LPGEMM kernels" — true of the
**built `.so`** (symbols not exposed), false of the **source**
(kernels are present, addon not enabled). Memory entry needs
correction.

**Net result:** ~1 day engineering instead of 1-2 weeks. Same
library, same RPATH, same linkage. AMD's own kernel tuned for
Zen 5. No dep additions, no symbol conflicts.

### 3.1 Engineering scope

**Step 1 — Enable LPGEMM addon in the AOCL-BLIS build.**

Update `scripts/install-aocl-blis.sh`:

```diff
 ./configure \
     --enable-cblas \
     --enable-threading=openmp \
     --enable-shared \
+    --enable-addon=aocl_gemm \
     --prefix="$INSTALL_DIR" \
     amdzen
```

Force-rebuild (~5-10 min on Zen 5):

```bash
rm vendor/aocl-install/lib/libblis-mt.so*
./scripts/install-aocl-blis.sh
```

Verify symbol presence:

```bash
nm -D vendor/aocl-install/lib/libblis-mt.so | grep -i aocl_gemm_bf16
# Expect: aocl_gemm_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32,
# aocl_get_reorder_buf_size_bf16bf16f32of32 (and friends)
```

**Step 2 — Add Rust FFI bindings.**

New module `crates/gelo-protocol/src/aocl_lpgemm.rs`:

```rust
//! FFI bindings for AOCL-BLIS LPGEMM addon's bf16 GEMM kernel.
//!
//! API contract (from `vendor/aocl-blis/addon/aocl_gemm/`):
//! - Type: `bfloat16` = `int16_t` (binary layout matches IEEE-754
//!   bfloat16). Maps to Rust's `half::bf16`.
//! - Reorder-then-compute: bf16 GEMM uses a pre-pack pattern for
//!   the B matrix (the activations side in our usage). Reorder
//!   buffer size is queried via `aocl_get_reorder_buf_size_*`;
//!   buffer populated via `aocl_reorder_*`; GEMM called against
//!   the reordered buffer.
//! - Runtime ISA check: kernel falls back to ref impl if
//!   AVX512_BF16 not detected. Strix Halo Zen 5 passes.

unsafe extern "C" {
    fn aocl_get_reorder_buf_size_bf16bf16f32of32(
        order: c_char,        // 'r' for row-major
        trans: c_char,        // 'n' for no-transpose
        mat_type: c_char,     // 'B' since we reorder the B matrix
        k: i64,
        n: i64,
    ) -> usize;

    fn aocl_reorder_bf16bf16f32of32(
        order: c_char,
        trans: c_char,
        mat_type: c_char,
        src: *const bf16,
        dst: *mut bf16,
        k: i64,
        n: i64,
        ldb: i64,
    );

    fn aocl_gemm_bf16bf16f32of32(
        order: c_char,
        transa: c_char,
        transb: c_char,
        m: i64, n: i64, k: i64,
        alpha: f32,
        a: *const bf16, lda: i64, mem_format_a: c_char,
        b: *const bf16, ldb: i64, mem_format_b: c_char,
        beta: f32,
        c: *mut f32, ldc: i64,
        post_op: *const c_void,  // null for plain GEMM
    );
}
```

(Exact signatures to be confirmed against the header
`vendor/aocl-blis/addon/aocl_gemm/aocl_gemm_interface_apis.h`
at implementation time. The skeleton above captures the shape.)

**Step 3 — Wire into `mask.rs`.**

```rust
pub fn matmul_bf16_lpgemm(
    a: ArrayView2<'_, bf16>,
    b: ArrayView2<'_, bf16>,
) -> Array2<f32> { /* call sequence: reorder B → gemm */ }

pub fn matmul_bf16_lpgemm_trans_a(
    a: ArrayView2<'_, bf16>,
    b: ArrayView2<'_, bf16>,
) -> Array2<f32> { /* pass 't' for transa */ }
```

These replace the proposed hand-rolled `matmul_bf16` /
`matmul_bf16_trans_a`. Same signature surface so the
MaskFamily integration is unchanged.

**Step 4 — Cargo feature.**

The `blas` feature already gates AOCL-BLIS linkage. The LPGEMM
addon symbols are now part of the same `.so` once we enable
the addon, so no new feature gate is needed. The new functions
sit alongside `sgemm_blis` / `matmul_blis` in `mask.rs` under
the existing `#[cfg(feature = "blas")]` umbrella.

**Step 5 — Runtime fallback.**

The AOCL LPGEMM call itself runtime-checks `AVX512_BF16` and
falls back internally. We add a Rust-side guard via
`is_x86_feature_detected!("avx512bf16")`: on hosts without
the ISA, we widen bf16 → f32 in TEE and call the existing
`matmul_blis` instead, so the protocol still works (just at
the precision and bandwidth of today's f32 path).

**Wire it into `MaskFamily::apply` / `unapply`:**

- `MaskFamily::Haar(GeloMask)` — currently does `A.dot(&hidden)` at
  f32. Change to: downcast `A` to bf16 once at mask-construction
  time (cached on the struct), downcast `hidden` to bf16 at
  apply-time (TEE-side, single AVX-512 pass), call
  `tee_matmul_bf16_sbgemm`. The bf16-cached `A` is ~4 GB at
  s=2048+k=8 (vs ~8 GB f32) — within executor memory budget.
- `MaskFamily::Hd3` — already O(s·d·log s) via FWHT; doesn't go
  through a GEMM kernel. **Skip 3a for HD₃**; HD₃ stays at f32
  FWHT. (Future 3c-style rework could push HD₃ to bf16 sign
  vectors + bf16 FWHT, but it's not on this plan.)
- `MaskFamily::Dct4` — same as HD₃; structured-transform, not
  GEMM. **Skip 3a for DCT-IV.**

So 3a's win specifically targets the **Haar mask family** —
the paper-parity default. HD₃ and DCT-IV are already cheaper than
Haar at their target shapes and don't pay the same cost.

### 3.2 Mask cache update

`GeloMask` currently stores `a: Array2<f32>`. Add a parallel
`a_bf16: Option<Array2<bf16>>` field, lazily populated on first
`apply` call. Memory cost: +50 % per mask. Mitigated by: the bf16
cache replaces nothing today, so peak working-set increases by
~4 GB at s=2048. Within tolerance for production hardware.

Alternative: cache ONLY bf16, drop f32 after the QR sample
completes. Saves memory. Trade: `unapply` uses `Aᵀ · M` which is
also bf16 GEMM — same kernel, same precision. Sample → downcast →
discard f32. **Recommend this** to avoid the +50 % peak.

### 3.3 Parity tests

- New test in `crates/gelo-protocol/tests/mask_bf16_parity.rs`:
  generate random `A` (Haar) and `H`, compute `A · H` via both
  f32 (`A.dot(&H)`) and bf16 (`tee_matmul_bf16_sbgemm`). Assert
  abs delta ≤ **1e-3** per element, relative delta ≤ **2e-3**.
- Re-run `qwen3_generation_e2e` and `qwen3_4b_batched_mask_sweep`
  benches with `with_haar_mask_bf16()` opt-in; token parity
  preserved.

### 3.4 Default-flip

Builder opt-in: `InProcessTrustedExecutor::with_haar_mask_bf16()`.
Default off at first commit. Flip default-on after the perf gate
at §1.1 clears AND the parity tests pass on Qwen3-4B real weights.

### 3.5 Failure modes & rollback

- If OpenBLAS dep breaks the `[env]` propagation work from
  `feedback_cargo_env_propagation` memory, fall back to **hand-
  rolled AVX-512_BF16 kernel** (Q2 alt option) — ~1-2 weeks
  engineering, no new dep.
- If parity test fails at `bf16-floor 1e-3` — investigate per-row
  whether OpenBLAS's sbgemm uses f32 accumulators (check); if
  not, switch to a stricter accumulation policy or use Eigen's
  bf16 GEMM via FFI.
- If perf gate misses (≥ 20 % prefill reduction not achieved): the
  win was over-extrapolated from B=1 numbers. Revisit at B=8
  with HD₃ in play; if HD₃ wins anyway, retire bucket 3a in
  favour of flipping HD₃ default-on.

---

## 4. Sub-bucket 3b — bf16 activation storage end-to-end

Reaches for the full structural rework: every `Array2<f32>` in the
forward pass becomes `Array2<bf16>`. Composes with 3a's GEMM kernel
and removes the per-call f32 → f16 conversion at the GPU offload
boundary (the same conversion the bucket-2 abort post-mortem
diagnosed as costing ~134 ms per attention call at decode shape).

### 4.1 Files touched

Enumerated by category. Specific file lists are illustrative — every
forward-pass-tensor occupant.

**Activation tensor type migration:**

- `crates/gelo-embedder/src/decoder/forward.rs` — `h`, `h_norm`,
  `residuals`, attention `ctx`, FFN `gate`/`up`/`down` outputs all
  switch from `Array2<f32>` to `Array2<bf16>`.
- `crates/gelo-embedder/src/decoder/attention.rs` —
  `causal_gqa_attention*` Q/K/V/out tensors, in-TEE matmul calls.
- `crates/gelo-embedder/src/decoder/generation.rs` — `h_last`,
  decode-step intermediates.
- `crates/gelo-reranker/src/causal_discriminator.rs` — score path.
- `crates/gelo-snp-runner/src/extraction.rs` — runtime activation
  storage.

**Elementwise kernel widen-and-narrow:**

- `rms_norm` — read bf16, widen sum-of-squares to f32, reciprocal-
  sqrt, narrow output to bf16. ~50 LOC change. The f32 widen is
  load-bearing for numerical correctness at d=2560 RMS reductions.
- `rope::apply` — bf16 in, f32 sin/cos compute, bf16 out. RoPE
  table stays f32 (small, cached, no benefit to bf16).
- `apply_qk_norm` — same shape as `rms_norm`.
- residual adds — bf16 + bf16 → bf16 (single-precision pass, no
  widening needed; sum is bounded).
- softmax (in `causal_gqa_attention_cached`) — bf16 score input,
  widen to f32 for max + exp + sum, narrow result back to bf16
  (or stay f32 if the result is consumed by the next matmul
  immediately).

**Matmul boundaries:**

- `tee_matmul_bf16` — already bf16-aware; native fit.
- `engine.matmul` / `engine.matmul_dynamic_batched` — currently
  takes `ArrayView2<'_, f32>`. Two options:
  - **Option A (smaller diff):** keep trait signature f32; convert
    bf16 → f32 at the call site in substrate (`offload_linear`).
    Pays a per-call conversion but moves the boundary to one
    place. Simpler migration.
  - **Option B (larger diff):** widen trait signature to accept
    `ArrayView2<'_, bf16>` (or add bf16 variants). Eliminates the
    conversion entirely. Substrate-level refactor; every engine
    impl updated.
  - **Recommend A first, then B as a follow-up** if conversion
    cost is binding (per the bucket-2 post-mortem diagnosis,
    f32→f16 conversion is ~134 ms/call at large shapes; f32→bf16
    is similar; bf16→f16 is ~half that).

**Mask boundary:**

- `MaskFamily::apply` / `unapply` — input is bf16, output is bf16.
  For Haar with 3a's `tee_matmul_bf16_sbgemm`: input bf16, internal
  f32 accumulate, output bf16 (narrow at end). For HD₃ / DCT-IV:
  bf16-aware FWHT / DCT inner loop — sign-flip + butterfly stay in
  bf16, no widening needed.

### 4.2 GPU boundary

Today: engine receives f32 input, converts to f16 inside
`array2_to_tensor_f16` (`gelo-gpu-wgpu/src/lib.rs:216`). Tomorrow:
substrate hands the engine a bf16 input. Two paths:

- **Path α (Option A above):** substrate converts bf16 → f32 at
  call site, engine converts f32 → f16. **Two conversions.** Same
  perf as today for this hop.
- **Path β (Option B above):** add `engine.matmul_bf16` /
  variants; engine converts bf16 → f16 directly via
  `array2_bf16_to_tensor_f16` (already exists at
  `gelo-gpu-wgpu/src/lib.rs:235` for the bf16 weight upload path).
  **One conversion**, half the bandwidth, half the time.

**Recommend Path α for the initial 3b ship** (smaller diff,
incremental). File Path β as a follow-up perf optimization that
lands once the rest of 3b is stable.

### 4.3 Migration order (incremental)

1. **Week 1:** Ship 3a behind `with_haar_mask_bf16` flag (perf
   validation + parity tests). Default off at commit.
2. **Week 2:** Convert `decoder::forward::run` and
   `decoder::forward::run_batched` activation tensors to bf16,
   with f32 widen at GPU offload boundary (Path α). Keep
   elementwise kernels at their current f32 (transient widening
   internally). Greedy parity test on `qwen3_generation_e2e`.
3. **Week 3:** Convert elementwise kernels (`rms_norm`,
   `apply_qk_norm`, `rope::apply`, residual adds, softmax) to
   bf16-in/bf16-out with internal f32 widening. Greedy parity test
   re-run; tolerate bf16-floor delta.
4. **Week 4:** Convert `tee_matmul_bf16` consumers (e.g.,
   `tee:qkv_direct`, `tee:o_direct`, `tee:swiglu_proj_direct`) —
   skip-layer paths today. Re-baseline `qwen3_m1_12_r1_q1_microbench`.
5. **Week 4+:** Perf gate at §1.2. Flip `with_bf16_activations()`
   builder default on. Update memory + plan with measured numbers.

### 4.4 Parity contract

- Per-kernel parity tests re-baselined to bf16-floor (~1e-3 abs).
- Greedy generation parity on `qwen3_generation_e2e`: token
  sequences byte-stable on canonical prompts. Argmax-flips at
  tied logits are acceptable (M1.11 D3 contract).
- Extraction bench `extract_and_query_bench`: entity/relation set
  delta ≤ 1 % (semantic equivalence; not bit equality).

### 4.5 Failure modes & rollback

- **RMSNorm overflow at long n:** if the f32 widen-internally
  policy isn't sufficient (e.g., bug in the conversion path,
  Welford reduction needed), revisit per-op widening policy.
  Q3-rejected option (mixed precision per op) is the fallback —
  costs ~30 % more engineering for the precision matrix.
- **Greedy parity flips outside argmax-tied range:** indicates
  a real numerical bug somewhere in the bf16 path. Revert to
  flag-gated; debug at the per-kernel level via the existing
  parity tests.
- **Perf gate misses:** if 3a + 3b only delivers ~20 % prefill
  reduction instead of ≥ 30 %, the activation-storage win is
  smaller than projected. Possible cause: Path α's substrate-
  side bf16 → f32 conversion at the offload boundary is paying
  back the upload conversion. Mitigation: ship Path β (engine
  trait widening) as a follow-up.

---

## 5. Precision argument (why no AloePri gate)

The masked operand the GPU observes today is `A · H` quantised to
**f16** (via the `array2_to_tensor_f16` upload). Tomorrow it's
`A · H` quantised first to **bf16** (in TEE GEMM) then to **f16**
(at upload). The mantissa precision strictly decreases — adversary's
observation is noisier.

GELO's security argument is information-theoretic on `A`: `A` is
sampled Haar-uniform on O(s) and observed only through `A · H` for
unknown `H`. Quantisation noise on the observation increases the
adversary's lower bound on recovery error — every observed entry
has rounding noise ~2¯⁷ of its magnitude (bf16) vs ~2¯¹⁰ (f16). The
AloePri attack drivers (anchor_ica, JADE, JD, gram_error) all
operate on observation noise floors; a noisier observation strictly
weakens the attacks' recovery rates.

No new shape regime. No new topology. No new attack surface. The
existing c1-c5 conditions cover the dispatch shapes; precision is
orthogonal to them.

**Caveat:** the `aloepri_attack_harness_disparities` memory
(2026-05-20) flags that math-vs-attack-data disparities exist
in the c1-c5 harness — some drivers (VMA / IA static-attack,
IMA-EmbedRow-ridge) have known methodology gaps. The math-only
argument here assumes the noise-floor monotonicity holds; if a
future revision of the attack drivers surfaces a precision-dependent
recovery rate, this argument needs revisiting.

---

## 6. Out of scope / follow-ups

- **HD₃ bf16 conversion (3c-style):** HD₃ FWHT could in principle
  run on bf16 sign vectors + bf16 butterflies. Today HD₃ uses
  ChaCha20-sampled ±1 signs (fits any precision) but f32 butterfly
  arithmetic. A bf16 HD₃ would save memory bandwidth but not
  compute (FWHT is already memory-bound). Skip in v1.
- **Engine trait widening (Path β):** add `matmul_bf16` /
  variants to `GpuOffloadEngine`. Substrate-level refactor; lands
  if Path α's residual conversion cost is measurable.
- **f16 throughout the forward (Q3 option b):** explicitly
  rejected per the precision-contract analysis in Q3. Filed as
  research item if/when a need arises (e.g., dGPU bring-up with
  f16-native compute kernels).
- **bf16-mask FWHT for HD₃:** see HD₃ note above.
- **Activation quantisation (int8 / int4):** different milestone.
  Per the round-3 research memo, Q4 GPU weights is the relevant
  follow-up. Composes with bucket 3 but separate engineering.
- **bucket 4 (R4 async pipelining)** — order interaction noted in
  roadmap. Shipping bucket 3 first collapses R4's payoff on iGPU
  (no CPU mask bucket left to overlap with). Q#2 RADV-async spike
  still owed independently to settle whether R4 ships at all.

---

## 7. Open questions

1. **OpenBLAS dep co-existence with AOCL-BLIS** — does linking
   both libraries in the same binary cause symbol clashes? Per
   `feedback_cargo_env_propagation`, AOCL-BLIS uses
   `$ORIGIN`-relative RUNPATH; OpenBLAS may have its own
   linkage requirements. Verify at first wire-up.

2. **bf16 mask GEMM with Mezzadri sign correction** — `A` is
   Haar-sampled with Mezzadri sign correction after the QR
   factorisation (load-bearing per the comment at `mask.rs:553`).
   The bf16 downcast happens AFTER the sign correction. Confirm
   the bf16-cached `A` preserves the Haar property (it should:
   bf16 quantisation doesn't introduce systematic bias). Spot-
   check empirically: sample 100 random A's, verify
   `mean(A_bf16) ≈ 0` and `cov(A_bf16) ≈ I`.

3. **HD₃ + 3b interaction** — HD₃ today uses f32 FWHT butterflies
   on bf16-storage input would need either (a) widen butterflies
   to f32 internally (current path; works) or (b) bf16-native
   butterflies (smaller diff, lower precision). Recommend (a)
   for v1 — same widening discipline as the elementwise kernels.

4. **Verify probe at bf16** — U-Verify's Freivalds check
   (`verify_probes > 0` mode) compares engine output against a
   TEE-side reference computation. At bf16, the precision
   tolerance widens. Update probe threshold; verify no false
   positives.

---

## 8. Reproducing the baseline before 3a starts

```bash
# Existing post-R3 baseline (bucket-3 gate measured against)
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 \
GELO_BENCH_MAX_TOKENS=64 \
  cargo test -p gelo-gpu-wgpu --release \
  --test qwen3_m1_12_r1_q1_microbench -- --ignored --nocapture \
  m1_12_per_op_breakdown_prefill_decode

# Existing mask round-trip parity tests (re-baselined at bf16-floor)
cargo test -p gelo-protocol --release mask_

# Existing greedy generation test (token parity preserved post-3b)
cargo test -p gelo-embedder --release --test qwen3_generation_e2e
```

The numbers in §1 are measured against these.

---

## 9. References

- `bf16_mask_gemm_skipped.md` (2026-05-19) — original deferral
  rationale; the share-vs-effort calculus at B=1 vs B=8 shift
- `feedback_cargo_env_propagation` — AOCL-BLIS RUNPATH setup
  (relevant for OpenBLAS co-existence)
- `feedback_memory_efficiency_priority.md` — "never upcast bf16 →
  f32" rule; 3b satisfies it more thoroughly than today's path
- `feedback_benches_use_gelo_gpu.md` — measurement discipline
- `aloepri_attack_harness_disparities.md` (2026-05-20) — known
  attack-harness methodology gaps (caveat for §5's math-only
  argument)
- `crates/gelo-protocol/src/mask.rs:556` — `sample_haar_orthogonal`
  + Mezzadri sign correction (load-bearing for §7 Q2)
- `crates/gelo-protocol/src/sim.rs:113` — `rng: ChaCha20Rng` (main
  mask RNG; precision-orthogonal)
- `docs/handoffs/2026-05-22-perf-bucket-roadmap-r3-default.md` §3
  — bucket-3 spec (3a + 3b split)
