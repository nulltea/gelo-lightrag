# Handoff — HD₃ Hadamard-cascade mask: non-pow2 regression, suggested fixes, attack-defence gate

> **Subject:** Round 3 perf step B (HD₃) landed as opt-in. Wins
> −28 % TTFT at pow2-aligned shapes (n=2040) but regresses
> +51 % at the canonical n=2048 shape due to power-of-two padding.
> Closing the regression needs a non-pow2 orthogonal cascade — no
> standard "Bluestein FWHT" exists; the candidates are listed below.
> The security gate (B.3 attack-suite re-run vs Haar) is also still
> open and is what blocks flipping the executor default.
>
> **Reference artifacts** (read first; this handoff does not duplicate
> their content):
> - Commit `2e8db60` — the HD₃ landing (`crates/gelo-protocol/src/hd3.rs`, `mask.rs` `MaskFamily`/`MaskKind`, `sim.rs` `with_hd3_mask()`, bench env knob)
> - `docs/research/private-llm-inference-round-3.md` — round 3 research doc; HD₃ is plan B; §B.3 is the attack-suite gate
> - `docs/prototype/gelo-complexity-analysis.md` — bottleneck breakdown that motivates HD₃
> - `crates/gelo-protocol/src/hd3.rs` module docs — math, SIMD/rayon kernel, padding contract
> - `memory/hd3_mask_landed.md` — short-form summary
> - Predecessor handoffs: `2026-05-19-bf16-mask-deferred.md` (bf16 closeout), `2026-05-18-m1-10-perf.md` (round-2 perf handoff)

---

## 1. What landed (and verified)

`InProcessTrustedExecutor::with_hd3_mask()` swaps the dense Haar mask
for `A = D₃·H·D₂·H·D₁·H` (QuIP#/QuaRot primitive). Implementation
highlights, all in commit `2e8db60`:

- **Hd3Mask primitive** (`crates/gelo-protocol/src/hd3.rs`):
  power-of-two-only side length; 3·s ±1 bits per mask;
  in-place row-axis FWHT with SIMD-intrinsic butterfly
  (AVX-512F: 16 f32/inst, AVX-2 fallback: 8 f32/inst, scalar last
  resort), rayon-parallel above
  `FWHT_RAYON_WORK_THRESHOLD = 65 536` elements.
- **MaskFamily enum** (`crates/gelo-protocol/src/mask.rs`) wraps
  Haar (`GeloMask`) and Hd3 (`Hd3Mask`); shared `apply` / `unapply` /
  `n()` API.
- **Executor integration** (`crates/gelo-protocol/src/sim.rs`):
  `mask_kind: MaskKind` field; `build_shielded_and_apply` dispatches
  on kind. For non-pow2 `n + k`, zero-pads the stacked-with-shield
  operand to `s_pad = (n+k).next_power_of_two()` (the round-trip
  identity holds because the FWHT cascade is orthogonal — see
  hd3.rs module docs for why the padded rows have to flow through
  the GPU too).
- **Bench knob**: `GELO_BENCH_MASK_KIND=hd3` swaps the `gpu_gelo`
  cell to HD₃ in `qwen3_long_context_bench`.
- **Tests** (7 new, all green):
  - `hd3::tests::hd3_round_trip_preserves_matmul` (7 shape combos
    including long-context)
  - `hd3::tests::hd3_orthogonality`
    (`AᵀA = I` at n ∈ {8, 16, 32, 64, 128, 256})
  - `hd3::tests::hd3_deterministic_from_seed`
  - `hd3::tests::hd3_rejects_non_pow2`
  - `hd3::tests::hd3_round_trip_relative_error_at_long_n`
    (relative RMS error < 1e-4 at n=4096, d=2048)
  - `sim::tests::hd3_executor_agrees_with_plaintext`
  - `sim::tests::hd3_qkv_agrees_with_plaintext`

## 2. The regression — what we measured

Apples-to-apples bench at threads=16, Qwen3-1.7B prefill, `max_tokens=4`:

| shape | Haar TTFT | HD₃ TTFT | delta vs Haar | overhead vs gpu_plain |
|---|---:|---:|---:|---:|
| **n=2040** (n+k=2048 pow2) | 14.97 s | **10.72 s** | **−28 %** ✓ | Haar +138 %, HD₃ **+78 %** |
| **n=2048** (n+k=2056, pad → 4 096) | 15.40 s | 23.30 s | **+51 %** ❌ | Haar +120 %, HD₃ +255 % |

Per-bucket diagnosis at n=2048 (the problem case):

```
gelo:mask_sample    3 134 → 0.01 ms    (−3.13 s  ✓ Haar QR gone)
gelo:mask_apply     1 492 → 3 001 ms   (+1.51 s  ↑ FWHT at 2× padded shape)
gelo:mask_unapply   2 626 → 5 860 ms   (+3.23 s  ↑ ditto)
engine:matmul_many  2 600 → 6 378 ms   (+3.78 s  ↑ GPU does 2× more rows)
engine:matmul       1 849 → 4 038 ms   (+2.19 s  ↑ same)
                                       ───────
                                       +7.6 s net regression
```

**Root cause** — at our default `k_shield = 8` and `n=2048`, we have
`s = n + k = 2 056` which is **not** a power of two. The FWHT
cascade requires pow2 side length, so the executor zero-pads to
`s_pad = 4 096`. The padded operand flows through the engine (so the
unapply has the full s_pad rows to recover from), doubling GPU
matmul cost. The CPU FWHT is also working at s_pad=4096 — even
SIMD+rayon can't make the 2× memory traffic disappear at the
memory-bandwidth ceiling (~80 GB/s on Strix Halo).

**Where HD₃ does win** — at any shape where `n + k` is already a
power of two:

```
n=2040 (k=8, s=2048):   exact pow2  →  −28 % TTFT  (validated)
n=4088 (k=8, s=4096):   exact pow2  →  expected −30 %+  (not benched)
n=8184 (k=8, s=8192):   exact pow2  →  expected −35 %+  (not benched)
```

For long-context generation regimes (n ≥ 4k) the relative padding cost
shrinks — at n=4088 padding is exact (no overhead). The non-pow2
regression is concentrated at the n=2048 boundary case.

## 3. Suggested fix — option matrix

There is **no standard "Bluestein FWHT"**. The Bluestein chirp-z trick
relies on `w^{kn} = w^{(k²+n²−(k−n)²)/2}` which uses DFT's complex
roots of unity; WHT has no analogous factorisation, and the
literature has no canonical non-pow2 fast Hadamard transform.
Candidates that close (or partially close) the gap are listed below
in increasing implementation cost:

| # | approach | what it is | effort | cost-vs-Haar at n=2048 | security argument |
|---|---|---|---|---|---|
| A | **Adaptive Haar fallback** | Auto-dispatch in `with_hd3_mask()`: at pow2 `n+k` use Hd3Mask, at non-pow2 fall through to GeloMask. | ~2 hours | strict improvement: HD₃ when possible, Haar elsewhere (no regression risk) | inherits each component's existing argument |
| B | **FFT-with-random-phase cascade** | `A = D₃·F⁻¹·D₂·F·D₁·F⁻¹` over real-FFT; `D_i` are real ±1 with palindromic symmetry (`D[k] = D[N-k]`) so Hermitian symmetry of the spectrum is preserved → real output. Use `rustfft` for arbitrary N. | 3-4 days microkernel + ~1 week security spike | likely net win at non-pow2; ~2-3× FWHT cost per element but no padding | **No QuIP#/QuaRot proof for FFT cascade**; new incoherence argument needed |
| C | **Mixed-radix WHT** | For `N = 2^a × b`: FWHT_{2^a} on one tensor axis, dense Haar (size b) on the other. | 2-3 days | at n=2048 → b=257 prime → dense 257×257 cost ~ s²·d / (s/log s)·d — same asymptotic as Haar. Not a win. | inherits Haar+QuIP from the parts |
| D | **DCT-II cascade** | `D₃·DCT·D₂·DCT·D₁·DCT`; DCT-II is orthogonal at any N, O(N log N) via FFT. | 3-4 days | similar trade-off to B (~3-4× FWHT cost per element) | same gap as B |
| E | **Prompt-side alignment** | Document that callers should pick `n = pow2 − k_shield` (e.g., n=2040 instead of 2048). Trim or pad prompts as needed. | 0 (docs only) | gives HD₃ win to anyone who aligns | unchanged |

### Recommended sequence

1. **Land A first** (adaptive fallback, ~2 hours). Lets `with_hd3_mask()`
   stop accidentally regressing users at non-pow2 shapes — pure
   improvement, no security regression, easy revert. This is the
   pragmatic "make it safe to opt in" change.
2. **Document E** (prompt-side alignment) in
   `docs/prototype/gelo.md` and the round-3 doc — tell long-context
   users they get a 28 % TTFT win by trimming 8 tokens of context.
3. **Plan B** (FFT-cascade) as a separate research effort.
   It's the only candidate that genuinely closes the gap at n=2048
   without a security-axis trade. Requires:
   - Real-FFT cascade design + complex-vs-real bookkeeping
   - Palindromic-D entropy bound (lower than HD₃'s 3·s bits)
   - **A fresh security analysis** — no published incoherence proof
     for FFT-cascade obfuscation. Either prove it ourselves or
     accept a weaker hardness claim than HD₃'s QuIP#-inherited
     bounds.

C, D, E are mostly redundant once A + B are in place. Don't pursue
C at our shape (the 257-prime factor kills the asymptotic).

## 4. Remaining HD₃ testing surface — attack defence vs Haar

The HD₃ implementation lands as **opt-in research-grade**, **not**
default-on, because of one open question that the perf trade alone
can't answer: **does the discrete `2^{3·s}`-element HD₃ orbit defeat
the GELO §4.3 attack pipeline as well as the continuous Haar measure
does?** Until we re-run the paper's published attacks against
HD₃-with-shield at our shapes and confirm parity with Haar-with-shield,
the default has to stay Haar.

### What needs to be run

The GELO paper §4.3 (Belikov & Fedotov, arXiv:2603.05035) defines
the attack pipeline against the obfuscated `U = A·H` observable. We
need each component re-run against `MaskFamily::Hd3` at the
Qwen3-1.7B activation shapes, with results compared head-to-head
against `MaskFamily::Haar`:

| § | attack | implementation hint | metric |
|---|---|---|---|
| 4.3.3 | Anchor-based recovery — k known plaintexts | ridge LS for `A_K = UH_K^T·(H_K H_K^T + λI)⁻¹` + (a) deflation/FastICA, (b) projection, (c) constrained ICA. `linfa-ica` for FastICA. | p95 non-anchor cosine similarity per Table 6 (k ∈ {1, 10, 50, 100, 200}) |
| 4.3.3 | JADE | Cardoso 1993, ~200-400 LOC reference impl | p95 cosine similarity |
| 4.3.3 | Joint Diagonalization (JD) | Belouchrani et al. 1997 | p95 cosine similarity |
| 4.3.4 | Geometric recovery — Hungarian-matched Gram error | Hungarian matching via existing crate; row-side Gram on matched rows; Frobenius error vs identity | matched-subset Gram error per Table 7 |

### Acceptance criterion (gate B.3 in the round-3 doc)

For HD₃ to be promoted from opt-in research to default:

```
non-anchor p95 cosine similarity (HD₃, shield k=8, σ=4.0)
   ≤ non-anchor p95 cosine similarity (Haar, shield k=8, σ=4.0) + 0.05
   for each anchor count k ∈ {1, 10, 50, 100, 200}

AND

Frobenius Gram error (HD₃, shield k=8, σ=4.0)
   ≥ Frobenius Gram error (Haar, shield k=8, σ=4.0) − 20 %
   at the matching anchor counts
```

That is: HD₃ should be **at most marginally worse** than Haar across
the published attack metrics. The ±0.05 / ±20 % bands are the
paper's noise/error reporting tolerances.

### Suggested crate layout

New crate `crates/gelo-attacks` (referenced in round-3 doc §B.3):

```
crates/gelo-attacks/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── anchor.rs       # ridge LS + ICA/projection/constrained variants
    ├── ica.rs          # FastICA wrapper + JADE + JD ports
    ├── metrics.rs      # p95 cosine sim, matched-subset Gram error
    ├── harness.rs      # run an attack vs MaskFamily; collect numbers
    └── bin/
        └── hd3_vs_haar.rs   # the actual A/B comparison script
```

Dependencies likely needed:
- `linfa-ica` (FastICA implementation, MIT/Apache-2.0)
- `pathfinding` or `lapjv` (Hungarian-algorithm row matching)
- The existing `gelo-protocol::MaskFamily`

### What we already have on the security side

- The `Hd3Mask::hd3_orthogonality` test confirms `AᵀA = I` to f32
  noise — the orthogonal-mixing property is verified.
- The `hd3_executor_agrees_with_plaintext` and
  `hd3_qkv_agrees_with_plaintext` tests confirm round-trip
  correctness on the protocol-side: nothing in the executor or
  pipeline is silently breaking the mask round-trip.
- Per-batch freshness is preserved: every `begin_forward_pass`
  samples 3·s fresh sign bits via the executor's RNG.
- Shield rows still work the same way — the existing
  shield-stack code path runs unchanged before the HD₃ apply.

What we **don't** have:
- Comparison of HD₃ vs Haar under any published attack (the gate).
- A formal incoherence-style proof for HD₃ in the GELO threat
  model. QuIP# proves incoherence for quantisation; that's a
  different downstream property than BSS-hardness. The proof gap
  has to be closed empirically (the attack suite) or theoretically
  (a research paper) before HD₃ becomes default.

## 5. Concrete next-step list (priority order)

1. **Implement A (adaptive Haar fallback)** in
   `crates/gelo-protocol/src/sim.rs`. Modify
   `build_shielded_and_apply` so that when `mask_kind == Hd3` and
   `n + k` is not a power of two, it transparently uses Haar for
   that forward pass and logs once. Add a parity test (in `sim.rs`'s
   tests module) that exercises the fallback at n=2048. ~2 hours.
2. **Document E (prompt-side alignment)** in
   `docs/prototype/gelo.md` §HD₃ — explain that callers picking
   `n = next_pow2(n_target) − k_shield` get the 28 % win.
3. **Stand up `crates/gelo-attacks`** crate with one binary that
   runs anchor-recovery on (a) MaskFamily::Haar, (b)
   MaskFamily::Hd3 at the Qwen3 shapes. Compare numbers against
   GELO paper Table 6. ~1 week.
4. **Add JADE + JD ports** to `crates/gelo-attacks`. ~1 week.
5. **If gate B.3 passes**: flip `MaskKind::Hd3` to default for the
   long-context regime; document the migration in
   `memory/paper_parity_default.md`.
6. **If gate B.3 fails or shows a gap**: tune shield row count /
   energy to close it (cheap), or fall back to Haar as default
   permanently and treat HD₃ as a research-context option.
7. **Plan B (FFT-cascade)** in `docs/research/` only if (3-5)
   establish that HD₃-pow2 is a real win in production. Otherwise
   the FFT-cascade investment isn't justified.

## 6. Suggested skills for next session

- **`grill-me` / `grill-with-docs`**: when designing the security
  spike, stress-test the HD₃ attack-resistance claim. The honest
  question is "what specifically does HD₃ inherit from QuIP#'s
  incoherence proof, and does that imply BSS-hardness in the GELO
  threat model?" The grill-me skill is well-suited to working that
  out interactively.
- **`diagnose`**: if the adaptive Haar fallback (item 1 above) ends
  up regressing something subtle, the diagnose skill is the right
  reproduce-minimise-fix loop. Signal source:
  `cargo test -p gelo-protocol --lib hd3_` + the long-context
  bench.
- **`improve-codebase-architecture`**: if implementing B
  (FFT-cascade) — the real-FFT bookkeeping is non-trivial and the
  module structure decision (extend `Hd3Mask` vs. add a new
  `FftMixingMask` vs. generalise to a `StructuredOrthogonal`
  trait) is worth the skill's deeper analysis.

## 7. One non-obvious gotcha

The `Hd3Mask::apply` and `::unapply` always allocate a fresh
`Array2::zeros(self.n, *)` buffer per call. At our prefill shape
(s_pad=4096, d=2048) that's a 32 MB allocation per call × ~308
calls/forward = ~10 GB of allocator churn per prefill forward.

The existing `build_shielded_and_apply` scratch buffer
(`stacked_scratch: HashMap<usize, Array2<f32>>`) covers the
input-side of `apply` but the OUTPUT of `apply` is freshly
allocated by `Hd3Mask` and there's no internal scratch reuse. This
is probably part of why the FWHT wall is dominated by memory
bandwidth — we're paying both the FWHT memory traffic AND the
allocator's bzero-of-32 MB-per-call.

Fix candidate: add a `Hd3Mask::apply_into(&mut self, hidden,
out_buf)` variant that writes into a caller-supplied scratch. The
executor can hold the scratch in
`InProcessTrustedExecutor::stacked_scratch` (or a parallel field
keyed by `(d_out, s_pad)`). Worth ~1-2 days; might be worth bundling
with item 1 (adaptive fallback) since both touch the same code path.
