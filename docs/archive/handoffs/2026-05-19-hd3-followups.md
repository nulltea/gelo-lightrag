---
type: handoff
status: current
created: 2026-05-19
updated: 2026-05-19
tags: [hd3, mask]
---

# Handoff — HD₃ Hadamard-cascade mask: non-pow2 regression, suggested fixes, attack-defence gate

> **Subject:** Round 3 perf step B (HD₃) landed as opt-in. Wins
> −28 % TTFT at pow2-aligned shapes (n=2040) but regresses
> +51 % at the canonical n=2048 shape due to power-of-two padding.
> Closing the regression needs a non-pow2 orthogonal cascade — no
> standard "Bluestein FWHT" exists; the candidates are listed below.
> The security gate (B.3 attack-defence re-run vs Haar) is also
> still open and is what blocks flipping the executor default — but
> the merge from `path-2-aloepri-gemma` (commit 829110d) brought in
> `evals/aloepri-attacks/`, which is most of the attack-harness
> infrastructure already and shortcuts B.3 considerably.
>
> **Reference artifacts** (read first; this handoff does not duplicate
> their content):
> - Commit `2e8db60` — the HD₃ landing (`crates/gelo-protocol/src/hd3.rs`, `mask.rs` `MaskFamily`/`MaskKind`, `sim.rs` `with_hd3_mask()`, bench env knob)
> - Commit `829110d` — the merge from `path-2-aloepri-gemma` that brought in `evals/aloepri-attacks/` (Rust snapshot-capture + Python attack drivers + 3-condition runner)
> - `evals/aloepri-attacks/README.md` — the harness's own operator runbook (capture → attack matrix → JSON)
> - `crates/gelo-protocol/src/snapshot.rs` — the PCIe-side snapshot capture API the harness consumes
> - `docs/research/private-llm-inference-round-3.md` — round 3 research doc; HD₃ is plan B; §B.3 is the attack-defence gate
> - `docs/archive/prototype/gelo-complexity-analysis.md` — bottleneck breakdown that motivates HD₃
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
   `docs/dev/prototype/gelo.md` and the round-3 doc — tell long-context
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

HD₃ lands as **opt-in research-grade**, **not** default-on, because
of one question the perf trade alone can't answer: **does the
discrete `2^{3·s}`-element HD₃ orbit defeat published attacks
against masked activations as well as the continuous Haar measure
does?** Until we re-run attacks against HD₃-with-shield and confirm
parity with Haar-with-shield at our shapes, the default has to stay
Haar.

### Big update: `evals/aloepri-attacks/` is already most of the infrastructure

The merge from `path-2-aloepri-gemma` (commit 829110d) brought in
`evals/aloepri-attacks/`, which is *exactly* the attack-harness
infrastructure I previously scoped as "new `crates/gelo-attacks/`
scaffold, ~1 week." It already has:

- A **Rust snapshot-capture binary**
  (`evals/aloepri-attacks/src/bin/capture_snapshots.rs`) that runs
  Qwen3-1.7B forward through the GELO `InProcessTrustedExecutor`,
  taps the PCIe-side snapshot via
  `crates/gelo-protocol/src/snapshot.rs`, and exports per-condition
  `<slug>.safetensors` + `<slug>.meta.json`.
- A **three-condition control framework** (per its README):
  `c0_plain` (no mask, baseline) / `c1_mask_only` (mask, no shield)
  / `c2_default` (mask + shield k=8 σ=4.0).
- **Six attack drivers in Python**
  (`evals/aloepri-attacks/attack_drivers/`):
  `run_vma.py` Vocab-Matching, `run_ima.py` Inverse-Mapping,
  `run_isa.py` Inverse-Subspace, `run_tfma.py` Token-Frequency-
  Matching, `run_sda.py` Statistical-Distance, `run_ia.py`
  Inversion.
- A **`run_all.py` orchestrator** that runs the 3×6 matrix and
  emits an `aloepri_attack_results_v1` JSON with an
  `acceptance_gate` block.

Threat model alignment: the snapshots are exactly `U = A·H` (and
`U·W` engine outputs) — the GELO observables. So the harness IS
testing GELO's threat model directly, just through AloePri-flavoured
attack families.

### Attack-family coverage: AloePri vs GELO §4.3

The two attack families overlap but don't fully substitute:

| concern | AloePri family (already in harness) | GELO §4.3 family (still needed) |
|---|---|---|
| Token-level leakage (vocabulary, frequency) | ✓ VMA, TFMA | not directly tested |
| Per-row inverse mapping | ✓ IMA, ISA, IA | partially via anchor-based recovery |
| Statistical distance to baseline | ✓ SDA | partially via Gram error |
| **Anchor-based recovery** with ridge LS + ICA variants (paper §4.3.3) | ✗ | needs FastICA / projection / constrained ICA |
| **JADE** (Cardoso 1993) | ✗ | needs port |
| **Joint Diagonalization** (Belouchrani 1997) | ✗ | needs port |
| **Matched-subset Gram error** (paper §4.3.4) | ✗ | needs port |

Verdict: **the existing harness is necessary but not sufficient** for
the B.3 gate. The right plan is to (a) add HD₃ as a fourth condition
to the existing 3-condition framework, then (b) add the missing
GELO §4.3 attack drivers alongside the existing AloePri ones.

### Concrete plan to satisfy the B.3 gate

**Phase 1 (≤ 1 day) — adopt HD₃ into the existing harness**:

- Add `c3_hd3` condition to
  `evals/aloepri-attacks/src/bin/capture_snapshots.rs`. The
  condition selector and `run_condition` arm are at lines ~89-110
  and ~414+; mirror the c2_default branch and chain
  `.with_hd3_mask()` on the executor builder. Update the matching
  arms in `to_conditions()` and the meta.json writer.
- Add `c3_hd3` to the conditions table in `run_all.py` and the
  acceptance-gate logic.
- Capture snapshots: `cargo run --release -p aloepri-attack-snapshot-runner --bin capture_snapshots -- --condition c3 --max-prompts 64`.
- Run the 6 AloePri attacks against the new condition:
  `python run_all.py --conditions c2_default,c3_hd3
   --snapshot-root snapshots/qwen3-1.7b
   --output results/hd3-vs-haar-aloepri.json`.
- Compare. Either the AloePri attacks distinguish HD₃ from Haar (bad
  — HD₃ regresses against this attack family) or they don't (good
  but only proves part of the threat model).

**Phase 2 (~ 1 week) — add the missing GELO §4.3 attacks**:

Add new attack-driver scripts in
`evals/aloepri-attacks/attack_drivers/`:

| file | attack | source | metric |
|---|---|---|---|
| `run_anchor_ica.py` | Anchor-based recovery — ridge LS for `A_K = UH_K^T (H_K H_K^T + λI)^{-1}`, then FastICA / projection / constrained-ICA variants | paper §4.3.3 + `linfa-ica` for FastICA | p95 non-anchor cosine similarity at k ∈ {1, 10, 50, 100, 200} |
| `run_jade.py` | JADE | Cardoso 1993, ~200-400 LOC port | p95 cosine similarity |
| `run_jd.py` | Joint Diagonalization | Belouchrani 1997 | p95 cosine similarity |
| `run_gram_error.py` | Hungarian-matched Gram error | paper §4.3.4 | matched-subset Frobenius error vs identity |

Wire each into `run_all.py` so the same 4-condition × 10-attack
matrix runs end-to-end. The harness's `AttackResult` /
`AcceptanceGate` types in `attack_drivers/common.py` should
accommodate the new metrics without structural change.

**Phase 3 — flip the default**:

If both phases pass acceptance (HD₃ at most marginally worse than
Haar across all metrics) → set `MaskKind::Hd3` as the default in
`InProcessTrustedExecutor::new` / `::with_seed`, document the
migration in `memory/paper_parity_default.md` and the round-3 doc.

### Acceptance criterion (gate B.3 in the round-3 doc)

For HD₃ to be promoted from opt-in research to default:

```
For each metric M ∈ {p95 cosine similarity, Gram error,
                    VMA/IMA/ISA/TFMA/SDA/IA TTRSR}:
    M(c3_hd3, shield k=8 σ=4.0)
        within paper's reported tolerance of
    M(c2_default = Haar, shield k=8 σ=4.0)

For the paper-defined metrics specifically:
    non-anchor p95 cosine sim:
        HD₃ value ≤ Haar value + 0.05
    Frobenius Gram error:
        HD₃ value ≥ Haar value − 20 %
```

That is, HD₃ should be **at most marginally worse** than Haar across
both the AloePri-family and GELO §4.3-family attacks. The ±0.05 /
±20 % bands are the paper's noise/error reporting tolerances.

### What we already have on the security side

- `Hd3Mask::hd3_orthogonality` confirms `AᵀA = I` to f32 noise.
- `hd3_executor_agrees_with_plaintext` / `hd3_qkv_agrees_with_plaintext`
  confirm round-trip correctness on the protocol side.
- Per-batch freshness preserved (every `begin_forward_pass` samples
  3·s fresh sign bits).
- Shield rows still work — the existing shield-stack code path runs
  unchanged before the HD₃ apply.

What we still don't have:
- HD₃ snapshots captured through the merged harness (need the
  `c3_hd3` condition wired in).
- Either AloePri or GELO §4.3 attack runs against HD₃.
- A formal incoherence-style proof for HD₃ in the GELO threat
  model. QuIP# proves incoherence for quantisation downstream;
  that's a different property than BSS-hardness. The proof gap
  has to be closed empirically (the harness) or theoretically
  (a research paper) before HD₃ becomes default.

## 5. Concrete next-step list (priority order)

1. **Implement A (adaptive Haar fallback)** in
   `crates/gelo-protocol/src/sim.rs`. Modify
   `build_shielded_and_apply` so that when `mask_kind == Hd3` and
   `n + k` is not a power of two, it transparently uses Haar for
   that forward pass and logs once. Add a parity test (in `sim.rs`'s
   tests module) that exercises the fallback at n=2048. **~2 hours.**
2. **Document E (prompt-side alignment)** in
   `docs/dev/prototype/gelo.md` §HD₃ — explain that callers picking
   `n = next_pow2(n_target) − k_shield` get the 28 % win.
   ~30 minutes.
3. **B.3 Phase 1 — adopt HD₃ as `c3_hd3` condition** in the merged
   `evals/aloepri-attacks/` harness. Edit
   `src/bin/capture_snapshots.rs` (~lines 89-110, 414+, condition
   table) to add the new condition that builds an
   `InProcessTrustedExecutor::with_seed(...).with_hd3_mask()` arm.
   Update `run_all.py` and the acceptance-gate JSON shape.
   Capture snapshots, run the 6 AloePri attacks (VMA / IMA / ISA /
   TFMA / SDA / IA) against c2_default + c3_hd3. Compare. **≤ 1
   day** of code; snapshot + attack runtime depends on prompt
   count (~1-2 hours at the gate cap of 256 prompts).
4. **B.3 Phase 2 — add GELO §4.3 attack drivers** alongside the
   AloePri ones in `evals/aloepri-attacks/attack_drivers/`:
   - `run_anchor_ica.py` (anchor + FastICA/projection/constrained,
     `linfa-ica` for FastICA)
   - `run_jade.py` (Cardoso 1993 port)
   - `run_jd.py` (Belouchrani 1997 port)
   - `run_gram_error.py` (Hungarian-matched Frobenius)
   Wire into `run_all.py`. **~1 week.**
5. **If B.3 passes (Phase 1 + Phase 2 acceptance criterion held)**:
   flip `MaskKind::Hd3` to default for the long-context regime in
   `InProcessTrustedExecutor::new` / `::with_seed`. Document the
   migration in `memory/paper_parity_default.md` and the round-3
   doc.
6. **If B.3 fails or shows a gap**: tune shield row count / energy
   to close it (cheap, ~hours), or fall back to Haar as default
   permanently and treat HD₃ as a research-context option.
7. **Plan B (FFT-cascade for non-pow2)** in `docs/research/` only if
   (3-5) establish that HD₃-pow2 is a real win in production.
   Otherwise the FFT-cascade investment isn't justified — the
   adaptive-Haar-fallback from (1) plus prompt-side alignment from
   (2) covers all known callers without the new security spike.

## 6. Suggested skills for next session

- **`grill-me` / `grill-with-docs`** — for B.3 Phase 2 (adding the
  GELO §4.3 attacks), it's worth stress-testing the HD₃ attack-
  resistance claim against someone playing red-team. Honest open
  question: "what does HD₃ inherit from QuIP#'s incoherence proof,
  and does that imply BSS-hardness in the GELO threat model?"
- **`diagnose`** — if any phase of B.3 shows an unexpected gap
  between HD₃ and Haar, use the diagnose loop to reproduce-
  minimise-instrument. Signals: TTRSR per condition in
  `results/<run>.json`, plus per-attack output logs from
  `attack_drivers/*.py`.
- **`improve-codebase-architecture`** — only if pursuing item 7
  (FFT-cascade). The real-FFT bookkeeping is non-trivial and the
  module-structure decision (extend `Hd3Mask` vs. add a new
  `FftMixingMask` vs. generalise to a `StructuredOrthogonal`
  trait) benefits from the skill's deeper analysis. Not needed for
  items 1-6.

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
