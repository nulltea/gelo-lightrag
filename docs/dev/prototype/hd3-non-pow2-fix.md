---
type: prototype-note
status: current
created: 2026-05-20
updated: 2026-05-20
tags: [hd3, gpu, mask]
---

# HD₃ non-pow2 fix — closing the 32 s vs Haar 25.9 s regression at n = 2048

> **Scope.** Design-grade write-up of options to fix the HD₃ pad-to-pow2
> GPU-side regression. The Walsh-Hadamard cascade requires power-of-two
> side length; at our long-context shape (n=2048, k=8 → s=2056) the
> caller pads to s_pad=4096, doubling the GPU GEMM cost. Measured
> 2026-05-20 on Qwen3-4B: HD₃ TTFT 32 012 ms at n=2048 vs Haar 25 873 ms —
> HD₃ regresses 24 % at the canonical production shape.
>
> **Reference artifacts:**
> - `crates/gelo-protocol/src/hd3.rs` — current HD₃ implementation
> - `crates/gelo-protocol/src/sim.rs` `build_shielded_and_apply` — caller-side pow2 pad
> - `docs/research/private-llm-inference-round-3.md` §2.1 — HD₃ adoption rationale
> - `memory/qwen3_4b_perf_2026_05_20.md` — measured 4B numbers
> - `memory/hd3_mask_landed.md`, `memory/hd3_radix8_and_scratch_reuse.md` — prior HD₃ work

---

## Definitions

| symbol / term | meaning |
|---|---|
| `n` | data row count (prompt-token count at prefill) |
| `k` | shield row count (paper default 8) |
| `s` | stacked-with-shield size = `n + k` |
| `s_pad` | pow2 padding of s = `s.next_power_of_two()` |
| `d` | hidden-state width (or any per-call operand width) |
| HD₃ | randomised Hadamard cascade `A = D₃·H·D₂·H·D₁·H` (QuIP#/QuaRot primitive) |
| Haar | dense Householder-QR-sampled orthogonal mask from the GELO paper |
| FWHT | Fast Walsh-Hadamard Transform, `O(n log n)` butterfly on pow2 `n` |
| DCT | Discrete Cosine Transform (real-valued, orthogonal at any `n`) |
| BSS | Blind Source Separation — the family of attacks (anchor recovery, ICA, JADE, JD) the GELO mask must defeat |
| incoherence (QuIP#) | the property that mask rows have bounded `O(1/√n)` entries — load-bearing for QuIP# weight quantisation accuracy |

---

## 1. The problem, structurally

`Hd3Mask::fresh(s)` asserts `s.is_power_of_two()`. When `s = n+k` is not
pow2 (which is most production shapes — n=2048 → s=2056, n=4096 → s=4104,
n=8192 → s=8200), the caller in
`sim.rs:build_shielded_and_apply` pads the stacked-with-shield operand
to `s_pad = s.next_power_of_two()`. At n=2048 this is `s_pad = 4096`
(2× s).

The CPU FWHT cost on `(s_pad, d)` is cheap — `O(s_pad · d · log s_pad)`
even after radix-8 SIMD. Measured 4B HD₃ `mask_apply` at s_pad=4096
costs ~2 100 ms across all 144 calls/forward — ~14 ms/call, dominated
by memory bandwidth, not FLOPs.

**The GPU sees s_pad rows.** At 4B with s_pad=4096:
- `engine:matmul` 6 233 ms (vs 3 363 ms at n=2040 s_pad=2048) — 1.85× more
- `engine:matmul_many` 9 399 ms (vs 4 320 ms at n=2040) — 2.18× more

Total GPU work jumps from ~7.7 s to ~15.6 s — **+7.9 s wall time, +43%
of total TTFT**.

### Why we can't trim rows post-FWHT

HD₃ is structurally orthogonal: `A · stacked` mixes ALL s_pad rows of the
input into ALL s_pad output rows. The first n+k output rows are NOT a
function of just the first n+k input rows — they're a weighted sum
across all s_pad inputs (including the zero-padding rows).

If we send fewer than s_pad rows to the GPU and unmask the partial
output, the unmasking identity fails:

```
Let P = row-selection projection onto first (n+k) rows.
Then Aᵀ · P · A ≠ I    (P is non-trivial; rank-(n+k) idempotent)
The unmask round-trip leaks energy from pad rows into data rows.
```

The pad rows being zero before HD₃ doesn't help — after one FWHT stage
they're mixed with data rows, and the linear combinations that
reconstruct data rows depend on those (now non-zero) pad-row
contributions. Truncation in the middle of the cascade breaks round-trip.

This rules out every option of the form "do HD₃ at s_pad but ship fewer
rows to GPU". The mathematical structure of orthogonal full-mixing
transforms forbids it.

---

## 2. Options surveyed

Each option includes the cost regime, what it costs to ship, and the
security argument. Ranked by expected ROI.

### Option A — Hybrid auto-dispatch: HD₃ at pow2 `s`, Haar elsewhere

In `build_shielded_and_apply`, choose mask family per-call:
- If `(n + k).is_power_of_two()`: HD₃ (FWHT-cascade)
- Otherwise: Haar (dense Householder-QR)

Per-forward-pass mode picks once at `begin_forward_pass` based on `n`;
per-offload mode picks per call.

**Performance.** At pow2-aligned shapes (n=2040, 4088, 8184, …) users
get the full HD₃ win (−28 % TTFT measured at 4B). At non-pow2 shapes
they get exactly today's Haar latency — no regression vs the current
default. The hybrid is monotonic: HD₃ ≤ Haar at every shape.

**Security.** Both primitives are already shipped and gated by their
own attack suites (Haar is paper-baseline; HD₃ cleared B.3 attack
re-run per user's note 2026-05-20). The composition is sound: each
forward picks one or the other based on a public input (`n`). No
cross-family leakage.

**Cost.** 1-2 days. Code change is local to `sim.rs` and the executor's
`set_mask_kind` API:
- Add `MaskKind::Auto` variant
- `build_shielded_and_apply` reads `mask_kind`; if `Auto`, picks at call site
- `begin_forward_pass` does the same for per-forward-pass mode
- Tests: parity tests with both families fire at the corresponding shapes

**Limitation.** Users hitting "round-numbered" prompt sizes (2048, 4096,
8192) still get Haar latency. Users who can pick (2040, 4088, 8184)
get HD₃. Production prompt lengths are usually shaped by tokeniser
output, not user choice — so most prompts will fall on the wrong side
of the boundary by chance. Realistic expected behaviour: ~50 % of
prompts get HD₃, ~50 % get Haar, averaging ~−15 % TTFT over a workload.

**This is the right ship-it-now option.** It guarantees no regression
and harvests the HD₃ win wherever shape allows. It is **not** the
final fix.

---

### Option B — Caller-side prompt alignment (`n → next_pow2(n+k) − k`)

Front-end pads or truncates prompts to a nearest-pow2-minus-k length
before they reach the executor. Two sub-variants:
- B.1: round UP (pad with attention-mask-zero tokens). At n=2049, pad
  to n=4088 with 2039 dummy tokens — pays full s_pad=4096 cost on GPU
  but with one extra cell of overhead.
- B.2: round DOWN (truncate). At n=2049, drop to n=2040 — loses 9 tokens
  of context.

**Performance.** Worst case B.1 pays the same cost as A's Haar fallback
(in fact slightly worse — Haar at s=2056 vs HD₃ at s=2048+pad). B.2
gets the full HD₃ win but at the cost of lost context.

**Security.** Identical to vanilla HD₃; no new analysis.

**Cost.** Trivial code — a `pow2_align_prompt(ids, k)` helper. But:
- User-visible API contract change (prompt length not preserved).
- The padding/truncation behaviour has to be agreed at every API
  boundary (CLI, RAG, agent harness).
- Production RAG queries often hit specific token counts due to
  chunking; this forces them off the natural shape.

**Limitation.** Pushes the problem to the caller. Most callers can't
trade prompt content for latency without semantic loss.

**Verdict.** Skip as primary fix. Acceptable as an opt-in flag for
power users who can tolerate the trade.

---

### Option C — Block-diagonal HD₃ at non-pow2

Partition `s` into pow2-aligned blocks. e.g., s=2056 = 1024 + 1024 + 8:
three blocks. Run HD₃ independently on each block with independent
sign cascades. Mask matrix is block-diagonal:

```
A = diag(HD₃_1024, HD₃_1024, HD₃_8)
```

**Performance.** CPU FWHT cost: Σ block_size · log(block_size) ≈
2048 · 11 + 8 · 3 = 22 552 + 24 = ~22.6k FLOPs/d vs monolithic FWHT
at s_pad=4096 of 4096 · 12 = 49 152 FLOPs/d. Block-diagonal is ~2.2×
faster on CPU. GPU sees s = 2056 rows (no pad) — saves the +7.9 s
that pad-to-4096 costs. Projected TTFT at 4B n=2048 with block-
diagonal HD₃: ~18 500 ms (matches measured n=2040 HD₃).

**Security.** Block-diagonal A has a known weakening:
- Within-block: full HD₃-with-shield protection (sign cascade orbit
  size 2^{3·block_size}).
- Cross-block: correlations between data in block i and block j leak
  through the public block structure. An attacker who can solve the
  per-block separation problem now gets O(s/B) linear constraints
  per token, vs O(1) for monolithic HD₃.

The tail block (size 8 at our shape) is the load-bearing weakness:
2^{24} = 16M sign cascades is **brute-force enumerable** by an
attacker who controls the GPU. Specifically, the attacker:
1. Samples a candidate sign-vector triple for the tail block.
2. Computes `HD₃_8_cand · masked_tail_block`.
3. Checks correlation with known cleartext anchors (e.g., system-prompt
   tokens that always start the prompt).
4. Repeats over 16M candidates — feasible in <1 s on a single GPU.

This collapses the security of the tail block to "anchor-leak +
brute-force key recovery", which is exactly the attack class GELO §3.2
is designed to defeat.

**Mitigation: enforce minimum block size.** If all blocks ≥ 2^{16},
brute force is infeasible (2^{48} ≈ 3 × 10^{14}). At s=2056 we'd need
blocks {1024, 1024, 8} → no, the 8 is forced by `s - 2·1024 = 8`.

Possible re-partition: pick block sizes that avoid tiny tails. For
s=2056:
- {2048, 8} — tail 8 is unsafe
- {1024, 1024, 8} — same problem
- {512, 512, 512, 512, 8} — same
- {1024, 1024} ignoring 8 tokens — but then 8 rows are unmasked, which
  defeats the shield's purpose

**Verdict.** Theoretically the right structural fix, but the tail-block
problem at our shape kills it without a security spike. The spike
needs to either (a) prove the small tail is BSS-safe under shield
absorption, or (b) re-architect to avoid small tails. Estimate:
2-4 weeks of security analysis + attack-suite re-run (the existing
HD₃ B.3 gate adapted for block-diagonal).

**Filed.** Defer to security spike. If the spike clears, this is the
best long-term answer. The aloepri-attacks harness (`evals/aloepri-attacks/`)
should add a `c4_hd3_blockdiag` condition mirroring c3_hd3.

---

### Option K — DCT-cascade `A = D₃ · C · D₂ · C · D₁ · C`

Replace each `H` in the HD₃ cascade with `C`, the orthonormal DCT-II
matrix. DCT-II is real-valued, orthogonal, and computable in
`O(n log n)` at **arbitrary n** via FFT-based algorithms (or the
direct Lee/Loeffler recursions).

**Performance.** DCT-II has ~3× the operation count of FWHT at the
same n (one DCT-II = one length-n real FFT plus pre/post twiddles).
But it works at the exact s = n+k, no padding. Projected total CPU
mask cost at 4B n=2048: ~3.5× the current FWHT cost ≈ ~4.2 s ÷ 3 ×
1 → ~1 400 ms apply + ~3 500 ms unapply (assuming unapply
scratch-reuse lands per Task #4) → ~4 900 ms total mask CPU.
GPU sees s=2056 — no pad regression. Projected TTFT: ~20-21 s at 4B
n=2048.

Net vs current HD₃-at-n=2048 (TTFT 32 s): saves ~11 s.
Net vs Haar at n=2048 (TTFT 25.9 s): saves ~5 s.
Net vs proposed block-diagonal HD₃ (TTFT ~18.5 s): worse by ~2 s.

**Security.** Open question, this is the load-bearing concern:
- **Orthogonality**: DCT-II is exactly orthogonal at any n. Round-trip
  identity holds.
- **Incoherence (QuIP#-style)**: DCT-II row entries are
  `√(2/n) · cos((2j+1)iπ/2n)` (with adjusted first row). Max entry is
  `√(2/n)` — same `O(1/√n)` bound as Hadamard rows. The incoherence
  property survives.
- **BSS-distinguishing-game hardness**: this is where DCT differs from
  Hadamard. Hadamard rows are uniform ±1 sequences — they look like
  random noise to an attacker. DCT rows are cosine basis vectors —
  they live in a public, low-dimensional, structured space.
  An attacker can project the masked output onto the DCT basis and
  recover the diagonal signs:
  ```
  D · C · x  -- projecting onto C-basis gives D · x (modulo a known transform)
  Recovering D from {D · x_i} is the standard "sign-recovery from
  many-anchored observations" problem, hard but not "Hadamard-noise" hard.
  ```
- The composition `D₃ · C · D₂ · C · D₁ · C` does NOT obviously cure
  this. Each C is the same public matrix; the variation is in the
  Dᵢ signs only. The orbit size is `2^{3n}` — same as HD₃ — but the
  attack distinguishing structure is different.

**No published cryptographic analysis specific to DCT-cascade-as-mask.**
QuIP#/QuaRot use Hadamard specifically because of its noise-like
properties; the literature has not (to our knowledge) explored DCT
substitution.

**Cost.** 2-3 weeks implementation + 2-4 weeks security analysis:
- Real-valued DCT-II SIMD kernel (or wrap `rustfft` with real-only
  input via Hermitian symmetry).
- Plumbing through `MaskFamily::Dct3`.
- Attack-suite extension: rerun all §4.3 attacks against
  `c5_dct3` condition + add a DCT-projection-specific attack
  variant.

**Verdict.** A serious option but **higher risk than C and more
expensive than A**. Only worth pursuing if both A (ship-it-now) and
C (security spike) fail to deliver. Filed as a research follow-up.

---

### Options ruled out

| option | why ruled out |
|---|---|
| Bluestein-style chirp-z for FWHT | No analog — chirp-z is specific to DFT's `w^{kn}` structure |
| Mixed-radix Hadamard at non-2-power orders | Hadamard at order 257 doesn't exist; Williamson/Paley constructions don't fit s=2056 |
| Truncated/sub-pow2 FWHT | Orthogonality lost; round-trip identity fails |
| Pad-row "smart" filling (extra shield rows, deterministic structure) | Pad rows still cost GPU compute — the regression is on the GPU side, not CPU |
| GPU-side fused HD₃ kernel | Breaks GELO threat model — engine sees unmasked H briefly |
| Shrink k to align (n+k) to pow2 | At n=2048, requires k=0 (defeats shield) or k=2048 (worse) |
| Per-forward HKDF-derived mask material | Orthogonal direction — saves Haar QR cost, doesn't help non-pow2 |
| Caller-side defer to next decode step | Tactical workaround; user-visible TPOT spike |

---

## 3. Recommended phased plan

### Phase 1 — Ship Option A (hybrid auto-dispatch)

**Effort.** 1-2 days.
**Deliverable.**
- `MaskKind::Auto` variant
- `build_shielded_and_apply` and `begin_forward_pass` consult shape
- Bench cell `gpu_gelo_auto` in `qwen3_long_context_bench.rs` to track
  the per-shape dispatch
- Default-on (replace current `MaskKind::Haar` default with
  `MaskKind::Auto` after Phase 1 lands and passes attack suite)

**Expected payoff.** At pow2-aligned shapes (n+k pow2), full HD₃ win.
At non-pow2, today's Haar baseline. Guarantees no regression at any
shape. Removes the largest current footgun ("HD₃ helps at n=2040,
hurts at n=2048" is invisible to most callers).

This is the **primary deliverable**. Everything below is optional
optimization.

### Phase 2 — Security spike on Option C (block-diagonal HD₃)

**Effort.** 2-4 weeks (security analysis + attack-suite re-run + impl).
**Deliverable.**
- Written security analysis: block-diagonal HD₃ with minimum-block-size
  constraint. The analysis needs to answer:
  1. Can the tail block be safely absorbed into the preceding block
     (i.e., make blocks {1024, 1032} with overlap)? Doesn't break
     orthogonality (still block-diagonal modulo the boundary), might
     close the small-tail attack.
  2. What's the minimum block size that's BSS-resistant under the
     shield?
- `c4_hd3_blockdiag` condition in `evals/aloepri-attacks/`
- If spike clears: implement `MaskFamily::Hd3BlockDiagonal`, wire
  into hybrid dispatcher as the non-pow2 branch (replacing Haar)
- If spike fails: file in `docs/research/future-rnd.md`, fall back
  to Option A as the permanent answer

**Expected payoff.** ~30 % additional TTFT win at non-pow2 shapes
(from Haar's ~25.9 s to block-diagonal HD₃'s projected ~18.5 s at
4B n=2048). Lifts the average-over-prompt-shapes gain from ~15 %
(Option A alone) to ~28 % (matching HD₃-at-pow2).

### Phase 3 — DCT-cascade research (Option K), only if Phase 2 fails

**Effort.** 4-6 weeks (research + impl).
**Deliverable.** A standalone research deliverable. Don't pursue unless
Phase 2 closes with a hard "block-diagonal fails BSS" verdict.

---

## 4. Out-of-scope clarifications

- **This is orthogonal to bf16-mask.** That decision was closed
  separately (see `memory/bf16_mask_gemm_skipped.md`). bf16 attacks
  CPU mask-GEMM cost; this proposal attacks GPU-pad cost. The two
  don't interact.
- **This is orthogonal to Q4-GPU weights.** Once HD₃ default-on lands
  via Phase 1, Q4-GPU is the next strategic step (paper-target compound
  stack). HD₃ rotation is the QuIP#/QuaRot preprocessor that makes Q4
  numerically safe.
- **The `unapply_in_place` + scratch-reuse refactor (Task #4) lands
  independently of Phase 1.** It's already merged or in flight; this
  doc assumes the unapply optimisation is in.

---

## 5. Open questions for the next agent

1. Is there a published analysis we missed of "Walsh-Paley-like
   transforms at arbitrary order N preserving QuIP#-style incoherence"?
   The OFFT/OFDM literature on real orthogonal transforms might have
   leads.
2. For Option C tail-block absorption: does the overlapping-block
   variant `{1024, 1032}` (last 8 of block 1 also in block 2, with
   block-2 covering rows [1016, 2048)) preserve orthogonality? The
   block-overlap matrix is the union of two block-rotation matrices,
   which is generally not orthogonal. Probably needs a Gram-Schmidt
   on the boundary, which costs O(k²·d) — small at our k=8.
3. Is `MaskKind::Auto` worth a separate executor-config knob, or should
   it just be the default? Argument for being default: it's strictly
   ≤ Haar at every shape, so "do nothing worse than today" is the
   conservative behaviour.

---

## 6. Deep dive — 2026-05-20 update

This section overrides §3 (the recommended phased plan) after stress-
testing the Option C security argument and refining Option K. The
top-level recommendation is **unchanged for Phase 1 (Option A)**;
Phase 2 changes from block-diagonal HD₃ to **DCT-IV cascade** because
block-diagonal is structurally broken at our shapes under a realistic
multi-anchor adversary.

### 6.1 Option C revisited — multi-anchor attack on block-diagonal HD₃

The original §2 analysis flagged only the tail-block enumerability
problem (`HD₃_8` has 2²⁴ orbit, brute-force feasible). **A sharper
attack defeats block-diagonal HD₃ at any block size relevant to our
shapes.**

Setup. Block-diagonal `A = diag(A_1, …, A_B)` with each `A_i = HD₃` at
order `b_i`. Each `A_i` is defined by `3·b_i` random sign bits per
forward. The cleartext within block `i` is `H_i ∈ R^{b_i × d}`; the
masked block is `U_i = A_i · H_i`.

Adversary model. Realistic for GELO: the attacker knows the system
prompt (a fixed boilerplate of ~20-50 tokens that prepends every user
query). Those tokens project to specific known hidden-state rows after
the embedding lookup and first few rmsnorm+layer ops. Call these
anchored rows `H_i[anchor]` for some indices in block i.

Attack on block i. The attacker observes `U_i = A_i · H_i`. For each
anchored row `H_i[a, :]` they have:

```
U_i[a, :] = A_i[a, :] · H_i             — one row equation
         = sum_j A_i[a, j] · H_i[j, :]   — d scalar equations
```

`A_i[a, :]` is a single row of HD₃ at order b_i — a deterministic
function of the 3·b_i sign bits in `A_i`. With **k anchored rows in
block i**, the attacker has `k·d` scalar equations in `3·b_i`
unknowns (the sign bits).

At Qwen3-4B shapes:
- d = 2 560 (or up to 9 728 for FfnDown intermediate)
- Suppose `b_i = 1024` (the largest pow2 block from {1024, 1024, 8}).
- Then `3·b_i = 3 072` sign-bit unknowns.
- With **just 2 anchored rows**: `2 · 2 560 = 5 120` scalar equations.
- Over-determined system in sign-cascade space.

The system is nonlinear (each output coordinate depends multiplicatively
on three signs), but the structure is bounded: each equation is a
polynomial of total degree 3 in the sign-bits. With over-determined
constraints and a moderate amount of compute, **MILP / SAT solvers
recover the sign cascade in minutes** (typical hardness for
multilinear-3 systems of this size in cryptanalysis practice — see
e.g., Albrecht-Cid-Faugère 2009 on linearization attacks).

**Worse for the production threat model:** the system prompt is
identical across many user queries. The attacker can accumulate
anchored-row observations across a session; even one anchored token in
each block per query gives them a fresh constraint per query. Within a
few thousand queries (one user session) they have multi-thousand
anchored rows per block — wildly over-determined.

**Result.** Block-diagonal HD₃ at any block size `b_i` with `d > 1.5 b_i`
(satisfied at every Qwen3 projection) is broken under the multi-anchor
adversary. The original §2 analysis only considered the 8-block tail;
this generalises the problem to every block.

Mitigation candidates:
- **Add shield rows per block** to confuse anchor identification: turns
  block-diagonal back into "monolithic HD₃ at slightly bigger order".
  Defeats the purpose (we lose the pow2-fit advantage).
- **Increase cascade depth** (HD_k cascade with k > 3): adds `n·(k-3)`
  more sign bits per block. At cascade depth 5, b_i=1024 has 5 120 sign
  bits; the multi-anchor break point moves to `k > 5 120/d ≈ 2` anchors.
  Still trivially breakable.
- **Mix blocks** via a small "binding" structure (e.g., a sparse
  permutation across blocks): adds coupling but reintroduces public
  structure that ICA/JADE can exploit.

**Verdict.** Option C is dead under our adversary model. Remove from
the Phase 2 plan. The multi-anchor weakness is a structural property
of block-diagonal sign cascades, not specific to our block size choice.

### 6.2 Option K refined — DCT-IV cascade `A = D₃ · C^IV · D₂ · C^IV · D₁ · C^IV`

The original §2 worried about DCT-II's constant first row (entries all
`1/√n`). The first DCT-II row leaks `(D₁)_0 · ⟨1, x⟩/√n` — a direct
function of the input row-sum and one diagonal sign.

**DCT-IV does not have this weakness.** Its row formula is:

```
C^IV[i, j] = √(2/n) · cos((2i+1)(2j+1) π / 4n)
```

For all `i ∈ [0, n)` the row is a balanced cosine sequence with no DC
component and entries bounded by `√(2/n)` — exactly the QuIP#-style
incoherence bound. No row is constant. Crucially: DCT-IV is **also
exactly orthogonal** (`(C^IV)ᵀ · C^IV = I`) and computable in
`O(n log n)` at arbitrary `n` via standard DCT-IV-from-DFT algorithms
(see, e.g., Tolimieri-An-Lu 1997, or
`rustfft` + Bluestein twiddles).

Cost analysis at 4B n=2048 (s = 2056, no pad):

| pass | FLOP / call | wall (BLIS-mt / FFT cost model) |
|---|---:|---:|
| 3 × DCT-IV at s=2056 via Bluestein-DFT at 4096 (~4× FWHT-pow2 cost) | ~150 M FLOPs/call (3-cascade) | ~33 ms / call at d=2560 |
| × 144 apply calls/forward | — | ~4 800 ms (apply CPU) |
| same for unapply (in-place + scratch reuse) | — | ~4 800 ms (unapply CPU) |
| mask_sample (3 sign diagonals) | ~6 k bits | ~0 ms |
| GPU `engine:matmul*` at s = 2056 (no pad) | same as HD₃-at-pow2 | ~7 700 ms |
| `tee:attn_cached`, residuals, other | unchanged | ~3 000 ms |
| **TTFT predicted at 4B n=2048** | — | **~20 300 ms** |

Comparison:
- Current HD₃ at n=2048 (pad → 4096): TTFT 32 000 ms
- Haar at n=2048: TTFT 25 900 ms
- **DCT-IV cascade at n=2048: ~20 300 ms** — saves ~5.6 s vs Haar, ~11.7 s vs current HD₃-padded
- HD₃ at n=2040 (pow2-aligned, after unapply fix): 16 600 ms

So DCT-IV cascade at n=2048 is **slower than HD₃-at-pow2 by ~3.7 s but
faster than Haar by ~5.6 s**. It closes the gap. It cannot beat HD₃ at
pow2 because the inner transform is intrinsically ~4× slower than FWHT
(Bluestein overhead for non-radix-2 sizes).

Security argument:
- **Orthogonality**: exact at any n.
- **Incoherence**: `max|C^IV[i,j]| = √(2/n) = O(1/√n)`, same bound as
  Hadamard rows. The QuIP# incoherence proof transfers.
- **BSS hardness**: this is the open question, but DCT-IV's row
  structure has favourable properties absent in DCT-II:
  - No constant row → no row-sum leak.
  - Rows are pairwise orthogonal cosine sequences with co-prime
    frequency relations → not concentrated in any low-dimensional
    subspace.
  - The sign-cascade `D_k · C^IV · D_{k-1} · C^IV · ... · D_1 · C^IV`
    creates a depth-k mixing that's structurally equivalent to
    HD_k except for the inner transform basis.
  - Per QuIP# / QuaRot analyses, the load-bearing property is
    "noise-like row distribution after `D · M` for random `D`". DCT-IV
    rows × random ±1 signs give a uniform-bounded sequence with mean
    zero and entries `±√(2/n) · cos(known frequency)`. This is
    "structured noise" — not as noise-like as pure ±1/√n Hadamard
    rows, but the structure is in a public basis that the attacker can
    project out only by also recovering the signs.

- **Multi-anchor attack on DCT-IV cascade**: same analysis as
  monolithic HD₃ — k anchored rows give k·d constraints in 3n sign
  bits. At our shapes (d ≈ n) the system is under-determined with
  k=1 anchor and only weakly over-determined with k=2 anchors.
  Monolithic HD₃ is safe because the sign cascade's nonlinearity
  resists structured solving even when over-determined; DCT-IV
  inherits the same resistance because the structure is identical at
  the cascade level. **DCT-IV cascade is NOT vulnerable to the
  Option-C multi-anchor attack** because there are no separate
  per-block sign cascades to solve — only the monolithic 3·s bits.

Implementation surface:
- New `MaskFamily::Dct4Cascade` variant.
- DCT-IV kernel:
  - Reference: `rustfft` + Bluestein wrapper.
  - SIMD-optimised: split-radix real DFT with sin/cos pre-tables.
  - Estimated 3-4 weeks of focused implementation including SIMD.
- Attack-suite extension: rerun §4.3 attacks at `c5_dct4` condition.
- Estimated 2-3 weeks of attack runs + analysis.

**Verdict.** Promote to Phase 2.

### 6.3 New Option L — Generalised orthogonal cascade framing

The HD₃ → DCT-IV → "any-fast-orthogonal" generalisation is worth
naming explicitly because it constrains future research:

```
A = D_k · M · D_{k-1} · M · ... · D_1 · M
```

where:
- `M` is any fixed, public, exactly-orthogonal matrix at order `s`
  with a fast (sub-quadratic) multiplication algorithm.
- Each `D_i` is a fresh `(s × s)` random sign diagonal sampled per
  forward.
- Depth `k ≥ 3` (paper baseline; QuIP# uses 3).

The full design space:

| `M` choice | order constraint | apply cost | security baseline |
|---|---|---:|---|
| Sylvester Hadamard | pow2 | `O(s log s)` | HD₃, **already shipped** |
| DCT-II | arbitrary | `O(s log s)` | Has constant-row leak; ruled out |
| **DCT-IV** | **arbitrary** | **`O(s log s)`** | **Best non-pow2 candidate** |
| DST-IV | arbitrary | `O(s log s)` | Variant of DCT-IV; structurally identical |
| Real DFT (cosine-sine basis) | arbitrary | `O(s log s)` | Complex bookkeeping; same incoherence |
| Cayley-rotation from random skew-symmetric | arbitrary | `O(s²)` (dense) | Slower; not worth pursuing |
| Householder cascade (depth-k Householders) | arbitrary | `O(k·s²)` (dense) | k = O(s) for full mixing → not fast |
| Butterfly / Monarch (Dao 2022) | arbitrary | `O(s log s)` | Public sparsity pattern → BSS-vulnerable (round-3 doc §2.2) |

Among fast-multipliable orthogonals, **DCT-IV is the only published
real-valued option at arbitrary n that preserves incoherence**.

The framing matters because if DCT-IV's attack suite fails, the search
space for replacements is narrow (DST-IV is the obvious next try).

### 6.4 Revised phased plan (supersedes §3)

#### Phase 1 — Ship Option A (hybrid auto-dispatch)

Unchanged from §3. 1-2 days. Default-on after attack suite clears.
**Primary deliverable.**

#### Phase 2 — DCT-IV cascade (was: block-diagonal HD₃)

**Effort.** 3-4 weeks impl + 2-3 weeks attack suite ≈ 5-7 weeks total.

**Deliverable.**
- `crates/gelo-protocol/src/dct4.rs` — `Dct4Mask` type + Bluestein-based
  DCT-IV kernel.
- `MaskFamily::Dct4Cascade` variant, wired through `sim.rs`.
- Unit tests: orthogonality, round-trip preservation, deterministic
  seeding, incoherence-bound check.
- Attack-suite condition `c5_dct4` in `evals/aloepri-attacks/`,
  mirroring `c3_hd3`.
- Bench cell `gpu_gelo_dct4` in `qwen3_long_context_bench.rs`.
- If attack suite clears: wire DCT-IV cascade as the non-pow2 branch
  of `MaskKind::Auto` (replacing the Haar fallback).
- If attack suite fails: file at `docs/research/future-rnd.md`,
  consider DST-IV as next try; default `MaskKind::Auto` keeps Haar
  fallback.

**Expected payoff.** Closes the n=2048 gap. Average TTFT over a
mixed-shape workload at 4B drops from ~21 s (Option A: 50/50
HD₃-at-pow2 + Haar-at-non-pow2) to ~18-19 s (DCT-IV at non-pow2,
HD₃ at pow2). ~10-15 % additional win on top of Option A.

#### Phase 3 — Optional: cascade-depth tuning

Once DCT-IV cascade lands, audit whether `k=3` is optimal. Higher `k`
(e.g., `k=5`) increases sign-bit entropy from `3s` to `5s` bits,
strengthening multi-anchor resistance. Cost: `(k/3)×` apply/unapply.
Likely net-positive at k=4 (8 % more cost, meaningful security
margin); investigate if attack suite shows marginal residual leak at
k=3.

### 6.5 Why Option C was an attractive trap

The original §2 ranked block-diagonal as Phase 2 because:
- The CPU/GPU cost story is correct (no pad regression at non-pow2).
- Block boundaries are a public input, so "cross-block leakage" feels
  like a theoretical-only concern.
- The Pareto-front search ("cheapest primitive that preserves
  orthogonality") returns block-diagonal as the answer.

What the original missed:
- The per-block sign-cascade entropy `3·b_i` is dimensional, not
  asymptotic. At our shapes `d ≈ b_i`, just 2 anchored data rows
  break the block via linearisation.
- System-prompt anchored rows are realistic and ubiquitous.

This is a recurring pattern in mask-design: cheap primitives that
"factor" the orthogonality across structural boundaries are
broken by adversaries with side-information about those boundaries.
The round-3 doc §2.2 ruled out banded / butterfly / circulant for
similar reasons; block-diagonal HD₃ falls into the same category.

The principled position going forward: **only consider primitives
that preserve full s-dimensional mixing**. That eliminates
block-diagonal, banded, sub-pow2-projection, etc. It leaves: dense
Haar (current Haar fallback), HD₃ at pow2, and arbitrary-order
fast orthogonal cascades (DCT-IV, DST-IV, real DFT).
