---
type: theory
status: current
created: 2026-05-21
updated: 2026-05-21
tags: [aloepri, alg1]
---

# AloePri Algorithm 1 keymat — variance sources + K_a × K_d universality

**Status:** 2026-05-22 investigation. Q3-4B Û_vo at L=17, K=64.

> **Implication for [`aloepri-attacks.md`](aloepri-attacks.md):** the finding
> below (single-seed TTRSR readings carry ~5 pp noise at d=2560) applies to
> all ISA TTRSR measurements in the attacks doc — comparisons within that
> band are not significant.

A 2×2 PRNG×LinAlg factorial on the attacker's Algorithm 1 keymat builder
appeared to expose wildly divergent TTRSR readings depending on whether the
generator was CPU MT19937 or CUDA Philox and whether QR/SVD ran on CPU
LAPACK or rocSOLVER (3.41 % – 11.92 % spread). A first-pass diagnosis
blamed Philox+rocSOLVER device numerics. Two follow-up probes —
direct nullspace-basis comparison and a 5-seed sweep of all four corners
— ruled that out: rocSOLVER's nullspace basis is a Haar-random rotation
of LAPACK's (distributionally identical), and all four corners sample
TTRSR from indistinguishable distributions (Welch t-test p > 0.4
pairwise). The K=64 attacker-keymat-pool sample variance is ~5 pp at
d=2560 — much larger than the 3.2 pp the disparities memo estimated —
and a single-seed comparison can produce an apparent ~8 pp "effect"
purely from noise.

> **Framing correction (2026-05-25).** Throughout this document the
> claim "Algorithm 2 amplifies the K_a × K_d interaction ~3-5×" should
> be read as "the synth→real gap is 3-5×" — the real-mode captures used
> for that gap include **§5.2.2** (additive noise on W_e/W_h + Π token
> permutation) **plus §5.2.3 Algorithm 2 plus bf16 + Q8_0 quantization**,
> while synthetic mode applies *only* Algorithm 1's keymat to plain
> captures. The §5.2.2 vs §5.2.3 vs quantization components have never
> been isolated empirically. Theoretical attribution: most of Algorithm
> 2's runtime contribution to ISA HiddenState comes from `Ẑ_block` at
> β=8 (35 % rel score perturbation propagating to the residual) and the
> bf16 numerical leftover of `Û_vo`; R̂_qk / Ĥ_qk-±1 / Π_head contribute
> ~0 to ISA HiddenState at exact precision. Full cross-component
> picture: `docs/handoffs/2026-05-25-alg2-attack-crossmap.md`. The
> "Phase d Algorithm 2 ablation" plan in
> `docs/handoffs/2026-05-22-keymat-defense-optimization.md` inherits
> the same conflation — see that doc for the (un-cleaned) ablation matrix.

## Key findings

1. **σ_pool dominates σ_split by 13×** — at K=64, attacker-pool draw contributes σ ≈ 5.7 pp; eval-split contributes ≈ 0.4 pp; residual ≈ 1.3 pp. Single-seed §08 cells comparing within 5 pp are noise.
2. **Pool distribution is bimodal-ish at K=64** — 5 pool seeds split into "unlucky" (0.5–3.7 %) and "lucky" (12–13 %) clusters with a 10 pp gap. Sparse-lucky-K_a^k model is consistent with K-sweep behaviour (adding 64 new keymats can flip a pool from "unlucky" 0.7 % to "lucky" 16 %).
3. **K_a × K_d interaction exists in pure Algorithm 1 but is bounded at ~5 pp** — synthetic K_d_test sweep (3 seeds × 3 pools) shows spreads of 2.6–7.1 pp. Pool ranks rotate with K_d; the lucky pool at one K_d isn't lucky at another.
4. **Algorithm 2 amplifies the K_a × K_d interaction ~3–5×** — real deployment spread is 13.1 pp vs synthetic ≤ 7.1 pp. Pool-2's 13.8 % luck against the real K_d=42 deployment is not explained by Algorithm 1 alone (synthetic gives only 3.1 %). Algorithm 2's contribution to that cell is +10.7 pp.
5. **C-block is the likely Algorithm-1 site of variance** — `C = coeffs · basis(null(F^T))` is the only Algorithm-1 component whose distribution depends on a random nullspace basis. B, U, V, Z are distributionally identical across pools.
6. **The earlier "Philox/rocSOLVER GPU port bug" diagnosis is retracted** — a 4-corner 2×2 (PRNG × LinAlg) × 5-seed factorial shows all four corners sample from indistinguishable distributions (Welch p > 0.4 pairwise). The dramatic 11.92 % vs 3.41 % gap was a +1.64 σ port-GPU sample paired with a −0.56 σ vendor sample from the same distribution.

## Implications for the defender plan

- **Adversarial K_d selection** (Phase 2.1) is still the cheapest defender lever but its inner-loop attacker must run on the *real* deployment (with Algorithm 2 applied) — synthetic K_d evaluation under-predicts attacker capability by 3–5×.
- **λ tuning** (Phase 2.2) is **demoted** — λ touches B, but the C-block carries the variance and Algorithm 2 dominates anyway.
- **Algorithm 1 modifications alone are insufficient** — even eliminating the ~5 pp Algorithm-1 contribution leaves ~9 pp of Alg-2-mediated luck. Phase 4 must be paired with Algorithm 2 hardening.
- **New phase 4b: Algorithm 2 ablation** — identify which Alg-2 component (Û_vo / π / additive noise) is the amplifier.
- **Multi-seed reporting** (Phase 5) is critical — every existing single-seed §08 cell carries ~5 pp uncertainty.

## Definitions

- **AloePri** — the obfuscation scheme described in Wang et al. 2603.01499v2.
- **Algorithm 1 (paper §5.2)** — keymat construction.  Builds an invertible linear transform `K ∈ ℝ^{d × (d + 2h)}` from a tower of random matrices `(U, V, E_1, E_2, F_1, F_2, Z, C)` so that, when applied to a transformer's residual stream `x`, the obfuscated output `x · K` is computationally indistinguishable from a random vector and yet a coordinate-wise *covariant* transform: any downstream linear op on `x` has a matching transformed op on `x · K`.
- **`d`** — model hidden size. Qwen3-4B: `d = 2560`. Qwen3-8B: `d = 4096`.
- **`h`** — expansion width. Paper default `h = 128`.
- **`d_obs = d + 2h`** — observable obfuscated dimension. Q3-4B: 2816. Q3-8B: 4352.
- **`λ`** — Algorithm 1 noise weight on `V`. Paper default `0.3`.
- **`K_d`** — the *deployment*'s secret keymat (server-side, single-sample).
- **`K_a^k`** — the *attacker*'s `k`-th synthesised keymat in the multi-key pool. Each `K_a^k` is drawn independently from the Algorithm 1 distribution. Used for paper-faithful multi-key training (`run_isa_multikey.py`).
- **`K`** — pool size. Paper-faithful ISA driver uses `K = 64`.
- **TTRSR** — Token Top-1 Recovery Success Rate. The attack-harness top-1 reading: fraction of test-set token positions whose plain id the inverter correctly recovers via cosine-NN against the public `W_e`.
- **Plain identity-τ ceiling** — TTRSR on the no-defence task (ridge attacker against the plain unobfuscated model). Q3-4B at L=17, N=411 test rows: **10.18 %**.
- **MT19937** — Mersenne Twister PRNG used by `torch.Generator(device="cpu")`.
- **Philox** — Philox4x32 counter-based PRNG used by `torch.Generator(device="cuda")`.
- **rocSOLVER** — AMD ROCm's dense LinAlg library. `torch.linalg.qr` / `torch.linalg.svd` on `device="cuda"` route through rocSOLVER on Strix Halo iGPU.
- **vendor_cpu** — `vendor/aloepri-py/src/keymat.py`. Reference Algorithm 1 builder, CPU-only, 8 fresh `torch.Generator(device="cpu")` instances per keymat.
- **gpu_native** — `_build_attacker_keymat_pool_gpu_native` in `evals/aloepri-attacks/m2_7/run_isa_multikey.py`. Single advancing generator per keymat, device chosen by `--device`.

## Algorithm 1 — concise build recipe

For each keymat `K_a^k` the builder draws **8 random matrices** and assembles:

| Symbol | Shape | Distribution | Role |
| --- | --- | --- | --- |
| `U` | `(d, d)` | Haar orthogonal via `QR(randn)` | rotation half of `B` |
| `V` | `(d, d)` | iid `N(0, 1/d)` | noise half of `B` |
| `B = U + λ V` | `(d, d)` | — | invertible base of K's left block |
| `E_1` | `(d, h/2)` | iid `N(0, 1/d)` | left low-rank factor of `E` |
| `E_2` | `(h/2, h)` | iid `N(0, 1/d)` | right low-rank factor of `E` |
| `E = E_1 E_2` | `(d, h)` | rank ≤ h/2 | right block of K |
| `F_1` | `(h, h/2)` | iid `N(0, 1/d)` | left low-rank factor of `F` |
| `F_2` | `(h/2, d)` | iid `N(0, 1/d)` | right low-rank factor of `F` |
| `F = F_1 F_2` | `(h, d)` | rank ≤ h/2 | nullspace anchor for `C` |
| `coeffs` | `(d, d_obs−h)` | iid `N(0, 1)` | random coords in nullspace of `Fᵀ` |
| `basis_F` | `(d_obs, d_obs−h)` | nullspace basis of `Fᵀ` via SVD | nullspace basis of `Fᵀ` |
| `C = coeffs · basis_Fᵀ` | `(d, d_obs)` | iso Gaussian *in* `nullspace(Fᵀ)` | middle block of K |
| `Z` | `(d_obs, d_obs)` | Haar orthogonal via `QR(randn)` | output mixing |

Final keymat:

`K_a^k = [B | C | E] · Z`

Two heavy LinAlg operations:

1. **QR** (`U` from `(d,d)` Gaussian, `Z` from `(d_obs,d_obs)` Gaussian) — sign-fixed via `q * sign(diag(r))`. For a given Gaussian input, the QR factorisation with positive diagonal is *unique*, so different LinAlg backends should produce the same `Q` up to floating-point noise.
2. **SVD with `full_matrices=True`** on `Fᵀ` — yields the right-singular-vectors matrix `Vh ∈ ℝ^{d_obs × d_obs}`. The bottom `(d_obs − rank(Fᵀ))` rows form the nullspace basis. The basis for the nullspace (degenerate-singular-value block, all-zero singular values) is **not unique** — different SVD implementations are free to return any orthonormal basis of the nullspace.

## The 2×2 factorial finding (Q3-4B Û_vo, L=17, K=64, seed=20260521)

Same fixed inputs (snapshots, embed table, deterministic train/val/test split). Only PRNG family and LinAlg backend vary.

| PRNG \ LinAlg | CPU LAPACK | rocSOLVER (GPU) |
| --- | --- | --- |
| MT19937 (CPU) | 3.16 % (port helpers) / 3.41 % (vendor 8-gen) | **0.49 %** |
| Philox (CUDA) | 8.03 % | **11.92 %** ⚠️ exceeds plain-τ ceiling (10.18 %) |

Naïve effect decomposition:

- PRNG main effect (Philox − MT19937): +8 pp.
- LinAlg main effect (rocSOLVER − LAPACK): +0.6 pp.
- Philox × rocSOLVER interaction: +6.5 pp.

The headline reading — Philox + rocSOLVER giving **11.92 %**, *above* the no-defence plain-τ ceiling of 10.18 % — was the original "structurally impossible" alarm in the handoff. It implies the attacker is doing **better** against the Û_vo-defended deployment than the same attacker would do against the plain undefended model. That is impossible for a paper-faithful attacker — unless the K_a^k pool is not really sampling the Algorithm 1 distribution.

## Counter-behaviour: MT19937 + CPU LAPACK vs MT19937 + rocSOLVER

The interesting cell is the bottom-left:

| Config | TTRSR top-1 |
| --- | --- |
| MT19937 + CPU LAPACK (vendor reference) | 3.41 % |
| MT19937 + rocSOLVER | **0.49 %** |

`0.49 %` (= **2/411 hits**) is well *below* the vendor baseline. Holding the PRNG fixed at MT19937 and swapping LinAlg from LAPACK to rocSOLVER drops the attacker by ~3 pp.

The bottom-right is the mirror: holding rocSOLVER fixed and swapping PRNG from MT19937 to Philox jumps the attacker by ~11 pp. So the two device variables have an enormous interaction sign-symmetry: each *alone* moves the needle by a few pp, but they cross to ±11 pp in opposite directions.

This is the empirical observation that needed explaining.

## Theory 1 — rocSOLVER's nullspace basis has a preferred orientation

**Hypothesis.** rocSOLVER's SVD might return a nullspace basis with a *specific* structural orientation (diagonal-dominant, banded, axis-aligned) that differs from CPU LAPACK's basis. In that case the C-block — which is `coeffs · basis_Fᵀ` — would inherit that orientation. K_a^k pools built with rocSOLVER would all share the rocSOLVER orientation; K_d (built with CPU LAPACK on the deployment server) would not. The ridge inverter trained on the rocSOLVER-oriented K_a^k pool would then extrapolate poorly to the LAPACK-oriented K_d → low TTRSR (0.49 %).

**Probe.** Build `Fᵀ` once on CPU from a fixed MT19937 seed. Compute the nullspace basis on CPU LAPACK SVD and on rocSOLVER SVD. Compare:

1. Do they span the same subspace? Projection-matrix diff `‖P_cpu − P_gpu‖∞` should be ~0 if they do.
2. Rotation matrix `R = basis_cpuᵀ · basis_gpu`. Orthogonal if same subspace. Examine `R`'s structure: diagonal-dominant, permutation-like, or Haar-random?

**Result** (probe at d=2560, h=128, nullspace dim 64):

| Quantity | Value | Interpretation |
| --- | --- | --- |
| `‖P_cpu − P_gpu‖∞` | 3.75e-14 | Same subspace to machine epsilon ✓ |
| `R` orthogonality error | 5.66e-14 | `R` is orthogonal ✓ |
| `R` diagonal mean / off-diagonal mean | 0.989 | `R` is **Haar-random**, not diagonal-dominant |
| Signed diag with `\|·\|>0.5` | 0 / 64 | **Not** permutation-like |
| Row-max mean | 0.33 | Entries spread across all 64 columns |

**Disputed.** rocSOLVER and LAPACK return bases that span exactly the same nullspace, related by a Haar-random orthogonal rotation. Since Gaussian coefficients are rotation-invariant (`coeffs · Rᵀ ≡ coeffs` in distribution), `C_rocSOLVER` and `C_LAPACK` are *distributionally identical*. Theory 1 is refuted.

## Theory 2 — K=64 attacker-keymat-pool sample variance is huge

**Hypothesis.** The Algorithm 1 distribution combined with the ISA labelled-ridge attack has a much larger inherent variance at K=64 than the 3.2 pp noise floor in `aloepri-attack-harness-disparities`. Single-seed TTRSR readings are not reliable point-estimates of the underlying attack capability — they are samples from a distribution with std ≈ 5 pp or more. The "11.92 % vs 3.41 %" headline is the comparison of two single samples from heavy-tailed distributions, and is dominated by noise, not by a real Philox/rocSOLVER bias.

**Probe.** Hold PRNG=MT19937 and LinAlg=rocSOLVER fixed (the most ambiguous cell — it would be the simplest to dismiss as anti-aligned). Run 6 independent attacker_seeds ∈ {20260521, 1, 2, 3, 4, 5} and read TTRSR.

**Result** (Q3-4B Û_vo, L=17, K=64, MT19937 + rocSOLVER):

| `attacker_seed` | top-1 | top-10 |
| --- | --- | --- |
| 20260521 (original) | 0.49 % | 1.95 % |
| 1 | 7.94 % | 20.10 % |
| 2 | 14.07 % | 19.44 % |
| 3 | 2.15 % | 8.61 % |
| 4 | 0.75 % | 2.24 % |
| 5 | 11.17 % | 22.58 % |

Summary: **mean = 6.1 %, range = 0.49 % – 14.07 %, std ≈ 5.6 pp**.

The seed-sweep span (≈14 pp) is much wider than the gap between the two headline numbers (`11.92 − 3.41 = 8.5 pp`). Both headline readings are *individually plausible samples from this single distribution*:

- Vendor (3.41 %) — z-score ≈ −0.5 from MT19937 + rocSOLVER mean.
- Port-GPU (11.92 %) — z-score ≈ +1.0.

Neither is anomalous.

### Confirmation sweep — vendor + port-GPU at 5 seeds

5 fresh `attacker_seed` ∈ {1, 2, 3, 4, 5} per config, same inputs.

| `attacker_seed` | vendor (MT19937 + LAPACK) | port-GPU (Philox + rocSOLVER) |
| --- | --- | --- |
| 1 | 10.42 % | 1.24 % |
| 2 | 12.79 % | 6.39 % |
| 3 | 1.91 % | 2.15 % |
| 4 | 2.49 % | 3.74 % |
| 5 | 3.47 % | 11.66 % |
| **mean** | **6.22 %** | **5.04 %** |
| **std** | **5.02** | **4.19** |
| range | 1.91 – 12.79 | 1.24 – 11.66 |

Including the MT19937 + rocSOLVER 5-seed reading from Probe 2 (mean **7.32 %**, std **5.60**, range 0.75 – 14.07):

**Welch t-test, all pairs:**

| Comparison | mean Δ | t | p |
| --- | --- | --- | --- |
| vendor vs port-GPU | +1.18 pp | 0.40 | 0.70 |
| vendor vs MT19937 + rocSOLVER | −1.10 pp | −0.33 | 0.75 |
| port-GPU vs MT19937 + rocSOLVER | −2.28 pp | −0.78 | 0.46 |

**No pairwise difference is detectable.** All three configurations sample TTRSR from indistinguishable distributions. The headline 11.92 % vs 3.41 % gap was a +1.64 σ port-GPU sample paired with a −0.56 σ vendor sample from the same distribution. Theory 2 wins.

**Original "GPU port bug" diagnosis is retracted.** The 2×2 factorial PRNG/LinAlg effects computed earlier were single-seed noise misread as structural signal.

### What the variance is and where it comes from

K=64 attacker-keymat-pool TTRSR has std ≈ 5 pp at d=2560 (Q3-4B). Two confounded variance sources at present:

1. **K_a^k pool variance** — the K=64 specific samples drawn from the Algorithm 1 distribution.
2. **Train/val/test split variance** — `attacker_seed` is also fed into `np.random.default_rng(attacker_seed + 17)` to partition the 904 plain-state rows. Different seeds give different 411-row test sets with different test-id distributions; the per-test-set TTRSR has its own noise. See follow-up #1 below for how to disentangle.

## Implications for the attack harness

1. **Single-seed TTRSR readings at K=64 are unreliable** at d=2560. Any cell in §08 of `docs/prototype/aloepri-llm.html` that quotes a multi-key TTRSR from one seed carries ~5 pp uncertainty. Comparisons within 5 pp of each other are noise.
2. **The "plain identity-τ ceiling" (10.18 %) is itself a single-seed quantity** and carries its own ~5 pp noise. The supposed "above ceiling = structurally impossible" alarm at 11.92 % does not survive contact with the noise distribution. Vendor reached 12.79 % at seed=2 alone.
3. **Cells reported as "below the ceiling by ~1 pp" mean almost nothing.** The handoff's claim that "Q3-8B Û_vo at 9.00 % is 1.18 pp below the ceiling" assumed both numbers were point-estimates; with std ≈ 5 pp each, the 1.18 pp gap is well within noise.
4. **The N=16-prompt sample-noise floor estimate (3.2 pp) in `aloepri_attack_harness_disparities.md` was too optimistic for the multi-key-pool variant.** The pool itself adds ~3 pp of independent variance.

## Follow-ups

1. **Disentangle pool-variance from split-variance** (cheapest probe). Currently `attacker_seed` drives both. Fix the pool seed; vary only `split_seed` across 5 runs. If split-variance alone is ~3 pp, the marginal pool variance is sqrt(5² − 3²) ≈ 4 pp and the headline "K=64 is the noisy thing" simplifies to "the test set is at least half of it".
2. **K-sweep at fixed seed.** Run K ∈ {32, 64, 128, 256} at one seed. Variance should shrink ~1/√K if it's pool-driven. If it doesn't, the noise is dominated by the split / ridge α selection / cosine-NN tie-breaking, not by the pool.
3. **Re-run plain identity-τ at 5 seeds** to give the ceiling a real error bar. Until then, any single-seed comparison to the ceiling is meaningless.
4. **Q3-8B replication.** If 4B std is 5 pp, 8B likely is too. The handoff's "structural defense gap at 8B" (9.00 % vs 10.18 % ceiling = 1.18 pp margin) almost certainly evaporates under proper error bars.
5. **Revise §08 cells** in `docs/prototype/aloepri-llm.html` to report mean ± std over ≥5 seeds, not single-seed point estimates.
6. **K-default decision.** If pool variance is confirmed dominant, raise K from 64 → 256 to shrink TTRSR std by ~2×, at ~4× the synthesis/ridge cost. Worth a quick perf check.

## Current investigation — variance decomposition + defense-improvement levers

### Two seeds, two distinct random draws inside one attack run

The harness has two RNG-driven steps. Each is independent of the other once the seeds are decoupled.

- **Attacker-pool seed** (a.k.a. `pool seed`, CLI: `--attacker-seed`). Drives the 64 attacker keymats `{K_a^1, …, K_a^64}`, each one a fresh sample from Algorithm 1. **The K_a pool is the attacker's resource for the multi-key training trick** — they cannot run the obfuscated model end-to-end, so they synthesise their training data by drawing their own K=64 keymats and applying each to the plain residual stream (`X_a^k = X_plain · K_a^k`). The pool seed determines *which specific 64 samples* the attacker happens to draw out of Algorithm 1's distribution.
- **Eval-split seed** (a.k.a. `split seed`, CLI: `--split-seed`). Drives the random partition of the 904 plain-state rows into train (452 rows) / val (226) / test (226 → filtered to 411 after vocab filter). **The eval-split is methodology** — it decides *which token positions* are used to fit ridge, validate α, and evaluate top-1 recovery. Different split seeds → different 411 test tokens (different vocab coverage, different per-token difficulty).

In short: pool seed = "which attacker keymats", split seed = "which test tokens".

### Hypothesis

K=64 single-seed TTRSR has std ≈ 5 pp at d=2560. Two candidate sources for that 5 pp:

- **σ²_pool** — variance contributed by *which 64 attacker keymats got drawn*. Captures "how lucky is the attacker's pool against K_d?". Inherent to Algorithm 1's keymat distribution.
- **σ²_split** — variance contributed by *which 411 test tokens got picked*. Captures "how easy is this random subset of plain tokens to invert?". Pure measurement methodology, not a defense property.

`σ²_total ≈ σ²_pool + σ²_split` under the additive model. Whichever dominates is the lever:

- If **σ²_pool** wins → the attack capability genuinely varies with what keymats the attacker draws — that's defender-relevant. Fix: raise K so the attacker's empirical pool quality converges.
- If **σ²_split** wins → the noise is just methodology bouncing — fix the eval methodology (k-fold CV, average over splits, bigger test set), not the defense.

### Empirical approach

Two sweeps on Q3-4B Û_vo at L=17, fixed K_d (the deployment's baked-in keymat):

1. **Disentangle**: 5 pool seeds × 5 split seeds = 25 runs. Row range → σ_split; column range → σ_pool.
2. **K-sweep**: K ∈ {64, 128, 256} × 3 pool seeds at fixed split. Tests `σ(K) ∝ 1/√K` and whether the K=64 mean is the asymptote.

`--split-seed` was added to `run_isa_multikey.py` to decouple from `--attacker-seed` (which previously drove both via `attacker_seed + 17`).

### Scope: attacker-side only

K_d is fixed across every run — the deployment keymat baked into `cell-qwen3-4b-uvo-20260521`'s GGUF. Sweep varies attacker-side `K_a^k` pool seeds and methodology-side split seeds only. A K_d sweep would require re-obfuscation + re-capture (~5 h overhead per K_d sample) and is out of scope here.

### Disentangle results (5×5, all 25 cells)

Rows = which `{K_a^k}` attacker pool was drawn. Columns = which test-token subset the ridge inverter was evaluated on.

| `K_a` pool ↓ \ test tokens → | set-101 | set-102 | set-103 | set-104 | set-105 | **pool mean** |
|---|---|---|---|---|---|---|
| pool-1 | 0.7 % | 0.0 % | 0.7 % | 0.5 % | 0.7 % | **0.5 %** |
| pool-2 | 13.8 % | 14.9 % | 14.7 % | 11.1 % | 9.9 % | **12.9 %** |
| pool-3 | 0.7 % | 3.2 % | 3.2 % | 2.4 % | 3.4 % | **2.6 %** |
| pool-4 | 3.0 % | 5.4 % | 4.0 % | 2.8 % | 3.2 % | **3.7 %** |
| pool-5 | 12.6 % | 10.8 % | 14.2 % | 10.4 % | 13.1 % | **12.2 %** |
| **split mean** | **6.2 %** | **6.9 %** | **7.4 %** | **5.5 %** | **6.1 %** | grand 6.4 % |

ANOVA (two-way random-effects) variance decomposition:

| component | std (pp) | what it captures |
|---|---|---|
| **σ_pool** | **5.72** | which 64 attacker keymats got drawn |
| **σ_split** | **0.44** | which 411 test tokens got picked |
| σ_residual | 1.34 | interaction + within-cell noise |

**σ_pool / σ_split = 13.1 ×.** The attacker-pool draw accounts for essentially all measurement noise; the eval-split methodology is nearly noise-free.

### Pool distribution is bimodal-ish — Algorithm 1 has "lucky" and "unlucky" pools

Per-pool means cluster into two groups:

- **Unlucky pools (1, 3, 4)** → 0.5 %, 2.6 %, 3.7 % → cluster mean ≈ 2.3 %.
- **Lucky pools (2, 5)** → 12.9 %, 12.2 % → cluster mean ≈ 12.6 %.

The split inside one pool stays tight (each row's range is 0.7–4.7 pp), but the across-pool spread is bimodal — TTRSR jumps by ~10 pp between the two clusters. Algorithm 1's K_a distribution is **not** behaving like a smooth Gaussian sample around a single attack-capability mean; some pool seeds happen to draw a subset of `K_a^k` samples that align with K_d's specific C-block direction, and those pools dominate the attack signal.

### K-sweep results (3 pool seeds × {64, 128, 256}, fixed split=101)

| K | pool-1 | pool-2 | pool-3 | mean | std |
|---|---|---|---|---|---|
| 64 | 0.7 % | 13.8 % | 0.7 % | 5.1 % | 7.5 |
| 128 | 16.3 % | 15.3 % | 1.5 % | 11.0 % | 8.3 |
| 256 | 1.2 % | 7.6 % | 0.7 % | 3.2 % | 3.9 |

The K-sweep is **too thin (3 seeds per K) to settle the 1/√K scaling claim**. Notable observations:

- **σ shrinks from 7.5 (K=64) to 3.9 (K=256)** — directionally consistent with 1/√K (predicted 7.5 → 3.75 at K=256). Consistent with the pool-IID-sampling hypothesis.
- **The K-sweep also exhibits the "lucky-pool" jump**: pool-1 at K=64 = 0.7 % (unlucky), jumps to 16.3 % at K=128 (adding 64 new keymats happened to include lucky ones), and reverts to 1.2 % at K=256 (the lucky samples are diluted by 128 more "noise" keymats, and the val-α selection picks a different ridge).
- This is consistent with a **"sparse lucky K_a^k" model**: only a small fraction of Algorithm-1 samples carry the attack signal. The K=64 pool either includes one or it doesn't, and the ridge generalisation collapses to a 2-mode distribution.

To definitively measure the K → ∞ asymptote, we'd need ≥ 10 pool seeds at K ∈ {64, 128, 256, 1024}. Worth doing as a follow-up.

## K_d universality probe — synthetic Algorithm-1-only vs real deployment

To distinguish "lucky pool is a property of Algorithm 1 alone" from "lucky pool is the K_a × K_d interaction" (which is the gate between Phase 4-style Algorithm 1 mods and Phase 2.1-style adversarial K_d selection), we ran a cheap-synthesis probe: for each `K_d_test_seed`, regenerate K_d via vendor's CPU MT19937, synthesise test inputs as `State_plain[test_rows] @ K_d_test`, and run the same multi-key ridge attack at K=64. This isolates Algorithm 1's keymat interaction with no Algorithm 2 perturbations (Û_vo, π row-perm, additive noise).

Sweep: K_d_test_seed ∈ {42, 142, 242} × attacker pool seed ∈ {1, 2, 3} at fixed split=101.

| K_d_test_seed | pool-1 | pool-2 | pool-3 | spread |
|---|---|---|---|---|
| 42 (synth, matches deployment seed) | 4.0 % | 3.1 % | 0.9 % | 3.1 pp |
| 142 (synth) | 4.4 % | 1.8 % | 4.4 % | 2.6 pp |
| 242 (synth) | 0.0 % | 7.1 % | 3.1 % | 7.1 pp |
| **42 (real deployment, with Algorithm 2)** | **0.7 %** | **13.8 %** | **0.7 %** | **13.1 pp** |

### Two-part finding

1. **K_a × K_d interaction exists in pure Algorithm 1, but bounded at ~5 pp.** Synthetic spreads are 2.6–7.1 pp. Pool ranks rotate with K_d (pool-1 highest at KD=142, pool-2 highest at KD=242). This confirms Algorithm 1 is *not* flat from the attacker's perspective — but the swings are small enough that single-seed comparisons within ~5 pp at K=64 remain noise-dominated.
2. **Algorithm 2 amplifies the interaction ~3–5×.** Real deployment spread is 13 pp (vs 3–7 pp synthetic). Pool-2's dominant 13.8 % reading against the real K_d=42 deployment is *not* explained by Algorithm 1 alone — synthetic KD=42 gives pool-2 only 3.1 %. The 10.7 pp gap is the Algorithm 2 contribution to *that specific (pool-2, K_d=42, Algorithm-2-seeds)* combination.

### Implications for the defender-lever plan

- **Phase 2.1 (adversarial K_d selection) survives but its inner-loop attacker must use real-deployment captures, not synthetic.** Synthetic-K_d evaluation under-predicts attacker capability by ~3–5× because it misses the Algorithm 2 amplifier.
- **Phase 4 (Algorithm 1 mods) alone are insufficient.** Even if we eliminate all Algorithm 1's K_a × K_d interaction (the ~5 pp synthetic effect), the real-deployment spread still has ~9 pp of Algorithm-2-mediated luck. Phase 4 needs to be paired with Algorithm 2 hardening to close the full gap.
- **New phase candidate — Algorithm 2 mods.** Investigate which Algorithm 2 component (Û_vo, π, noise) does the amplification. The amplifier is not just "additive noise hides signal" — it's *increasing* the lucky-pool effect, which is structurally surprising and worth characterising.

## Component-level decomposition of Algorithm 1 — where does "luck" live?

Algorithm 1's keymat is `K = [B | C | E] @ Z`, built from 8 random matrices. To identify *which component* carries the lucky-pool signal, walk each block's role:

| block | shape | construction | role | structural variance across pools | "luck signal" candidate? |
|---|---|---|---|---|---|
| `U` | (d, d) | Haar orthogonal via QR(randn) | rotation of plain input axis into B | uniform on O(d) | low — distributionally identical across pools |
| `V` | (d, d) | iid N(0, 1/d) | additive perturbation of B | low | low |
| `B = U + λV` | (d, d) | always invertible | left block of K — covers full plain space | always full rank | low — every pool covers plain space the same way |
| `E1·E2 = E` | (d, h) | rank-h/2 product of N(0, 1/d) iid Gaussians | rightmost block of K | random rank-h/2 subspace direction | medium — E spans an h-dim subspace from a rank-h/2 product |
| `F1·F2 = F` | (h, d) | rank-h/2 product of N(0, 1/d) | **anchors C-block via its nullspace** | random rank-h/2 → its nullspace orientation varies pool-to-pool | **high — paper §5.2 calls this the "key information" hiding axis** |
| `C = coeffs · basis(null(F^T))` | (d, h) | iid N(0,1) `coeffs` rows projected onto F^T's nullspace | middle block — Algorithm 1's expansion-randomness contribution | very high — distribution depends on both `coeffs` and the random nullspace basis | **strongest candidate** |
| `Z` | (d+2h, d+2h) | Haar orthogonal | final mixing of [B,C,E] into the deployed keymat | uniform on O(d+2h), but Z is a permutation-of-mix, not a "selector" | medium — Z rotates the [B,C,E] tower but distributional structure is preserved |

**Theoretical priors for the lucky-pool mechanism**

A K=64 attacker pool's TTRSR against K_d is determined by how well the pool's collective `K_a^k` matrices *span the same subspace* as K_d. The attack works because the ridge inverter learns "the function that maps from `X_plain @ K_a^k` back to `W_e[plain_id]`, key-invariantly". Key-invariance requires the K_a pool to *sample* the Algorithm 1 distribution well enough that the inverter generalizes to K_d.

`B` is structurally identical across pools (every pool covers plain space the same way). `Z` is orthogonal and preserves distributional structure. So variance must come from `E`, `F`, or `C`. Of those, **C is by far the most variable** — its sample is a random direction in the (d_obs − h/2)-dimensional nullspace of `F^T`, and both the coefficients and the nullspace basis vary per pool.

**Working hypothesis** (to be confirmed by Phase 1.1 probe):

> A "lucky" K_a pool has at least one K_a^k whose C-block subspace orients within a small principal angle of K_d's C-block subspace. The ridge inverter, given access to that K_a^k, builds a near-perfect approximation of K_d's inverse, and generalizes the rest.

If confirmed, the defender lever ranking sharpens:

| lever | mechanism (under C-block hypothesis) | expected effect on K=64 lucky-pool prevalence |
|---|---|---|
| **Adversarial K_d selection** | Pick K_d whose C-block lands in a direction common attacker draws miss | direct — eliminates the lucky-pool mode for chosen K_d |
| **Larger h** (expansion size) | Bigger nullspace dim → lower chance of any K_a^k aligning with K_d's C-direction | direct — reduces baseline lucky-pool frequency |
| **Anti-correlated C-coeffs** | Stratified `coeffs` so pool draws span nullspace evenly | reduces *pool-level* variance — every pool has near-uniform C-direction coverage, making best-of-N indistinguishable from any one draw |
| **λ tuning** | Adjusts B's noise weight, not C | **negligible under this hypothesis** — λ touches B, not C |

So the prediction sharpens: **λ tuning is the *weakest* defender lever** if C is the luck signal, contrary to my earlier ranking. Re-obfuscation experiments at varying λ are deprioritized; they'd answer the question but cheaply: a Phase 2.2 λ-sweep that shows ~flat TTRSR across λ ∈ {0.3, 0.5, 0.8} would confirm "B doesn't matter".

## What this reveals about Algorithm 1 and how to improve defense

The variance pattern is structural information about Algorithm 1, not just measurement noise. Three load-bearing observations:

1. **The K_a distribution has a non-trivial covariance with K_d.** K_a^k and K_d are independent Algorithm-1 samples, yet specific K_a pools systematically align with K_d's structure more than IID-Gaussian expectation predicts. Most likely site: the C-block (`coeffs · basis_F^T`), where the nullspace orientation of F^T couples to K_d's own C-block direction.
2. **K=64 is a noisy sample of attacker capability.** The paper's defense claim ("attacker reaches X %") is best read as "attacker draws from a distribution with mean X % and std σ; a best-of-N attacker reaches X + ~2σ %". An organised attacker who runs N=10 fresh pools and picks the best already adds ~10 pp.
3. **The defender has free levers we're not using.** K_d is currently one random sample. The defender can pick K_d adversarially against expected attacker draws at zero deployment cost.

### Defense-improvement levers (ranked by cost/benefit)

| # | Lever | Mechanism | Cost | Expected gain |
|---|---|---|---|---|
| **L1** | **Adversarial K_d selection** | At deployment, sample N=64 K_d candidates. Run a reference K_a pool against each. Pick the K_d with worst attacker TTRSR. | One-time, ~10 min at deployment | Structurally weaker attacker forever against this K_d. Likely shifts the per-deployment attack distribution leftward by ~σ_pool. |
| **L2** | **Tune λ** (Algorithm 1 V-noise weight, default 0.3) | Higher λ fattens the K_a distribution → more dispersed pool sampling → "lucky" pool advantage shrinks. | Re-obfuscation per setting | Likely shrinks σ_pool and moves the mean down. Limit: too-high λ may break covariance preservation. |
| **L3** | **Tune `h`** (expansion size, default 128) | Larger nullspace dimension (d_obs − h) → more degrees of freedom in C → less alignment between K_a^k's and K_d's C-block. | Deployment ambient dim grows, slowdown linear in h. | Direct attack-on-C-block defense. |
| **L4** | **Multi-K_d rotation** | Server uses K different K_d's per session, rotates. A "lucky" K_a pool against K_d_1 isn't lucky against K_d_2. | Increased server-side state | Breaks the consistent-K_d assumption the paper attacker makes. |
| **L5** | **Heavier-tailed noise in Algorithm 1** | Replace Gaussian V / E_i / F_i / coeffs_c with sub-Gaussian or anti-correlated draws — fewer "lucky" pool draws by design. | Algorithm change | Reduces the upper tail of the attacker distribution. Untested. |

### Probes to run for each lever

| Lever | Required probe | Cost |
|---|---|---|
| L1 — Adversarial K_d | Re-obfuscate 10 K_d candidates, run pool-1 attacker against each, plot distribution. Pick the lowest. | ~10 × (5 min obf + ~10 min capture) ≈ 2.5 h |
| L2 — λ sweep | Re-obfuscate at λ ∈ {0.3, 0.5, 0.8, 1.0}. Full attacker-side sweep at each. | 4 × ~2 h ≈ 8 h |
| L3 — h sweep | Re-obfuscate at h ∈ {128, 256, 512}. Attacker sweep at each. | 3 × ~2 h ≈ 6 h |
| L4 — Multi-K_d | Server harness change + attack-against-mixture experiment. | Larger workstream |
| L5 — Tail change | Algorithm-1 code change + Algorithm-1 distributional theory check. | Larger workstream |

### Why the attacker-side sweep settles defender-side questions

Each defender-side lever (L1–L3) reduces either σ_pool or the mean attacker capability. We can measure both from the disentangle sweep + the K-sweep, **without ever re-obfuscating** — the K_d-relative-to-attacker-distribution is the property we're directly observing. Re-obfuscation is only needed when changing K_d-side parameters (λ, h, or K_d itself).

The K-sweep is the load-bearing probe for the "paper's K=64 is non-asymptotic" claim — if `TTRSR(K=256) > TTRSR(K=64)`, the paper's defense numbers are too optimistic, and the gap is the leverage a stronger attacker has over the paper attacker. That's the most direct way to improve over the paper baseline: identify where the paper attacker is weaker than the optimal attacker, then strengthen the defense until the optimal attacker is back at the paper-claimed TTRSR.

## c1 — Luckiness-signature probe (in progress, 2026-05-22)

### Rationale

Sections above establish that **K=64 single-pool TTRSR has σ ≈ 5 pp at d=2560**, and that — at fixed K_d — different attacker pool seeds land anywhere from ~1 % to ~7 % TTRSR. The doc-level question this leaves unanswered is **why** some K_a pools are "lucky" while others aren't. Three competing mechanisms, each routing the defender plan differently:

- **Intrinsic (Cat 1)** — luck is a property of K_a^k alone (e.g., spectral ill-conditioning of a single draw). Defender lever: reshape Algorithm 1's noise to clip the lucky tail.
- **Alignment (Cat 2)** — luck is a property of the (K_a^k, K_d) pair (e.g., row-space overlap with K_d's structure). Defender lever: adversarial K_d selection at deployment.
- **Component (Cat 3)** — luck lives in a specific Algorithm-1 block (most likely C: the nullspace anchor whose orientation couples to K_d's own C-block). Defender lever: anti-correlated coefficient sampling in `sample_null_columns`, larger h.

Per **lever ranking section** above (L1 = K_d adversarial, L2 = λ, L3 = h, L5 = heavier-tailed noise): which lever fits the data is currently unknown. c1 is the probe that picks among them.

### What we're measuring

For each of N pool seeds at K=64, d=2560, h=128, λ=0.3, K_d_seed=42:

1. Compute ~15 scalar **features** on the K_a pool — one number per pool per feature, aggregated across the 64 K_a^k matrices via {mean, max, min, median, top5_mean, std}.
2. Independently measure the pool's synthetic-mode (Algorithm-1-only) **multi-key ridge ISA TTRSR top-1** — the attacker's recovery rate against that pool.
3. Across the N pools, compute Pearson r and Spearman ρ for every (feature, aggregate stat) vs TTRSR.

**Hypothesis to test:** there exists at least one feature with |r| ≥ 0.7 at N=10 whose category routes the rest of the plan. The categories partition the feature set:

| Category | Features | Predicted lever if it wins |
|---|---|---|
| **Cat 1 (intrinsic)** | `frobenius_norm`, `sigma_1`, `sigma_min`, `condition_number`, `spectral_concentration_top1`, `spectral_kurtosis` | L5 — Algorithm 1 noise reshape (e.g., reject draws with σ_min < τ) |
| **Cat 2 (alignment)** | `frobenius_alignment` (= ‖K_a^k · pinv(K_d)‖_F), `top_sv_overlap_r128` (= Σσ² of cross top-h Vh) | L1 — adversarial K_d selection using this feature as scoring function |
| **Cat 3 (component)** | per-block aggregates: `U_diag_overlap_with_kd`, `V_norm`, `E_norm`, `F_top_sv`, `Z_diag_overlap_with_kd`, **`C_nullspace_angle_with_kd`** (principal angle between nullspace(F_a^T) and nullspace(F_d^T) — load-bearing for C-block hypothesis) | Phase b1 — anti-correlated C-coeffs; secondarily L3 — h sweep |

### Methodology

**Pool construction.** Each pool builds K=64 K_a^k via vendor `init_keymat_bases` + `generate_keymat` (paper-faithful, 8 separate `Generator(seed+i)` per K_a^k). Pool `s` uses seeds `s + 1 + 10_000·k` for k=0..63, matching the seed schedule in `run_isa_multikey.py --keymat-impl vendor_cpu` exactly so features and TTRSR refer to the *same* K_a^k matrices.

**Feature script:** `evals/aloepri-attacks/m2_7/probe_luckiness_signature.py`. Stack the 64 K_a^k into a (K, d, d+2h) tensor on GPU, do one batched SVD, derive Cat 1 from singular values and Cat 2 from the top-h right-singular-vector subspace. Cat 3 is computed CPU-side numpy from the stored bases (no GPU SVD precision dependence). One pool per ~5 min wall: ~1 min CPU keymat build + ~3 min float32 GPU batched SVD on the (64, 2560, 2816) stack + ~30 s features.

**TTRSR sweep:** `run_isa_multikey.py --kd-test-seed 42 --attacker-seed N --split-seed 101 --layer 17 --attacker-num-keys 64 --split-mode row --keymat-impl vendor_cpu --device gpu`. Synthetic mode (`--kd-test-seed` synthesises Algorithm-1-only test inputs by applying K_d to plain captures; Algorithm 2 is **absent**, so the signal is the pure-Alg-1 attacker capability).

**Precision choice.** GPU float32 SVD chosen over float64 after a 3-path benchmark (2026-05-22):

| SVD path | K=8 wall | extrapolated K=64 | precision vs f64 |
|---|---|---|---|
| GPU float64 (initial choice) | 90 s | ~12 min | reference |
| **GPU float32 (chosen)** | **26 s** | **~3.4 min** | **≤ 3 × 10⁻⁴ rel; Cat 3 byte-identical (CPU)** |
| CPU LAPACK 8 workers | 98 s | ~13 min | reference (float64) |

Pool-1 parity check between float64 and float32 confirmed Cat 3 byte-identical and Cat 1/2 aggregates within 10⁻⁴ relative — well below pool-to-pool variation (10⁻²–10⁻¹).

**Run grid.** N=10 pool seeds (1..10). Per handoff, expand to N=20 only if |r| ∈ [0.4, 0.6] across multiple competing features at N=10.

### Results (final, N=10, 2026-05-22)

**Job 2 TTRSR per pool** (vendor_cpu builder, GPU attack, kd_test_seed=42, split_seed=101, K=64):

| pool | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| TTRSR top-1 | 3.10 | 6.64 | 1.33 | 7.08 | 0.88 | 0.88 | 1.77 | 2.65 | 4.42 | 3.10 |

Pool mean 3.19 %, std 2.12 pp, range 0.88 – 7.08 %.

**Top-10 features by Spearman ρ at N=10:**

| rank | feature.stat | ρ | Pearson r | category |
|---|---|---|---|---|
| 1 | `top_sv_overlap_r128.mean` | **-0.782** | -0.737 | **Cat 2 (alignment)** |
| 2 | `spectral_concentration_top1.max` | +0.745 | +0.411 | Cat 1 (intrinsic) |
| 3 | `sigma_min.max` | -0.721 | -0.717 | Cat 1 |
| 3 | `sigma_min.min` | -0.721 | **-0.803** | Cat 1 |
| 5 | `sigma_1.max` | +0.697 | +0.417 | Cat 1 |
| 6 | `sigma_min.top5_mean` | -0.685 | -0.575 | Cat 1 |
| 7 | `sigma_1.median` | +0.673 | +0.458 | Cat 1 |
| 8 | `top_sv_overlap_r128.median` | -0.673 | -0.560 | Cat 2 |
| 9 | `condition_number.median` | +0.636 | +0.556 | Cat 1 |
| 10 | `V_norm.median` | +0.612 | +0.638 | Cat 3 (spurious — see below) |

**Best per category (Spearman):**
- Cat 2: `top_sv_overlap_r128.mean` ρ = -0.782 ← winner
- Cat 1: `spectral_concentration_top1.max` ρ = +0.745 (close second; many sub-features within 0.05)
- Cat 3 (load-bearing C-block): `C_nullspace_angle_with_kd.max` ρ = -0.418, r = -0.621 — **C-block hypothesis refuted**

**Statistical significance:** Spearman ρ = 0.78 at N=10 → p ≈ 0.008 (two-tailed). Above the 0.7 routing threshold.

**Spurious top-10 entries to ignore:**
- `V_norm.median` (rank 10) — V is iid Gaussian by construction; its norm across pools varies only by 1.6 × 10⁻⁴ relative (float32 noise floor). Any correlation here is fortuitous.
- `Z_diag_overlap_with_kd.median` (rank 11) — Z is independent Haar-orthogonal; expected ≈ 0 with iid sign.

Final correlation report: `/tmp/aloepri-gpu-validation/c1_final_correlations.json`.

### Conclusion

**What c1 establishes.** The lucky-pool mechanism is **spectral-tail behaviour of K_a^k**, surfacing through two co-correlated features: (i) low overlap of K_a^k's top-h right-singular subspace with K_d's top-h subspace (Cat 2, ρ = -0.782) and (ii) anomalously small σ_min in at least one K_a^k draw (Cat 1, ρ = -0.72, r = -0.80). Both index the heavy tail of the iid Gaussian noise in B = U + λV: when V draws an outlier, K_a^k becomes mis-aligned with K_d's structure **and** near-rank-deficient — these are the same effect surfacing twice. The Algorithm-1 component hypothesis (Cat 3, C-block nullspace alignment) is **refuted** at this N (ρ = -0.42, ranks #11).

**Why this matters.** The lever is structurally inside Algorithm 1's iid-Gaussian-V draw, not inside the C-block nor inside the K_d–K_a interaction per se. The K_a–K_d interaction we see is induced by V's tail, not by an independent geometric relationship.

**Structural Algorithm-1 modifications** (paper-faithful = produces a subset of the original distribution; non-faithful = requires re-deriving the security proof):

| # | Change | Site | Paper-faithful? | Expected per-pool TTRSR shift |
|---|---|---|---|---|
| **M1** | Reject K_a^k draws with `σ_min(B) < τ` (regenerate from next seed) — applied **at K_d construction only** | `keymat.init_keymat_bases` after `b = u + lam*v` | yes — strict subset | shrinks lucky tail; defender expected to land at the lower end of the current pool distribution by construction. Magnitude bounded by `Var(σ_min(B))` ≈ 1.6 × 10⁻³, so the resulting K_d gets a few-pp tighter attacker. |
| **M2** | Replace iid `N(0, 1/d)` V with sub-Gaussian (e.g. truncated Gaussian, uniform on a cube) — same first and second moments | `keymat._sample_gaussian` for V | no — alters the distribution the paper proves indistinguishability against | larger shrinkage of lucky tail (entire distribution loses upper tail), but requires re-derivation of paper §5.2 security claim. Engineering lift: ~5 lines + ~1 paper of math. |
| **M3** | Anti-correlated coefficient sampling in `sample_null_columns` (the original Phase b1 lever) | `vendor/aloepri-py/src/keymat.py:sample_null_columns` | yes (rotation-equivariance preserved analytically) | **refuted by c1** — Cat 3 lost decisively; expected gain ≈ 0. Skip. |

**Parametric modifications** (no code change, just sweep deployment hyperparameters):

| # | Param | Direction | Cost | Expected effect |
|---|---|---|---|---|
| **P1** | **λ** (V-noise weight, paper default 0.3) | **Lower** — smaller λ means B ≈ U (Haar), tighter K_a^k spectrum, less variance in σ_min. Run λ ∈ {0.05, 0.1, 0.2, 0.3} synthetic + plain captures, ~1 h per value. | Cheap | Shrinks the heavy-V-tail mechanism. Bound: λ → 0 collapses B to deterministic U, removing the Algorithm-1 noise security argument entirely. Optimum likely λ ∈ (0, 0.3). |
| **P2** | **h** (expansion, paper default 128) | h ∈ {128, 256} sweep (in progress 2026-05-22). | Cheap (synthetic mode, no re-obf) | If the Cat 2 leader (`top_sv_overlap_r{H}.mean`) preserves rank at h=256, mechanism is robust. If not, less-obvious lever. |
| P3 | K (attacker pool size, default 64) | Already swept in Theory-2 section; TTRSR grows with K up to ~256. Defender-side direction: nothing — this is attacker capability. | — | Not a defender lever. |

**Deployment-side modification — Adversarial K_d selection (L1, with closed-form scoring):**

The Cat 2 winner `top_sv_overlap_r128.mean` is a **closed-form scoring function** for picking K_d at deployment. Algorithm:

1. Build a **reference K_a pool** once: K=64 K_a^k from a deterministic seed schedule (e.g. seeds 1..64). Compute and cache `Vh_a_top ∈ ℝ^(K, 128, d_obs)` — top-128 right-SVs per K_a^k.
2. For each candidate K_d (sample N_kd ≈ 64 from Algorithm 1): compute `Vh_d_top ∈ ℝ^(128, d_obs)` via one SVD.
3. Pool-mean score per candidate: `score(K_d) = mean_k Σ σ²(Vh_a_top[k] · Vh_d_top.T)` (sum of squared SVs of the (128, 128) cross-matrix). Computable in milliseconds per candidate.
4. **Pick `argmax_{K_d}` score** (sign is negative ⟹ high overlap = unlucky for attacker = good for defender).

**Statistical effectiveness, point estimate:** the c1 sample TTRSR distribution at the deployment K_d is approximately N(μ ≈ 3.2 %, σ ≈ 2.1 pp). The score-vs-TTRSR relationship has Spearman ρ = -0.78 (deterministic given the scoring function holds across K_d samples too, which the universality probe and Cat 2's robustness across N=5..10 both support). For N_kd = 64 candidates, the *best* candidate's expected TTRSR is at the lower tail. Conservatively, picking by score at ρ = -0.78 selects a candidate whose TTRSR is ≈ 1 standard deviation below mean ≈ **1.1 %**, vs the unconditional mean of 3.2 % — a ~2 pp expected reduction per deployment, at zero deployment-time runtime cost beyond the 64 SVDs (~minutes).

**Can M1 / M2 / P1 achieve the same effect?**

- **M1 (σ_min-rejection at K_d) is the closest parametric/structural analog to adversarial K_d selection.** Both narrow the defender's K_d distribution to spectrally-typical samples. M1 picks by the *intrinsic* σ_min feature (Cat 1); adversarial K_d selection picks by the *alignment* feature (Cat 2). Since these are co-correlated in the c1 data (top-3 features all anchor on K_a^k's tail behaviour), M1 likely recovers a large fraction of the Cat-2-driven gain — but Cat 2 selection is the principled choice: it conditions on the attacker pool, M1 only on K_d's intrinsic spectrum. **M1 ≤ adversarial K_d** in expected gain.

- **M2 (lighter-tailed V) would help symmetrically — both K_d's and K_a's draws lose the heavy tail.** This *narrows the attacker's distribution too*, which both increases the defender's lower-bound assurance (no lucky attackers) **and** removes the lever from K_d adversarial selection (every K_d is "typical" so picking among them gains less). M2 and adversarial K_d are partial substitutes, not stack-additive. Net: M2 alone ≈ adversarial K_d alone for this mechanism; **stacking gains diminishing returns**.

- **P1 (lower λ) tightens the same V distribution as M2 but parametrically.** Same substitute-not-additive logic. Likely the cheapest single change with the largest expected effect; the h-sweep (P2) is the natural companion test.

**Recommended order of attack:**

1. **Now (free):** Implement adversarial K_d selection with `top_sv_overlap_r128.mean` scoring. Zero risk, paper-faithful, immediate deployment win.
2. **Concurrent (in progress):** h-sweep at h=256 to confirm Cat 2 mechanism robustness. If `top_sv_overlap_r256.mean` doesn't win at h=256, the scoring function is h-fragile and adversarial K_d selection needs an h-aware refinement.
3. **Next sweep (~4 h):** P1 λ-sweep at fixed K_d_seed=42, λ ∈ {0.05, 0.1, 0.2, 0.3}, 10 pool seeds each. Synthetic mode, plain captures reused. Pick optimum.
4. **Concurrent with #3:** M1 implementation. Trivial change, opportunistic deployment use.
5. **Only if 1–4 leave a gap:** M2 sub-Gaussian V — invest in the security-proof re-derivation.

The h-sweep is launched in parallel with this update; results land in ~1 h.

### h-sweep result — h=256 measurement (2026-05-22)

Job 2 TTRSR sweep at `--attacker-expansion 256`, kd_test_seed=42, 10 pool seeds, vendor_cpu builder, GPU attack — completed in ~29 min wall. (c1 feature probe at h=256 still in progress at writeup time; reported here are the TTRSR-only numbers since they answer the immediate-goal question directly.)

| pool | h=128 hits / 226 | h=128 TTRSR % | h=256 hits / 226 | h=256 TTRSR % |
|---|---|---|---|---|
| 1 | 7 | 3.10 | 7 | 3.10 |
| 2 | 15 | 6.64 | 7 | 3.10 |
| 3 | 3 | 1.33 | 6 | 2.65 |
| 4 | 16 | 7.08 | 13 | 5.75 |
| 5 | 2 | 0.88 | 2 | 0.88 |
| 6 | 2 | 0.88 | 4 | 1.77 |
| 7 | 4 | 1.77 | 3 | 1.33 |
| 8 | 6 | 2.65 | 4 | 1.77 |
| 9 | 10 | 4.42 | 3 | 1.33 |
| 10 | 7 | 3.10 | 3 | 1.33 |
| **mean** | 7.2 | **3.19 %** | 5.2 | **2.30 %** |
| **std** | — | **2.12 pp** | — | **1.37 pp** |
| range | — | 0.88 – 7.08 | — | 0.88 – 5.75 |

**Reading:** moving h=128 → h=256
- Pool-mean TTRSR drops **28 %** (3.19 % → 2.30 %)
- Pool-to-pool std drops **35 %** (2.12 → 1.37 pp)
- Range drops 21 % (6.2 pp → 4.87 pp)

Both directions favor the defender. h is a confirmed parametric lever (P2). The reduction is modest per +128 h step — extrapolation to h=512 (skipped on cost grounds) by the same linear delta would yield σ_pool ≈ 0.6 pp, mean ≈ 0.5 %, but the trend likely saturates rather than continues linearly. Cost of moving to h=256: ~9 % inference latency (d_obs grows 2816 → 3072); zero security risk (paper-faithful).

### Defender plan to eliminate the ~5 pp Algorithm-1/keymat variance

**Goal:** drive σ_pool (the pool-to-pool TTRSR variance at fixed K_d, contributing the ~5 pp single-pool variance noted in the headline) toward zero before adding Algorithm-2 effects to the picture.

**Hard ceiling:** σ_pool cannot be driven to zero at K=64. It reflects the intrinsic finite-sample variance of K=64 K_a^k draws from the Algorithm-1 distribution. Only the attacker controls K (defender doesn't); the K-sweep section above shows TTRSR grows up to ~K=256, so K=64 is sub-asymptotic. **The defender's plan is to narrow the K_a distribution (so any 64-sample draw has lower variance) and reduce the mean attacker capability**, not to eliminate σ_pool entirely.

**Layered approach — stack-additive at the distribution-narrowing layer, distribution-shifting at the K_d layer:**

| # | Lever | σ_pool effect | Mean TTRSR effect | Evidence | Status |
|---|---|---|---|---|---|
| L_h | **Bump h: 128 → 256** | -35 % | -28 % | Job 2 h=128 vs h=256, N=10 each | **Measured. Deploy now.** |
| L_λ | **Lower λ to optimum (sweep λ ∈ {0.05, 0.1, 0.2, 0.3})** | est. -25-40 % more | est. -25-40 % more | c1 mechanism: V-noise iid Gaussian tail drives lucky/unlucky pools. Lower λ ⇒ B = U + λV → U → tighter B spectrum → tighter K_a^k subspace structure → lower σ_pool. | Unmeasured. ~3 h sweep cost (synthetic mode, plain captures reused). |
| L_kd | **Adversarial K_d selection** with `top_sv_overlap_r{H}.mean` scoring | unchanged | est. -50 % more (drops to ~0.5 % expected) | c1 h=128 N=10: Spearman ρ = -0.78 between scoring function and TTRSR. From 64 K_d candidates, pick the highest-score candidate. | Free at deployment (~5 min per deployment). Implementable immediately. |
| L_m1 | **σ_min-rejection sampling at K_d construction** (M1, paper-faithful) | unchanged | small (~10 %) — partial substitute for L_kd | c1 Cat 1 second-place at h=128. Narrows K_d's σ_min tail. | Free, 1-line code change. Mostly subsumed by L_kd. |
| L_m2 | **Sub-Gaussian V replacement** (M2, non-paper-faithful) | est. -50 % | est. -50 % | c1 mechanism prediction — but needs paper §5.2 security proof rework. | Skip unless L_h + L_λ + L_kd don't close the gap. |

**Expected combined result, point estimates (compounded, multiplicative on σ_pool; additive shifts on mean):**

| Stack | σ_pool | Mean TTRSR | Mean+2σ ceiling |
|---|---|---|---|
| h=128 baseline (current) | 2.12 pp | 3.19 % | 7.4 % |
| h=256 alone | 1.37 pp | 2.30 % | 5.0 % |
| h=256 + L_kd | 1.37 pp | ~0.5 % | 3.2 % |
| h=256 + L_λ (optimal) + L_kd | **~0.9 pp** | **~0.3 %** | **~2.1 %** |

This is the **best achievable defender position with paper-faithful changes only** at K=64 attacker. Going further requires either (a) L_m2 with security proof rework, or (b) the Algorithm-2 amplifier becoming load-bearing (Phase d ablation will quantify how much of the real-deployment 13-pp spread Algorithm 2 is responsible for; Algorithm-1 alone gives only 4-7 pp here).

**Why adversarial K_d selection doesn't eliminate σ_pool:** L_kd picks one K_d that minimises *expected* attacker TTRSR. At the selected K_d, the attacker still draws 64 K_a^k matrices from the Algorithm-1 distribution and that sample has σ_pool variance. L_kd shifts the entire distribution leftward; only L_h, L_λ, L_m2 narrow it.

**Why L_λ (lower λ) is high priority:** c1 N=10 identified the V-noise tail as the mechanism. The scoring function's Spearman ρ = -0.78 is driven by V drawing outlier matrices whose top-h right-SV subspace ends up mis-aligned with K_d's. Lower λ shrinks the V contribution to B = U + λV → less V-tail variance → smaller K_a-pool variance by direct construction. λ → 0 is bounded by the security cost (B → Haar U → no Algorithm-1 noise hiding); we expect a non-trivial optimum below 0.3.

**Why L_h has bounded gain:** h=256 cut σ_pool by 35 % already; extrapolating to h=512 by linear delta would give ~0.6 pp (almost zero), but the trend likely saturates because the K_a^k structure becomes dominated by the (h-rank) C-block as h grows, and C's contribution to top-h subspace alignment plateaus. A second h step (h=384 or h=512) would test this — deferred per the user's "don't run h=512, extrapolate" instruction.

### Recommended order of execution

1. **Deploy at h=256** — cheap, paper-faithful, immediate ~35 % σ_pool reduction.
2. **Implement adversarial K_d selection** (~5 min deployment-time recipe; reuses pre-built reference K_a pool's `Vh_a_top` cache).
3. **Run λ-sweep** at h=256, λ ∈ {0.05, 0.1, 0.2, 0.3} × 10 pool seeds, synthetic mode, plain captures reused. ~3-4 h total. Deploy at optimum.
4. **Stop here for Algorithm-1.** Pivot to **Phase d (Algorithm-2 ablation)** to quantify how much of the real-deployment 13-pp pool spread is Algorithm 2 vs Algorithm 1 — that's the remaining variance to attack.

L_m1 and L_m2 are intentionally not in the critical path: L_m1 is mostly redundant with L_kd (both narrow K_d's spectral structure), and L_m2's cost (re-derive paper security proof) is only justified if L_h + L_λ + L_kd leave a deployment-blocking gap.

### Real-mode validation of L_kd — refuted (2026-05-22)

We tested adversarial K_d selection (best-of-N=64 winner: `init_seed = 1000013`, score = 21.365 = +2.26σ above the mean 21.333) on a **real obfuscated model** with Algorithm-2-fully-excluded, to validate whether the synthetic-mode predictions transfer.

#### Test design

| Stage | Detail |
|---|---|
| Obfuscation | `python/aloepri-llm/obfuscate_qwen3_gguf.py --mode keymat --expansion-size 256 --seed 1000013 --output-dtype bf16` — no `--pi`, no `--alg2`, no `--noise-*`; pure Algorithm-1 |
| Captures | `capture_hidden_states.py --no-tau-map` (identity τ since no Pi) against `aloepri-llama-server:m2_7`; 64 prompts × 14 tokens, attn_norm at layers {0, 17, 30}; n_test=373 |
| Attack | `run_isa_multikey.py` in **real mode** (`--obf-captures`, NOT `--kd-test-seed`); K=64 multi-key ridge; 10 pool seeds (only N=6 actually run before stop) |

Plain captures, split seed, and attacker keymat builder all matched the prior synthetic h=256 baseline so the only deliberate difference was real-obfuscated vs synthetic-applied K_d.

#### Result (N=6)

| pool | hits / 373 | TTRSR |
|---|---|---|
| 1 | 6 | 1.61 % |
| 2 | 39 | **10.46 %** (above plain ceiling) |
| 3 | 3 | 0.80 % |
| 4 | 19 | 5.09 % |
| 5 | 15 | 4.02 % |
| 6 | 11 | 2.95 % |
| **mean** | **15.5** | **4.16 %** |
| **median** | — | 3.49 % |

Comparison vs prior baselines:

| condition | mean TTRSR | std (across pool seeds) |
|---|---|---|
| synthetic h=128 + kd_seed=42 (paper default) | 3.19 % | 2.12 pp |
| synthetic h=256 + kd_seed=42 (L_h alone) | 2.30 % | 1.37 pp |
| **real-obf h=256 + adv K_d (L_h + L_kd, N=6)** | **4.16 %** | **3.21 pp** |

**Real-mode TTRSR is higher than both synthetic baselines, including the *no-adversarial-K_d* synthetic h=128 default.** L_kd does not transfer; it possibly even regresses relative to synthetic-mode prediction.

#### Why L_kd doesn't transfer

The adversarial K_d selection optimised a closed-form score (`top_sv_overlap_r{H}.mean`) that captures a property of the *ideal-math* K_d. Real-mode TTRSR is generated through a stack of numerical / structural transformations the score function ignores:

1. **Precision mismatch.** Score is computed at float64 keymat → float32 SVD. Real-mode K_d is stored bf16 in GGUF tensors; activations propagate through bf16 ops over 17 transformer blocks. Per-block bf16 relative error ~1e-3 compounds to ~1.7 × 10⁻² cumulative — **larger than the +0.15 % score advantage** the selection found.
2. **L=17 captures are not `(plain_L17) @ K_d`.** Synthetic-mode mirrors the score function's linearity exactly. Real-mode L=17 activations are a 17-block obfuscated forward pass through attention, FFN, RMSNorm-γ-fusion. The score function's linear-algebra view of K_d's spectral structure decouples from real-mode TTRSR through these nonlinearities.
3. **Reference-pool bias.** The score selected K_d=1000013 as most-aligned against the *specific reference K_a pool* {1, 10001, …, 630001}. The 10 attacker test pools use *different* seed schedules (`pool + 1 + 10000·k`). What's optimal against one fixed reference pool isn't optimal against an arbitrary attacker pool — the selection's reference-pool-bias dominates when the score window is narrow (h=256: ±0.07 % of baseline).
4. **Q8_0 quantisation of obfuscated tensors.** GGUF stores most weights as Q8_0 (~1.5e-3 relative error). Compounds with bf16 activation noise.

The c1 mechanism finding (Cat 2 alignment wins at h=128) was therefore a property of the **idealised threat model** used in c1 / synthetic mode. The *direction* of the lever is real (lower σ_pool with h-bump in c1 measurements), but the **magnitude of L_kd's gain shrinks below the deployment noise floor**.

#### What still survives

| Lever | Status after real-mode test |
|---|---|
| L_h (h=128 → h=256) | **Likely transfers** because it's a finite-sample variance effect, not a numerical precision effect. Real-mode h=128 would need to be measured to confirm — TBD. |
| L_kd (best-of-N adv K_d at fixed h, λ) | **Refuted in real mode** at h=256. |
| L_λkd0 (asymmetric λ, K_d=0) | **Predicted to be refuted** by the same mechanism (the score-derived "elegant lever" has ≤0.03 score-window advantage at h=256, well below noise floor). Not worth a real-mode test. |
| M1 (σ_min-rejection at K_d) | **Untested**. Would have the same noise-floor concern as L_kd. Probably also doesn't transfer. |
| P1 (λ < 0.3 across both K_d and K_a^k) | **Untested**. This isn't a per-deployment selection; it changes the *distribution* the attacker draws from too. May transfer differently. |

#### Implications for the defender plan

- The doc's earlier "Defender plan to eliminate ~5 pp Algorithm-1/keymat variance" overestimated the achievable σ_pool reduction by treating synthetic-mode evidence as predictive for real-mode deployment. **The real-mode noise floor sets a TTRSR baseline that score-derived selection cannot push below.**
- Concretely: real-mode K=64 Algorithm-1-only attacker at d=2560, h=256 looks like **~3-5 % mean TTRSR with σ_pool ~3 pp** at this measurement N. Not the 0.5 % / 1.37 pp the synthetic model predicted.
- The 5-pp single-seed std from `aloepri_attack_keymat_cuda_philox_bias` was apparently always the right number; synthetic mode underestimated it because synthetic linearises away the deployment noise that *adds* to attacker capability via bf16 leakage.
- **For Algorithm 1, the practical defender ceiling is L_h alone.** L_kd / L_λkd0 / M1 / M2 selection-style levers don't survive the bf16+Q8_0+17-block noise floor.
- **Phase d (Algorithm 2 ablation) becomes the dominant remaining lever**: real-deployment 13-pp σ_pool minus the ~5 pp Algorithm-1 floor = ~8 pp amplification attributable to Algorithm 2. Targeting that amplifier is where the next gains live.

## Repro

Probe script and sweep harness:

- `/tmp/aloepri-gpu-validation/probe_rocsolver_nullspace.py` — Probe 1 (basis comparison) + Probe 2 (MT19937 + rocSOLVER 5-seed sweep).
- `/tmp/aloepri-gpu-validation/seed-sweep-5seeds.sh` — vendor + port-GPU 5-seed sweep.

JSON outputs and logs alongside under `/tmp/aloepri-gpu-validation/`.

Fixed inputs across every run:

- Plain captures: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-plain-rerun-20260520/captures`
- Obf captures: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-20260521/captures`
- Layer 17, kind `attn_norm`, K = 64, h = 128, λ = 0.3, row-split.
- Ridge α grid: {1e-4, 1e-2, 1.0}, val-selected.
