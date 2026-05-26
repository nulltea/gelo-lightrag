---
type: theory
status: current
created: 2026-05-21
updated: 2026-05-21
tags: [aloepri, alg1]
---

# AloePri Algorithm 1 keymat — K=64 sample-variance dominates TTRSR at d=2560

**Status:** investigation closed 2026-05-21. Theory 2 (K=64 sample variance) confirmed; Theory 1 (rocSOLVER basis orientation) refuted; the original "Philox + rocSOLVER GPU port bug" diagnosis is **retracted**.

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
