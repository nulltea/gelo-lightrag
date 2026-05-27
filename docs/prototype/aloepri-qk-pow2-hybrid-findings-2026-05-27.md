# AloePri Q/K hardening + bf16-safe UVO findings, 2026-05-27

## Goal

Test whether the defense gains from paper-style Q/K algebra can be kept while replacing dense/raw `U_vo` with the bf16-safe `pow2-monomial` family discovered on 2026-05-26.

The working hypothesis was:

```text
paper/QK-side defense + pow2-monomial UVO => lower ISA row-split recovery without dense-UVO accuracy collapse
```

## Baselines carried forward

| Cell | Quality | HumanEval n=20 | Strongest relevant attack reading |
|---|---:|---:|---|
| Non-UVO canonical, h128 beta8 | pass | 6/20 = 30% | utility reference |
| Dense/default UVO, h128 beta8 | pass | 3/20 = 15% | utility damaged by dense bf16 V/O rounding |
| pow2-monomial UVO, h128 beta8 | pass | 6/20 = 30% | recovers utility; does not defend `kq` row split |
| true-paper-K + default/stabilized UVO (`PAPERKTRUE`) | fail | skipped | L0/L5 micro-probe defense improved, but quality failed |
| old no-R `PAPERLIT` A1+A2 | fail | skipped | best measured row-split defense; used raw/dense paper-ish UVO and no-R K |

## Mathematical model

There are two independent-looking error channels, but they interact in deployment:

1. **V/O UVO numerical error.** Dense `U_vo` performs many-channel sums before bf16 storage. Even if `cond(U)=1`, the inverse in `W_o` cannot reconstruct mantissa bits already rounded away. Power-of-two monomial UVO avoids this because, for normal-range bf16 values, `Q_bf16(2^k w) = 2^k Q_bf16(w)`.
2. **Q/K score perturbation.** The strongest row-split attack reads `QK^T` or attention output. V/O-only changes are invisible to raw `kq`. Paper/no-R K-side algebra changes the score surface and can lower row-split recovery, especially after L0.

So the deployable target is not `more UVO`; it is **Q/K-side non-covariance with bf16-commuting V/O transforms**.

## New cells tested today

### Cell A: true-paper-K + pow2-monomial UVO, beta8

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-paperK-uvo-pow2e1-bf16-native.gguf
```

Build flags:

```bash
--mode keymat --expansion-size 128 --pi --noise-alpha-e 1.0 --noise-alpha-h 0.2 --alg2 --alg2-qk-norm-matrix --alg2-h-hadamard-signs --alg2-beta 8 --alg2-paper-literal-k --alg2-u-vo --alg2-u-vo-mode pow2-monomial --alg2-u-vo-pow2-exp 1 --output-dtype bf16
```

Quality result:

| Probe | Result |
|---|---|
| 5-prompt quality gate | **fail** |
| HumanEval | skipped |

Observed outputs were repetitive/off-manifold (`che che`, repeated `sum`, repeated fragments). This falsifies the narrow hypothesis that `PAPERKTRUE` failed only because default dense/stabilized UVO damaged bf16 V/O cancellation. Replacing UVO with pow2 was not enough at beta8.

### Cell B: true-paper-K + pow2-monomial UVO, beta1

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-beta1-paperK-uvo-pow2e1-bf16-native.gguf
```

Quality result:

| Probe | Result |
|---|---|
| 5-prompt readability gate | pass |
| Semantic quality | degraded: France -> Belgium, arithmetic repeats |
| HumanEval n=20 | killed at user request after 6/17 observed |

Reading: beta1 removes enough block-permutation disruption to pass the simple readability gate, but task behavior is still degraded and generations usually hit the token cap. This matches the previous beta1 observation: readable is not the same as semantically healthy.


### Cell C: no-R Q/K hardening + pow2-monomial UVO, beta8

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-noR-uvo-pow2e1-bf16-native.gguf
```

Quality result:

| Probe | Result |
|---|---|
| 5-prompt quality gate | **fail** |
| HumanEval | skipped |
| Attacks | skipped |

Observed outputs were worse than true-paper-K beta8: repeated `measure`, `sum`, and bracket fragments across every prompt. This falsifies the hope that the old no-R defense gain was mainly blocked by raw/dense UVO. The no-R K-side perturbation itself is too far off the model manifold in this Qwen3 matrix-Γ implementation.

Updated reading: **the utility bottleneck is now Q/K score-manifold distortion, not V/O bf16 cancellation**. `pow2-monomial` fixed the V/O side, but strong K-side non-covariance still ruins generation.


### Cell D: true-paper-K + pow2-monomial UVO, beta8, no Hadamard signs

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-paperK-uvo-pow2e1-bf16-native.gguf
```

Difference from Cell A: removed `--alg2-h-hadamard-signs`, so `H=I` while keeping true-paper K, matrix-Γ, `Z_block` beta8, and pow2 UVO.

Quality result:

| Probe | Result |
|---|---|
| 5-prompt quality gate | **fail** |
| HumanEval | skipped |
| Attacks | skipped |

Observed outputs still contained repeated `measure`, ellipses, and invalid code-like fragments. This rules out the Hadamard-sign perturbation as the primary cause of beta8 failure. The remaining suspect is the magnitude of Q/K non-covariance induced by `Z_block^T` at beta8 under the matrix-Γ QK-norm implementation.


### Cell E: true-paper-K + pow2-monomial UVO, beta2, no Hadamard signs

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-beta2-paperK-uvo-pow2e1-bf16-native.gguf
```

Difference from Cell D: shrink `Z_block` window from beta8 to beta2 while keeping `H=I`, true-paper K, matrix-Γ, and pow2 UVO.

Quality result:

| Probe | Result |
|---|---|
| 5-prompt quality gate | **pass** |
| Semantic sample | much healthier; still repetitive |
| HumanEval n=20 | **6/20 = 30%** |

Sample reading: France -> Paris, arithmetic -> 68, code prompt emits plausible function prefix. This is the first useful signal for a calibrated Q/K defense: beta8 is too large, beta1/H-sign is semantically weak, beta2/no-H may be the survivable middle.

## Tradeoff update

| Lever | Defense direction | Accuracy effect observed | Current interpretation |
|---|---|---|---|
| pow2-monomial UVO | preserves L0 `kqv_out` defense vs plain; no raw `kq` defense | good: 6/20 on h128 beta8 without paper-K | keep as the safe UVO family |
| dense/default UVO | may obscure V/O basis | bad: 3/20 | avoid for bf16 deployment |
| true-paper-K beta8 | improves `kqv_out`, L5 `kq` in prior micro-probe | **bad: quality fail with H signs and without H signs** | beta8 `Z_block^T`/paper-K perturbation is too large |
| true-paper-K beta1 | likely weaker defense than beta8/no-R | mixed: readable gate pass, semantic degradation, HumanEval killed at 6/17 | not viable without further calibration |
| true-paper-K beta2, no-H | default-like on AttnScore; strong on HiddenState | **quality pass; HumanEval 6/20** | first calibrated utility candidate, but not enough score-surface defense |
| no-R Q/K | strongest prior row-split defense | **bad: quality fail even with pow2 UVO** | not deployable at beta8/current matrix-Γ |


Focused ISA row-split micro-test, first 256 prompts, 3 seeds, GPU ridge:

| Surface | Layer | beta2/no-H top1 | Prior default obf top1, 256p | Prior true-paper-K beta8 top1, 256p | Reading |
|---|---:|---:|---:|---:|---|
| `kq` | 0 | 41.97 ± 0.81% | 42.27 ± 1.24% | 40.14 ± 1.38% | tiny gain only |
| `kq` | 5 | 34.55 ± 0.59% | 33.90 ± 0.87% | 14.43 ± 0.45% | defense mostly gone |
| `kqv_out` | 0 | 72.14 ± 1.03% | 70.98 ± 0.79% | 47.06 ± 0.14% | default-like, worse than beta8 |
| `kqv_out` | 5 | 45.05 ± 1.10% | 44.25 ± 1.44% | 12.66 ± 0.74% | default-like, no useful defense |

Interpretation: beta2/no-H is the first utility-preserving calibrated Q/K candidate, but the useful Q/K defense disappears with this much shrinkage. The tradeoff frontier is now bracketed:

- beta8: strong defense, quality fails.
- beta2/no-H: utility survives, defense nearly disappears.
- beta4/no-H is the next natural quality-only test.

## Model-size scaling tradeoff

Qwen3-8B has d=4096, so it gives more utility headroom than the 4B model: the same absolute keymat expansion h is a smaller fraction of the residual dimension, and stronger Q/K perturbations such as beta4 may be less likely to push generation off-manifold. This is a reason to test beta4/no-H and h256 on 8B even though they are too sharp for the 4B utility cell.

That headroom is not the same thing as automatic privacy. Larger d also gives ridge-style static and internal-state attacks more observable coordinates. In particular, prior 8B HiddenState/IMA measurements showed that UVO attenuation can shrink with model size, and static ridge can become easier when the attacker has more dimensions to fit. The practical reading is:

| Scaling effect | Helps | Hurts | Sweep implication |
|---|---|---|---|
| Larger d lowers relative perturbation cost for fixed h and beta | accuracy / quality | none directly | use 8B to retry beta4 and h256 behind a quality gate |
| Larger d gives more observed coordinates to linear attackers | none directly | ridge/static recovery can improve | do not treat 8B as a privacy proof; rerun harness surfaces |
| Larger d may tolerate higher alpha or beta before visible collapse | AttnScore / embedding-noise defense if quality survives | generation coherence if overdriven | increase one lever at a time, not beta+h+alpha together |

So the 8B hypothesis is **more deployable defense budget**, not **free defense from dimensionality**. A useful 8B result must show both: quality gate survives and the strongest row-split/HiddenState surfaces actually move.

## Next running test

Cell C built no-R K-side hardening with pow2 UVO and failed quality. The next test should reduce Q/K perturbation amplitude rather than changing UVO again. Simple beta ramp is exhausted: beta2 preserves utility but loses defense; beta4+ fails quality. Next direction should change the kind of Q/K perturbation, not just beta magnitude.

```bash
--alg2-paper-literal-k-no-r --alg2-u-vo --alg2-u-vo-mode pow2-monomial --alg2-u-vo-pow2-exp 1 --alg2-beta 8
```

Purpose: isolate whether the old no-R defense can keep its attack gain once raw/dense UVO is removed. Run quality-only first. Do not run HumanEval or attacks if quality fails.


### Cell F: true-paper-K + pow2-monomial UVO, beta4, no Hadamard signs

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-beta4-paperK-uvo-pow2e1-bf16-native.gguf
```

Quality result:

| Probe | Result |
|---|---|
| 5-prompt quality gate | **fail** |
| HumanEval | skipped |
| Attacks | skipped |

Sample outputs collapsed into repeated numerals/tokens (`1. 1. 1`, repeated `wal`, repeated `6666`). This gives a useful bracket:

- beta2/no-H: quality pass, HumanEval 6/20, but attack defense is default-like.
- beta4/no-H: quality fail.
- beta8/no-H and beta8/H: quality fail.

Therefore a simple beta ramp does not expose a useful operating point. The defense/accuracy curve is steep: enough Q/K non-covariance to affect row-split ridge already damages generation.


## ISA HiddenState paper-faithful result

Ran K=64 paper-faithful labelled-ridge ISA on beta2/no-H at `attn_norm-17`, 64 prompts, row split, attacker seed 20260521, vendor-correct keymat pool, GPU ridge/cosine evaluation.

Result file:

```text
evals/aloepri-attacks/results/sweep/cell-qwen3-4b-paperK-noH-uvo-pow2e1-b2-20260527/isa-hiddenstate/isa-multikey-attn_norm17-seed20260521.json
```

| Metric | Result |
|---|---:|
| top-1 TTRSR | **3.89%** |
| top-10 TTRSR | 20.92% |
| risk | low |
| selected alpha | 1e-4 |

This is the strongest positive signal for the beta2/no-H utility cell: HiddenState improves versus the prior default-UVO 3-seed mean (8.54% ± 4.74), while AttnScore remains weak/default-like.

## Working conclusion so far

`pow2-monomial` solves the V/O bf16 accuracy problem, but it does **not** automatically make paper-style Q/K perturbations deployable. The accuracy bottleneck has moved from V/O cancellation error to Q/K score-manifold distortion. Future viable defense likely needs a calibrated Q/K perturbation: strong enough to break row-split ridge, but small/norm-preserving enough to keep generation on-manifold.
