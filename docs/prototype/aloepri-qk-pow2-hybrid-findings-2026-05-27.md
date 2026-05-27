# AloePri Q/K hardening + bf16-safe UVO findings, 2026-05-27

## Goal

Test whether the defense gains from paper-style Q/K algebra can be kept while replacing dense/raw `U_vo` with the bf16-safe `pow2-monomial` family discovered on 2026-05-26.

The working hypothesis was:

```text
paper/QK-side defense + pow2-monomial UVO => lower ISA row-split recovery without dense-UVO accuracy collapse
```

## Baselines carried forward

| Cell                                                 | Quality | HumanEval n=20 | Strongest relevant attack reading                                        |
| ---------------------------------------------------- | ------: | -------------: | ------------------------------------------------------------------------ |
| Non-UVO canonical, h128 beta8                        |    pass |     6/20 = 30% | utility reference                                                        |
| Dense/default UVO, h128 beta8                        |    pass |     3/20 = 15% | utility damaged by dense bf16 V/O rounding                               |
| pow2-monomial UVO, h128 beta8                        |    pass |     6/20 = 30% | recovers utility; does not defend `kq` row split                         |
| true-paper-K + default/stabilized UVO (`PAPERKTRUE`) |    fail |        skipped | L0/L5 micro-probe defense improved, but quality failed                   |
| old no-R `PAPERLIT` A1+A2                            |    fail |        skipped | best measured row-split defense; used raw/dense paper-ish UVO and no-R K |

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

| Probe                 | Result   |
| --------------------- | -------- |
| 5-prompt quality gate | **fail** |
| HumanEval             | skipped  |

Observed outputs were repetitive/off-manifold (`che che`, repeated `sum`, repeated fragments). This falsifies the narrow hypothesis that `PAPERKTRUE` failed only because default dense/stabilized UVO damaged bf16 V/O cancellation. Replacing UVO with pow2 was not enough at beta8.

### Cell B: true-paper-K + pow2-monomial UVO, beta1

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-beta1-paperK-uvo-pow2e1-bf16-native.gguf
```

Quality result:

| Probe                     | Result                                          |
| ------------------------- | ----------------------------------------------- |
| 5-prompt readability gate | pass                                            |
| Semantic quality          | degraded: France -> Belgium, arithmetic repeats |
| HumanEval n=20            | killed at user request after 6/17 observed      |

Reading: beta1 removes enough block-permutation disruption to pass the simple readability gate, but task behavior is still degraded and generations usually hit the token cap. This matches the previous beta1 observation: readable is not the same as semantically healthy.

### Cell C: no-R Q/K hardening + pow2-monomial UVO, beta8

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-noR-uvo-pow2e1-bf16-native.gguf
```

Quality result:

| Probe                 | Result   |
| --------------------- | -------- |
| 5-prompt quality gate | **fail** |
| HumanEval             | skipped  |
| Attacks               | skipped  |

Observed outputs were worse than true-paper-K beta8: repeated `measure`, `sum`, and bracket fragments across every prompt. This falsifies the hope that the old no-R defense gain was mainly blocked by raw/dense UVO. The no-R K-side perturbation itself is too far off the model manifold in this Qwen3 matrix-Γ implementation.

Updated reading: **the utility bottleneck is now Q/K score-manifold distortion, not V/O bf16 cancellation**. `pow2-monomial` fixed the V/O side, but strong K-side non-covariance still ruins generation.

### Cell D: true-paper-K + pow2-monomial UVO, beta8, no Hadamard signs

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-paperK-uvo-pow2e1-bf16-native.gguf
```

Difference from Cell A: removed `--alg2-h-hadamard-signs`, so `H=I` while keeping true-paper K, matrix-Γ, `Z_block` beta8, and pow2 UVO.

Quality result:

| Probe                 | Result   |
| --------------------- | -------- |
| 5-prompt quality gate | **fail** |
| HumanEval             | skipped  |
| Attacks               | skipped  |

Observed outputs still contained repeated `measure`, ellipses, and invalid code-like fragments. This rules out the Hadamard-sign perturbation as the primary cause of beta8 failure. The remaining suspect is the magnitude of Q/K non-covariance induced by `Z_block^T` at beta8 under the matrix-Γ QK-norm implementation.

### Cell E: true-paper-K + pow2-monomial UVO, beta2, no Hadamard signs

Artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-beta2-paperK-uvo-pow2e1-bf16-native.gguf
```

Difference from Cell D: shrink `Z_block` window from beta8 to beta2 while keeping `H=I`, true-paper K, matrix-Γ, and pow2 UVO.

Quality result:

| Probe                 | Result                           |
| --------------------- | -------------------------------- |
| 5-prompt quality gate | **pass**                         |
| Semantic sample       | much healthier; still repetitive |
| HumanEval n=20        | **6/20 = 30%**                   |

Sample reading: France -> Paris, arithmetic -> 68, code prompt emits plausible function prefix. This is the first useful signal for a calibrated Q/K defense: beta8 is too large, beta1/H-sign is semantically weak, beta2/no-H may be the survivable middle.

## Tradeoff update

| Lever                    | Defense direction                                            | Accuracy effect observed                                                  | Current interpretation                                                   |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| pow2-monomial UVO        | preserves L0 `kqv_out` defense vs plain; no raw `kq` defense | good: 6/20 on h128 beta8 without paper-K                                  | keep as the safe UVO family                                              |
| dense/default UVO        | may obscure V/O basis                                        | bad: 3/20                                                                 | avoid for bf16 deployment                                                |
| true-paper-K beta8       | improves `kqv_out`, L5 `kq` in prior micro-probe             | **bad: quality fail with H signs and without H signs**                    | beta8 `Z_block^T`/paper-K perturbation is too large                      |
| true-paper-K beta1       | likely weaker defense than beta8/no-R                        | mixed: readable gate pass, semantic degradation, HumanEval killed at 6/17 | not viable without further calibration                                   |
| true-paper-K beta2, no-H | default-like on AttnScore; strong on HiddenState             | **quality pass; HumanEval 6/20**                                          | first calibrated utility candidate, but not enough score-surface defense |
| no-R Q/K                 | strongest prior row-split defense                            | **bad: quality fail even with pow2 UVO**                                  | not deployable at beta8/current matrix-Γ                                 |

Focused ISA row-split micro-test, first 256 prompts, 3 seeds, GPU ridge:

| Surface   | Layer | beta2/no-H top1 | Prior default obf top1, 256p | Prior true-paper-K beta8 top1, 256p | Reading                         |
| --------- | ----: | --------------: | ---------------------------: | ----------------------------------: | ------------------------------- |
| `kq`      |     0 |   41.97 ± 0.81% |                42.27 ± 1.24% |                       40.14 ± 1.38% | tiny gain only                  |
| `kq`      |     5 |   34.55 ± 0.59% |                33.90 ± 0.87% |                       14.43 ± 0.45% | defense mostly gone             |
| `kqv_out` |     0 |   72.14 ± 1.03% |                70.98 ± 0.79% |                       47.06 ± 0.14% | default-like, worse than beta8  |
| `kqv_out` |     5 |   45.05 ± 1.10% |                44.25 ± 1.44% |                       12.66 ± 0.74% | default-like, no useful defense |

Interpretation: beta2/no-H is the first utility-preserving calibrated Q/K candidate, but the useful Q/K defense disappears with this much shrinkage. The tradeoff frontier is now bracketed:

- beta8: strong defense, quality fails.
- beta2/no-H: utility survives, defense nearly disappears.
- beta4/no-H is the next natural quality-only test.

## Model-size scaling tradeoff

Qwen3-8B has d=4096, so it gives more utility headroom than the 4B model: the same absolute keymat expansion h is a smaller fraction of the residual dimension, and stronger Q/K perturbations such as beta4 may be less likely to push generation off-manifold. This is a reason to test beta4/no-H and h256 on 8B even though they are too sharp for the 4B utility cell.

That headroom is not the same thing as automatic privacy. Larger d also gives ridge-style static and internal-state attacks more observable coordinates. In particular, prior 8B HiddenState/IMA measurements showed that UVO attenuation can shrink with model size, and static ridge can become easier when the attacker has more dimensions to fit. The practical reading is:

| Scaling effect                                                     | Helps                                                   | Hurts                              | Sweep implication                                          |
| ------------------------------------------------------------------ | ------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------- |
| Larger d lowers relative perturbation cost for fixed h and beta    | accuracy / quality                                      | none directly                      | use 8B to retry beta4 and h256 behind a quality gate       |
| Larger d gives more observed coordinates to linear attackers       | none directly                                           | ridge/static recovery can improve  | do not treat 8B as a privacy proof; rerun harness surfaces |
| Larger d may tolerate higher alpha or beta before visible collapse | AttnScore / embedding-noise defense if quality survives | generation coherence if overdriven | increase one lever at a time, not beta+h+alpha together    |

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

| Probe                 | Result   |
| --------------------- | -------- |
| 5-prompt quality gate | **fail** |
| HumanEval             | skipped  |
| Attacks               | skipped  |

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

| Metric         |    Result |
| -------------- | --------: |
| top-1 TTRSR    | **3.89%** |
| top-10 TTRSR   |    20.92% |
| risk           |       low |
| selected alpha |      1e-4 |

This is the strongest positive signal for the beta2/no-H utility cell: HiddenState improves versus the prior default-UVO 3-seed mean (8.54% ± 4.74), while AttnScore remains weak/default-like.

## Working conclusion so far

`pow2-monomial` solves the V/O bf16 accuracy problem, but it does **not** automatically make paper-style Q/K perturbations deployable. The accuracy bottleneck has moved from V/O cancellation error to Q/K score-manifold distortion. Future viable defense likely needs a calibrated Q/K perturbation: strong enough to break row-split ridge, but small/norm-preserving enough to keep generation on-manifold.

## Qwen3-8B sweep (added 2026-05-27)

### Pipeline correctness check first

Before reading the 8B obfuscated numbers, we confirmed the plaintext baseline
through the _exact same_ `/completion` pipeline (raw token-ID continuation,
no chat template, no thinking-mode injection — AloePriClient hardcodes
`/completion` and the request body uses `prompt: [int IDs]`, never
`messages`).

| Cell                       | Pipeline            |     HumanEval n=20 |  wall |
| -------------------------- | ------------------- | -----------------: | ----: |
| Plain Qwen3-8B (reference) | `/completion`, no τ | **10/20 = 50.0 %** | 687 s |

Per-completion check: 0/20 plain completions contain `<think>`/`</think>`
tokens; 3/20 show plain-English reasoning drift mid-code which does not
crash plain accuracy. So Qwen3-8B's instruct/thinking training does not
silently leak into our `/completion` regime, and the 50 % plain ceiling is
on-distribution for raw base-completion (chat-mode + thinking on this model
reaches 70 – 80 %).

### Obfuscated cells on 8B

All cells share: `keymat / Π / αₑ=1.0 / αₕ=0.2 / Alg2 matrix-Γ /
paper-literal-K / Ûvo pow2-monomial e=1 / H=I (no Hadamard signs) / bf16
output`. They differ only in (h, β):

|   h |     β | κ(K_d) | Quality         |    HumanEval n=20 | Δ vs plain 8B |
| --: | ----: | -----: | --------------- | ----------------: | ------------: |
| 128 | **2** |   7.79 | pass            | **8/20 = 40.0 %** |    **−10 pp** |
| 128 |     4 |   7.79 | **fail**        |           skipped |             — |
| 256 |     2 |  10.67 | pass (readable) |      0/20 = 0.0 % |        −50 pp |
| 256 |     4 |  10.67 | **fail**        |           skipped |             — |

Quality-fail mode at β=4 is single-token loops (`is is is is …`,
`What\n\nWhat\n\nWhat\n\n…`). At h=256/β=2 the readability heuristic
passes (the model says "Paris", "68", etc.) but every HumanEval completion
collapses into deterministic statement-level loops
(`if not isinstance(strings, list): return None` repeated 15×).
The h=128/β=2 cell is the only operating point that keeps multi-step
coherence: real `is_prime` trial-division, correct French translation,
8/20 HumanEval.

### Per-cell quality probe outputs (5 fixed prompts × 48 tok, τ-routed)

The probe runs the same five prompts through every cell via
`AloePriClient` (τ applied to the prompt, τ⁻¹ to the response). Each
cell row shows the first 60 chars of each de-obfuscated completion.

| Prompt                                      | h=128 / β=2 (8/20)                                                    | h=128 / β=4 (quality fail)                                    | h=256 / β=2 (0/20)                                                 | h=256 / β=4 (quality fail)                                         |
| ------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| "The capital of France is"                  | `Paris, and it is also a city with a long history, and it is`         | `is is is is is is is is is is is is is is is is is is is is` | `Paris. The capital of France is Paris. The capital of Franc`      | `the great company of the\n\n.\n\n.\n\n.\n\n.\n\n.\n\n…`           |
| "Q: What is 17 times 4? A:"                 | `68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,` (correct then loop) | `what\n\nWhat\n\nWhat\n\nWhat\n\nWhat\n\nWhat\n\n…`           | `17 times 4 is 68. 17 times 4 is 68. 17 times 4 is 68. 17 ti`      | `<48 spaces>`                                                      |
| "Write a haiku about autumn leaves:"        | `first line: 5 syllables, second line: 7 syllables, third li`         | `a\n\na\n\na\n\na\n\na\n\na\n\na\n\na\n\na\n\na\n\na\n\n…`    | `3-3-3, 2-2-2, 1-1-1, 0-0-0, 0-0-0, 0-0-0, 0-0-0`                  | `a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a`      |
| "def is_prime(n):"                          | `\n    if n < 2:\n        return False\n    for i in range(2, in`     | `is, is is is is is is is is is is is is is is is is is is i` | `\n    if n == 1:\n        return True\n    if n == 2:\n        r` | ` (index (index (index (index (index (index (index (index (in`     |
| "Translate to French: 'Good morning, how '" | `'Bonjour, comment allez-vous?'.\nHow do you say 'I am happy'`        | `\n\nI am a\n\na\n\na\n\na\n\na\n\na\n\na\n\na\n\na\n\n…`     | `'Good morning, how are you?' 'Good morning, how are you?' '`      | `\n\nF\n\nF\n\nF\n\nF\n\nF\n\nF\n\nF\n\nF\n\nF\n\nF\n\nF\n\nF\n\n` |
| `all_readable`                              | true (heuristic flags `68, 68,` loop)                                 | **false** (5/5 collapsed)                                     | true                                                               | **false** (5/5 collapsed)                                          |
| HumanEval n=20                              | **8/20 = 40 %**                                                       | skipped                                                       | **0/20**                                                           | skipped                                                            |

**Reading.** The two β=4 cells produce single-character or
single-token loops on every prompt — the model has lost
prompt-conditioning entirely. The h=256/β=2 cell still reads
the prompt (correct France→Paris, 17×4=68, correct French
translation echo, plausible `is_prime` prefix) but cannot
hold multi-step state across more than ~10 tokens of generation:
every HumanEval task ends in a deterministic statement-level
loop. The h=128/β=2 cell shows real Python (`if n < 2: return
False; for i in range(2, in…`) and a correct French translation
(`Bonjour, comment allez-vous?`); short-form Q&A drifts after
the correct answer (`68, 68, 68…`) but multi-step coherence
holds long enough to land 8/20 HumanEval.

### Per-task HumanEval overlap, 8B h=128 / β=2 vs plain 8B

Of 10 plain passes:

- **7 retained under obfuscation:** `HumanEval/28, 35, 22, 8, 7, 23, 55`
  — these are shallow-reasoning tasks (sort, slice, format,
  predicate-list).
- **3 lost:** `HumanEval/6` (nested-parenthesis parser),
  `HumanEval/57` (two-list arithmetic with conditional
  monotonicity), `HumanEval/62` (polynomial derivative — multi-step
  index math).

Of 10 plain fails:

- **1 newly passes:** `HumanEval/26` (list dedup) — short
  generation, low-reasoning-depth.

Net change: −3 + 1 = −2 (= 50 % plain → 40 % obf). The lost tasks
share a pattern: they need more than ~30 coherent generation
tokens of multi-step state. The recovered task is a short
list-comprehension. This is consistent with the depth-compounding
model in
`docs/research/aloepri-h-beta-interaction-2026-05-27.md` § 2:
the keymat perturbation flattens the decoder's distribution as
generation accumulates, so prompt-conditioned outputs that resolve
in one or two coherent steps survive while longer plans drift into
loops.

### Per-task HumanEval overlap, 4B h=128 / β=2 vs 8B h=128 / β=2

Carrying the prior 4B β=2 reference (Cell E, 6/20 = 30 %) against
today's 8B β=2 number (8/20 = 40 %):

|                   | plain ceiling | h=128 / β=2 obf | absolute Δ | relative gap |
| ----------------- | ------------: | --------------: | ---------: | -----------: |
| Qwen3-4B (d=2560) |  not measured |     6/20 = 30 % |        n/a |          n/a |
| Qwen3-8B (d=4096) |  10/20 = 50 % | **8/20 = 40 %** | **−10 pp** |    **−20 %** |

The absolute number rises with model size as expected (plain
8B > plain 4B), but with no plain-4B reference at n=20 we can't
quantify whether the relative deficit is smaller on 8B. The
working-conclusion read is: 8B is more useful in absolute terms
because plain 8B is more capable, _not_ because the obfuscation
hurts it less.

### Per-task overlap, 8B h=128 β=2 vs plain 8B

Of 10 plain passes: 7 also pass under obfuscation, 3 lost
(`HumanEval/6` parenthesis parser, `/57` two-list arithmetic, `/62`
polynomial derivative).
Of 10 plain fails: 1 newly passes (`HumanEval/26` list dedup).
Lost tasks tend to require longer multi-step state; recovered tasks are
shallow-reasoning string ops.

### Attack-harness readings on 8B h=128 β=2

Ran `run_all_m2_7.py` (static + IMA-EmbedRow) against the working cell:

| Attack                   | TTRSR top-1 | TTRSR top-10 | Risk     | Notes                                                          |
| ------------------------ | ----------: | -----------: | -------- | -------------------------------------------------------------- |
| VMA                      |      1.17 % |       3.52 % | low      | 256 test prompts, 36 layers, 7 projection kinds                |
| IA-gate                  |      0.07 % |       0.27 % | low      | 4096 test tokens, layer 0                                      |
| IA-attn                  |      0.02 % |       0.05 % | low      | 4096 test tokens, layer 0                                      |
| IMA-EmbedRow-ridge       | **96.88 %** |      99.22 % | **high** | deprecated reference; over-triggers on identity-fixed specials |
| IMA-EmbedRow-transformer |  **0.00 %** |       0.00 % | low      | paper-faithful trained inverter (Qwen-backbone)                |

The IMA-EmbedRow-ridge 97 % is the known-stale ridge baseline
(see `aloepri_attack_harness_disparities.md` memory note): it
collapses on the ~293 identity-fixed special-token pairs and is
not the paper's actual IMA attack surface. The
paper-faithful transformer inverter — the same attacker that
recovered Qwen3-4B baseline ImaActivation rows in prior cells —
lands at **0.00 %** TTRSR on this cell. Static + IA are also low.

### ISA AttnScore + HiddenState (8B paperK / no-H / pow2 / h=128 / β=2)

Captures: 512 prompts, plain + obf via patched llama-server with
`--tensor-filter '^(kq|kqv_out)-(0|5)$|^attn_norm-(17)$'` + `--flash-attn off`.
Sweep: `diagnose_isa/gpu_sweep.py` on ROCm, row split, 3 seeds × layers
× cells.

| Surface   | Layer |    PLAIN top-1 |          OBF top-1 |     Δ defense |
| --------- | ----: | -------------: | -----------------: | ------------: |
| `kq`      |     0 | 49.02 % ± 0.61 | **48.56 % ± 0.67** |      +0.46 pp |
| `kq`      |     5 | 39.52 % ± 0.51 | **39.59 % ± 0.65** |      −0.07 pp |
| `kqv_out` |     0 | 93.02 % ± 0.93 | **82.26 % ± 1.38** | **+10.76 pp** |
| `kqv_out` |     5 | 50.23 % ± 0.93 | **49.44 % ± 1.22** |      +0.79 pp |

**Reading.** `kq` defense is essentially zero — predicted exactly by
the β-bifurcation (paper-literal-K at β=2 reduces to default-K, so
`Q · Kᵀ` cancels just like the un-paper-K construction). Ûvo
pow2-monomial drives +10.76 pp at `kqv_out` L=0 but vanishes by L=5
(+0.79 pp). All four absolute obf TTRSR numbers stay in the
48–82 % band — far above any "defended" threshold — so this surface
is **not effectively protected by static obfuscation alone**.
Matches the 4B β=2 pattern (default-like AttnScore at L=0,5 per
prior Cell E micro-probe).

#### ISA HiddenState multikey K=64 at attn_norm-17

Paper-faithful labelled-ridge ISA, attacker seed 20260521, vendor
CPU keymat pool. Ridge solve attempted on GPU; fell back to CPU
LAPACK on a `RuntimeError` (OOM at 28928 × 4352 normal-equations
form). CPU fallback is correctness-preserving.

| Cell                                     | top-1 TTRSR | top-10 TTRSR | best α | risk       |
| ---------------------------------------- | ----------: | -----------: | -----: | ---------- |
| 4B β=2 / no-H / pow2 / h=128 (prior ref) |      3.89 % |      20.92 % |   1e-4 | low        |
| **8B β=2 / no-H / pow2 / h=128**         | **10.22 %** |      20.68 % |   1e-4 | **medium** |

The **8B HiddenState defense is _worse_ than 4B's** under the same
recipe — top-1 jumps from 3.89 % (low) to 10.22 % (medium). This
re-confirms the chronicle's 2026-05-21 finding: at d=4096, Ûvo
attenuation drops sharply (4B: 33 % relative; 8B: 7.5 % relative)
because the ridge attacker has more observed coordinates to fit.
Larger d does **not** automatically buy more HiddenState privacy
— it gives the attacker more leverage too.

### What the 8B data falsifies

The prototype-doc § _Model-size scaling tradeoff_ hypothesised that 8B's
extra d=4096 headroom would let us run stronger Q/K perturbations than 4B
tolerates. Measurement disagrees:

- **h does not scale with d.** Pushing h from 128→256 on 8B drops
  pass@1 from 40 %→0 %, even though 2h/d is similar to the 4B operating
  point (12.5 % vs 10 %). h=128 is the binding ceiling at both model
  sizes.
- **β=4 fails on 8B too.** Quality collapses at β=4 regardless of h,
  matching the 4B β-ramp's β≥4 cliff. Larger d does not absorb the
  Q/K perturbation magnitude either.
- **No 8B-specific operating point opens up.** The only working
  (quality + accuracy + paper-literal-K) cell on 8B is h=128/β=2,
  same combinatorics as 4B's working cell. We get +33 % relative
  HumanEval at 8B (8/20 = 40 % vs 4B 6/20 = 30 %) just because plain
  8B is more capable; the obfuscation deficit (relative to plain) is
  not smaller.

### Working conclusion update

The β-ramp is exhausted on both 4B and 8B. There is no useful operating
point along the β axis under paper-literal-K + matrix-Γ + Ûvo pow2 on
Qwen3 dense. Two structural reasons make this expected (see
`docs/research/aloepri-h-beta-interaction-2026-05-27.md` for full
derivation):

1. **β bifurcation.** Window permutations Z are involutive iff every window
   has fewer than 3 mobile elements. β=2 ⇒ Z² = I always, so
   paper-literal-K reduces to the default construction at the score
   surface. β=4 is the first β where Z² ≠ I generically — and it kills
   generation.
2. **h drives layer-depth compounding.** K_d's condition number scales
   with √(h/d). Through Qwen3's L=36 transformer blocks the per-layer
   distortion compounds multiplicatively, so a 37 % κ increase
   (h=128→256) becomes a ~10⁵× amplification at the final logit.

Future viable defense must change the Q/K perturbation kind, not the
β magnitude (e.g. low-rank R̂_qk additive deltas; randomised mixed-β
windows where some windows stay involutive; cross-layer correlated
keys to break compounding). See the research note for candidates.

## Conclusions (full sweep, 4B + 8B, 2026-05-27)

### Accuracy axis — what we learned

1. **The only viable defended cell across 4B + 8B is paper-literal-K /
   no-Hadamard-signs / pow2-monomial UVO / h=128 / β=2.** Every other
   (h, β) point we measured either failed the readability gate (β≥4 at
   any h; non-H-paper-K / no-R variants), produced readable but
   task-incoherent output (h=256 at β=2 on 8B: 0/20 HumanEval), or
   damaged utility below the working-cell baseline (dense Ûvo at any
   β: 3/20 on 4B).

2. **Plain-baseline accuracy through this `/completion` pipeline is
   50 % on 8B at n=20** — `<think>` tokens never emit in raw
   completion mode (0/20 of plain runs), so the comparison to
   obfuscated cells is a clean A/B against the model's true
   base-completion capability, not a thinking-mode-confused
   measurement.

3. **The HumanEval gap of −10 pp at 8B (40 % obf vs 50 % plain) is
   pure obfuscation deficit, not a measurement artefact.** Same wall
   time (≈ 11 min for n=20 either way), same tokenizer, same prompts,
   same temperature=0 / seed=0 / max_tokens=384 settings, same `/completion`
   path, same `_PlainClient` shape as the obfuscated `AloePriClient`.

4. **Larger d (=4096) does not give us a stronger defense lever.**
   The prior "Model-size scaling tradeoff" hypothesis predicted that
   8B's extra residual headroom would absorb h=256 and/or β=4.
   Empirics: h=256 destroys 8B harder than expected (0/20 even when
   the readability heuristic passes), and β=4 collapses generation at
   every h. The accuracy-preserving cell on 8B is the _same_ recipe
   as on 4B; the absolute pass@1 number rises only because plain 8B
   is more capable.

5. **The failure mode at h=256 / β=2 is mode-collapse looping, not
   token salad.** Completions like
   `if not isinstance(strings, list): return None` repeated 15× look
   syntactically Python-shaped because the prompt-conditioned step
   from `def foo(strings, ...):` is intact for the first ~10 tokens.
   The decoder loses multi-step state once the prompt-condition fades,
   and at temperature=0 / seed=0 it falls into a deterministic
   low-entropy attractor. This isn't fixed by giving the model more
   capacity (8B failed harder than 4B) and isn't fixed by raising
   max_tokens (the loop is per-token deterministic so it never
   escapes).

### Defense axis — what the working cell actually buys

On 8B paperK / no-H / pow2 / h=128 / β=2, full static + IMA-EmbedRow
sweep:

| Attack                                                 |                             TTRSR top-1 | Risk                                                  |
| ------------------------------------------------------ | --------------------------------------: | ----------------------------------------------------- |
| VMA                                                    |                                  1.17 % | low                                                   |
| IA-gate / IA-attn                                      |                         0.07 % / 0.02 % | low                                                   |
| IMA-EmbedRow-transformer (paper §F.1 trained inverter) |                                  0.00 % | low                                                   |
| Per-head fingerprint attn_q / attn_o                   |                         3.73 % / 3.21 % | low (≈ 32-head random floor 3.13 %)                   |
| Per-head fingerprint attn_k / attn_v                   |                       13.19 % / 13.89 % | medium                                                |
| V/O channel-pair vo_v / vo_o / vo_pair                 |               12.50 % / 3.21 % / 3.82 % | medium / low / low                                    |
| ISA AttnScore `kq` L=0 / L=5 (obf vs plain Δ)          |  48.56 % / 39.59 % (Δ +0.46 / −0.07 pp) | **structurally undefended** (β-bifurcation confirmed) |
| ISA AttnScore `kqv_out` L=0 / L=5 (obf vs plain Δ)     | 82.26 % / 49.44 % (Δ +10.76 / +0.79 pp) | high absolute; +10.76 pp from Ûvo at L=0              |
| ISA HiddenState multikey K=64 at attn_norm-17          |                             **10.22 %** | **medium** (worse than 4B β=2's 3.89 %, low)          |

The K/V-side ~13 % numbers reflect **head-identity** recoverability
(which physical head a tensor row belongs to), _not_ row-content
recovery — Q/O sides sit at the 32-head random-guess floor (3.13 %).
This is incidental statistical leak from the bf16-commuting
pow2-monomial UVO (signed permutation × power-of-two scaling
preserves L2 ranks per channel by construction); no surface lands
in the "high" risk band.

### The β-bifurcation pre-empts further β-ramp work

Per `docs/research/aloepri-h-beta-interaction-2026-05-27.md` § 1.3,
under paper-literal-K with H=I:

```
Prob(Ẑ² = I globally)   = 1                     (β=2)
                        = 4.4 × 10⁻⁷           (β=4)
                        = 1.7 × 10⁻¹⁴          (β=8)
```

So paper-literal-K is _exactly identical_ to default-K at β=2 (zero
K-side score defense, regardless of d), and the K-side surface is
_certainly_ distorted at β≥4 (where it also kills generation). The
"smooth midpoint" we kept hoping for does not exist in the
fixed-window sampler. **No further β-ramp measurements are
informative**; the next defense lever has to change the _kind_ of
Q/K perturbation, not its magnitude.

### Recommendations

1. **Adopt 8B paperK / no-H / pow2 / h=128 / β=2 as the current
   8B operating cell.** It is the only quality-coherent point on
   the (h, β) surface we measured and lands all static / static-
   inversion attacks in the low-risk band. HumanEval n=20 = 8/20 =
   40 % (−10 pp vs plain).

2. **Stop ramping β.** The bifurcation is structural. Time spent on
   β=3, β=5, mixed-β-ramps within the current sampler is
   bounded-zero return.

3. **Next defense exploration should pursue one of:**
   - **Mixed-window γ-sampler** — most windows involutive, γ fraction
     non-involutive. Rank-2 score perturbation. 1-day patch in
     `lib/alg2.py::generate_block_perm`.
   - **Low-rank additive R̂_qk delta** — `S → S + α · u · vᵀ` per
     layer. Smoothly tunable in α. 1-day patch.
   - **Anti-correlated cross-layer keys** — pair layers so per-layer
     perturbations telescope; compound bound moves from O(L·δ) to
     O(δ). 3-day patch.
   - **Layer-shared Ẑ_block** — one global draw, applied at every
     layer. Rank-1-in-depth perturbation. 3-day patch.

   Each comes with a concrete prediction in
   `docs/research/aloepri-h-beta-interaction-2026-05-27.md` § 5.

4. **For protection on `kq` and `kqv_out` at L=0** (where paper-
   literal Alg2 still leaks ~43–47 % under row-split ridge), the
   gold-standard answer remains **TEE-protected attention (path-1)**.
   Even the best Q/K-side static defense cannot close the L=0
   embedding-noise shadow on the attention surface. AloePri's path-2
   protocol is best read as "as much static defense as Qwen3 dense
   can absorb without depth-compounding into generation failure" —
   _not_ as a strict replacement for path-1.
