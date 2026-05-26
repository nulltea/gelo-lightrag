# AloePri pow2-monomial UVO findings — 2026-05-26

## Question

The default `--alg2-u-vo` cell preserved semantic readability but hurt HumanEval relative to the non-UVO canonical cell. The working hypothesis was that UVO is algebraically exact but bf16 storage makes the V/O inverse-pair cancellation lossy.

This note documents the mathematical diagnosis, implementation changes, utility experiments, and attack-harness results for the new bf16-friendly UVO family.

## Executive summary

Older obfuscation setups were optimizing attack-defense knobs that are exact or mostly benign in real arithmetic, but not in the deployed bf16 GGUF model. In particular, dense UVO transforms preserve the value/output product algebraically, yet store two independently rounded transformed matrices. The lost bf16 mantissa bits cannot be recovered by the inverse transform, so accuracy degrades even though the symbolic construction says the function should be unchanged.

The best current configuration is:

```text
untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-pow2e1-bf16-native.gguf
```

It uses h128, beta8, Alg2 matrix-gamma/Hadamard keymat, and `pow2-monomial` UVO with exponent range +/-1. It passed the quality gate and scored 6/20 on HumanEval n=20, matching the non-UVO canonical cell and improving over default dense-ish UVO's 3/20.

Completed attack-harness status for this config:

- Static VMA/IA: low risk.
- Token TFMA/SDA: low risk.
- IMA EmbedRow transformer: low risk.
- IMA EmbedRow ridge: high risk, top1 0.5547.
- Runtime ISA row-split `kq`: high recovery / not defended, as expected for a Q/K surface.
- Runtime ISA row-split `kqv_out`: L0 defense preserved (~97.46% plain -> ~82.45% obf), later layers mostly neutral; per-head L17 obf mean 12.86%, max 19.11%.

## Root issue: defense knobs ruined accuracy through non-commuting quantization

The recurring pattern is:

1. The paper-level transformation is exact in real arithmetic.
2. The implementation materializes transformed weights into bf16 GGUF tensors.
3. Some transformations require dense mixing or ill-conditioned inverse pairs.
4. bf16 rounding happens before inference, so inverse-pair cancellation is only approximate.
5. Runtime accuracy degrades even when the algebraic invariant is correct on paper.

For UVO:

```text
W_v' = U^T W_v
W_o' = W_o U^{-T}
```

Real arithmetic gives:

```text
X W_v'^T W_o'^T = X W_v^T U U^{-1} W_o^T = X W_v^T W_o^T
```

The deployed model instead uses:

```text
B_v = Q_bf16(U^T W_v)
B_o = Q_bf16(W_o U^{-T})
output = X B_v^T B_o^T
```

Dense `U` mixes many channels before bf16 rounding. Even when `cond(U)=1`, the rounded product is not the same as the original product. If `U` is raw Gaussian or QR plus perturbation, condition number and inverse amplification make the problem worse.

The bf16-safe alternative is to use transforms that commute with bf16 storage. Signed permutations commute exactly. Power-of-two monomial transforms commute for normal-range bf16 values:

```text
Q_bf16(2^k w) = 2^k Q_bf16(w)
```

That is why `pow2-monomial` recovers accuracy while preserving an explicit UVO channel transform.

## Configs compared

Baseline artifacts:

- Non-UVO canonical: `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-bf16-native.gguf`
- Default UVO: `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf`

New artifacts:

- `pow2-monomial`, h128, beta8: `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-pow2e1-bf16-native.gguf`
- signed-permutation, h128, beta8: `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-signedperm-bf16-native.gguf`
- `pow2-monomial`, h128, beta1: `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-beta1-uvo-pow2e1-bf16-native.gguf`
- `pow2-monomial`, h256, beta8: `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h256-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-pow2e1-bf16-native.gguf`

All runtime experiments were launched through `evals/aloepri-attacks/m2_7/spawn_obfuscated_server.sh`; server logs confirmed `Vulkan0 : Radeon 8060S Graphics`.

## Mathematical diagnosis

UVO is exact in real arithmetic. For the value/output path:

```text
W_v' = U^T W_v
W_o' = W_o U^{-T}
```

Then:

```text
X W_v'^T W_o'^T = X W_v^T U U^{-1} W_o^T = X W_v^T W_o^T
```

The deployed bf16 model stores:

```text
B_v = Q_bf16(U^T W_v)
B_o = Q_bf16(W_o U^{-T})
```

so the real deployed path is:

```text
X B_v^T B_o^T
```

Dense orthogonal UVO fixes condition number but not the core rounding issue: `U^T W_v` is a dense sum across channels and then gets rounded to bf16. The inverse in `W_o` cannot recover the lost mantissa bits.

The better bf16-safe family is monomial power-of-two UVO:

```text
U = signed_permutation * diag(2^k_i)
```

For normal-range bf16 values:

```text
Q_bf16(2^k w) = 2^k Q_bf16(w)
```

so the transform approximately commutes with storage quantization. It preserves near-zero extra V/O storage error while adding channel permutation and magnitude perturbation.

Microprobe summary:

| UVO family | condition | incremental bf16 V/O error |
| --- | ---: | ---: |
| signed permutation | 1 | ~0 |
| pow2 monomial, exp ±1 | 4 | ~0 |
| pow2 monomial, exp ±2 | 16 | ~0 |
| dense orthogonal / Hadamard / QR | 1 | ~0.33% per V/O pair |
| current QR + 0.05 Gaussian perturb | ~5-6 | worse |
| raw Gaussian | median ~707, worst observed ~28501 | often catastrophic |

## Lever cross-effects

The useful way to categorize the knobs is by which term they perturb: model utility, static recovery surfaces, or runtime attention/output recovery surfaces.

### Accuracy-positive or accuracy-preserving

| lever | accuracy effect | defense effect | reason |
| --- | --- | --- | --- |
| `pow2-monomial` UVO, exponent +/-1 | positive vs dense UVO; observed HumanEval 6/20 | still under runtime test | commutes with bf16 scaling, avoiding dense pre-rounding error |
| signed-permutation UVO | mostly preserving; HumanEval 5/20 | weaker than pow2 if magnitude cues matter | exact channel relabeling with no bf16 loss |
| h128 with beta8 | current utility-safe point | moderate defense capacity | enough expansion/noise without pushing layer outputs off manifold |
| bf16-native storage with monomial transforms | preserving | compatible with deploy target | avoids requiring f32/f64 artifacts that the real runtime will not use |

### Accuracy-negative

| lever | observed accuracy effect | defense motivation | why it hurts |
| --- | --- | --- | --- |
| default dense-ish UVO `qr-perturb` | HumanEval 3/20 despite readable quality | hides V/O channel basis | dense bf16 rounding plus inverse amplification |
| raw Gaussian UVO | expected catastrophic | closest to some paper-literal readings | high condition number at head_dim=128; inverse amplifies bf16 noise |
| h256 in current stack | failed quality gate | more residual/noise room | current Alg2 approximation/noise scaling becomes off-manifold |
| beta1 with pow2 UVO | readable but semantically worse | less aggressive key scaling/noise interaction probe | not enough current evidence, but quality outputs lost semantic reliability |
| dense orthogonal/Hadamard UVO | better than raw Gaussian but still lossy | mixes channels while keeping cond=1 | condition number is not enough; dense bf16 rounding loses information |

### Defense-positive but utility-risky

| lever | defense direction | utility risk | mathematical note |
| --- | --- | --- | --- |
| stronger residual expansion/noise | can reduce static alignment and embedding recovery | can break generated semantics | defense comes from randomizing hidden basis, but model layers still expect a narrow activation distribution |
| dense UVO | can reduce output-surface channel matching | hurts bf16 utility | paper-exact only before quantized materialization |
| larger h | may increase nullspace/randomization capacity | h256 failed quality | h is not a direct fix for UVO; it expands keymat/residual path, while UVO acts inside head_dim |
| non-monomial scaling | may confuse magnitude-based attacks | high inverse/rounding risk | arbitrary scaling does not commute with bf16 |

### Defense-negative or invariant surfaces

| lever/surface | effect | why |
| --- | --- | --- |
| `kq` AttnScore row split | expected hard to defend with V/O-only UVO | attention scores depend on Q/K, not V/O; UVO is invisible to this surface |
| `--split vocab` | ignored by design | user identified it as weaker and not worth optimizing |
| static IMA EmbedRow ridge | still high risk | static embedding-table signal remains available outside the paper runtime threat model |

## Implementation changes

Added UVO modes in `python/aloepri-llm/lib/alg2.py` and exposed them in `python/aloepri-llm/obfuscate_qwen3_gguf.py`:

```bash
--alg2-u-vo-mode {qr-perturb,orthogonal,signed-permutation,pow2-monomial,raw-gaussian}
--alg2-u-vo-pow2-exp E
```

The default remains `qr-perturb` for artifact reproducibility. `pow2-monomial` samples signed channel permutations with power-of-two scales `2^k`, `k in [-E, E]`.

## Utility results

| config | quality gate | HumanEval n=20 |
| --- | ---: | ---: |
| default UVO, h128, beta8, QR-perturb | pass | 3/20 = 15% |
| non-UVO canonical, h128, beta8 | pass | 6/20 = 30% |
| signed-permutation UVO, h128, beta8 | pass | 5/20 = 25% |
| pow2-monomial UVO, h128, beta8 | pass | 6/20 = 30% |
| pow2-monomial UVO, h128, beta1 | readable but semantically worse | not run |
| pow2-monomial UVO, h256, beta8 | failed quality | skipped |

Result files:

- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-b8-20260526/quality-only.json`
- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-b8-20260526/humaneval-n20.json`
- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-signedperm-b8-20260526/quality-only.json`
- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-signedperm-b8-20260526/humaneval-n20.json`
- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-h256-uvo-pow2e1-b8-20260526/quality-only.json`

Main utility finding: `pow2-monomial` recovered the full observed default-UVO HumanEval drop in the n=20 gate: 15% -> 30%, matching the non-UVO canonical score while preserving an explicit UVO transform.

`h=256` is not an accuracy fix in the current stack. It failed the quality gate with repeated underscores/numbers/angle-bracket patterns. Larger residual expansion may still help defense capacity, but it pushes the current RMS/keymat/noise approximation off-manifold before utility is acceptable.

## Current best config and measured results

Current best artifact:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-pow2e1-bf16-native.gguf
```

Key:

```text
/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-pow2e1-bf16-native.gguf.key.npz
```

Parameters:

| parameter | value |
| --- | --- |
| base | Qwen3-4B Q8_0 untied |
| keymat dimension | h128 |
| beta | beta8 |
| noise | ae1.0, ah0.2 |
| Alg2 form | matrix-gamma |
| structured mixer | Hadamard |
| UVO | enabled |
| UVO mode | `pow2-monomial` |
| UVO exponent range | +/-1 |
| storage/runtime target | bf16-native GGUF |

Accuracy:

- Quality gate: pass.
- HumanEval n=20: 6/20 = 30%.
- Relative to default UVO: improves from 3/20 to 6/20.
- Relative to non-UVO canonical: matches 6/20 in this small gate.

Completed attack harness:

| attack family | metric | result | risk |
| --- | --- | ---: | --- |
| Static VMA | top1 | 0.0898 | low |
| Static VMA | top10 | 0.2344 | low |
| Static IA | top1 | 0.0005 | low |
| IMA EmbedRow ridge | top1 | 0.5547 | high |
| IMA EmbedRow ridge | top10 | 0.6016 | high |
| IMA EmbedRow transformer | top1 | 0.0000 | low |
| IMA EmbedRow transformer | top10 | 0.0000 | low |
| Token TFMA | top1 | 0.0078125 | low |
| Token TFMA | top10 | 0.046875 | low |
| Token SDA | BLEU4 | 1.2810572317089155e-05 | low |

Runtime attack-harness result:

- `kq` row split remains high and essentially invariant to pow2 UVO.
- `kqv_out` row split preserves the useful L0 output-surface defense of default UVO while recovering HumanEval, but does not solve later layers.

## Attack results completed so far

Target cell: `pow2-monomial`, h128, beta8.

Static-weight attacks:

- Result: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-b8-20260526/m2_7-static.json`
- VMA: top1 0.0898, top10 0.2344, risk low
- IA: top1 0.0005, gate 0.0005, attn 0.0000, risk low

IMA-EmbedRow attacks:

- Result: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-b8-20260526/m2_7-ima-embedrow.json`
- Ridge: top1 0.5547, top10 0.6016, risk high
- Paper-like transformer: top1 0.0000, top10 0.0000, risk low

Token-stream attacks:

- Captures: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-b8-20260526/m2_7-token-streams.jsonl`
- Result: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-b8-20260526/m2_7-token.json`
- TFMA: top1 0.0078125, top10 0.046875, risk low
- SDA: BLEU4 1.2810572317089155e-05, risk low

## Runtime tensor attack status

The current strongest attack thread is the 512-prompt row-split GPU ridge sweep over `kq` and `kqv_out` surfaces at layers {0, 5, 11, 17, 23}, using `evals/aloepri-attacks/m2_7/diagnose_isa/gpu_sweep.py`.

Capture completed:

- Output dir: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-attn-and-output-512-20260526/`
- Tensor filter: `^(kq|kqv_out)-(0|5|11|17|23)$`
- Corpus: `evals/aloepri-attacks/corpora/release-gate-512.txt`
- Server: flash-attn off, tensor dump enabled, Vulkan backend confirmed
- Capture result: 512 prompts, 5120 tensors

Comparison baselines used:

- Plain kq captures: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-plain-attn-512-20260526/captures/`
- Plain kqv_out captures: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-plain-attnoutput-512-20260526/captures/`
- Default UVO documented baseline: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-attn-512-20260526/captures/` and `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-attnoutput-512-20260526/captures/`

## Interim interpretation

`pow2-monomial` is a real utility improvement over default dense-ish UVO. It addresses the bf16 implementation failure mode directly rather than merely reducing condition number.

It does not yet solve all defense gaps. Static VMA/IA and token attacks pass, but IMA-EmbedRow ridge remains high. That ridge attack is a static embedding-table attack and has previously been treated as a stricter/out-of-paper threat-model probe, but as a raw attack metric it remains a fail.

The decisive runtime question is now partly answered: `pow2-monomial` preserves the useful default-UVO L0 `kqv_out` defense while recovering quality, but `kq` remains high as predicted by the AttnScore invariance theorem and later `kqv_out` layers remain mostly undefended.

## Runtime attack results as they arrive

This section is intentionally append-friendly: it records live attack-harness results before waiting for every long-running probe to finish, so context compaction does not lose the numbers.

### `kq` row-split ISA AttnScore, 512 prompts, 10 seeds

Command ran through `evals/aloepri-attacks/m2_7/diagnose_isa/run_gpu.sh`, so PyTorch used ROCm with device name `Radeon 8060S Graphics`. This is the expected invariant/hard surface for V/O-only UVO.

| layer | cell | top1 mean | top1 std | top10 mean | n |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | PLAIN | 48.63% | 0.80 | 58.20% | 10 |
| 0 | OBF | 47.20% | 0.85 | 54.32% | 10 |
| 5 | PLAIN | 38.68% | 0.81 | 51.80% | 10 |
| 5 | OBF | 38.19% | 0.69 | 49.98% | 10 |
| 11 | PLAIN | 27.73% | 0.79 | 43.50% | 10 |
| 11 | OBF | 26.93% | 0.69 | 43.19% | 10 |
| 17 | PLAIN | 22.41% | 0.81 | 38.31% | 10 |
| 17 | OBF | 22.02% | 0.64 | 37.82% | 10 |
| 23 | PLAIN | 30.12% | 0.82 | 44.55% | 10 |
| 23 | OBF | 29.51% | 0.57 | 44.36% | 10 |

Per-head ridge at L17: plain mean 9.87%, max 11.66%; obfuscated mean 9.31%, max 11.03%.

Interpretation: pow2 UVO does not materially defend `kq`, as expected. The L0 delta is only -1.43 pp, and later-layer deltas are also tiny.

### `kqv_out` row-split ISA AttnScore, 512 prompts, 10 seeds

Live progress file:

```text
evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-attn-and-output-512-20260526/logs/gpu_sweep_kqv_out.progress.jsonl
```

Completed layer sweep so far:

| layer | cell | top1 mean | top1 std | top10 mean | n |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | PLAIN | 97.46% | 0.40 | 99.83% | 10 |
| 0 | OBF | 82.45% | 0.57 | 90.08% | 10 |
| 5 | PLAIN | 50.79% | 0.89 | 61.62% | 10 |
| 5 | OBF | 48.26% | 0.83 | 59.15% | 10 |
| 11 | PLAIN | 18.50% | 0.40 | 31.86% | 10 |
| 11 | OBF | 18.55% | 0.52 | 30.38% | 10 |
| 17 | PLAIN | 16.68% | 0.45 | 27.82% | 10 |
| 17 | OBF | 16.68% | 0.59 | 26.70% | 10 |
| 23 | PLAIN | 25.29% | 0.48 | 39.29% | 10 |
| 23 | OBF | 26.40% | 0.97 | 39.35% | 10 |

Interpretation so far: pow2 UVO keeps the important L0 `kqv_out` runtime-output defense while recovering accuracy. L0 drops from plain 97.46% to obfuscated 82.45%, comparable to the older default UVO L0 result (~83.77%) but without the HumanEval regression. At L5 and above, defense deltas are small or neutral, so the remaining runtime threat is not solved.

Per-head `kqv_out` at L17: plain mean 11.81%, median 11.72%, p90 13.51%, max 17.50% at head 12; obfuscated mean 12.86%, median 12.89%, p90 15.11%, max 19.11% at head 19. Interpretation: pow2 UVO does not improve the per-head L17 runtime-output probe; if anything, the strongest single obfuscated head is higher than plain.

## Attack logging transparency fix

`evals/aloepri-attacks/m2_7/diagnose_isa/gpu_sweep.py` now supports live progress logging:

```bash
--progress-jsonl PATH
--quiet-progress
```

The script flushes human-readable progress lines and appends JSONL events for start/device/embed load, each layer/cell load, every seed start/end, running means, layer summaries, per-head seed results, per-head summaries, and final `done`. This prevents long ROCm-container runs from becoming opaque and preserves intermediate results if the session is compacted or interrupted.

Current live/completed files:

- Full event stream: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-attn-and-output-512-20260526/logs/gpu_sweep_kqv_out.progress.jsonl`
- Compact summary: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-attn-and-output-512-20260526/logs/gpu_sweep_kqv_out.summary.json`

## Next actions

1. Treat `pow2-monomial` UVO h128 beta8 as the current best utility-preserving UVO config.
2. Do not spend time defending `--split vocab`; it is not the strongest threat model.
3. For <=15% runtime recovery, focus beyond V/O-only UVO: Q/K-side defenses, per-layer/per-head transformations that affect `kq`, and static EmbedRow ridge mitigation.
4. If exploring UVO further, prefer bf16-commuting families: signed permutations, power-of-two monomial, and blockwise pow2 monomial. Avoid dense QR/Gaussian UVO unless stored at higher precision or fused so quantization happens after cancellation.
5. Use `gpu_sweep.py --progress-jsonl` for all future long attack sweeps so intermediate results survive interruptions.
