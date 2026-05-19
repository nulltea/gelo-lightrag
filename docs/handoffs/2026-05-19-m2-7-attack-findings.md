# Handoff — M2.7 attack-resistance findings on §05 obfuscated GGUF

**Date:** 2026-05-19
**Status:** measurement complete. Next session: triage the failures, not run more attacks.

## TL;DR

Two attacks **fail** §6.3's 15 % gate on the §05 deployment
(`keymat-h128-pi-noise-alg2-fp32.gguf`); five attacks pass.
The two failures share a single root cause: **partial Algorithm 2 is
deployed without the intra-head transforms (R̂_qk, Ĥ_qk, Ẑ_block),
so layer-0 hidden state remains inverter-recoverable**.

Don't run more attacks until that's fixed.

## Full attack ledger

| Attack | Plain | Obfuscated | §6.3 gate | Outcome | Capture |
|---|---:|---:|---|---|---|
| NN | 100.0 % | 0.0 % | ≤ 15 % | ✓ Pass | 64-prompt vocab-disjoint |
| IMA basic | 98.1 % | **88.9 %** | ≤ 15 % | **✗ FAILS 6×** | 64-prompt vocab-disjoint |
| **IMA paper-like (NEW)** | **49.3 % @N=256** / **72.8 % @N=512** | **38.0 % @N=256** / **46.8 % @N=512** | ≤ 15 % | **✗ FAILS 2.5–3×** | 512-prompt prompt-disjoint |
| ISA HiddenState | 8.7 % | **16.3 %** | ≤ 15 % | **✗ FAILS marginally** | 64-prompt vocab-disjoint |
| ISA AttnScore | 2.9 % | 0.0 % | ≤ 15 % | ✓ Pass | 64-prompt vocab-disjoint |
| TFMA | n/a | 1.6 % top-1 / 5.1 % top-10 | ≤ 15 % | ✓ Pass | 64-prompt token-stream |
| SDA | n/a | BLEU-4 = 1.6×10⁻⁵ | ≤ 5 BLEU | ✓ Pass | 64-prompt token-stream |
| VMA, IA | — | — | — | out of scope (weight privacy) | — |

## The three failures explained

### 1. IMA basic at 88.9 % obfuscated

**What it tests.** Multi-α linear ridge regression from layer-0
hidden state (`attn_norm-0`) to plaintext token embeddings. Vocab-
disjoint train/val/test splits (paper-faithful — generalising
attacker, not a memoriser).

**Why it fails.** The §05 keymat expansion adds 256 decoy dimensions
(`d_eff = 2048 + 2·128 = 2304`). Paper Algorithm 2 cancels these via
the intra-head R̂_qk / Ĥ_qk / Ẑ_block transforms. **§05 ships only
the inter-head shuffle (Π_head), not the intra-head transforms.**
The decoy dims therefore carry recoverable structure that a single
linear fit (with dim-bridging 2304→2048) recovers cleanly.

**Plain comparison.** Plain hits 98.1 % — the attack is genuinely
strong; obfuscation only knocks ~9 pp off. Paper Table 1 reports
0 % with full Algorithm 2 on Qwen2.5-14B.

### 2. IMA paper-like at 38–47 % obfuscated *(measured this session)*

**What it tests.** Paper §F.1's canonical IMA: a trained 2-layer
8-head transformer inverter (inverter_hidden=256, 2 epochs default;
we used **epochs=16** to drive the inverter to fit). Prompt-disjoint
75/25 split.

**Why it matters.** This is a **stronger** attacker than IMA basic
(non-linear, more capacity). If linear ridge already breaks the
defence, the transformer inverter should do at least as well — and
it does, just more sample-hungry.

**Data-scale sweep (epochs=16, default hyperparameters otherwise):**

```
  N   |  Plain top1 | Obf top1  | Plain top10 | Obf top10
 -----+-------------+-----------+-------------+-----------
   64 |   7.6 %    |   6.2 %  |  26.2 %     |  17.8 %  ← both undertrained
  128 |  24.4 %    |  11.5 %  |  45.5 %     |  37.9 %  ← plain crosses 15 % gate
  192 |  39.1 %    |  34.1 %  |  54.9 %     |  47.1 %
  256 |  49.3 %    |  38.0 %  |  66.3 %     |  47.5 %  ← paper-matched data scale
  384 |  62.4 %    |  47.6 %  |  76.7 %     |  55.3 %
  512 |  72.8 %    |  46.8 %  |  87.9 %     |  55.1 %  ← saturating
```

**Minimal viable N for this attack:** **N=128** — the smallest
corpus where plain TTRSR crosses the 15 % paper gate. At paper's
data scale (~N=256, ~3 400 train rows ≈ paper's 128 × 32 = 4 096),
plain reaches 49.3 % and obfuscated 38.0 %.

**Defence ratio** (1 − obf/plain): 12–36 % across the sweep —
obfuscation knocks at most a third off the attack. Paper Table 1
reports 0 % with full Algorithm 2 on Qwen2.5-14B.

**Same root cause as IMA basic.** Both inverters target layer-0
hidden state; both succeed because the intra-head transforms are
missing. The transformer inverter just needs more data to fit (it
has more parameters than the ridge); once it does (N ≥ 192), the
defence ratio is comparable to ridge.

### 3. ISA HiddenState at 16.3 % obfuscated *(higher than plain!)*

**What it tests.** Multi-α ridge at layer 23 (deep). Vocab-disjoint
splits.

**Why it fails.** The keymat expansion's 256 decoy dims persist
through to layer 23 (the residual stream is wide). Plain ceiling
at layer 23 is only 8.7 % — the deep hidden state is heavily
processed and far from the embedding. But on §05 obfuscated, the
2304-dim observation gives the dim-bridging ridge **more usable
structure than plain**, so TTRSR *rises* to 16.3 %. Paper Table 3
ablation: "Noise only" reads 40 % HS on Qwen2.5-14B, "Noise +
KeyMat" drops to 0.82 %, "Noise + KeyMat + Head&BlockPerm" hits 0 %.
We're between "Noise only" and "Noise + KeyMat" in defensive depth
on this surface, even with our (partial) Head perm.

**This is the strangest finding.** Obfuscation makes the attack
*easier*. The decoy expansion is a defensive footgun without the
intra-head transforms.

## The shared root cause — one diagnosis, three symptoms

§05 deploys:
- ✅ keymat h=128 expansion
- ✅ Π token permutation
- ✅ α-noise (per-layer noise injection)
- ✅ Algorithm 2 *inter-head* shuffle (Π_head)
- ❌ Algorithm 2 *intra-head* transforms (R̂_qk, Ĥ_qk, Ẑ_block)

The missing intra-head transforms are what make Algorithm 2 close
the gap between "Noise + KeyMat" (0.82 % HS) and "full Algorithm 2"
(0 % HS) in paper Table 3. Without them, the keymat decoy expansion
is recoverable structure rather than noise.

**Next session's task is to land the intra-head transforms**, not
to run more attacks against the broken deployment. See
`docs/plans/handoff-aloepri-quantisation-and-alg2-gaps.md` for the
prior pass at scoping that work.

## What passes — and what those passes tell us

- **NN** (cosine retrieval): the rotation alone defeats naïve
  truncation. Trivially expected.
- **ISA AttnScore at 0.0 %**: head-shuffle alone defends the
  `Q·Kᵀ` surface against a generalising ridge attacker — no
  intra-head transforms needed at this surface for this attack.
  Useful data point: paper Table 3 said full Algorithm 2 was
  needed; our data says head-shuffle suffices for AttnScore.
- **TFMA / SDA**: τ permutation defeats token-frequency and
  bigram-substitution attacks on the wire-side response stream.
  Behaviour matches paper Table 4 within ~1 pp.

These three passes establish that the obfuscation pipeline *works*
for the attack surfaces that don't depend on the missing intra-head
transforms. The deployment is one component short, not fundamentally
broken.

## Artifacts produced this session

**Code (new):**
- `evals/aloepri-attacks/m2_7/build_corpus_512.py` — extends the
  64-prompt hand-curated corpus to 512 via filtered PIQA.
- `evals/aloepri-attacks/m2_7/sweep_ima_paper_like.py` — view-
  truncates a single capture and sweeps the IMA paper-like inverter
  over corpus sizes. Doesn't re-capture, just re-trains.
- `evals/aloepri-attacks/corpora/release-gate-512.txt` — 512 prompts,
  mean 15.4 words. First 65 = original hand-curated; rest = PIQA
  goal+solution pairs filtered to 6–25 words.

**Captures (new, ~160 MB total — not committed):**
- `evals/aloepri-attacks/snapshots/m2_7-plain-512/`
- `evals/aloepri-attacks/snapshots/m2_7-hidden-512/`
  Both contain `hidden.safetensors` + `hidden.meta.json` (attn_norm-0
  only — 512 prompts × ~14 tok mean × 2048 (plain) or 2304 (obf) dims).

**Results JSONs (new — small, worth committing):**
- `evals/aloepri-attacks/results/m2_7-ima-paper-like-sweep-plain.json`
  (epochs=4)
- `evals/aloepri-attacks/results/m2_7-ima-paper-like-sweep-plain-e8.json`
- `evals/aloepri-attacks/results/m2_7-ima-paper-like-sweep-plain-e16.json`
- `evals/aloepri-attacks/results/m2_7-ima-paper-like-sweep-obf-e16.json`

**Doc updates:**
- `docs/prototype/aloepri-llm.html` §08:
  - IMA paper-like row replaced (was "0% undertrained" placeholder →
    now full sweep + headline numbers + verdict).
  - Acceptance gates: `ima_paper_like_obfuscated_below_15pct` now
    ✗ 38 % @N=256 / 46.8 % @N=512; `ima_paper_like_plain_at_least_50pct`
    now ✓ 49.3 % @N=256 / 72.8 % @N=512.

## Next session — recommended focus

**Don't run more attacks against this deployment.** All informative
attacks have been exercised. Two fail by a wide margin (IMA basic,
IMA paper-like), one fails marginally (ISA HS). The remediation is
the same for all three: deploy the full Algorithm 2 intra-head
transforms.

**Suggested work, in priority order:**

1. **Read `docs/plans/handoff-aloepri-quantisation-and-alg2-gaps.md`**
   — prior session's scoping of the Algorithm 2 gap. Confirm the
   listed missing transforms still match the current code.
2. **Plan the intra-head transform deployment.** The paper specifies
   R̂_qk, Ĥ_qk, Ẑ_block in §F.1. Decide where they live (build-time
   GGUF rewriter vs runtime in the patched llama.cpp).
3. **Rebuild the GGUF as `keymat-h128-pi-noise-alg2-FULL-fp32.gguf`**
   (or equivalent slug) — one of the rewriter scripts under
   `python/path-2/` likely needs to grow the intra-head step.
4. **Re-run M2.7 against the full Algorithm 2 GGUF.** Both
   IMA-basic + IMA paper-like should drop to ≤ 15 %. ISA HS should
   drop below plain ceiling (≤ 8.7 %). All other passes should hold.
   The sweep harness from this session is ready to re-run unchanged.

**Dropped follow-up:** the prior handoff scoped a 256-prompt ISA
re-run as the remaining test surface. That's no longer load-bearing
— the same intra-head-transforms fix closes both the IMA failures
(large magnitude) and the ISA HS marginal one. No new attack data is
needed before the fix; the harness will collect it after.

## Working-tree state at handoff

```
 M Cargo.toml                                         # untouched this session
 M docs/prototype/aloepri-llm.html                    # §08 updated
?? docs/plans/handoff-aloepri-quantisation-and-alg2-gaps.md
?? docs/prototype/aloepri-attack-harness-findings.md
?? docs/prototype/aloepri-attack-harness-followups.md
?? evals/                                             # full M2.7 harness + captures
?? docs/handoffs/2026-05-19-m2-7-attack-findings.md   # this doc
```

Plus uncommitted changes in `vendor/llama.cpp/common/{common.h, common.cpp, debug.h, debug.cpp, arg.cpp}` (M2.7 tensor-dump patch).

**Recommended commit split for next session:**
1. **vendor/llama.cpp tensor-dump patch** — 5 files, ~80 lines, the
   M2.7 capture hook.
2. **evals/aloepri-attacks/ harness + result JSONs + corpus + §08
   doc** — everything else.

**Don't commit** the large capture safetensors
(`evals/aloepri-attacks/snapshots/m2_7-{plain,hidden,plain-512,hidden-512,tokens}/`
+ `qwen3-1.7b-*/`). Add a `snapshots/` entry to `.gitignore`. Result
JSONs are small and worth committing as ground truth.

## Boundaries the session was operating under (still apply)

- Don't kill containers we didn't spawn (`llama-swap` is untouched).
- Ask before running benchmarks (each significant run authorised
  explicitly this session).
- Weight-privacy attacks (VMA + IA) are out of scope.
- No path-1 (GELO) references in `aloepri-llm.html`.

## Suggested skills for next session

- `/grill-with-docs` against AloePri's `vendor/aloepri-py/` to nail
  down the exact intra-head transform formulae before reimplementing.
- `/diagnose` if the rebuilt GGUF still fails — the layer-0 finding
  is structural enough that a "still fails" result would need a
  closer look at how the rewriter writes the transforms.
- Direct execution otherwise — the M2.7 capture + attack harness is
  ready to re-run unchanged against a fixed GGUF.
