---
type: handoff
status: current
created: 2026-05-26
updated: 2026-05-26
tags: [aloepri, attacks, c1, arrowmatch, sequence-ima]
---

# Handoff — three new deobfuscation attacks (C1, ArrowMatch, Sequence-IMA)

**Date:** 2026-05-26 (late)
**Branch:** `path-2-aloepri-gemma`
**Commit:** `d20b34b path-2: 3 new deobfuscation attacks`
**Builds on:**

- `2026-05-26-isa-attnscore-theorem-and-paper-disparity.md` (theorem + 2B.1)
- `2026-05-26-alg2-paper-literal-defense-gap.md` (paper-literal Alg2 + Lever 1 vocab-disjoint)

This handoff covers a research+implement session that added three new
deobfuscation-attack drivers to `evals/aloepri-attacks/m2_7/`. Per
session directive, **no attack runs were executed in this session** —
all three are implementation-only deliverables ready for next-session
empirical work.

## Why these three (and not others)

Coming out of the prior session's two findings — (1) paper Table 4's
"AttnScore = 0%" is reproducible under paper-literal Alg2 +
vocab-disjoint methodology, and (2) the 87.14 % "Noise + KeyMat"
baseline is still unexplained — a literature scan + internal
prioritisation identified four classes of un-tested deobfuscation
attacks (see §"Research notes" below). The user grilled through the
priority list and committed to implementing three:

| #   | Attack                                            | Why chosen                                                                                                                                                                                                                                                                                                                              | Deferred items |
| --- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| 1   | **C1: TFMA-seeded ridge**                         | Cheapest to land; directly tests the load-bearing claim in `aloepri-attacks.md` §"Why AloePri is safe from ridge in practice" that ≤ 20 TFMA-leaked pairs are below the ridge bootstrap threshold. Decision matrix is clean: if ridge generalises from 20 pairs, AloePri-under-strong-Π is broken.                                      | —              |
| 2   | **ArrowMatch port** (Wang et al., USENIX Sec '25) | Stronger direction-similarity attack than current VMA. User raised a critical caveat (verified): ArrowMatch's expected applicability to AloePri is _bounded_ because the paper's Obs2 identifies matrix-multiplication obfuscation as immune — AloePri uses matrix multiplication (Q̂). The port measures the empirical residual signal. | —              |
| 3   | **Sequence-IMA scaffold**                         | Paper §F.1 IMA is per-row; AloePri's α_e=1.0 embedding noise is large enough to defeat per-row inversion (Vec2Text reproducibility study, arXiv 2507.07700, confirms λ=0.01 already breaks Vec2Text — we're 100× over that). Sequence-level conditioning averages noise across n_q positions; expected SNR gain ≈ √n_q.                 | —              |
| 4   | Paper gradient-opt ISA (Appendix D.1)             | **NOT implemented.** Only test that could explain the 87 % baseline; demoted because the construction-side story for 0 % is already closed. 3-5 day implementation cost. Resurrect if a reviewer demands paper-Table-4-baseline reproduction.                                                                                           | —              |

## The three implemented attacks

### 1. C1: TFMA-seeded ridge

**File:** `evals/aloepri-attacks/m2_7/run_ima_embedrow_attacks.py` (extended; not new). Commit `d20b34b`.

**What it does.** Existing IMA-EmbedRow-ridge attack now has an option
to pick the _top-N most-frequent tokens_ (rather than a uniform-random
sample) as the ridge training set. Models a realistic deployment-day
attacker whose τ-mapping leaks via TFMA on the high-frequency tail
(paper §7.6: ≤ 20 such pairs even at distribution-aware setting).

**Key code.** New functions in the driver:

- `_tfma_seeded_splits()` — picks `train_ids = argsort(-counts)[:N]` for
  the top-N most-frequent tokens; samples val/test/candidate from
  remaining vocab.
- `_load_corpus_token_ids_from_captures()` — reads `prompt_token_ids`
  from existing capture meta.json files (cheap; reuses already-
  tokenised data).
- `_load_corpus_token_ids_from_text()` — fallback that tokenises a raw
  text file via the HF tokeniser for `--baseline-model-dir`.

**Strength bound.** This driver gives the attacker EXACT τ on the
top-K — paper-faithful TFMA has ~20 % error rate per §7.6, not
modelled here. Treat as a worst-case bound for the attacker.

**How to run.**

```bash
cd /home/timo/repos/private-rag-path-2

# Sweep over n_train to find the elbow on the deployed Q3-4B cell:
python evals/aloepri-attacks/m2_7/run_ima_embedrow_attacks.py \
  --plain      /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf \
  --obfuscated /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf \
  --key        /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf.key.npz \
  --ridge-tfma-seeded \
  --ridge-tfma-from-captures \
    evals/aloepri-attacks/results/sweep/cell-qwen3-4b-plain-attnoutput-512-20260526/captures/attn.meta.json \
  --ridge-train-size-sweep 20 50 100 200 500 1024 2048 \
  --skip-transformer \
  --output evals/aloepri-attacks/results/sweep/c1-tfma-seeded-ridge-q3-4b-default-uvo.json
```

For control runs swap the obfuscated cell with the no-R PAPERLIT cell
(`…-uvo-PAPERLIT-bf16-native.gguf`) or the true-paper-K cell
(`…-uvo-PAPERKTRUE-bf16-native.gguf`) and rerun. Output emits a
`ima_embedrow_ridge_sweep` array of (n_train, top1, top10) rows.

**Expected outcome.** Memory note + handoff text predicts the elbow
lands at n_train > 20 (paper-safety argument holds). If the elbow
lands at n_train ≤ 20, AloePri's strong-Π argument breaks and
ridge-with-deployment-day-leakage is a real path-2 threat.

#### Update 2026-05-27 (later): C1 not run, driver deleted

C1 was never measured. The TFMA-seeded variant lived inside
`run_ima_embedrow_ridge()` (it added `_tfma_seeded_splits()` to swap
in top-frequency training pairs); the entire ridge code path —
function, splits, corpus loaders, `--ridge-*` CLI args, and the
`run_all_m2_7.py` orchestrator wiring — was deleted on 2026-05-27
in commit `ab6c1f5` because the **uniform-random ridge variant
that shared the same code path** was over-triggering on the ~293
identity-fixed special-token pairs (CONTROL / USER_DEFINED / UNUSED,
kept at identity-τ for server stop-on-EOS compatibility) and
reading 96.88 % top-1 on the 8B β=2 cell — not τ-recovery, just
identity-match on the fixed positions.

The decision **not to revive C1** rests on two grounds:

1. **The driver as implemented gave the attacker EXACT τ on the
   top-K most-frequent tokens.** Realistic TFMA error rate is ~20 %
   per paper §7.6, not zero. So the C1 driver was a strictly
   stronger-than-realistic upper-bound — a worst-case probe rather
   than a deployment-faithful one.

2. **The TFMA prerequisite is empirically near zero in our
   regime.** Measured TFMA against the 4B paperK β=2 / no-H / pow2
   cell reads **0.78 % top-1 / 4.69 % top-10** on the 64-prompt
   release-gate corpus (chronicle 2026-05-26 + carry-forward
   2026-05-27). At those rates, the realistic attacker-side harvest
   of "reliable top-K" pairs the paper claims (≤ 20 at the head of
   the frequency distribution) is unlikely to materialise at all
   on this deployment. C1 measures "what if attacker had ≥ 20
   reliable seed pairs"; in practice the wire-side leak doesn't
   produce 20 reliable pairs to begin with.

Net: **C1 measurement isn't urgent.** It tests an attacker
assumption (large reliable TFMA harvest) whose prerequisite isn't
holding empirically. If TFMA recovery ever climbs (longer corpora,
distribution-aware models, repeated-prompt analysis), C1 becomes
worth re-implementing — but as a follow-up to TFMA being
non-trivial, not as a standalone defense check. A re-implementation
would graft `_load_corpus_token_ids_from_*` (preserved in the
deletion history) onto a transformer or small-MLP inverter, not the
deleted ridge. Estimated ~1 day if it becomes load-bearing.

### 2. ArrowMatch port (Game of Arrows direction-similarity)

**File:** `evals/aloepri-attacks/m2_7/run_arrowmatch.py` (new, 486 LOC). Commit `d20b34b`.

**What it does.** Two-stage direction-similarity attack on the
mask-in-weight obfuscation family (Wang et al., USENIX Sec '25 — paper
PDF: ziqi-zhang.github.io/files/SEC25-GameOfArrows.pdf):

- **Stage 1 (`arrowmatch_stage1`):** chunked GPU cosine matching.
  For each obfuscated weight row `w̃_i`, find
  `σ̂(i) = argmin_j cos(w̃_i, w_j)` over all plaintext rows `w_j` (paper
  Eq. 1). Reports σ̂ vs ground-truth τ⁻¹ top-1 / top-K recovery.
- **Stage 2 (`arrowmatch_stage2`):** length adjustment
  `ŝ_i = ||w_{σ(i)}|| / ||w̃_i||`. Reconstructs the scalar-multiplied
  weight estimate `W_init = ŝ · W̃` and reports reconstruction residual.

**Dim alignment.** Plain `W` has shape (V, d=2560) and obfuscated
`W̃` has shape (V, d_obs=2816) for Q3-4B. Aligned via
`--align-dims {truncate, pad}` (default truncate to first
`min(d_plain, d_obs)` cols).

**Threat-model framing.** ArrowMatch's foundational claim is Obs2:
"lightweight obfuscation algorithms (Π+D per-vector ops) DO NOT change
vector directions" — only matrix multiplication does, and matrix
multiplication is too expensive for on-device TEEs to apply at runtime.
**AloePri's `W̃ = Π · W · Q̂ + ε` uses matrix multiplication** (Q̂ is
non-diagonal invertible), exactly what ArrowMatch identifies as immune.
AloePri pays the matrix-multiplication cost OFFLINE (one rewrite of the
obfuscated model), avoiding the per-inference TEE bottleneck.

So **ArrowMatch's expected applicability to AloePri is bounded.** The
empirical question: at λ=0.3 the keymat `P̂ = [B|C|E]·Z` with
`B = U + λV` (U orthogonal, V random Gaussian) is close-to-orthogonal
in its top-d×d block. Whether enough direction signal survives the
remaining 2h columns + per-row noise for cosine-matching to recover
σ partially is what this driver measures.

**How to run.**

```bash
cd /home/timo/repos/private-rag-path-2

python evals/aloepri-attacks/m2_7/run_arrowmatch.py \
  --plain      /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf \
  --obfuscated /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf \
  --key        /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf.key.npz \
  --tensor token_embd \
  --device cuda \
  --output evals/aloepri-attacks/results/sweep/arrowmatch-q3-4b-default-uvo.json
```

Memory: peak ~128 MB per chunk at defaults (`--chunk-obf 1024
--chunk-pre 32768`). Pairwise cosine matrix (V × V ≈ 23 GB at
V=151K) is never materialised — only the per-chunk slice. Runtime
expected ~30-60 s on Radeon 8060S iGPU.

Plain control: `--identity-tau` makes τ = identity; attack should
recover ~100 % since the bijection is trivial. Verifies the attack
itself works.

**Expected outcome (prediction).** Significantly lower than current
VMA (which gets 13.51 % on Q2.5-14B / 25.05 % on Q3-14B). Best-case
~5-15 % top-1 — above random (1/V ≈ 10⁻⁵) but well below VMA's
RowSort-based recovery. If ArrowMatch beats VMA on Q3-4B,
direction-recovery is meaningful here and the deployment recommendation
shifts. If it's worse, AloePri's matrix-multiplication keymat
successfully defeats Obs2.

### 3. Sequence-IMA scaffold

**File:** `evals/aloepri-attacks/m2_7/run_sequence_ima.py` (new, 635 LOC). Commit `d20b34b`.

**Status: scaffold only.** Architecture, training loop, and eval
harness all implemented. CLI parses. No training runs were executed
this session — pipeline syntax-validated but un-tested against real
data.

**What it does.** A small encoder transformer (2 layers, 8 heads, pre-
LN — matches paper §F.1 IMA architecture budget) that takes a SEQUENCE
of n_q obfuscated activation rows and emits a SEQUENCE of n_q
plain-token logits jointly. Differs from paper IMA-EmbedRow-transformer
which inverts each row INDEPENDENTLY:

- Paper IMA: per-row inversion → SNR bounded by `1 / α_e ≈ 1` at
  α_e=1.0; per-row Vec2Text-class noise (≥ λ=0.01) already breaks it
  (arXiv 2507.07700).
- Sequence-IMA: cross-attention over n_q positions averages noise →
  effective SNR ~ `√n_q / α_e`. For n_q ≈ 16-32, that's a 4-6× SNR
  gain over per-row.

**Architecture (`SequenceIMAInverter`).**

- Input projection `d_obs → d_model` (defaults equal — no proj).
- Learned positional embedding (seq_len positions).
- N transformer blocks (default 2). Each is pre-LN: LN → multi-head
  self-attention → residual; LN → FFN (GELU) → residual.
- Output `lm_head: d_model → V` produces per-position logits.

**Training (`train_inverter`).** Per-position cross-entropy on
plaintext token IDs. AdamW with paper §F.2 defaults (lr=3e-4,
weight_decay=0, batch_size=8, epochs=2). Per-step loss + per-epoch val
top-1 logged.

**Data synthesis for `embed` surface (`synthesize_embed_pairs`).**
Kerckhoffs-faithful: attacker picks own `(τ_a, Q̂_a, noise_a)`, runs
paper Algorithm 1 mock on public-corpus tokens, generates (plain_ids,
obf_embed_rows) training pairs. The inverter must learn τ-INVARIANT
inversion (different τ_a per training pair would be ideal; current
implementation uses one τ_a per run — extensible).

**Surface flag.** `--surface embed` implemented; `hidden_l0`,
`kqv_out_lN`, `kq_lN` raise `NotImplementedError` with a docstring
pointer to the capture pipeline at
`evals/aloepri-attacks/m2_7/capture_hidden_states.py`. The architecture

- training loop accept (B, n_q, d_obs) tensors agnostically — only the
  data-synthesis pipeline needs extension for non-embed surfaces.

**How to run.**

```bash
cd /home/timo/repos/private-rag-path-2

# Smoke test (build model + one synth pair + one forward pass; no training):
python evals/aloepri-attacks/m2_7/run_sequence_ima.py \
  --plain /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf \
  --corpus-file evals/aloepri-attacks/corpora/release-gate-512.txt \
  --tokenizer-model Qwen/Qwen3-4B \
  --no-run \
  --output evals/aloepri-attacks/results/sweep/sequence-ima-smoke.json

# Full embed-surface train + eval (default n_train=128, n_val=16, n_test=16,
# seq_len=32, epochs=2):
python evals/aloepri-attacks/m2_7/run_sequence_ima.py \
  --plain /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf \
  --corpus-file evals/aloepri-attacks/corpora/release-gate-512.txt \
  --tokenizer-model Qwen/Qwen3-4B \
  --device cuda \
  --output evals/aloepri-attacks/results/sweep/sequence-ima-embed-q3-4b.json
```

Compare against the existing per-row paper IMA-EmbedRow-transformer in
`run_ima_embedrow_attacks.py` (without `--skip-transformer`) on the
SAME synthesis hyperparameters to isolate the sequence-vs-per-row
contribution.

**Expected outcome.** If the sequence-level hypothesis holds, TTRSR
top-1 on `embed` surface significantly exceeds the per-row paper IMA
baseline at the same attacker budget. If it doesn't, AloePri's α_e=1.0
noise survives cross-row averaging — strong defence-side evidence.

#### Update 2026-05-27 (later): scaffold ran, but lands at corpus-prior floor (no attack signal)

The Sequence-IMA scaffold was run end-to-end on Q3-4B paperK β=2 +
a `--attacker-alpha-e 0` plain control. Both landed at the same
number:

| Cell                               | α_e |      top-1 |  top-10 |       val_top1 ep0 → ep1 |
| ---------------------------------- | --: | ---------: | ------: | -----------------------: |
| Obfuscated 4B paperK β=2 keymat    | 1.0 | **4.10 %** | 22.66 % |     4.30 % → 3.91 % flat |
| Plain control (α_e=0, same keymat) | 0.0 | **3.91 %** | 21.88 % | (same; immediately flat) |

**The scaffold did NOT measure obfuscation defense at this
budget.** Diagnosis (five compounding issues, full writeup
2026-05-27):

1. **Wildly over-parameterised.** 618 M-param inverter
   (d_model=d_obs=2 816; vocab head 2 816 × 151 936 = 428 M
   dominates) on n_train=128 × seq_len=32 = 4 096 supervised
   labels ⇒ ~151 000 parameters per training label. Optimiser
   pushes the vocab head onto the empirical token-marginal
   distribution of the public corpus (the cheapest loss-decrease
   direction) and stops there.
2. **Val curve goes the WRONG direction.** ep0 4.30 % →
   ep1 3.91 % — the model is slightly overfitting the train set's
   token-frequency mix in the second epoch, hurting val. Real
   attack learning would push val_top1 up, not down.
3. **Plain control = obfuscated number.** Removing the entire
   Gaussian noise budget (α_e: 1.0 → 0.0) moved top-1 by 0.19 pp.
   If the inverter were actually learning to undo Q̂_a it should
   land near 100 % at α_e=0 (it's a fixed linear inversion on
   4 096 examples). It can't even solve the noiseless version at
   this budget.
4. **Corpus-prior baseline IS ~4 % / ~22 %.** Qwen3's tokenizer +
   Wikipedia/code corpus has a head dominated by whitespace,
   punctuation, and common subwords. Predicting "the most likely
   token regardless of input" hits top-1 ≈ 3-5 %, top-10 ≈
   20-25 %. Both obf and plain numbers sit exactly in that band —
   no attack signal at all.
5. **No CLI for the right plain control.** `--attacker-alpha-e 0`
   removes noise but keeps the keymat. A useful **identity
   baseline** would set `Q̂_a = I` so the inverter just learns
   identity over the embedding table (should hit ~100 % even at
   tiny budget). The script doesn't have that flag — needs a
   one-line patch in `synthesize_embed_pairs` to bypass `@ Q̂_a`.

**What "fixed scaffold" needs** (concrete patch list, ~½ day):

| Lever                 |                   Current |                           Target | Why                                                                      |
| --------------------- | ------------------------: | -------------------------------: | ------------------------------------------------------------------------ |
| n_train sequences     |                       128 |                      **≥ 4 096** | Paper's hidden experiments use ≥ 4 096; ours is smoke-test budget.       |
| Epochs                |                         2 |                       **8 – 16** | Loss curve doesn't even start descending in 2 epochs at this model size. |
| Inverter d_model      |           2 816 (= d_obs) |                  **512 – 1 024** | Cuts vocab head from 428 M → ~78 M; rebalances params/label ratio.       |
| Plain-control flag    |                only α_e=0 | add `--attacker-keymat-identity` | Gives a real ~100 % ceiling for sanity-checking the inverter.            |
| Per-pair τ_a rotation | one τ_a for all 160 pairs |                 rotate per batch | Forces _key-invariant_ inversion (paper-faithful threat model).          |

Until those land, **any Sequence-IMA TTRSR is "did the inverter
escape the corpus-frequency prior", not "did obfuscation hold".**
The handoff's per-row-vs-sequence-level SNR hypothesis remains
**untested** — neither falsified nor confirmed.

## What was researched but NOT implemented (and why)

Full research report lives in the session transcript; condensed
priority + rationale here. Web sources cited inline.

| Attack class                                                                            | Status                                            | Reason not implemented                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Paper gradient-opt ISA** (Appendix D.1)                                               | Researched; designed; not coded                   | Paper-Table-4 baseline reproduction is the _only_ remaining gap, but the construction-side explanation in `2026-05-26-alg2-paper-literal-defense-gap.md` already accounts for the 0 % observation. 3-5 days of HF-transformers + GGUF→HF conversion work for a paper-reproduction lever. Demoted; revive if a reviewer demands.                                                                                                                                                  |
| **Vec2Text per-row sequence inversion** (Morris et al., NAACL 2023, arXiv 2310.06816)   | Researched; not coded                             | Reproducibility study (arXiv 2507.07700) shows λ=0.01 Gaussian noise already breaks Vec2Text. AloePri's α_e=1.0 is 100× that. Per-row Vec2Text would mostly fail on AloePri obfuscation. The interesting variant is SEQUENCE-LEVEL, which is what Sequence-IMA implements — so Vec2Text-per-row would have been redundant.                                                                                                                                                       |
| **GEIA / ZSinvert** (zero-shot embedding inversion via auxiliary LLM, arXiv 2504.00147) | Researched; not coded                             | Requires API access to a strong external LLM (GPT-4 / Llama-3-405B) as inverter prior. Mostly defeated by AloePri's α_e=1.0 noise per the same reasoning as Vec2Text. Also: AloePri's effective embedding API is gated by Π; ZSinvert's black-box construction doesn't transfer trivially.                                                                                                                                                                                       |
| **Hidden No More — full pipeline** (Thomas et al., ICML 2025, arXiv 2505.18332)         | Researched; **closed** — no implementation gap    | AloePri's existing VMA (`run_vma_seed_sweep.py`) IS the HNM core attack (paper [25] reference). HNM's "additional pipeline" beyond what we implement is theoretical (effectiveness analysis under noise) and the defence proposal (Cascade — multi-party token sharding) which doesn't apply to AloePri's single-party setting.                                                                                                                                                  |
| **SGT (Stained Glass Transform)** cryptanalysis                                         | Researched; **no published attack**               | Roberts et al., arXiv 2506.09452, Jun 2025. SGT is structurally per-prompt-fresh (closer to GELO than AloePri); no published cryptanalysis as of 2026-05-26. Not directly relevant to AloePri attack surface. Cited in the theorem doc as the per-prompt-fresh design class AloePri _should_ be in.                                                                                                                                                                              |
| **2602.11088 precomputed-basis attack** (Saini et al., Feb 2026)                        | Researched; closed — methodology doesn't transfer | Attack targets schemes with a _static secret basis + per-query random coefficients_ (Shadow Net, SLIP, TransLinkGuard). Requires K+δ zero-vector queries to extract the noise subspace. AloePri is _strictly more static_ (no per-query randomness at all), so the attacker just reads W̃ directly — the query-phase step is trivialised. The attack's methodology doesn't transfer; AloePri faces a _strictly worse_ threat model where weight inspection is the starting point. |
| **ObfuscaTune-class ArrowMatch parent**                                                 | Researched; partially covered                     | ArrowMatch (now implemented per #2 above) is the canonical mask-in-weight attack. The broader "ObfuscaTune-class" framing is the same attack family.                                                                                                                                                                                                                                                                                                                             |
| **Higher-order spectral attacks on (W, W̃)**                                             | Researched; not coded                             | Defender-side luckiness probe in `python/aloepri-llm/select_adversarial_kd.py` already exploits per-keymat singular-value signatures (Spearman ρ = −0.78 between adversarial K_d score and attacker TTRSR per `aloepri-keymat-variance.md`). Flipping to attacker-side would be ~1-2 days; not prioritised because the signal is already visible from VMA. Worth a 1-page memo if a reviewer asks.                                                                               |
| **Composite TFMA + SDA**                                                                | Researched; not coded                             | C1 (TFMA-seeded) covers the strongest leg. SDA adds only weak supervision (paper §7.6 BLEU-4 ≈ 2). Composition would add ~0.5 days for marginal lift. Not prioritised.                                                                                                                                                                                                                                                                                                           |
| **PCIe / GPU bus snooping (TEE.Fail-class)**                                            | Researched; out of scope                          | Outside paper's threat model (honest-but-curious server) but inside path-2's. AloePri's static keymat IS vulnerable to per-batch bus snooping; design-level finding, not an implementation deliverable. Cited as motivation for path-1 TEE-protected attention.                                                                                                                                                                                                                  |
| **Repeated-prompt analysis (RAG-specific)**                                             | Researched; out of scope                          | Real attacker, expensive to simulate; depends on workload assumptions. Logged as a future research item.                                                                                                                                                                                                                                                                                                                                                                         |
| **Active server / Byzantine adversary**                                                 | Out of scope                                      | AloePri's threat model is honest-but-curious. Byzantine deviation is a different paper.                                                                                                                                                                                                                                                                                                                                                                                          |

## Repo state at handoff

```
$ git log --oneline -4
d20b34b path-2: 3 new deobfuscation attacks (C1 TFMA-seeded ridge, ArrowMatch port, Sequence-IMA scaffold)
7d34016 Merge remote-tracking branch 'origin/master' into path-2-aloepri-gemma
e75b317 path-2: paper-literal Alg2 (A1+A2) closes most of Table 4 87→0 pp gap
1b435d9 path-2: ISA-AttnScore theorem tightened to attn-output surface + 2B.1 measurement
```

Working tree has unrelated modifications (other handoffs, capture
dirs, the PDF) untouched by this session; those belong to other
threads.

`docs/prototype/aloepri-llm.html` §08 ISA AttnScore plain (4B) column
was updated to show kq L=0 (48.63 %) + kqv_out L=0 (97.46 %) as the
two canonical plain ceiling readings (commit pending — was a follow-up
to the staleness flag; not yet committed at handoff time).

## Next session priorities

1. **Run C1 sweep first** — cheapest decisive test of the
   "ridge-bootstrap threshold ≤ 20 pairs" claim. Multiple cells in
   parallel (default-UVO, no-R PAPERLIT, true-paper-K PAPERKTRUE) on
   the existing GPU container. ~30 min wall.
2. **Run ArrowMatch on Q3-4B all three cells** — cosine-direction
   recovery vs ground-truth τ. ~30 s × 3 cells.
3. **Sequence-IMA smoke test → embed-surface training** — `--no-run`
   first to confirm pipeline, then a short embed-surface training
   compared against the existing per-row paper IMA. ~1-3 hours.
4. **Compare results across all three attacks vs VMA + per-row IMA**
   in a single report (extending the §08 ledger).
5. **Decide on follow-ups:** if ArrowMatch beats VMA → escalate.
   If Sequence-IMA beats per-row IMA → that's the new IMA SOTA on
   AloePri.

The deferred items from §"What was researched but NOT implemented" are
all valid follow-ups but lower priority than running what we have.

## Suggested skills for next session

- **`/diagnose`** for empirical runs — disciplined "reproduce →
  hypothesise → instrument → fix" loop on the three attack results
  once they land.
- **`/code-review`** before any cell is built or attack result becomes
  load-bearing for path-2 deployment — these are new attack drivers,
  and `run_arrowmatch.py` in particular has GPU memory + numerical
  edge cases that warrant a pass.
- **`/ask before running attacks/captures`** — memory note
  `feedback_ask_before_running_attacks` applies; user runs benches on
  separate worktrees and concurrent GPU contention on Strix Halo
  corrupts measurements. **Default to one attack run at a time** until
  there's signal that contention is OK.

## Files to read first (next agent)

1. **This handoff** — top-level orientation.
2. **`evals/aloepri-attacks/m2_7/run_arrowmatch.py`** — full attack
   driver with detailed docstring including the
   AloePri-applicability-bound discussion.
3. **`evals/aloepri-attacks/m2_7/run_sequence_ima.py`** — architecture
   - training loop scaffold; docstring explains the per-row vs
     sequence-level SNR argument.
4. **`evals/aloepri-attacks/m2_7/run_ima_embedrow_attacks.py`** —
   look for `_tfma_seeded_splits()` and the new `--ridge-tfma-seeded`
   / `--ridge-train-size-sweep` CLI flags.
5. **`docs/handoffs/2026-05-26-alg2-paper-literal-defense-gap.md`** —
   prior session's context including the two-factor decomposition of
   paper Table 4's 0 % and the 87 % baseline open question.
6. **`docs/research/aloepri-attacks.md`** — current threat-model
   framing, theorem state, paper-literal results, K_a-invariance
   reasoning.
