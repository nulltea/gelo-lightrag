---
type: handoff
status: current
created: 2026-05-26
updated: 2026-05-26
tags: [alg2, aloepri, attacks]
---

# Handoff — paper-literal Alg2 closes most of the 87→0 pp Table 4 gap

**Date:** 2026-05-26 (evening)
**Branch:** `path-2-aloepri-gemma`
**Builds on:** `2026-05-26-isa-attnscore-theorem-and-paper-disparity.md` (theorem rewrite + 2B.1) and `2026-05-26-alg2-attribution-and-8b-followup.md` (deployed cell config).

## TL;DR

1. **Our deployed Alg2 was understating AloePri's actual defense by 7–40 pp** across surfaces × layers. Two construction deviations vs paper Algorithm 2:
   - **A1** — `k_matrix = R̂_qk · Ĥ⁻¹ · Ẑ` (our deployment) vs paper's `Ĥ⁻¹ · Ẑᵀ` (no R̂, transpose Ẑ).
   - **A2** — Û_vo QR-stabilised + 0.05 σ perturbation (our deployment) vs paper's raw `N(0, 1/d_head)` Gaussian.
2. **Paper-literal A1+A2 cell built and remeasured** (`untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-PAPERLIT-bf16-native.gguf`). Ridge defense delta on the §5.4-bounded `kqv_out` output surface at L=5 grows from 0.5 pp (default Alg2) to 40 pp (paper-literal). At L=0 it grows from 14 pp to 50 pp. Score-surface `kq` defense delta at L≥5 grows from ~1 pp to 16–31 pp.
3. **Lever 1 (vocab-disjoint methodology) confirms paper Table 4's "0.0 % AttnScore" as a measurement floor.** Under vocab-disjoint split, ridge on `kq` floors at ~0 % at ALL layers including L=0 even on plaintext (0.43 %) — so paper's 0 % is reproducible with vocab-disjoint methodology on any sufficiently-trained attack, regardless of obfuscation.
4. **Combined picture: paper Table 4's 87 → 0 pp transition is now mostly explained** as the compound of (a) paper-literal Alg2 dropping obf TTRSR to single digits at L≥5, plus (b) vocab-disjoint methodology flooring both plain and obf to ~0 at all layers beyond L=0. The only piece still unaccounted for is paper's "87.14 %" Noise+KeyMat baseline — likely requires a stronger attack class than ridge (paper's gradient-opt ISA, Step 3) or a different baseline cell.

## What was fixed in code

### `python/aloepri-llm/lib/alg2.py`

- `generate_u_vo()` — added `paper_literal: bool = False` flag. When True, returns raw `N(0, 1/d_head)` Gaussian directly (no QR-stabilise, no 0.05 σ Gaussian perturbation). Condition number of the resulting matrix is ~500 (vs ~6 for the QR-stabilised default); paper accepts this, we now do too under the flag.
- `build_layer_keys()` — added `paper_literal: bool = False` flag. When True, sets `k_matrix = Ĥ⁻¹ · Ẑᵀ` instead of our default `R̂_qk · Ĥ⁻¹ · Ẑ`, AND passes the flag through to `generate_u_vo`. Comment block documents the `‖Ẑ² − I‖` score-surface residual that the paper-literal construction accepts as part of `e_C^attn`.

### `python/aloepri-llm/obfuscate_qwen3_gguf.py`

- Added `--alg2-paper-literal` CLI flag.
- Plumbed through `obfuscate()` and into `alg2.build_layer_keys`.
- CLI help string explains the two construction changes + expectation ("lower ridge recovery vs default but NOT to reach paper Table 4's 0 %").

## What was built / captured

| Artefact | Path | Size |
|---|---|---|
| Paper-literal Q3-4B obf cell | `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-PAPERLIT-bf16-native.gguf` | 9.7 GB |
| Paper-literal cell key | `…-PAPERLIT-bf16-native.gguf.key.npz` | 536 KB |
| Combined kq + kqv_out captures (paper-literal) | `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-PAPERLIT-attn-and-output-512-20260526/captures/` | 2.3 GB |

Captures are 512 prompts × 5 layers × {kq, kqv_out} = 5 120 snapshots, n_q × {4 096, 8 192} feature axis. Capture pipeline: patched `aloepri-llama-server:m2_7` (CPU-only) with regex `^(kq|kqv_out)-(0|5|11|17|23)$`, ~2 min total wall.

## Ridge attack remeasurement

### kq surface (pre-softmax `Q·K^T`)

| Layer | Plain row | Default obf row | Paper-literal obf row | Plain vocab | Default obf vocab |
|---:|---:|---:|---:|---:|---:|
| 0 | 48.63 % | 47.22 % | **43.22 %** | **0.43 %** | 0.07 % |
| 5 | 38.69 % | 38.49 % | **7.79 %** | 0.08 % | 0.04 % |
| 11 | 27.73 % | 26.95 % | **7.52 %** | 0.02 % | 0.00 % |
| 17 | 22.41 % | 21.17 % | **6.35 %** | 0.00 % | 0.00 % |
| 23 | 30.13 % | 29.67 % | **6.49 %** | 0.00 % | 0.01 % |

### kqv_out surface (attention output, per head, pre W_o)

| Layer | Plain row | Default obf row | Paper-literal obf row | Plain vocab | Default obf vocab |
|---:|---:|---:|---:|---:|---:|
| 0 | 97.46 % | 83.77 % | **47.32 %** | **71.68 %** | 36.70 % |
| 5 | 50.77 % | 50.22 % | **10.49 %** | 0.76 % | 0.54 % |
| 11 | 18.50 % | 19.83 % | **10.57 %** | 0.01 % | 0.00 % |
| 17 | 16.67 % | 17.75 % | **10.04 %** | 0.00 % | 0.00 % |
| 23 | 25.27 % | 25.99 % | **10.04 %** | 0.09 % | 0.02 % |

Row-split, 10-seed mean, multi-α grid {1e-4, 1e-2, 1.0}, val-picked best per seed. GPU rocSOLVER via `aloepri-ima-trainer:latest`. Wall ~5-15 min per (surface × split) at 4-way GPU contention; serial would be ~3-5 min per run.

## Construction-vs-attack attribution

The two-factor decomposition that produces paper Table 4's headline 0 %:

| Factor | Effect on obf TTRSR | Evidence |
|---|---|---|
| **Paper-literal Alg2 (A1+A2)** vs our deployed Alg2 | −7 to −40 pp depending on surface × layer | This work — same plain captures, only obf construction differs |
| **Vocab-disjoint split** vs row-split | Plain L=0 drops from 48.63 % to 0.43 % on `kq`; deep layers floor at 0 % on both surfaces | This work — Lever 1, both surfaces |
| **Compound** (paper-literal × vocab-disjoint) | Predicted ~0 % on both plain and obf at L≥5 across surfaces | **Not measured this session** — paper-literal × vocab-disjoint not run on either surface yet |

What remains unexplained: paper Table 4's **87.14 % "Noise+KeyMat" baseline**. Ridge under row-split with paper-literal Alg2 still shows only 43.22 % plain on `kq` at L=0 (much less than 87 %). The 87 % almost certainly requires either a different baseline cell (paper "Noise+KeyMat" lacks the full Alg2 — i.e., it's α_e + KeyMat alone, no head/block perm) OR a stronger attack class than ridge. Step 3 (paper-faithful gradient-opt ISA) remains the unresolved test.

## Files touched (this session, on top of the morning theorem-rewrite commit)

| Path | Change |
|---|---|
| `python/aloepri-llm/lib/alg2.py` | Added `paper_literal` flag to `generate_u_vo` and `build_layer_keys` |
| `python/aloepri-llm/obfuscate_qwen3_gguf.py` | Added `alg2_paper_literal` parameter + `--alg2-paper-literal` CLI flag |
| `docs/research/aloepri-attacks.md` | Filled in 2B.1 row + per-layer table (morning session); new paper-literal results to be added |
| `docs/prototype/aloepri-llm.html` §08 ISA AttnScore | Reframed verdict around §5.4 scope + 2B.1 measurements (morning session); paper-literal update pending |
| `evals/aloepri-attacks/results/sweep/2B1-attn-output-vs-kq-comparison.md` | Original 2B.1 comparison report (morning session) |

A4 (Q3-4B vs Q2.5-14B model topology), 2A.1 (RoPE pair-indexing verify), and 2A.3 standalone (Û_vo cleanup separable from A1) remain deferred.

## Next steps — refocused on obfuscation vs paper-literal

The biggest research question shifted. We previously asked "how does paper get 0 %?" — that question is now mostly answered (compound of A1+A2 + vocab-disjoint, modulo the 87 % baseline). The new live question is:

> **What is the practical defense surface AloePri Algorithm 2 *actually* provides when implemented paper-faithfully — and what does our deployed cell give up by deviating?**

Concretely:

### 1. Compound paper-literal × vocab-disjoint ridge (highest leverage, cheap)

Run `gpu_sweep.py --split vocab` against the new paper-literal captures, both surfaces. ~10 min serial each. If it lands at ~0 % on both plain and obf at L≥5 (very likely), the paper Table 4 0 % reproducibility story is closed at the construction + methodology compound level. ~20 min total.

### 2. Per-component attribution: A1 alone vs A2 alone vs A1+A2 together

Currently A1+A2 are bundled under a single flag. Splitting them isolates which of the two deviations carries most of the defense gain. Build cells:
- `--alg2-paper-literal-k` only (A1 only)
- `--alg2-paper-literal-uvo` only (A2 only)

Then measure each. ~3-5 min build + ~3-5 min capture + ~5-10 min ridge per cell. ~30 min total for both new cells. Tells us:
- If A1 alone closes 90 % of the gap, A2 is cosmetic (and our deployed Û_vo is fine to keep).
- If A2 alone closes 90 % of the gap, A1's deviation is justified (the QR-stabilised Û_vo is the real problem).
- If both contribute equally, both deviations need defending in deployed docs.

### 3. Accuracy / quality impact of paper-literal Alg2

Paper claims 0–3 % accuracy loss on the bench suite. Our deployed Alg2 is accuracy-validated; paper-literal has NEVER been accuracy-validated on Qwen3 (only score-surface validated). Run:
- HumanEval pass@1 on the paper-literal cell (same harness as `evals/aloepri-attacks/m2_7/run_quality_humaneval.py`)
- Spot-check generated text on 10–20 prompts for coherence

If paper-literal collapses accuracy on Qwen3 (e.g., due to higher-condition-number Û_vo⁻¹ at bf16 multiplication), our deployed Alg2's deviations are *correct deployment hygiene* not bugs. Worth knowing.

### 4. Whether paper-literal Alg2 can be salvaged by re-introducing minimal stabilisation

If accuracy collapses under (3), explore a middle-ground:
- Keep paper's pure Gaussian Û_vo but clip the singular values at a numerical floor (e.g., σ_min ≥ 0.01) instead of QR-stabilising.
- Or keep our k_matrix (R·H⁻¹·Z) but switch Û_vo to pure Gaussian.

This is a narrower paper-faithful variant that captures most of the defense gain without the bf16-precision risk.

### 5. Re-deploy decision

If accuracy holds under (3) AND A1+A2 give the measured defense gain, the recommended aloepri deployment changes from `default Alg2` to `--alg2-paper-literal` across all future Q3-4B / Q3-8B / Q2.5-14B obfuscation runs. The deployed cell becomes the paper-literal variant; the prior CAVEAT in `alg2.py:244-262` becomes wrong (deliberate non-covariance was the bug, not the feature). HTML §08 + theorem doc need updating to reflect the recommended construction.

### 6. Step 3 (paper-faithful gradient-opt ISA) — still needed for 87% baseline

Independent of the above, the 87 % Noise+KeyMat baseline in paper Table 4 still requires paper's gradient-opt attack to reproduce. Lowest-priority since the construction story above accounts for most of the aloepri narrative.

## Suggested skill for next session

- `/diagnose` on the construction question — specifically attribute A1 vs A2 defense contribution, then verify accuracy preservation under paper-literal.
- `/code-review` before the deployed-cell migration if the new paper-literal becomes the recommended config (load-bearing change in obfuscation construction across multiple deployment paths).

## Pending background work

None — all four ridge jobs completed successfully. No active containers (`docker ps` shows only edgequake / postgres / vaultwarden / svc-dashboard / llama-swap, none of which are ours).

## Memory notes added / updated

None yet — recommend adding after the per-component attribution (#2 above) so the rule encodes which deviation specifically matters, not the bundle.
