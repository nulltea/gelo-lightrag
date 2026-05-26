---
type: handoff
status: current
created: 2026-05-20
updated: 2026-05-20
tags: [ima, attacks]
companion: [2026-05-21-ima-transformer-paper-disparity]
---

# Handoff — IMA-EmbedRow-transformer investigation (capacity-limited attack)

**Date:** 2026-05-20
**Branch:** `path-2-aloepri-gemma` (uncommitted: m2_7 driver renames, paper-IMA → IMA-EmbedRow, identity-τ plain control, four trained-transformer architecture attempts).
**Status:** Plain controls for VMA / IA / IMA-EmbedRow-ridge validated. IMA-EmbedRow-transformer fails the identity plain control across every architecture variant we tried. Root cause identified via minimal-pipeline test. Recommendation: drop or downgrade the IMA-EmbedRow-transformer row in §08.

## What we set out to do

After porting paper-IMA-basic / paper-IMA-paper-like as `IMA-EmbedRow-ridge` / `IMA-EmbedRow-transformer` and adding a `--identity-tau` plain-control mode, we needed to validate the four new prompt-inversion attacks (VMA, IA, IMA-EmbedRow-*) had real plain-side baselines, not "n/a (attack requires θ̃)" placeholders. The motivating concern: the trained-transformer's 0.0 % obfuscated reading could be a defence signal OR a broken attack — only the plain control disambiguates.

## Results that are settled

| Attack | Plain control top-1 | Obfuscated top-1 | Verdict |
|---|---|---|---|
| VMA | **99.6 %** | 0.0 % | ✅ Attack works, defence real |
| IA | **98.6 %** | 0.0 % | ✅ Attack works, defence real |
| IMA-EmbedRow-ridge | **99.2 %** | **97.66 %** | ✅ Attack works, **defence fails by 6.5×** |
| IMA-EmbedRow-transformer | **0.0–0.4 %** | 0.0 % | ❌ **Broken attack** — fails plain control |

Result files:
- `evals/aloepri-attacks/results/m2_7-static-FULL-zfix-hadamard.json` — obfuscated VMA + IA.
- `evals/aloepri-attacks/results/m2_7-static-plain-control.json` — plain VMA + IA.
- `evals/aloepri-attacks/results/m2_7-ima-embedrow-FULL-zfix-hadamard.json` — obfuscated ridge + transformer.
- `evals/aloepri-attacks/results/m2_7-ima-embedrow-plain-control.json` — plain ridge + transformer v1.
- `evals/aloepri-attacks/results/m2_7-ima-embedrow-plain-control-v2.json` — transformer v2 (no bottleneck, seq=32).
- `evals/aloepri-attacks/results/m2_7-ima-embedrow-plain-control-v2-32ep.json` — transformer v2 with epochs=32.
- `evals/aloepri-attacks/results/m2_7-ima-embedrow-plain-control-v3.json` — transformer v3 (identity-init blocks).
- `evals/aloepri-attacks/results/m2_7-ima-embedrow-plain-control-v3-mlp.json` — transformer v4 (residual MLP only, lr=1e-3).

## Things tried on IMA-EmbedRow-transformer (every config fails the plain control)

| Variant | Architecture | Hyperparams | Plain top-1 | Train loss trajectory | Cosine |
|---|---|---|---|---|---|
| **v1** | seq=1 row in, MHA(8h) + FFN, hidden=256 (bottleneck), 2 blocks | lr=3e-4, wd=0, bs=64, epochs=4 | 0.39 % | 0.020 → 0.0018 | 0.030 |
| **v2** | seq=32 sequences, MHA(8h) + FFN, hidden=obs_dim=2048 (no bottleneck), 2 blocks | lr=3e-4, wd=0, bs=8, epochs=8 | 0.00 % | 0.149 → 0.003 | 0.008 |
| **v2 + 32ep** | same as v2 | epochs=32 | 0.00 % | 0.149 → 0.0017 | 0.008 |
| **v3** | v2 + identity-init blocks (zero-init MHA `out_proj` + FFN `[2]`) | epochs=8 | 0.00 % | 0.073 → 0.0016 | 0.020 |
| **v4** | Residual MLP only (no MHA), identity-init blocks, hidden=obs_dim, 2 blocks | lr=1e-3, wd=1e-3, bs=64, epochs=16 | 0.00 % | 1.35 → 0.007 (unstable) | 0.027 |
| **min** | Single `nn.Linear(2048, 2048)` — pure GD ridge equivalent | lr=3e-4, wd=0, bs=64, epochs=16 | **2.3 %** | 0.00150 → 0.00025 | **0.24** |

The minimal single-Linear test is the decisive diagnostic: it **does** make monotonic progress (cosine 0.14 → 0.24, top1 0 → 2.3 %, top10 0 → 16 %). The pipeline isn't broken — it's just that gradient descent on a 2048×2048 weight matrix doesn't converge to identity in 256 update steps.

## Root cause

The IMA-EmbedRow attack's optimal inverter is the closed-form least-squares solution `W = (XᵀX + αI)⁻¹ Xᵀ Y`. Ridge computes this in 0.8 s and gets 99.2 % on the plain control. The trained variants try to approach the same `W` via AdamW gradient descent.

With 4096 training rows × batch 64 × 16 epochs = ~1 024 update steps, each parameter's total movement is bounded by `epochs × lr × ||grad / batch||` ≈ 16 × 3e-4 × O(1) per param. Identity has diagonal 1.0 vs random init `|W_ij| ~ 0.022`, so each diagonal needs ~30 unit-grad steps to climb. The minimal Linear test shows `||W - I||_F` drops only 2 % in 16 epochs (52.2 → 51.2), consistent with this rate. The deeper architectures (v1–v4) add more parameters and slow convergence further while not adding inductive bias toward the correct solution.

The AloePri reference's `_PaperLikeIMAInverter` uses the same protocol (lr=3e-4, epochs=2, batch=8) and reports IMA-paper-like = 0.0 % on AloePri Qwen2.5-14B. The paper **does not publish** the plain-control number for IMA-paper-like, so we don't know if the reference's inverter also fails identity recovery — but the architecture + hyperparam match suggests it does.

This means: under paper-default training budget, IMA-EmbedRow-transformer is fundamentally redundant with IMA-EmbedRow-ridge. Ridge already gets the closed-form upper bound; the transformer can only approach it asymptotically and never reaches it within paper-compatible budgets.

## What this means for the §08 attack table

The current §08 IMA-EmbedRow-transformer row (with plain "0.4 %" / obfuscated "0.0 %") is misleading: it looks like a real measurement against a real attack. It isn't — the attack itself doesn't reach paper-claimed capability.

**Recommendation:** drop the IMA-EmbedRow-transformer row entirely. Cite IMA-EmbedRow-ridge as the load-bearing measurement on the static embedding-row surface. Document the trained-transformer's GD-convergence limitation in `evals/aloepri-attacks/README.md` so a future operator knows why this attack isn't in the harness's regular run set.

**Alternative:** keep the row but mark it as "GD-budget-limited" with a numbered plain-control reading + the same caveat. Less clean but preserves measurement provenance.

## Things NOT tried (defer-able)

- **Closed-form init + GD refine.** Initialize the inverter's output_proj to the ridge solution, then train. Should reach ridge's 99 % immediately and possibly exceed via non-linear refinement. Distinct attack only if it strictly beats ridge.
- **Huge epoch budget.** Run v3 / v4 with epochs=1000+. Compute cost: ~12 h on the current CPU host. Probably matches ridge but does not exceed it.
- **Reference's Qwen2 backbone.** Use `AutoModel.from_config` with full Qwen2 architecture instead of vanilla pytorch MHA. Adds dependency on `transformers`; expected behaviour is the same since GD limitation is generic.
- **Contrastive loss instead of MSE.** Train against direct cosine-similarity-to-target objective. Could converge faster than MSE because the discriminator is also cosine. Untested.

## Files in this session (uncommitted)

**Code:**
- `evals/aloepri-attacks/m2_7/run_ima_embedrow_attacks.py` — renamed from `run_paper_ima_attacks.py`; final state is the residual-MLP v4 driver with `--identity-tau` flag for plain controls.
- `evals/aloepri-attacks/m2_7/run_all_m2_7.py` — orchestrator step 1b for IMA-EmbedRow.
- `evals/aloepri-attacks/m2_7/m2_7_common.py` — registered `ima_embedrow_attacks` phase (25 GB pre-flight).

**Docs:**
- `docs/prototype/aloepri-llm.html` §08 — updated with 4 new attack rows (VMA, IA, IMA-EmbedRow-ridge, IMA-EmbedRow-transformer) including plain-control numbers for VMA / IA / IMA-EmbedRow-ridge. **IMA-EmbedRow-transformer row needs a final pass** to reflect the v4 plain-control 0.0 % result and the "broken attack" verdict from this handoff.
- `docs/prototype/aloepri-llm.html` §03 / §04 / §09 — naming-note + cross-reference updates.

**Results:** see table above.

**Minimal isolation test:** `/tmp/iso/test.py` (single Linear, 16-epoch identity recovery). Useful for future GD-convergence experiments; consider parking under `evals/aloepri-attacks/m2_7/scripts/` if we want it tracked.

## Suggested ordering for the next session

1. **Update §08 IMA-EmbedRow-transformer row** to reflect the v4 plain-control reading (0.0 %) and the broken-attack verdict. Recommended copy: "Plain control 0.0 % — attack itself fails identity recovery at paper-default training budget (1 024 GD steps insufficient to converge to ridge's closed-form solution). The 0.0 % obfuscated reading is therefore not interpretable as defence; cite IMA-EmbedRow-ridge as the load-bearing measurement on this surface."
2. **Update §08 acceptance gates table** — change `ima_embedrow_transformer_below_15pct` from "provisional pass" to "skipped (attack broken)".
3. **Update §08 headline reading** — change "8 of 11 pass" to "7 of 10 pass; 1 attack broken (IMA-EmbedRow-transformer)".
4. **Commit the m2_7 renames + plain-control changes.** Branch has been uncommitted for two sessions; the static-weight harness state is reproducible enough to ship as-is.
5. (Optional) Re-attempt IMA-EmbedRow-transformer with closed-form init OR contrastive loss; only worth the effort if aloepri wants a "trained-inverter" plain-side claim. Otherwise drop the variant.

## Pre-flight notes for the next session

- **Memory:** 51 GB free on this host; the two-GGUF working set (plain Q8_0 + obfuscated fp32) needs 25 GB.
- **Plain GGUF:** `~/.cache/huggingface/hub/models--bartowski--Qwen_Qwen3-1.7B-GGUF/snapshots/dcb19155b962dbb6389f4691a982043a8e651022/Qwen_Qwen3-1.7B-Q8_0.gguf`
- **Obfuscated FULL target:** `~/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-alg2-FULL-zfix-hadamard-fp32.gguf` (+ `.key.npz`).
- **Harness state:** `evals/aloepri-attacks/m2_7/run_ima_embedrow_attacks.py` is functional and passes `--help`; `run_static_attacks.py` is unchanged. Both write JSON consumable by the §08 ingestion path.
