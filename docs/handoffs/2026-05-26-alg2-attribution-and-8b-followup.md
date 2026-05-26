# Handoff — Alg2 attribution complete on Q3-4B; ISA-AttnScore deeper diagnose + 8B re-run pending

**Date:** 2026-05-26
**Branch:** `path-2-aloepri-gemma`
**Companion docs (don't duplicate, read first):**
- `docs/handoffs/2026-05-25-alg2-attack-crossmap.md` — the living crossmap; updated with VMA bisection, §5.2.2 × Alg2 interaction, broadly-optimal config decision, synthetic probe scope correction.
- `docs/research/aloepri-attacks.md` — added VMA, Per-head fingerprint, V/O channel-pair sections.
- `docs/prototype/aloepri-llm.html` §08 — updated Conditions under test with the current best-defense config + per-attack Q3-4B numbers.

## TL;DR (2026-05-26)

- **Algorithm 2 component attribution on Q3-4B is empirically resolved** for the static-weight attack ledger. Five new isolation cells built today on the Alg1+§5.2.2 substrate (cells A/B/C/D/E + F). VMA 3-seed sweep across all five.
- **Two Alg2 components carry real defense; two are dead-weight:**
  - `R̂_qk + Ĥ_qk±1` (matrix-Γ kernel): −17.8 pp on VMA via §5.2.2 × Alg2 superadditive interaction
  - `Û_vo`: −12.9 pp on VMA via per-head column-magnitude perturbation
  - `Ẑ_block β=8`: **0 pp on every measured attack**; introduces 35 % rel RoPE attention-score Δ (accuracy cost). Strict improvement to set β=1.
  - `Π_head` (τ_kv + τ_group): 0 pp; Alg1's rectangular keymat already shadows it.
- **§5.2.2 (Π + α_e + α_h) is the load-bearing defense**, contributing −63 pp on VMA. Alg2 components add −26 pp combined on top via superadditive interaction with §5.2.2 substrate.
- **ISA-AttnScore re-classified "no defense possible" — and the "11.87 % noise floor" framing retracted 2026-05-26.** The K_a-invariance claim still holds (surface is invariant by paper §3.2 construction; confirmed at every measured layer, max plain-obf Δ = 1.41 pp ≈ 1.2 σ). But the magnitude was a 64-prompt sample artefact (only 34 % test/train vocab overlap capped memorisation). With 512 prompts (75 % overlap, GPU-native sweep), the true surface ceiling at L=0 is **48.63 % ± 0.80 (plain) vs 47.22 % ± 0.84 (obf)**, and at L=17 is **22.41 vs 21.17**. The attack reads ~half of all token identities at L=0 even under the deployed best-defense config. Old "Pass at 0 %" reading was outlier-low single-seed with vocab-disjoint methodology (vocab-disjoint mechanically floors near 0 % on small data — see [[feedback_isa_attnscore_methodology]]).
- **Synthetic probe is now flagged as scope-limited** (`probe_alg2_static_attacks.py`): measures Alg2 on no-§5.2.2 substrate, under-predicts deployed contribution by ~9× due to missing the §5.2.2 × Alg2 superadditive amplification.

## Current best-defense config for Q3-4B (deployed Û_vo cell)

File: `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf` (9.7 GB bf16).

Build command:

```bash
python obfuscate_qwen3_gguf.py \
    --in  /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf \
    --out untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf \
    --mode keymat --expansion-size 128 \
    --pi --noise-alpha-e 1.0 --noise-alpha-h 0.2 \
    --alg2 --alg2-qk-norm-matrix --alg2-h-hadamard-signs \
    --alg2-beta 8  \
    --alg2-u-vo \
    --output-dtype bf16
```

**Broadly-optimal variant: change `--alg2-beta 8` → `--alg2-beta 1`.** Zero security cost on the measured ledger, frees ~35 % rel RoPE attention-score perturbation (accuracy gain). Not yet built/measured.

Full per-component breakdown lives in `docs/prototype/aloepri-llm.html` §08 "Conditions under test" row "Obfuscated (4B) — current best-defense".

## Attack matrix on this cell (Q3-4B, 2026-05-26)

Headline numbers (3-seed or 10-seed mean where noted; full table in `docs/handoffs/2026-05-25-alg2-attack-crossmap.md` and HTML §08):

| Attack | Q3-4B obf top-1 | Plain ceiling | Status |
|---|---:|---:|---|
| NN @ attn_norm-0 | 0.0 % | 100 % | defense complete (Alg1) |
| IA Gate-IA | 0.05 % | 98.5 % | defense complete (Alg1) |
| IA Attn-IA | 0.00 % | (same) | defense complete (Alg1) |
| Per-head fingerprint Q | 4.25 % | 100 % | at random 1/32 (Alg1) |
| Per-head fingerprint K | 13.54 % | 100 % | at random 1/8 (Alg1) |
| Per-head fingerprint V | 12.50 % | 100 % | at random 1/8 (Alg1) |
| Per-head fingerprint O | 3.21 % | 100 % | at random 1/32 (Alg1) |
| V/O channel-pair V | 12.50 % | — | at random 1/8 (Alg1) |
| V/O channel-pair O | 3.12 % | — | at random 1/32 (Alg1) |
| V/O channel-pair (V, O) joint | 3.12 % | — | at random (Alg1) |
| **VMA** (3-seed mean ± std) | **9.51 % ± 2.50** | 98.4 % | defense partial; §5.2.2 + Alg2 interaction |
| IMA-EmbedRow-transformer multi-key K=64 | 3.13 % (per §08) | 13.5 % | defense partial; §5.2.2 + Alg2 |
| **ISA HS multi-key K=64 paper-faithful** @ L=17 (3-seed mean) | **8.54 % ± 4.74** | 15.04 % | defense partial; high seed variance |
| ISA-AttnScore (10-seed row-split mean) @ kq-17, 64-prompt | 11.87 % ± 3.44 | 11.87 % | **rebased 2026-05-26** — 64-prompt sample artefact |
| ISA-AttnScore (10-seed row-split mean) @ kq-0, 512-prompt | **47.22 % ± 0.84** | 48.63 % | **no defense possible** (K_a-invariant, headline) |
| ISA-AttnScore (10-seed row-split mean) @ kq-17, 512-prompt | 21.17 % ± 0.86 | 22.41 % | same — Δ 1.24 pp within seed-spread |
| ISA-AttnScore (10-seed row-split mean) @ kq-23, 512-prompt | 29.67 % ± 0.71 | 30.12 % | same — Δ 0.45 pp |
| TFMA | 0.4 % (per §08) | — | defense complete (Π) |
| SDA | BLEU-4 7.8 × 10⁻⁶ (per §08) | — | defense complete (Π) |
| QK-norm Γ eigendecomposition | ~100 % in 5 ms / layer | n/a | **undefended by design** (structural Alg2 break) |

## What's new in code + docs since 2026-05-25 morning

| Path | Change |
|---|---|
| `python/aloepri-llm/lib/alg2.py` | Bug #1 fix — added matrix-Γ caveat block with β-sweep table (lines 241-262); Bug #2 fix (legacy `LayerAlg2Keys` reconstruction was already done) |
| `python/aloepri-llm/obfuscate_qwen3_gguf.py` | Bug #2 fix from 2026-05-25 (legacy `LayerAlg2Keys` reconstruction passes `u_vo`/`u_vo_inv` through) |
| `python/aloepri-llm/scripts/check_alg2_invariance.py` | Bug #3 fix (stale `python/path-2` import → `python/aloepri-llm`) |
| `evals/aloepri-attacks/m2_7/probe_alg2_ia_invariant.py` | NEW — synthetic per-component IA Attn-IA probe |
| `evals/aloepri-attacks/m2_7/probe_alg2_static_attacks.py` | NEW — synthetic per-component VMA + IA Gate-IA probe; **header annotated 2026-05-26 with scope-limitation warning** (under-predicts deployed contribution ~9× due to missing §5.2.2 substrate) |
| `evals/aloepri-attacks/m2_7/run_per_head_fingerprint.py` | NEW (background agent 2026-05-25) — per-head SVD-spectrum static attack |
| `evals/aloepri-attacks/m2_7/run_vo_channel_pair.py` | NEW (background agent 2026-05-25) — per-head V/O channel-pair static attack |
| `evals/aloepri-attacks/m2_7/run_isa_attn_score_multikey.py` | NEW — stub driver returning `not_applicable` with rationale (K_a-invariant surface; multi-key paper-faithful inapplicable) |
| `docs/handoffs/2026-05-25-alg2-attack-crossmap.md` | Living crossmap; major sections added: Alg1-only attack matrix, Alg1+min-Alg2 attack matrix, VMA × Alg2 ranked table, cross-param interactions (within-Alg2 + §5.2.2 × Alg2), VMA-optimal vs broadly-optimal config, within-Alg2 bisection (Cells A-F), synthetic probe scope correction |
| `docs/research/aloepri-keymat-variance.md` | Bug #6 fix — framing-correction block at top explaining "Alg2 amplifies" was a §5.2.2 + §5.2.3 + quantization conflation |
| `docs/handoffs/2026-05-22-keymat-defense-optimization.md` | Bug #6 fix — Phase d table now has §5.2 column + Ẑ_block + Π_head rows |
| `docs/research/aloepri-attacks.md` | Added VMA, Per-head fingerprint Q/K/V/O, V/O channel-pair V/O sections with mechanism + Q3-4B numbers |
| `docs/prototype/aloepri-llm.html` §08 | Updated Conditions under test "Obfuscated (4B)" row with full per-component config; updated VMA / ISA HS multi-key / ISA-AttnScore / IA rows with 2026-05-26 numbers; added Per-head fingerprint Q/K/V/O + V/O channel-pair attack rows; Plain (4B) and Obfuscated (4B) updated, Obfuscated (8B) marked `pending` for new attacks |

## Bisection cells on disk (Q3-4B)

All in `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/`:

| File | Config | VMA top-1 |
|---|---|---:|
| `Qwen3-4B-Q8_0-untied.gguf` | plain (untied) | — |
| `untied-keymat-h128-alg1-only-bf16.gguf` | Alg1 only | 98.4 % |
| `untied-keymat-h128-alg2min-zblock-bf16.gguf` | Alg1 + minAlg2 (R̂+H+Ẑ+Π_head, no Û_vo, no §5.2.2) | 96.5 % |
| `untied-keymat-h128-pi-noise-ae1.0-ah0.2-bf16.gguf` | Alg1 + §5.2.2 | 35.4 % |
| `untied-keymat-h128-522-headperm-bf16.gguf` | Alg1 + §5.2.2 + Π_head only | 35.4 % |
| `untied-keymat-h128-522-matrixgamma-bf16.gguf` | Alg1 + §5.2.2 + matrix-Γ (no Û_vo) | 17.6 % |
| `untied-keymat-h128-522-matrixgamma-beta1-bf16.gguf` | same with `--alg2-beta 1` | 17.6 % |
| `untied-keymat-h128-522-uvoonly-bf16.gguf` | Alg1 + §5.2.2 + Π_head + Û_vo | 22.5 % |
| `untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf` | **deployed (current best-defense)** | 9.5 % |

Plus existing captures under `evals/aloepri-attacks/results/sweep/`:
- `cell-qwen3-4b-alg1only-attn-20260525/captures/` — Alg1-only hidden + kq at L={0, 17}
- `cell-qwen3-4b-alg2min-attn-20260525/captures/` — Alg1+minAlg2
- `cell-qwen3-4b-plain-attn-multilayer-20260525/captures/` — plain at L={0, 5, 11, 17, 23}
- `cell-qwen3-4b-uvo-attn-multilayer-20260525/captures/` — full-deployed multi-layer kq + hidden

## Next steps (in priority order)

### 1. Diagnose ISA-AttnScore failure deeper — UPDATED 2026-05-26

**Resolved 2026-05-26 via 512-prompt re-capture + GPU layer sweep.** Prior "11.87 % noise floor" framing was wrong — it was a 64-prompt sample-noise artefact. The K_a-invariance claim still holds.

**Findings (512 prompts, row-split, 10-seed mean, plain vs deployed-best-defense obf):**

| Layer | Plain top-1 ± std | Obf top-1 ± std | Δ |
|---|---:|---:|---:|
| 0 | **48.63 ± 0.80** | **47.22 ± 0.84** | +1.41 pp (~1.2σ, not sig) |
| 5 | 38.68 ± 0.81 | 38.49 ± 0.48 | +0.19 |
| 11 | 27.73 ± 0.79 | 26.95 ± 0.89 | +0.78 |
| 17 | 22.41 ± 0.81 | 21.17 ± 0.86 | +1.24 |
| 23 | 30.12 ± 0.82 | 29.67 ± 0.71 | +0.45 |

**Per-head ridge @ L=17 (3 seeds, α=1e-2):** best-head plain 11.66 % / obf 11.20 %. Per-head signal is much weaker than cross-head flattened — ridge over 8192-col flatten extracts genuine cross-head structure, not just per-head noise.

**Post-softmax @ L=0 (paper's surface, 10-seed row-split, 512 prompts):** plain 32.09 % ± 0.90 / obf 30.60 % ± 0.64. Δ = 1.49 pp — essentially identical to pre-softmax Δ of 1.41 pp. Post-softmax compresses signal by ~16 pp (softmax + causal mask flatten the magnitude info ridge uses), but K_a-invariance survives the softmax non-linearity (softmax is deterministic + equivariant: plain≈obf at pre → plain≈obf at post). Confirms: paper's 0 % reading is NOT explained by post-softmax surface choice alone — must be vocab-disjoint methodology compounded with smaller corpus.

**Reframed conclusions:**
- Attention scores leak **~half of all token identities at L=0** even under the deployed obfuscation. Not a noise floor.
- K_a-invariance is empirically confirmed at every layer (max Δ 1.41 pp ≈ 1.2σ).
- The transformer-inverter spike (originally Plan step 2) is no longer needed to confirm invariance — ridge already shows the surface signal AND the structural invariance. It would still be valuable to know if a stronger attacker pushes plain → 100 % (which would let us quantify what % obf would defend against a competent attacker, but the absolute defense delta would still be ~0).

**Companion `/diagnose`:** zero-defense-vs-ISA-AttnScore@L=0 — param-by-param attribution; see latest diagnose section in `docs/research/aloepri-attacks.md`.

**Threat-model framing for §08 readers:**
- Our `kq` capture is **pre-softmax** Q·K^T (custom debug build + `--flash-attn off`). Paper reference impl uses **post-softmax** `outputs.attentions[L]` (standard HF hook).
- Both within AloePri §3.2 threat model ("attacker captures intermediate activations from deployment"). Pre-softmax is a more privileged attacker (operator + debug build); post-softmax is plausibly leakable via interpretability features.
- Switching to paper's post-softmax surface drops the plain ceiling 48 → 32 pp at L=0 but the obf-vs-plain Δ stays ~1.5 pp. K_a-invariance carries through softmax.
- Under path-1 (TEE-with-PCIe), attention is in-TEE → both surfaces are `not_applicable` per `run_isa_attn_score.py` docstring.

**Open follow-ups (lower priority):**
- Build the paper-like 2-layer transformer inverter at kq-0 — quantify maximum recoverable signal on plain (likely > 90 %), confirm obf still matches within seed-spread. ~4-6 hr GPU.
- Post-softmax L=17 + L=23 sweep (currently only post-softmax @ L=0 measured). Cheap — apply softmax to existing kq captures, ~2 min per cell on GPU. Expected: same K_a-invariance with proportional signal compression.
- Bug: `aloepri-llama-server:m2_7` is CPU-only (no Vulkan/ROCm libs compiled in). Rebuild with Vulkan backend would unlock GPU captures and let 8B captures finish in <2 min instead of ~15 min. See [[infra_aloepri_llama_server_cpu_only]].

### 2. Rerun Q3-8B on optimal config

**Current state on disk:** Q3-8B deployment GGUF from 2026-05-21 (`cell-qwen3-8b-uvo-20260521/`) has hidden captures at L=17 (attn_norm only). No multi-layer kq captures. No Alg1-only / minAlg2 / bisection cells built.

**Steps:**
1. Build deployed Q3-8B optimal cell:
   ```bash
   python obfuscate_qwen3_gguf.py \
       --in  /home/timo/.cache/huggingface/hub/models--bartowski--Qwen_Qwen3-8B-GGUF/.../Qwen_Qwen3-8B-bf16.gguf \
       --out /home/timo/.cache/huggingface/path-2-aloepri/qwen3-8b/keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16.gguf \
       --mode keymat --expansion-size 128 --pi --noise-alpha-e 1.0 --noise-alpha-h 0.2 \
       --alg2 --alg2-qk-norm-matrix --alg2-h-hadamard-signs --alg2-beta 8 --alg2-u-vo \
       --output-dtype bf16
   ```
   (Q3-8B has `tie_word_embeddings: false` — no untie step needed.)
   
   Expected ~3-5 min build.

2. Spawn patched server (`aloepri-llama-server:m2_7`) with `--tensor-filter '^(attn_norm-(0|17)|kq-17)$' --flash-attn off`.

3. Capture hidden + kq at L={0, 17}: 64 prompts × ~80 s each = ~3 min per kind.

4. Run full attack matrix on this cell:
   - Static (no captures needed): VMA + IA + Per-head fingerprint + V/O channel-pair + IMA-EmbedRow multi-key
   - Runtime: NN @ L=0, ISA HS multi-key K=64 @ L=17 (3 attacker seeds), ISA-AttnScore @ kq-17 (10 seeds, row-split)
   - Compare to Q3-4B numbers cell-by-cell.

**Hypothesis going in:** 8B numbers should be similar to 4B for the "defense complete" attacks (NN, IA, per-head fingerprint, V/O — Alg1 alone defeats; dimension shouldn't change). VMA might be lower on 8B due to extra dimensional headroom (d_obs=4352 vs 4B's 2816); §08 currently shows VMA 8B = 5.1 % vs 4B = 18.4 % (stale). ISA HS multi-key is the most likely to differ — 8B at L=17 was 9.0 % single-seed in 2026-05-21; needs 3-seed re-measure to verify.

Cost: ~30-45 min for the full sweep.

### 3. Per-component bisection on Q3-8B (lower priority)

If 8B results differ significantly from 4B per-attack, repeat the Cells A-F bisection on Q3-8B to verify the §5.2.2 × Alg2 superadditivity scales. Same 7 cells × VMA 3-seed = ~30 min compute. Only worth doing if 4B numbers don't predict 8B.

### 4. (Optional) IMA-EmbedRow-transformer multi-key on Q3-4B + Q3-8B

Currently §08 quotes 3.13 % (4B and 8B) from earlier 2026-05-21 work. Re-running on the current-best-defense cell would either confirm the headline number or expose drift. Cost: ~30 min training per model.

## Bugs noted but not fixed

- **Bug #4** (`--alg2-gamma`, `rope_base` are dead CLI args after 2026-05-19 Ẑ_block fix): left as documented cruft. Signature ripple to remove cleanly is uglier than the benefit.
- **Bug #5** (path-2 deviates from paper Alg2 line 6: adds R̂_qk to K, removes Z^T): documented; security-proof implications open. Decision-pending.
- **Alg2 seed is in the public obfuscate script** (line 832 default `987654321`): treat as deployment secret. Future PR.

## Synthetic probe limitations summary

`probe_alg2_static_attacks.py` and `probe_alg2_ia_invariant.py` measure Alg2 components on a NO-§5.2.2 substrate. Under-predict real-cell contributions by ~9× for VMA (R̂_qk+H gives 0.4 pp synthetic vs 17.8 pp real on Alg1+§5.2.2). The probes are **correct in their setting** (verified by Alg1 → Alg1+minAlg2 marginal of −1.9 pp on real cell, matching synthetic ~0.4-2 pp). They just don't capture §5.2.2 × Alg2 superadditive interaction.

**Fix proposed but not implemented:** add a §5.2.2-equivalent substrate option (add W_e noise + row-perm before measuring) and run probes in both settings. See `probe_alg2_static_attacks.py` docstring (annotated 2026-05-26) for the full caveat.

## Suggested skills for next session

- **`/diagnose`** for the ISA-AttnScore deeper investigation (transformer inverter vs ridge).
- **`/grill-with-docs`** when reconciling §08's stale 8B numbers with fresh measurements + propagating naming consistency.
- **`/code-review`** before any deployment strip (changing `--alg2-beta 8` → `--alg2-beta 1` in the default).

## Files to read first (for the next agent)

1. `docs/handoffs/2026-05-25-alg2-attack-crossmap.md` — primary living document with all attribution analysis
2. `docs/research/aloepri-attacks.md` § VMA, § Per-head fingerprint, § V/O channel-pair
3. `docs/prototype/aloepri-llm.html` §08 Conditions under test row "Obfuscated (4B) — current best-defense"
4. `evals/aloepri-attacks/m2_7/run_per_head_fingerprint.py` + `run_vo_channel_pair.py` if running real-data probes
5. This handoff for next-step planning + cells-on-disk inventory
