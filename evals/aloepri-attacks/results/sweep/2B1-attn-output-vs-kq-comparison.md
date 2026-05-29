# 2B.1 — Attention-output (`kqv_out`) vs pre-softmax score (`kq`) ridge comparison

**Date:** 2026-05-26
**Model:** Qwen3-4B (deployed Û_vo cell `untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf`)
**Corpus:** `evals/aloepri-attacks/corpora/release-gate-512.txt` (512 natural-language prompts, ~9 463 token rows)
**Attack:** multi-α ridge `(X^T X + αI)^{-1} X^T Y`, α∈{1e-4, 1e-2, 1.0}, 10-seed row-split mean, cosine-NN top-1 over full Q3-4B embedding table (V=151 936). Driver: `evals/aloepri-attacks/m2_7/diagnose_isa/gpu_sweep.py` via `aloepri-ima-trainer:latest` (rocSOLVER on Radeon 8060S, gfx1151).
**Capture tag:** `cell-qwen3-4b-{plain,uvo}-attnoutput-512-20260526/` (kind=`kqv_out`, shape `(n_q, 4096)` = 32 heads · 128 head_dim, per-head attention output pre-W_o).

## Headline

| Surface | L=0 plain | L=0 obf | Δ | §5.4 bound applies? |
|---|---:|---:|---:|:---:|
| `kq` (pre-softmax `Q·K^T`) | 48.63 % ± 0.80 | 47.22 % ± 0.84 | −1.41 pp | **No** (qualitative `≈_{e_C^attn}` only) |
| `kqv_out` (attn output, per head, pre W_o) | **97.46 % ± 0.40** | **83.77 % ± 0.64** | **−13.69 pp** | **Yes** (composed into accuracy bound) |

The §5.4-bounded surface has a real but bounded defense delta at L=0 (~14 pp); at deeper layers it collapses to <1.5 pp.

## Per-layer

| Layer | Plain `kqv_out` top-1 | Obf `kqv_out` top-1 | Δ | Plain top-10 | Obf top-10 |
|---:|---:|---:|---:|---:|---:|
| 0  | 97.46 % ± 0.40 | 83.77 % ± 0.64 | **−13.69 pp** (~21σ) | 99.83 % | 91.55 % |
| 5  | 50.77 % ± 0.89 | 50.22 % ± 0.64 | −0.55 pp (~0.5σ) | 61.62 % | 61.22 % |
| 11 | 18.50 % ± 0.39 | 19.83 % ± 0.65 | +1.33 pp (~1.7σ) | 31.86 % | 32.89 % |
| 17 | 16.67 % ± 0.45 | 17.75 % ± 0.53 | +1.08 pp (~1.6σ) | 27.83 % | 28.42 % |
| 23 | 25.27 % ± 0.47 | 25.99 % ± 1.04 | +0.72 pp (~0.6σ) | 39.32 % | 39.28 % |

Wall: ~57 s per (layer, cell) on rocSOLVER GPU; ~10 min total.

## Signal-correlation diagnostic

Element-wise plain–obf correlation, first 3 prompts:

| Surface | L=0 | L=17 |
|---|---:|---:|
| `kq` (pre-softmax) | 0.58–0.66 | 0.89–0.91 |
| `kqv_out` (attn output) | −0.009 / 0.015 / 0.030 | 0.014 / 0.031 / −0.008 |

The output surface is **essentially uncorrelated** between plain and obf at the per-element level (corr ≈ 0). Yet ridge still recovers 83.77 % at L=0 — the obfuscation rotates / scales the feature space but preserves the *learnable* relationship between activation and token identity. This is the signature of K_a-covariant (paper §3.3) defense: signal-level perturbation is large, but the rank-1 covariance structure between features and labels is preserved up to a learned linear map, which ridge fits.

## Interpretation

### The tightened theorem is empirically supported at L ≥ 5

The tightened theorem (`docs/research/aloepri-attacks.md` §"Attention-output covariance"): defense delta on the output surface is `O(L_A · e_C^attn_output)`. At L=5, 11, 17, 23 the measured delta is within noise (±1.5 pp at 0.5-1.7σ). Consistent with the theorem.

### L=0 has a real but bounded 14 pp delta

Embedding noise α_e=1.0 enters at the residual input to the first decoder layer. The attention output of L=0 inherits this noise; the obf cell adds it directly, the plain cell does not. The 14 pp gap is consistent with that one-shot noise injection (Algorithm 1 line 4) rather than with Algorithm 2's per-layer covariance. Confirming this would require a noise-isolated cell (α_e=0 obf) at L=0; out of scope here.

### Paper Table 4's 87 → 0 pp transition is NOT reproduced on either surface

- Pre-softmax `kq`: paper would expect 87 % → 0 %; we measure 49 % → 47 % (delta 1.4 pp).
- Post-V output `kqv_out`: paper would expect 87 % → 0 %; we measure 97 % → 84 % (delta 14 pp).

Neither surface matches paper Table 4's defense pattern. The three reconciling hypotheses listed in the theorem doc remain open:
1. **Surface mismatch** — paper measures a third surface we haven't captured (e.g., post-W_o residual contribution, or attention output as `outputs.attentions[L]` which is the post-softmax probability not the post-V output). The latter we already tested via offline softmax reconstruction at 30.60 % obf — still nowhere near 0 %.
2. **Attack class mismatch** — paper's ISA per Appendix D.1 is gradient-opt on input embeddings; ridge bypasses the optimization landscape.
3. **Π handling in metric** — paper TTRSR may compare against the obfuscated-vocab ground truth.

Step 3 of the post-grilling plan (paper-faithful gradient-opt ISA) is the highest-leverage next test for hypothesis (2).

### Implication for AloePri

Both the score surface (47 %) and the attention-output surface (84 % at L=0, ~25 % at L=23) leak heavily to a ridge attacker. The full Alg2 obfuscation provides at most 14 pp of defense on the worst-case L=0 output surface, and ~0 pp elsewhere. **AloePri-style obfuscation is not sufficient for attention-surface privacy under a ridge threat model.** TEE-protected attention (path-1) or a non-covariant score-/output-surface perturbation that intentionally violates §5.4's invariance bound is required.

## Raw artefacts

- Plain captures: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-plain-attnoutput-512-20260526/captures/{attn.safetensors,attn.meta.json}` (775 MB)
- Obf captures: `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-attnoutput-512-20260526/captures/` (775 MB)
- Ridge log: `/tmp/2B1-attn-output-ridge.log`
- Driver: `evals/aloepri-attacks/m2_7/diagnose_isa/gpu_sweep.py --kind kqv_out --skip-per-head`
- Sanity diff: plain–obf element-wise correlation ~0.01-0.03 (above; not regenerated as a separate artefact — `compare_plain_obf.py` template can be retargeted at these paths if needed).

## What we did not measure

- **Vocab-disjoint split** (memory note `feedback_isa_attnscore_methodology` flags this as the methodology that better matches a real attacker with disjoint train/test vocab). Defer to Step 2B.3 of the post-grilling plan.
- **Per-head ridge** on `kqv_out` (skipped via `--skip-per-head`; cheap to add if needed).
- **Post-W_o residual contribution** (`kqv_out` named in llama-graph.cpp:2132 — semantically different from our captured tensor; would require a separate capture pass). Open question whether paper's "AttnScore" surface is this.
