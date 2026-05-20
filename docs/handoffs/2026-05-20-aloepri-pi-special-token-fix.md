# Handoff — AloePri Π special-token bug + accuracy regression diagnosis

**Date:** 2026-05-20
**Branch:** `path-2-aloepri-gemma`
**Status:** Root cause identified, fix landed in obfuscator, accuracy gap shrunk from −45pp to −15pp.

## The 30-second version

The path-2 Qwen3-1.7B obfuscator was permuting token IDs in `[0, 151669)` **including the EOS, BOS, im_start, im_end, fim_*, and tool-call special tokens**. After Π, the inference server's standard "stop on EOS token id 151645" check never fires (the model has learnt to emit `inv_τ[151645]` to mean stop, but the server sees a different obf id). Every obfuscated `/completion` runs to `max_tokens`, the model drifts off-manifold past its natural stop point, eventually emits an invalid utf-8 sequence, and llama-server's JSON encoder returns 500.

HumanEval pass@1 ledger:

| Stage | Server | pass@1 (n=20) | Δ vs plain | Notes |
|---|---|---|---|---|
| plain Q8_0 (reference) | upstream | **10/20 = 50%** | — | baseline |
| gamma-only (§5.2.5 fusion regression test) | upstream | 9/20 = 45% | −5pp | within sampling noise |
| keymat-h128-fp32 (bare Alg-1) | upstream | 8/20 = 40% | −10pp | matches paper claim |
| keymat-pi-noise (α_e=1.0, **broken Π**) | patched | 0/20 = 0% | −50pp | broken |
| keymat-pi-noise (α_e=1.0, **broken Π**) | upstream | 0/20 = 0% | −50pp | **identical to patched → exonerates our kernel patch** |
| keymat-pi-noise-alg2 partial (broken Π) | patched | 1/20 = 5% | −45pp | broken |
| keymat-pi-noise-alg2 FULL-zfix-hadamard (broken Π) | patched | 1/20 = 5% | −45pp | broken |
| keymat-pi-noise (α_e=0.3, ah=0.1, **broken Π**) | upstream | 2/20 = 10% | −40pp | quality coherent, server crashes |
| **keymat-pi-noise (α_e=0.3, ah=0.1, FIXED Π)** | upstream | **7/20 = 35%** | **−15pp** | **fix applied** |

## What the diagnosis looked like

Started from a baseline `keymat-h128-pi-noise-alg2-FULL-zfix-hadamard` GGUF reporting HumanEval pass@1 = 5%, plain Q8_0 = 50%. Paper claims <3pp accuracy loss; we were measuring −45pp.

Ran a top-down ablation: stripped one stage at a time, measured HumanEval n=20 + 5-prompt quality probe at each. Found:
1. **§5.2.5 norm fusion** (gamma-only mode) is mathematically exact (κ=1.0) — bit-identical to plain mod fp32 quant. Regression test passes.
2. **Bare Algorithm 1 keymat** (`keymat-h128-fp32`, no Π / noise / Alg 2) gives 8/20 = 40% — matches paper's expected accuracy budget at this n.
3. **The cliff is at Π + noise** stacking on top of keymat — pass@1 drops to 0%.

Tested the patched llama.cpp kernel against upstream `ghcr.io/ggml-org/llama.cpp:server-vulkan`: **identical 0/20 with identical server-500s on identical HumanEval problems**. Our matrix-Γ patch is a pure `if (aloepri_qk_norm_matrix) { matrix path } else { original path }` conditional; the `else` is bit-identical to upstream. Kernel exonerated.

Switched to lowering noise (α_e: 1.0 → 0.3, α_h: 0.2 → 0.1). Pass@1 only climbed to 10%, but quality probe showed dramatic coherence improvement (textbook `is_prime` response, correct French translation, factually-correct Paris). HumanEval still hit **11/20 server-500s**, more than at α_e=1.0 (3/20). That was the smoking gun — server-side crash rate was uncorrelated with noise level, suggesting a non-noise cause.

Server log showed the 500s came from llama-server's response encoder failing on multi-language gibberish token sequences in long outputs. Cross-checking against Stage 1 (no Π): **0 server-500s**, regardless of how the model performs accuracy-wise. The differentiator was Π itself, not noise.

Read `python/path-2/obfuscate_qwen3_gguf.py:308-323`:

```python
pi_active_size = 151669
perm = pi_rng.permutation(pi_active_size).astype(np.int32)
tau = np.arange(n_vocab, dtype=np.int32)
tau[:pi_active_size] = perm
```

`[0, 151669)` includes the EOS tokens (151643, 151645) and all chat-template markers (151662-151664, etc — 26 special tokens in total per `tokenizer.ggml.token_type`). Π was permuting them. Confirmed via inv_τ trace.

## The fix

`python/path-2/obfuscate_qwen3_gguf.py:297-368` now reads `tokenizer.ggml.token_type` from the source GGUF, filters to `type ∈ {NORMAL=1, BYTE=6}` for the permutable set, and leaves all other token IDs (CONTROL, USER_DEFINED, UNUSED) at identity. For Qwen3-1.7B this is 151,643 permuted, 26 kept identity.

**Privacy implication of leaving specials at identity:** none. Special token IDs are public knowledge (the tokenizer config is part of the GGUF metadata, distributed openly). Permuting them gains zero confidentiality and costs the entire inference-server stop-token plumbing.

## Cell metrics after the fix

`keymat-h128-pi-noise-ae0.3-ah0.1-fp32.gguf` (fixed Π):
- HumanEval pass@1 (n=20): **7/20 = 35.0%** (Δ vs plain = −15pp)
- Quality probe: 5/5 readable, is_prime textbook, French translation correct
- IMA-EmbedRow-ridge top-1: **99.22%** (unchanged from prior 99.22% / 97.66% baselines — this is a structural attack at d=2048/h=128/λ=0.3, neither Π fix nor noise change can move it)
- Residual server-500s on HumanEval: 5/20 (down from 11/20)

## Outstanding issues

1. **Residual 5/20 server-500s.** The model still degenerates on certain long HumanEval prompts (HumanEval/163, 28, 70, 57, 143) into multi-language gibberish that breaks llama-server's JSON encoder. Smaller failure mode than before but real. Suspected residual noise effect — next cell will test α_e=0.1 to confirm.
2. **IMA-EmbedRow-ridge at 99.22%.** This is the structural attack on the embed-row bijection. Requires keymat parameter changes (h, λ) to move, not Π/noise. Out of scope for this fix.
3. **Algorithm 2 cells haven't been re-measured** with the fixed Π. All previous Alg-2 measurements were against broken Π. Need to rebuild Alg-2 GGUFs from scratch.

## What was deleted

To prevent confusion, all Π-permuted GGUFs were removed from `~/.cache/huggingface/path-2-aloepri/qwen3-1.7b/`:
- `keymat-h128-pi-*.gguf` and `.key.npz` (Stages 2-6, ae0.3-v1)
- `keymat-h2*.gguf` and `keymat-h128.gguf` (older test artifacts)
- `keymat-h128-Q5_K_M.gguf`, `Q6_K`, `Q8_0` (quantised Stage 1 variants)

Kept as ablation references (no Π):
- `gamma-only.gguf` — §5.2.5 fusion correctness regression test
- `keymat-h128-fp32.gguf` — bare Algorithm 1 (Stage 1)

Tracked JSONs from before today's investigation were `git rm`'d:
- All `evals/aloepri-attacks/results/m2_7-*.json` (FULL-hadamard / FULL-zfix attack reads, plain / plain-vocab snapshots, paper-like sweep snapshots, hidden / token-stream snapshots). Recoverable from git history.

## File map (today's session)

**New:**
- `evals/aloepri-attacks/m2_7/run_quality_humaneval.py` — per-cell quality probe (5 prompts) + HumanEval pass@1 driver. Uses AloePriClient for τ-mapping. Supports `--plain-mode` for capturing the plain reference. Default n_humaneval=50; pass `--n-humaneval 20` for fast sweep cells (≈5 min each).
- `docs/handoffs/2026-05-20-aloepri-pi-special-token-fix.md` — this document.
- `evals/aloepri-attacks/results/sweep/` — full investigation record (8 cells × {run.log, quality-humaneval.json, optional ima-embedrow.json}). Includes plain reference, ablation stages 1-3, ae0.3 broken vs fixed comparison.

**Modified:**
- `python/path-2/obfuscate_qwen3_gguf.py` — Π special-token exclusion (the fix).
- `evals/aloepri-attacks/m2_7/run_all_m2_7.py` — `--allow-quality-humaneval` step that calls the new driver.

## Suggested next steps

1. **Build + measure α_e=0.1 cell** (immediate next). If residual 500s clear, confirms the noise/quality tradeoff curve for Qwen3-1.7B's d=2048.
2. **Rebuild + remeasure Algorithm 2 cells** with the fixed Π. The old Alg-2 GGUFs were stacking Alg-2 on top of broken Π; we need clean numbers.
3. **IMA-EmbedRow-ridge mitigation** is a separate workstream — requires keymat redesign (h up, λ up, or alternative P̂ family). Sweep can wait until accuracy/quality cell selection is settled.
4. **Update `docs/prototype/aloepri-llm.html` §08** with the new accuracy numbers + the Π fix note once the cell selection is final.

## Test-the-fix recipe (for reproducibility)

```bash
# Rebuild
PLAIN_GGUF=~/.cache/huggingface/hub/models--bartowski--Qwen_Qwen3-1.7B-GGUF/snapshots/dcb19155b962dbb6389f4691a982043a8e651022/Qwen_Qwen3-1.7B-Q8_0.gguf
OUT=~/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-ae0.3-ah0.1-fp32.gguf
python/path-2/.venv/bin/python python/path-2/obfuscate_qwen3_gguf.py \
  --in "$PLAIN_GGUF" --out "$OUT" --mode keymat \
  --expansion-size 128 --seed 42 --lam 0.3 \
  --pi --pi-seed 42424242 \
  --noise-alpha-e 0.3 --noise-alpha-h 0.1 --noise-seed 13371337

# Spawn upstream server
docker run --rm -d --name aloepri-m2_7-server \
  -p 127.0.0.1:8061:8080 -v "$(dirname $OUT):/models:ro" --device /dev/dri \
  ghcr.io/ggml-org/llama.cpp:server-vulkan \
  -m "/models/$(basename $OUT)" -ngl 999 -np 1 --flash-attn on \
  -c 4096 --ubatch-size 1024 --host 0.0.0.0 --port 8080

# Measure
python/path-2/.venv/bin/python evals/aloepri-attacks/m2_7/run_quality_humaneval.py \
  --endpoint http://127.0.0.1:8061 --key "$OUT.key.npz" \
  --output /tmp/cell.json --n-humaneval 20
```
