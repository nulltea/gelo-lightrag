---
type: dev-log
status: current
created: 2026-05-20
updated: 2026-05-26
tags: [aloepri, aloepri, qwen3, llama-cpp, deployment]
companion: [2026-05-20-aloepri-pi-special-token-fix, 2026-05-20-quantization-tried-deferred, 2026-05-21-strong-pi-server-patch, 2026-05-21-aloepri-next-steps, 2026-05-26-aloepri-recommendations]
---

# AloePri deployment — distilled fix log

> Knowledge layer for the aloepri (AloePri on Qwen3-1.7B via patched
> llama.cpp) deployment workstream. Server-side patches, special-token
> handling, quantization verdicts, and the 2026-05-26 paper-literal
> Algorithm 2 migration decision.

## Current deployment config

**Target:** Qwen3-1.7B, pre-norm-only (28 layers × 2 norms + 1 output = 57 residual norm sites)

**Recommended flags (per 2026-05-26):**
- `--mode keymat --expansion-size 128 --seed 42 --lam 0.3`
- `--alg2-paper-literal` (migration from default; higher defense on attention surfaces — see below)
- `--pi --pi-seed 42424242` with special-token exclusion (fixes EOS/BOS/control token handling)
- `--noise-alpha-e 0.3 --noise-alpha-h 0.1 --noise-seed 13371337` (calibrated for residual 5/20 server-500s with coherent output)

**Artifact storage:**
- fp32 keymat required for training/tuning; Q8_0 requantization tested and stable at inference per 2026-05-21 (Gate A cleared).
- Deployment format: `keymat-h128-pi-noise-ae0.3-ah0.1-fp32.gguf` (9.1 GB fp32) or Q8_0 variant (~2.5 GB).

---

## Server-side patches

### Special-token fix (EOS/BOS exclusion in Π)

**Date:** 2026-05-20. **Status:** Landed.

- **Symptom:** HumanEval pass@1 dropped to 0/20 with multi-language gibberish; llama-server returned 500 errors on 11/20 prompts.
- **Root cause:** `obfuscate_qwen3_gguf.py` permuted *all* tokens in `[0, 151669)` including EOS (151645), BOS (151643), and 26 special tokens (chat template markers, control tokens). After Π, the inference server's "stop on EOS token id 151645" check never fired. The model learned to emit `inv_τ[151645]` to mean stop, but the server saw a different obfuscated id and ran to `max_tokens`, eventually emitting invalid UTF-8.
- **Fix:** `obfuscate_qwen3_gguf.py:297-368` now reads `tokenizer.ggml.token_type` from the source GGUF, filters permutable set to `type ∈ {NORMAL=1, BYTE=6}`, leaves `CONTROL, USER_DEFINED, UNUSED, UNKNOWN` at identity. For Qwen3-1.7B: 151,643 tokens permuted, 26 special tokens kept identity.
- **Privacy implication:** None. Special token IDs are public (tokenizer config in GGUF metadata, distributed openly).
- **Accuracy after fix:**
  - keymat-h128-pi-noise-ae0.3-ah0.1-fp32 (fixed Π): **7/20 = 35.0 %** (Δ vs plain = −15 pp)
  - Residual server-500s: 5/20 (down from 11/20 with broken Π)
  - Quality probe: 5/5 readable, is_prime textbook response, correct French translation

### Strong-Π server patch (chat_parser=epsilon workaround)

**Date:** 2026-05-21. **Status:** Landed.

- **Goal:** keep llama-server `/completion` 100 % robust under multi-language gibberish output from strong-Π (all 151,669 active tokens permuted).
- **Solution:** supply custom epsilon PEG parser to `chat_parser` request field. No llama.cpp source patch required.
- **Mechanism:** stock llama.cpp's `task_result_state::update_chat_msg` runs `common_chat_parse` on every chunk with default PEG `content(rest()) + end()`. With strong-Π, the cumulative de-tokenized text is gibberish, throwing "Failed to parse input" on ~5–9 % of token sequences.
- **Fix:** `python/aloepri-llm/aloepri_client.py` sends:
  ```json
  {"parsers":[{"type":"epsilon"}],"rules":{},"root":0}
  ```
  The epsilon primitive matches the empty prefix and never fails. `update_chat_msg` sees `new_msg.empty()` and falls through without throwing. The streamed `tokens` field (independent code path) is unaffected.
- **Tested:** 65/65 corpus prompts pass through both raw HTTP and `AloePriClient.complete()` paths.

---

## Quantization × aloepri

### fp32 keymat constraint

Q8_0 quantization on the obfuscated weights collides with low-variance blocks (max=55, std=4.7 at layer 27) where Q8_0's per-32-element scaling (0.4 % block-max error) rounds small values to zero. **Empirically: fp32 required for weight obfuscation fidelity; Q8_0 requantization at inference is stable (Gate A cleared 2026-05-21).**

### Q4, Q6_K, f16 variants (tried, deferred)

Per `2026-05-20-quantization-tried-deferred.md`:
- **Q4 GPU spike (Vulkan/HIP):** 0.71–0.82× speedup on Qwen3-4B; relative error 11–13 % (pre-rotation Q4 baseline, matches QuIP#). cubek-matmul 0.9.0 doesn't emit WMMA INT4 intrinsics on Strix Halo iGPU (gfx1151). **Deferred to discrete-GPU deployment.**
- **f16 engine:** only FfnDown (wide weights) wins meaningfully (1.47×); three of four shapes tied or slower. Decision rule: selective-f16 routing only if measured ≥2 % TTFT gain at n=2048 (not yet implemented).

### Verdict

AloePri uses fp32 keymat with no quantization on obfuscated weights. Discrete-GPU deployment can revisit Q4 once WMMA kernels are available.

---

## Recommended config (per 2026-05-26 paper-literal migration)

**Deployment construction:** paper-literal Alg2, not prior default.

**Key measurement shift:** our deployed cell was understating AloePri's actual defense by 7–40 pp on both attention-output and score surfaces.

**Defense surface deltas under paper-literal (vs prior default):**
- Attention output (§5.4-bounded): **50 pp at L=0**, 40 pp at L≥5 (vs prior 14 pp at L=0, 0.5 pp at L≥5).
- Score surface (kq): 16–31 pp at L≥5 (vs prior ~0 pp), single-digit obf TTRSR at L≥5 (vs prior 47 %).

**Accuracy preservation requirement:** bf16 inverse loss for paper-literal Û_vo (500× higher condition number) is the new precision risk; needs verification before finalisation.

**Threat-model reading:** TEE-protected attention (path-1) remains gold standard for L=0 adversaries. Even paper-literal Alg2 leaks 43 % on kq at L=0 (embedding-noise shadow); only an in-TEE first decoder layer eliminates it.

For full per-attack measurement detail see [`aloepri-attack-bench-log.md`](aloepri-attack-bench-log.md).

---

## Pending work

1. **Build + measure α_e=0.1 cell** — confirm residual 5/20 server-500s clear; validate noise/quality tradeoff curve for d=2048.
2. **Rebuild + remeasure Algorithm 2 cells with fixed Π** — all prior Alg-2 GGUFs stacked on broken Π.
3. **Verify bf16 accuracy under paper-literal Û_vo** before deployment recommendation is final.
4. **IMA-EmbedRow-ridge mitigation** — separate workstream requiring keymat redesign (h up, λ up, or alternative P̂ family); ridge attack retired per [`aloepri-attack-bench-log.md`](aloepri-attack-bench-log.md).
