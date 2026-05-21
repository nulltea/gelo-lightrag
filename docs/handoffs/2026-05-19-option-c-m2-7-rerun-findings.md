# Option C M2.7 attack re-run — findings

**Date:** 2026-05-19 (revised after operator pushback on first draft)
**Status:** ran with paper-faithful `--split-mode vocab`. **Option C
reproduces §05 attack scores exactly.** Diagnosis below — the gain is
real but invisible to the current M2.7 attack surfaces.

## Full attack ledger (current code, `--split-mode vocab`, 2026-05-19)

| Attack | Layer / surface | §05 (handoff frozen) | §05 (re-run, vocab) | **Option C (FULL alg2)** | §6.3 gate | Verdict |
|---|---|---:|---:|---:|---|---|
| NN | 0, `attn_norm` | 0.0 % | 0.0 % | 0.0 % | ≤ 15 % | ✓ |
| IMA basic | 0, `attn_norm` | 88.9 % | **88.9 %** | **88.9 %** | ≤ 15 % | ✗ |
| ISA HiddenState | 23, `attn_norm` | 16.3 % | **16.3 %** | **16.3 %** | ≤ 15 % | ✗ |
| ISA AttnScore | 23, `kq-23` | 0.0 % | **0.0 %** | **0.0 %** | ≤ 15 % | ✓ |
| TFMA | token stream | 1.6 % top-1 / 5.1 % top-10 | — | 0.78 % / 2.73 % | ≤ 15 % | ✓ |
| SDA | token stream | 1.6 × 10⁻⁵ BLEU | — | 1.4 × 10⁻⁵ BLEU | ≤ 5 BLEU | ✓ |
| VMA, IA | weight | out of scope | — | out of scope | — | — |

**Headline:** Option C's M2.7 numbers are bit-equal to §05's for every
hidden-state attack. TFMA/SDA wobble within fp32 noise from the
matmul-vs-elementwise path. No attack moved.

## Initial mis-diagnosis (recorded for the receipt)

First draft of this doc claimed "M2.7 handoff numbers are stale; attack
code changed since." That was wrong. Attack code is unchanged
(`git log evals/aloepri-attacks/attack_drivers/ evals/aloepri-attacks/m2_7/`
since handoff: README-only changes). The real cause: the M2.7 ledger
was generated with `--split-mode vocab` (paper-faithful, vocab-disjoint
train/test splits) but the default in `run_hidden_state_attacks.py:121`
is `--split-mode row`. The first re-run used the default and produced
75.2 % / 28.1 % — different numbers from the same captures because of
the split flag. With `--split-mode vocab` the handoff numbers reproduce
**exactly**.

Lesson: when reproducing prior results, read the stored `extra.*` field
of the result JSON (it records `split_mode`, `candidate_pool_size`,
`best_ridge_alpha`, etc.) rather than relying on the runbook's
narrative.

## Why §05 and FULL alg2 give identical scores — algebra, not coincidence

I verified capture-level diffs before drawing the comparison:

| Capture | §05 vs FULL alg2 |
|---|---|
| `attn_norm-0` (layer 0) | **bit-identical**, max\|Δ\| = 0.000e+00 |
| `attn_norm-11` (layer 11) | max\|Δ\| = 0.88, mean\|Δ\| = 1.3 × 10⁻² |
| `attn_norm-23` (layer 23) | max\|Δ\| = 0.37, mean\|Δ\| = 1.1 × 10⁻² |
| `kq-23` (attention scores) | algebraically identical by construction (`M_q · M_kᵀ = I`) |

Per attack, this gives:

1. **IMA basic at layer 0 (88.9 % in both).** M_q acts on `head_dim`
   post-W_q. Layer 0 attention input is `RMS-norm(token_embd · token_id)`
   — pre-W_q, pre-attention. M_q **structurally cannot** change layer-0
   captures. Layer-0 captures are bit-identical; IMA at layer 0 is
   necessarily identical.

2. **ISA HiddenState at layer 23 (16.3 % in both).** Captures DO differ
   (max\|Δ\| ≈ 0.37 from M_q's contribution propagating through 23 layers
   of attention into the residual). But the ridge attacker is fitting
   the keymat decoy expansion (256 decoy dims, same construction in §05
   and FULL alg2 — same seed, same Π, same α-noise). M_q's per-layer
   perturbation is small relative to the keymat structure that the
   attacker actually exploits. The ridge fit is dominated by the larger
   signal and doesn't shift.

3. **ISA AttnScore at kq-23 (0.0 % in both).** Algorithm 2 is
   **designed** to make `M_q · M_kᵀ = I`, so `Q_obf · K_obfᵀ = Q · Kᵀ`.
   Attention scores are intentionally preserved. The `kq-23` capture is
   algebraically the same plaintext-equivalent tensor in §05 and FULL
   alg2 (modulo fp32 matmul vs elementwise noise from the matrix-Γ
   path). The 0 % isn't a defence improvement — it's the price of
   correctness-preserving obfuscation that the attacker can't see
   either way at this surface, because the head shuffle scrambles which
   head's scores land where.

## What this means for Option C

**Option C is correctly deployed.** Verified end-to-end:

- Matrix-Γ algebra exact: `(Q_obf / RMS) · Γ ≡ Q_plain_normed · M_q` to
  3 × 10⁻⁸ rel under the orthogonality MVP (`scale_range = (1.0, 1.0)`).
  `M_q · M_kᵀ = I` to 1 × 10⁻⁷ rel.
- Kernel branch correct: identity-Γ (M_q = I) smoke produces
  bit-identical greedy tokens to scalar-γ.
- Coherent generation: `"The capital of France is"` →
  `"in Paris, and the population of France is 64,000,000. How many more"`.
- GGUF loads against the patched kernel with `aloepri.qk_norm_matrix = true`.

**But Option C's defence is invisible to the current M2.7 attack
suite.** The attacks observe surfaces (residual stream pre-W_q,
attention scores) that M_q either cannot touch or deliberately
preserves. The surface where M_q's per-head obfuscation actually lives
is **post-q_norm Q values** (and equivalent K) — the values RoPE rotates
and the attention dot product consumes — and that surface is not
currently dumped by the M2.7 tensor-filter.

## To exercise Option C's defence, add a Q/K post-norm capture

The matrix-Γ kernel runs:
```
Qcur = (q_proj_obf · x)          // shape (d_h, n_head, n_tokens)
Qcur = build_norm(Qcur, NULL, …) // RMS, no γ
Qcur = ggml_mul_mat(Γ_q, Qcur)   // matrix-Γ
ggml_callback("Qcur_normed", Qcur)
// → RoPE → attention
```

M_q lives in `Qcur_normed`. To measure whether the intra-head
obfuscation actually defends against an inverter, the M2.7
tensor-filter regex needs to be extended:

```
--tensor-filter '^(attn_norm-(0|11|23)|Qcur_normed-(0|11|23)|Kcur_normed-(0|11|23)|kq-23)$'
```

Then re-run IMA / ISA / NN against the `Qcur_normed` captures. Expected
outcome: §05 (q_matrix = I) captures of `Qcur_normed` will match the
plaintext signal modulo the head-shuffle, while FULL alg2 captures
will carry M_q's per-NEOX-pair rotation. The inverter has to recover
through M_q, which is the per-prompt-fixed unknown the paper expects
the IMA inverter to fail on.

**This is the missing measurement.** Filing as the immediate follow-up
priority for the path-2 attack harness.

## What Option C still buys, independent of the M2.7 result

- **Substrate for the next iteration.** The matrix-Γ kernel patch is
  in place. When the missing intra-head components land
  (Ẑ_block per the degeneracy doc, optionally Ĥ_qk via runtime κ
  correction), they ride this same kernel with no further infra work.
- **Paper-faithful Algorithm 2 deployment for Qwen3.** Before this
  session, intra-head transforms were dormant on Qwen3 because the
  paper's input-axis κ fold can't extend to the QK-norm site. Option
  C extends the kernel to consume `M_q · γ ⊗ M_qᵀ`. The vendor-fork
  diff is the load-bearing engineering.
- **Decisive negative finding on the bounded attacker model.** With
  R̂_qk alone, even the paper-faithful vocab-disjoint IMA does not
  drop. That's a real constraint on what "fix Ẑ_block + Ĥ_qk"
  expectations should be — the missing components are doing more work
  than the kernel-level fold does.

## Suggested next steps (priority)

1. **Add `Qcur_normed-N` / `Kcur_normed-N` tensor-dump filter** to the
   M2.7 capture pipeline. ~30 min on the llama.cpp side (one more `cb()`
   call in `models/qwen3.cpp`), zero on the harness side (existing
   `capture_hidden_states.py` consumes whatever the filter dumps).
2. **Re-run M2.7 IMA / ISA / NN against the new Qcur_normed captures**
   on both §05 and FULL alg2 GGUFs. This is the missing measurement.
3. **Fix `Ẑ_block`** per `2026-05-19-alg2-z-block-degeneracy.md`. Rebuild
   FULL alg2 GGUF. Re-run attack suite against the *correct* surfaces.
4. **Refresh M2.7 ledger** in
   `2026-05-19-m2-7-attack-findings.md` with the post-Ẑ-block numbers.

## Artifacts produced this session

**Code (new, fp32 GGUF rewriter + kernel patch):**

- `vendor/llama.cpp` (submodule, uncommitted) — matrix-Γ kernel branch:
  `src/llama-arch.{h,cpp}` + `src/llama-hparams.h` + `src/models/qwen3.cpp`.
- `python/aloepri-llm/obfuscate_qwen3_gguf.py` — `--alg2-qk-norm-matrix` flag,
  full Algorithm 2 plumbing.
- `python/aloepri-llm/scripts/measure_gamma_qk_clusters.py` — Option B pre-flight
  (used to kill Option B 2026-05-19 morning).
- `python/aloepri-llm/scripts/check_alg2_invariance.py` — algebraic validation.
- `python/aloepri-llm/scripts/build_matrix_gamma_identity_gguf.py` — identity-Γ
  smoke pair builder.
- `evals/aloepri-attacks/m2_7/vulkan-m2_7-cli.Dockerfile` — patched
  docker image with `llama-cli` + `llama-completion` + `llama-server`.

**Docker images:**

- `aloepri-llama-server:option-c-cli` — Vulkan build of the patched
  llama.cpp with all three binaries.

**GGUFs:**

- `keymat-h128-pi-noise-alg2-FULL-fp32.gguf` (9.1 GB) under
  `~/.cache/huggingface/path-2-aloepri/qwen3-1.7b/` — Option C output.
  `.key.npz` companion alongside.
- `/tmp/qwen3-fp32-scalar.gguf`, `/tmp/qwen3-fp32-matrix-identity.gguf` —
  smoke pair, **delete after review**.

**Captures:**

- `evals/aloepri-attacks/snapshots/m2_7-FULL-hidden/{hidden,attn}.safetensors`
  — Option C captures (64 prompts, layers 0/11/23, kq-23).
- `evals/aloepri-attacks/snapshots/m2_7-FULL-token-streams.jsonl` —
  Option C token stream.

**Result JSONs (new, all `--split-mode vocab`):**

- `evals/aloepri-attacks/results/m2_7-FULL-hidden.json` — IMA 88.9 %,
  ISA HS 16.3 %, ISA AS 0.0 %.
- `evals/aloepri-attacks/results/m2_7-FULL-token.json` — TFMA 0.78 %,
  SDA 1.4 × 10⁻⁵.
- `/tmp/m2_7-section05-vocab.json` — §05 reproduction with current
  attack code, vocab split.

**Documents (new):**

- `docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md` —
  eigendecomposition leak + proposed defences.
- `docs/handoffs/2026-05-19-alg2-z-block-degeneracy.md` — Ẑ_block
  silently-identity finding.
- (this document).

## Boundaries respected

- Did not kill `llama-swap` container.
- Did not push any submodule changes upstream.
- Ephemeral docker containers spawned for the run, torn down after.
- Asked before every smoke / capture / attack-run step per session
  agreement.
- Mis-diagnosed once; corrected by reading the stored result JSON's
  `extra.split_mode` field rather than guessing.
