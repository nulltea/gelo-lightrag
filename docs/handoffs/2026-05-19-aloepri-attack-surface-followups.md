# Handoff — AloePri attack-surface followups + Algorithm 1 sweep + GELO-like dynamic defence + Gram leakage

**Date:** 2026-05-19
**Branch:** `path-2-aloepri-gemma` (uncommitted: matrix-Γ kernel patch in
`vendor/llama.cpp`, `lib/alg2.py` repairs, obfuscator new flags,
attack-driver 3-D snapshot flatten).
**Status:** Option C deployment + steps 0/1/2a done, M2.7 re-measured.
Five follow-up threads queued for the next session.

## Where we are

The 2026-05-19 ramp deployed full Algorithm 2 on Qwen3 1.7B via a
**matrix-Γ kernel extension** in the patched `llama.cpp`, plus
repairs in `python/aloepri-llm/lib/alg2.py` (Ẑ_block + M_k construction)
and the obfuscator (`--alg2-qk-norm-matrix`, `--alg2-h-hadamard-signs`).
The current GGUF is
`~/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-alg2-FULL-zfix-hadamard-fp32.gguf`.

Six of eight §6.3 gates pass at this condition; **IMA-L0-activation
and IMA paper-like (path-2 variants, activation surface at layer 0)
fail at 88.9 % / 38–47 %** — Algorithm 2 cannot reach the
`attn_norm-0` surface (pre-W_q). See
[`2026-05-19-option-c-steps-0-1-2a-findings.md`](2026-05-19-option-c-steps-0-1-2a-findings.md)
for the full ledger.

User-facing doc updated at `docs/prototype/aloepri-llm.html` §08:
attack table now split into in-scope activation/token attacks vs
collapsible weight-inversion table, with a highlighted row for the
new QK-norm Γ eigendecomposition attack. Threat-model doc lives at
`docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md`.

## Five threads for the next session

### 1. Reframe and re-measure the "weight-inversion" attacks as prompt-inversion via weights

**Diagnosis.** Path-2's docs mislabel VMA / IA / paper-IMA-basic /
paper-IMA-paper-like as "weight inversion attacks (out of scope
because weight privacy isn't load-bearing)". All four are actually
**prompt-inversion attacks** — they recover τ (the token
permutation) from static weight tensors, and recovering τ lets an
attacker decode every wire-side prompt to the deployment. The "weight
inversion" framing in the doc was conflating *observation surface*
(weight tensor rows) with *attack goal* (recover τ → decode
prompts).

**Why this matters.** Path-2 distributes the obfuscated GGUF openly,
so an attacker has both `W_e_obf` and `W_e_plain` simultaneously.
They can run any of VMA / IA / paper-IMA-basic / paper-IMA-paper-like
*offline*, no server access needed. Paper Table 1 reports these all
pass the 15 % gate on Qwen2.5-14B under full Algorithm 2, but **we
have not measured them against the path-2 Qwen3-1.7B
FULL-zfix-hadamard deployment**. If they fail on Qwen3, prompt
privacy is already broken regardless of activation-side defences.

**What to do.**
- VMA + IA: harness already built at
  `evals/aloepri-attacks/m2_7/run_static_attacks.py`. Run against the
  current FULL-zfix-hadamard GGUF + plain Qwen3-1.7B safetensors.
  Expected runtime ~5 min. Publish numbers in §08.
- paper-IMA-basic + paper-IMA-paper-like: port from
  `vendor/aloepri-py/src/security_qwen/ima.py:run_ima_baseline` and
  `run_ima_paper_like`. Both load `model.embed_tokens.weight` from
  the safetensors of both plain and obfuscated models. ~1-2 hours of
  porting.
- Update `docs/prototype/aloepri-llm.html` §08: rename the
  "Weight-inversion attacks (out of scope)" collapsed table to
  "Prompt-inversion attacks via static weights" and move it back to
  in-scope. Re-classify each row's verdict (pass/fail) once measured.

### 2. Sweep Algorithm 1 parameters to close the path-2 IMA-L0-activation gap

**Diagnosis.** IMA-L0-activation at 88.9 % reflects that
`attn_norm-0` is dominated by:
1. **`α_e = 1.0 × σ(W_e)` α-noise** — paper-default, modest.
2. **`h = 128` keymat** — paper-default, linear expansion.
3. **Linear keymat fold** — `P̂_R · Q̂_R = I`, easy to invert with
   ridge.

Algorithm 2 (intra-head) cannot reach this surface — M_q lives
post-W_q. The path-2 deployment is at paper defaults that the paper
chose for the *wire-side* attacker, not the activation-side
attacker.

**What to do.** Two-dimensional sweep on the obfuscator, measuring
IMA-L0-activation TTRSR + generation quality (MMLU subset, perplexity
on a held-out corpus, coherent-generation smoke):

| Lever | Range | Cost |
|---|---|---|
| `α_e` | 1, 2, 5, 10, 20 | trivial (param flag); each rebuild ~30 s |
| `h` (keymat expansion) | 128, 256, 384, 512 | storage grows linearly |

Pre-flight question: at what `(α_e, h)` does IMA-L0-activation drop
≤ 15 %? At what point does perplexity diverge unacceptably from
plain? The crossover sets the path-2 deployment defaults.

**Stretch.** Non-linear keymat fold (small MLP applied after
`P̂_R · x`). Paper-divergent; new research. Defer until the linear
sweep results are in.

### 3. QK-norm Γ eigendecomposition — measure and document the eigendecomposition leak

**Diagnosis.** The matrix-Γ kernel extension writes 2D
`Γ = M<sup>T</sup> · Diag(γ_qk) · M` at `blk.*.attn_q_norm.weight`.
This is a similarity transform of a diagonal matrix. Anyone with
`numpy.linalg.eig` recovers both `γ_qk` (eigenvalues) and `M`
(eigenvectors, up to head_dim-index permutation) in milliseconds —
defeating the intra-head obfuscation entirely.

This is documented in
`docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md` and
flagged with an orange-highlighted row in `aloepri-llm.html` §08, but
**not yet measured empirically**.

**What to do.**
- Implement the attack: load
  `keymat-h128-pi-noise-alg2-FULL-zfix-hadamard-fp32.gguf`, extract
  each layer's 2D Γ_q and Γ_k, run `np.linalg.eig`, recover M.
- Verify that the recovered M allows downstream IMA-L0-Q-activation to
  drop from 88 % to plaintext-level (~98 %), i.e. M-recovery undoes
  the obfuscation.
- Defence options (all break "static obfuscation, no infra change"):
  - **Per-prompt fresh M** (GELO-style) — see thread 5.
  - **Additive noise on Γ** — tune σ such that eigendecomp is noisy
    but generation stays coherent. Risk: γ_k values span [−1, 68] so
    SNR varies by orders of magnitude across eigenvalues.
  - **Hide M in the M̃ = R · H · Z product** by replacing
    `aloepri.qk_norm_matrix` form with a different parameterisation
    that doesn't expose M as a similarity transform — unknown
    feasibility, research question.

### 4. Port GELO-like dynamic defences and measure Gram leakage

**Why this is the strategic move.** Threads 1-3 are tuning the
existing AloePri/Option-C static-obfuscation envelope. Even fully
tuned, the static design has fundamental limits — the
eigendecomposition attack (thread 3) and the IMA-L0-activation
attack (which static keymat + α-noise can only partially defend)
both point at the same root: a single-key static transform is
information-theoretically recoverable given enough observation.
GELO addresses this by sampling **fresh masks per forward pass**
(see `docs/prototype/gelo-llm.html` and the path-1 implementation
under `crates/gelo-*/`).

**What to do.**
- Survey the path-1 GELO masking primitives: Haar, HD₃ Hadamard
  cascade (memory `[hd3_mask_landed]`), per-forward-pass +
  per-offload variants.
- Spec a path-2 deployment story: where would per-prompt fresh masks
  land in the obfuscation pipeline? Candidate: replace the static
  Γ_q / Γ_k tensors with a per-prompt-sampled M_q · γ · M_q<sup>T</sup>
  shipped via a thin client handshake. Breaks the "no protocol
  change" thesis but matches GELO's deployment model.
- **Measure Gram leakage.** GELO's path-1 research round 2 / 3
  (memory `[gelo_research_round_2]`, `[private_llm_inference_round_3]`)
  identifies Gram-matrix attacks as the main residual surface even
  under fresh masking. The path-2 attack harness extension for these
  attacks is in
  `docs/research/aloepri-attack-harness-followups.md` (per memory
  `[aloepri_hd3_gate_phase_a_b]`) — four attack drivers
  (anchor_ica / jade / jd / gram_error) already in evals. Run them
  against the static path-2 deployment + the proposed dynamic-mask
  variant, compare leakage.

This thread is the largest of the five — likely 2-3 weeks of
spike work — but if path-2's mission is durable prompt privacy on
Qwen3, it's the right destination.

### 5. Resolve the paper-vs-path-2 surface-mismatch in public docs

**Diagnosis.** Section §08 of `aloepri-llm.html` and the surrounding
prose still has some "out of scope" framing that conflated weight
privacy with prompt-inversion-via-weights (see thread 1). The
"covariant obfuscation" mismatch the user surfaced near the end of
this session is worth a dedicated callout in §03 + §08:

- AloePri's paper threat model: attacker has wire I/O + θ̃ + θ; the
  secret to protect is τ. All §F.1 attacks reduce to τ-recovery.
- Path-2's deployment: the wire-side threat is the same, but the
  activation-side attacker (with τ known) is also relevant if the
  honest-but-curious server has runtime memory access. Layer-0
  activation surface defends differently from the deep-layer surface
  the paper isolates.

**What to do.** Add a short section in §08 (probably a callout
between the in-scope and weight-inversion tables) titled "What
'covariant obfuscation' covers and doesn't" — restate that all
paper attacks are τ-recovery routes, define the additional
threat-model assumptions path-2 brings (activation access + known
τ), and link to the threat-model doc.

Threads 1 and 5 are coupled — best to do them together as a single
doc-correction pass.

## Suggested ordering

1. **Thread 1 (½ day)** — port the four paper prompt-inversion-via-weights
   attacks, measure on the current GGUF, update §08 framing. Cheap,
   high-leverage, settles whether the paper-claimed defence holds at
   path-2's scale (Qwen3-1.7B with current keymat/noise defaults).
2. **Thread 5 (parallel-safe with 1)** — rewrite the "out of scope"
   framing in §08; add the covariant-obfuscation callout. Doc-only.
3. **Thread 3 (½ day)** — measure the eigendecomposition leak
   end-to-end; add the empirical number to the §08 row.
4. **Thread 2 (1-2 days)** — α_e / h sweep. Measures the
   accessible-by-tuning improvements before committing to anything
   more invasive.
5. **Thread 4 (2-3 weeks)** — GELO-port + Gram-leakage measurement.
   Largest investment; only worth doing if threads 2-3 confirm static
   obfuscation can't reach the desired privacy gate at acceptable
   generation quality.

## Pre-flight notes for the next session

- **Branch state**: `path-2-aloepri-gemma`, uncommitted matrix-Γ +
  alg2.py + obfuscator + attack-driver changes. Consider committing
  before further work so the bench is reproducible.
- **Docker image**: `aloepri-llama-server:option-c-cli` (patched
  llama.cpp with matrix-Γ branch + M2.7 tensor dump + llama-cli +
  llama-server + llama-completion). All three GGUF variants land at
  `~/.cache/huggingface/path-2-aloepri/qwen3-1.7b/`.
- **Memory budget**: confirmed 48 GB available on this host. Static
  attacks need ~25 GB; weight-inversion harness fits comfortably.
- **§05 baseline numbers**: reproduced exactly under
  `--split-mode vocab`. The "stale handoff" diagnosis earlier in the
  session was wrong — see addendum at top of
  `2026-05-19-m2-7-attack-findings.md`.

## Suggested skills

- `/diagnose` if any of threads 1-3 produce unexpected numbers
  (especially thread 1 — if weight-IMA fails on path-2 but passes on
  paper's Qwen2.5-14B, the gap is informative).
- `/grill-with-docs` against the threat-model doc when starting
  thread 4 — GELO terminology + threat model needs to be precisely
  carried over.
- `/init` skill **not** needed; `CLAUDE.md` and codebase docs are
  current.

## Open questions left for the next session

- **Did the paper's IMA basic / IMA paper-like really test what I
  inferred?** The reference impl loads `model.embed_tokens.weight`;
  the paper text may describe a different attack that the reference
  implements differently. A 30-minute reading pass through the AloePri
  paper (PDF lives at `vendor/aloepri-py/docs/`) would settle this
  before porting them as "prompt inversion via weights".
- **Is M_q · M_k<sup>T</sup> = I really invariant under non-trivial
  Ẑ_block?** The numerical test at
  `python/aloepri-llm/scripts/check_alg2_invariance.py` shows fp32 noise
  for M_q · M_k<sup>T</sup> = I but ~32 % attention-score drift after
  RoPE — generation tolerates it but it's a known quality cost. Worth
  documenting the perplexity hit before scaling β beyond 8.
- **Where does the path-2 dynamic-mask handshake live?** GELO uses
  the `InProcessTrustedExecutor` boundary; path-2 doesn't have an
  equivalent. Likely requires a thin client handshake over the
  HTTP/completion endpoint — sketch the protocol before measuring
  Gram leakage.

## Artifacts produced in this session (uncommitted, ready to commit)

**Code (path-2 Python + attack harness):**
- `python/aloepri-llm/obfuscate_qwen3_gguf.py` — `--alg2-qk-norm-matrix`,
  `--alg2-h-hadamard-signs`.
- `python/aloepri-llm/lib/alg2.py` — `generate_block_perm` rewrite,
  `generate_h_qk` Hadamard mode, M_k construction repair.
- `python/aloepri-llm/scripts/measure_gamma_qk_clusters.py` — Option B
  pre-flight (used + done).
- `python/aloepri-llm/scripts/check_alg2_invariance.py` — algebraic
  validation script (now at realistic params).
- `python/aloepri-llm/scripts/build_matrix_gamma_identity_gguf.py` —
  identity-Γ smoke pair builder.
- `evals/aloepri-attacks/attack_drivers/common.py` — 3-D snapshot
  flattening in `stack_prompt_observations`.
- `evals/aloepri-attacks/m2_7/vulkan-m2_7-cli.Dockerfile` — patched
  docker image (cli + server + completion).

**Code (vendor/llama.cpp submodule, uncommitted):**
- `src/models/qwen3.cpp` — matrix-Γ branch on
  `hparams.aloepri_qk_norm_matrix`.
- `src/llama-hparams.h`, `src/llama-arch.h`, `src/llama-arch.cpp` —
  new `LLM_KV_ALOEPRI_QK_NORM_MATRIX` metadata key.

**Docs (committed-shaped, paths above):**
- `docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md` —
  eigendecomposition leak + defence options.
- `docs/handoffs/2026-05-19-alg2-qwen3-shape-analysis.md` (updated
  §4a with Option B verdict).
- `docs/handoffs/2026-05-19-alg2-z-block-degeneracy.md` — task #9
  finding.
- `docs/handoffs/2026-05-19-option-c-m2-7-rerun-findings.md` —
  Option C re-measure ledger.
- `docs/handoffs/2026-05-19-option-c-steps-0-1-2a-findings.md` —
  full step 0-1-2a ledger.
- `docs/handoffs/2026-05-19-m2-7-attack-findings.md` (updated with
  post-ramp addendum at top).
- `docs/prototype/aloepri-llm.html` — §08 rewrite, §03/§09 updates,
  collapsed weight-inversion section, eigendecomposition row.

**GGUFs (regenerable; on disk):**
- `keymat-h128-pi-noise-alg2-FULL-fp32.gguf` — R̂_qk only.
- `keymat-h128-pi-noise-alg2-FULL-zfix-fp32.gguf` — R̂ + Ẑ.
- `keymat-h128-pi-noise-alg2-FULL-zfix-hadamard-fp32.gguf` — full
  (current target).

**Captures + results (committed-shape):**
- `evals/aloepri-attacks/snapshots/m2_7-{section05,FULL,FULL-zfix,FULL-hadamard}-hidden-qk/`
- `evals/aloepri-attacks/results/m2_7-*.json` (per condition × surface)

## Boundaries respected throughout this session

- Did not push any submodule changes upstream.
- Ephemeral docker containers spawned per-condition, torn down after.
- `llama-swap` container untouched.
- Asked before every smoke / capture / attack-run step.
- Smoke GGUFs at `/tmp/qwen3-fp32-*` flagged for cleanup (still on
  disk; safe to `rm /tmp/qwen3-fp32-*.gguf` between sessions).
