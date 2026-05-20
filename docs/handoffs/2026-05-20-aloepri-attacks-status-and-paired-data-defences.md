# Handoff — AloePri attacks: measured status + paired-data attacker is the load-bearing limit

**Date:** 2026-05-20
**Branch:** `path-2-aloepri-gemma` (clean, merged with master at commit
`4c7aff5`).
**Supersedes:** `2026-05-19-aloepri-attack-surface-followups.md` —
threads 1-3 of that doc are resolved; threads 4-5 evolved into the
"paired-data attacker" framing below.
**Status:** Full attack re-eval landed on two ae0.1 cells (no Alg 2,
+ Alg 2 + matrix-Γ). Three gates remain open. Static-obfuscation
tuning has hit the structural ceiling at d=2048; further progress
requires either a higher-d model or a different threat model.

## What landed since 2026-05-19

Five commits on this branch, all in `path-2-aloepri-gemma`:

| Commit | Subject |
|---|---|
| `51d932b` | Π special-token fix — recovers HumanEval pass@1 from −45pp to −15pp |
| `abcef5a` | AloePri client — stream + ignore_eos + bytes-mode to dodge llama-server PEG chat-parser 500s |
| `f11e525` | Alg 2 + matrix-Γ + α_e=0.1 cell — end-to-end working, paper-sized accuracy cost |
| `e297c5c` | Fill in attack figures for ae0.1 no-Alg-2 cell |
| `ce0b661` | Full attack re-eval — hidden-state + token-stream attacks across both ae0.1 cells |
| `09a1049` | Doc: aloepri-llm.html §06 + §09 — algorithm-spec callouts + dimensional headroom note |
| `61b9ac2` | Doc: FIG. 03b SVG updated to reflect full Alg 2 + matrix-Γ deployment |
| `b9d4f04` | Doc: FIG. 03a — Stage 1 (setup) added above Stage 2 (request) |

Two bugs found and closed:
1. **Π was permuting EOS / chat-template special tokens** → server's
   stop-on-EOS never fired, generation drifted to max_tokens, model
   degenerated. Fix: read `tokenizer.ggml.token_type` and keep
   non-NORMAL/BYTE tokens at identity (26 special tokens for
   Qwen3-1.7B).
2. **llama-server PEG chat-parser** (`common/chat.cpp:2536`) ran on
   the aggregated detokenised text at end of every `/completion`
   and threw 500 on multi-language gibberish (inevitable with Π
   plus the unpermuted server tokenizer string table). Fix in the
   AloePri client: `stream: True` + `iter_lines(decode_unicode=False)`
   + per-line bytes decode with `errors='replace'` + `ignore_eos: True`
   for fixed-budget eval semantics.

## Current attack ledger (ae0.1 cells, n=20 HumanEval, gate < 15 %)

| Attack | Surface | no Alg 2 | + Alg 2 + matrix-Γ | Status |
|---|---|---|---|---|
| VMA top-1 | static W tensors sort-and-match | 0.00 % | 0.00 % | ✓ pass |
| IA top-1 | invariants on (W_e, W_gate/Q/K) | 0.0000 % | 0.0244 % | ✓ pass |
| **IMA-EmbedRow-ridge top-1** | static W_e row pairs | **99.22 %** | **99.22 %** | **✗ structural** |
| NN @ attn_norm-0 | layer-0 residual | 0.00 % | 0.00 % | ✓ pass |
| **IMA-L0-activation @ attn_norm-0** | pre-W_q residual | **22.58 %** | **22.58 %** | **✗ pre-W_q, Alg 2 can't reach** |
| ISA HiddenState @ attn_norm-23 | layer-23 residual | 9.68 % | 9.68 % | ✓ pass |
| **IMA @ Qcur_normed-0** | post-W_q + Q-norm | **35.48 %** | **38.71 %** | **✗ +3pp from Alg 2** |
| ISA @ Kcur_normed-0 | post-W_k + K-norm | 35.48 % | 35.48 % | ✗ |
| TFMA top-1 (top-10) | wire token freq | 0.0 % (5.08 %) | 0.0 % (4.69 %) | ✓ pass |
| SDA BLEU-4 | wire bigrams | 1.5e-5 | 1.4e-5 | ✓ pass |
| HumanEval pass@1 (n=20) | accuracy | 8/20 = 40 % | 7/20 = 35 % | paper-sized −5pp Alg 2 cost |
| Server-500s | wire reliability | 0/20 | 0/20 | ✓ pass |

Full per-cell artifacts under
`evals/aloepri-attacks/results/sweep/cell-ae0.1-ah0.033-final/` and
`.../cell-ae0.1-alg2/`. The harness used:
- `m2_7_static.json` — VMA + IA
- `ima-embedrow.json` — IMA-EmbedRow-ridge
- `quality-humaneval.json` — HumanEval pass@1 + 5-prompt quality probe
- `hidden-attacks.json` — NN + IMA-L0 + ISA on `attn_norm-{0,11,23}`
- `hidden-attacks-qk.json` — IMA + ISA on `Qcur_normed-0` / `Kcur_normed-0`
- `token-streams.jsonl` + `token-attacks.json` — TFMA + SDA

## Why the three failing gates fail (one root cause, three surfaces)

All three failing gates have the same root cause:

**Static linear obfuscation cannot defeat a paired-data ridge attacker.**

The AloePri threat model (paper §3.2) explicitly grants the attacker:
- Plaintext model weights `W` (Qwen3 is open-source, downloaded from HF)
- Obfuscated model weights `W̃` (the server hosts them, attacker IS the
  honest-but-curious server)
- Wire-side I/O
- Knowledge of the obfuscation algorithm (Kerckhoffs's principle)

What the attacker does NOT have: τ, the keymat P̂/Q̂/M_q/Û_vo, and the
noise seeds.

Given paired weights, the attacker constructs (W[i], W̃[τ[i]]) pairs
across the vocab and fits a single linear map V via closed-form ridge:
`V · W̃[τ[i]] ≈ W[i]`. Once V is fit, applying V to any obfuscated
row recovers approximately the original token. The only thing standing
between V and τ-recovery is the per-row noise α_e·σ_e·ε and the
conditioning of P̂ — both of which fail at d=2048 / α_e=0.1.

This breaks down per-surface:

1. **IMA-EmbedRow-ridge (99.22 %)** — most direct case. Attack target is
   the raw W_e tensor; no model run needed. Algorithm 2 by construction
   doesn't touch W_e, so its presence or absence doesn't matter.

2. **IMA-L0-activation @ attn_norm-0 (22.58 %)** — pre-W_q residual is
   just RMSNorm(W̃_e[obf_id]). Same bijection as IMA-EmbedRow-ridge but
   normalised. Algorithm 2 lives post-W_q so it structurally cannot
   reach this surface.

3. **IMA @ Qcur_normed-0 (35-39 %)** — post-W_q + Q-norm. This is where
   Alg 2 should win, but matrix-Γ's design *deliberately* makes
   `(Q̃ / RMS(Q̃)) · Γ_q = Q_plain_normed · M_q` for orthogonal M_q.
   The attacker observes Q_plain_normed rotated by a static orthogonal
   M_q — and a paired-data ridge inverter learns M_q^T in closed form.
   Alg 2 actually makes this gate marginally worse (+3pp) because the
   noise that was helping in the no-Alg-2 baseline gets smoothed
   through M_q's orthogonality.

## Why paper's IMA = 0 % on Qwen2.5-14B but ours = 99.22 % on Qwen3-1.7B

Dimensional headroom argument, captured in §09 of `aloepri-llm.html`:

- Paper config: d=5120, α_e=1.0 → per-row noise norm `1.0·σ·√5120 ≈ 71.5σ`
- Path-2 config: d=2048, α_e=0.1 → per-row noise norm `0.1·σ·√2048 ≈ 4.5σ`
- Ratio: ~16× weaker per row, in 2.5× fewer dimensions → ~25× weaker
  effective scatter in the attack-utility subspace.

We cannot raise α_e to paper's 1.0 at d=2048: at the FULL-zfix-hadamard
baseline (broken Π era, but the noise effect persists) HumanEval pass@1
went to 0 % with multi-language gibberish on hard prompts.
α_e=0.1 is the accuracy ceiling we found.

The static linear obfuscation envelope has been characterised. Further
progress requires either (a) more dimensional headroom (bigger model),
or (b) a different threat model (paired-data attacker excluded or
neutralised by design).

## Next steps

### Direction 1 — test higher-d models with paper-default obfuscation

**Hypothesis:** at d ≥ 4096, paper's α_e=1.0 should be back in scope
without breaking accuracy, and the IMA-EmbedRow / IMA-L0 gates should
land in paper-claimed range. Confirms the dimensional argument
empirically and gives path-2 a deployable demonstrator at a model
size that actually clears all gates.

**Concrete experiments:**

| Cell | Model | d | α_e | α_h | h | λ | Alg 2 | Expected ridge | Expected HE |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-4B | 2560 | 1.0 | 0.2 | 128 | 0.3 | matrix-Γ + Hadamard | ~80-90 % (still ✗) | ~40-50 % |
| 2 | Qwen3-4B | 2560 | 1.0 | 0.2 | 256 | 0.3 | matrix-Γ + Hadamard | ~60-80 % | ~35-45 % |
| 3 | Qwen3-4B | 2560 | 1.0 | 0.2 | 512 | 0.3 | matrix-Γ + Hadamard | ~40-65 % | ~30-40 % |
| 4 | Qwen3-8B | 4096 | 1.0 | 0.2 | 128 | 0.3 | matrix-Γ + Hadamard | **~10-20 % (likely ✓)** | ~55-65 % |
| 5 | Qwen3-8B | 4096 | 1.0 | 0.2 | 256 | 0.3 | matrix-Γ + Hadamard | **< 10 % (✓)** | ~50-60 % |

Cell 4 is the smallest model where the structural-attack gate is
plausible to close at paper defaults. Cell 5 is the cleanest
publishable result.

**Pre-work needed:**
- Plain Qwen3-4B and Qwen3-8B GGUFs (bartowski has both; download +
  hash-check).
- The patched llama.cpp image (`aloepri-llama-server:option-c`) should
  work as-is — the matrix-Γ kernel branch keys on
  `aloepri.qk_norm_matrix` metadata and the rest is upstream Qwen3
  graph. Sanity-check by spawning a 4B plain server and verifying
  /health.
- Disk: Qwen3-4B fp32 obfuscated ~16 GB; 8B ~32 GB. ~64 GB needed if
  we want both on disk at once. Current free space (per recent
  cleanup): 1.3 TB → plenty of room.
- Memory: 4B forward pass should fit comfortably on the 96 GB iGPU;
  8B at fp32 needs careful unload of any other resident model.
- Time: ~10 min per cell for build + ~20 min for full attack suite.

**Order recommendation:** Cell 1 first (cheapest signal on the
dimensional argument; partial confirmation if it lands at ~85 %).
Cell 4 second (the headline result). Cells 2/3/5 only if needed for
the sweep coverage.

### Direction 2 — design defences against the paired-data attacker

If we want a privacy-preserving result at Qwen3-1.7B specifically (or
at any d where paper-default α_e+λ+h doesn't reach the gate), the
threat model has to change. Three structural options, in increasing
order of engineering cost:

#### 2A. Hide the obfuscated model

**Idea:** make the obfuscated GGUF non-public. The attacker still has
plain `W` (HF download) but not `W̃` — so the paired-data assumption
breaks. Activation-side attacks (IMA-L0 on captured activations)
still require server-side access; if the server runs in a TEE or
otherwise isolates its memory, the attack collapses to wire-side only.

**Pros:** zero changes to the obfuscation algorithm, zero changes to
inference. Just access control.
**Cons:** breaks paper's "compatible with public LMaaS infrastructure"
thesis. Requires trust in the operator who holds `W̃` and key custody.
**Status:** known design lever, never explicitly considered in path-2
docs. Probably the right move if "no AI on this codebase" is too
strong a constraint to lift cleanly.

#### 2B. Obfuscate the model inside a TEE

**Idea:** keep `W̃` public, but run obfuscation generation inside a
TEE (SEV-SNP CVM or similar) so the keymat / τ stay inside the
attestation perimeter. Attacker on the server side has `W̃` and the
forward graph, but cannot extract `W_plain` because the obfuscation
process consumed it inside the TEE before producing `W̃`.

This still doesn't help against a paired-data ridge attacker who has
the public `W_plain` from HF — they reconstruct the pair themselves.
**So 2B alone is insufficient.** It only helps if combined with 2A
(hide `W̃`) or with a model that's NOT publicly available.

**Pros:** matches the path-1 GELO threat model. Clean composition with
the existing attestation infrastructure.
**Cons:** doesn't actually move the paired-data gate unless `W_plain`
is also unavailable. Re-read the threat model carefully — useful as
a *building block* but not a standalone fix.

#### 2C. Dynamic masking (GELO-style fresh keymat per request)

**Idea:** the structural fix. Replace the static `W̃ = Q̂·W·P̂` and
`Γ = M^T·Diag(γ)·M` with per-prompt fresh masks. Attacker sees a
different W̃ on every forward pass; cross-batch pair collection
cannot accumulate statistical leverage; ICA-style attacks lose the
fixed-mixing assumption.

This is the **direct generalisation of what GELO does** in the path-1
embedder. Memory references: `[gelo_research_round_2]`,
`[private_llm_inference_round_3]`, `[hd3_mask_landed]`. The path-1
implementation lives under `crates/gelo-*/`.

**Architectural questions to resolve:**

a. **Where does the fresh mask come from?** Two candidates:
   - **Client-resident generator**: client samples a per-prompt
     `M_q^{(t)}` and ships it to the server alongside the prompt.
     Server applies it to W̃ before the forward pass. Breaks the
     "stateless server" thesis but matches GELO's deployment.
   - **TEE-resident generator**: server-side TEE samples the mask
     each forward pass. Path-2 doesn't currently have a TEE
     primitive; would need to either embed one in llama-server or
     pre-stage a stream of masks signed by an attested generator.

b. **What gets masked?** Just W_q,k or also W_e? The embed mask is
   what closes IMA-EmbedRow / IMA-L0; the W_q,k mask is what closes
   IMA@Qcur. If only one of the two, the other gate stays open.

c. **Inference cost?** GELO benchmarks (memory `[hd3_mask_landed]`,
   `[blis_default_on_and_layer_skip_regression]`) report −28 % TTFT
   at pow2-aligned `n+k` with HD₃ Hadamard cascade. Cost on the
   AloePri side will differ because the mask applies to weights not
   activations.

d. **Compatibility with stock llama.cpp?** Almost certainly no.
   Per-prompt mask refresh requires either rewriting W̃ in-place
   per request (expensive) or a fused kernel that applies the
   mask on the fly. The matrix-Γ kernel patch is a precedent —
   adding a similar branch for "mask before W_q/W_k" is the right
   shape.

**Pros:** structurally addresses the paired-data attacker. Matches the
research direction the project is already pursuing (GELO).
**Cons:** by far the most engineering. Probably 4-8 weeks of spike
work. Breaks the "no infrastructure change" thesis but matches the
project's actual research goal.

**Gram-leakage measurement (deferred from 2026-05-19 thread 4).**
The path-1 round-2/round-3 research already identified Gram-matrix
attacks as the residual surface under fresh masking. The four new
attack drivers under `evals/aloepri-attacks/attack_drivers/`
(`run_anchor_ica.py`, `run_jade.py`, `run_jd.py`, `run_gram_error.py`)
are exactly the toolkit to validate any dynamic-masking design. Should
run them against (i) the current static path-2 deployment, (ii) any
prototyped dynamic-mask variant — comparative numbers settle the
design.

## What's now done from the 2026-05-19 thread list

| 2026-05-19 thread | Status |
|---|---|
| 1 — re-frame VMA/IA/paper-IMA as prompt-inversion via static weights | **done**. All four measured. VMA + IA pass. IMA-EmbedRow-ridge confirmed at 99.22 %. §08 + §09 of `aloepri-llm.html` updated. |
| 2 — sweep (α_e, h) to close IMA-L0-activation | **partly done**. We learned α_e>0.3 destroys accuracy at d=2048; α_e=0.1 became the operational default. h sweep not run; deferred to direction 1 (do it at d=4B/8B instead). |
| 3 — measure QK-norm Γ eigendecomposition leak | **deferred**. The matrix-Γ design ships `Γ = M^T·Diag(γ)·M` which is a similarity transform — `numpy.linalg.eig` recovers M directly. Threat-model doc covers this. Empirical measurement against the actual GGUF still TBD but the analytical attack is unambiguous. |
| 4 — port GELO-like dynamic defences + Gram leakage | **carried forward** into direction 2C. |
| 5 — resolve paper-vs-path-2 surface-mismatch in public docs | **done**. §08 attack table reorganised; §09 dimensional-headroom note added; FIG. 03a/03b updated. |

## Suggested ordering for the next session

1. **Direction 1 cell 1 + 4** (4-6 hours) — bring up Qwen3-4B and 8B
   plain GGUFs, build the +Alg2 cells at paper defaults, run the full
   attack suite. This is the highest-information experiment available
   and confirms or refutes the dimensional argument.
2. **If cells 1+4 confirm the argument** — write up "path-2 demonstrator
   pivot from Qwen3-1.7B to Qwen3-{4,8}B" as the recommendation; update
   `aloepri-llm.html` §04 (supported models) and §09 (status).
3. **If cells 1+4 don't close the gate** — Direction 2 becomes
   load-bearing. Start with 2A (access-control analysis: who needs to
   see `W̃`, can it be private?) as the cheapest mitigation; if that
   doesn't fit the project's intent, scope 2C (dynamic masking) as a
   multi-week spike.
4. **Eigendecomposition leak measurement** (Thread 3 from 2026-05-19) —
   30 min, settles the only remaining unmeasured open question from
   the 2026-05-19 ledger. Should be done before either Direction 1 or
   Direction 2 lands so the threat-model doc is fully grounded.

## Pre-flight notes

- **Branch**: `path-2-aloepri-gemma`, clean, merged with master at
  `4c7aff5`. All changes from today committed (8 commits since the
  prior handoff).
- **GGUFs**: only the path-2-aloepri/qwen3-1.7b/ directory has the
  current cells. The ablation GGUFs (gamma-only, keymat-only, etc.)
  were cleaned up during the disk-recovery pass earlier today; they
  rebuild in ~40 s from
  `python/path-2/obfuscate_qwen3_gguf.py` if needed.
- **Disk**: 1.3 TB free post-cleanup. Room for both Qwen3-4B and
  Qwen3-8B obfuscated variants.
- **Docker image**: `aloepri-llama-server:option-c` (~710 MB). The
  m2_7 and option-c-cli tags were pruned earlier — `option-c` has
  matrix-Γ + tensor-dump support which is everything we need.
- **Memory hygiene rule**: see new global memory
  `[feedback-docker-build-disk-hygiene]` — when rebuilding the
  patched llama.cpp Dockerfile, always
  `docker builder prune -af` after the tagged image lands. The
  prior 185 GB build-cache fill came from forgetting this.
- **Attack harness**: full coverage now wired. Per-cell run pattern:
  ```
  # static (no server needed)
  run_static_attacks.py --plain $PLAIN --obfuscated $OBF  → VMA + IA
  run_ima_embedrow_attacks.py --plain $PLAIN --obfuscated $OBF --key $KEY → ridge
  # server-side (needs patched llama-server up)
  capture_hidden_states.py --tensor-filter '^(Qcur_normed|Kcur_normed|attn_norm)-(0|11|23)$' …
  run_hidden_state_attacks.py → NN + IMA-L0 + ISA
  capture_token_streams.py + run_token_attacks.py → TFMA + SDA
  run_quality_humaneval.py → HumanEval pass@1 + quality probe
  ```
  Full driver inventory under `evals/aloepri-attacks/m2_7/` and
  `evals/aloepri-attacks/attack_drivers/`.

## Suggested skills for the next session

- **`/diagnose`** if any Direction 1 cell produces unexpected numbers
  (e.g., Qwen3-4B at paper defaults still fails ridge — would mean
  the dimensional argument is incomplete and there's an additional
  factor we haven't isolated).
- **`/grill-with-docs`** when sketching the Direction 2C protocol —
  GELO terminology + threat model needs to carry over cleanly into
  AloePri's vocabulary.
- **No `/handoff`** at the end unless the session opens up another
  branch of work; the current state is well-captured.

## Open questions deferred

- **Eigendecomposition leak empirical number** (carried over from
  2026-05-19 thread 3). The Γ matrices are on disk in the current
  GGUF; `np.linalg.eig` on each layer's
  `blk.{i}.attn_q_norm.weight` should yield γ_q + M_q in ~seconds.
- **Per-prompt fresh-mask handshake design** for Direction 2C —
  needs a sketch before measuring Gram leakage. The path-1 GELO
  protocol lives at `crates/gelo-protocol/` but assumes an in-process
  TEE boundary; path-2's HTTP /completion endpoint is a different
  shape entirely.
- **Effective ridge sample-budget at d=4096/d=5120** — does the
  attacker still get clean closed-form recovery with the same
  vocab-train split (1024 train rows) at higher d? If the answer is
  yes, the dimensional argument is incomplete and α_e+λ+h must do
  more work than just "scatter in higher-d directions".

## Artifacts referenced

- This handoff: `docs/handoffs/2026-05-20-aloepri-attacks-status-and-paired-data-defences.md`
- Predecessor: `docs/handoffs/2026-05-19-aloepri-attack-surface-followups.md`
- Companion handoffs from today's session:
  - `docs/handoffs/2026-05-20-aloepri-pi-special-token-fix.md` —
    full diagnostic record for the Π fix
- Public docs: `docs/prototype/aloepri-llm.html` §03–§09 — all in
  sync with the current measurements
- Threat-model doc: `docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md`
- Per-cell attack JSONs: `evals/aloepri-attacks/results/sweep/cell-ae0.1-{ah0.033-final,alg2}/`
