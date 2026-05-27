# Option C steps 0 / 1 / 2a — full ledger and diagnosis

**Date:** 2026-05-19
**Status:** all three steps landed; one §6.3 gate now passes (ISA HS) on
all surfaces. IMA basic still fails by ~6× on all measurable surfaces
because the intra-head M_q does not reach the layer-0 capture site.

## TL;DR

After the matrix-Γ kernel (this session's Option C) plus the alg2.py
repairs (Ẑ_block + M_k construction + ±1 Walsh-Hadamard Ĥ_qk),
**Ẑ_block is the only step that materially moves an attack metric**:
ISA HS at attn_norm-23 drops 16.3 % → 11.5 % and clears the gate.
Adding ±1 Hadamard Ĥ_qk on top changes nothing measurable.

Layer-0 IMA stays at 88.9 % regardless of intra-head fixes — that
surface is **structurally unreachable** by any M_q transformation
(M_q acts on head_dim post-W_q; the IMA observation site is pre-W_q
residual stream).

## Full ledger

`--split-mode vocab` everywhere (paper-faithful, matches the §05
ledger frozen in `2026-05-19-m2-7-attack-findings.md`).

| Attack | Surface | §05 (frozen) | + R̂_qk (Option C) | + Ẑ_block fix | + Ĥ_qk ±1 | Gate |
|---|---|---:|---:|---:|---:|---|
| NN | * | 0.0 % | 0.0 % | 0.0 % | 0.0 % | ≤15% ✓ |
| IMA basic | attn_norm-0 | **88.9 %** | 88.9 % | 88.9 % | 88.9 % | ≤15% ✗ |
| IMA basic | Qcur_normed-0 | 88.9 % | 88.0 % | 88.0 % | 88.0 % | ≤15% ✗ |
| IMA basic | Kcur_normed-0 | 76.4 % | 76.9 % | 76.9 % | 76.9 % | ≤15% ✗ |
| ISA HS | attn_norm-23 | **16.3 %** | 16.3 % | **11.5 %** | 11.5 % | ≤15% ✓ |
| ISA HS | Qcur_normed-23 | 12.5 % | 9.6 % | 13.5 % | 13.5 % | ≤15% ✓ |
| ISA HS | Kcur_normed-23 | 8.7 % | 7.7 % | 7.7 % | 7.7 % | ≤15% ✓ |
| ISA AS | kq-23 | 0.0 % | 0.0 % | not re-run | not re-run | ≤15% ✓ |
| TFMA | token stream | 1.6 / 5.1 % | 0.8 / 2.7 % | not re-run | not re-run | ≤15% ✓ |
| SDA | token stream | 1.6e-5 | 1.4e-5 | not re-run | not re-run | ≤5 BLEU ✓ |

Bolded numbers are §6.3-gate verdicts that changed in this session.

## What each step did and why

### Step 0 — add Qcur_normed / Kcur_normed to M2.7 tensor-dump filter

**Why it was needed.** The original M2.7 filter only captured
`attn_norm-{0, 11, 23}` and `kq-23` — i.e., residual stream and final
attention scores. M_q lives between W_q and RoPE, on head_dim, *inside*
the attention block. The intra-head obfuscation never showed up in any
captured surface.

**What landed.** Extended the filter to also dump `Qcur_normed-{0, 11, 23}`
and `Kcur_normed-{0, 11, 23}` (callbacks already named in
`qwen3.cpp:112,134`). Added 3-D-snapshot flattening in
`attack_drivers/common.py:stack_prompt_observations` so the ridge can
consume `(n_tokens, n_head, d_h)` tensors flattened to
`(n_tokens, n_head·d_h)`.

**What it revealed.** Re-running IMA / ISA on §05 captures across all
three kinds confirmed §05 was reproducible at the new surfaces.
Captured both §05 and Option-C-R̂_qk-only states for the four reruns
that followed.

### Step 1 — repair Ẑ_block

**Diagnosis (filed 2026-05-19 morning):** `generate_block_perm` with
default params (β=8, γ=1e3, rope_base=1e6) produces the identity
permutation every time — softmax window-selection collapses to
size-1 windows. Documented in
`docs/handoffs/2026-05-19-alg2-z-block-degeneracy.md`.

**Fix.** Replaced the dynamic-window softmax with fixed β-wide windows
shuffled internally. Each consecutive group of β RoPE pairs is
randomly permuted; cross-window permutation is forbidden. Preserves
RoPE-locality enough that attention drift is bounded (~32 % relative
on numerical test, but model still generates coherently).

**Bonus algebra repair.** The original M_k construction
`R · H⁻¹ · Z⁻¹` only collapses `M_q · M_kᵀ` to I when Z² = I — which
the identity-Z degeneracy silently provided. With non-trivial Z this
breaks. Changed to `M_k = R · H⁻¹ · Z` (same Z, no inverse); now
`M_q · M_kᵀ = R · H · (Z·Zᵀ) · H⁻¹ · Rᵀ = I` for any orthogonal Z,
verified numerically.

**Result.** ISA HS attn_norm-23 dropped 16.3 % → 11.5 % (passes
gate). Q/K post-norm ISA HS already passed at §05; small ripple on
those surfaces. IMA unchanged on all surfaces.

**Generation.** Smoke prompt still coherent: "Paris … Málaga …
Washington" with one hallucinated locale ("Espin"). Attention drift
is within the model's tolerance band.

### Step 2a — Ĥ_qk as ±1 Walsh-Hadamard

**Rationale.** With Ĥ_qk = I (current MVP), M_q is rotation +
permutation only. ±1 Hadamard adds per-pair sign flips while keeping
M_q orthogonal (`H·H = I` for ±1 diag).

**Implementation.** Added `hadamard_signs=True` mode to
`generate_h_qk` and wired through the obfuscator as
`--alg2-h-hadamard-signs`. Constructed `H = diag(±1)` with the
paired-halves layout the existing code expects (same sign in both
NEOX halves of each RoPE pair).

**Result.** Numerically identical to the Z-fix-only ledger. The
ridge attacker is trivially insensitive to ±1 sign flips on a feature
axis — the ridge fit simply absorbs the sign pattern. Algebraic
dead-end as an obfuscation primitive against the ridge attack class.

**Algebraic note.** With ±1 H, `M_q = R · H · Z` and `M_k = R · H⁻¹ · Z`
(but `H⁻¹ = H` since H is involutive), so `M_q = M_k`. No Q/K
diversity. To get diversity we'd need non-unit H, which makes M_q
non-orthogonal, which requires runtime κ correction in the kernel
(handoff Option C variant, ~1 week kernel work).

## Why IMA basic still fails

IMA basic observes layer-0 hidden state, which in our pipeline is:

```
attn_norm-0 = RMS-norm(token_embd · token_id)
```

This is computed **before** any attention block runs — therefore
**before** any W_q · M_q product or RoPE. The intra-head M_q has no
effect at this layer. Capturing IMA at `Qcur_normed-0` (where M_q
*does* act) lowers TTRSR by ~1 pp (88.9 → 88.0); still 6× over the
15 % gate.

The 88-89 % IMA TTRSR on Qwen3-1.7B-§05 reflects that:

1. The keymat decoy expansion adds 256 dims (2048 → 2304) of
   structurally linear content. A ridge regressor with vocab-disjoint
   splits learns to strip these decoy dims and recover the original
   2048-dim embedding linearly.
2. The §5.2.2 α-noise on the embedding table is too weak (`α_e = 1.0
   × σ(W_e)` is a modest perturbation that doesn't move ridge top1
   meaningfully — the ridge fits the average over noise instances).
3. The Π token permutation acts on the vocab axis (input/output), not
   on hidden-state rows. A ridge inverter that finds a left-inverse
   recovers the obfuscated embedding; then the client decodes via τ⁻¹.
   But the ridge doesn't need to apply τ⁻¹ — the embedding it
   recovers is already in the obfuscated id space, and the IMA TTRSR
   counts top-1 matches in that same space. So Π gives no defence
   against IMA at all (it defends against TFMA / SDA which observe
   the output stream and need to align frequencies to plaintext).

**To move IMA basic below 15 %, levers outside Algorithm 2:**

- Larger keymat expansion (`h = 256` or higher). More decoy dims to
  blow up ridge regularisation requirement.
- Heavier α-noise on the embedding (`α_e = 5–10` instead of 1.0).
  Trade-off: degrades coherent generation.
- Non-linear keymat fold (`P̂_R` with non-linear post-processing).
  Paper-divergent; new research.
- Move the IMA observation surface so it's not L0 — but this is
  attacker's choice, not defender's.

None of these are Algorithm 2's job, which is why steps 0 / 1 / 2a
move the dial on ISA but not IMA.

## What changed in artifacts

**Code (uncommitted):**

- `python/aloepri-llm/lib/alg2.py` — `generate_block_perm` rewritten as
  β-wide-window permutation; `generate_h_qk` extended with
  `hadamard_signs=True` mode; `build_layer_keys` takes
  `h_hadamard_signs` flag and uses repaired M_k construction.
- `python/aloepri-llm/obfuscate_qwen3_gguf.py` — `--alg2-h-hadamard-signs`
  CLI flag.
- `python/aloepri-llm/scripts/check_alg2_invariance.py` — bumped to realistic
  params (head_dim=128, β=8).
- `evals/aloepri-attacks/attack_drivers/common.py` — `stack_prompt_observations`
  flattens 3-D tensors to 2-D so Q/K post-norm snapshots feed the
  existing ridge attackers.

**GGUFs (locally, regenerable):**

- `keymat-h128-pi-noise-alg2-FULL-fp32.gguf` — R̂_qk only.
- `keymat-h128-pi-noise-alg2-FULL-zfix-fp32.gguf` — R̂_qk + Ẑ_block.
- `keymat-h128-pi-noise-alg2-FULL-zfix-hadamard-fp32.gguf` — full
  R̂·Ĥ·Ẑ.

**Captures:**

- `evals/aloepri-attacks/snapshots/m2_7-section05-hidden-qk/`
- `evals/aloepri-attacks/snapshots/m2_7-FULL-hidden-qk/`
- `evals/aloepri-attacks/snapshots/m2_7-FULL-zfix-hidden-qk/`
- `evals/aloepri-attacks/snapshots/m2_7-FULL-hadamard-hidden-qk/`

**Results JSONs:**

- `m2_7-FULL-{hidden,token,qcur,kcur}.json`
- `m2_7-FULL-zfix-{attn_norm,qcur,kcur}.json`
- `m2_7-FULL-hadamard-{attn_norm,qcur,kcur}.json`

## Suggested next steps (priority order)

1. **Refresh the M2.7 ledger** in
   `docs/handoffs/2026-05-19-m2-7-attack-findings.md` with the
   step 0-1-2a numbers and the Q/K post-norm surfaces. The
   existing handoff predates this work.
2. **Update `docs/prototype/aloepri-llm.html` §08** acceptance gates:
   ISA HS now passes; IMA basic still fails; document why.
3. **(Open)** Decide whether to invest in IMA-specific defences
   (keymat / noise / non-linear fold) or accept that L0 IMA is
   the path-2 deployment's known weakness and document the threat
   model boundary accordingly. This is a strategic call.
4. **Commit the alg2.py + obfuscator + attack-driver changes.**
   They're production-quality; the GGUF naming convention with
   `-zfix` and `-hadamard` suffixes can be retired once we settle
   on a single configuration.

## Boundaries respected

- Did not push any submodule changes upstream.
- Ephemeral docker containers spawned + torn down per condition.
- Asked before every capture / attack run.
- Repaired silently-broken alg2.py M_k construction (Z² requirement)
  in passing; documented inline + in handoff.
