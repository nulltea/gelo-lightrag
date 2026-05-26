# Handoff — two AloePri gaps awaiting a research pass

**Worktree:** `/home/timo/repos/private-rag-path-2/` · branch
`path-2-aloepri-gemma`. Tip:

```
a5003a9 merge master into path-2-aloepri-gemma · pulls in AloePri attack-harness handoff
5e54cc7 path-2: AloePri Qwen3 1.7B — Π token permutation + α_e/α_h noise + partial Algorithm 2
ee932ed path-2: handoff doc — three gates before deferred privacy work
```

The committed state is correct. Reference for the wider context:

- Plan: [`docs/plans/path-2-aloepri-gemma.md`](docs/plans/path-2-aloepri-gemma.md) · running status [`docs/plans/path-2-status.md`](docs/plans/path-2-status.md) · next-steps handoff [`docs/plans/path-2-aloepri-next-steps.md`](docs/plans/path-2-aloepri-next-steps.md)
- Protocol doc: [`docs/prototype/aloepri-llm.html`](docs/prototype/aloepri-llm.html) — §07 perf, §08 gaps
- Implementation: `python/aloepri-llm/{obfuscate_qwen3_gguf.py, aloepri_client.py, lib/alg2.py, evals/}`

This handoff covers **only the two open research questions** the user
flagged. The independent Phase-2/3 attack-harness work is tracked in
[`docs/plans/handoff-aloepri-attack-resistance.md`](docs/plans/handoff-aloepri-attack-resistance.md);
do not duplicate that work.

---

## (a) Obfuscated-artifact quantisation — current state

**Verdict landed: fp32 is required.** All three llama.cpp
production-quant formats break the obfuscation chain.

Numbers (from `docs/plans/path-2-status.md` "Gate A" entry):

| Format | Size | Smoke output on "What is the capital of France?" |
|---|---:|---|
| fp32 (reference) | 8.6 GB | coherent on-topic continuation |
| Q8_0 | 2.3 GB | `" ( ( ( ( ,chein,zech…"` — degenerate |
| Q6_K | 1.8 GB | server 500 — output failed re-tokenisation |
| Q5_K_M | 1.6 GB | `"phenhqymologyaverholes…"` — word salad |

**Mechanism.** AloePri-keymat weights are heavy-tailed per row (max ≈ 55,
std ≈ 4.7 on `blk.27.*` of Qwen3 1.7B). Q8_0 stores 32-element blocks
with one fp16 scale; within a heavy-tailed block the small values
round to zero, breaking the per-row covariant chain. K-quants (Q6_K,
Q5_K_M) have more flexible scaling but still can't hold the precision.

**What was tried in-session:** the three formats above via stock
`llama-quantize`. No knob-tuning beyond format choice.

### Next steps for (a)

**Goal:** find an obfuscation construction whose weights survive Q8_0
without losing the covariant property, so the artifact shrinks 4× and
decode-bandwidth penalty disappears.

1. **QR-project P̂_R onto the orthonormal manifold.** Paper Algorithm 1
   builds `P̂ = [B C E]·Z` where `B = U + λV` (U orthogonal, V Gaussian,
   λ=0.3). With `λ → 0` or QR(B) → orthonormal U, the per-row variance
   tightens dramatically. Cost: smaller obfuscation group at the
   keymat layer, which weakens IA defense; cleanest path is to measure
   the new per-row variance distribution before fully committing.
   Implementation hook: `python/aloepri-llm/obfuscate_qwen3_gguf.py:233-244`
   already loads keymat from `vendor/aloepri-py`; either add a
   `--qr-project` flag or fork the reference's `keymat.py` locally.
2. **Smaller h.** Default h=128 gives `d → d+2h = 2048 → 2304`. Trying
   h=64 gives 2176, h=32 gives 2112. Less internal expansion =
   smaller off-axis components = tighter per-row distribution. Cost:
   smaller AloePri obfuscation group on the residual stream. Cheap to
   test: re-run rewriter with `--expansion-size 64` then `32`,
   re-attempt Q8_0 quantise.
3. **Empirical knob sweep before tweaking math.** Measure per-row max
   and std of the obfuscated `blk.*.attn_q.weight` across {h, λ}
   ∈ {128, 64, 32} × {0.3, 0.1, 0.03}; plot vs Q8_0 within-block
   loss. May find a sweet spot that survives Q8_0 without giving up
   IA defense entirely.
4. **K-quant with calibration.** llama.cpp's K-quant variants accept
   an importance matrix (i-mat) via `llama-imatrix`. Generate i-mat
   from a calibration corpus run against the fp32 obfuscated artifact;
   K-quant should then preserve the high-magnitude rows. Untested in
   this session; could be the cheapest fix if it works.

**Skill suggestion:** `grill-me` before committing to any specific
combo — there's enough free variables (h, λ, format, i-mat) that a
stress-test of the chosen path is worth 30 minutes.

**Expected effort:** ~1 week to land a Q8_0-friendly construction
that also passes Gate B/C accuracy. Validation harness is reusable as-is.

---

## (b) Internal-state inversion / Algorithm 2 solution gap — current state

**Verdict landed: ship inter-head shuffle only on Qwen3.** R̂_qk /
Ĥ_qk / Ẑ_block / Û_vo — the intra-head transforms paper §5.2.3
specifies — are not deployed in the current implementation.

### What was tried (chronologically)

Each step was a fresh GGUF rewrite + container spawn + smoke test
with `"What is the capital of France?"`, max_tokens=24, seed=0.

| Configuration | Smoke output | Verdict |
|---|---|---|
| Full Algorithm 2: intra-head dense (R̂_qk · Ĥ_qk · Ẑ_block) + Ẑ_block dynamic_window + head-shuffle (τ_kv, τ_group) + §5.2.5 QK-norm fold | `" ..................... a again  ..."` | degenerate |
| Same minus Ẑ_block (`--alg2-beta 1` → window size 1 = identity perm) | same kind of degenerate | Ẑ_block not the bug |
| Head-shuffle only (no intra-head dense, no Ẑ_block) + QK-norm fold still applied | `" of ............     ... ... ..."` | degenerate — still bad |
| QK-norm fold ONLY (intra-head identity, head-shuffle identity) | `" of .................     Let.........................."` | **QK-norm fold is the bug** |
| Head-shuffle only, **without** QK-norm fold (γ_qk stays per-element; q_matrix = k_matrix = I; τ_kv ∈ S_8, τ_group ∈ S_2) | `"The capital of France is Paris.\n\nNo, that's not right…"` | **coherent — byte-identical to items 6+8 baseline** |

So the bisect landed cleanly: **§5.2.5 fusion of the QK-norm γ_qk
into W_q/W_k breaks the model.** Once the fold is removed, the
remaining transforms (head-shuffle) compose cleanly with the
unmodified QK-norm.

### Why §5.2.5 fold for QK-norm doesn't work

Paper §5.2.5 fuses a per-element `γ` into the adjacent linear weight
and replaces the norm site's γ with scalar `κ`. The construction is
exact in expectation when `κ = E[RMS(x·γ)/RMS(x)]` and `x` is i.i.d.
Gaussian. Under that assumption:

```
RMS(q · γ) / RMS(q) = sqrt(mean(γ²))      (i.i.d. Gaussian q)
```

so we set `κ_q = sqrt(mean(γ²))` and the fold cancels in expectation.

**Failure mechanism on Qwen3 QK-norm.** Trained Qwen3 Q/K vectors are
not i.i.d. Gaussian — the model puts more activation in the head-dim
indices where γ_q is large (the model has learned which dims matter
and γ has co-adapted). So `q² · γ²` is positively correlated, not
independent, and per-input:

```
RMS(q · γ) > κ_q · RMS(q)   when q and γ correlate
```

The fold then over- or under-scales the post-norm output by a
prompt-dependent factor. Inside softmax(Q·Kᵀ/√d_h), even small
per-input scale errors push attention toward uniformity (high-prior
token loops) or away from semantic peaks. The model degenerates to
high-frequency-token emission ("a again ... ... ...").

The same dynamic is the reason path-2-status.md flagged Gemma 4
post-norms as a blocker — 5 norm sites × multiplicative κ-bias
compounding pushes accuracy loss past ~10%. For QK-norm, even one
site breaks attention because softmax amplifies the error.

### Why the public reference didn't catch this

Paper §7.1 explicitly lists Qwen3 in evaluated models. Reference repo
`vendor/aloepri-py @ sheng1feng/Aloepri 60e8ea3` has zero Qwen3 path —
every attention module imports `transformers.models.qwen2.modeling_qwen2`.
Qwen2.5 has no QK-norm so the issue never surfaces in the public code.
ByteDance's internal industrial build presumably has a fix; it's not
in the academic release. (Filed as known gap in §08 of the protocol
HTML.)

### Current deployment

`python/aloepri-llm/lib/alg2.py:147-189` generates the full per-layer key
set (R̂_qk, Ĥ_qk, Ẑ_block, τ_kv, τ_group) — the math is there. The
rewriter (`obfuscate_qwen3_gguf.py:333-388`) only emits **identity**
q_matrix / k_matrix and applies the head-shuffle via
`apply_qkv_output_transform(out_arr, None, q_feat)` (dense_transform
forced to None). The intra-head dense path is wired but dormant — set
`q_matrix` and `k_matrix` back to the real `keys.q_matrix` /
`keys.k_matrix` and apply the QK-norm fold block (currently commented
out / replaced with the head-shuffle-only construction) to re-enable
once a workaround lands.

### Proposed next steps for (b)

Ordered by effort × likelihood. The first one is the highest-value
research path.

1. **Per-norm-site κ calibration** *(most paper-faithful, ~2 days)*.
   Replace the i.i.d. Gaussian `κ_q = √(mean(γ²))` with an
   empirically-measured `κ_site = E_{x∼corpus}[RMS(W_q·x ⊙ γ_q) /
   RMS(W_q·x)]`. Steps:
   - run plaintext Qwen3 1.7B on a ~256-prompt calibration corpus
     (the AloePri default-prompts set in `vendor/aloepri-py/src/defaults.py`
     is convenient and gives comparable numbers to the paper),
   - at each QK-norm site dump the pre-norm Q (and K) vectors,
   - compute the per-site κ_site as the empirical mean of the ratio,
   - bake into the rewriter: each site gets its own κ.
   This brings the fold within ~ε of exact on real inputs. Test by
   smoke + a partial Gate C re-run.

2. **Structured γ-commuting R̂_qk** *(elegant but small group, ~1 day
   to spec)*. Restrict the rotation to commute with diag(γ_q) by
   only rotating RoPE pairs `(i, i+head_dim/2)` where γ_q[i] ≈
   γ_q[i+head_dim/2]. In the NEOX-layout RoPE, paired positions have
   the same frequency but different γ values; we'd select the
   sub-set where γ values match within some tolerance and only
   rotate inside that sub-set. Yields a much smaller rotation group
   (probably 8-32 dim instead of 128) but no κ approximation
   needed. Likely too weak alone; consider combining with (1).

3. **Modify llama.cpp** *(opposite of the design goal, ~1 week)*.
   Add a "scaled-RMSNorm" op that applies a per-element correction
   inside the norm kernel, undoing the per-input fold bias from the
   server side. Fastest to land but violates AloePri's "no infra
   change" deployment thesis. Document, defer.

4. **Switch demonstrator backbone** *(cleanest path, hardest pivot,
   ~3 days)*. Drop Qwen3, pin v1 to Qwen2.5-1.5B or any non-QK-norm
   variant. Full Algorithm 2 becomes deployable verbatim from the
   paper. Cost: loses architectural alignment with the GELO-LLM
   demonstrator on the GELO route (also Qwen3). Loses comparability.

**Gating signal:** the attack-resistance benchmark (Phase 2 from the
sibling handoff). If ISA TTRSR with head-shuffle-only stays ≤ 15 %,
the gap is academic and (1)-(4) are optional. If ISA TTRSR > 15 %,
work the list. **Empirical check first, math second.**

**Skill suggestions:**
- `grill-me` before committing to (1) — the per-site κ calibration
  has a few sub-decisions worth stress-testing (calibration corpus
  choice, online vs per-batch κ, accuracy of the i.i.d.-corrected
  bound).
- `diagnose` if (1) lands but ISA TTRSR is still above threshold —
  the disciplined reproduce → minimise → hypothesise loop fits.

---

## What's NOT in this handoff

- The Phase-2 attack-harness work (Rust `attack_export.rs` serialiser
  + Python `evals/aloepri-attacks/`) is tracked separately in
  [`docs/plans/handoff-aloepri-attack-resistance.md`](docs/plans/handoff-aloepri-attack-resistance.md).
  That's the prerequisite for empirical gating of (b)'s next-steps —
  but the actual handoff for it lives there, not here.
- IFEval re-run (timeout-deferred Gate C task) — small loose end,
  not load-bearing for (a) or (b).
- Streaming / EOS handling — deferred for unrelated reasons.

## Suggested fresh-session opening moves

1. Read this handoff + `docs/plans/path-2-status.md` (entries dated
   2026-05-18) end to end.
2. Read protocol doc §07 and §08 in `docs/prototype/aloepri-llm.html`
   (the doc is served at `http://127.0.0.1:8000/aloepri-llm.html`
   if the python http server is still running; otherwise re-spawn
   per the existing pattern).
3. If pursuing (a) first: read `python/aloepri-llm/obfuscate_qwen3_gguf.py`
   focus on the `keymat` mode + the existing identity-pad / gamma-only
   modes which act as regression fixtures.
4. If pursuing (b) first: read `python/aloepri-llm/lib/alg2.py` end to
   end — the math is annotated and the QK-norm fold removal is
   commented in `obfuscate_qwen3_gguf.py:333-388`.
5. Either path: do NOT touch `gelo-protocol` API (frozen per Phase 1
   handoff).
