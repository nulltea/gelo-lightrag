# Handoff — full Algorithm 2 vs Qwen3 architectural shape

**Date:** 2026-05-19
**Status:** research / scoping. Next session can pick a path and prototype.
**Companion to:** `docs/handoffs/2026-05-19-m2-7-attack-findings.md` (the
attack-resistance failures whose root cause is the gap analysed here).

## 0. The underlying problem — one sentence

**Qwen3's `attn_q_norm` and `attn_k_norm` sit between `W_q`/`W_k` and the
RoPE+dot-product. Algorithm 2's intra-head transforms (R̂_qk, Ĥ_qk,
Ẑ_block) need to bake into the output axis of `W_q`/`W_k`, but the
per-element γ_qk inside the RMS denominator destroys the
M_q-commutativity that makes those transforms invisible to the model —
and the paper's §5.2.5 fold construction, which would normally bridge
this, only proves correct for *input-axis* RMSNorm sites.** Qwen2 has no
QK-norm; the public reference repo is Qwen2-only, so there is no
worked example for this case.

### Expanded — what the model topology actually looks like

Qwen3 dense attention, per layer (head_dim = 128, n_q = 16, n_kv = 8 on
1.7B):

```
x_residual ──► attn_norm(γ_input) ──► W_q ──► attn_q_norm(γ_q) ──► RoPE ──┐
                              │                                          ├─► softmax(Q·Kᵀ/√d_h) ──► attn_v ──► W_o ──► +
                              └► W_k ──► attn_k_norm(γ_k) ──► RoPE ──────┘
```

- `attn_norm.γ_input` has shape `(d,) = (2048,)` — operates on the
  **residual stream** (input axis of W_q/k/v).
- `attn_q_norm.γ_q`, `attn_k_norm.γ_k` have shape `(head_dim,) = (128,)` —
  operate on the **head_dim axis** of Q/K, **after** projection.
- Algorithm 2 needs M_q = R̂_qk · Ĥ_qk · Ẑ_block on head_dim. The
  obvious fold is `W̃_q = W_q · M_q` (post-multiply on output axis), so
  that `Q_obf = W_q · x · M_q`. But then:

  ```
  attn_q_norm(Q_obf, γ_q) = (Q_obf · γ_q) / RMS(Q_obf · γ_q)
                         = ((W_q · x · M_q) · γ_q) / RMS((W_q · x · M_q) · γ_q)
  ```

  For this to equal `attn_q_norm(Q_plain, γ_q) · M_q` we would need
  `M_q` to commute with `Diag(γ_q)` and `RMS(Q_obf · γ_q) = RMS(Q · γ_q)`.
  In general it doesn't, because γ_q is non-uniform across head_dim
  indices.

That's the gap. Without an answer to this, the §05 deployment can
deploy *only* the inter-head shuffle (τ_kv, τ_group) and the
residual-stream pre-norm fold — which is exactly what `obfuscate_qwen3_gguf.py:333-376`
currently ships.

---

## 1. Paper analysis — what §5.2.3 and §5.2.5 actually say

### 1.1 Algorithm 2 itself is silent on QK-norm

§5.2.3 + Algorithm 2 produce four obfuscated weight tensors —
`(W̃_q, W̃_k, W̃_v, W̃_o)` — and stop there. The pseudocode line:

```
W̃_q = Q̂_q · W_q · R̂_qk · Ĥ_qk · Ẑ_block
W̃_k = Q̂_k · W_k · R̂_qk · Ĥ_qk⁻¹ · Ẑ_block_η
```

assumes `Q_plain = W_q · x_norm` flows directly into RoPE and the dot
product. No norm site appears between W_q and the next op. That
assumption holds for **Qwen2.5 / Llama3 / DeepSeek-R1-Distill-Qwen** —
none of those have QK-norm. It fails for Qwen3.

§5.2.3 also says:

> As for MLA, low-rank matrices are employed for query and key weights,
> with the integration of decoupled RoPE. Therefore, we obfuscate the
> low-rank weights in MLA using another set of invertible transformations.

So the paper explicitly recognises that MLA needs a tweaked recipe.
But it makes no equivalent statement about GQA-with-QK-norm — which is
exactly Qwen3's shape. That's the silence the public reference repo
inherits.

### 1.2 §5.2.5 fold is *only* the input-axis case

The §5.2.5 text:

> Let `W_norm = Diag(w_norm)` denote the diagonal matrix corresponding
> to the RMSNorm weights. Assuming that the input data `x` of any
> normalization layer follows a Gaussian distribution, we use
> `κ = E[||x·P̂||/||x||]` as the coefficient for the obfuscated
> normalization layer to adjust for the bias induced by the `P̂`
> transformation. We then fuse an RMSNorm layer with weights
> `w̃_norm = 1·κ` and a linear layer with weights `W_norm` to replace
> the plaintext RMSNorm layer. **The weights of the linear layer
> `W_norm` can be merged into the layer adjacent to the RMSNorm layer
> before applying weight obfuscation.**

That last sentence is the load-bearing one. "The layer adjacent to the
RMSNorm layer" = the linear layer **downstream** of the norm, on whose
**input** axis γ_diag is applied. For our case:

- ✅ `attn_norm → W_q/k/v` — γ_input folds into W_q's *input* axis as
  `W_q' = W_q · Diag(γ_input)`. **Algebraically exact**, no Gaussian
  assumption needed. This is what `fuse_gamma_pre` in
  `obfuscate_qwen3_gguf.py:167` already does.
- ❌ `W_q → attn_q_norm → RoPE → dot` — γ_q sits *after* W_q, so it
  would have to fold into W_q's *output* axis. The paper construction
  *does not derive this case*. The κ correction in §5.2.5 is specifically
  for the dim expansion (`||x·P̂|| / ||x||` measures the keymat blow-up,
  not the per-input γ-correlation effect we hit on QK-norm).

So when the prior session tried fold formula `κ_q = √(mean(γ_q²))` and
got degenerate output ("a again ... ... ..."), it wasn't a paper
construction misimplemented — it was a reasonable extension of §5.2.5
to a case the paper doesn't cover, and the extension was wrong for
trained models. The handoff already diagnosed *why* (q and γ are
correlated by training); the new framing is that **the paper itself
doesn't promise §5.2.5 works on output-axis folds**.

### 1.3 The Qwen3 evaluation in §7 vs the public release

§7.1 explicitly lists Qwen3 in evaluated models. §7.2 shows AloePri at
≤15% TTRSR on Qwen2.5-14B-Instruct. Table 3 shows accuracy + privacy
across several models including Qwen3 variants. So the ByteDance
internal industrial build *does* successfully run AloePri on Qwen3.

What's not in the public artifact:

- No §F.1 / §G implementation appendix surfaces a Qwen3-specific QK-norm
  recipe (EdgeQuake's full-text retrieval over the paper found nothing).
- The reference repo `sheng1feng/Aloepri @ 60e8ea3` has zero Qwen3 path
  (next section).

The conservative inference: **the industrial build at ByteDance has a
QK-norm fix they didn't ship in the academic release.** What that fix
is, the paper doesn't say. That's the literal gap.

---

## 2. Reference implementation analysis

`vendor/aloepri-py @ 60e8ea3` is the only published implementation of
Algorithm 2.

### 2.1 It's Qwen2-only

Every attention module imports from `transformers.models.qwen2.modeling_qwen2`:

```
src/stage_b.py:10
src/stage_g_attention.py:8,14,15
src/obfuscate_attention_complex.py:8
src/keymat_attention_bridge.py:8,9
```

Class names: `TracingQwen2Attention`, `ComplexQwen2Attention`,
`KeyMatFusedQwen2Attention`, `StaticizedQwen2Attention`. Zero
references to `q_norm` / `k_norm` in implementation code (one mention
in a Chinese-language stage-E history doc which is annotation only,
and one cosmetic `.norm()` call inside the IMA scorer).

So when the paper claims Qwen3 evaluation, the public repo cannot
substantiate it — the code only handles Qwen2.

### 2.2 What the reference *does* fold (and why it works for Qwen2)

`StaticizedQwen2Attention.__init__` (`stage_h_attention_static.py:83-204`)
takes `input_norm_weight` as an explicit constructor argument and
builds:

```python
right_bridge = q * norm_weight.unsqueeze(0)
base_q_weight = attention_module.q_proj.weight @ right_bridge.T
```

This is the input-axis fold of attn_norm.γ_input into W_q (and
similarly W_k, W_v). It's the same construction our rewriter applies in
`PER_BLOCK_FUSION_MAP` (`obfuscate_qwen3_gguf.py:91-94`). No QK-norm
appears anywhere because Qwen2 doesn't have it.

The intra-head transform is then applied to W_q's output axis cleanly:

```python
q_dense = _block_diag_repeat(q_matrix, num_heads)
q_weight, q_bias = _apply_output_transform(
    base_q_weight, q_bias,
    dense_transform=q_dense,
    feature_order=q_feature_order,
)
```

On Qwen2 this works because `RoPE(W_q · x · M_q) = RoPE(W_q · x) · M_q`
when M_q is block-diagonal in the RoPE-aware (R̂_qk, Ẑ_block) layout
(see `attention_keys.generate_r_qk` and §5.2.3 RoPE-compatibility
argument). And the dot product `Q · Kᵀ` cancels because `M_q · M_k⁻¹ = I`
by construction (when M_k uses Ĥ_qk⁻¹ and Ẑ_block_η matched to M_q).

On Qwen3, the per-element `Diag(γ_q)` between W_q and RoPE breaks both
identities.

### 2.3 No upstream fix branch / fork

WebSearch turned up no public fork or branch of the reference repo
adding Qwen3 support. The arXiv ID `2603.01499` is the technical
report itself (just published — the model knowledge-cutoff is January
2026 and the paper appears as a 2026 work, so any successor publication
postdates this session). No follow-up GitHub project found.

So we are on our own for the Qwen3 recipe. ByteDance's industrial
solution is invisible to us.

---

## 3. Creative analysis — what to try next

Three options, ordered by **expected effort × likelihood of landing**.
Pick one before re-running M2.7.

### Option A. Empirical per-site κ_qk via calibration corpus *(elegant but theoretically thin)*

**Idea.** Promote `κ_q = √(mean(γ_q²))` (the i.i.d.-Gaussian bound that
breaks) to `κ_q_site = E_{x∼corpus}[ ||W_q·x ⊙ γ_q|| / ||W_q·x|| ]`,
measured empirically per attention layer.

**Why this might land.** The §5.2.5 fold *correctness* only requires
`κ ≈ true ratio`. Replacing a closed-form (i.i.d.-Gaussian) constant
with an empirical mean over real Qwen3 activations changes "exact in
the wrong distribution" into "exact in expectation in the right
distribution". The per-prompt error becomes the variance of the ratio
around its mean — small if γ_q's correlation with q is roughly uniform
across the corpus.

**Why it might fail.** Softmax is non-Lipschitz around peaks. If the
ratio's variance is high, you still get bad attention on outlier
prompts. The 16 GQA query heads each have their own γ_q — 28 layers ×
16 heads × 2 (q, k) = 896 κ_site values. Calibration needs enough
prompts to get a stable estimate on each.

**Concrete next step.** Reuse the AloePri default-prompts set
(`vendor/aloepri-py/src/defaults.py`) — ~256 prompts, paper-comparable.
Patch `obfuscate_qwen3_gguf.py` to:
1. After loading γ_q, run plaintext Qwen3 on the calibration corpus
   with hooks dumping pre-norm Q at each `attn_q_norm` site.
2. Compute `κ_q_site = mean(||q ⊙ γ_q|| / ||q||)` per layer per head.
3. Fold γ_q into `W_q` output axis as `W_q · Diag(γ_q)` (the same
   per-element operation), AND set `attn_q_norm.γ_q` to a scalar
   `κ_q_site` (broadcast back to head_dim).

If this works (smoke test passes coherent French capital), the full
Algorithm 2 (R̂_qk, Ĥ_qk, Ẑ_block) can then be applied to W_q's output
axis on top, because the QK-norm site has become a per-input scalar
ratio that commutes with M_q.

**Effort:** ~2 days to implement + 1 day to calibrate + validate. Risk:
empirical mean might still not be tight enough; softmax amplifies even
small ratio variance.

### Option B. RoPE-frequency-aware γ-commuting M_q *(constrains M_q, no κ approximation)*

**Idea.** Restrict the intra-head transform `M_q = R̂_qk · Ĥ_qk · Ẑ_block`
so that it commutes with `Diag(γ_q)` *exactly*, eliminating the κ
problem entirely.

`M_q · Diag(γ_q) = Diag(γ_q) · M_q` holds iff M_q only mixes head_dim
indices `i, j` where `γ_q[i] ≈ γ_q[j]`. Concretely:

- NEOX-layout RoPE pairs index `(i, i + head_dim/2)`. The two halves
  share the same RoPE frequency. **If we further require
  `|γ_q[i] − γ_q[i + head_dim/2]| < ε`**, the 2-D rotation `R̂_qk` on
  that pair commutes with `Diag(γ_q)` up to ε.
- Then `Ẑ_block` (which permutes RoPE pairs) is allowed only between
  pairs in the same γ-iso-tonic cluster.
- `Ĥ_qk` (diagonal scaling per pair) freely commutes with any diagonal.

**Why this might land.** No κ approximation, no Gaussian assumption, no
training-distribution dependence. Algebraically exact ⇒ no softmax
amplification of per-input error.

**Why it might fail.** The γ-iso-tonic clustering might leave a very
small obfuscation group. If γ_q values are all distinct (no near-ties
in head_dim), the only commuting M_q is `Diag` × `R̂_qk` with axis-
aligned rotation (no useful mixing) plus a 1-step `Ẑ_block` (identity
within trivial 1-element clusters). Worth measuring before committing:
**count the γ-iso-tonic clusters at ε ∈ {0.01, 0.05, 0.1, 0.25}** across
all 28 layers × 16 heads. If clusters of size ≥ 8 cover ≥ 50% of
head_dim positions, this is a real option.

**Concrete next step.** Write a 30-line `python/aloepri-llm/scripts/measure_gamma_qk_clusters.py`
that loads plaintext Qwen3 GGUF, extracts every `blk.*.attn_q_norm.weight`
and `attn_k_norm.weight`, clusters within each head, and reports cluster-
size distribution.

**Effort:** 30 minutes to measure, then if numbers are favourable, ~3
days to specialise `lib/alg2.py` to γ-iso-tonic rotation groups.

### Option C. Patch llama.cpp to compute per-input κ on the server *(opposite of paper goal, fastest to land)*

**Idea.** Modify the GGML attention kernel (in our nulltea fork) so the
QK-norm op accepts an obfuscated γ_q_baked weight and computes the
correction ratio at runtime:

```
q_normed = q * γ_q_baked / RMS(q)
correction = ||q * γ_q_baked|| / ||q||     # cheap, two reductions
q_out = q_normed * (correction / κ_target) # rescale to plaintext path
```

Then the algebraic identity `M_q · attn_q_norm(W_q · x) = attn_q_norm(M_q · W_q · x · M_q⁻¹) · M_q`
holds *per-input*, because the per-input κ correction is computed
exactly. Full Algorithm 2 then deploys verbatim with no math
modification.

**Why this might land.** Doesn't depend on training distribution; works
for any γ_q whether or not co-adapted with q. Same construction as
"scaled-RMSNorm" mentioned in `handoff-aloepri-quantisation-and-alg2-gaps.md`
option 3, but framed as a single per-attention runtime correction
rather than a new norm op.

**Why it might fail / not be acceptable.** Violates AloePri's stated
"no infra change" deployment thesis (paper Constraint 3 — software
compatibility). The fork only runs against our patched llama.cpp; can't
ship to upstream vLLM/SGLang without their cooperation. **However:** we
already ship a patched llama.cpp (vendor/llama.cpp submodule pinned at
nulltea fork commit `49680b131` for M2.7's tensor-dump). The bar is
already "we ship a fork"; adding one more attention-kernel patch is
incremental.

**Concrete next step.** Spec the kernel patch in `vendor/llama.cpp/ggml/`
to add `ggml_rms_norm_with_correction` op. Wire `attn_q_norm` to it.
Bake `γ_q_baked = γ_q` (unchanged), with a new metadata flag
`aloepri.qk_norm_correction = true`.

**Effort:** ~1 week. Same magnitude as option (b) but with a different
risk profile (engineering complexity vs unknown empirical viability).

### Option D (rejected, for completeness). Switch demonstrator backbone

Drop Qwen3, pin v1 to a non-QK-norm model (Qwen2.5-1.5B, Llama-3.2-1B).
Full Algorithm 2 deploys verbatim from the paper.

**Why rejected.** v1 GELO-LLM demonstrator (path-1) is on Qwen3-1.7B
(per memory `qwen3_v1_demonstrator.md`). Switching backbones on the
AloePri side breaks architectural comparability between the two paths,
which is the entire point of having two paths in this repo. Document
and move on.

---

## 4. Recommended ordering

1. **Measure first** (option B's pre-flight): 30-minute γ-iso-tonic
   cluster measurement. If clusters of size ≥ 8 cover ≥ 50% of
   head_dim, option B becomes the cleanest path (no approximation, no
   infra patch). If clusters are tiny, B is dead.
2. **If B viable**: prototype option B. 3-day spike. Smoke test +
   partial Gate C. Re-run M2.7 if it passes.
3. **If B is dead or marginal**: prototype option A (empirical κ_qk).
   2-day spike. Risk is high but cost is moderate.
4. **If A and B both fail**: commit to option C (runtime correction in
   llama.cpp). 1-week effort, but algebraically certain to work.

**Sequence the spikes; don't parallelise.** Each option's outcome
informs whether the next is worth attempting. In particular, option C
is a known-good fallback — only invest the week if A and B both die.

---

## 4a. Option B pre-flight result — DEAD (measured 2026-05-19)

Ran `python/aloepri-llm/scripts/measure_gamma_qk_clusters.py` against
`Qwen_Qwen3-1.7B-Q8_0.gguf`. Raw output at
`evals/aloepri-attacks/results/m2_7-gamma-qk-cluster-preflight.txt`.

**Aggregate across 56 vectors (28 layers × {q, k}):**

| ε    | mean pair-symm % | mean Ẑ-coverage %  | verdict   |
|-----:|-----------------:|-------------------:|:----------|
| 0.01 |              3.6 |                0.0 | dead      |
| 0.05 |             16.0 |                0.0 | dead      |
| 0.10 |             25.9 |                1.4 | dead      |
| 0.25 |             43.8 |               22.0 | dead      |

Coverage = % of head_dim positions inside a strict γ-band cluster of
size ≥ 8 *pairs*, restricted to pairs that are already γ-symmetric at
the same ε (so both R̂_qk and Ẑ_block are simultaneously enabled).

**Even at ε=0.25 (a 17 % relative tolerance on the median-γ scale of
~1.5 — already too loose for softmax-around-peaks stability), only
22 % of head_dim positions land in useful Ẑ_block groups.** At
operational ε ≤ 0.10 the structure is essentially absent.

**Why so hostile, especially on the K side.** Several `attn_k_norm`
tensors have extreme outliers: γ_max = 68.0 at layer 0, ≥ 20 on
layers 1, 3, 4, 5, 12, 15. Qwen3 trains QK-norm γ to compensate for
attention-score saturation, and that compensation is per-head_dim-
index, not per-pair. The trained vectors are essentially asymmetric
across NEOX pair index (i, i + d_h/2). The handoff anticipated this
exact case ("If γ_q values are all distinct (no near-ties in
head_dim)…").

**Bonus observation worth recording.** A handful of layers do clear
50 % coverage on their *own* (layers 4, 8, 22, 24 on the Q side at
ε=0.25). A "use full Algorithm 2 where γ is gentle, fall back to
identity elsewhere" hybrid is feasible mechanically, but **the M2.7
attacks target layer 0** — and layer 0 fails the threshold (20 %
Q coverage, 25 % K coverage at ε=0.25). The hybrid does not defend
the surface that's actually failing.

### Decision

Option B is **dead** for Qwen3-1.7B. Do not invest the 3-day spike.

### Forward path

- **Option A (empirical κ_qk) is also at high risk** because the K-
  side γ-heterogeneity (γ_max = 68 in one head) blows up the
  variance of the empirical ratio estimator. Q-side might land
  (most γ_q stay in a benign range), but K-side is the more
  defensible surface against IMA-style attackers at layer-0 since
  attention scores fold through both. A Q-only A is therefore
  unlikely to close the gap on the IMA HiddenState surfaces.
- **Option C (runtime κ correction in patched llama.cpp) becomes
  the recommended path.** The bar "we ship a forked llama.cpp" is
  already met (vendor/llama.cpp pinned at `49680b131` for the M2.7
  tensor-dump). One more attention-kernel patch is incremental,
  and is algebraically certain to work regardless of γ-distribution.

**Updated ordering (post-measurement):**

1. ~~Option B pre-flight~~ — done, dead.
2. Spec Option C in detail (kernel-level: which GGML op extends, how
   the metadata flag flows in, fp32-only or quantisation-aware) — 1
   day.
3. Implement and merge into `vendor/llama.cpp` fork — 3–5 days.
4. Rebuild `keymat-h128-pi-noise-alg2-FULL-fp32.gguf` with full
   Algorithm 2 intra-head transforms applied and the κ-correction
   metadata flag set — ~1 hour after the kernel lands.
5. Re-run M2.7 — both IMA attacks should drop ≤ 15 %, ISA HS should
   fall below plain ceiling (8.7 %).

A pragmatic detour worth considering before committing the week:
**implement Option A for the Q side only and re-measure IMA at
layer 0**. If Q-only empirical κ closes 50 %+ of the IMA gap, that
suggests the K-side outliers are not the dominant signal for the
attacker and Option A might suffice with both sides. If it closes
< 20 %, commit to Option C without further detour.

---

## 5. Surfaces this report deliberately does not address

- **Whether the intra-head transforms actually defend IMA at layer-0
  hidden state.** The M2.7 handoff asserts they do (paper Table 3
  citation). Mechanism: keymat decoy dims propagate through attention
  weights, and intra-head transforms scramble that propagation. There
  is a subtler question — does the layer-0 ridge attack actually
  observe enough of the propagated structure to justify the paper's
  ablation drop from 0.82% to 0%? — that empirical question is for the
  re-run after the fix, not this scoping pass.

- **Quantisation interaction with γ_qk fold.** The (a) handoff
  established fp32 is required for the residual-stream fold. Folding
  γ_qk into W_q output axis multiplies the per-row variance of W_q
  even further (γ_q values up to ~10× the row scale on some heads).
  Q8_0 will be even worse. Plan to stay fp32 unless the (a) work
  lands first.

- **The Ẑ_block-η for K's separate sub-permutation.** Algorithm 2 lines
  6 use `Ẑ_block^η(i)` for K, distinct from `Ẑ_block` for Q. Our
  `lib/alg2.py` uses the same z_block for both — possible bug
  unrelated to the QK-norm problem, worth a separate trace.

- **Bias handling.** Qwen3 has q_proj.bias / k_proj.bias / v_proj.bias.
  When γ_qk folds into W_q output, biases need γ_qk multiplied in too,
  exactly as the reference's `_apply_output_transform` does. Ensure
  the chosen option threads bias through.

## Appendix — exact code sites for the fix

- `python/aloepri-llm/lib/alg2.py:140-192` — `LayerAlg2Keys.build_layer_keys`
  already generates the full key set; only `q_matrix`/`k_matrix` are
  overridden to identity at the rewriter site.
- `python/aloepri-llm/obfuscate_qwen3_gguf.py:333-376` — the dormant-wiring
  block. To enable full Algorithm 2, replace the
  `q_matrix=np.eye, k_matrix=np.eye` overrides with the real
  `full_keys.q_matrix, full_keys.k_matrix` AND emit the chosen γ_qk
  treatment (per option A/B/C above).
- `python/aloepri-llm/obfuscate_qwen3_gguf.py:475-496` — Algorithm 2 apply
  block. Currently passes `None` for `dense_transform`; passing
  `_block_diag_repeat(q_matrix, num_heads)` re-enables intra-head.
- `python/aloepri-llm/obfuscate_qwen3_gguf.py:91-101` — the
  PER_BLOCK_FUSION_MAP. Option A would add `attn_q_norm.weight` and
  `attn_k_norm.weight` here with a new "post" direction. Option B
  leaves this map unchanged. Option C touches llama.cpp, not this
  rewriter.

## Sources

- [Towards Privacy-Preserving LLM Inference via Covariant Obfuscation (Technical Report) — arXiv abs](https://arxiv.org/abs/2603.01499)
- [Same paper — PDF mirror](https://arxiv.org/pdf/2603.01499)
- [sheng1feng/Aloepri reference repo — `stage_h_attention_static.py` Algorithm 2 anchor](https://github.com/sheng1feng/Aloepri/blob/60e8ea3cc04353b7a0058e9c86d67461c7d25763/src/stage_h_attention_static.py#L83-L204)
- [HuggingFace transformers `modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) — authoritative Qwen3 attention shape (QK-norm sites)
