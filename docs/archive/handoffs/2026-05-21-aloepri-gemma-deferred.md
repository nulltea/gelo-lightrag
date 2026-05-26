---
type: handoff
status: current
created: 2026-05-18
updated: 2026-05-21
tags: [aloepri, gemma]
---

# Handoff — AloePri on Gemma 3/4 (deferred)

> **Status:** Deferred. Active AloePri work pivoted to Qwen3.5 2B / 4B
> on 2026-05-18 because Gemma's per-block **post-norms** cannot be made
> covariantly obfuscated by the paper §5.2.5 fuse-and-scale construction
> alone — they require either a custom RMSNorm op or an alternative
> algebraic derivation we have not built yet.
>
> **Audience:** future agent / future-us picking this back up after the
> protocol has been validated end-to-end on a pre-norm-only model
> (Qwen3.5). At that point the AloePri pipeline (offline rewriter,
> Algorithm 1 / 2 / Π / noise, client wrapper, attack benchmark) is
> mature; the only Gemma-specific work left is the covariant-norm
> question and the PLE / hybrid-attention deltas. This doc is the
> entry point for that work.
>
> **What this doc does NOT duplicate** (reference instead):
>
> - Plan: [`../../plans/path-2-aloepri-gemma.md`](../../plans/path-2-aloepri-gemma.md)
> - Running status: [`../../dev/logs/path-2-status.md`](../../dev/logs/path-2-status.md)
> - Protocol reference: [`./aloepri-llm.html`](./aloepri-llm.html)
> - Round-2 architecture analysis (Gemma 4 PLE, K=V, p-RoPE, hybrid):
>   [`../../research/private-llm-inference-round-2.md`](../../research/private-llm-inference-round-2.md)
>   §D
> - GELO vs AloePri comparison: [`../../research/aloepri-vs-gelo.md`](../../research/aloepri-vs-gelo.md)
> - Paper: AloePri, arXiv 2603.01499 — §5.2.5 *Layer Normalization Transformation*

---

## 1. Why we pivoted away from Gemma 4

Paper §5.2.5 fuses each RMSNorm's per-dim γ into the adjacent linear
weight and replaces γ at the norm site with a constant κ scalar.
The construction is mathematically exact in plaintext for
**pre-norms**:

```
(x · γ / RMS(x)) @ W  ==  (x / RMS(x)) @ (Diag(γ) · W)
```

For **post-norms** (the norm sits between a linear's output and the
residual add: `... → Linear(W) → RMSNorm(γ) → +residual`), the same
trick is **not exact** because RMS reduces over all dims and is
sensitive to γ's per-dim values:

```
post-norm target:    y  = (out · γ) / RMS(out)
"fuse γ backward":   y' = (out · γ) / RMS(out · γ)
y'/y                    = RMS(out) / RMS(out · γ)
```

The ratio is 1 only when γ is constant (or the operand is balanced in
a specific way), which isn't true for trained LLMs. Empirically the
ratio is ≈ 1/√d per site; for Gemma 4 with 3+ post-norms per block
and 35 blocks, the error compounds catastrophically (verified — produces
gibberish output).

Gemma 4 dense has FIVE residual-stream norm sites per block of which
**three are post-norms** (`post_attention_norm`, `post_ffw_norm`,
`post_norm` ≡ per_layer_post_norm). Plus the global `output_norm`
(pre-norm — fine). The paper-tested Qwen/Llama/DeepSeek dense
architectures use only **pre-norms** (2 per block), so the paper
construction works for them cleanly. Gemma 4 specifically is the
outlier in this generation.

**Active pivot target:** Qwen3.5 2B and 4B. Pure pre-norm
architectures; §5.2.5 fusion is exact; AloePri's empirical accuracy
results (0–3.5 % loss) were measured on this family of models, so we
have a baseline to validate against. Once the protocol is end-to-end
on Qwen3.5 — offline rewriter + Algorithm 1 + Algorithm 2 + Π + noise
+ client wrapper + attack benchmark — we revisit Gemma.

## 2. Findings worth carrying forward

These are validated and load-bearing for any future Gemma 4 attempt;
re-using them avoids re-discovering the same things.

### 2.1 llama.cpp upstream Gemma 4 status — supported

`LLM_ARCH_GEMMA4` is a full architecture class in mainline
llama.cpp (`vendor/llama.cpp/src/models/gemma4.cpp`). Type-detection
handles E2B (35 layers), E4B (42), 26B A4B (30, MoE), 31B (60).
**No fork needed for the architecture itself** — only for the
covariant-norm op (see §3).

### 2.2 `hidden_size = d + 2h` propagates cleanly

`gemma4.embedding_length` is read from GGUF metadata at line
`load_arch_hparams`; no assertion anywhere requires
`hidden_size == n_heads · head_dim`. The identity-padding artifact
(commit `b50f807`, ~10 GB on `:11438`) loaded and served correctly
with `d + 2h = 1792`. **This means the AloePri dim expansion does
not require a llama.cpp patch.**

### 2.3 K=V tying is a no-op at the GGUF level

`gemma4.attention.shared_kv_layers = 20` for E2B describes
**KV-cache runtime sharing** between layers (`n_layer_kv_from_start`
indexing), **not weight tying**. The GGUF stores `attn_k.weight`
and `attn_v.weight` as separate tensors at every layer. The AloePri
plan's M2.2b "K=V untie" sub-step is therefore not needed for Gemma 4
GGUF artifacts.

**Open constraint (untested):** for layers in the cache-sharing
group, the Algorithm-2 `Q̂_k` and `Q̂_v` matrices probably need to
match the values used by the layer they share with, otherwise Q·K^T
lands in mismatched obfuscation spaces. To be confirmed by reading
llama.cpp's KV cache binding code when Gemma 4 work resumes.

### 2.4 PLE table is one fused `[262144 × 8960]` tensor

Gemma 4 E2B / E4B stores Per-Layer Embeddings as a single
`per_layer_token_embd.weight` tensor of shape
`[n_embd_per_layer × n_layer, vocab]` = `[256 × 35, 262144]` =
`[8960, 262144]`. Not a per-layer-indexed structure.

Implication for AloePri: the **τ permutation on the vocab axis is
one numpy operation** (`arr_new[:, i] = arr_old[:, τ(i)]`), mirroring
the regular `token_embd` permutation. Earlier plans assumed per-layer
work; that's wrong.

The per-layer projection `per_layer_model_proj.weight` (shape
`[hidden, 8960]`) gets standard Q̂_R left-multiplication on the
residual axis (reads from residual).

### 2.5 Algorithm 1 KeyMat math verified

`python/aloepri-llm/lib` (and the vendored
`vendor/aloepri-py/src/keymat.py`) generate `(P̂, Q̂)` pairs
satisfying `P̂·Q̂ = I_d` to ≤ 3·10⁻⁷ max-absolute-error at fp64.
Tested on E2B (d=1536) and E4B (d=2560) dimensions.

### 2.6 κ for d=1536, h=128, λ=0.3 ≈ 8.0

`E[‖xP̂‖/‖x‖] ≈ 8.01` for Gaussian inputs at these dimensions
(measured 2000 samples). The dominant contribution is the `C ∈
R^{d×h}` matrix in P̂ = `[B C E]Z`: C entries have ≈ unit variance
and dim `(d, h)`, so `‖xC‖²/‖x‖² ≈ h ≈ 128`.

The §5.2.5-corrected κ used at the norm site is
`κ_correct = κ_E · √(d/(d+2h)) ≈ 7.42` — derived in
[`../../dev/logs/path-2-status.md`](../../dev/logs/path-2-status.md). This
accounts for the change in RMSNorm denominator dim. Documented but
not yet sufficient to make Gemma 4 work because of the post-norm
issue.

### 2.7 The offline rewriter (`obfuscate_gemma4_gguf.py`) is mostly correct

[`python/aloepri-llm/obfuscate_gemma4_gguf.py`](../../python/aloepri-llm/obfuscate_gemma4_gguf.py)
implements:

- `identity-pad` mode: zero-pad expansion. Validated end-to-end —
  loads, serves, produces correct "Paris" answer.
- `keymat` mode: §5.2.5 fusion (pre-norms only after the post-norm
  fix), Algorithm 1 obfuscation, output untying. Loads, serves,
  but produces gibberish due to the post-norm e_C^norm not being
  bounded.
- `gamma-only` mode: §5.2.5 fusion only, no dim expansion. Used as
  a regression test for fusion correctness.

Three tensor-classification subtleties surfaced and are encoded in
the script (re-use them on any future Gemma return):

1. `token_embd` writes to residual → use `arr @ P̂_R`; `output`
   reads from residual → use `arr @ Q̂_R^T`. They have the same GGUF
   shape but different transforms. The script's
   `RESIDUAL_NE0_READ_TENSORS` vs `RESIDUAL_NE0_WRITE_TENSORS`
   classification captures this.
2. `gguf-py`'s `GGUFWriter` already reverses shape internally —
   pass natural numpy shape, not pre-reversed ggml shape, or get
   offset drift in the tensor data section.
3. Mixed-quant passthrough doesn't work; everything must be
   dequantised + written F16/F32 uniformly. The Q8_0 `per_layer_token_embd`
   in particular is the trip-wire.

## 3. What's needed to bring Gemma 4 back into scope

Three independent work items, ordered by criticality.

### 3.1 Covariant post-norm — the actual blocker

Pick **one** of these approaches:

**Option A — llama.cpp patch for post-norm sites.** Add an op
`ggml_rms_norm_then_scale(x, γ_per_dim_obf, κ)` that:

```
y = κ · x / RMS(x)      # RMSNorm with scalar γ
y = y ⊙ γ_per_dim_obf   # elementwise post-multiply
```

This separates the RMS computation from the γ application; both
can then be made covariant individually. In the gemma4 forward
graph, replace the three post-norm `build_norm` calls with this
fused op. Mechanical, ~30 lines diff, guarded by a metadata flag
(`gemma4.aloepri.enabled = true`) so plaintext models still load.

Engineering scope: 2–3 weeks (op + graph wiring + tests + upstream
PR or vendored fork). The fork lives at
`vendor/llama.cpp/` already; this would add a worktree branch.

**Option B — algebraic reformulation.** Find a different §5.2.5-
style construction that handles `out · γ / RMS(out)` offline.
Candidates:

- Approximate γ by replacing it with its mean ḡ at the norm site,
  and fuse `(γ - ḡ) ⊙ ·` into adjacent layers. Probably loses
  significant accuracy because Gemma's γ has non-trivial variance.
- Insert a learned "γ adjustment" into a pre-existing scaling op
  (Gemma 4 has `layer_output_scale`, a per-layer scalar — already
  in the architecture, not yet used by AloePri).
- A novel construction. No known prior work.

Option A is the recommended path. Option B is a research project
on its own.

### 3.2 PLE token-axis permutation under τ

Once τ is added to the rewriter (general AloePri milestone, not
Gemma-specific):

```python
arr_new = arr_old[:, tau]    # for per_layer_token_embd natural shape (8960, vocab)
```

Same τ as `token_embd`. One tensor operation. The corresponding
per-layer projection `per_layer_model_proj` is already classified
as a ne0-read (Q̂_R^T transform) — needs no PLE-specific code.

**Novel attack surface from PLE under τ:** the per-layer gather
addresses (one per `(layer ℓ, position t)`) observe `τ(token_id)`
on the address bus. Paper §7 doesn't test TFMA under PLE — Gemma 4
gives the attacker `N_layers × seq_len` observations per prompt vs
1 in Qwen-class baselines. This goes in the M2.9 attack benchmark
as a Gemma-specific row.

### 3.3 p-RoPE-aware R̂_qk for global layers

Gemma 4 global layers apply RoPE only to the first `p · d_head`
dims (`p=0.25`). Algorithm 2's `R̂_qk = diag({R_i}_{i ≤ d_head/2})`
constructs a 2D rotation across all `d_head/2` RoPE-block pairs;
under p-RoPE this only makes sense on the rotated subset. The
non-rotated subset must use identity (or a different transform —
TBD).

Likewise `Ĥ_qk` and `Ẑ_block` must be restricted to the rotated
subset. Two `BlockPerm` configs are needed per layer-class:

- Local layers: `rope_base = 10_000`, RoPE on all `d_head/2` blocks
- Global layers: `rope_base = 1_000_000`, RoPE on first `0.25 ·
  d_head/2` blocks only

This is mechanical bookkeeping in the Algorithm 2 portion of the
rewriter (which doesn't exist yet).

### 3.4 Hybrid attention (sliding-window vs global)

Algorithm 2's attention-side transforms don't fundamentally depend
on whether the attention is sliding-window or global — they apply
per-head. The 4:1 (E2B) / 5:1 (E4B) pattern is consumed by the
gemma4 forward graph via the `sliding_window_pattern` metadata;
AloePri doesn't touch this.

**No special handling required** beyond the §3.3 p-RoPE concern.

## 4. Artifacts and pointers

### 4.1 Commits on `path-2-aloepri-gemma` branch

```
f8c1482  protocol HTML doc + plan retarget to llama.cpp
7788c02  M2.0 — Vulkan baseline running for Gemma 4 E2B Q8_0
b50f807  M2.3 gate cleared — llama.cpp accepts hidden_size = d + 2h
338c062  RMSNorm covariance finding (later reverted)
51363b7  Revert M2.3 verdict downgrade — §5.2.5 provides offline RMSNorm
```

The subsequent post-norm-fusion bug and pivot decision are
uncommitted at the time this handoff was written (working tree may
still contain the broken `mode=keymat` / `gamma-only` machinery in
`obfuscate_gemma4_gguf.py`).

### 4.2 Running containers (status as of pivot)

| Port | Container | Artifact | Status |
|---|---|---|---|
| 11437 | `llama-gemma4-e2b-aloepri-baseline` | Plaintext Gemma 4 E2B Q8_0 (from `unsloth/gemma-4-E2B-it-GGUF`) | ✅ correct output |
| 11438 | `llama-gemma4-e2b-aloepri-identity-pad` | Identity-padded F16, h=128 | ✅ correct output (math = plaintext) |
| 11439 | `llama-gemma4-e2b-aloepri-keymat` | Real Algorithm 1 + (broken) post-norm fusion | ❌ gibberish |
| 11440 | `llama-gemma4-e2b-aloepri-gamma` | γ-only mode, no dim expansion | ❌ gibberish (post-norm fusion bug) |

Stop the broken ones with `docker rm -f llama-gemma4-e2b-aloepri-{keymat,gamma}` when reclaiming resources. The baseline (`:11437`) and identity-pad (`:11438`) are worth keeping as regression references if disk allows.

### 4.3 Files of interest

- Rewriter:
  [`../../python/aloepri-llm/obfuscate_gemma4_gguf.py`](../../python/aloepri-llm/obfuscate_gemma4_gguf.py)
- Vendored Algorithm 1 reference:
  `vendor/aloepri-py/src/keymat.py` (gitignored — clone
  `sheng1feng/Aloepri @ 60e8ea3`)
- Vendored llama.cpp:
  `vendor/llama.cpp/` (gitignored — clone `ggml-org/llama.cpp`).
  Gemma 4 source: `vendor/llama.cpp/src/models/gemma4.cpp` —
  load_arch_hparams at line 3, load_arch_tensors at 31, forward
  graph at 145, per-layer-embedding helpers at 399 and 437.
- Reference's runtime RMSNorm bridge (research aid, NOT what we'll
  ever use in production):
  `vendor/aloepri-py/src/keymat_norm.py::KeyMatRMSNormBridge`

### 4.4 Hyperparameter values from the Gemma 4 attempt

- `d = 1536` (E2B), `d = 2560` (E4B)
- `h = 128` (paper default), gives expansion `2h = 256`
- `λ = 0.3` in Algorithm 1 (paper default)
- `seed = 42` used for all artifacts above
- `κ_E ≈ 8.01` measured for E2B
- `κ_correct ≈ 7.42` = κ_E · √(d/(d+2h)) per the §5.2.5 derivation

## 5. Suggested skills for the next session

If Qwen3.5 has cleared all the milestones by the time this is picked
back up, the natural sequence is:

1. **Re-verify the §3.1 post-norm constraint** still holds —
   check whether llama.cpp has added a `RMSNorm + scale` fused op
   upstream (it might, given other architectures may want similar).
   Use the `claude-code-guide` subagent or a quick read of
   `vendor/llama.cpp/src/ggml.h` for new ops.
2. **Port the Qwen3.5 rewriter** (which by then is correct, tested,
   and complete) to Gemma 4. Mostly file-list and per-block-norm-map
   changes. Use `Explore` to find the AloePri-pipeline files in their
   final form.
3. **Implement covariant post-norm** — §3.1 Option A.
4. **Add PLE permutation** — §3.2.
5. **Add p-RoPE-aware Algorithm 2** — §3.3.
6. Run the attack benchmark with PLE-aware TFMA variant.

If picking this up before Qwen3.5 is done, just read this doc, the
plan, the status doc, and the AloePri paper, then continue Qwen3.5
work first — there is no useful Gemma 4 progress to make until the
protocol is mature on a simpler architecture.

---

*Written by Claude Opus 4.7 (1M context) on 2026-05-18 as a session
handoff. Conversation log: see git history of
`docs/dev/logs/path-2-status.md` for the chronological account.*
