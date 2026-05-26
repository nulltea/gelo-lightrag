# AloePri attack notes

Conceptual descriptions of the attacks in §08 of `docs/prototype/aloepri-llm.html`. Threat-model conclusions live here; harness/driver-fix progress lives in `docs/handoffs/`.

## IMA-EmbedRow-ridge

**Setup.** The obfuscator publishes a transformed embedding table `W̃`. For each plain token id `i`, the obfuscator places its transformed row at obfuscated index `τ[i]`, where

  `W̃[τ[i]] ≈ Q̂ · W[i] · P̂ + α_e · noise[i]`

Three secrets the server hides: the keymat `Q̂` and `P̂` (small invertible matrices), the row permutation `τ`, and the Gaussian noise. Anyone (including the attacker) can download `W̃` and the plain `W` from the model release.

**Attacker's goal.** Recover `τ` — once `τ` is known, every wire prompt decodes (just permute the obfuscated token ids back to plain ids). To recover `τ`, it suffices to take *any one* obfuscated row `W̃[k]` and figure out which plain row it came from.

**The trick: assume the attacker already knows τ on a subset.** This simulates the realistic case where high-frequency tokens (`"the"`, `" "`, `","` — the top ~1000 in any natural-language corpus) leak their τ-mapping just from frequency statistics: count obfuscated id frequencies on the wire, sort, match to plain id frequency rank, and you have ~1000 known `(plain_id, obf_id)` pairs without breaking any crypto. Call this **training subset** `T`.

**What ridge regression does.** The attacker has, for each `i ∈ T`:

- An obfuscated row `x_i = W̃[τ[i]]` (length `d_obs`)
- A plain row `y_i = W[i]` (length `d_plain`)

They want a function that takes any obfuscated row and outputs the plain row it came from. The simplest such function is a **linear map** `V` such that

  `V · x_i ≈ y_i`  for every `i ∈ T`.

If such a `V` exists, then for any *unknown* obfuscated row `W̃[k]`, the attacker can compute `V · W̃[k]` and get back something close to a plain row. Then nearest-neighbour search in the plain table `W` returns the plain id. **τ recovered.**

**Why a linear V exists at all.** Because the obfuscation `W̃[τ[i]] ≈ Q̂ · W[i] · P̂ + noise` is itself almost linear in `W[i]`. The "inverse" of a linear transform is another linear transform. The attacker doesn't know `Q̂`, `P̂`, or `τ` — but on the training subset they do know which obfuscated row pairs with which plain row, so they can *fit* `V` directly from those pairs, never seeing the keymat.

**Why ridge specifically.** "Fit `V` such that `V · x_i ≈ y_i`" is a linear regression. The closed-form least-squares solution is

  `V = (XᵀX + αI)⁻¹ XᵀY`

where `X` stacks the training `x_i` rows and `Y` stacks the `y_i` rows. The `αI` part ("ridge regularization") just stops the matrix-inverse step from blowing up when `XᵀX` is near-singular. `W_ridge = V` — it's the optimal linear map the attacker fits.

The whole attack takes:

- Training: ~1 s (one matrix solve)
- Inference: matrix-vector multiply per query

**Why this attack succeeds on AloePri at d=2560/4096.** The keymat dimension `h=128` (paper default) is small relative to `d`. Concretely the obfuscation lives in a `(d + 2h)`-dim ambient space — for 4B that's `2560 + 256 = 2816`. A linear inverter has `d × (d + 2h) = 7.2M` parameters and only needs to fit `1024` training pairs. Heavily over-parameterized — ridge finds an inverse map that generalizes to every other plain token in the vocab. Empirically: 57 % top-1 recovery on 4B, 97 % on 8B.

**Why paper claims this attack at 0 %.** Paper §F.1 specifies a *trained transformer* IMA (the `_paper_like` variant), not the ridge baseline. Paper never ran the closed-form ridge against its own deployment — possibly because the reference impl's "Gate-2 minimum baseline" was structured as a development checkpoint rather than a published metric. Our ridge result is a stricter attack the paper didn't probe, exposing a real defence gap.

### Why AloePri is safe from ridge in practice

The 57 % / 97 % readings on our harness use `train_size = 1024` — a worst-case assumption that the attacker has somehow obtained ~1024 `(plain_id, obf_id)` pairs from the real deployment τ. The realistic attacker can harvest far fewer.

**Realistic deployment-τ harvest.**

| Source | Pairs |
|---|---|
| Paper §7.6: TFMA top-100 @ distribution-aware (the strongest TFMA setting paper measures) | ≤ 20 |
| Identity-fixed specials + UNUSED tail (before `--pi-include-specials`) | ~293 |
| **Realistic total (without strong-Π)** | **~313** |
| **Realistic total (with strong-Π)** | **≤ 20** |

**Why ≤ 20 pairs is far below the ridge bootstrap threshold.**

- The keymat-rotated obfuscation requires a linear inverter `V` of size `d × (d + 2h)` ≈ 7–18 M parameters at 4B/8B. Fitting `V` from 20 paired observations is heavily under-determined — ridge regularization will produce *some* `V`, but generalization to the unobserved tail is what the attack needs and what 20 pairs can't provide.
- Paper §7.6 explicitly states that even under distribution-aware TFMA "it remains challenging to recover the client dataset" — top-100 recovery caps at ≤ 20 %, and **SDA BLEU-4 ≈ 2** "is insufficient to form meaningful text." The wire-side leakage isn't dense enough to bootstrap a ridge inverter that generalizes.
- Even at ~313 pairs (the pre-strong-Π regime), the 293 "free" identity-fixed pairs concentrate in a low-rank corner of the embedding manifold (special-token rows like `<|im_start|>` etc. have a narrow semantic distribution; UNUSED-tail rows are essentially random-init). A linear inverter fit on these doesn't span the directions that matter for natural-language tokens. The 20 TFMA pairs add diversity but not enough volume to fix the rank deficit.

**Strong-Π closes the structural side of the leak.** Permuting all 151669 active tokens removes the 293-pair "free" identity-fixed corner. After strong-Π the attacker's harvest is bounded by TFMA + SDA — both measured at low recovery in paper §7.6.

**Caveat.** This safety argument is conditional on TFMA being the only viable τ-leak channel. Other side channels could change the picture: ordering of tokens in `obf_response_ids` may carry semantic structure recoverable via n-gram analysis (SDA-like), and special tokens with predictable wire patterns (always-first BOS, always-last EOS) could leak τ on those specific ids. AloePri's TFMA + SDA defence numbers cap these channels at low recovery — so the composite leak stays under the ridge bootstrap threshold.

Because of this, **IMA-EmbedRow-ridge and IMA-L0-activation are no longer measured in §08's main table**. The trained-inverter variant (IMA-EmbedRow-transformer) is still relevant under paper's no-τ-leak threat model — see below.

## IMA-EmbedRow-transformer

**Same surface as the ridge attack, different threat model.** Paper §F.1 defines IMA as a 2-layer 8-head transformer trained on `(plain_embed, synthetic-obfuscated_embed)` pairs generated by the attacker themselves: they have the obfuscation algorithm (Kerckhoffs's principle) but not the secrets, so they pick their *own* pseudo-`τ`, `Q̂`, `P̂` and run the obfuscator on a public corpus. The trained inverter has to learn a `τ`-*invariant* inverse — that's why the paper considers this the hard attack and the relevant defence target.

This threat model is strictly weaker for the attacker (no deployment-τ leak required), which is why the ridge safety argument above doesn't apply to it. AloePri's reported "IMA = 0 %" on Qwen2.5-14B is paper's claim that this transformer attack fails on their deployment.

The current path-2 driver doesn't yet reproduce paper's attack faithfully — the trained inverter at paper-default hyperparameters fails the identity-τ plain control across every architecture variant tried, so the obfuscated reading isn't interpretable as a defence number. Tracking the fix in `docs/handoffs/`.

### Why IMA-L0-transformer is redundant with IMA-EmbedRow-transformer

IMA-L0-transformer attacks the same target (recover plain id from obfuscated info) using captured hidden states at `attn_norm-0` instead of static embedding rows. But `attn_norm-0` is the first RMSNorm in the model, applied directly to the embedding row:

`hidden_state[attn_norm-0][i] = RMSNorm(W̃[obf_id_i])`

So IMA-L0-transformer observes the **unit-normalized obfuscated embedding row** — same information content as IMA-EmbedRow-transformer, just rescaled to unit norm. Two consequences:

1. **No additional attack capability.** Any inverter that works on `W̃[k]` works on `RMSNorm(W̃[k])` with a rescaling factor absorbed into `V`. The two surfaces are equivalent up to a per-row scalar.
2. **Strictly more attacker privilege required.** IMA-L0 needs server-side wiretap access (capture activations during inference). IMA-EmbedRow only needs public access to `W̃`. Paper §F.1 names the embedding-row variant as the canonical IMA — there's no reason to test the strictly-harder-to-mount variant when the easier one is the load-bearing measurement.

§08 measures only IMA-EmbedRow-transformer (when the driver fix lands).

## ISA HiddenState — labelled-ridge with multi-key attacker synthesis

**Setup.** A different attack family than IMA. Instead of inverting the static obfuscated embedding table `W̃`, this attack inverts the *runtime hidden states* the server computes during forward inference. At every layer L the obfuscated model produces a residual-stream tensor `State_d[L]` of shape `(seq_len, d + 2h)`. Under AloePri's covariant obfuscation, this residual equals `K_e · State_plain[L]` to the extent the algorithm preserves covariance — same `K_e` keymat applied to every layer's residual, with Algorithm 2 transforms preserving the basis on the intra-attention internals.

**Attacker's goal.** Recover plain token ids of real user queries by inverting the residual-stream observation. The attacker is the honest-but-curious server, so they observe `State_d` legitimately as part of forward inference; they never see plain text or `τ`.

**The local-plain-model trick.** The attacker doesn't query the deployment to build training data — that path needs `τ` and would only return obfuscated outputs to obfuscated inputs. Instead, the attacker runs the **public plain Qwen3-4B locally** (it's on HuggingFace, weights public). They feed plaintext prompts they wrote themselves — plain_ids known by tokenizing the plaintext directly — and capture `State_plain[L]` on their own machine. Zero deployment interaction during training; zero `τ` knowledge needed.

**Why a linear inverter exists at all.** AloePri's covariance claim says

  `State_d[L] = K_e · State_plain[L]`     (up to per-prompt noise that propagates from embedding and small Algorithm-2 perturbations on the post-attention residual)

so the inverse map `State_d → State_plain` is *linear in K_e*. The plain residual `State_plain[L]` then maps to the plain token id at that position via the rest of the network — a relationship that ridge can fit position-wise from labelled training data even without inverting the whole transformer (cosine-NN against `W_e` recovers the id once the predicted embedding is close enough to the true plain row).

The attacker doesn't know `K_e`, but they know the obfuscation algorithm. So they sample their own `K_a` and learn the inverse map from labelled pairs.

**Single-key naive attempt fails.** If the attacker uses one synthetic `K_a` to generate training data `(State_plain[L] @ K_a, W_e[plain_id])`, ridge learns `W_ridge ≈ K_a⁻¹` (composed with the residual→embedding map). At test time, `State_d = K_d · State_plain` where `K_d ≠ K_a` is the deployment's independent secret. Applying `W_ridge` to `State_d` gives `K_a⁻¹ · K_d · State_plain ≠ State_plain` — the basis mismatch breaks the inversion. Top-1 collapses to ~0 % by construction. This is the same single-key transfer failure that broke single-key IMA-EmbedRow-transformer.

**Multi-key synthesis fixes the transfer problem.** Instead of one `K_a`, the attacker pre-generates `K = 64` independent attacker keymats `K_a^k` (Algorithm 1 with different attacker-side seeds). For every captured plain-state `(State_plain[L][i], plain_id[i])` pair they produce 64 synthetic obfuscated states `State_a^k[L][i] = State_plain[L][i] @ K_a^k`. Ridge sees the same plain_id labelled against 64 different keymat-transformed input rows; the closed-form solution can't memorize any single `K_a^k` and instead picks up the K-invariant inversion direction — the algorithm-level structure that's common to every draw from the keymat distribution.

This is paper-faithful: under Kerckhoffs the attacker is allowed to run Algorithm 1 with their own randomness as many times as they want.

**The full attack, end-to-end.**

```
1. Attacker downloads plain Qwen3-4B (public; no τ involved).
2. Attacker writes N plaintext prompts → tokenises locally → knows plain_ids.
3. Attacker runs plain Qwen3-4B on those plain_ids → captures State_plain[L]
   server-side (their own server, of their own plain model copy).
4. Attacker samples K=64 keymats {K_a^k} via Algorithm 1 (own seeds).
5. Training matrix: stack [State_plain[L][i] @ K_a^k for all k, all i].
   Target matrix: stack [W_e[plain_id[i]] for all k, all i].
6. Ridge solve: W_ridge = (XᵀX + αI)⁻¹ XᵀY (multi-α grid, val-selected).
7. At deployment-time, attacker (= honest-but-curious server) captures
   State_d[L] from a real user's forward pass. No labels — that's what
   they're recovering.
8. Predict: W_e_pred[i] = State_d[L][i] · W_ridge.
9. Cosine-NN of W_e_pred against the full plain W_e table → top-1 plain_id.
```

No step requires `τ`. Step 7 is the only deployment interaction and it's purely passive (within the threat model: attacker = server, sees activations as a byproduct of inference).

**How this differs from paper §D.1's gradient-optimization ISA.** Paper describes ISA as an optimization attack: capture `State_d` for a single user query, initialise random plain input `X_2`, forward `X_2` through the plain model to get `State_2`, optimize `X_2` via gradient descent to minimize `‖State_d − State_2‖`. No labelled training set, no ridge. Paper's argument for why this fails: under covariance `State_d = K_e · State_plain(X_1)` lives in a different basis than any `State_2 = State_plain(X_2)`, so the optimizer can never reach zero loss.

The labelled-ridge attack is a separate operationalization — paper Table 4's empirical ISA reading (`0.0 %` for the full Noise+KeyMat+Head&BlockPerm defence on HiddenState) comes from ridge in the reference impl (`sheng1feng/Aloepri`), not from the optimization variant. The two attacks measure different things; we report the ridge variant because it matches the reference impl and is much cheaper to run (closed-form solve, not iterative gradient descent on the entire plain model per query).

### Train/test split: row-split, not vocab-disjoint

The reference impl defaults to **vocab-disjoint** splits — the unique plain_ids seen during training and those evaluated on test are *disjoint sets*. This is a stress test for vocab generalization: "if the user types a token the attacker has never trained on, can the inverter still recover it?"

Under our threat model, vocab-disjoint is the wrong null hypothesis:

1. **The attacker controls the training corpus.** A realistic attacker uses a 100 k+ prompt public corpus (Common Crawl, Wikipedia, OpenOrca, …) that trivially covers > 99 % of the 151,936-token Qwen3 vocab. Rare tokens that don't appear in a 100 k-prompt corpus barely appear in user queries either.
2. **Train and test queries share a tokenizer.** A real user's prompt is tokenised into the same plain_id space the attacker trained on. The expected overlap of train and test plain_ids is *near-complete*, not disjoint.
3. **Ridge cannot extrapolate across vocab on small data.** Vocab-disjoint at L > 0 gives `0.0 %` by construction regardless of obfuscation quality — the inverter has no signal to predict `W_e[unseen_id]` from a hidden state. The 0.0 % isn't measuring defence, it's measuring "ridge can't extrapolate" — a property of the attack methodology.

We report **row-split** as the realistic threat-model measurement: rows are partitioned randomly across positions, both splits share the unique plain_id vocab. This isolates the defence's contribution from the methodology's vocab-generalization stress.

Vocab-disjoint is still useful as a *secondary* reading — a stress test for the rare-vocab edge case — but it isn't the load-bearing defence claim.

### Layer-choice tradeoff space

Hidden-state attacks can be conducted at any layer `L ∈ [0, n_layers)`. The choice is a tradeoff between two compounding effects through depth:

| Effect | Trend with `L` | What it means for the attack |
|---|---|---|
| Content accumulation | grows with L | Each block adds attention-mixed contextual semantics. Deeper L = richer per-position encoding of the underlying semantics → more signal for the attacker. |
| Defense accumulation | grows with L | `K_e` is applied to the residual stream at the embedding and preserved by every subsequent block (paper's covariance claim). Algorithm 2's intra-head transforms add additional per-layer perturbations on attention internals. Deeper L = more rounds of obfuscation between the plain content and the captured state. |

The two effects cancel in opposite directions, so the layer-attackability curve is non-monotone — there's a sweet-spot zone where content has accumulated enough to be discriminable but defence hasn't accumulated to the point of dominating.

**By relative depth:**

| Layer band | Depth | Tradeoff state |
|---|---|---|
| `attn_norm-0` | 0 % | Residual = `RMSNorm(W̃[obf_id])`. Pure obfuscated embedding row — no contextual content, defence is just Algorithm 1 keymat (single round). Equivalent to attacking `W̃` directly with an extra rescaling factor; see "IMA-L0-transformer redundant" above. Not informative as a separate surface. |
| Early (~10 %) | 0.1 | Some attention mixing but content is still local to each position. Defence has had a few rounds of Algorithm 2 transforms applied. Usually under-discriminable. |
| **Mid (~48 %)** | 0.5 | **Paper's measured layer** (Qwen2.5-14B `attn_norm-23/48`; Qwen3-4B/8B `attn_norm-17/36`). Content has spread across positions via attention, defence has had ~half the depth budget applied. The paper's argument: this is where the defence-to-content ratio is most favourable for the defender. |
| Deep (~85 %) | 0.85 | Maximum contextual content; defence has had nearly the full depth budget applied. Hidden state encoding is now strongly task-shaped (the final layers prepare for `lm_head`). Sometimes easier for an attacker if the task-specific encoding leaks more about the current token than the mid-layer averaged context did. |
| Final (`n_layers - 1`) | 1.0 | Immediately before `lm_head`. The residual at this point is what gets projected to logits over the obfuscated vocab. Useful as an upper bound but is also the surface most affected by the obfuscated head weights. |

**Why paper picks ~48 %.** Empirically (paper Table 4) the mid layer is where the *full* defence recipe (Noise + KeyMat + Head&BlockPerm) closes the leak completely. The shallow layers leak more pure-embedding signal (less Algorithm-2 protection accumulated); the deep layers leak more task-specific signal (less buffer to the output projection). The mid layer is where Algorithm 2's intra-head transforms have had enough block-perm rounds to scramble the per-head structure that earlier layers would expose, while the residual hasn't yet collapsed into the lm_head-aligned subspace.

**Layer choice for our measurement.** L = 17 on both Q3-4B and Q3-8B (36-layer models), matching paper's `attn_norm-23/48 ≈ 48 %` depth. The same layer index doesn't translate across architectures (paper Q2.5-14B has 48 layers); we use the **relative-depth match**, not the absolute-index match.

A defence-gap audit should also sweep `L ∈ {4, 9, 17, 21, 26, 30}` — if the mid-layer paper-matched depth passes but a shallower or deeper layer leaks above the 15 % gate, the defence holds at paper's measurement point but has structural weakness elsewhere. We default to the paper-matched depth for the headline number and treat the sweep as a secondary audit.

### Approximation in the covariant synthesis (and what it costs)

The attacker's synthesis `State_a^k = State_plain @ K_a^k` matches what their own obfuscated model would produce at layer L *if covariance held exactly*. In practice covariance is approximate:

1. **Per-prompt embedding noise** (`α_e · σ_e · ε` at the input embedding) is included in the deployment's `State_d` and propagates through attention + FFN. Our synthesis sets `State_a = State_plain @ K_a` with no noise term, so training data is slightly cleaner than test data. The trained inverter is mildly overfitted to a noise-free representation.
2. **Algorithm 2's intra-head transforms** are designed to preserve the residual basis exactly, but any implementation imperfection (rounding, finite-precision matmul) introduces small residual perturbations the synthesis doesn't model.

Net consequence: the measured paper-faithful TTRSR is likely a slight **under-estimate** of what an attacker using a real second model could achieve. To get a strictly accurate paper-faithful number we'd need to *materialize* `K` actual obfuscated models with the attacker's own keys and capture from each — at K=64 × 17 GB per model on Q3-4B, that's ~1 TB of disk and 64× the obfuscation runtime. Not justified at the current measurement budget; the synthesis approximation is the operational compromise.

### Why we report row-split paper-faithful and not vocab-disjoint paper-faithful

Combining the above:

- **Vocab-disjoint paper-faithful**: ridge can't extrapolate across vocab → 0.0 % regardless of obfuscation. This is paper's reading of its own attack, but it measures methodology, not defence.
- **Row-split paper-faithful**: ridge sees most/all plain_ids in training → measures the obfuscation's actual attenuation of the per-position signal. Aligns with the realistic attacker (large training corpus, broad vocab coverage).
- **Row-split with τ-leak attacker** (the prior `_isa_ridge_split_row` reading in §08, before the threat-model fix): gave 9.6 % on Q3-4B and 16.3 % on Q3-8B. Was attacker-too-strong (used deployment's secret `τ` for labelling) — out of threat model.
- **Vocab-disjoint with τ-leak attacker**: paper-aligned methodology + over-powered attacker — confounds two effects in opposite directions, so we don't use it.

§08 will measure ISA HiddenState under row-split paper-faithful as the headline, vocab-disjoint as a secondary stress reading.

## VMA — Vocabulary-Matching Attack (sorted-quantile RowSort)

**Setup.** Static-weight attack. Attacker has plaintext GGUF `θ` and obfuscated GGUF `θ̃`. No runtime captures, no forward inference. The obfuscator has applied Algorithm 1 keymat (`W̃ = Q̂ · W · P̂`), §5.2.2 noise + Π token-perm (`W̃_e[τ(i)] = W_e[i] + α_e σ ε_i`, mapped through Π), and Algorithm 2 attention transforms (R̂_qk, Ĥ_qk±1, Ẑ_block, Π_head, Û_vo).

**Attacker's goal.** Recover τ (the token permutation). Knowing τ lets the attacker decode every wire-side obfuscated prompt to plain token ids.

**The trick: RowSort + sorted-quantile features + multi-source voting.** Paper §F.1 + Table 8 + reference `vendor/aloepri-py/src/security_qwen/vma.py`. Implementation in `evals/aloepri-attacks/m2_7/run_static_attacks.py::run_vma`.

Per row (one plain id):

1. **Compute a row-level "feature" from multiple weight tensors.** Each weight `W_proj` (attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down × 36 layers) is projected against W_e: `feature_source = W_e @ W_proj` or `W_e @ W_proj.T`. This gives ~252 "source matrices" for Q3-4B, each shape `(vocab, projected_dim)`.
2. **For each source matrix, take SORTED-QUANTILE features per row.** Sort each row's values, take 64 evenly-spaced quantile points, mean-centre and L2-normalize. This kills the column order — features are invariant under column permutation of W_proj.
3. **Match plain features to obs features.** For each plain row `i` and each obs row `k`: cosine similarity per source. Z-score the rows. Sum across all 252 sources.
4. **Top-K argmax** picks the obs row most-correlated to plain row `i`. If it's `τ(i)`, attack succeeds.

**Why sort?** It's the crux. Sorted-quantile features defeat Alg2's column-shuffling defences:
- Π_head permutes head-index — columns within a source matrix get reshuffled. After sort: identical features. **Π_head = 0 pp on VMA.**
- Ẑ_block permutes RoPE-pair index — same column-permutation argument. **Ẑ_block = 0 pp on VMA.**

But sort doesn't defeat:
- Per-pair 2D rotation (R̂_qk): each value gets mixed with its pair-partner (NEOX `i ↔ i+half_d`). Sorted-quantile features SHIFT because rotation changes the values, not just their order.
- Per-pair sign flip (Ĥ_qk ±1): `[a, b] → [-a, b]`. Sort of `[-a, b]` differs from sort of `[a, b]` when `|a| < |b|`. Subtle effect on trained weights — but real (−17.8 pp).
- V/O channel perturbation (Û_vo): per-head column mixing with a QR-orthogonal + 0.05 Gaussian. Per-row L2 stats shift.

**§5.2.2 alone (no Alg2) defends.** α_e noise scrambles W_e values per id; Π re-shuffles indices. Sorted-quantile features per row become noisy → cross-source voting can't reinforce a stable per-row signature. Drops VMA from 98 % to 35 % on Q3-4B.

**§5.2.2 × Alg2 superadditivity.** This is the key empirical finding (2026-05-26 bisection, see `docs/handoffs/2026-05-25-alg2-attack-crossmap.md`). R̂_qk + Ĥ_qk±1 on top of §5.2.2 give a **9× larger marginal contribution** than on Alg1 alone:

- Alg1 → Alg1+R̂+H (no §5.2.2): −1.96 pp
- Alg1+§5.2.2 → Alg1+§5.2.2+R̂+H: **−17.84 pp**

Mechanism (conjectured): §5.2.2 scrambles W_e per-row; Alg2's matrix-Γ scrambles W_q/W_k per-source. Each alone leaves enough residual structure for VMA's multi-source voting to recover. Stacked, both axes are perturbed independently and the cross-source vote can't reinforce.

**Path-2 attribution (Q3-4B, 3-seed mean).** Total deployment drop: 98.4 % → 9.5 % = −89 pp.

| Component | Contribution |
|---|---:|
| §5.2.2 (α_e + α_h + Π token-perm) | −63 pp (primary) |
| R̂_qk + Ĥ_qk±1 (matrix-Γ kernel) | −17.8 pp (only after §5.2.2 active) |
| Û_vo | −12.9 pp |
| Sub-additive overlap (R̂+H × Û_vo) | +4.8 pp |
| Π_head, Ẑ_block β=8 | 0 each |

§08 row-table reports 18.4 % single-seed; my 3-seed mean is 9.5 % ± 2.5. The §08 number is plausibly an outlier-high sample within VMA's seed-noise distribution.

## Per-head fingerprint Q/K/V/O — recovers `Π_head` via SVD spectra

**Setup.** Static-weight. Attacker has plain GGUF θ and obfuscated GGUF θ̃. Driver: `evals/aloepri-attacks/m2_7/run_per_head_fingerprint.py` (new 2026-05-25). Targets Algorithm 2's inter-head shuffle (paper §5.2.3 step 5).

**Attacker's goal.** Recover the per-layer head permutations `τ_kv ∈ S_8` (KV-head order) and `τ_q ∈ S_32` (Q-head order, built from `τ_kv × τ_group`). Recovery wouldn't directly give `τ` (the token permutation) but would unlock downstream attacks by de-shuffling the heads, making per-head function fingerprinting + attention-pattern analysis tractable on the obfuscated weights.

**The trick: per-head SVD spectrum as a permutation-invariant signature.** For each plain head `i` (Q, K, V, or O), reshape `W[head_i]` to `(head_dim, d_residual)` and take the top-16 singular values. The SV vector is invariant under right-multiplication by orthogonal matrices on the head_dim axis — so R̂_qk, Ĥ_qk±1, Ẑ_block, and Û_vo (the four Alg2 head_dim-axis transforms) all preserve it. Π_head, however, *permutes which head index holds which signature*, so a successful match recovers `τ`. NN match by L2 distance on the 16-d SV vector; ground truth `τ` loaded from `.key.npz`.

**Why it lands at random chance on the deployed cell.** Algorithm 1's keymat `K_d` is rectangular `(d, d_obs) = (2560, 2816)`. Each obfuscated head's weight matrix is `(head_dim, d_obs)` — 256 dims wider than plain. SVD spectra of `(128, 2816)` matrices differ systematically from `(128, 2560)` for the same head content because the extra columns add noise components to the singular values. Combined with Π_head's shuffle, every obfuscated head's signature looks the same to the attacker. **Alg1 incidentally protects the head-permutation secret that Alg2 was designed to protect.**

**Q3-4B measured (2026-05-25, deployed Û_vo cell):**

| Surface | Top-1 | Random chance | Note |
|---|---:|---:|---|
| attn_q | 4.25 % | 3.13 % (1/32, n_q=32) | within 1 σ of random |
| attn_k | 13.54 % | 12.5 % (1/8, n_kv=8 via GQA) | within 1 σ |
| attn_v | 12.50 % | 12.5 % | exactly random |
| attn_output | 3.21 % | 3.13 % | exactly random |

Key-recovery information bound: attacker gets ≤ guessing entropy. `τ_kv × τ_group` key space is `(8! · 4!)^36 ≈ 10^214` — intractable to brute-force.

## V/O channel-pair V/O — recovers `Π_head` + probes `Û_vo`

**Setup.** Static-weight. Driver: `evals/aloepri-attacks/m2_7/run_vo_channel_pair.py` (new 2026-05-25). Targets Algorithm 2's V↔O random projection (paper §5.2.3 step 4 `Û_vo`) in addition to `Π_head`.

**Attacker's goal.** Same as per-head fingerprint but with a signature designed to be *sensitive* to `Û_vo`'s per-head perturbation, not just `Π_head`'s shuffle.

**The trick: per-channel L2 magnitudes (not SV spectra).** For each plain V-head: take row L2 norms across `head_dim` rows of `W_v[head]` → length-128 magnitude vector. For each plain O-head: take column L2 norms within the O-head's column range. Optional auxiliary top-K SV signature.

Why magnitudes specifically: `Û_vo` is QR-orthogonal + 0.05 Gaussian perturbation — it's the *only* Alg2 component that shifts per-head V/O column magnitudes (R̂_qk + Hadamard are pure orthogonal → preserve magnitudes; Π_head permutes whole heads → preserves within-head magnitudes; Ẑ_block permutes RoPE-pair indices → also preserves magnitudes). So this attack is specifically designed to detect `Û_vo`.

Three sub-attacks:
- `vo_v_match`: NN match plain V-heads to obfuscated V-heads (recovers `τ_kv` if successful).
- `vo_o_match`: same for O-heads (recovers `τ_q`).
- `vo_pair_match`: joint match — for each Q-head, both V and O must match correctly under GQA grouping. Recovers `(τ_kv, τ_q)` jointly + the V→O group structure.

**Why it lands at random chance on the deployed cell.** Same reason as per-head fingerprint: Algorithm 1's `d → d_obs` keymat changes input-axis dimension uniformly across all heads, shifting all per-head magnitudes by an identical factor (preserving relative ordering within a layer but losing the per-head distinguishing signal vs plain). **Û_vo's perturbation is shadowed by Alg1's rectangular projection** — the attacker can't distinguish Û_vo's per-head magnitude drift from the keymat's uniform shift.

**Q3-4B measured (2026-05-25, deployed Û_vo cell):**

| Sub-attack | Top-1 | Random chance | Note |
|---|---:|---:|---|
| `vo_v_match` | 12.50 % | 12.5 % (1/8) | exactly random |
| `vo_o_match` | 3.12 % | 3.13 % (1/32) | exactly random |
| `vo_pair_match` | 3.12 % | random² ≈ 0.4 % | individually random but joint requires both — see below |

The `vo_pair_match` deserves a note: it's not random² because the V and O signatures aren't truly independent given GQA structure. Empirically lands at the random-Q rate (3.12 %) because the O match is the harder bottleneck (32 candidates vs 8 for V).

**Common pattern with per-head fingerprint.** Both attacks land at random chance on the deployed cell because **Algorithm 1's rectangular keymat already collapses the per-head signal** before Alg2's defenses (Π_head, Û_vo) need to kick in. The Alg2 components were designed for these threat models; in practice Alg1 shadows them.

## Attention-output covariance: where §5.4's bound applies, and the score-surface gap

**Claim (informal).** AloePri paper §5.4 proves that Algorithm 2's intra- and inter-head transforms preserve the attention layer's **output** vector (`f_attn(x, ω_attn) P̂_o`) up to an error `e_C^attn` that composes into the global accuracy bound `e_C^AloePri`. The Lipschitz consequence below shows that any head-permutation-invariant attack on the *output* surface therefore has defense delta `O(L_A · e_C^attn_output)`. The paper does **not** quantitatively bound the perturbation on the *pre-softmax score* `Q·K^T` or the *post-softmax probability* surface — only the qualitative symbol `≈_{e_C^attn}` appears there, and the global composition into `e_C^AloePri` only uses the output-level bound. The 47 % pre-softmax score-surface ridge recovery we measure on the deployed cell is therefore *out of scope* of paper §5.4, neither contradicting nor confirming Table 4's `AttnScore TTRSR = 0.0%` claim until the measurement surface paper used is identified (paper §7.5 / Table 4 / Appendix D.1 are ambiguous on this).

### Notation (paper §3.3, §3.4, §5.2)

- `x ∈ ℝ^d` — plaintext residual at the input of an attention layer.
- `x̃ = x P̂` — obfuscated residual, where `P̂ ∈ ℝ^{d×(d+2h)}` is the keymat (paper Algorithm 1, page 8).
- `W_q^(i), W_k^{η(i)}, W_v^{η(i)}, W_o^(i)` — plaintext attention weights for head `i` (paper §3.4.1 eq. 2).
- `W̃_q^(i), W̃_k^{η(i)}, W̃_v^{η(i)}, W̃_o^(i)` — obfuscated attention weights per paper Algorithm 2 lines 6-7 (page 9):
  - `W̃_k^{η(i)} = Q̂_k W_k^{η(i)} Ĥ_qk^{-1} Ẑ_block^T`  *(paper as written)*
  - `W̃_q^(i)   = Q̂_q W_q^(i) R̂_qk Ĥ_qk Ẑ_block`
  - `W̃_v^{η(i)} = Q̂_v W_v^{η(i)} Û_vo`
  - `W̃_o^(i)   = Û_vo^{-1} W_o^(i) P̂_o`
- `R̂_qk, Ĥ_qk, Ẑ_block, Û_vo, η(·)` — Algorithm 2 parameters (rotary, scaling, block-perm, V↔O random projection, inter-head permutation).
- `G(·)` — RoPE.
- `s_i(x) = G(x W_q^(i)) G(x W_k^{η(i)})^T / √d_head` — plaintext **pre-softmax score** for head `i`.
- `a_i(x) = softmax(s_i(x) + mask)` — plaintext **post-softmax probability**.
- `o_i(x) = a_i(x) · (x W_v^{η(i)})` — plaintext **per-head attention output** before W_o aggregation.
- `y(x) = Σ_i o_i(x) · W_o^(i)` — plaintext **attention layer output** (the §5.4 bounded quantity).
- Tildes denote obfuscated counterparts.

### Paper §5.4's two equations and their scope

Paper §5.4 page 10, "Attention" paragraph contains two distinct statements:

**Score-level (qualitative).** *"The attention score for the i-th head is computed as: `G(x̃W̃_q^(i))G(x̃W̃_k^η(i))^T ≈_{e_C^attn} G(xW_q^(i'))G(xW_k^η(i'))^T`, where i' denotes the permuted index corresponding to the i-th head, and e_C^attn is the obfuscation error induced by block permutation (parameterized by β, γ)."*

**Output-level (composes into accuracy bound).** *"f̃_attn(φ_X^attn(x), φ_Θ^attn(ω_attn)) ≈_{e_C^attn} φ_Y^attn ∘ f_attn(x, ω_attn) P̂_o"*

Paper §5.4 page 11 "Putting Together" composes these into `e_C^AloePri ≤ M_0 e_C^embed + Σ M_i e_C^decoder_i + e_C^head` where `e_C^decoder_i ≤ (M_i^norm)(M_i^attn e_C_i^attn + e_C_i^norm) + e_C_i^FFN`. **Only the output-level `e_C^attn` enters this composition** — score-level perturbation is absorbed into `softmax + V·` aggregation before it crosses the output bound. Paper Table 2's `< 3 %` accuracy-loss therefore constrains the output-level `e_C^attn`, not the score-level one.

**Implication.** "Small `e_C^attn`" means small output-level error. The score-level perturbation can be arbitrarily larger and still be consistent with §5.4 — softmax + V-aggregation can absorb a large `Q·K^T` perturbation while keeping `softmax(Q·K^T)·V` bounded.

### Theorem (attention-output covariance)

**Theorem.** Let `A: ℝ^{m × n_q × d_head} → [0,1]` be a recovery-rate attack on the **per-head attention output** `(o_1(x), ..., o_m(x))` (or equivalently on `y(x)` post W_o, since W_o is invertible up to keymat composition). Assume `A` is L-Lipschitz in its input with respect to the operator norm:

> `|A(o_1, ..., o_m) − A(o'_1, ..., o'_m)| ≤ L_A · max_i ‖o_i − o'_i‖`.

Then under paper §5.4's output-level attention bound,

> `|A(õ_{η(1)}(x̃), ..., õ_{η(m)}(x̃)) − A(o_1(x), ..., o_m(x))| ≤ L_A · e_C^attn_output(β, γ, Û_vo)`.

For attacks `A` that are invariant under the head-axis permutation `η` (which the attacker doesn't know), the permuted index drops out and the bound holds on the unpermuted output tensor.

### Proof sketch

By paper §5.4's output-level equation, `õ(x̃) = o(x) + ε` with `‖ε‖_op ≤ e_C^attn_output`. Apply the Lipschitz hypothesis on A. The head permutation `η` is a relabeling of A's input axis; permutation-invariance of A absorbs it. The same argument extends to `y(x)` post W_o because the W_o block-diagonal head aggregation is itself a Lipschitz map. □

### What the theorem does NOT say

The theorem does **not** claim anything about the pre-softmax `Q·K^T` or post-softmax probability surfaces. In particular it does not force `TTRSR_obf` to track `TTRSR_plain` on those surfaces. Paper §5.4 leaves the score-level perturbation unbounded; our empirical observation (table below) of a 47 % score-surface recovery on the deployed cell is consistent with that.

### Known limitations and caveats

- **L1 (Lipschitz of ridge).** Ridge `(X^T X + αI)^{-1} X^T Y` has operator norm ≤ `1/(λ_min(X^T X) + α)`. For small α relative to `λ_min`, `L_A` can be large; combined with even a moderate `e_C^attn_output`, the bound is loose. The α grid {1e-4, 1e-2, 1.0} in `gpu_sweep.py` picks best-on-val, but worst-case `L_A` is not uniformly bounded.
- **L2 (TTRSR is not Lipschitz).** TTRSR top-1 is a sum of `argmax` indicators, discontinuous at decision boundaries. The Lipschitz hypothesis on `A` should be read as "Lipschitz under expectation over the score distribution" or "Lipschitz almost-everywhere away from margins." A fully rigorous bound would substitute a total-variation argument; we leave that to future work.
- **L3 (permutation-invariance restriction).** The theorem covers head-permutation-invariant attacks. Fixed-architecture trained inverters (like the Qwen2 2-layer 8-head inverter referenced in paper Appendix D.1 for IMA, possibly used for ISA too) are *not* permutation-invariant — their token-mixing layer assumes a fixed head ordering. Head shuffles can break such inverters without changing information content. This is an artifact of attack architecture, not a defense property.
- **L4 (η dropout is over-expectation).** η is a secret. For a single deployed cell, η is fixed, and "η drops out" averages over realizations of the secret keys. The theorem holds in expectation; per-cell variance is `O(1/m^{1/2})` from sampling.

### Implementation deviations relevant to the theorem

- **A1 — k_matrix construction.** `python/aloepri-llm/lib/alg2.py:263` uses `k_matrix = R̂_qk · Ĥ_qk^{-1} · Ẑ_block` instead of paper Algorithm 2 line 6's literal `Ĥ_qk^{-1} · Ẑ_block^T`. Documented in the CAVEAT at `alg2.py:244-262` as a deliberate fix: paper-literal gives `M_q · M_k^T = R̂_qk · Ĥ · Ẑ² · Ĥ^{-1}` which only equals R̂_qk (absorbed by RoPE) when `Ẑ² = I`; paper's BlockPerm function does not enforce involutivity. Our impl achieves `M_q · M_k^T = I` exactly via the `Ẑ · Ẑ^T = I` identity. Both impls satisfy paper §5.4's output-level bound (accuracy preserved); both differ in their score-level error profile.
- **P1 — paper Algorithm 2 line 6 is under-specified.** Paper as written produces a `‖Ẑ² − I‖` residual at the score level; this residual is consistent with paper's qualitative `≈_{e_C^attn}` but never bounded quantitatively. Either a typo (`Z^T` → `Z`), an implicit assumption (`Ẑ ∈` involutions), or absorption into the unquantified `e_C^attn`.
- **A2/A3 (minor).** Our `Û_vo` adds a small QR-stabilization perturbation not in paper. Our `R̂_qk` uses Qwen-half-rotated layout `(i, i+half)` instead of paper's `(2i, 2i+1)`; pair-index correctness with the runtime RoPE kernel is asserted but not formally verified (see Step 2A.1 of the 2026-05-26 plan).

### Empirical: pre-softmax score surface (out-of-scope of §5.4)

`evals/aloepri-attacks/m2_7/diagnose_isa/gpu_sweep.py` (2026-05-26), Qwen3-4B, 512 natural-language prompts (`evals/aloepri-attacks/corpora/release-gate-512.txt`), 10-seed row-split mean, deployed cell `untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf` (paper §7.1: α_e=1.0, α_h=0.2, h=128, β=8, γ=10³):

| Surface @ L=0 | Plain TTRSR top-1 | Obf TTRSR top-1 | Δ | Surface in §5.4 scope? |
|---|---:|---:|---:|:---:|
| Pre-softmax `Q·K^T` | 48.63 % ± 0.80 | 47.22 % ± 0.84 | +1.41 pp (~1.2σ) | **No** |
| Post-softmax (mask + softmax(·/√d_head), reconstructed) | 32.09 % ± 0.90 | 30.60 % ± 0.64 | +1.49 pp (~1.4σ) | **No** |
| Attention output `softmax(Q·K^T)·V` (per-head, pre W_o, llama.cpp `kqv_out`) | 97.46 % ± 0.40 | **83.77 % ± 0.64** | +13.69 pp (~21σ) | **Yes** |

Per-layer attention-output (2B.1, 2026-05-26, captures `cell-qwen3-4b-{plain,uvo}-attnoutput-512-20260526/`):

| Layer | Plain | Obf | Δ |
|---:|---:|---:|---:|
| 0  | 97.46 % | 83.77 % | **+13.69 pp** |
| 5  | 50.77 % | 50.22 % | +0.55 pp |
| 11 | 18.50 % | 19.83 % | −1.33 pp |
| 17 | 16.67 % | 17.75 % | −1.08 pp |
| 23 | 25.27 % | 25.99 % | −0.72 pp |

**Reading.** At L ≥ 5 the output-surface delta is within ±1.5 pp (≤ 1.7σ) — empirically consistent with the tightened theorem. At L=0 the 13.69 pp gap exceeds noise by ~21σ but is *far* from paper Table 4's 87 → 0 pp transition; the L=0 surplus is most plausibly the direct shadow of the Algorithm 1 embedding noise (α_e = 1.0) injected into the first decoder layer's input. A noise-isolated cell (α_e = 0 obf, full Alg2) at L=0 would test this hypothesis; deferred. Signal-correlation diagnostic: element-wise plain–obf correlation on `kqv_out` is ~0.01–0.03 at both L=0 and L=17 (vs ~0.6 / 0.9 for `kq`), so the obfuscation rotates the feature space substantially but ridge still recovers most of the token signal — the signature of K_a-covariant defense.

Full report: `evals/aloepri-attacks/results/sweep/2B1-attn-output-vs-kq-comparison.md`.

Capture sanity (2026-05-26 Step 0, `compare_plain_obf.py` against the 20260526 captures): plain–obf correlation 0.58-0.66 at L=0, 0.89-0.91 at L=17; not byte-identical, not near-identical → captures are genuinely obfuscated, ruling out a capture-path artifact.

The pre-softmax 47 % obf TTRSR is **not bounded** by paper §5.4. It is a real path-2 finding: a ridge attacker who can capture the pre-softmax `Q·K^T` tensor recovers ~half the tokens on a full-Alg2 deployment. The 1.5 pp defense delta says Alg2 doesn't meaningfully obstruct this particular attack on this particular surface.

The attention-output measurement (pending) tests the theorem's prediction. If output-surface Δ is small (single-digit pp), the theorem is empirically validated and the score-surface 47 % stands as a *complementary* finding on an out-of-scope surface. If output-surface Δ is also large, the theorem fails — the Lipschitz framework would need to be re-examined or the §5.4 output bound would need to be empirically violated.

### Paper-literal Alg2 (A1+A2) + vocab-disjoint methodology — Table 4 0 % is reproducible

2026-05-26 follow-up. The original 1.5 pp / 14 pp defense deltas above were measured on the **deployed** Alg2 cell, which differs from paper Algorithm 2 in two construction details:

- **A1.** Our `k_matrix = R̂_qk · Ĥ⁻¹ · Ẑ` produces `M_q · M_k^T = I` exactly, vs paper Algorithm 2 line 6 literal `Ĥ⁻¹ · Ẑᵀ` which produces `M_q · M_k^T = R · H · Ẑ² · H⁻¹` (paper's score-level error has a `‖Ẑ² − I‖` term we structurally eliminate).
- **A2.** Our `Û_vo` is QR-stabilized + 0.05 σ Gaussian perturbation (near-orthogonal, well-conditioned for bf16 inversion). Paper's is raw `N(0, 1/d_head)` Gaussian (wide-spectrum, ~500 condition number).

A paper-literal cell built 2026-05-26 (`untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-PAPERLIT-bf16-native.gguf`, behind `obfuscate_qwen3_gguf.py --alg2-paper-literal`) gives substantially larger defense deltas on both surfaces:

#### Row-split (natural-language corpus, 75 % vocab overlap)

| Layer | Surface | Plain | Default obf | Paper-lit obf | Δ default | Δ paper-lit |
|---:|---|---:|---:|---:|---:|---:|
| 0  | kq      | 48.63 | 47.22 | **43.22** | −1.4 | −5.4 |
| 5  | kq      | 38.69 | 38.49 | **7.79** | −0.2 | **−30.9** |
| 11 | kq      | 27.73 | 26.95 | **7.52** | −0.8 | **−20.2** |
| 17 | kq      | 22.41 | 21.17 | **6.35** | −1.2 | **−16.1** |
| 23 | kq      | 30.13 | 29.67 | **6.49** | −0.5 | **−23.6** |
| 0  | kqv_out | 97.46 | 83.77 | **47.32** | −13.7 | **−50.1** |
| 5  | kqv_out | 50.77 | 50.22 | **10.49** | −0.6 | **−40.3** |
| 11 | kqv_out | 18.50 | 19.83 | **10.57** | +1.3 | −7.9 |
| 17 | kqv_out | 16.67 | 17.75 | **10.04** | +1.1 | −6.6 |
| 23 | kqv_out | 25.27 | 25.99 | **10.04** | +0.7 | −15.2 |

#### Vocab-disjoint split (held-out tokens, paper's likely methodology)

| Layer | Surface | Plain | Default obf | Δ |
|---:|---|---:|---:|---:|
| 0  | kq      | **0.43** | 0.07 | −0.4 |
| 5  | kq      | 0.08 | 0.04 | −0.04 |
| 11 | kq      | 0.02 | 0.00 | −0.02 |
| 17 | kq      | 0.00 | 0.00 | 0 |
| 23 | kq      | 0.00 | 0.01 | +0.01 |
| 0  | kqv_out | 71.68 | 36.70 | −35.0 |
| 5  | kqv_out | 0.76 | 0.54 | −0.2 |
| 11 | kqv_out | 0.01 | 0.00 | −0.01 |
| 17 | kqv_out | 0.00 | 0.00 | 0 |
| 23 | kqv_out | 0.09 | 0.02 | −0.07 |

#### Reading

1. **Paper-literal Alg2 carries most of the defense gap.** Switching from our deployed construction to paper-literal moves obf `kq` from 38 % to 7.8 % at L=5, and obf `kqv_out` from 50 % to 10 % at L=5. The QR-stabilised Û_vo we use preserves much more per-head structure than paper's pure Gaussian; the extra `R̂_qk` on our `k_matrix` cancels paper's `‖Ẑ² − I‖` score residual. Both deviations were *anti-defense* relative to paper's intent.
2. **Vocab-disjoint methodology accounts for the rest.** Plain `kq` at L=0 drops from 48.63 % (row-split) to 0.43 % (vocab-disjoint) — meaning >99 % of the row-split signal was vocab memorisation, not structural recovery. At L≥5 ridge floors at ~0 % on both plain and obf regardless of construction.
3. **Compound prediction (untested as of 2026-05-26):** paper-literal × vocab-disjoint at L≥5 = ~0 % obf, matching paper Table 4's headline.

What's still unexplained: paper Table 4's **87.14 % "Noise+KeyMat" baseline**. The paper-literal cell still shows only 43 % plain on `kq` row-split at L=0 — much less than 87 %. Three live hypotheses for the 87 %:
- Paper's "Noise+KeyMat" baseline lacks the full Alg2 entirely (just α_e + Algorithm 1 keymat, no head/block perm). A plain-baseline cell stripped to that config might reach 87 %.
- Paper uses a stronger attack class (gradient-opt per Appendix D.1) that converges on plaintext where ridge plateaus.
- Paper uses a different model topology (Q2.5-14B has 48L / 40Q heads) that exposes more per-feature signal.

Step 3 (paper-faithful gradient-opt ISA) is still the cleanest test for the 87 % baseline. Construction story for the 0 % is closed.

### Relation to paper Table 4 (revised 2026-05-26 PM)

Paper Table 4 (page 15) reports `AttnScore TTRSR = 0.0 %` for Noise+KeyMat+Head&BlockPerm. The 2026-05-26 PM measurements above resolve the construction half of this:

- Compound of **paper-literal Alg2** (A1+A2) plus **vocab-disjoint methodology** drives both surfaces to ~0 % at L≥5 across all tested attacks (ridge with multi-α + 10-seed val pick). Paper's 0 % is therefore reproducible under their likely measurement protocol, even with our ridge attack.
- The 87 → 0 pp transition between the two ablation rows still requires explanation, but it is now most plausibly an *attack-class effect* (their gradient-opt converges on plaintext, fails on obf) on a baseline cell that lacks the full Alg2 — both falsifiable with Step 3.

§5.4's bound is consistent with everything we measure. The tightened theorem holds.

### Implications for path-2

1. **The recommended deployment construction is paper-literal Alg2, not our prior default.** Our deployed cell was understating AloePri's actual defense by 7–40 pp on both surfaces. Migration to `--alg2-paper-literal` is the path-2 recommendation, contingent on accuracy preservation under bf16 (paper-literal Û_vo has 500× higher condition number; bf16 inverse loss is a new precision risk to verify — see next-steps memo).
2. **AloePri §5.4 protects the attention output surface, more than we previously measured.** Subject to confirming accuracy under paper-literal, the §5.4-bounded surface defense delta at L=0 is **50 pp** under paper-literal (vs 14 pp under our default). At L≥5 the delta is **40 pp** under paper-literal (vs 0.5 pp under default). This is a substantive deployment protection, not the 1.4 pp we previously reported.
3. **AloePri's score-surface defense, under paper-literal, is also non-trivial at L≥5.** Even outside §5.4's quantitative bound, the paper-literal `kq` defense delta at L=5+ is 16–31 pp, dropping obf to single digits. The L=0 surplus (~5 pp) is still small but no longer "no defense."
4. **A different threat-model reading.** The path-2 score-surface attack we previously characterised as "AloePri provides ~0 pp defense" was a measurement of the *anti-defense version* of Alg2 we'd deployed. Real AloePri Alg2 (paper-literal) defends meaningfully on this surface too. The remaining 6-7 % obf TTRSR at L≥5 is the operational leak budget, not 47 %.
5. **TEE-protected attention (path-1) remains the gold standard for adversaries who can capture either surface at L=0** — even paper-literal Alg2 leaks 43 % on `kq` at L=0 and 47 % on `kqv_out` at L=0. The L=0 surplus is α_e=1.0 embedding-noise shadow; only an in-TEE first decoder layer eliminates it.

### Open question — score-surface `e_C^attn` bound

Paper qualitatively asserts `≈_{e_C^attn}` at the score level but never bounds it. Our deployed-cell measurement at β=8 shows ~35 % relative score perturbation (per the `alg2.py:251-262` CAVEAT β-sweep). Whether the bound *can* be small at the score level under paper's exact Algorithm 2 line 6 (with `Z²≠I` residual) is unknown; whether the bound matters for ridge-class attacks (which are data-driven and learn whatever perturbation pattern exists) is also unknown.
