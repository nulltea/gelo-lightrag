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
