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
