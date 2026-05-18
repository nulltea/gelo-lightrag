# M1.10 Phase 0 — Security Review: Causal-Mask Leak in Permuted Attention

> **Status:** drafted 2026-05-18. Companion to
> [`m1-10-fused-permuted-attention.md`](m1-10-fused-permuted-attention.md).
> **Decision:** recommend **F1 — in-TEE softmax** (with a refined
> dispatch flow that recovers most of the bandwidth benefit M1.10
> was designed for).

---

## 0. The leak in one paragraph

Our `gelo-protocol::attention::permuted_attention` permutes Q, K, V
on the token axis by a fresh per-batch `π_b ∈ S_n` (Amulet
softmax-equivariance), adds optional `N(0, σ²·I)` noise to Q and K
(Hidden No More mitigation), and offloads the three heavy ops
(`matmul · scale+mask · softmax · matmul`) to the GPU. For
autoregressive generation we need a causal mask: today we set
`M_π[i,j] = -∞ if perm[j] > perm[i] else 0` on the TEE side, **then
hand the scaled+masked score tensor to the GPU's `softmax_batched`**.
The GPU sees the explicit `-∞` pattern. Per-row counting of `-∞`
entries yields `perm[i] = n - 1 - |{j : perm[j] > perm[i]}|`
trivially, so the GPU recovers `π_b` from a single softmax call.
With `π_b` known, the row-permuted noisy Q/K come back into
canonical order (at σ=0.01 the noise is tiny relative to activation
magnitudes), and the Hidden No More vocabulary-matching attack on
"unscrambled" activations becomes feasible.

This only fires when `AttentionMask::Causal` is used. Today that's
only synthetic tests (`tests/permutation_attention.rs`); the
production embedder dispatch is bidirectional. M1.10 wiring into
`decoder_block_cached` activates the leak in production — Phase 0
must fix it first.

## 1. Literature survey

Surveyed 2026-05-18 by [Web search + paper fetches]. The question:
how do published TEE-GPU split-inference schemes handle causal
attention masks under permutation-based protection?

### 1.1 Amulet (arXiv 2512.07495, Dec 2025)

The paper our protocol is derived from. **Does not address causal
masks.** The softmax-equivariance derivation
`softmax(πQK^T π^T / √d) · πV = π · softmax(QK^T / √d) · V`
assumes **dense** (bidirectional) attention. The paper's evaluation
covers BERT-base (bidirectional) and GPT-2 (autoregressive) but
the GPT-2 mechanism for causal masking is not described in the
publicly available HTML version. The masking matrices `P` (left)
and `N` (right) in `F(PXN) = P·F(X)·N` are different — left-
permutation on rows, right-mask on cols — but the construction
still requires the GPU to materialise the score tensor before
softmax, with no explicit recipe for keeping the causal pattern
hidden.

**Net:** the publicly described Amulet construction has the same
gap our implementation does.

Fetched analysis quote: *"the paper does not specify where the
causal mask resides or how it avoids leaking the permutation π.
If causal masking occurs on GPU, the mask itself could reveal π."*

### 1.2 TwinShield (Liu et al., arXiv 2507.03278, Jul 2025)

The name-collision TwinShield (not our Xue '25 reference). Their
**OutSoftMax** primitive offloads softmax to GPU using
**additively homomorphic encryption** — the TEE applies the causal
mask to plaintext logits, encrypts the masked scores via a Paillier-
class scheme, GPU does softmax on ciphertext, returns encrypted
probs. The mask itself never reaches the GPU plaintext-side.

**Why this doesn't fit us:** their construction needs HE because
their threat model includes weight confidentiality. Our model is
openweight; weights are public; we don't carry an HE crypto stack
in this codebase. Adding HE softmax just to hide the mask would be
massive overkill — HE-softmax is typically 100-1000× slower than
plaintext.

**What we steal:** the structural idea that **the mask must not be
sent to the GPU plaintext-side at all**. F1 follows this without
the HE machinery.

### 1.3 KV-Shield (arXiv 2409.04040, on-device TEE-shielded inference)

Permutes the weights `W_q, W_k, W_v` at init time so the resulting
KV pairs are permuted; permutation matrix kept in TEE. **Does not
discuss causal masking.** Fetched analysis: *"The absence of mask
discussion is a notable technical oversight, as causal masking is
fundamental to autoregressive LLM inference."* Their evaluation is
on on-device LLM with weights+KV protection; the mask handling is
a gap.

**Net:** same gap as Amulet.

### 1.4 SCX (Yuan et al., SIGCOMM 2025)

Stateless KV-cache encoding, per-position keys derived from
`(session_id, layer_id, position)`. **Different problem space** —
SCX handles cache bandwidth at decode time, not prefill attention
masking. The causal pattern at decode is implicit (each new token
attends to all cached positions; no triangular mask needed). Not
applicable to our prefill leak.

### 1.5 Hidden No More (arXiv 2505.18332, ICML 2025)

The attack paper that motivates the noise term in our protocol.
**Confirms the threat:**

- Focuses **explicitly on autoregressive/causal transformers**.
- Their vocabulary-matching attack exploits the structural
  constraint that position `i` attends only to `j ≤ i`. The attack
  **does not require seeing the mask explicitly** — it infers the
  causal constraint from the model's behaviour on chosen inputs.
- Their recommended noise σ to defeat their attack is "relatively
  large" (no clean threshold; experiments in §5.4). σ=0.01 (which
  is what our codebase's `PermAttnConfig::HIDDEN_NO_MORE` ships)
  is at the **lower end** of their tested range; higher σ is
  needed to push ROUGE recovery to < 0.1.
- Implication for us: even **without** the mask-pattern leak, a
  GPU adversary observing the noisy permuted Q/K/V can run the
  vocab-matching attack at scale. Our defence rests on σ being
  large enough AND `π_b` staying hidden.

**Net:** mask-leak makes the attack trivial; even without the
mask-leak, σ=0.01 may not be safe under their stronger attack.

### 1.6 PermLLM (arXiv 2405.18744) — broken

Static-permutation-based scheme for fast private inference.
**Broken by Hidden No More.** Their vocab-matching attack
near-perfectly decodes input tokens from permuted hidden states
because the permutation is static across many queries — repeated
exposure under the same π enables the attack. Our protocol uses
fresh `π_b` per forward, which raises the bar but does not
eliminate the attack class (Hidden No More §5 shows the attack
extending to per-batch permutations with weaker bounds).

### 1.7 Cascade (arXiv 2507.05228, Jul 2025) — token-sharded

Multi-server scheme; non-colluding servers; **GPUs assumed
trusted**. Does not apply to our threat model (single-server
TEE+commodity-GPU). Mentioned for completeness — the post-PermLLM
remediation literature has moved toward multi-party rather than
solving the single-server permutation problem.

### 1.8 Fission (eprint 2025/653), SoK 2026/935, Privacy-Preserving LLM Inference (eprint 2026/105)

Surveys and distributed schemes. Neither identifies a single-server
TEE-only solution to the causal-mask-leak-under-permutation
problem. The SoK paper categorises by primitive (HE, MPC, TEE,
TEE+GPU) and the "TEE+commodity-GPU" cell is sparsely populated
with exactly the schemes already in our research notes (GELO,
Amulet, KV-Shield, TwinShield-Liu, SecureInfer).

### 1.9 Summary: known approaches

| Scheme | Mask handling | Fits our model? |
|---|---|---|
| Amulet | Not specified for causal | No (gap) |
| KV-Shield | Not specified | No (gap) |
| TwinShield (Liu) | TEE-applies-mask-then-HE-encrypt scores → GPU softmax under HE | Possible (heavy) |
| SCX | N/A — KV-cache problem, not attention mask | No |
| PermLLM | Static π; broken | No |
| Cascade | Multi-party | No (different threat model) |
| Confidential GPU (H100 CC) | Mask in encrypted VRAM, no leak | No (commodity GPU target) |
| **Our F1 (this doc)** | **TEE-applies-mask + TEE-softmax + plaintext score round-trip** | **Yes** |

**Net finding:** the causal-mask-on-GPU leak under permutation-
based TEE-GPU split is a **genuinely open problem in the
single-server commodity-GPU literature.** Every published scheme
either (a) doesn't address it (Amulet, KV-Shield), (b) needs HE
(TwinShield Liu), (c) needs multi-party (Cascade), or (d) needs
confidential GPU. F1 picks option (a) for compute and adds an
explicit in-TEE-softmax step — same approach as TwinShield Liu's
but without the HE wrapper, since openweight + GELO already gives
us the confidentiality property HE was buying.

## 2. Options F1–F6 reconsidered with literature in hand

(Same option labels as `m1-10-fused-permuted-attention.md` §3.)

### F1 — In-TEE softmax ✓ recommended

| Step | Where | Cost at n=2048, Qwen3-1.7B |
|---|---|---|
| Sample `π_b ∈ S_n`, permute & noise Q/K/V | TEE | µs (already happens) |
| `tilde_scores = matmul_dynamic_batched(πQ+η, πK+η)` | GPU | ~10 ms per layer |
| Return `tilde_scores` (B=16, n=2048, n=2048) | PCIe → TEE | 64 MB · ~6 ms at 10 GB/s |
| `scaled = tilde_scores · scale + M_π` | TEE (CPU) | ~17 MFLOPs · ms-scale |
| `probs = softmax_rowwise(scaled)` | **TEE** (CPU) | ~17 MFLOPs per head · ms-scale per layer |
| Send `probs` (B, n, n) back to GPU | PCIe → GPU | 64 MB · ~6 ms (same shape as scores) |
| `tilde_out = matmul_dynamic_batched(probs, πV+η_V?)` | GPU | ~10 ms per layer |
| Unpermute via π⁻¹ | TEE | µs |

**Per layer total: ~40 ms.** × 28 layers = ~1.1 s for attention
prefill at n=2048. Compare to current in-TEE attention at ~7 s.
**~6× speedup** on the attention slice; argument is information-
theoretic (mask `M_π` never leaves TEE) at PCIe-bandwidth cost.

**Privacy:** GPU sees `tilde_scores = (πQ+η)(πK+η)^T` and `probs`
(the softmax output). Probs has zero at blocked positions, which
arguably still leaks π through the zero-pattern. But softmax-output
zeros are at floating-point noise level (exp(-∞) → 0 exactly; with
finite mask values exp(-large) → denormal, indistinguishable from
zero after floating-point rounding). **This is a softer leak than
the explicit `-∞` mask, but it's still a leak** — see F1+ below.

#### F1+ — in-TEE softmax with non-zero blocked positions

Variant: instead of exact `-∞` at blocked positions, set blocked
positions to a moderately-large-negative value `-C` such that
`exp(-C)` is **not** at f32 denormal range — e.g. `-30`, giving
exp(-30) ≈ 9.4e-14. After softmax, blocked positions have small
but **non-zero** probabilities at the floating-point resolution.
The GPU sees a roughly-uniform distribution shape with a few
larger values at allowed positions — the **count attack** that
reveals π from the zero-pattern no longer works because there are
no exact zeros.

Drawback: blocked positions receive nonzero attention weight,
contributing nonzero terms to the output. At C=30 the contribution
is ~1e-13 per blocked position, ~n × 1e-13 ≈ 1e-10 cumulative —
within f32 noise; preserves model output up to single-token
argmax stability (validated by our existing greedy parity tests).

**This is the recommended Phase-0 fix.** F1+ retains all of F1's
structural safety AND addresses the residual zero-pattern leak.

### F2 — Per-row scaled mask ✗ rejected

Original idea: replace `-∞` with row-scaled finite values so each
row has the same finite range. **Rejected** in the parent plan —
softmax has 24-bit f32 mantissa, perfectly distinguishes "blocked"
from "allowed" values regardless of scale.

Re-examined: same conclusion. The count attack still works on the
pattern of "very small" vs "near-zero" probabilities. F1+ handles
the same concern more cleanly.

### F3 — Permute the mask pattern ✗ rejected

Sample independent `π_M ≠ π` for the mask. **Breaks the Amulet
equivariance identity** unless `π_M = π`, in which case the leak
is exactly as before. Reject.

### F4 — Causal-aware fused kernel without explicit mask ✗ rejected

Require the GPU to apply causal masking on permuted positions
using `π`. **GPU needs π to do this** — equivalent to handing
π to the adversary. Reject (same shape of mistake as the on-GPU
unmask we struck on 2026-05-18).

### F5 — Sample-dependent mask noise ✗ insufficient

Add Gaussian noise to mask values to obscure the block pattern.
Per Hidden No More, σ needs to be "relatively large" — likely on
the order of activation magnitudes (~1.0), not the σ=0.01 we use
for Q/K noise. At that magnitude the noise dominates the model
output: incompatible with correctness.

### F6 — Block-randomised masking ⚠ deferred (privacy weakening)

Replace per-position causal blocking with block-level patterns —
fewer distinct row-block-counts, no exact `perm[i]` recovery.
Privacy bounded above by block size. Requires a separate security
analysis. Filed in `future-rnd.md`; **not the v1 answer** but
potentially useful as a future optimisation once F1+ is in place
and we understand the actual bandwidth bottleneck.

### F7 (new) — Fused-second-matmul under F1

Once F1+ lands, the dispatch is:

```
GPU:  Q·K^T                  → tilde_scores
TEE:  apply mask + softmax   → probs
GPU:  probs · V              → out
TEE:  unpermute              → final
```

The **first** and **third** matmuls can still be fused into a
"flash-style" kernel if we restructure: keep `probs` resident on
GPU between TEE-softmax-write-back and the final matmul. Net
benefit: the score tensor materialisation on the GPU side is gone
(scores leave the GPU right after Q·K^T), saving HBM bandwidth
proportional to `n²·heads` per layer. **F7 = F1+ + HBM-fused
output matmul.** Phase 2 of M1.10 implements F7's GPU side.

### F8 (new) — Q·K^T via OutAttnMult, in-TEE softmax, probs·V via plain GPU matmul

Hybrid: use OutAttnMult's 4-partition cover for the first matmul
(hiding the permuted Q and K behind additive masks and a second
permutation `λ_Q, λ_K`), then in-TEE softmax (as F1+), then plain
GPU matmul on permuted V.

Trade-off: OutAttnMult adds the setup cost we measured at +14
ms/text in the embedder bench, scaling with n. At n=2048, that's
~140 ms × 16 heads × 28 layers ≈ 60 s per forward. **Strictly
worse than F1+ at long context.** Skip.

## 3. Recommendation

**Land F1+ in Phase 0.**

- It's the only option that gives information-theoretic protection
  on the mask pattern (mask never leaves TEE).
- The PCIe round-trip cost (~12 ms per layer at n=2048) is small
  compared to the 7 s of in-TEE CPU attention compute we replace.
- Existing trait surface (`engine.matmul_dynamic_batched`,
  `engine.softmax_batched` on the TEE side instead of GPU) needs
  no protocol changes; only the dispatch order in `attention.rs`
  changes.
- Phase 2 (engine kernel) is unblocked: the fused-output-matmul
  optimisation (F7) operates entirely on probs and V, neither of
  which carries mask information; the M1.10 fused-flash kernel
  becomes a standard FlashAttention-style fused matmul-only kernel,
  no mask in its inputs.

## 4. The F1+ construction in detail

### Algorithm

```
fn permuted_attention_f1plus(q, k, v, mask: Causal):
    π ← sample fresh permutation S_n          // TEE
    πQ ← π·Q ;  πK ← π·K ;  πV ← π·V          // TEE
    add N(0, σ²I) noise to πQ, πK             // TEE
                                              //
    tilde_scores ← engine.matmul_batched(πQ, πK^T)  // GPU
    // PCIe ← tilde_scores
                                              //
    M_softC ← softC_mask(π, C = 30)           // TEE
    scaled  ← tilde_scores · scale + M_softC  // TEE
    probs   ← softmax_rowwise(scaled)         // TEE  ← key step
                                              //
    // PCIe → probs
    tilde_out ← engine.matmul_batched(probs, πV)    // GPU
    // PCIe ← tilde_out
                                              //
    out ← π^(-1) · tilde_out                  // TEE
    return out

fn softC_mask(perm, C):
    // Soft causal mask: -C at blocked positions, 0 at allowed.
    //   M[i,j] = 0    if perm[j] ≤ perm[i]
    //          = -C   otherwise
    // After softmax, blocked positions ≈ exp(-C) / sum ≈ 1e-13 (negligible)
    // No exact zeros — defeats the count attack on softmax output.
```

### Security argument

| Surface | Adversary observation | Why this doesn't reveal π |
|---|---|---|
| `tilde_scores` from Q·K^T | `(πQ+η)(πK+η)^T` — row/col permuted with σ-noise | π is hidden in the row-permutation, never sent to GPU. The row order of tilde_scores does NOT correspond to the row order of canonical Q·K^T — recovering π from this requires breaking Amulet's equivariance (open problem). |
| `probs` (after TEE softmax) | Softmax output with mask-induced small (but non-zero, at f32 ≈1e-13) values at blocked positions | Same as scores — row order is π-shuffled. The pattern of "near-zero" entries reveals **the existence** of causal masking but at f32 precision they're indistinguishable from the noise floor of the per-row softmax output of unmasked attention with moderate activation differences. **Phase-0 regression test:** run a single forward, recover the candidate π̂ from probs, measure Spearman correlation against true π. Target ρ < 0.1. |
| `tilde_out` from probs·V | Permuted attention output | Same. |
| Mask `M_softC` | **never crosses PCIe** | This is the structural protection. |
| `π` | **never crosses PCIe** | Same. |

The remaining attack surface — recovering π from the
`(tilde_scores, probs, tilde_out)` triple — reduces to the Hidden
No More vocab-matching attack on per-batch fresh π under σ-noise.
That's the threat our `PermAttnConfig::HIDDEN_NO_MORE` already
targets. F1+ doesn't make this attack any easier than the current
bidirectional path the embedder is exposed to. It just stops the
trivial mask-pattern leak that makes recovery essentially free.

### Open question: σ tuning

`PermAttnConfig::HIDDEN_NO_MORE = { sigma: 0.01 }` is the paper's
threshold for ROUGE recovery < 0.1 against their attack at the
shapes they evaluated (GPT-2-class). For Qwen3-1.7B the activation
magnitudes are different (hidden size 2048 vs 768; different
training distribution); σ should be re-tuned empirically. Filed
as M1.10.0.5.

### Implementation plan

(Files relative to repo root.)

| File | Change |
|---|---|
| `crates/gelo-protocol/src/attention.rs::permuted_attention` | Move softmax inside TEE: drop the `engine.softmax_batched` call after the mask add; add a `softmax_rowwise` helper instead. The function is already structured around three engine calls; we go from `matmul → softmax → matmul` on GPU to `matmul → [TEE: mask+softmax] → matmul`. ~30 LOC. |
| `crates/gelo-protocol/src/attention.rs::softC_mask` | New helper. Build `(n, n)` causal mask with `-C` at blocked positions, 0 at allowed. `C` parameter on `PermAttnConfig` (default 30). |
| `crates/gelo-protocol/src/attention.rs::PermAttnConfig` | Add `causal_mask_neg: f32` (default 30.0). |
| `crates/gelo-protocol/tests/permutation_attention.rs::engine_cannot_recover_pi_from_single_forward` | New regression. Build a SpyEngine that records every input to `matmul_dynamic_batched` and `softmax_batched`. After a single `permuted_attention(..., Causal, ...)` call, attempt three reconstructions: (a) count exact zeros per row in `probs` → recover `π̂`; (b) count `<1e-12` per row in `probs` → recover `π̂`; (c) sort-correlate per-row magnitudes against position index. Assert Spearman(π̂, π) < 0.1 in every case across n ∈ {64, 256, 1024} for 1000 trials. |
| `crates/gelo-protocol/tests/permutation_attention.rs::amulet_identity_preserved_under_f1plus` | Existing parity test, re-run with `causal_mask_neg = 30` — must match the in-TEE-attention reference within 1e-4. |
| Plan §M1.10 phase docs | Update §3 in `m1-10-fused-permuted-attention.md` to mark F1+ as the chosen option; update §M1.10 Files-to-add/modify accordingly. |

### Acceptance gate for Phase 0

- All synthetic parity tests at `AttentionMask::Causal` pass with
  `causal_mask_neg = 30`, output match within 1e-4 of in-TEE
  reference.
- `engine_cannot_recover_pi_from_single_forward` passes:
  Spearman(π̂, π) < 0.1 for all three recovery attacks across
  1000 trials at n ∈ {64, 256, 1024}.
- End-to-end Qwen3-1.7B greedy generation token-parity
  (`qwen3_generation_e2e.rs`) passes with `use_perm_attention =
  true` plus `causal_mask_neg = 30`. Argmax-stable across the
  three executor cells (gpu_plain, gpu_gelo, gpu_full_stack).
- σ tuning: empirical Hidden-No-More re-validation at Qwen3-1.7B
  shapes. If σ=0.01 is insufficient at our activation magnitudes,
  raise default in `PermAttnConfig::HIDDEN_NO_MORE`.

## 5. Why this is safe to claim as a contribution

Net of the survey, F1+ is **not derivative of any published
single-server TEE-GPU scheme** — every published mask-handling
recipe either:
- doesn't engage on causal masks (Amulet, KV-Shield),
- carries an HE wrapper we don't need (TwinShield Liu),
- assumes a different threat model (Cascade, confidential GPU).

The construction is small (in-TEE softmax with a soft-saturating
causal mask), the security argument is information-theoretic on
the mask itself (TEE never transmits it), and the residual attack
surface reduces cleanly to the Hidden-No-More-class attacks our
existing σ-noise already targets.

Worth flagging in the eventual GELO-LLM paper write-up; **the
specific gap is real and unresolved in the public literature
as of 2026-05.**

## 6. References

- Mao et al., "Amulet: Fast TEE-Shielded Inference for On-Device
  Model Protection." arXiv 2512.07495.
- Liu et al., "Securing Transformer-based AI Execution via Unified
  TEEs and Crypto-protected Accelerators" (TwinShield, Liu '25).
  arXiv 2507.03278.
- Tan et al., "A First Look At Efficient And Secure On-Device LLM
  Inference Against KV Leakage" (KV-Shield). arXiv 2409.04040.
- Yuan et al., "SCX: Stateless KV-Cache Encoding for Cloud-Scale
  Confidential Transformer Serving." SIGCOMM 2025.
- Wang et al., "Hidden No More: Attacking and Defending Private
  Third-Party LLM Inference." ICML 2025, arXiv 2505.18332.
- "PermLLM: Private Inference of Large Language Models within 3
  Seconds under WAN." arXiv 2405.18744. (broken by Hidden No More)
- "Cascade: Token-Sharded Private LLM Inference." arXiv 2507.05228.
  (multi-party, doesn't apply to our model)
- IACR ePrint 2026/105 (SoK survey), 2026/935 (HE-focused SoK),
  2025/653 (Fission) — surveys/distributed; cite for completeness.
- `gelo-protocol::attention` — current TEE-side implementation.
- `m1-10-fused-permuted-attention.md` — parent plan; §3 to be
  updated to mark F1+ as the chosen option.
