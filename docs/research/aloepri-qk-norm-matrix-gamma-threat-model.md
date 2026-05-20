# Threat model — matrix-γ QK-norm extension for AloePri on Qwen3

**Status:** design-stage. Companion to the M2.7 attack-resistance
handoff and the Qwen3 architectural shape analysis. Documents the
security implications of deploying full AloePri Algorithm 2 on a
Qwen3-class backbone, where the QK-norm site requires extending the
per-head γ tensor from a `(d_h,)` vector to a `(d_h, d_h)` matrix.

## 0. Definitions

- **AloePri.** Privacy-preserving LLM inference scheme of Sheng et al.
  (arXiv 2603.01499). Combines a residual-stream keymat expansion
  (Algorithm 1) with per-head intra-head transforms (Algorithm 2) and
  additive noise (§5.2.2) so that the server holds only obfuscated
  weights and observes only obfuscated activations.
- **Keymat / Algorithm 1.** `P̂_R ∈ ℝ^{d × (d+2h)}`, `Q̂_R ∈ ℝ^{(d+2h) × d}`
  with `P̂_R · Q̂_R = I_d`. Expands the residual stream from `d` to
  `d + 2h` dimensions; the extra `2h` dimensions carry "decoy" mass
  designed to leak no token information.
- **Intra-head transforms / Algorithm 2.** Per-attention-head 
  block-diagonal transforms `M_q[h] = R̂_qk · Ĥ_qk · Ẑ_block` acting
  on the `head_dim` axis. Orthogonal by construction (RoPE-aware
  rotation × ±1 Hadamard × block permutation). `M_k[h]` chosen so
  that `M_q · M_kᵀ = I` (paper §5.2.3), cancelling in `Q·Kᵀ`.
- **QK-norm site.** RMSNorm applied to projected Q and K with a
  per-`head_dim` γ vector, between `W_q`/`W_k` and RoPE. Present in
  Qwen3 (and Qwen3-style models such as Olmo-2); absent in Qwen2,
  Llama-3, DeepSeek-R1-Distill.
- **Matrix-γ kernel extension.** The proposed kernel-level
  modification that replaces the elementwise γ multiplication at the
  QK-norm site with a `(d_h × d_h)` matmul against a per-head
  `Γ[h] = M[h]ᵀ · Diag(γ) · M[h]`. Subject of this threat-model
  document.
- **IMA.** Inverse-Mapping Attack (paper §F.1). A trained inverter,
  typically a 2-layer 8-head transformer, mapping a layer-0
  hidden-state observation back to plaintext token embeddings.
  Paper's main strength-test for the obfuscation property.
- **ISA.** Information-State Attack (paper §F.2). Linear/non-linear
  ridge from a chosen layer's hidden state or attention scores to
  plaintext tokens.
- **VMA / IA.** Vocabulary-Matching Attack / Inversion Attack — paper
  attack classes targeting *weight* privacy (not activation
  privacy). **Out of AloePri's defended scope** per §6.3.
- **TTRSR.** Top-token Recovery Success Rate, the primary attack
  metric. §6.3 sets the deployment gate at ≤ 15 % obfuscated TTRSR.

## 1. The Qwen3 algebraic gap (recap)

On Qwen3, attention has γ-multiplication between `W_q` and the dot
product:

```
Q_plain         = x · W_q
Q_plain_normed  = γ_q ⊙ (Q_plain / RMS(Q_plain))
                = (Q_plain / RMS(Q_plain)) · Diag(γ_q)
```

Algorithm 2 wants to fold `M_q` into `W_q`'s output axis
(`W̃_q = W_q · M_q`) and produce `Q_obf_normed = Q_plain_normed · M_q`,
so that downstream the dot product cancels via `M_q · M_kᵀ = I`.
With the standard QK-norm op:

```
Q_obf_normed_naive = (Q_obf / RMS(Q_obf)) · Diag(γ_q)
                  = (Q_plain · M_q / RMS(Q_plain)) · Diag(γ_q)   [M_q orthogonal]
```

For this to equal `Q_plain_normed · M_q`, `M_q` and `Diag(γ_q)` must
commute. The companion shape-analysis document measured the
γ-iso-tonic structure of Qwen3-1.7B's QK-norm γ vectors and found
the commuting subgroup empty at any operationally useful threshold
(≤ 22 % `head_dim` coverage at ε = 0.25, < 2 % at ε ≤ 0.10).

## 2. The matrix-γ kernel extension

Replace the elementwise γ multiplication with a per-head matmul:

```
Γ_q[h]  =  M_q[h]ᵀ · Diag(γ_q) · M_q[h]      shape (d_h, d_h)
Γ_k[h]  =  M_k[h]ᵀ · Diag(γ_k) · M_k[h]      shape (d_h, d_h)
```

The kernel evaluates:

```
Q_obf_normed = (Q_obf / RMS(Q_obf)) · Γ_q[h]
            = (Q_plain · M_q / RMS) · M_qᵀ · Diag(γ_q) · M_q
            = (Q_plain / RMS) · Diag(γ_q) · M_q
            = Q_plain_normed · M_q                ✓
```

Per-input exact (no approximation, no calibration corpus), provided
`M_q` is orthogonal. The paper's `R̂_qk · Ĥ_qk · Ẑ_block` is the
composition of an orthogonal rotation, a ±1 Hadamard, and a
permutation — all orthogonal.

### 2.1 Storage and compute

For Qwen3-1.7B (`d_h = 128`, `n_q = 16`, `n_kv = 8`, `n_layer = 28`):

- γ tensor grows from 128 floats to 128² floats per (layer, head, q|k).
- Per layer: `(16 + 8) × 128² × 4 B ≈ 1.5 MB`.
- Whole model: ~42 MB extra against ~3.4 GB fp32 base. < 1.3 %.
- Decode-time per token per layer: from 128 multiplies to 128² MACs
  per head; ~11 M MACs/token total — < 0.5 % of forward FLOPs.

Negligible on both axes.

## 3. Threat model — what the new tensor leaks

The new tensor `Γ_q[h] = M_q[h]ᵀ · Diag(γ_q) · M_q[h]` is a
**similarity transform of a diagonal matrix**. Its eigendecomposition
recovers both:

- **Eigenvalues** = `γ_q` (multiset; consistent across heads, so a
  consistency check is available to the attacker).
- **Eigenvectors** = columns of `M_q[h]`, up to a `head_dim`
  permutation and sign.

A server-side adversary with code execution can therefore run
`numpy.linalg.eig(Γ_q[h])` and recover `M_q[h]` in `O(d_h³)` time per
head — for `d_h = 128`, well under a millisecond. Once `M_q[h]` is
known, the adversary computes `M_q[h]⁻¹ · Q_obs = Q_plain` at every
QK-norm observation site.

### 3.1 Is this a regression versus the paper's defended threat model?

**No.** The paper's evaluation regime (§6.3) covers only activation-
side attackers:

| Attack    | Surface                                      | Class                   |
|-----------|----------------------------------------------|-------------------------|
| NN        | wire-side embedding lookup                   | cosine retrieval        |
| IMA       | layer-0 hidden state → plaintext token       | trained inverter        |
| ISA HS    | layer-K hidden state → plaintext token       | ridge / inverter        |
| ISA AS    | attention scores → plaintext token           | ridge / inverter        |
| TFMA      | output token stream → token frequency        | frequency analysis      |
| SDA       | output token stream → bigram substitution    | substitution cipher     |
| **VMA**   | **plaintext W_v vs `W̃_v`**                  | **weight matching** ⊘   |
| **IA**    | **`W̃_q`, `W̃_k`, `W̃_v`, `W̃_o` → Q̂_q**     | **weight inversion** ⊘  |

VMA and IA target weight privacy and are listed **out of scope** by
the paper authors. The M2.7 handoff confirms this for our deployment:
"Weight-privacy attacks (VMA + IA) are out of scope."

The matrix-γ kernel extension introduces a *new* weight-privacy
weakness (the `MᵀDM` similarity tensor). But that class is already
excluded from AloePri's defended surface. So:

- **Within the paper's stated threat model** (IMA, ISA, TFMA, SDA, NN
  — all activation-side), the matrix-γ extension is as strong as
  vanilla Algorithm 2. The IMA attacker does not eigendecompose
  weights; they fit a linear/transformer inverter to observed
  activations.
- **Outside the paper's stated threat model** (an adversary that
  reads weights and runs `numpy.linalg.eig`), the matrix-γ extension
  is broken in *milliseconds*. Vanilla Algorithm 2 is also broken in
  this regime — see §3.2 — but more slowly.

The qualitative security property does not change; the leak speed
does. Within scope, both are equivalently safe; out of scope, both
fail, and the matrix-γ extension fails faster.

### 3.2 Is vanilla Algorithm 2 (Qwen2) safe against weight analysis?

**No — but the attack is different.** Under vanilla Algorithm 2 the
server holds:

```
W̃_q  =  Q̂_q · W_q · M_q                shape (d+2h, n_q · d_h)
W̃_k, W̃_v, W̃_o   similarly
```

These are dense matrices; there is no `MᵀDM` similarity tensor.
Eigendecomposition is not directly applicable. But the underlying
plaintext `W_q` is **publicly known** — base models are released on
Hugging Face. An attacker can attempt to solve:

```
W̃_q  =  Q̂_q · W_q · M_q
```

for unknowns `(Q̂_q, M_q)` with known plaintext `W_q`. Without the
keymat (`Q̂_q` absent or known), `M_q = W_q⁺ · W̃_q` recovers via
pseudoinverse — a one-line attack. The keymat factor `Q̂_q` adds an
unknown `(d+2h, d)` left-multiplier; recovering `M_q` from `W̃_q`
given known `W_q` becomes a constrained matrix-factorisation problem
of moderate difficulty. Block-diagonal-per-head structure on `M_q`,
public W_q row-norms, and the orthogonality constraint give the
attacker substantial leverage. Not cryptographically hard; not yet
formally analysed in the literature.

So vanilla Algorithm 2 already excludes weight-analysis attacks from
its defended surface (the paper does this explicitly with VMA/IA).
The matrix-γ extension does not *add* an excluded class — it makes
one specific attack in that already-excluded class trivial.

### 3.3 Practical reading

Inside the paper's evaluation regime, the matrix-γ extension is a
faithful instantiation of full Algorithm 2 for the Qwen3 backbone
and inherits the paper's measured defence numbers (Table 3:
IMA HS → 0 %, paper-faithful reproduction target).

Outside the paper's evaluation regime, AloePri-class static
obfuscation does not provide weight privacy on any backbone —
**this is a property of the scheme, not of our deployment**. Strong
weight privacy requires either:

- **Per-prompt fresh masking** (GELO, ObfuscaTune-on-activations,
  CAPRISE) — see `docs/research/private-llm-inference-round-2.md`
  and `docs/prototype/gelo-llm.md` for the path-1 instantiation.
- **HE / MPC primitives** with formal indistinguishability bounds.

Both options break the "no infrastructure change on the server"
deployment thesis that motivated AloePri in the first place.

## 4. Proposed defences (mitigations within AloePri's design space)

Ranked by deployment cost.

### 4.1 Status-quo: scope the deployment to the paper's threat model

**Cost.** Zero (documentation only).

**Mechanism.** The threat-model section of `aloepri-llm.html` already
excludes VMA/IA. Extend that exclusion explicitly to "weight-analysis
attacks of any form, including eigendecomposition of QK-norm tensors
in the matrix-γ form". The deployment defends what it always
defended: the seven attacks in §6.3 minus VMA/IA. Nothing about the
security argument changes.

**When this is enough.** When path-2's goal is paper-faithful
replication of §6.3 / Table 3 against the paper's own attack suite.
This is the current M2.7 mission.

**When this is not enough.** When the deployment story extends to
defending against an adversary that reads `numpy` documentation.

### 4.2 Additive noise on Γ_q

**Cost.** Tuning (1–2 weeks research) + ~1 day to wire.

**Mechanism.** Replace the published tensor with
`Γ̃_q[h] = Γ_q[h] + σ · N[h]`, where `N[h]` is i.i.d. Gaussian.
Eigendecomposition of `Γ̃_q` recovers `M_q` only up to a noise floor
proportional to `σ / (λ_i − λ_j)` for adjacent eigenvalues; with
small enough `σ` the kernel output is close to plaintext, with large
enough `σ` the recovery error makes `M_q⁻¹ · Q_obs` an unreliable
recovery of `Q_plain`.

**Why this is fragile.** γ_q's eigenvalue spread is large and
extremely non-uniform — `γ_k` values range from ~-1 to +68 across
heads (the QK-norm γ_k vector at layer 0 alone has values from 0.07
to 68.0). For a fixed σ, the noise-to-signal ratio across the
eigenvalue spectrum varies by orders of magnitude. The largest-γ
directions (which carry the most attention-score information) are
recoverable from σN-noised Γ̃ basically untouched; only the tiny-γ
directions get protection. This is an analogue of the embedding-
research finding that static-key obfuscation breaks against
ArrowMatch when eigenvalue heterogeneity is high — see memory
`private_embedding_findings.md`.

**Verdict.** Plausible research direction; not a 1-week deployment.
Path-2 should not invest here unless the threat model
unambiguously requires weight-privacy and HE is rejected on
performance grounds.

### 4.3 Per-prompt fresh M_q via thin client handshake

**Cost.** Breaks the "no infrastructure change" thesis. ~2 weeks of
plumbing on top of full Algorithm 2 + Option 4.1.

**Mechanism.** Trusted client samples fresh `M_q[h]_t` per inference
request, computes `Γ_q[h]_t` and ships it to the server alongside
the prompt. Server applies that prompt's `Γ` and unloads after.
Server eigendecomposes per-prompt → recovers `M_q_t` for that single
prompt → has one `(Q_obs, label)` pair under that transform. Trained
inverters require thousands of prompts under a consistent transform
to fit. Fresh `M_q_t` per prompt drives that signal to zero.

**Why this is correct.** It is what GELO already does on path-1 (see
`docs/prototype/gelo-llm.md` and memory `gelo_research_round_2.md`).
The two paths converge: static intra-head obfuscation is structurally
unable to defend against weight-analysing adversaries; fresh-per-
prompt masking is structurally able.

**Why this is out of scope for path-2.** Path-2 exists to deliver
the AloePri deployment exactly. Per-prompt M_q sampling makes it a
GELO variant, not AloePri. If we wanted that, we already have path-1.

**Verdict.** Not the right move for path-2. If we need this defence,
the right answer is "use path-1", not "rebuild path-2 to look like
path-1".

### 4.4 Hide γ_q's eigenvalue structure entirely (folded into W_q)

**Cost.** Negative — *fewer* tensors to ship — but requires a paper
extension that has not been derived in the literature.

**Mechanism.** Fold the QK-norm γ multiplication *into* `W̃_q` itself,
not into a separate Γ tensor. The kernel then runs the standard
RMS-without-γ followed by the dot product, with the γ-correlation
baked into the projection weights. This is precisely the case the
paper §5.2.5 does *not* cover (γ on the *output* axis of `W_q`); the
companion shape-analysis document documents why the §5.2.5 κ
construction does not extend cleanly to this case.

**Verdict.** Open research problem. Worth a literature pass; not a
short-term deliverable.

## 5. Decision and durable record

For the M2.7 mission (paper-faithful replication of §6.3 attack
numbers against Qwen3-1.7B), **defence 4.1 (scope-narrow) is the
chosen posture**. The matrix-γ kernel extension lands; the
deployment defends against IMA, ISA, TFMA, SDA, NN exactly as the
paper's full Algorithm 2 deployment does on Qwen2-class backbones;
weight-analysis attacks (VMA, IA, the new Γ eigendecomposition) stay
explicitly out of scope, matching the paper's own exclusion list.

If a future deployment target requires weight privacy, the answer is
**not** a tweaked AloePri — it is the path-1 GELO stack. This is
already documented in `docs/research/aloepri-vs-gelo.md`; the present
document extends that comparison with the specific observation that
**QK-norm-bearing backbones make the activation-side / weight-side
threat-model boundary even sharper than it is for Qwen2-class
models.**

## Sources

- Sheng et al., *Towards Privacy-Preserving LLM Inference via
  Covariant Obfuscation*, arXiv:2603.01499. §5.2.3 (Algorithm 2),
  §5.2.5 (§κ fold), §6.3 (attack suite), §7 (evaluation).
- `docs/handoffs/2026-05-19-alg2-qwen3-shape-analysis.md` —
  γ-iso-tonic measurement, §4a verdict.
- `docs/handoffs/2026-05-19-m2-7-attack-findings.md` — M2.7 attack
  results that motivated this work.
- `docs/research/aloepri-vs-gelo.md` — broader threat-model
  comparison between AloePri and GELO.
- `docs/research/private-llm-inference-round-2.md` — round-2 survey
  of weight-analysis defences in the wider literature.
