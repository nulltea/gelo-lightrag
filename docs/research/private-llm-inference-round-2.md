# Private LLM Inference — Research Round 2

> **Research date:** 2026-05-18. Follow-up to
> [`private-inference.md`](private-inference.md) (2026-04-21), focused on
> what's changed since that survey and on three model-architecture
> classes we have not yet validated GELO+TwinShield against: **MoE**
> (Qwen3-MoE / DeepSeek-V3), **hybrid attention** (Gemma 3), and **Per-Layer
> Embeddings** (Gemma 3n).
>
> **Hardware scope:** SEV-SNP CVM + commodity Vulkan GPU passthrough — the
> "replicate H100-CC-class workflows on a $50/mo Hetzner box" deployment.
> We are not chasing confidential-GPU adoption; we are chasing the
> protocol primitives that make commodity-GPU passthrough as safe as
> H100-CC.
>
> **Companion docs**:
> [`private-inference.md`](private-inference.md) (round 1, the canonical
> survey),
> [`../prototype/gelo.md`](../prototype/gelo.md) (embedding prototype that
> implements the GELO+TwinShield design),
> [`../prototype/gelo-llm.md`](../prototype/gelo-llm.md) (forward-looking
> LLM-generation extension),
> [`../prototype/future-rnd.md`](../prototype/future-rnd.md) (committed
> roadmap items).

---

## Definitions

| Term | Meaning |
|---|---|
| **AloePri** | ByteDance covariant-obfuscation scheme (arXiv 2603.01499). |
| **Amulet** | Softmax-permutation-equivariance technique (arXiv 2512.07495). |
| **ArrowMatch** | Attack class against weight-axis static-mask schemes; defeats STIP/ObfuscaTune. |
| **BSS** | Blind Source Separation — the inversion class GELO+shielding defeats. |
| **CC (in H100-CC, B200-CC)** | NVIDIA Confidential Compute mode (encrypted VRAM + attested driver). |
| **CryptoMoE** | 2PC MPC scheme with balanced-routing defense (arXiv 2511.01197). |
| **CVM** | Confidential VM (SEV-SNP, TDX). Hardware-isolated guest. |
| **GELO** | Activation-mask split inference (arXiv 2603.05035). Our base protocol. |
| **Haar-uniform** | Uniform distribution over the orthogonal group O(n); GELO's mask source. |
| **Hidden No More** | Attack class against fixed-permutation schemes (arXiv 2505.18332). |
| **ICA** | Independent Component Analysis — BSS sub-class. |
| **MoE** | Mixture of Experts. Sparse activation of K-of-N experts per token. |
| **MoEcho** | Side-channel attack on MoE routing (CCS 2025, arXiv 2508.15036). |
| **OSNIP** | Obfuscated-semantic-null-space client-side encryptor (arXiv 2601.22752). |
| **OutAttnMult** | TwinShield's 4-partition Q·Kᵀ masking primitive. |
| **PLE** | Per-Layer Embeddings — Gemma 3n's per-(layer, token_id) cache table. |
| **SCX** | Stateless KV-cache encoding for cloud LLM serving (SIGCOMM '25). |
| **SEV-SNP** | AMD Secure Encrypted Virtualization-Secure Nested Paging. |
| **SEV-TIO** | AMD's TDISP-based secure I/O extension. Linux 6.19 has foundations. |
| **SWIOTLB** | Software I/O TLB — bounce-buffer DMA for SEV-SNP guests. |
| **TDISP** | PCI-SIG TEE Device Interface Security Protocol. |
| **TDX** | Intel Trust Domain Extensions. The Intel counterpart to SEV-SNP. |
| **TEE.Fail** | Oct-2025 DDR5 bus-interposer attack on SGX/TDX/SEV-SNP. |
| **TwinShield (ours)** | Xue et al. 2025 (our citation, arXiv 2505.x). Shield rows + OutAttnMult + U-Verify. |
| **TwinShield (Xue '25)** | Unrelated Jul-2025 paper (arXiv 2507.03278). Name collision; see §A1. |
| **U-Verify** | Freivalds-style integrity probe; part of our TwinShield citation. |
| **Vec2Text** | Morris et al. EMNLP '23 embedding-inversion attack. |

---

## TL;DR

1. **No new attack invalidates GELO's per-batch full-rank Haar mask.**
   The TEE.Fail bus-interposer attack (Oct 2025) defeats H100-CC's
   attestation-rooted trust, but **strengthens** the case for per-batch
   activation masking: even a fully compromised TEE only loses the
   single batch's mask state, not the model. The Hidden No More and
   precomputed-basis attack families remain limited to static / fixed-
   permutation schemes that GELO is not in. AloePri (ByteDance, Mar
   2026) is a new same-family competitor but we cannot yet confirm
   whether its rotation is per-prompt or per-session — the latter would
   place it in the broken family.

2. **MoE under GELO is a real research project, not a port.** The
   routing histogram is a first-class leak with a published attack
   recovering 99.9% of templated prompts on DeepSeek-V2 (MoEcho, CCS
   2025) and at least two independent defenses (CryptoMoE,
   "Expert Selections Reveal..."). The mask mechanics carry over
   cleanly (one shared `A` per batch, slice by expert), but the
   router must stay in-TEE and dispatch must be flattened
   (CryptoMoE-style `t = 2mk/n`). Engineering: ~1 month for a leaky
   demo, **3–6 months for a publishable result**.

3. **Hybrid attention (Gemma 3) is a clean win.** Sliding-window
   attention drops in-TEE attention cost ~3.86× at n=8K, keeping all
   5/6 local layers comfortably in-TEE out to n≈16K and reducing the
   regime where we need OutAttnMult to the 1/6 global layers.
   Permuted attention does **not** extend to sliding-window (mask is
   not permutation-invariant); block-diagonal permutation works with
   ~7× entropy reduction, still astronomical.

4. **PLE (Gemma 3n) is a P0 leak that we discovered, not the paper.**
   The Per-Layer Embedding table is keyed by `token_id`. Observing
   gather addresses recovers the prompt **in plaintext, without
   inverting any hidden state**. Strictly worse than Vec2Text. The
   fix is mechanical (PLE table lives in TEE DRAM — 1.9 GB int8,
   fits) but is **non-negotiable for any Gemma 3n deployment** under
   our threat model.

5. **OSNIP is not for our threat model.** Defends against a cloud
   LLM *provider* with the model and gradient access, not a
   TEE-co-located GPU/host adversary. Cannot be applied to commercial
   LLM APIs (no gradient cooperation; APIs take tokens, not embeddings).
   Possible future composable layer if we ever self-host a serving
   stack and want input privacy against our own operators. Not on
   the critical path.

6. **Hardware: TDISP foundations landed in Linux 6.19** (Oct 2025).
   End-to-end confidential-device-passthrough is realistically 12+
   months out, and NVIDIA has not committed to TDISP. Our consumer-GPU
   passthrough + GELO bet remains correct through 2026.

---

## A. New 2025–Q4 / 2026 Systems in the GELO Family

### A1. TwinShield (Xue et al.) — name collision warning

Xue, Zhao, Zheng, Yao, Solihin, Lou, *Securing Transformer-based AI
Execution via Unified TEEs and Crypto-protected Accelerators*. arXiv
2507.03278 (Jul 2025). **Different team, different scheme, identical
name.** Disambiguate in writing: "TwinShield (Xue '25)" vs
"TwinShield (Liu et al. '25)" (ours, shield-rows + OutAttnMult +
U-Verify).

**What it is.** TEE = SGX, accelerator = non-CC GPU (GTX-series and
A100 tested). Additive secret sharing for linear ops; multiplicative
mask blinding for Q·Kᵀ; **`e^(X+R) = e^X · e^R` blinding for softmax
outsourcing**. Claims 87% of compute on GPU, 4–6.1× speedup.

**Why we care.** The softmax-outsourcing trick is the only published
2025 scheme that pushes softmax to GPU *without* the Amulet-style
permutation-equivariance identity. We should compare:

| Scheme | What stays in TEE | What moves to GPU | Security argument |
|---|---|---|---|
| Ours (`gelo.md` §3.5) | softmax | matmul (under GELO mask) | Per-batch fresh Haar |
| Amulet-style (`gelo.md` §3.5b) | mask, π sample, σ-noise | softmax (under π) | Equivariance + shield rows + σ-noise |
| TwinShield-Xue softmax | scaling, R sample | exp evaluation (under R) | Multiplicative blinding; not yet attacked |

The Xue scheme is **paper-only** (no code surfaced), so we cannot
benchmark it. The construction is worth a security-analysis spike
to see whether `e^(X+R)` blinding survives a Hidden-No-More-style
adversary that observes many `(X+R, exp(X+R))` pairs and the
softmax denominator structure.

### A2. AloePri (ByteDance) — resolved as not applicable under open weights

Lin et al., *Towards Privacy-Preserving LLM Inference via Covariant
Obfuscation*. arXiv 2603.01499 (Mar 2026). ByteDance + Nanjing Univ.
Reference code (community reproduction): `sheng1feng/Aloepri`.

**Resolution (2026-05-18, after full paper + code read).** AloePri's
rotation is **per-deployment static** — one `τ`, one set of key
matrices `{P̂_i, Q̂_i}`, one set of weight noise vectors, all baked
into shipped weights. **Security relies on the server not having
access to the original model weights** (§3.2 threat model). Under
our openweight assumption, the adversary has public `θ` and can
trivially recover `τ` from `(θ, θ̃)` pairs. **AloePri provides no
security under our threat model.**

Full deep-dive and applicability matrix in
[`aloepri-vs-gelo.md`](aloepri-vs-gelo.md). Of the 10 distinct
techniques surveyed, **9 are inapplicable** to our open-weight
setting; **1 is highly portable** — the empirical attack suite at
`src/security_qwen/` (VMA, IA, ISA, IMA, NN, TFMA, SDA) is the
broadest open-source attack codebase for split/obfuscated inference
and should be ported to validate GELO empirically (~2–3 weeks).

### A3. Other 2025–2026 same-family entries

| System | arXiv / venue | Family fit | Verdict |
|---|---|---|---|
| **Comet** | 2505.07239, S&P '25 | MPC + activation-sparsity prediction | Orthogonal to TEE setting; cite as MPC baseline only. |
| **CMIF** | 2509.09091, DASFAA '25 | Thin client-TEE (embedding only) + GPU | Strictly weaker than GELO's per-layer mask; already on our radar. |
| **Privacy-Aware Split Inference (Cunningham '26)** | 2602.16760 | Layer-split, no mask | 35–59% token-recovery attack rate confirms empirically *why* GELO masking matters. Cite as cautionary baseline. |

---

## B. New Attacks (2025–2026)

### B1. TEE.Fail — DDR5 bus interposition (Oct 2025) — most consequential result of the year

Tech, Seto, Berrios, van Schaik, Garman, Genkin (Georgia Tech + Purdue
+ Synkhronix). [tee.fail](https://tee.fail/), Oct 28 2025.

**What it does.** Sub-$1000 DDR5 memory-bus interposer extracts
plaintext from **Intel SGX, Intel TDX (including ciphertext-hiding
mode), and AMD SEV-SNP**. Documented chain: extract CPU-TEE
attestation keys → impersonate a CPU-TEE → compromise NVIDIA H100-CC
attestation chain → run adversary workloads bypassing GPU CC.

**Status.** Paper + public site, no code released. Vendor advisories
AMD-SB-3021 and INTEL-2025-10-28-001 acknowledge.

**Implication for us.** GELO's per-batch full-rank Haar mask was
designed against an honest-but-curious GPU; it was not designed
against a *fully compromised TEE*. But the mask material exists in
TEE memory **for the duration of one forward pass only**: even if
TEE.Fail-class adversaries dump CVM memory pages, they only recover
the mask for batches sampled while their interposer was active and
time-aligned with the forward-pass window — minutes per batch of
attack window, vs ~hundreds of milliseconds per batch of mask
lifetime. The compute time-alignment requirement is hard.

**Concrete check for our threat-model write-up.** Add a paragraph
to `gelo.md` §6 "What it does not protect" stating:
- DDR5 bus interposition with time-aligned mask exfiltration is
  out of scope (no protocol-level defense; mitigation is
  physical security of the host).
- This attack class **strengthens** the architectural case for
  per-batch full-rank masking vs longer-lived secrets (precomputed
  bases, per-session keys, weight-baked rotations) — those leak
  far more under the same primitive.

**Does not change our roadmap.** Validates it.

### B2. Shadow in the Cache / KV-Cloak — KV-cache attacks + defense

arXiv 2508.09442 (Aug 2025). Three attacks against unprotected
KV-cache: direct Inversion, Collision, Injection. Defense (KV-Cloak)
applies per-block secret invertible linear transform + one-time
random permutation. vLLM integration.

**Relevance.** SCX (Yuan '25, our planned decode-phase primitive per
`gelo-llm.md` §4.3) shields KV-cache via per-user keys. Need to
check whether SCX's construction survives the three KV attacks here
or needs to compose with KV-Cloak's per-block permutation. **Action
for the decode-phase research spike** (`gelo-llm.md` §6 step 7):
include this composition check in the SCX evaluation.

### B3. MoEcho — routing-histogram side channel

Already covered in §C.

### B4. What did NOT appear

Despite explicit search, **no published attack** against per-batch
fresh full-rank Haar masking (GELO-family) appeared in 2025-Q4 /
2026. The closest is the precomputed-basis attack (arXiv 2602.11088,
already in our threat model), and that paper's empirical claim that
"random subset sampling provides no meaningful defense" is
specifically about K-of-N subset draws from a *fixed* basis — not
fresh full-rank sampling. **GELO's family-immunity argument
(`gelo.md` §6) holds at 2026-05-18.**

---

## C. MoE Private Inference

### C.1 The routing histogram is the dominant new leak

**MoEcho** (arXiv 2508.15036, CCS 2025) is the load-bearing result:
GPU performance counters + TLB Evict+Reload recover **99.9% of
templated healthcare prompts and 56.5% of free-form prompts on
DeepSeek-V2** purely from observing expert-load and expert-sequence
patterns. No root needed. 92.8% response-reconstruction accuracy on
Mixtral-class models.

A second independent result, **"Expert Selections Reveal (Almost) As
Much As Text"** (arXiv 2602.04105), reaches the same conclusion via
a different attack vector.

**Implication.** The expert-routing histogram leaks token-level
semantic information *before* any activation masking is applied.
GELO's mask defends activation values; routing identity is
out-of-scope of the mask's promise.

### C.2 GELO mask transports cleanly; router must stay in-TEE

Mechanically:

1. **One shared `A` per batch** suffices. After the router dispatches
   tokens to expert `e`, the slice `H_e = H[indices_e, :]` is a
   contiguous row subset; applying the same `A` (suitably
   row-permuted to match) gives `A · H_e`. The fresh-per-batch
   guarantee carries over to per-batch-then-routed sub-batches.
   No per-expert fresh `A_e` needed.
2. **The router itself must stay in TEE.** Gate matrix is small
   (~10 MB for Qwen3-MoE 128-expert) so the cost is negligible.
3. **Unmask-and-combine in TEE.** The `softmax(gate) · expert_out`
   weighted sum must happen post-unmask in TEE, since gate weights
   are routing-decision-derived sensitive signal.

The "8–128× mask machinery overhead vs dense" worry is unfounded.
Mask cost is per-batch, not per-expert.

### C.3 The dispatch-flattening primitive (load-bearing)

**CryptoMoE** (arXiv 2511.01197, Nov 2025) is the canonical
defense. Their construction:

- Force every expert to receive exactly `t = 2mk/n` tokens
  (m = batch tokens, k = active, n = experts).
- Pad short experts with dummies; drop overflow tokens by
  routing-confidence rank.
- "Confidence-aware token selection" recovers 99.2% of insecure-
  baseline accuracy on DeepSeekMoE-16.4B, OLMoE-6.9B,
  QwenMoE-14.3B.

CryptoMoE is honest-but-curious 2PC (SS+BFV), not TEE+GPU. But the
**dispatch-flattening primitive is directly portable to
GELO+TwinShield** — independent of the underlying crypto. Run the
gate in-TEE, flatten the histogram to constant `t`, dispatch
masked sub-batches of identical shape `[t, d]` to N GPU experts.
The histogram is gone; the GPU sees N indistinguishable shapes.

### C.4 Cost model

For Qwen3-MoE at 8-of-128 with CryptoMoE-style balancing
(`t = 2·m·8/128 = m/8` per expert):

- Total expert-FFN compute: `N · t · 2·d·d_ff = 2·m·k·2·d·d_ff` =
  **~2× the active-expert work**, i.e. **~2× dense FFN cost**.
- Mask GEMM: 1 apply per layer on `H` of shape `[m, d]`; ~1×
  dense cost.

Net overhead vs dense Qwen3 is ~2×, not 8–128×. The 2× is the
price of flat dispatch; there is no cheaper privacy-preserving
alternative without losing accuracy.

### C.5 Engineering verdict

| Component | Effort |
|---|---|
| Per-batch shared `A`, sliced per expert | Trivial (existing primitive) |
| Router in TEE | Trivial (~10 MB gate) |
| Histogram flattening (CryptoMoE `t = 2mk/n`) | 2–4 weeks; lift directly |
| Unmask-and-combine in TEE | ~1 week of kernel work |
| Validation: replay MoEcho against our prototype to verify histogram is flat under load | 2–4 weeks |
| Accuracy measurement under confidence-aware dropping | 2–3 weeks (real corpus, real model) |

**~1 month for a leaky demo, 3–6 months for a publishable result.**
The CryptoMoE primitive must be in v1, not v2 — without it, the
demo leaks the histogram and a careful security review will reject it.

### C.6 Open question — primary or secondary?

MoE is on the agenda because the user named Qwen MoE specifically.
But the existing prototype embeds with Qwen3-Embedding-0.6B
(dense). The MoE work is upstream of any code we have today.
Decision needed: is MoE serving the next milestone after the
dense LLM serving harness (`gelo-llm.md` §6 step 1), or after?
**Recommendation**: start the new ADR file
`docs/adr/NNNN-moe-private-inference.md` to capture the
routing-in-TEE + balanced-dispatch decision before any MoE-target
code lands; sequencing can defer.

---

## D. Hybrid Attention (Gemma 3) + PLE (Gemma 3n)

### D.1 Architectural ground truth

**Gemma 3** (1B / 4B / 12B / 27B): 5:1 interleave of local
sliding-window attention (W=1024, RoPE base 10k) and global
attention (RoPE base 1M). Context 32K–128K. head_dim=256. Last
layer was sometimes local (architectural quirk).

**Gemma 3n E4B** (effective 4B / raw 8B):
- Gemma 3 attention backbone.
- **PLE table**: `[262144 tokens × 30 layers × 256 dims]`, ~2 GB
  fp16 / ~1 GB int8. Indexed by `token_id` (not position).
- **MatFormer**: nested submodels (E2B ⊂ E4B). Compile-time
  weight selection; doesn't change attention shape.

**Gemma 4** (released ~Q1 2026): same hybrid family; deltas vs
Gemma 3 worth tracking for our analysis:

| Model | Layers | Ratio | W (local) | Notes |
|---|---:|---:|---:|---|
| E2B | 35 | **4:1** | **512** | Effective 2.3B / raw 5.1B |
| E4B | 42 | 5:1 | **512** | Effective 4.5B / raw 8B |
| 26B A4B | — | 5:1 | 1024 | **MoE: 128 experts, 8 active, 3× shared expert** |
| 31B | — | 5:1 | 1024 | Dense |

Plus three architectural changes that affect our analysis:
- **K=V in global layers** — K and V tensors share storage; combined
  with 8-to-1 GQA and doubled key dimensionality.
- **p-RoPE (p=0.25)** in global layers.
- **Last layer is always global** (closes the Gemma 3 quirk).
- **Native audio + video encoders** (~150 M params each) on E2B/E4B
  alongside text — see §D.9.
- **PLE persists** in E2B/E4B at table shape `[262144 × 256 × N]`
  (N = layer count, 35 or 42). E2B int8 ≈ 1.1 GB; E4B int8 ≈ 1.3 GB.
  Still fits comfortably in TEE DRAM.

### D.2 Hybrid attention — clean win for the in-TEE path

In-TEE attention cost goes from `O(n² · d_head · H)` to
`O(n · min(n,W) · d_head · H)` for local layers. Speedups at n=8192:

| Model | Ratio | W | Speedup vs dense at n=8K |
|---|---:|---:|---:|
| Gemma 3 (any) | 5:1 | 1024 | 3.86× |
| Gemma 4 E4B | 5:1 | **512** | **4.57×** |
| Gemma 4 E2B | 4:1 | 512 | 4.00× |
| Gemma 4 26B A4B / 31B | 5:1 | 1024 | 3.86× |

Gemma 4's small models (E2B/E4B) are *more* in-TEE-friendly than
Gemma 3 because the window halved. At n=32K, W=512 the E4B speedup
grows further to ~7×.

**Practical consequence.** The current 28% in-TEE wall-clock on
Qwen3-Embedding-0.6B at n≈400 (`gelo.md` §8) is the dense baseline.
With Gemma-3/4-style hybrid at n=8K, **local layers stay comfortably
in-TEE; only the 1/6 (E4B, 31B) or 1/5 (E2B) global layers need
OutAttnMult or permuted attention**. The "all-or-nothing"
attention-placement decision becomes a per-layer-class decision.

### D.3 Permuted attention does NOT extend to sliding window

The identity `softmax(πQKᵀπᵀ) = π · softmax(QKᵀ) · πᵀ` requires
the attention to be permutation-equivariant under π. Adding a
mask M changes the requirement to:

```
M ⊙ (π Q Kᵀ πᵀ) = π (M ⊙ Q Kᵀ) πᵀ
⇔ M[π(i), π(j)] = M[i, j] for all i, j
```

For sliding window `M[i,j] = 1 ⟺ |i-j| < W`, this requires π to
preserve the band structure — only translations satisfy this, which
is a single-parameter subgroup, **not** the full symmetric group
`S_n`. Permuted attention as designed is broken on local layers.

**Workaround: block-diagonal π.** Permute within each W-sized
window. Security entropy degrades from `log₂(n!)` to
`(n/W) · log₂(W!)`. For W=1024, n=8192: 8 × log₂(1024!) ≈ 8 × 8769
≈ 70152 bits — still astronomical but ~7× fewer bits than full π.
Acceptable.

**For global layers**: full permuted attention works as-is. On
Gemma 4 global layers specifically, the K=V trick (§D.1) cuts our
mask work roughly in half: we sample one mask for the shared K/V
tensor instead of two for separate K and V. Free win, no security
change.

### D.4 OutAttnMult on sliding window — also non-trivial

OutAttnMult's 2n×2n product is dense by construction. Sliding-window
sparsity is lost; the local layers pay full O(n²) for blocks that
softmax will zero. Sparse-aware OutAttnMult is non-obvious because
the orthogonal mask linearly combines off-band entries with on-band
entries — band structure is destroyed by the mask.

**Practical answer:** in-TEE for local layers (cheap thanks to D.2);
full-dense OutAttnMult for global layers. Don't try to sparsify
OutAttnMult.

### D.5 PLE — a P0 leak we discovered

**The attack.** Gemma 3n's PLE table is indexed by `token_id`.
Same token id retrieves the same vector at all positions. A GPU
observer watching PLE gathers sees `(layer_ℓ, addr_t)` for each
token; `addr_t` is a 1:1 function of token id. **The cloud
recovers the prompt in plaintext from memory access patterns
alone, with zero inversion.**

This is strictly worse than Vec2Text: Vec2Text requires a learned
inverter; PLE access-pattern recovery requires only address bus
observation.

**The fix.** Keep the PLE table in TEE DRAM. 1.9 GB int8 fits
comfortably within a SEV-SNP CVM RAM budget; gather operations
in-TEE are bandwidth-bound, not compute-bound. Then pre-gather
the PLE vector, treat it as an activation, mask with `A` like
any other activation before GPU-side projections.

This is **non-negotiable** for any Gemma 3n deployment. PLE was
designed for on-device inference (CPU-resident on edge); keeping
it CPU-resident in the CVM costs nothing structurally and closes
the leak completely.

**Related literature.** SecEmb (HPCA 2025) discusses Path-ORAM for
embedding lookups at ~3× overhead. Unnecessary here — the whole
table fits in TEE memory.

### D.6 Could PLE-style amortization help our own scheme?

No. Per-batch fresh Haar mask is load-bearing for GELO's
security argument; caching `A_t` across batches collapses to a
fixed mask (known broken family). Householder sampling is already
cheap (~9 ms / forward per `gelo.md` §8); there's no significant
cost to amortize.

### D.7 The "all-CPU-TEE" alternative for Gemma E4B (3n or 4)

E4B at int8: weights ≈ 4–4.5 GB (Gemma 3n) or ≈ 4 GB (Gemma 4) +
PLE table ≈ 1.3 GB int8. Total in-TEE footprint ≈ 5.5–6 GB.
A SEV-SNP CVM on Genoa (96 cores) hits ~5–10 TFLOPS int8 —
comfortably enough for ~5–15 t/s on a 4B model running entirely on
CPU without any GPU offload.

**If 10 t/s is acceptable** for the target use case, the
GELO+TwinShield + GPU machinery is unnecessary complexity for E4B.
PLE leak disappears (no off-TEE accesses). Attention placement
becomes trivial (everything is in-TEE). Mask machinery becomes
unnecessary.

**Recommendation**: before designing Gemma E4B GPU offload (either
3n or 4), **run the all-CPU-TEE benchmark on Genoa**. One week of
work; the answer determines whether the whole offload story for
E4B is worth building.

### D.9 New surface in Gemma 4 — multimodal encoders

Gemma 4 E2B/E4B include native audio (~300 M params) and vision
(~150 M params) encoders. Audio and image tokens enter the LLM
through these encoders before hitting the transformer trunk.

Open questions not yet addressed by this round:
- Do encoder activations leak the same way text embeddings do
  (Vec2Text-style), or differently (e.g., is the audio-encoder
  spectrogram-recoverable from its output)?
- Does GELO masking compose with the audio/vision encoders, or only
  with the transformer trunk downstream of them?
- Are there per-modality attack classes? (E.g., does the image
  encoder leak via patch-routing patterns the way MoE leaks via
  expert routing?)

These are P3-priority research items, gated on whether
audio/video are target modalities for our RAG pipeline.

### D.10 26B A4B is composite (hybrid + MoE)

Gemma 4 26B A4B is the only model in the family that combines
hybrid attention (5:1, W=1024) with MoE (128 experts, 8 active,
3× shared expert). **It needs both the §C MoE defenses
(router-in-TEE, CryptoMoE balancing) and the §D hybrid-attention
placement strategy** on the same forward pass. Engineering is the
union, not the sum — most of the code paths are independent (MoE
in FFN, hybrid in attention).

### D.8 Engineering effort summary

| Component | Composes with GELO? | Code reuse | Engineering cost |
|---|---|---|---|
| Local sliding-window layers in-TEE | Cleanly | ~85% | Add windowed-attention kernel in-TEE |
| Global attention layers | Cleanly via existing OutAttnMult / permuted-attn | ~95% | Reuse existing path |
| Block-diagonal permuted attention for local layers | New construction | ~70% | ~1 month + security write-up |
| PLE table in TEE DRAM (gather + mask) | Required | ~70% | ~2 weeks; ~2 GB CVM mem budget |
| MatFormer slice selection | Orthogonal | ~100% | None (compile-time) |
| **All-CPU-TEE bench for E4B** | Trivial — no GPU | — | **1 week — should happen first** |

---

## E. OSNIP — Not on the Critical Path

Cao et al., arXiv 2601.22752 (Jan 2026). The user named this paper
explicitly, so the detailed analysis is in this section. Bottom line:
**OSNIP defends against a different adversary than GELO and cannot be
applied to commercial LLM APIs.**

### E.1 What OSNIP does

A **client-side encryptor network** `R_φ(h, k)` transforms a
token-embedding sequence `h = g(x)` into an "encrypted" embedding `z`,
then iso-norm-rescales to `z̃` so attention dot products are
unperturbed. The server runs the unchanged LLM forward pass on `z̃`.

The "semantic null space" `N_{δ,ϵ}^f(h)` is the intersection of the
δ-utility ball and the ϵ-orthogonality cone. Theorem 2.5 proves this
intersection is non-empty in high dimension with measure decreasing
as `exp(-(d-2)ϵ²/2)`. `R_φ` is **trained** to project into it via a
KL-utility + cosine-orthogonality + key-diversity loss.

### E.2 Threat model mismatch — disqualifying

**OSNIP's adversary:** semi-honest cloud LLM provider with full
white-box model access. Attempts KNN-retrieval / vocabulary-matching
reconstruction.

**Our adversary** (`gelo.md` §2): honest-but-curious GPU /
host operator with VRAM read access, co-located with our TEE.
Trusted: TEE silicon. Untrusted: the actor running the GPU.

These are different actors. OSNIP defends against the LLM **provider**
seeing your prompt; GELO defends against the **operator** seeing
activations crossing PCIe. Composing them defends against both
adversaries at once — but for our prototype, the LLM provider is
inside the TEE (we run the model ourselves), so the OSNIP defense
applies to no adversary we care about.

### E.3 Cannot apply to commercial LLM APIs

OSNIP requires:
1. A **trusted third party** with gradient access to the server's
   LLM, to train `R_φ`. OpenAI/Anthropic don't expose gradients.
2. The server to accept **raw embedding inputs** at layer 0.
   Commercial APIs take tokens, not embeddings.

For commercial-API RAG, OSNIP is inapplicable in principle.

### E.4 Possible future composability

If we ever ship a self-hosted serving stack where untrusted **clients**
want input privacy against our **operators** — the inverse of our
current setup — OSNIP could be a client-side layer composed atop
GELO+TwinShield. The defenses don't conflict (one is at layer 0
client-side, the other is in-TEE per-offload). But this is a
hypothetical deployment, not a roadmap item.

### E.5 Reproducibility flags

- No GitHub code as of 2026-05-18.
- Encryptor architecture and key-injection mechanism not specified
  in the paper HTML — reproducibility gap.
- The "100.1% BERTScore" figure cited in
  [`private-inference.md`](private-inference.md) §F.18 appears to be
  a retention ratio for a different model row (Llama-3.2 small).
  Qwen3-32B BERTScore is 99.9% (0.865 vs baseline 0.866). The cited
  number should be corrected on next pass.
- Not evaluated against the TEE/split-inference family (GELO,
  ObfuscaTune, STIP, SOTER, TSQP, TransLinkGuard, Centaur,
  ShadowNet). The "SOTA" claim covers DP-text mechanisms only
  (Cape, DYNTEXT, InferDPT).
- Not evaluated against an adaptive inverter trained on `(h, z̃)`
  pairs. "Depth Gives a False Sense of Privacy" (arXiv 2507.16372,
  USENIX Sec '25) exploits exactly this class of static-trained
  noisy-embedding defenses — OSNIP is in scope.

---

## F. Hardware Tracking

### F.1 TDISP / SEV-TIO — foundations landed, production ~12 months out

**Linux 6.19 (Oct 2025)** lands PCIe link-encryption + device-
authentication scaffolding (TSM core) and first SEV-TIO patches.
End-to-end confidential device assignment targets v6.20/v7.0.

**AMD SEV-TIO whitepaper** is published; builds on TDISP + IDE + SPDM.

**NVIDIA has not committed to TDISP.** Their CC path remains H100-CC
(integrated, expensive) or B200-CC. AMD GPU TDISP support is also
unannounced.

**Implication for us.** GELO+commodity-Vulkan-passthrough remains
the right bet through 2026. We should re-evaluate after Linux 7.0
ships and the first TDISP-capable consumer GPU appears.

### F.2 B200-CC shipping; price doesn't change the bet

Intel Trust Authority composite TDX+B200 attestation is GA in 2026.
Pricing ~$45–50K/GPU vs ~$2–15/hr cloud. H100-CC overhead 2–8%;
B200-CC overhead numbers not public yet.

**Implication for us.** Confirms commodity GPU + GELO wins by
~10× on capex. Strengthens, doesn't weaken, our position.

### F.3 AMD MI400 — no CC story

MI400 (CDNA-Next, 40 PFLOPS FP4, 432 GB HBM4) ships 2026, but AMD
has **no public confidential-compute roadmap** analogous to H100-CC.
Commodity Radeon (RX 7900 XTX) remains the only AMD option for
TEE-co-located GPU work.

---

## G. Recommended Research Spikes (Prioritized)

Ordered by ROI. Each item has been scoped against the agent
findings above.

### G.1 P0 — PLE table in TEE DRAM (Gemma 3n)

**Why first**: only blocking issue for Gemma 3n support; without
it, every prompt's token IDs leak via address bus observation.

**Effort**: ~2 weeks. 1.9 GB int8 table in TEE memory; gather +
mask kernel; verify access pattern from outside TEE shows no
table-side gathers.

**Gate**: Gemma 3n is on the model wishlist.

### G.2 P0 — All-CPU-TEE benchmark for Gemma 3n E4B

**Why first**: if this hits ~10 t/s on Genoa, the entire
GPU+GELO+PLE-leak-fix machinery is unnecessary for E4B.
**Cheapest possible answer.**

**Effort**: ~1 week. CPU-only inference (no Vulkan), measure
tokens/sec at int8 on 96-core EPYC CVM.

**Gate**: Same as G.1. Decide G.1 vs G.2 ordering by whether
the E4B numbers come in fast or slow.

### G.3 P1 — MoEcho replay against routing-in-TEE design

**Why**: validates the routing-flattening primitive against a
real attack before we land any MoE code.

**Effort**: ~3 weeks. Implement a strawman router-in-TEE +
balanced-dispatch on Qwen3-MoE-30B-A3B (smallest open MoE
without shared experts); replay MoEcho's expert-load attack;
confirm histogram is flat under realistic load.

**Gate**: MoE is on the model wishlist. The ADR
`docs/adr/NNNN-moe-private-inference.md` should land first.

### G.4 P1 — CryptoMoE balanced-dispatch + confidence-aware drop port

**Why**: the v1 MoE defense per §C.3–C.5. Without it, MoEcho
recovers prompts.

**Effort**: 2–4 weeks engineering + 2–3 weeks accuracy
measurement on a real benchmark.

**Gate**: G.3 confirms the primitive defeats MoEcho.

### G.5 P2 — TwinShield-Xue softmax outsourcing security analysis

**Why**: only published 2025 scheme that pushes softmax to GPU
without the Amulet identity. If `e^(X+R)` blinding survives
Hidden-No-More-class analysis, it's a candidate alternative to
permuted attention.

**Effort**: 1–2 weeks security analysis spike. No engineering
yet. Outcome is a write-up: either "broken because X" or "worth
prototyping."

**Gate**: None — independent of model roadmap.

### G.6 ~~P2 — AloePri rotation cadence read~~ → COMPLETED 2026-05-18

Result: static per-deployment. AloePri does not apply under
openweight; see [`aloepri-vs-gelo.md`](aloepri-vs-gelo.md).

### G.6b P1 — Port AloePri empirical attack suite

**Why**: the seven attacks in `sheng1feng/Aloepri/src/security_qwen/`
(VMA, IA, ISA, IMA, NN, TFMA, SDA) are model-agnostic and apply
to any obfuscation scheme. Currently we have no empirical
attack benchmark for GELO; `gelo.md` §8 lever #5 mentioned this
as a future spike. AloePri's suite is broader and better-curated
than the Game-of-Arrows reference originally suggested.

**Effort**: 2–3 weeks. Phase 1: snapshot harness in
`InProcessTrustedExecutor`. Phase 2: wire attacks via pinned
AloePri commit at `evals/aloepri-attacks/`. Phase 3: integrate
into release-gate CI with TTRSR threshold.

**Gate**: None — high ROI standalone item.

### G.7 P3 — Block-diagonal permuted attention for sliding-window

**Why**: gives us Gemma 3 long-context (>16K) with local layers
on GPU. Not needed until we're chasing long context.

**Effort**: 1 month math + engineering. Security write-up needs
~2 weeks of review.

**Gate**: Demand for long-context Gemma 3 generation.

### G.8 P3 — KV-Cloak composition check during SCX adoption

**Why**: when we eventually land SCX (`gelo-llm.md` §4.3), need to
verify it survives the Shadow-in-the-Cache attacks or compose
with KV-Cloak's per-block permutation.

**Effort**: ~1 week reading + ~1 week composition design.

**Gate**: SCX work commences (`gelo-llm.md` §6 step 7).

### G.9 P4 — TEE.Fail threat-model paragraph

**Why**: minor documentation addition to `gelo.md` §6 explaining
why per-batch full-rank masking is stronger than longer-lived
secrets under bus-interposition adversaries.

**Effort**: 1 hour.

**Gate**: Next pass on `gelo.md`.

---

## H. What Did Not Move

For completeness, here are the round-1 findings that round-2 did
not invalidate or extend:

- **Permuted attention is correct and the engineering is reusable**
  (`gelo.md` §3.5b). Round 2 only adds the negative result for
  sliding window (§D.3); the dense path is unchanged.
- **SCX is still the right decode-phase primitive** (`gelo-llm.md`
  §4.3), with one new caveat (§G.8: compose with KV-Cloak).
- **Petridish-style full-CVM LLM serving** with B200-CC remains
  available at the high end; we still don't target it because
  cost is ~10× ours.
- **Pure-MPC / pure-FHE LLM inference** has not crossed the
  interactivity threshold. PermLLM's 3 s/token (LAN) is still the
  MPC frontier. Euston (S&P '26) makes FHE LLaMA-3-8B prefill
  ~15 s on GPU but autoregressive decode multiplies by N tokens.

---

## References (new in round 2)

- Xue, Zhao, Zheng, Yao, Solihin, Lou. *Securing Transformer-based
  AI Execution via Unified TEEs and Crypto-protected Accelerators*.
  arXiv 2507.03278 (Jul 2025). **Naming collision; not our scheme.**
- Lin et al. *AloePri: Towards Privacy-Preserving LLM Inference via
  Covariant Obfuscation*. arXiv 2603.01499 (Mar 2026). ByteDance.
- Yan et al. *Comet: Accelerating Private Inference via Activation
  Sparsity Prediction*. arXiv 2505.07239, S&P 2025.
- Yu et al. *CMIF: Towards Confidential and Efficient LLM Inference
  with Dual Privacy Protection*. arXiv 2509.09091, DASFAA 2025.
- Cunningham. *Privacy-Aware Split Inference with Speculative
  Decoding*. arXiv 2602.16760 (Feb 2026).
  github.com/coder903/split-inference.
- Tech, Seto, Berrios, van Schaik, Garman, Genkin. *TEE.Fail: DDR5
  Bus Interposition Defeats SGX, TDX, and SEV-SNP*. tee.fail (Oct
  2025).
- *Shadow in the Cache* + KV-Cloak defense. arXiv 2508.09442 (Aug
  2025).
- Le, Wang et al. *MoEcho: Side-Channel Attacks on Mixture-of-Experts
  Routing*. arXiv 2508.15036, CCS 2025.
- Anonymous. *Expert Selections Reveal (Almost) As Much As Text*.
  arXiv 2602.04105.
- *CryptoMoE: Secure 2PC Inference for Mixture-of-Experts with
  Balanced Routing*. arXiv 2511.01197 (Nov 2025).
- *SecMoE: Select-Then-Compute Private MoE Inference via Lattice HE*.
  arXiv 2601.06790.
- Carlini et al. *Stealing User Prompts from Mixture-of-Experts*.
  arXiv 2410.22884, ICLR 2025.
- Cao et al. *OSNIP: Breaking the Privacy-Utility-Efficiency Trilemma
  in LLM Inference via Obfuscated Semantic Null Space*. arXiv
  2601.22752 (Jan 2026).
- Gemma Team. *Gemma 3 Technical Report*. arXiv 2503.19786.
- DeepWiki / Alan Dao notes on reverse-engineering Gemma 3n PLE.
- Wang, Huang, Liang et al. *SoK: Analysis of Accelerator TEE
  Designs*. NDSS 2026.
- Chrapek, Copik, Mettaz, Hoefler. *Confidential LLM Inference:
  Performance and Cost Across CPU and GPU TEEs*. arXiv 2509.18886
  (Sep 2025).
- Tang, Flemings, Wang, Annavaram. *DP-KSA: Differentially Private
  Retrieval-Augmented Generation*. arXiv 2602.14374.
- Nillion. *Fission: Hybrid MPC + Evaluator Network for Non-Linear
  Operators*. eprint 2025/653.
- AMD. *SEV-TIO Whitepaper*. Public, 2025.
- Linux Kernel 6.19 PCIe link-encryption merge. Phoronix coverage,
  Oct 2025.
- Intel Trust Authority composite TDX+B200 attestation GA, 2026.
