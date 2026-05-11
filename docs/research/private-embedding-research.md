# Private Embedding Generation

> Research date: 2026-04-21. Embedding-specific framing for private transformer inference: **how to run an embedding model (BERT-base, gtr-t5-base, mxbai-embed-large, etc.) such that the operator cannot see the input text**.
>
> **This doc is a lens on `private-inference.md`.** The general catalogue of MPC / FHE / TEE / obfuscation / TEE-split-inference systems lives there. Below we (1) anchor the embedding-specific threat (Vec2Text, EDNN), (2) list embedding-stage-specific applicability notes, and (3) map recommendations to the RAG embedding step.
>
> **Sibling docs:**
> - `private-inference.md` §A MPC, §B FHE, §C TEE, §E TEE Split-Inference, §F Obfuscation — canonical system entries
> - `private-reranking-research.md` — reranker-stage adaptations
> - `private-information-retrieval.md` — retrieval-stage privacy (query + access-pattern)
> - `fhe-encrypted-vector-db.md` — encrypted vector storage

---

## Why This Matters: The Embedding Inversion Threat

Before examining defenses, it is worth anchoring on what the threat actually is.

**Vec2Text** (Morris, Kuleshov, Shmatikov, Rush — EMNLP 2023, "Text Embeddings Reveal (Almost) As Much As Text").
An iterative correction-based attack that re-embeds reconstructed text and nudges toward the target embedding. Recovers **92% of 32-token inputs exactly** from `text-ada-002` and GTR-base; full names recovered from clinical notes. Published code.
→ *Implication*: sending raw embeddings to an untrusted vector DB or embedding service is essentially equivalent to sending plaintext.

**EDNN** (Lin et al. 2024) — model-agnostic nearest-neighbor inversion; ~100% token recovery even without the embedding model's weights.

These two attacks are the empirical bar any private-embedding scheme must survive.

---

## Where Privacy Can Be Applied in the Embedding Step

The embedding step has three parties: the **text owner** (client), the **embedding-model host** (server), and — for RAG — the **vector store** (which may be the same server or separate). Privacy can be applied at three points:

| Point | What is hidden | What the approach looks like |
|---|---|---|
| **A. Input text ↔ model** | Plaintext text from the embedding-model host | MPC / FHE / TEE inference (`private-inference.md` §A–C) |
| **B. Intermediate activations / output embedding** | Activations from a split server; output from a downstream consumer | Split inference (`private-inference.md` §E), DP perturbation (RemoteRAG, DP-Forward), obfuscation (SGT, ObfuscaTune, OSNIP) |
| **C. Model weights** | Model from the client | Same MPC/FHE/TEE approaches, both-way privacy; commercial TEE products |

Most RAG deployments care about (A) and (B): the user wants the text hidden from the embedding service *and* the stored embedding to not leak the text to whoever accesses the vector DB.

---

## Applicability to RAG Embedding Stage

Embedding models are **small transformers** (BERT-Base / Large class, 110M–340M params) — much smaller than generation LLMs. This shifts the cost curve:

- **MPC systems** (PUMA, SHAFT, NEXUS, BOLT, BumbleBee) achieve seconds-per-BERT-base inference on LAN (see `private-inference.md` §A for full numbers). Encoder-only inference is 5–20× cheaper than decoder-only LLM inference under MPC because there is no autoregressive loop.
- **FHE** (THE-X, CipherFormer) for BERT-Base is minutes-per-inference — still impractical for interactive embedding but feasible for batch ingestion.
- **TEE** (H100 Confidential Compute, TDX, SGX) is the clear production choice — 4–10% overhead for any model size, including embedding models (see `private-inference.md` §C and §commercial section).
- **TEE split-inference** (GELO, ObfuscaTune-encoder variant) applies cleanly to encoder models — the TEE holds a small embedding table + pooling head while the GPU runs transformer blocks on obfuscated activations. Smaller TEE footprint than running the whole encoder in the enclave.
- **DP / Obfuscation** (RemoteRAG, DP-Forward, SGT, OSNIP) apply specifically to the **output embedding** and are **embedding-stage-native** — they are not generation techniques and belong primarily here.

### Embedding-stage-native approaches

These are the systems whose threat model targets the embedding output (not generation) specifically:

- **RemoteRAG** (Cheng et al., ACL Findings 2025) — `(n,ε)-DistanceDP` on the client's computed query embedding before sending to the retrieval server. 0.67 s end-to-end, 46.66 KB communication at 10⁵ documents. Requires client-side embedding model.
- **DP-Forward** (Du et al., CCS 2023) — matrix Gaussian noise in the forward pass to satisfy `(ε,δ)-SeqLDP`. 88pp reduction in Vec2Text success, ~0 ms overhead. Applies at inference time without re-training.
- **Stained Glass Transform / SGT** (2025) — learned affine map that obfuscates the embedding sequence. No inference overhead; 93% nearest-neighbor reconstruction failure at −0.38pp accuracy on 70B models. Evaluation on Llama 3.2 1B through 70B.
- **OSNIP** (Cao et al. 2026, already in Edgequake) — null-space projection of input embeddings against a specific LLM's gradient. 0.96 ms/prompt overhead, KNN attack success 0.000. Target model must be known — incompatible with closed-source APIs.
- **TextObfuscator** (Zhou et al., ACL Findings 2023) — cluster-level obfuscation where token representations collapse to prototypes. Zero inference overhead; weakest privacy (cluster-level, not token-level).

### Split inference applied to embedding

Of the TEE split-inference systems in `private-inference.md` §E, the two relevant to encoder-only embedding are:
- **ObfuscaTune-encoder variant** (extrapolation from the published decoder version) — TEE holds embedding table + pooling head; GPU runs BERT transformer blocks on `Q·H`. Smallest TEE footprint.
- **GELO** (published for decoder LLMs) — TEE holds non-linear ops (GeLU, LayerNorm) and matrix `A`; GPU runs linear projections on `A·H`. Encoder variant is architecturally identical since BERT has the same linear+non-linear structure.

Both require a co-located TEE + GPU on the embedding server. Neither is published with encoder-only numbers.

---

## Practical Guidance for Embedding Generation

| If you need... | Best approach | Source |
|---|---|---|
| Cryptographic proof, no hardware trust | **NEXUS** (BERT-base, 37.3 s CPU / 0.88 s GPU, 164 MB comm, 1 round) | `private-inference.md` §A.6 |
| Practical performance, accept hardware trust | **H100 Confidential Compute** or TDX (4–8% overhead) | `private-inference.md` §C |
| Client controls embedding entirely; store encrypted in DB | **Client-side embed + CAPRISE / ADCPE** for storage | `fhe-encrypted-vector-db.md` |
| DP bound on query-embedding leakage | **RemoteRAG** (0.67 s, formal DP) or **DP-Forward** (~0 ms, SeqLDP) | Embedding-native, above |
| No model changes, just obfuscation | **Stained Glass Transform** (affine, ~0 ms overhead, 93% NN-fail) | Embedding-native, above |
| Minimize TEE footprint with GPU offload | **GELO** (76% on GPU, 20–30% overhead) or **ObfuscaTune-encoder variant** | `private-inference.md` §E.2, E.3 |
| Null-space projection for a known target LLM | **OSNIP** (0.96 ms/prompt, 0.000 KNN success) | Embedding-native, above |
| Commercial deployment today | **Privatemode / Opaque / Fortanix** | `private-inference.md` §commercial |

### The tradeoff spectrum for embedding generation

1. **Formal cryptographic security** — MPC (seconds/BERT-base), FHE (minutes/BERT-base). See `private-inference.md` §A, §B.
2. **TEE + GPU split** — 20–30% overhead, informational privacy (GELO, ObfuscaTune-encoder). See `private-inference.md` §E.
3. **TEE-only** — 4–10% overhead, hardware trust required. See `private-inference.md` §C.
4. **Embedding-output obfuscation** — near-zero overhead, computational privacy: SGT, OSNIP, DP-Forward (formal SeqLDP), TextObfuscator (weakest).
5. **Client-side embedding + encrypted storage** — zero server-side crypto, requires the client to hold the embedding model. See `fhe-encrypted-vector-db.md`.

### The unsolved problem

No system currently achieves all of: (a) formal cryptographic security, (b) sub-second latency, (c) no client-side embedding requirement, (d) compatibility with existing pre-trained weights. TEE + GPU split inference (GELO-class) is the closest compromise: 20–30% overhead, informational privacy, keeps pre-trained weights intact, runs server-side.

---

## Embedding-Specific Implications of the Shared Inference Systems

This section flags where a system in `private-inference.md` has embedding-specific nuances that are not obvious from its generation-oriented write-up.

- **PUMA / SHAFT / NEXUS / BOLT** (MPC): encoder-only (BERT-base) inference is ~5–20× cheaper than their reported LLaMA numbers because there is no autoregressive loop. Seconds-per-inference under LAN is realistic for batch embedding pipelines.
- **THE-X / CipherFormer** (FHE): designed for encoder models; numbers quoted are already for BERT-class. Extending to larger encoder models (`mxbai-embed-large`, 335M) scales linearly with depth.
- **GELO** (TEE split): encoder variant is architecturally the same as the decoder version — both have linear + non-linear op blocks. Per-batch `A` fresh per query is fine for RAG since each embedding call is independent.
- **ObfuscaTune**: the encoder variant replaces "embedding table + lm_head in TEE" with "embedding table + pooling head in TEE". The scoring head is tiny (a Linear layer, kilobytes), so the TEE footprint is dominated by the embedding lookup table (~90 MB for BGE-reranker-class; much less for compact embedding models).
- **Commercial TEE products** (Privatemode, Opaque, Fortanix): treat embedding inference as "one more model served in a CVM" — no embedding-specific path. Use their standard attested-TEE API.

---

## Open Research Gaps Specific to Embedding

- **No published encoder-specific MPC system outperforming the generic MPC transformer systems.** Encoder inference is structurally cheaper but no paper optimizes specifically for it.
- **No published ObfuscaTune-encoder or GELO-encoder variant with benchmark numbers.** The architectural pattern is sound; the engineering is not done.
- **Embedding-output DP bounds don't protect storage.** RemoteRAG / DP-Forward bound what is inferable from a noisy embedding, but once stored, the noisy embedding is still queryable. Storage-level distance-preserving encryption (CAPRISE / ADCPE — `fhe-encrypted-vector-db.md`) is the complementary piece.
- **Stained Glass Transform** (SGT) is the most practical obfuscation but has not been independently reproduced on standard embedding benchmarks (MTEB, BEIR) against current inversion attacks. Its PAC-Privacy advantage bound of 12.69% means residual leakage exists; empirical robustness at MTEB scale is unverified.

---

## Implementation Deep-Dive (2026-05-11 update)

> Research-scouting pass focused on practical algorithms + reference code for piecing together a private embedding model execution. Inputs: BERT-class encoders (also ColBERT, decoder-LLM-as-embedder), pooling strategies (mean/CLS/last-token), output dimension (truncation/quantization), and private-compute substrate (MPC / FHE / TEE / statistical).

### D1. Non-linear operator approximations (the BERT bottleneck)

The four ops below are 60–90 % of total cost in every published private-BERT pipeline. Picking the approximation defines the system.

#### Softmax

| System | Approach | Acc. hit (GLUE) | Code |
|---|---|---|---|
| **PUMA** (arXiv [2307.12533](https://arxiv.org/abs/2307.12533)) | clip negatives, truncated Taylor `exp` (5 iter, T=−14), one reciprocal | <0.011 abs | [secretflow/spu](https://github.com/secretflow/spu) (Apache-2.0) |
| **BumbleBee** (NDSS '25, [ePrint 2023/1678](https://eprint.iacr.org/2023/1678)) | segmented poly `exp`, mixed-ring | 0.31 % | [AntCPLab/OpenBumbleBee](https://github.com/AntCPLab/OpenBumbleBee) |
| **BOLT** (S&P '24) | Remez polynomial fit + HE+ASS | +0.4 % after FT | [Clive2312/BOLT](https://github.com/Clive2312/BOLT) (partial) |
| **NEXUS** (CCS '24) | pure RNS-CKKS, direct Taylor | 0.25 % (note: breaks on large input range) | [zju-abclab/NEXUS](https://github.com/zju-abclab/NEXUS) (GPL-3) |
| **Iron** (NeurIPS '22) | SIRNN-style LUT via OT | -0.41 % | [xingpz2008/Iron](https://github.com/xingpz2008/Iron) (rough) |
| **MPCFormer** (ICLR '23) | **2Quad**: `softmax_i ≈ (x_i+c)² / Σ(x+c)²` — quadratic replacement, needs KD | 1–9 % w/o KD; ~1 % after | [DachengLi1/MPCFormer](https://github.com/DachengLi1/MPCFormer) |
| **SecFormer** (ACL '24) | 2Quad + Goldschmidt normalization | 0.9 % | [jinglong696/SecFormer](https://github.com/jinglong696/SecFormer) |
| **SHAFT** (NDSS '25, [ePrint 2025/2324](https://eprint.iacr.org/2025/2324)) | **first constant-round softmax** (ODE characterization + input clipping) | 0.9–1.1 % | open-sourced per paper |
| **SIGMA** (PETS '24) | FSS LUT for max/exp/reciprocal, **GPU** | 0.34 % | MS internal — not on GitHub |
| **Nimbus** (NeurIPS '24) | distribution-aware cubic on active range, linear tails | 0.08 % | merged into SPU |

Take-away: for buildable + no retraining → **PUMA's polynomial** or **BumbleBee's segmented poly**. If retraining is OK → **MPCFormer/SecFormer 2Quad** (cheapest protocol). If WAN is the deployment → **SHAFT's constant-round** wins.

#### GELU

| System | Approach | Acc. hit | Code |
|---|---|---|---|
| **PUMA** | piecewise: 0 / cubic / deg-6 poly / linear on 4 intervals; max err 0.014 | <0.011 | SPU |
| **SHAFT** | Fourier-series characterization; max err 4.6e-3; one round + two muls fewer than BOLT | 51 % lower max err vs BOLT | open-sourced |
| **SecFormer** | segmented + Fourier on [-1.7,1.7] with 7 sine terms | 0.9 % | SecFormer |
| **MPCFormer** | **Quad**: `GELU(x) ≈ 0.125x² + 0.25x + 0.5` — needs KD | 4–9 % w/o KD | MPCFormer |
| **BumbleBee** | per-segment poly + mixed-ring | 0.31 % | OpenBumbleBee |
| **Nimbus** | dist-aware square poly on [-2.1, 0.2] | 0.08 % | SPU |

For pretrained checkpoints with no retraining → **PUMA's piecewise poly** is the de-facto baseline; **SHAFT's Fourier** is strictly more accurate and open-source.

#### LayerNorm (the hidden inverse-sqrt cost)

The expensive piece is `1/√(var+ε)`. SHAFT measures this as **43–45 % of all LayerNorm rounds**.

- **SecFormer's Goldschmidt iteration** for `1/√x` — multiply-only, 2–3 rounds; cleanest MPC building block. Recommended.
- **PUMA's `Π_rSqrt`** — directly computes `σ^(-1/2)` (not sqrt-then-reciprocal); buildable in SPU.
- **FHE-side:** **Panda's degree-N polynomial inverse-sqrt for CKKS** ([ePrint 2022/423](https://eprint.iacr.org/2022/423)) is the standard citation.

#### Embedding lookup (cheap but easy to get wrong)

PUMA pattern: V parallel equality tests `Π_Eq(i, id)` → one-hot → matrix multiply with cleartext table. Lossless, identical semantics, cheap at BERT-base vocab. If the **table itself is secret**, **FABLE** ([USENIX '25, ePrint 2025/1081](https://eprint.iacr.org/2025/1081)) is the buildable spec. For **pure-FHE large-vocab**, **HE-LRM** ([arXiv 2506.18150](https://arxiv.org/abs/2506.18150)) gives 56× speedup via digit-decomposition.

### D2. End-to-end private BERT-base benchmarks (seq-len 128, LAN)

| System | Protocol | Latency | Comm | Acc. delta |
|---|---|---|---|---|
| **MPCFormer** | 2PC CrypTen | 55 s | 12 GB | -1.1 % |
| **PUMA** | 3PC SPU | 34 s | 11 GB | ≤-1.1 % |
| **SecFormer** | 3PC | 20 s | 24 GB | -0.9 % |
| **BumbleBee** | 2PC HE+ASS | ~50–235 s | 14 GB | -0.3 % |
| **BOLT** | 2PC HE+ASS | 185 s | 26 GB | +0.4 % (after FT) |
| **SHAFT** | 2PC SS | ~50 s LAN; **best WAN** | 25–41 % less than BumbleBee | -0.9 % |
| **SIGMA** | 2PC FSS **GPU** | **1.84 s** (preproc excluded) | 1 GB | -0.34 % |
| **NEXUS** | non-interactive FHE | 857 s | 0.16 GB | -0.25 % |
| **Iron** | 2PC HE+OT | 475 s | 281 GB | +0.4 % |
| **Ditto** (ICML '24) | quantization-aware 3PC | ~14–24 s | ~10 GB | "negligible" |
| **Nimbus** | 2PC | ~50–85 s | ~10 GB | -0.08 % |
| **THE-X** (ACL '22) | FHE | ~4700 s (BERT-Tiny extrapolated) | — | -0.34 % |

Skeptical flags: SIGMA's 1.84 s excludes preprocessing (FSS keys 45 GB+); THE-X is BERT-Tiny only and has no public code; NEXUS's softmax is known-broken on large input ranges; Iron's comm is impractical; CipherFormer was only benchmarked on small classifiers.

### D3. Pooling, head, normalization — the cost the papers omit

Every BERT-private-inference benchmark stops at the encoder output. For *embedding generation* you still need:

| Op | Cost under MPC/FHE | Notes |
|---|---|---|
| CLS pool | **free** | Select position 0 on shared tensor |
| Mean pool | **free** | Sum on additive shares; divide by public `seq_len` (i.e. multiply by `1/n` in plaintext) |
| Last-token pool | free *if* `seq_len` public, oblivious-select otherwise | Common for decoder-LLM embedders (E5-Mistral, Qwen3-Embedding) |
| Projection head | one small linear (~1 % of total) | Same protocol as any linear |
| **L2 normalize** | **non-trivial** — one inverse-sqrt on `‖x‖²` | **Not in published latencies.** Budget ~one LayerNorm-equivalent. Use SecFormer Goldschmidt in MPC / Panda 2022/423 in FHE. |

If both query and stored vectors are L2-normalized, dot product == cosine — but the **‖q‖** normalize step under MPC is still a per-query cost. None of PUMA/BumbleBee/BOLT/NEXUS/SHAFT/SIGMA/Iron/MPCFormer reports it.

### D4. Architectural variants

- **ColBERT (multi-vector).** Same encoder forward pass, no pooling — protocol identical to BERT but token-level vectors are *worse* against Vec2Text than pooled vectors. The MaxSim late-interaction op (`Σ_q max_d cos(q_i, d_j)`) requires a max-per-query-token under MPC; no published private-ColBERT system exists. Adjacent: token-pooling research (Answer.AI 2024) suggests aggressive pre-pooling could reduce both leakage and private-retrieval cost.
- **Decoder-LLM-as-embedder (E5-Mistral 7B, NV-Embed v2, GTE-Qwen2-7B, Qwen3-Embedding-8B, Gemini Embedding).** No paper benchmarks single-forward-pass embedding under privacy. Loose extrapolation from PUMA/SIGMA Llama numbers: tens of minutes per embedding on LAN. Practical path: **fall back to BGE-base/E5-base/Qwen3-Embedding-0.6B** (110M–600M params, BERT-base-shaped) and accept the MTEB delta.

### D5. Embedding-output obfuscation — implementation details

#### DP-Forward (CCS '23, arXiv [2309.06746](https://arxiv.org/abs/2309.06746), code: [xiangyue9607/DP-Forward](https://github.com/xiangyue9607/DP-Forward))

Privacy: `(ε,δ)`-SeqLDP at the *whole-sequence* level. Mechanism: **Analytic Matrix Gaussian** (Balle-Wang 2018 lifted to matrices).

```python
# transcribed from DP-Forward's dp_noise.py
def matrix_gaussian_noise(eps, delta, sensitivity):
    phi = lambda t: (1 + erf(t/sqrt(2))) / 2
    delta_0 = phi(0) - exp(eps) * phi(-sqrt(2*eps))
    B_plus  = lambda v: phi( sqrt(eps*v))     - exp(eps)*phi(-sqrt(eps*(v+2)))
    B_minus = lambda u: phi(-sqrt(eps*u))     - exp(eps)*phi(-sqrt(eps*(u+2)))
    B = B_plus if delta >= delta_0 else B_minus
    u_star = bisect(B - delta, 0, 1e5)        # ~5000 iters
    alpha  = sqrt(1+u_star/2) + (-1 if delta>=delta_0 else +1)*sqrt(u_star/2)
    R      = sqrt(2*eps) / alpha
    return sensitivity / R                    # σ

def add_noise(x, eps, delta, C=1.0):
    x = l2_clip_rows(x, C)                    # per-row clip → Δ₂(f) = 2C
    sigma = matrix_gaussian_noise(eps, delta, sensitivity=2*C)
    return x + sigma * N(0, I, x.shape)
```

Defaults from the released code: `noise_layer=10`, position `add_and_norm_2`, `C=1.0`, `δ=1e-5`. For RAG you noise the *final pooled* embedding, not an internal layer. Empirical: Vec2Text success drops up to **88 pp** at moderate ε; <1–2 pp utility loss on SST-2/QNLI; 3× faster than DP-SGD. ~30 lines to port to Rust.

#### RemoteRAG (ACL Findings '25, arXiv [2412.12775](https://arxiv.org/abs/2412.12775))

Privacy: `(n,ε)`-DistanceDP where **`n` is the embedding dimension** (not a cluster count — common misreading). Mechanism: multivariate planar Laplace.

```python
# n = embedding dim
r ~ Gamma(shape=n, scale=1/eps)               # radial magnitude
z ~ N(0, I_n);  v = z / ||z||                 # uniform direction on S^{n-1}
e_q_noisy = e_q + r * v
```

Two-stage protocol:
1. Send `e_q_noisy` → server returns top-`k'` candidates (`k'>k`).
2. Send PHE-encrypted `e_q` → server computes encrypted cosine over the `k'` candidates → client decrypts and sorts.

Result: **100 % recall@k** at N=10⁶ documents, 0.67 s end-to-end, 46.66 KB comm. The over-fetch + PHE rerank is what recovers exact recall even when noise re-orders candidates. Code not released — re-implementation is ~200 lines plus a Paillier/CKKS dependency.

Recommended ε: `10n`–`50n` for dim ∈ [384, 1536], i.e. L2 perturbation 0.02–0.10 on unit-norm vectors.

#### Stained Glass Transform (Protopia, arXiv [2506.09452](https://arxiv.org/abs/2506.09452))

**Input-conditioned** Gaussian: `x̃ = x + μ_θ(x) + Σ_θ^{1/2}(x) · u`, with `μ_θ, Σ_θ` learned small transformers. PAC-Privacy bound 12.69 % is *per-token adversary advantage over uniform-vocab guessing*, **not** sequence reconstruction. Empirical: 93 % NN-reconstruction failure, 86.66 % BeamClean failure, −0.29 to −0.46 % MMLU/ARC/PIQA utility on 70B targets. **No public code** — productized in Protopia's closed-source engine. Two extra training stages (target-LLM-distillation + MI minimization) and a frozen target LLM make this a heavy lift.

#### OSNIP (Cao et al. 2026, arXiv [2601.22752](https://arxiv.org/abs/2601.22752))

Name is a *misnomer*: there is no analytic null-space projection. It's a trained encryption network `R_φ(h, k)` + iso-norm scaling.

```
z      = R_φ(h, k)                                  # learned per-user-key map
output = (h + z) * ||h||₂ / ||(h+z)||₂              # iso-norm projection
```

Training objective combines KL-with-frozen-target-LLM (utility), cosine-orthogonality `|cos(h,z)| < ε` (privacy), and key-separation `||R_φ(h,k₁) − R_φ(h,k₂)|| > δ` (unlinkability). Empirical: KNN inversion ASR 0.000–0.066 vs 0.334–0.578 for CAPE; ~0.96 ms/prompt on Qwen3-32B (network forward + scaling only). **Paper omits `R_φ` architecture, key-mixing recipe, and λ values** — reproducibility is non-trivial. **No code released** at submission.

#### TextObfuscator (ACL '23)

Replace token reps with `prototype + small Gaussian noise` at an intermediate layer. Trained with cluster-contrastive loss. **Weakest privacy** (cluster-level leakage = "this is a person name", "this is a verb of motion"). No code, no `(ε,δ)` guarantee. Skip for RAG.

#### 2024–2026 honorable mentions

- **Eguard** ([arXiv 2411.05034](https://arxiv.org/abs/2411.05034)) — 24-layer RoBERTa projector trained with MI loss; Vec2Text F1 94 → 5; no code; heavy.
- **SPARSE** (ICLR '26, [arXiv 2602.07090](https://arxiv.org/abs/2602.07090)) — concept-aware mask + **anisotropic Mahalanobis-Laplace** noise; 92 % Vec2Text reduction at ε=5; 65 % downstream utility; code "planned."
- **NVDP / NVIB** ([arXiv 2601.02307](https://arxiv.org/abs/2601.02307)) — variational information-bottleneck layer with Rényi-DP guarantee.
- **ObfuscaTune** ([arXiv 2407.02960](https://arxiv.org/abs/2407.02960)) — random orthogonal `Q` on hidden states with `Q⁻¹·W` weight adjustment. **Statically keyed → broken by Soter/TSQP-style attack** ([arXiv 2602.11088](https://arxiv.org/abs/2602.11088)) which recovers a Llama-3 8B layer in ~6 min. **Require per-batch refresh** to be secure.

#### D5 summary — embedding-output obfuscation

| System | Mechanism | Privacy notion | Best defense metric | Utility cost | Client latency | Needs training? | Code |
|---|---|---|---|---|---|---|---|
| **DP-Forward** | aMGM Gaussian on pooled emb | `(ε,δ)`-SeqLDP (formal) | Vec2Text −88 pp | ≤1–2 pp GLUE | ~10 µs (Rust) | no | [xiangyue9607/DP-Forward](https://github.com/xiangyue9607/DP-Forward) |
| **RemoteRAG** | planar Laplace + PHE rerank | `(n,ε)`-DistanceDP (formal) | Vec2Text SacreBLEU 50→10 | 100 % recall@k preserved | 0.67 s e2e | no | none |
| **SGT** | input-cond. Gaussian (`μ_θ, Σ_θ` nets) | PAC-Privacy 12.69 % adv. bound | NN-recon 93 % fail; BeamClean 87 % | −0.29 to −0.46 % MMLU/ARC | one small-tx forward | yes | none (Protopia closed) |
| **OSNIP** | learned `R_φ(h, k)` + iso-norm | empirical (KNN, vocab-match) | KNN ASR 0.000–0.066 | ~+0.1 % (within noise) | 0.96 ms | yes | none |
| **TextObfuscator** | prototype substitution at layer L | none (cluster-level leakage) | "meaningless words" | ~1 pp | trivial | yes | none |
| **Eguard** | 24-layer RoBERTa projector + MI loss | empirical MI bound | Vec2Text F1 94 → 5 | ~1–2 pp | heavy (24 tx) | yes | none |
| **SPARSE** | concept mask + Mahalanobis-Laplace | metric-DP (formal) | Vec2Text −92 % at ε=5 | 65 % utility retention | trivial | yes | planned |
| **NVDP / NVIB** | NVIB layer + noise | Rényi-DP (formal) | GLUE-tested | tunable | one layer | yes | none |
| **ObfuscaTune** | orthogonal `Q` on input + `Q⁻¹W` weights | obfuscation — **broken under TSQP** | trivial against static key | 0 (lossless) | one matmul | one-time | none |
| **SanText / CusText / CAPE** | metric-LDP token replacement | LDP at token level | weaker than emb-layer | substantial recall drop | per-token sample | no | [SanText](https://github.com/xiangyue9607/SanText) |

Top picks for a Rust thin client: **DP-Forward** (only formal-DP scheme with public code) and **RemoteRAG** (formal-DP + retrieval-aware). Stack with INT8 + MRL (§D6) as belt-and-braces.

### D6. Hybrid / split / statistical (encoder-applicable)

#### GELO ([arXiv 2603.05035](https://arxiv.org/abs/2603.05035)) — TEE+GPU split with per-batch fresh mask

Split: TEE keeps LayerNorm, softmax, GeLU, residuals, FFN, attention bookkeeping. GPU runs Q/K/V/O projections only. For each batch:

```
A   ← sample fresh invertible (n×n)         # orthogonal + high-energy "shield" vectors
U   = A · H                                  # in TEE
GPU computes U · W                           # plaintext-weight GEMM on commodity GPU
TEE applies A⁻¹ on return                    # recovers H · W
```

Bit-exact float32 output. ~20–30 % latency overhead vs plaintext GPU on Llama-2 7B. **Encoder applicable** (BERT shares the same Q/K/V/O + LN + FFN structure). **No code** — paper-only; you port the math.

Why this is the *right* shape for encoder embedding: pooling happens in TEE on the unmasked stream, so the GPU never sees a pooled embedding. The shield-vector construction defeats orthogonal-Gram attacks; **read GELO §4 carefully before implementing** — naive orthogonal `A` is broken.

**Follow-up status (2026-05-11 check):** zero direct citations on Semantic Scholar / Google Scholar; no successor paper, no public code release, no published cryptanalysis specifically targeting fresh-per-batch hidden-state masking. The paper is ~2 months old in a slow subfield — expected, but means it is community-unvetted. Read these **concurrent siblings** alongside GELO:

- **Amulet** ([arXiv 2512.07495](https://arxiv.org/abs/2512.07495), Dec 2025) — same primitive (per-round fresh invertible masks), pushes obfuscation through non-linearities via absorb-shuffle-squeeze. **Already publishes BERT-base + GPT-2 numbers** (2.8–4.8× overhead vs unprotected GPU; 8–9× speedup vs full TEE) — the encoder evaluation GELO is missing. Active adversary threat model. No code.
- **TwinShield** ([arXiv 2507.03278](https://arxiv.org/abs/2507.03278), Jul 2025) — bidirectional info-theoretic masking + secret sharing + **U-Verify integrity layer** (challenge-response checksums on GPU returns). 87 % offload, 4.0–6.1× speedup over TEE-only on SGX+CUDA. **GELO has no integrity layer** — if you need retrieval-correctness guarantees, port this.
- **SecureInfer** ([arXiv 2510.19979](https://arxiv.org/abs/2510.19979), Oct 2025) — *opposite* split decision: Q/K/V/attn/FFN/residuals in SGX, other linear ops offloaded under XOR-OTP. 2.06× latency vs unprotected GPU. Decoder-only Llama-2 evaluation. Does not consider BSS/ICA attacks (GELO's main contribution).

**Attack that doesn't break GELO but reviewers will ask about:** **ArrowMatch / Game of Arrows** ([USENIX Security '25](https://www.usenix.org/conference/usenixsecurity25/presentation/wang-pengli)) — recovers >98 % of obfuscated weights via direction-similarity to *public pretrained* weights. Targets static-key weight obfuscation, **not** fresh-per-batch hidden-state masking, so plausibly safe by construction; security argument should make this explicit.

**Engineering recipe for Recipe D today:** start from the GELO threat model and BSS-hardness argument (its actual contribution), borrow Amulet's BERT engineering, add TwinShield's U-Verify integrity, and cite ArrowMatch as the baseline attack to defeat.

#### Slalom ([arXiv 1806.03287](https://arxiv.org/abs/1806.03287), code [ftramer/slalom](https://github.com/ftramer/slalom))

OTP-based linear-op offload + Freivalds integrity check. Information-theoretic privacy *if pads are refreshed*. Code is 2018 TF/SGX1, CNN-only, with prominent "DO NOT USE THIS FOR REAL DATA" disclaimer. Useful as architectural reference; the pad/Freivalds primitive transfers, but you rewrite.

#### H100 Confidential Compute baseline

PCIe AES-GCM + on-chip key isolation; **4–8 % throughput overhead** for batched LLM workloads ([arXiv 2509.18886](https://arxiv.org/abs/2509.18886), [arXiv 2409.03992](https://arxiv.org/abs/2409.03992)). High-QPS small-payload embedding workloads drift closer to **7–15 %** because PCIe overhead is amortized worse. **PipeLLM** ([arXiv 2411.03357](https://arxiv.org/abs/2411.03357)) pipelines AES with compute, bringing vanilla H100 CC overhead from 52–88 % to <19.6 % on OPT-30B/66B/175B — useful primitive even if you don't take the rest.

#### Statistical / info-theoretic levers

- **INT8 quantization (absmax/zeropoint)** drops Vec2Text BLEU by **~60 %** while leaving retrieval recall ~intact ([Vec2Text reproducibility, arXiv 2507.07700](https://arxiv.org/abs/2507.07700)). INT4 is fragile under adaptive attack.
- **Matryoshka truncation** (MRL, [arXiv 2205.13147](https://arxiv.org/abs/2205.13147)): no paper measures Vec2Text-vs-`k`. Plausible information-theoretic reduction; **runnable in a week** by anyone with Vec2Text + a Matryoshka embedder. Open research seam.
- **JL + Gaussian noise** (Blocki et al., [arXiv 1204.2606](https://arxiv.org/abs/1204.2606)) — formal `(ε,δ)`-DP with `(1±ε)` distance preservation. The only formal "smaller-dim = less leakage" result.
- **CLUB / NVIB / Eguard / SPARSE** — variational MI lower-bound objectives for training private encoders. CLUB has [code](https://github.com/Linear95/CLUB); the others don't.

#### D6 summary — hybrid / split / statistical

Split-inference and TEE-anchored systems:

| System | Split / mechanism | Threat model | Accuracy | Overhead vs plaintext | Code |
|---|---|---|---|---|---|
| **GELO** ([2603.05035](https://arxiv.org/abs/2603.05035), Mar 2026) | TEE: LN/softmax/GeLU/FFN; GPU: Q/K/V/O w/ per-batch fresh invertible mask + shield vectors | honest-but-curious GPU; single-batch BSS hardness | bit-exact float32 | 20–30 % latency on Llama-2 7B | none — **0 citations as of May 2026** |
| **Amulet** ([2512.07495](https://arxiv.org/abs/2512.07495), Dec 2025) | per-round fresh invertible masks through *all* layers via absorb-shuffle-squeeze | active adversary w/ full OS/HW/mem access | bit-exact | 2.8–4.8× vs unprotected GPU; **BERT-base + GPT-2 measured** | none |
| **TwinShield** ([2507.03278](https://arxiv.org/abs/2507.03278), Jul 2025) | bidirectional IT-masking + secret sharing + **U-Verify integrity** | honest-but-curious + integrity-checked | bit-exact | 4.0–6.1× speedup vs TEE-only; 87 % offload | none |
| **SecureInfer** ([2510.19979](https://arxiv.org/abs/2510.19979)) | TEE: Q/K/V/attn/FFN; GPU: other linear ops w/ XOR-OTP | black-box query adversary (no BSS analysis) | n/a | 2.06× vs unprotected GPU; 4.7× over TEE-only | none |
| **ObfuscaTune** ([2407.02960](https://arxiv.org/abs/2407.02960)) | TEE: 5 % (embeds, LN, dropout); GPU: 95 % w/ static orthogonal `R_a, R_b` | honest-but-curious cloud — **broken under TSQP** ([2602.11088](https://arxiv.org/abs/2602.11088)) & **ArrowMatch >98 %** ([USENIX'25](https://www.usenix.org/conference/usenixsecurity25/presentation/wang-pengli)) | lossless | 1.5–4.3× | none |
| **Slalom** ([1806.03287](https://arxiv.org/abs/1806.03287)) | TEE: nonlins; GPU: all linear w/ OTP + Freivalds | IT-private if pads unique; integrity-verifiable | quantization-bound | 4–11× speedup vs SGX-only (CNN only) | [ftramer/slalom](https://github.com/ftramer/slalom) (CNN, "do not use") |
| **PipeLLM** ([2411.03357](https://arxiv.org/abs/2411.03357)) | H100 CC w/ pipelined PCIe AES | std CC | bit-exact | <19.6 % (vs vanilla 52–88 %) on OPT-30B/66B/175B | none |
| **H100 CC baseline** ([2509.18886](https://arxiv.org/abs/2509.18886)) | whole GPU enclaved | std CC (NVIDIA Hopper attest) | bit-exact | 4–8 % batched LLM; 7–15 % high-QPS small-batch embedding | NVIDIA stack |
| **TDX / SEV-SNP CPU TEE** | whole CPU VM enclaved | std CC (Intel TDX / AMD SEV) | bit-exact | <10 % throughput, <20 % latency (small-model embed) | Intel/AMD vendor |
| **DarkneTZ** ([2004.05703](https://arxiv.org/abs/2004.05703)) | TrustZone: last CNN layers | MIA defense (CNN-only) | n/a transformers | 7.3× speedup, 71 % mem reduction | [mofanv/darknetz](https://github.com/mofanv/darknetz) |

Statistical / information-theoretic levers (apply post-encoder, often stacked):

| Technique | Mechanism | Privacy notion | Defense gain | Utility cost | Code |
|---|---|---|---|---|---|
| **INT8 quantization** | absmax / zeropoint | empirical | Vec2Text BLEU −60 % | negligible recall drop | trivial |
| **INT4 quantization** | aggressive | empirical | marginal; defeated by adaptive attack | small drop | trivial |
| **Matryoshka truncation** | use prefix `e[:k]` | empirical, untested vs Vec2Text | speculative (info-theoretic capacity ↓) | 0 if `k` adequate | [HF s-t](https://github.com/huggingface/sentence-transformers) |
| **JL + Gaussian** ([1204.2606](https://arxiv.org/abs/1204.2606)) | random projection + noise | `(ε,δ)`-DP (formal) | `(1±ε)` distance | tunable | trivial |
| **PCA reduction** | drop low-var dims | empirical only | modest, ≪ Eguard | substantial recall drop | scikit-learn |
| **CLUB-trained min-I** ([2006.12013](https://arxiv.org/abs/2006.12013)) | MI lower-bound loss in training | empirical bound | needs retraining; strong if tuned | train cost | [Linear95/CLUB](https://github.com/Linear95/CLUB) |
| **Fisher-Approx. Shannon** ([2504.10016](https://arxiv.org/abs/2504.10016)) | tractable `I(in; activation)` bound | info-theoretic argument | argument only, not a defense by itself | 0 | none |

Cheapest practical stack (no retraining, no formal proof): **INT8 absmax + Matryoshka prefix `k=256/384`** before SAP encryption. Empirically blunts Vec2Text by ~60 % with negligible retrieval cost; ~5 LOC in Rust.

### D7. Concrete buildable recipes for our RAG codebase

The `crates/approach4` workspace already has SAP/CAPRISE storage encryption + AES-GCM payloads. The embedding-side recipes that compose cleanly:

**Recipe A — H100-CC + INT8 + MRL (ship in a week).** Qwen3-Embedding-0.6B inside a Privatemode-style attested H100 CC enclave; output truncated to MRL prefix `k=256/512` and INT8 absmax-quantized before SAP encryption. Threat model: honest-but-curious cloud op. Components: NVIDIA NIM/vLLM with CC, [edgelesssys/contrast](https://github.com/edgelesssys), Qwen3-Embedding, existing `approach4` SAP. No inversion proof, but ~60 % empirical Vec2Text BLEU drop from quantization alone.

**Recipe B — DP-Forward at pooled output (formal DP, simplest).** Run embedder in TEE; before SAP-encrypting the result, apply the aMGM mechanism (~30 lines of Rust, port `matrix_gaussian_noise` + L2-clip + Gaussian add). ε ∈ [2, 8]. Formally `(ε,δ)`-SeqLDP. Adds <10 µs per query on the client/TEE side. Stacks with Recipe A.

**Recipe C — RemoteRAG-style two-stage retrieval (preserves exact recall).** Wrap Recipe B in over-fetch + PHE rerank. Noise on the sent query, encrypted-but-clean query for rerank against `k'=2k–5k` candidates. ~200 LOC + a Paillier/CKKS dep. Recovers 100 % recall@k. Composable with SAP for at-rest encryption of the index.

**Recipe D — GELO-style TEE+commodity-GPU split (no H100 CC needed).** TDX or SEV-SNP VM holds the encoder spine (LayerNorm/softmax/GeLU/FFN/embedding table/pooling) + per-batch fresh orthogonal+shield masks. Commodity GPU (any L40S/H100, **CC mode off**) runs Q/K/V/O GEMMs on masked hidden states. ~25–40 % total overhead. Major cost win vs H100 CC. Implementation cost: ~2–3 weeks; the hardest part is the shield-vector construction and BF16 numerical stability of `A⁻¹` (use orthogonal `A` so `A⁻¹ = Aᵀ`).

**Recipe E — Pure-MPC PUMA-on-SPU baseline (cryptographic, no hardware trust).** Fork [secretflow/spu](https://github.com/secretflow/spu) with the BERT-base path cribbed from OpenBumbleBee. BGE-base or E5-base under PUMA's 3PC. Mean pool is free; L2 normalize via SecFormer Goldschmidt `1/√x` (the cost the papers omit — port it). Expected: 20–40 s per query, ~10 GB comm, <1 % MTEB delta. Cost: requires 3 non-colluding parties.

The **L2-normalize step is the missing-from-the-papers cost** for every Recipe E/F variant — budget one extra inverse-sqrt beyond the published encoder latency. In MPC use SecFormer Goldschmidt; in FHE use Panda 2022/423.

### D8. Code maturity grade (skeptical reading)

| Tier | Systems |
|---|---|
| **Working, modern, transformer-applicable** | SPU (PUMA + Nimbus + Ditto), OpenBumbleBee, MPCFormer, SecFormer, NEXUS, DP-Forward (DP), SanText (text-DP baseline), Matryoshka via HF sentence-transformers |
| **Partial / research-grade** | BOLT, Iron, Slalom (CNN-era), SHAFT (claimed open per paper — verify) |
| **Paper-only (no public code 2026-05)** | GELO, ObfuscaTune, SecureInfer, PipeLLM, CipherFormer, THE-X, SIGMA, AERO, HE-LRM, Eguard, SPARSE, NVDP, OSNIP, SGT, RemoteRAG, TextObfuscator |
| **Commercial / closed beta** | Privatemode (Edgeless), Cyborg + cuVS (NVIDIA), Opaque, Fortanix |

### D9. Updated open research gaps

- **L2 normalization cost** is omitted from every published private-BERT benchmark — measure and publish.
- **Matryoshka × Vec2Text** has no published curve — a 1-week experiment.
- **ObfuscaTune-style static-key obfuscations are broken** ([arXiv 2602.11088](https://arxiv.org/abs/2602.11088)). Per-batch refresh is mandatory; GELO is the architecturally-correct pattern.
- **No paper benchmarks decoder-LLM-as-embedder under MPC/FHE specifically** (single forward pass, last-token pool, L2 normalize). Worst-case extrapolation is tens of minutes per embedding.
- **No published private-ColBERT system.** Multi-vector + late-interaction MaxSim under MPC is open.
- **Eguard / SPARSE / OSNIP / SGT / RemoteRAG / GELO all lack public code.** Anyone reproducing one of these with MTEB/BEIR + Vec2Text/ZSinvert benchmarks fills a measurable gap.

### D10. Rust ecosystem audit (2026-05-11)

Snapshot of crates.io coverage for the primitives Recipes A–E need. Dates are latest stable release. "Pure Rust" = no C/C++ link-time dep.

#### Transformer / encoder inference

| Crate | Latest | Pure Rust? | Maintained? | Notes |
|---|---|---|---|---|
| [`fastembed`](https://crates.io/crates/fastembed) | 5.13.4 (2026-04-27) | No (uses `ort`) | Very active (~250k DL/mo) | **Already pinned in workspace.** BGE/E5/Qwen3 + reranker zoo. The practical Recipe A path. |
| [`ort`](https://crates.io/crates/ort) | 2.0.0-rc.12 (2026-03-05) | No (ONNX Runtime C++) | Very active (~1.2M DL/mo) | Behind `fastembed`. RC line is prod-stable. |
| [`candle-core`](https://crates.io/crates/candle-core) | 0.10.2 (2026-04-01) | Mostly (CUDA shaders opt) | Active (HF, ~500k DL/mo) | BERT/E5/Qwen3 examples in tree. CUDA/Metal optional. |
| [`tract-onnx`](https://crates.io/crates/tract-onnx) | 0.22.0 (2025-08) | Yes | Active (Sonos) | Pure-Rust ONNX. Slower CPU than ORT, no FFI. |
| [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) | 0.1.146 (2026-04-30) | No (llama.cpp bindgen) | Very active | Best perf for GGUF Qwen3-Embedding on consumer HW. |
| [`burn`](https://crates.io/crates/burn) | 0.21.0 (2026-05-07) | Yes | Very active | BERT/E5 not first-class, weight port required. |
| [`rust-bert`](https://crates.io/crates/rust-bert) | 0.23.0 (2024-09) | No (libtorch) | **Frozen since Sep 2024** | Avoid for new code. |

#### Sampling and special functions (Recipe B)

| Crate | State | Notes |
|---|---|---|
| [`rand_distr`](https://crates.io/crates/rand_distr) 0.6.0 | Active | Normal, Gamma, Laplace, Cauchy. ~7.6M DL/mo. |
| [`rand_chacha`](https://crates.io/crates/rand_chacha) 0.9 | **In workspace** | CSPRNG; seed from `OsRng` for DP noise. |
| [`statrs`](https://crates.io/crates/statrs) 0.18.0 | Active | `erf`, gamma, beta — aMGM bisection. |
| [`special`](https://crates.io/crates/special) 0.13.1 | Active | Smaller alternative for `f64` erf/gamma. |

**Constant-time DP noise:** none of the above are constant-time. If your threat model requires it, write a CDT/Karney sampler (~300 LoC).

#### Paillier and FHE (Recipe C)

| Crate | Latest | State | Notes |
|---|---|---|---|
| [`fast-paillier`](https://crates.io/crates/fast-paillier) | 0.3.2 (2025-11) | Active (Dfns) | **Use this for Paillier.** CRT-based + safe-prime keygen. Pulls in LGPL `rug` (GMP). Not formally audited. |
| [`kzen-paillier`](https://crates.io/crates/kzen-paillier) | 0.4.3 (2022-12) | **Stale** (3.5 yr) | Authors warn no side-channel hardening. |
| [`libpaillier`](https://crates.io/crates/libpaillier) | 0.7.0-rc0 (2024-11) | Slow | Used in threshold-ECDSA. "Not audited." |
| [`tfhe`](https://crates.io/crates/tfhe) (Zama) | 1.6.1 (2026-04) | Very active | TFHE only — **no CKKS.** Wrong shape for inner products. |
| [`sealy`](https://crates.io/crates/sealy) | 0.2.0 (2024-10) | Hobbyist | BFV + CKKS via SEAL FFI. "Early stages." |
| [`openfhe`](https://crates.io/crates/openfhe) (fairmath/openfhe-rs) | 0.2.8 (2024-06) | Active GH (2026-05), stale crate | Full OpenFHE incl. CKKS + bootstrapping. Releases out-of-band. |
| [`sunscreen`](https://crates.io/crates/sunscreen) | 0.8.1 (2023-09) | **Abandoned** | BFV only, AGPL-3.0. |

**Verdict:** Production CKKS in Rust does not exist. FFI to OpenFHE 1.5 via `openfhe-rs` is the only realistic path; pin C++ + Rust versions yourself.

#### MPC (Recipe E)

| Project | State | Notes |
|---|---|---|
| [`swanky`](https://github.com/GaloisInc/swanky) | Active GH, git-dep only | OT/GC/ZK/VOLE/PSI. **README warns of proven vulnerability in arithmetic garbled-circuit projection gates.** |
| MP-SPDZ Rust bindings | None | C++ only. |
| ABY3 / Cheetah Rust | None | Original Python/C++ research code. |

**Confirmed: Recipe E stays Python (SPU).** No production Rust MPC for 3PC ABY3.

#### TEE attestation (Recipes A, D)

| Crate | Latest | Notes |
|---|---|---|
| [`tdx-guest`](https://crates.io/crates/tdx-guest) (Intel) | 0.3.1 (2026-03) | TDCALL/TDVMCALL wrappers. ~21k DL/mo. |
| [`tdx-quote`](https://crates.io/crates/tdx-quote) (entropyxyz) | 0.0.5 (2026-01) | **Early dev, AGPL-3.0, unaudited.** `no_std` v4/v5 quote parse. |
| [`tdx_attest`](https://crates.io/crates/tdx_attest) | active | ioctl → TD report + DCAP quote. |
| [`sev`](https://crates.io/crates/sev) (virtee) | 7.1.0 (2025-09) | Very active. Canonical AMD SEV-SNP + KVM. ~84k DL/mo. Apache-2.0. |
| [`snphost`](https://crates.io/crates/snphost) / `snpguest` (virtee) | 0.7.0 (2025-11) | Companion host/guest CLIs. |
| [`aws-nitro-enclaves-nsm-api`](https://crates.io/crates/aws-nitro-enclaves-nsm-api) | 0.5.1 (2026-04) | First-party AWS SDK, ~260k DL/mo. CBOR/COSE. |
| [`apache/teaclave-sgx-sdk`](https://github.com/apache/teaclave-sgx-sdk) | v2.0 (2025-09) | Slow but alive. v1.1.1 on crates.io is the legacy artifact. |
| `nvtrust-rs` (community) | Research-grade fork | **Only Rust path for H100 CC attestation.** NVIDIA's own NVAT is C++/CLI. |

**H100 CC attestation is the soft spot.** Realistic plan: shell out to NVAT host-side, surface verified report to Rust verifier through a protobuf interface.

#### Linear algebra, ANN, tokenization, AES (well-served)

- `nalgebra` 0.34.2 + `nalgebra-lapack` for QR-sampled orthogonal `A` (Recipe D mask). Sample `A ~ N(0,1)^{d×d}` → `QR::new(M).q()` → Haar-uniform `O(n)`. Trivial.
- `ndarray` 0.17.2 for tensor math.
- `hnsw_rs` 0.3.4 for ANN. `instant-distance` works but frozen 3 yrs.
- `tokenizers` 0.23.1 (HF first-party) — loads BGE/E5/Qwen3 `tokenizer.json` cleanly.
- AES-GCM / ChaCha20-Poly1305 / HMAC / HKDF — RustCrypto suite, all NCC-audited and prod-grade. Already in workspace.

#### GPU offload for Recipe D

| Option | Pros | Cons |
|---|---|---|
| [`cudarc`](https://crates.io/crates/cudarc) 0.19.4 | Fine control over per-GEMM launch; ~650k DL/mo | Write your own kernels (or use cuBLAS via cudarc). |
| `candle` w/ CUDA | cuBLAS-backed GEMM, tested | Couples to candle tensor types. |
| Raw `bindgen` to cuBLAS/CUTLASS | Maximum control | Most boilerplate. |

For Recipe D specifically (per-batch fresh mask, choose which GEMMs go off-TEE) → **`cudarc` is the closest fit**. ~1–2k LoC for the mask-then-offload-then-unmask glue.

#### Per-recipe Rust verdict

- **Recipe A (H100/TDX + INT8 + MRL).** Buildable today. Stack: `fastembed`+`ort` (or `candle`), `tokenizers`, `ndarray` for INT8/MRL post-process, existing AES-GCM/SAP. Soft spot = H100 CC attestation (no first-party Rust).
- **Recipe B (DP-Forward aMGM).** Trivial. `rand_chacha` + `rand_distr::Normal` + `statrs::erf`. ~200 LoC including reference tests.
- **Recipe C (RemoteRAG + PHE rerank).** Doable but ugly. `fast-paillier` works (with LGPL caveat); CKKS realistically means OpenFHE FFI. Stay Paillier-only if possible.
- **Recipe D (TEE+GPU split, per-batch mask).** Rust math easy (`nalgebra` QR). GPU side via `cudarc`. ~1–2k LoC for the offload engine. GELO mask abstraction = you write it.
- **Recipe E (PUMA-on-SPU).** **Not Rust.** Out-of-process Python service.

#### Cross-cutting flags

- **No production-grade Rust path** for: CKKS (FFI OpenFHE), 3PC MPC (use Python SPU), first-party NVIDIA CC attestation (shell out to NVAT).
- **Stale crates** signaling frozen or abandoned: `kzen-paillier`, `instant-distance`, `sunscreen`, legacy `sgx_tstd`, `rust-bert`.
- **Early-dev, 1–2 maintainers** (vendor or fork before depending on for load-bearing code): `tdx-quote 0.0.5`, `sealy 0.2.0`, `nvtrust-rs`.
- **Licensing footguns:** `fast-paillier` pulls LGPL `rug`; `tdx-quote` is AGPL-3.0; `sunscreen` is AGPL-3.0.
