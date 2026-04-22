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
