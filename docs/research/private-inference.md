---
type: research
status: current
created: 2026-05-11
updated: 2026-05-11
tags: [inference]
---

# Private Transformer Inference (Embedding, Reranking, Generation)

> Research date: 2026-04-21. Covers private/confidential inference for **any** transformer in a RAG pipeline — embedding generation, cross-encoder reranking, and LLM response generation — where the inference server cannot see the plaintext input, the model weights, or both.
>
> **Scope:** MPC, FHE, TEE, obfuscation, TEE+GPU split-inference, and end-to-end RAG systems chaining private retrieval with private inference.
>
> **Cross-references:**
> - `private-embedding-research.md` — embedding-specific applicability (Vec2Text threat, client-side vs server-side embedding)
> - `private-reranking-research.md` — reranker-specific applicability (cross-encoder vs bi-encoder key management)
> - `private-information-retrieval.md` — retrieval-stage (PIR, ORAM, DP-on-query) systems
> - `fhe-encrypted-vector-db.md` — encrypted vector storage

---

## Context: The Retrieval-to-Generation Handoff

After private retrieval, the pipeline has `k` retrieved documents. The central question for private generation is: **what format does the context arrive in, and what does the LLM see?**

| Retrieval scheme | Context format at generation | LLM sees |
|---|---|---|
| DP perturbation (RemoteRAG) | Plaintext, decrypted client-side | Full plaintext prompt + context — cloud LLM sees everything |
| Dist-preserving enc (CAPRISE) | Decrypted client-side | Full plaintext prompt + context — cloud LLM sees everything |
| Secret sharing (p²RAG) | Reconstructed client-side from shares | Full plaintext prompt + context — cloud LLM sees everything |
| FHE (RAGtime-PIANO) | Ciphertext, never decrypted | Encrypted context — LLM must support homomorphic inference |
| TEE (Opal, PrivANN) | Decrypted inside enclave | Plaintext inside trusted hardware — cloud operator sees nothing |
| Obfuscation (GELO, OSNIP) | Obfuscated text, sent to cloud LLM | Obfuscated prompt + context — cloud LLM generates over garbled text |

Most retrieval systems (DP, DPE, SS) silently break privacy at generation: the decrypted context is sent in the clear to a cloud LLM. End-to-end privacy requires either (1) private LLM inference, (2) TEE-hosted LLM, or (3) obfuscation before sending.

---

## A. End-to-End Private RAG Systems

### 1. Opal — Private Memory for Personal AI
*(local corpus; Kaviani et al. 2026)*

**Target:**
- **Model:** Cloud LLM (GPT-4 class) used as the generator; embedding model runs client-side or in TEE
- **Retrieval:** TEE-hosted ORAM-backed private memory store; vector similarity search inside enclave
- **Embedding:** Client-side or enclave-side; not exposed to cloud

**Approach:**
Full-pipeline private personal AI: the user's memory (documents, notes, past interactions) is stored in an encrypted, ORAM-protected vault. At query time, the TEE retrieves relevant memories obliviously, assembles the context, and manages the LLM interaction. The cloud LLM only sees the assembled prompt (which may still include context — see privacy model). The focus is on protecting the **corpus** (personal memory) rather than the query, via hardware isolation.

**Privacy/security model:**
TEE (SGX/TDX) isolates all memory access and embedding computation. The ORAM layer hides which memory entries were accessed. The cloud LLM provider sees the assembled prompt but not the raw memory vault. Threat model: curious cloud infra operator; not a compromised LLM provider.

**Performance:**
- Specific latency numbers not publicly available (paper not on arXiv)
- ORAM-backed retrieval adds overhead vs plaintext memory access (see PrivANN: 2.4× over FHE for ORAM-based ANN)
- Full PDF in local library

**Implications:**
- TEE gives strong corpus privacy without expensive MPC/FHE at inference time
- The LLM provider still sees the assembled context; true end-to-end privacy requires a private-inference-capable LLM or an on-device model
- Practical for personal AI (notes, medical records, emails) where the corpus is sensitive but the final generation can go to a trusted API

**Used in:** Research prototype; no known commercial deployment

---

### 2. RAGtime-PIANO — FHE End-to-End Private RAG
*(local corpus; Januszewicz et al.)*

**Target:**
- **Model:** Unspecified LLM operating under FHE (CKKS + BFV)
- **Retrieval:** FHE-encrypted hybrid dense + sparse retrieval (see PIR doc §5)
- **Embedding:** Client-side; query encrypted before upload

**Approach:**
The only system in this survey attempting FHE privacy across the *entire* RAG pipeline — retrieval and generation. The encrypted query goes through a two-stage FHE cluster search, and the retrieved encrypted context is then fed into FHE-based generation. The offline pre-processing phase (encrypted cluster centroids, index structures) amortizes some per-query FHE overhead.

**Privacy/security model:**
FHE throughout: the server processes only ciphertexts at every stage. No hardware trust required. Both the query and the corpus remain encrypted on the cloud at all times.

**Performance:**
- Concrete latency not publicly available (paper not on arXiv; full PDF in local library)
- FHE inference for an LLM on top of FHE retrieval compounds two expensive operations
- Expected order-of-magnitude: minutes per query at research scale

**Implications:**
- Strongest privacy model of any system here — no hardware trust, no polynomial approximation of the *retrieval* step
- FHE inference for large LLMs is currently impractical (see THE-X: BERT-only; PUMA/BumbleBee: minutes per LLaMA-7B token)
- RAGtime-PIANO is architecturally sound but likely research-stage only

**Used in:** Research only

---

### 3. RemoteRAG — DP Retrieval + Cloud LLM Generation
https://arxiv.org/abs/2412.12775
**Cheng, Zhang, Wang, Yuan, Yao | ACL Findings 2025**

**Target:**
- **Model:** Any cloud LLM (not made private); embedding model runs client-side
- **Retrieval:** (n,ε)-DistanceDP + PHE (see PIR doc §4)
- **Embedding:** Client-side; perturbed before sending to cloud retriever

**Approach:**
Privacy covers only the retrieval step. After private retrieval, the top-k documents are decrypted client-side and sent **in plaintext** as context to the cloud LLM for generation. The generation step is explicitly not protected — the system's claim is that the cloud LLM already has the corpus, so seeing retrieved docs adds marginal new leakage beyond what the cloud embedding model would infer.

**Privacy/security model:**
(n,ε)-DistanceDP protects query embedding identity during retrieval. The LLM provider sees: the (perturbed) query embedding, the k retrieved documents, and the final query text at generation time. Corpus is already on the cloud — no corpus privacy. Generation is plaintext.

**Performance:**
- Retrieval: 0.67s, 46.66 KB comm for 100K documents
- Generation: standard cloud LLM latency (not measured)
- Vec2Text SacreBLEU on retrieved query embedding: ~50 (no noise) → ~10 (ε=0.2)

**Implications:**
- DP retrieval is a partial fix: the LLM still sees the query at generation time (either explicitly or via the context)
- For query privacy, the DP protection on the embedding is undermined unless the LLM API call is also anonymized
- Practical as a defense-in-depth measure; not a strong end-to-end privacy claim

**Used in:** Research

---

### 4. prRAG + CAPRISE — DPE Retrieval + Local/Cloud Generation
https://arxiv.org/abs/2601.12331
**Ye et al. | 2026**

**Target:**
- **Model:** Any LLM (not made private); embedding model client-side
- **Retrieval:** Distance-preserving encryption (CAPRISE) with DP query perturbation
- **Embedding:** Client-side; encrypted with CAPRISE before upload

**Approach:**
Three-phase pipeline: (1) corpus encrypted and uploaded, (2) encrypted query retrieves top-k' encrypted candidates, (3) client decrypts candidates, re-ranks to top-k, assembles context, generates response with LLM. The paper explicitly states generation uses "retrieved content combined with original query for LLM-based response generation" — generation is outside the private boundary.

**Privacy/security model:**
Corpus content hidden from cloud at rest (AES-encrypted). Query intent partially hidden by DP perturbation. At generation time, plaintext context is assembled client-side and sent to LLM — LLM provider sees the final prompt.

**Performance:**
- Encryption throughput: 2,339 vectors/s at 768-dim
- Encryption overhead: 15 ms per 128 queries
- Retrieval expansion at r=0.033: k=5 → k'=258 (52× more candidates returned)
- Generation latency: unspecified

**Implications:**
- Privacy of the corpus at rest is strong (AES + CAPRISE encryption)
- But the cloud LLM sees the full assembled plaintext prompt at generation time
- Practical if the threat is a storage-layer adversary (e.g., compromised vector DB), not an LLM provider adversary

**Used in:** Research

---

## B. MPC-Based Private LLM Inference

### 5. PermLLM — Permutation-Accelerated MPC Inference
https://arxiv.org/abs/2405.18744
**Zheng et al. | 2024**

**Target:**
- **Model:** ChatGLM-6B (6B parameters)
- **Retrieval:** Not addressed; assumes prompt + context is prepared client-side
- **Embedding:** Not addressed

**Approach:**
Introduces **secure random permutation** as a primitive to accelerate MPC-based transformer inference. The key insight: most of the cost of MPC transformer inference comes from evaluating non-linear functions (GeLU, Softmax, LayerNorm). PermLLM replaces expensive garbled circuit evaluations of these with a protocol based on random permutations of hidden-state columns, dramatically reducing communication. Uses additive secret sharing (A-SS) + Beaver triples for linear layers, BFV for the permutation protocol.

**How it connects to RAG:**
Client-side: assemble plaintext prompt (query + retrieved context). Secret-share the assembled prompt and send shares to two/three MPC servers. MPC servers jointly run the transformer on shares. Response is secret-shared back to the client, who reconstructs plaintext.

**Privacy/security model:**
Semi-honest 3-party (model provider P₀, user P₁, helper P₂). Neither the model provider nor helper learns the prompt or response. The user learns nothing about model weights. Security under honest-but-curious assumption.

**Performance:**
- Latency: **~3 seconds/token** (10ms RTT, 1 Gbps LAN); ~7s/token at 20ms/100 Mbps WAN
- Communication: **~20 MB/token** total; single transformer layer: **0.49 MB** (vs. 3,073 MB for MPCFormer — **6,000× reduction**)
- Model: ChatGLM-6B (6B params)
- Quality: **identical to plaintext** — no accuracy loss

**Implications:**
- 3s/token is the current MPC frontier for 6B-class models; first MPC system fast enough to consider for interactive use
- 20 MB/token over WAN still means ~150 MB for a 7-token response — non-trivial bandwidth
- The 6,000× comm reduction over MPCFormer makes WAN deployment feasible for the first time
- Does not address how retrieved context (potentially long) affects latency — longer prompts = more secret-sharing overhead

**Used in:** Research; PermLLM group at HUST

---

### 6. PUMA — Private LLM Serving with 2-out-of-3 Secret Sharing
https://arxiv.org/abs/2307.12533
**Dong, Chen, Lin et al. | 2023**

**Target:**
- **Model:** LLaMA-7B, GPT-2 (Base/Medium/Large), BERT-Base
- **Retrieval:** Not addressed
- **Embedding:** Not addressed

**Approach:**
2-out-of-3 replicated secret sharing (RSS) framework for LLM inference, building on Cheetah's matrix multiplication protocols. Key contributions: efficient MPC protocols for GeLU, SoftMax, and LayerNorm without model retraining; first system to run LLaMA-7B under MPC.

**How it connects to RAG:**
Same handoff pattern as PermLLM: client assembles plaintext prompt → secret-shares → MPC servers compute → reconstruct response. For RAG, the full context (query + retrieved docs) is secret-shared. Longer retrieved context means more input tokens to MPC, increasing latency and communication linearly.

**Privacy/security model:**
Semi-honest 3-party RSS (3 servers, 1 corrupted). Model provider and user mutually learn nothing about each other's inputs. Model weights are also secret-shared — model provider does not need to reveal weights to the cloud.

**Performance:**
- LLaMA-7B (8-token input, 1-token output): **~200 seconds/token**; 4-token input: **~122 seconds**
- GPT-2 Base (32 tokens): **15.5 seconds/token**
- BERT-Base (128 tokens): **33.9 seconds** per inference
- Communication: LLaMA-7B: **1.794 GB/token**; GPT-2-Large: **11.95 GB/token**; BERT-Base (128 tokens): **10.77 GB**
- Hardware: 3 × 128-thread / 1 TB RAM servers (Alibaba Cloud)
- Quality: GLUE accuracy drop ≤0.011; perplexity difference ≤0.02

**Implications:**
- 200s/token for LLaMA-7B: research-only, not interactive
- 1.794 GB/token comm for LLaMA-7B is impractical for WAN deployment
- BERT-Base at 33.9s for 128 tokens is borderline viable for batch applications (medical record analysis, legal RAG) where latency is less critical
- No retraining required — any existing model can be used

**Used in:** Research (PUMA team, ByteDance)

---

### 7. SIGMA — FSS-Based LLM Inference
*(PETS 2024)*

**Target:**
- **Model:** LLaMA-2-13B, GPT-2
- **Retrieval:** Not addressed
- **Embedding:** Not addressed

**Approach:**
**Function Secret Sharing (FSS)** replaces garbled circuits for non-linear functions (Softmax, GeLU, SiLU). FSS allows a dealer to generate correlated randomness offline, reducing the online phase to simple additions. This is more efficient than garbled circuits for comparison/ReLU-class operations.

**Privacy/security model:**
2-party computation, semi-honest adversaries, with an offline dealer (non-colluding).

**Performance:**
- LLaMA-2-13B: **44 seconds/token**
- GPT-2: **1.6 seconds/token**
- Throughput: **11–19× improvement** over prior MPC approaches (BOLT, Cheetah)
- Communication: reduced vs GC-based; specific MB/token not published

**Implications:**
- 44s/token for a 13B model is 4.5× faster than PUMA's LLaMA-7B (200s) — significant progress
- GPT-2 at 1.6s/token is close to usable for non-interactive RAG
- FSS preprocessing can be pipelined with retrieval — while retrieval happens, generate the correlated randomness

**Used in:** Research (ETH Zürich / University of Edinburgh)

---

### 8. SHAFT — Secure, Handy, Accurate, and Fast Transformer Inference
https://doi.org/10.14722/ndss.2025.242287
**Kei, Chow | NDSS 2025**

**Target:**
- **Model:** BERT-Base, RoBERTa (NLU tasks)
- **Retrieval:** Not addressed
- **Embedding:** Not addressed

**Approach:**
MPC-based transformer inference with optimized protocols for attention, GeLU, and LayerNorm. Targets NLU (classification) rather than generative inference. Used as the baseline in SPRINT (2026).

**Privacy/security model:**
MPC semi-honest setting; model weights and user inputs mutually protected.

**Performance:**
- Used as SPRINT baseline; SPRINT achieves 1.6× speedup over SHAFT
- 9 citations since NDSS 2025 — widely referenced as current MPC-NLU SOTA
- Specific latency numbers: paper behind NDSS paywall; not publicly available

**Implications:**
- NLU focus (BERT/RoBERTa) means it's relevant for RAG *classification/ranking* tasks but not generation
- SPRINT extends SHAFT to add DP for training data privacy in addition to input privacy

**Used in:** Research; serves as baseline for subsequent work

---

### 9. SPRINT — MPC Inference on DP-Finetuned Models
https://petsymposium.org/popets/2026/popets-2026-0008.pdf
**Capano, Böhler, Weggenmann | PETS 2026**

**Target:**
- **Model:** RoBERTa (GLUE benchmark)
- **Retrieval:** Not addressed
- **Embedding:** Not addressed

**Approach:**
Combines DP fine-tuning (to protect training data) with MPC inference (to protect input at inference time). Two levels of privacy: training data is ε-DP protected during fine-tuning; user input is protected by MPC at inference. Uses parameter-efficient fine-tuning (LoRA) to reduce DP noise impact, and cleartext public parameters to reduce MPC overhead.

**Privacy/security model:**
Dual: (1) training data DP-protected (formal ε guarantee); (2) inference input MPC-protected (semi-honest). Strongest formal guarantees of any MPC system here — both corpus and query have formal privacy bounds.

**Performance:**
- **1.6× faster** MPC inference than SHAFT
- **1.6× less communication** than SHAFT
- Accuracy: <1 percentage point gap vs cleartext for GLUE benchmark

**Implications:**
- Combining DP training + MPC inference addresses both "what did the model memorize?" and "what does the server learn about this query?"
- For RAG: if the RAG corpus is used to fine-tune the model (e.g., domain-adapted embedding model or generator), DP fine-tuning protects the training corpus from model extraction
- 1.6× faster than SHAFT is still in the same order of magnitude — not a breakthrough for latency

**Used in:** SAP Research

---

### 10. MERGE — Fast Private Text *Generation*
https://doi.org/10.1609/aaai.v38i18.29964
**Liang, Wang, Zhang et al. | AAAI 2024**

**Target:**
- **Model:** GPT-2 (NLG tasks: translation, code completion, summarization)
- **Retrieval:** Not addressed
- **Embedding:** Not addressed

**Approach:**
The only MPC system specifically designed for **autoregressive text generation** (NLG) rather than NLU. Key insight: standard MPC inference treats each token generation as an independent call, wasting the recomputed KV-cache under MPC. MERGE (1) reuses the output hidden state as the word embedding to skip redundant embedding lookup, (2) reorganizes linear operations in the transformer to batch secret-sharing communication across the autoregressive loop.

**How it connects to RAG:**
RAG generation is autoregressive: the model generates one token at a time, conditioning on both the retrieved context and prior generated tokens. MERGE directly addresses this pattern.

**Privacy/security model:**
Two-party MPC, semi-honest. Model weights and user prompt/context protected from the inference server.

**Performance:**
- **GPT-2-base (124M), seq len 128:** 171 s total → **~1.34 s/token** (CrypTen baseline: 1328 s / ~10.4 s/token → 7.75× speedup)
- **T5 (138M), seq len 128:** 144 s total → **~1.12 s/token** (CrypTen baseline: 1569 s → 10.89× speedup)
- At seq len 512: **26.5× speedup** over unoptimized encrypted model (amortization improves with length)
- **Communication — GPT-2:** 121 GB per 128-token inference (~0.95 GB/token); baseline: 322 GB (62% reduction)
- **Communication — T5:** 98 GB per 128-token inference; baseline: 380 GB (74% reduction)
- vs state-of-the-art approximated MPC (MPCFormer, THE-X): up to **10× speedup**

**Implications:**
- First MPC paper to address the autoregressive loop cost explicitly — the others measure single-token latency, which understates total generation cost
- ~1.3 s/token on a 124M-param model with 121 GB comm/inference: faster than other MPC approaches, but still impractical for interactive RAG even at GPT-2 scale
- 26.5× speedup at seq len 512 is vs the unoptimized CrypTen baseline — the fairer comparison (vs MPCFormer) is 10×
- Does not test LLaMA-class models; GPT-2-base is ~50× smaller than LLaMA-7B by parameter count

**Used in:** Research (XJTU)

---

## C. FHE-Based Private LLM Inference

### 11. THE-X — FHE for Transformer Inference
https://arxiv.org/abs/2206.00216
**Chen, Bao, Huang et al. | ACL Findings 2022 | 73 citations**

**Target:**
- **Model:** BERT-Base, BERT-Large; downstream NLU tasks (GLUE)
- **Retrieval:** Not addressed
- **Embedding:** Query encrypted under CKKS before sending to server

**Approach:**
FHE inference for pre-trained transformers without model retraining. All non-polynomial operations (GeLU, Softmax, LayerNorm) are approximated with low-degree polynomials compatible with CKKS. THE-X provides a workflow to handle the full transformer block under HE, not just individual layers.

**Privacy/security model:**
Pure FHE (CKKS). The server processes only ciphertexts and never sees the plaintext input. No hardware trust, no non-colluding server assumption. Single-server, non-interactive after initial key setup.

**Performance:**
- Model: BERT-Base and BERT-Large
- Accuracy: **negligible drop** vs plaintext across GLUE tasks (all within noise)
- Concrete latency: not published in the paper (theoretical framework); later work (BumbleBee, CipherFormer) provides numbers for FHE+GC hybrids
- Communication: single encrypted ciphertext upload/download — low compared to MPC

**Implications:**
- THE-X established the feasibility of FHE BERT inference without retraining; precursor to BumbleBee and Iron
- The polynomial approximation of Softmax/GeLU introduces approximation error that limits scalability to larger models
- No numbers for LLaMA-class models; FHE BERT is already slow — LLaMA would be orders of magnitude worse

**Used in:** Research (Microsoft Research Asia); 73 citations, widely extended

---

### 12. BumbleBee — HE + Garbled Circuits for LLaMA-7B
*(NDSS 2025)*

**Target:**
- **Model:** LLaMA-7B
- **Retrieval:** Not addressed
- **Embedding:** Not addressed

**Approach:**
Hybrid HE + garbled circuits (GC) approach targeting LLaMA-scale models. Uses homomorphic encryption for linear operations and garbled circuits for non-linear activations (GeLU, SiLU). Key optimizations: 80–95% reduction in communication for activation functions vs prior methods.

**Privacy/security model:**
Two-party secure inference (input holder vs model holder). Model weights protected from the querying user; user inputs protected from the model provider.

**Performance:**
- LLaMA-7B: **~8 minutes/token** (CPU-based)
- Communication: **80–95% less** for activations, **80–90% less** for matrix mult vs prior methods
- vs Iron (prior HE+GC SOTA): **>10× faster**; vs BOLT: **3× faster** with 1/10 communication
- Quality: not specified (LLaMA-7B is used as-is, no retraining)

**Implications:**
- 8 min/token is research-only; 10× faster than Iron shows the field is progressing rapidly but still far from interactive
- CPU-only: GPU-accelerated HE would reduce this significantly (CKKS on GPU showed 10–100× speedups in other contexts)
- BumbleBee is currently the largest model evaluated under FHE+GC — establishing that it's *possible*, not that it's *practical*

**Used in:** Research (NDSS 2025)

---

### 13a. NEXUS — Non-Interactive FHE Transformer Inference (NDSS 2025)
https://doi.org/10.14722/ndss.2025.230868
**Zhang, Yang, He, Chen, Lu, Wang, Hou, Liu, Ren, Yang | NDSS 2025**

**Target:**
- **Model:** BERT-Base (12 layers)
- **Retrieval:** Not addressed
- **Embedding:** Query encrypted under RNS-CKKS

**Approach:**
**First non-interactive** FHE protocol for secure transformer inference. Client sends one encrypted message; server computes; client decrypts. No back-and-forth. Novel primitives: SIMD ciphertext compression/decompression, SIMD slot folding, secure Argmax (improved from `O(m)` Phoenix to lower complexity for vocabulary-sized `m`).

**Privacy/security model:**
Single-server RNS-CKKS FHE. No hardware trust, no non-colluding assumption. Server sees only ciphertexts.

**Performance (BERT-Base):**
- **37.3 s CPU** / **0.88 s GPU** (42.3× GPU speedup)
- Bandwidth: **164 MB** (single round)
- vs BOLT (Oakland '24): **3.6× faster**, **372.5× less bandwidth** (59.6 GB → 164 MB)
- vs BumbleBee (NDSS '25): **53.6× less bandwidth** (~8.8 GB → 164 MB)
- Quality: comparable to plaintext

**Tradeoffs:**
- Non-interactive = no amortization across tokens; each inference independent
- 37.3 s CPU still far from real-time for BERT-Base; GPU version (0.88 s) starts to be interactive
- No LLaMA-class numbers; scales poorly past BERT-Base without hardware acceleration

**Used in:** Baseline for Euston (below) and other non-interactive FHE work.

---

### 13b. Euston — Non-Interactive FHE with SVD-Preprocessed HMM (IEEE S&P 2026)
**Gao, Fu, Liu, Liu, Luo, Wang | IEEE S&P 2026** (already in Edgequake) | [Code](https://github.com/FLL-Lab/Euston)

**Not TEE split-inference.** Pure FHE, single-server, non-interactive — extends NEXUS.

**Target:**
- **Model:** BERT-Base, GPT2-1.5B, LLaMA-3-8B (tested on both CPU and GPU)
- **Retrieval:** Not addressed
- **Embedding:** Query encrypted under RNS-CKKS before single-round upload

**Approach:**
Two orthogonal innovations over NEXUS:
1. **SVD-based preprocessing for Homomorphic Matrix Multiplication (HMM).** Mask matrix is SVD-decomposed; only the diagonal singular-value matrix is encrypted, shrinking transmitted ciphertext size drastically. Four HMM primitives (plaintext / ciphertext / batched-ciphertext) target column-packed, row-packed, diagonal-packed, and unpacked matrix formats — eliminating redundant homomorphic rotations and slashing user-side preprocessing cost.
2. **Rotation-free Homomorphic Non-linear Evaluations (HNE).** Column- and diagonal-packed ciphertext formats chosen so that GELU, Softmax, and LayerNorm can be evaluated without inter-slot rotations. Depth-regulation strategies reduce multiplicative depth and speed up bootstrapping.

**Privacy/security model:**
Single-server RNS-CKKS FHE. Identical security model to NEXUS; changes are purely efficiency optimizations, same standard IND-CPA assumption under LWE/RLWE.

**Performance (Figure 6 of the paper — amortized over a batch of 32 × 128-token inputs):**

| Model | NEXUS-CPU | Euston-CPU | NEXUS-GPU | Euston-GPU | CPU speedup | GPU speedup |
|---|---|---|---|---|---|---|
| BERT-Base | 2,300 s | **648 s** | 98 s | **46 s** | 3.5× | 2.1× |
| GPT-2-1.5B | 7,200 s | **1,300 s** | 340 s | **136 s** | 5.5× | 2.5× |
| LLaMA-3-8B | 31,800 s (~8.8 h) | **~3,600 s (~60 min)** | 1,800 s (~30 min) | **~484 s (~8 min)** | 8.8× | 3.7× |

Per-inference (single 128-token forward pass, Euston-only, amortized):

| Model | CPU per inference | GPU per inference |
|---|---|---|
| BERT-Base | **~20.3 s** | **~1.44 s** |
| GPT-2-1.5B | **~40.6 s** | **~4.25 s** |
| LLaMA-3-8B | **~113 s (~1.9 min)** | **~15.1 s** |

**Per-operation speedups vs NEXUS:** HMM 90×; GELU 17.7×; LayerNorm 27.7×; Softmax 165.7×; bootstrapping 1.25–1.3×.

**User-side offline (preprocessing) improvements vs NEXUS:** **3100× shorter runtime**, **4.4× lower communication**, **2.2× less storage** — brings user-side setup from hour-scale down to seconds.

**Communication:** ~2.8× less online communication than NEXUS (NEXUS was 164 MB for BERT-Base → Euston ~58 MB).

**Important benchmark caveat.** Fig 6 totals are **prefill-only** (one forward pass on 128-token input), not autoregressive generation. For a RAG-style decoder workload with N generated response tokens, multiply the per-inference number by ~N — a LLaMA-3-8B generation of 200 response tokens on top of a 128-token prompt would be ~30+ minutes on GPU, ~6+ hours on CPU. **Not interactive at LLaMA scale.**

**Implications:**
- **Euston BERT-Base GPU at ~1.4 s per inference is the current fastest FHE transformer inference**, ~2× faster than NEXUS (0.88 s per single query → Euston amortizes to 1.44 s, but this is over a 32-query batch). For single-query latency, NEXUS's reported 0.88 s GPU on BERT-Base and Euston's matching number are in the same ballpark — the big win is throughput under batching, not single-query latency.
- **LLaMA-3-8B under FHE is now tractable for batch / offline workloads** (minutes on GPU per query), but still 2–3 orders of magnitude from interactive.
- 3100× user-side preprocessing reduction is the deployment-changing result — NEXUS's hour-scale client setup made it impractical; Euston's seconds-scale setup enables real deployment.

**Tradeoffs:**
- CPU LLaMA-3-8B at ~2 min per prefill query — still batch-only
- Non-interactive FHE caps at polynomial-approximated GELU / Softmax / LayerNorm — small accuracy drift vs plaintext
- Fig 6 numbers are amortized over batch-32; single-query latency is higher (NEXUS single-query CPU was 37.3 s BERT-Base — the Fig 6 per-query number of 20 s for Euston suggests Euston's per-single-query is similar or slightly better, not dramatically so)
- Open-source C++/CUDA impl on GitHub (FLL-Lab/Euston); RNS-CKKS engineering cost applies

**Used in:** Research (IEEE S&P 2026); positions Euston as the current non-interactive FHE SOTA.

---

### 13c. CipherFormer — HE + GC, Low Round Complexity
https://arxiv.org/abs/2403.16860
**Wang, Kuang | 2024**

**Target:**
- **Model:** Small BERT variants (1–2 encoder layers, d=32–64)
- **Retrieval:** Not addressed
- **Embedding:** Encrypted under HE

**Approach:**
Reduces round complexity of HE+GC inference. Key contribution: homomorphic matrix multiplication protocol and modified attention mechanism designed for GC efficiency. Uses mixed-bitwidth to reduce inference latency.

**Privacy/security model:**
Two-party HE+GC; few communication rounds between client and server.

**Performance:**
- Online latency: **5.15 seconds** per inference (Yelp, 1–2 layer BERT-small)
- Offline setup: 14.07 seconds
- Communication: 32.0 MB online, 42.5 MB offline
- Accuracy: 90–92% on text classification (plaintext: 90–91%); IMDB: 78.4% vs 83.5% — larger gap on harder tasks
- vs HErBERT baseline: **7.7–11.9× faster**, 3–11% better accuracy

**Implications:**
- 5.15s for a 1–2 layer micro-BERT is not representative of RAG-scale models (BERT-Base has 12 layers)
- The round-complexity reduction is architecturally important but latency numbers need scaling to full models

**Used in:** Research

---

## D. TEE-Based Private Generation

### 14. Petridish — Confidential Prompting via CVM
https://arxiv.org/abs/2409.19134
**Li, Gim, Zhong | 2024**

**Target:**
- **Model:** Any LLM hosted inside a Confidential VM (CVM/TDX)
- **Retrieval:** Not addressed; applicable to any retrieval method
- **Embedding:** Not addressed

**Approach:**
Runs the LLM service inside a **Confidential Virtual Machine** (CVM — AMD SEV-SNP or Intel TDX). Introduces **Secure Partitioned Decoding (SPD)**: the service is split into a per-user process (handles prefill and attention with user's prompt) and a shared service process (batches attention scores from all users for token generation). The CVM protects both the LLM weights from the cloud operator and the user prompt from other users.

**How it connects to RAG:**
The user assembles plaintext context (query + retrieved docs) and sends to the CVM. The CVM handles generation internally. The cloud infrastructure operator cannot access the prompt or generated response. This is the simplest end-to-end solution for RAG with a cloud provider: use a CVM-hosted LLM.

**Privacy/security model:**
CVM hardware isolation (TEE-level): cloud operator cannot read memory inside the CVM. Remote attestation verifies the software stack. Both LLM weights and user prompts are protected. Limitation: the CVM's OS kernel and the LLM service code are trusted.

**Performance:**
- Full utility preserved — output identical to non-CVM deployment
- No latency numbers published (paper focuses on architecture, not benchmarks)
- Startup overhead: scales linearly with tenant count (manageable)

**Implications:**
- Practical path to private RAG generation today: use a CVM-hosted model API
- Requires trust in the CVM hardware (AMD/Intel) and the TEE stack — weaker than cryptographic guarantees
- Compatible with any retrieval scheme: after private retrieval, the assembled plaintext context goes to the CVM LLM

**Used in:** Research (Yale); architecture directly applicable to AMD SEV-based cloud LLM services

---

### 15. Portcullis — PII-Anonymizing Privacy Gateway for LLM
https://doi.org/10.1609/aaai.v39i1.32088
**Zhan, Zhang et al. | AAAI 2025**

**Target:**
- **Model:** GPT-4o mini and other cloud LLMs
- **Retrieval:** Not addressed; operates as a gateway between retrieval and generation
- **Embedding:** Not addressed

**Approach:**
A **TEE-attested privacy gateway** sitting between the user and the cloud LLM. Portcullis runs inside a secure enclave, receives the user's assembled prompt (query + context), anonymizes PII entities (names, emails, medical terms) via parallel substitution, forwards the anonymized prompt to the cloud LLM, and accurately reconstructs the response by reversing the substitution map. The gateway is attested — users can verify the anonymization code is correct.

**How it connects to RAG:**
Portcullis inserts itself after retrieval: assembled context (query + retrieved docs) enters the enclave, PII is replaced with pseudonyms, the anonymized RAG prompt goes to GPT-4, and the response is de-anonymized before returning to the user.

**Privacy/security model:**
TEE-attested gateway: cloud LLM sees anonymized text, not real PII. Threat model: curious cloud LLM provider (e.g., OpenAI seeing medical records). The gateway itself is attested to be running the correct anonymization code.

**Performance:**
- Masking/unmasking: **96× faster** than Hide-and-Seek
- Accuracy on Enron dataset: >0.1 better than Hide-and-Seek for GPT-4o mini
- False positive/negative rates: equal to or better than existing solutions

**Implications:**
- Does not hide query semantics — an adversary who sees the anonymized prompt can still infer topic/domain
- Strong defense for PII protection; weaker as a general query-privacy mechanism
- Most practical commercial-adjacent approach to private RAG generation today

**Used in:** Research (AAAI 2025); architecture is commercially deployable

---

### 16. SCX — Stateless KV-Cache Encoding for Cloud LLM
https://doi.org/10.1145/3718958.3750509
**Yuan, Zhang, Zeng et al. | 2025**

**Target:**
- **Model:** LLaMA-7B and other large transformers
- **Retrieval:** Not addressed; designed for the generation step
- **Embedding:** Not addressed

**Approach:**
The user sends the prompt to the cloud LLM in plaintext, but the LLM's **KV-cache** (key-value attention cache computed from the prompt) is **encoded with a user-controlled key** before storage on the server. Each autoregressive generation step requires the user to re-encrypt the KV-cache. The server can neither recover the input from the KV-cache nor complete next-token prediction without the user's encoding.

**Privacy/security model:**
The cloud server stores only encoded KV-caches; the plaintext context cannot be reconstructed without the user key. The server sees the token stream as it is generated (one token at a time) but cannot reconstruct the original prompt post-session.

**Performance:**
- LLaMA-7B: **36 ms** per autoregressive step (vs. minutes for MPC approaches)
- **85% further reduction** in KV-cache communication with advanced cache management
- Output: zero loss vs plaintext inference — identical outputs

**Implications:**
- 36ms is orders of magnitude faster than any MPC/FHE approach — the gap is 5,000×
- Privacy is weaker: server still processes the plaintext prompt during prefill; only post-hoc KV-cache recovery is blocked
- Useful for protecting prompt content from being *stored and later retrieved* by the provider; not protection against real-time interception
- Complementary to Petridish/CVM: a CVM + SCX gives both real-time prompt isolation and post-session KV-cache protection

**Used in:** Research (Fudan / SJTU)

---

## E. TEE Split-Inference (TEE + GPU Asymmetric Decomposition)

> Co-locates a TEE with a GPU and splits the transformer across the trust boundary. The TEE holds a small sensitive portion; the GPU runs the bulk of the compute on an obfuscated representation. Applicable to **any** transformer — embedding, reranker, generator — though the published works target either vision DNNs or decoder LLMs; encoder-only / reranker variants are extrapolations.

### E.1 Delta / AsymML / 3LegRace — Low-rank / Residual Asymmetric Split
**Niu, Zhang, Zhu, Yi 2022 (3LegRace, PoPETs 2022)**; **Niu et al. 2022 (AsymML)**; **Niu, Ali, Prakash, Avestimehr 2023 (Delta, arXiv 2312.05264, "All Rivers Run to the Sea")**

**Target in the paper:** DNN **training** on **vision models** (AlexNet, VGG, ResNet) — not transformers, not inference. Delta adds a formal DP bound to the asymmetric flow.

**Approach:** Every weight matrix `W = W_s + W_r` via SVD / rank-k truncation. TEE holds the low-rank sensitive factor `W_s`; GPU holds the high-rank residual `W_r`. Partial activations `W_s·x` + `W_r·x` are summed to reconstruct the full output. Privacy: `W_r` alone cannot reconstruct `W` without `W_s`; Delta adds DP noise on the residual path for a formal bound.

**Privacy model:** Honest-but-curious GPU that sees `W_r·x` and partial activations. Delta's DP bound extends to colluding GPU side channels.

**Performance:**
- **7.6× training speedup** over pure-TEE training (3LegRace, on AlexNet / ResNet — image models)
- No published transformer numbers; no reranker or embedding numbers
- Inference overhead depends on rank-k per layer: low k = smaller TEE, weaker security; high k converges to pure-TEE

**Applicability to RAG:**
- **Generation:** in principle yes, but no published LLM split via this approach
- **Embedding / reranker:** would require decomposing every BERT weight matrix (12 layers × 6 matrices = 72 decompositions) and post-factorization fine-tuning to recover NDCG@10 / MRR
- DP budget compounds across layers — loose `ε` at aggregate

**Status:** Research; published only for vision training. Transformer port is a research project, not drop-in.

---

### E.2 ObfuscaTune — Orthogonal Activation Obfuscation
**Frikha et al. 2024** (already in Edgequake) | [Paper](https://arxiv.org/abs/2407.02960)

**Target in the paper:** Decoder-only **LLM fine-tuning + inference** (GPT-2 family; extensible to Llama-class). Boundary layers in TEE, transformer blocks on GPU. RAG-reranking is listed as future work.

**Approach:** Input embedding table + lm_head (~5% of parameters) stay in TEE. Per-batch random orthogonal matrix `Q` (via QR decomposition) drawn in TEE. Hidden state sent to GPU as `H̃ = Q·H`; GPU runs transformer blocks (attention, FFN) on `H̃`. TEE applies `Q⊤` to recover. Orthogonality preserves inner products through self-attention when `Q` is applied consistently to `Q, K, V`; LayerNorm and softmax require targeted fixes.

**Privacy model:** Honest-but-curious GPU. Security argues that per-batch fresh `Q` defeats accumulation attacks (BSS/ICA). **No formal DP bound.**

**Performance (from the paper):**
- **1.5× (GPT2-small) to 4.3× (GPT2-XL)** inference slowdown vs unprotected baseline
- Accuracy within 1–2pp of unprotected across WebQuestions / OpenBookQA / PIQA / SciQ
- TEE footprint: **5.2% of GPT2-XL parameters** inside enclave

**Applicability to RAG:**
- **Generation:** the paper's primary target
- **Embedding (encoder):** architectural pattern ports — TEE holds embedding table + pooling head, GPU runs encoder blocks on `Q·H`. No published encoder variant; would need LayerNorm/softmax-preservation work
- **Reranker (cross-encoder BERT):** similar encoder-variant extrapolation — same boundary-layer split, small TEE footprint because no lm_head

**Status:** Published for decoder LLMs; encoder / reranker variants are extrapolations.

---

### E.3 GELO — GPU-offloaded Encrypted Linear Operations
**Belikova, Fedotov 2026** (already in Edgequake) | [Paper](https://arxiv.org/abs/2603.05035)

See §F.17 below for the full entry. GELO belongs in this category: TEE holds a non-orthogonal per-batch invertible matrix `A`, GPU runs linear layers on `A·H`. Non-linear ops (GeLU, LayerNorm, Softmax) stay in TEE — larger TEE footprint than ObfuscaTune but no orthogonality constraint on the obfuscation matrix.

---

### E.4 Shredder — Learned Additive Noise at a Cut Layer
**Mireshghallah, Taram, Jalali, Hashemi, Tullsen, Esmaeilzadeh 2020 (ASPLOS)**

**Target in the paper:** **CNN image classification** (AlexNet, VGG on CIFAR / ImageNet) — not transformers.

**Approach:** Vertical split at a chosen layer: layers 1..k in TEE / edge, k+1..N on GPU / cloud. At the cut, additive noise `η ~ learned-distribution` is injected. Noise distribution is trained (typically few epochs) to maximize mutual-information reduction between noisy activations and input, subject to a utility constraint. Model weights stay frozen.

**Privacy model:** Honest-but-curious cloud. **Empirical MI reduction**, not a formal DP bound. Vulnerable to inversion attacks with auxiliary information or many correlated queries.

**Performance (from the paper):**
- **74% MI reduction** at **1.58% accuracy loss** (AlexNet / VGG on image classification)
- Per-inference latency overhead: trivial (noise sampling only)
- No transformer experiments

**Applicability to RAG:**
- **Generation:** transferable; cut-layer choice on a 32-layer LLM is unexplored
- **Embedding / reranker:** cut layer on a 12-layer BERT is non-trivial — too early (after embeddings) leaks text; too late (final layer) means TEE does most of the work. Reasonable cut points: after layer 6 for embeddings; after final encoder block (before scoring head) for a reranker's `[CLS]` representation.

**Status:** Published for vision; no transformer numbers. Straightforward to port since no model modification needed.

---

### E.5 Privacy-Aware Split Inference (arXiv 2026)
[Paper: 2506.xxxxx]

**Target in the paper:** Generic LLM split inference over WAN. Embedding layers + first N transformer layers local; middle layers cloud; final layers optionally local.

**Approach:** Vertical split with no obfuscation applied at the cut — relies on the information-theoretic difficulty of inverting mid-network activations into tokens. Split depth chosen to trade off token-recovery risk against local-compute budget.

**Privacy model:** Practical, not cryptographic. **35–59% of tokens recoverable** from mid-network activations via InversionNet / gradient-based attacks depending on split depth.

**Performance:**
- **8–9 tokens/s** on a 7B model over 80 ms WAN
- **8 KB/token** transmitted (compressed activations)
- Local compute scales with split depth — suitable for M-series Mac / edge hardware, not thin clients

**Applicability to RAG:**
- **Generation:** direct fit
- **Embedding / reranker:** shallow models (12 layers) give fewer split points — cut after layer 6 transmits ~50% activation entropy

**Status:** Published; practical privacy, no formal guarantee.

---

### E.6 Comparison of E.1–E.4 (TEE Split-Inference mechanisms)

All three co-locate a TEE with a GPU and split a transformer across the trust boundary. They differ on **what gets placed where**, **what the obfuscation primitive is**, and **what kind of guarantee they deliver**.

| | **Delta / AsymML / 3LegRace** | **ObfuscaTune** | **GELO** | **Shredder** |
|---|---|---|---|---|
| **Published on** | Vision training (AlexNet / ResNet) | Decoder LLM fine-tuning + inference (GPT-2) | Decoder LLM inference (Llama-2-7B) | Vision inference (AlexNet / VGG) |
| **What splits** | Every weight matrix `W = W_s + W_r` | Layer-wise boundary: embedding + lm_head in TEE, transformer blocks on GPU | TEE holds non-linear ops + key `A`; GPU runs linear ops on `A·H` | Vertical cut at layer `k`; TEE runs 1..k, GPU runs k+1..N |
| **TEE holds** | Low-rank sensitive factor of every matrix | Input embedding + output head (~5% params) | Key matrix `A` + non-linear ops (GeLU, LayerNorm, Softmax) | Layers 1..k of the model |
| **GPU holds** | Residual factors (bulk of parameters) | All transformer blocks | Linear projections (76% of MACs) | Layers k+1..N in plaintext |
| **What crosses the boundary** | Partial activations `W_s·x` + `W_r·x` | Obfuscated hidden states `H̃ = Q·H` (orthogonal Q) | Obfuscated hidden states `U = A·H` (non-orthogonal, invertible A) | Noised activations `h̃ = h + η` |
| **Obfuscation primitive** | Matrix decomposition | Orthogonal linear transform | Non-orthogonal invertible linear transform | Additive learned noise |
| **Formal privacy** | Delta: DP bound (3LegRace: heuristic low-rank) | Informational — argues per-batch Q defeats BSS/ICA accumulation | Informational — `<0.28` cosine ICA recovery, no DP | Empirical 74% MI reduction, no DP |
| **Retraining needed** | Yes (post-factorization fine-tune) | No (Q sampled at inference) | No (A sampled at inference) | No model retrain; **noise distribution must be trained** |
| **Headline overhead** | 7.6× training speedup (vision) | 1.5–4.3× inference slowdown (GPT-2 family) | **20–30%** inference overhead (Llama-2-7B) | **1.58% accuracy loss** at 74% MI reduction (vision) |
| **Adversary it defends against** | Honest-but-curious GPU + DP for side channels | Honest-but-curious GPU; weak against batch-accumulation attacks | Honest-but-curious GPU with VRAM read access | Honest-but-curious cloud; weak against adversaries with auxiliary data |
| **TEE must be co-located with GPU?** | Yes | Yes | Yes | Yes |
| **Reranker-specific applicability** | Extrapolation; 72 decompositions for BERT | Extrapolation; small TEE (no lm_head) fits reranker well | Extrapolation; TEE must still run non-linear ops of the reranker | Extrapolation; cut after final encoder layer before scoring head |

### E.7 Honest caveats on "X-reranker" framings

None of these systems is published *specifically* for reranking. Any "ObfuscaTune-reranker", "Shredder-at-`[CLS]`", or "Delta-for-BGE" framing is an adaptation, not a cited result. The table above is descriptive; applicability claims to reranker / embedding workloads are extrapolations.

The closest **published** transformer-specific split-inference systems are:
- **ObfuscaTune** (decoder LLMs)
- **GELO** (decoder LLMs, already in Edgequake)
- **Privacy-Aware Split Inference** (generic LLM WAN split)

For reranker-specific adaptation, the clean near-term options remain the ones in `private-reranking-research.md` §3 proper — TEE-hosted cross-encoder (Opal) or client-side reranker (HR+QDA / BGE).

---

## F. Obfuscation-Based Private Generation

### 17. GELO — Good-Enough LLM Obfuscation
https://arxiv.org/abs/2603.05035
**Belikova et al. | 2026**

**Target:**
- **Model:** Llama-2-7B (7B parameters; hidden dim 4096, 32 attention heads)
- **Retrieval:** Compatible with any retrieval scheme; operates on assembled context
- **Embedding:** Not addressed (obfuscation acts on hidden states, not input embeddings)

**Approach:**
Splits the transformer across a client TEE and an untrusted cloud GPU. The client (inside a TEE) holds the embedding layer and a **random invertible matrix A**. Before sending the hidden state to the cloud GPU for expensive linear projections (Q/K/V/O projections, ~67% of multiply-adds), the client obfuscates it with A. The cloud GPU computes the obfuscated product and returns the result; the client applies A⁻¹ to recover the correct hidden state. Privacy relies on **Blind Source Separation (BSS) intractability** — an adversary with the obfuscated hidden state and the cloud's computation trace cannot recover the original text.

**How it connects to RAG:**
The assembled RAG prompt (query + retrieved context) enters the pipeline as normal text. The client's TEE handles tokenization and embedding; the obfuscated hidden states go to the cloud GPU. The cloud GPU never sees the plaintext prompt or the embedding layer weights. The final de-obfuscated hidden states are decoded by the client.

**Privacy/security model:**
Honest-but-curious threat model: adversary has real-time VRAM access on the cloud GPU. BSS intractability (not cryptographic hardness) is the privacy foundation. Defends against a GPU-side adversary reading VRAM; does not defend against a model-layer adversary or side-channel attacks on the TEE.

**Performance:**
- Overhead: **20–30%** at batch sizes 256–512
- Bottleneck: 81.4% of overhead from socket IPC (client-GPU communication), not the obfuscation math
- Offloads **~76%** of total linear algebra to untrusted GPU
- Quality: **100%** top-1 token match in float32; **98.8–99.8%** in bfloat16/float16

**Implications:**
- 20–30% overhead vs plaintext is the smallest overhead of any private generation scheme — orders of magnitude less than MPC/FHE
- The BSS-based privacy is weaker than cryptographic guarantees; adversary with enough samples may be able to invert
- Practical for cloud RAG deployments where the threat is a curious cloud provider with VRAM access, not a cryptographically sophisticated adversary
- The IPC bottleneck (81.4% of overhead) suggests that co-located TEE+GPU deployments would reduce overhead significantly

**Used in:** Research (2026)

---

### 18. OSNIP — Obfuscated Semantic Null Space Inference Privacy
https://arxiv.org/abs/2601.22752
**Cao et al. | 2026**

**Target:**
- **Model:** Llama-3.2-1B, 3B-Instruct, Qwen3-14B, Qwen3-32B
- **Retrieval:** Compatible with any retrieval; operates on input embeddings
- **Embedding:** Client-side; obfuscated before sending to cloud

**Approach:**
Projects the input embedding into a **semantic null space** — a subspace where the semantic content is preserved for downstream generation but is invisible to adversarial reconstruction. The null space is learned to be orthogonal to all known inversion attack directions. The obfuscated embedding is sent to the cloud LLM, which generates a response. The response is coherent because the null-space projection preserves the dimensions the LLM uses for generation, while destroying the dimensions that inversion attacks exploit.

**How it connects to RAG:**
The full RAG prompt (query + retrieved context) is embedded and obfuscated before sending to the cloud LLM. The LLM generates a response over the obfuscated representation. No client-side de-obfuscation is needed for the response — the response is directly readable.

**Privacy/security model:**
Defends against white-box adversaries with full model access and KNN retrieval / vocabulary-matching attacks. Formal privacy relies on the intractability of inverting the null-space projection given knowledge of the full model. Not cryptographically hard — a sufficiently powerful adversary may find the projection.

**Performance:**
- Overhead: **0.96 ms/prompt** (vs. 4.93–98.36 ms for competing obfuscation methods)
- Quality: **100.1% BERTScore** on CNN/DailyMail (negligible quality loss)
- Perplexity increase: **1–3%** (vs. 700–17,000% for competing methods)
- Models: up to Qwen3-32B tested

**Implications:**
- 0.96ms overhead is essentially free — the cheapest private generation approach in this survey
- 100.1% BERTScore means OSNIP introduces *no* quality loss, unlike DP perturbation approaches
- Privacy is stronger than DP-perturbation (more targeted against inversion attacks) but weaker than MPC/FHE (no formal cryptographic guarantee)
- Scales to 32B models — only MPC approaches with PermLLM match this scale, at 10,000× the latency overhead

**Used in:** Research (2026)

---

## G. Comparison Matrix

All entries normalized to per-inference or per-token numbers where reported; benchmark model indicated. LAN-1 Gbps unless noted.

| System | Approach | Benchmark model | Latency | Comm per inference | Quality loss | HW trust? |
|---|---|---|---|---|---|---|
| **Opal** | TEE + ORAM | Any (via API) | Small ORAM overhead on retrieval | Low | None | Yes (SGX/TDX) |
| **RAGtime-PIANO** | FHE end-to-end | Unspecified | Minutes–hours/query (estimated) | Not reported | Small (poly approx) | No |
| **RemoteRAG** | DP query + cloud LLM | Any cloud LLM (retrieval only) | **0.67 s retrieval** @ 10⁵ docs | **46.66 KB** | None on generation | No |
| **prRAG + CAPRISE** | Dist-preserving enc + cloud LLM | Any cloud LLM (retrieval only) | Enc: **15 ms / 128 queries**; gen: plaintext | Low enc; plaintext gen | Vec2Text BLEU 83→12.4 | No |
| **PermLLM** | 3-party MPC (perm + SS + HE) | ChatGLM-6B | **~3 s/token LAN**, ~7 s/token WAN | **~20 MB/token** | None (identical to plaintext) | No |
| **PUMA** | 3-party RSS-MPC | LLaMA-7B; BERT-Base | LLaMA-7B: **~200 s/token**; BERT-Base: **33.9 s / 128 tok**; GPT-2 Base: **15.5 s / 32 tok** | LLaMA-7B: **1.79 GB/token**; BERT-Base: **10.8 GB / 128 tok** | ≤0.011 GLUE acc drop | No |
| **SIGMA** | 2-party FSS | LLaMA-2-13B; GPT-2 | LLaMA-2-13B: **~44 s/token**; GPT-2: **~1.6 s/token** | Reduced vs GC (exact MB/token not published) | Comparable | No |
| **SHAFT** | 2-party MPC | BERT-Base (NLU) | Est. **~4.6 s LAN BERT-Base** (5.2× faster than BOLT's 25 s) | **~4.6 GB** (82% less than BOLT) | <1pp GLUE | No |
| **SPRINT** | 2-party MPC + DP fine-tune | RoBERTa | **~2.9 s LAN** (1.6× faster than SHAFT) | ~2.9 GB (1.6× less than SHAFT) | <1pp GLUE | No |
| **MERGE** | 2-party MPC (NLG) | GPT-2-base (124M); T5 (138M) | GPT-2 128-tok: **171 s** (~1.34 s/token); T5 128-tok: **144 s** (~1.12 s/token) | **121 GB / 128 tok** (GPT-2); **98 GB / 128 tok** (T5) | Small (poly approx) | No |
| **THE-X** | Single-server FHE (CKKS) | BERT-Base, BERT-Large | Minutes/inference (paper: no absolute number) | Low (single ciphertext round) | Negligible GLUE | No |
| **NEXUS** | Single-server FHE (non-interactive RNS-CKKS) | BERT-Base | **37.3 s CPU** / **0.88 s GPU** | **164 MB** | Comparable | No |
| **Euston** | Single-server FHE (non-interactive RNS-CKKS + SVD) | BERT-Base; GPT2-1.5B; LLaMA-3-8B | Per 128-tok prefill, amortized batch-32: BERT-Base **~20.3 s CPU / ~1.44 s GPU**; GPT2-1.5B **~40.6 s / ~4.25 s**; **LLaMA-3-8B ~113 s CPU / ~15.1 s GPU** | ~58 MB BERT-Base (2.8× less than NEXUS); user-side preproc 3100× less | Comparable (CKKS poly approx) | No |
| **BumbleBee** | 2-party HE + GC | LLaMA-7B | **~8 min/token** CPU | 80–95% less than prior HE+GC | Not specified | No |
| **CipherFormer** | 2-party HE + GC | BERT-small (1–2 layers) | **5.15 s online** + 14.07 s offline (L=1–2) | **32 MB online, 42.5 MB offline** | ~5% on hard tasks | No |
| **Petridish / CVM** | CVM (TDX/SEV-SNP) | Any LLM in CVM | **~4–8% overhead** (H100 CC); <10% CPU TEE | None | None | Yes (TDX/SEV) |
| **Portcullis** | TEE-attested PII gateway | GPT-4o class | Minimal (gateway overhead) | None | Equal/better on Enron | Yes |
| **SCX** | User-key KV-cache encoding | LLaMA-7B | **36 ms/autoregressive step** | 85% reduction with mgmt | None (identical output) | No (plaintext prefill) |
| **GELO** | TEE + GPU (non-orthogonal A·H) | Llama-2-7B | **20–30% overhead** vs plaintext GPU | IPC-bound (81.4% of overhead is socket IPC) | 100% top-1 fp32; 98.8–99.8% bf16/fp16 | Yes (TEE for client) |
| **ObfuscaTune** | TEE + GPU (orthogonal Q·H) | GPT-2 family | **1.5× (small) – 4.3× (XL) slowdown** | Not separately reported | Within 1–2pp baseline | Yes (TEE for client) |
| **Delta / AsymML / 3LegRace** | TEE + GPU (SVD `W=W_s+W_r`) | Vision training (no transformer benchmarks) | 7.6× training speedup vs pure-TEE (vision) | Not reported for inference | Unknown for transformers | Yes (TEE + GPU) |
| **Shredder** | Learned noise at cut layer | Vision (AlexNet/VGG) | Trivial noise overhead | Depends on cut point | **1.58% acc loss at 74% MI reduction** | Optional TEE |
| **Privacy-Aware Split Inference** | Vertical split over WAN | Generic 7B LLM | **8–9 tokens/s** over 80 ms WAN | **8 KB/token** | 35–59% token recovery risk | No |
| **OSNIP** | Null-space embedding projection | Up to Qwen3-32B | **+0.96 ms/prompt** | None extra | 100.1% BERTScore CNN/DailyMail | No |

---

## Key Observations

**1. No single system delivers strong cryptographic privacy end-to-end at usable speed.**
The strongest guarantees (MPC: PermLLM/PUMA/SIGMA, FHE: THE-X/BumbleBee) cost 44–200 seconds/token. The lightest approaches (GELO: 20–30% overhead, OSNIP: 0.96ms) provide BSS/geometric privacy without formal cryptographic hardness. Petridish/CVM sits in the middle: hardware privacy (no overhead, no formal math, requires hardware trust).

**2. The retrieval-generation handoff is the most common privacy gap.**
RemoteRAG, prRAG, and p²RAG all do private retrieval but expose the assembled plaintext context to the cloud LLM at generation time. This makes their "private RAG" claims partial: corpus privacy at rest, but query + context visible at generation. Only Opal (TEE), RAGtime-PIANO (FHE), and obfuscation-based approaches (GELO/OSNIP) span both steps.

**3. The practical private RAG stack today is TEE-based.**
Petridish (CVM/TDX) + any private retrieval scheme = private RAG with no quality loss and no MPC/FHE overhead. The trust assumption is hardware (Intel TDX or AMD SEV-SNP), which is commercially available. This is what Edgeless/Privatemode, Opaque Systems, and similar vendors offer.

**4. MPC generation is approaching viability for BERT-class models.**
SHAFT, SPRINT, SecFormer, and MERGE all demonstrate that private NLU/generation for BERT/GPT-2-scale models is within seconds per inference. For RAG applications where the generator is a small domain-specific model (not a 70B frontier model), MPC generation is already close to viable.

**5. Obfuscation is the only approach that works with existing cloud LLM APIs.**
GELO and OSNIP are the only schemes in this survey that can wrap around an *existing* cloud LLM (OpenAI, Anthropic, etc.) without requiring the provider's cooperation. MPC/FHE require the server to run special protocols; TEE requires the provider to use CVMs. GELO/OSNIP just change what the client sends.

**6. The generation step leaks query intent in all non-TEE/non-MPC systems.**
Even if retrieval is private, any system that sends the assembled context to a cloud LLM in plaintext leaks the query topic, the retrieved document content, and the response to the LLM provider. DP on the retrieval step provides no protection at generation time.
