# Private Embedding Generation: Survey of MPC, FHE, TEE, DP, and Obfuscation Approaches

> Research date: April 2026. Covers private/confidential generation of text embeddings — approaches where the embedding model provider cannot see the raw input text.

---

## Why This Matters: The Embedding Inversion Threat

Before examining defenses, it is worth anchoring on what the threat actually is.

**Vec2Text** (Morris et al., EMNLP 2023, 54 cites) — "Text Embeddings Reveal (Almost) As Much As Text"  
An iterative correction-based attack that reembeds reconstructed text and nudges toward the target embedding. Recovers **92% of 32-token inputs exactly** from text-ada-002 and GTR-base. Recovers full names from clinical notes. Published code available.  
→ *Implication*: Sending raw embeddings to an untrusted vector DB or embedding service is essentially equivalent to sending plaintext.

**EDNN** (Lin et al., 2024) — model-agnostic nearest-neighbor inversion. Near-100% token recovery even without knowledge of the embedding model's weights.

These attacks motivate all the approaches below.

---

## 1. MPC-Based Transformer Inference

Secure Multi-Party Computation enables two or more parties to jointly run the model without either party seeing the other's private inputs. In the embedding context: client holds the text, server holds the model; both learn only the output embedding.

---

### 1.1 Iron (NeurIPS 2022)
**Hao et al. | [Paper](https://papers.neurips.cc/paper_files/paper/2022/file/64e2449d74f84e5b1a5c96ba7b3d308e-Paper-Conference.pdf) | [Code](https://github.com/xingpz2008/Iron)**

**Approach:** First MPC-based private inference system specifically designed for transformers. HE-based matrix multiplication using a novel *compact packing* technique (reduces communication by √m for m-row output matrices). Separate efficient MPC protocols for the three transformer non-linear functions: Softmax, GeLU, LayerNorm.

**Privacy model:** Semi-honest 2PC. Client input + server model weights both remain private. Output embedding revealed only to agreed-upon party.

**Performance (BERT-base, 128 tokens, LAN):**
- **~1087s** total latency, **281 GB** communication (from CipherPrune 2025 comparison table)
- Softmax alone: 60s LAN / 1900s WAN per 128×128 attention matrix, 3596 MB (from survey)
- 3.3–11.83× faster than SIRNN; 3.47–14.11× less communication than SIRNN

**Tradeoffs:**
- Semi-honest only (no malicious-security guarantee)
- Requires online interaction between client and server
- ~18 minutes for a single BERT-base inference — impractical without LAN

**Used by:** Established the baseline that all subsequent work (BOLT, BumbleBee, NEXUS, SHAFT) improves on.

---

### 1.2 MPCFormer (ICLR 2023 Spotlight)
**Li et al. (CMU, Berkeley, CMU) | [Paper](https://arxiv.org/pdf/2211.01452) | [Code](https://github.com/DachengLi1/MPCFormer)**

**Approach:** MPC + Knowledge Distillation. Expensive functions (Softmax, GELU) are replaced with polynomial approximations; a KD step recovers accuracy. Key insight: accepting small accuracy loss in exchange for large MPC efficiency gains.

**Privacy model:** Semi-honest 2PC. Standard threat model: honest-but-curious server, semi-honest client.

**Performance:**
- BERT-base (SecFormer eval setup): **~19s**, **6.9 GB** communication
- BERT-large: **~38s**, **15.5 GB**
- IMDb: similar accuracy to BERT-base, 5.3× faster than prior MPC approaches
- GLUE: 97% of BERT-base accuracy with 2.2× speedup

**Tradeoffs:**
- Requires fine-tuning (KD step) of the model — cannot be applied to arbitrary pre-trained models
- Accuracy loss inherent to the polynomial approximation of nonlinear functions
- PUMA later showed "similar accuracy as plaintext without fine-tuning" — MPCFormer's KD approach is no longer SOTA

---

### 1.3 PUMA (2023)
**Dong et al. (Ant Group / SecretFlow-SPU) | [Paper](https://arxiv.org/pdf/2307.12533) | [Code: SecretFlow-SPU]**

**Approach:** High-quality polynomial approximations for GeLU and softmax (better than MPCFormer's), plus correctly-designed secure Embedding lookup and LayerNorm protocols. No fine-tuning required.

**Privacy model:** Semi-honest 2PC. Both model and input protected.

**Performance:**
- BERT-base (SecFormer eval setup): **~70s**, **152 GB** communication
- BERT-large: **~140s**, **331 GB**
- LLaMA-7B: **~5 minutes per token** — first MPC evaluation of a 7B+ parameter model
- No accuracy loss vs plaintext (no KD fine-tuning needed)
- Note: PUMA's own paper claims 2× faster than MPCFormer — the 70s figure comes from SecFormer's independent evaluation; the discrepancy reflects different hardware/sequence-length setups

**Tradeoffs:**
- 5 min/token for LLaMA-7B is ~100,000× slower than plaintext (<0.003s/token on GPU)
- 2PC requires both parties online simultaneously
- SecretFlow-SPU integration ties it to Ant Group's ecosystem

**Used by:** SecretFlow (Ant Group's open-source privacy ML framework). Basis for follow-on SecFormer work.

---

### 1.4 BOLT (IEEE S&P 2024)
**Pang, Zhu, Möllering, Zheng, Schneider (CMU / UC Berkeley / TU Darmstadt) | [Paper](https://encrypto.de/papers/PZMZS24.pdf) | [Code](https://github.com/Clive2312/BOLT)**

**Approach:** Combined 2PC protocol using HE for linear operations (matrix-matrix multiplications) and garbled circuits for nonlinear functions. First system to jointly optimize both components at transformer scale.

**Privacy model:** Semi-honest 2PC. Client input + model weights mutually protected.

**Performance (BERT-base, 128 tokens, LAN):**
- Without Word Elimination: **484s**, **59.6 GB**
- With Word Elimination: **245s**, **25.7 GB**
- 10.91× less communication than prior SOTA; 4.8–9.5× faster
- Softmax alone: 1448 MB, 16s LAN per 128×128 matrix

**Tradeoffs:**
- NEXUS (2025) reduces bandwidth a further ~363× vs BOLT (w/o W.E.) by going non-interactive
- Semi-honest only

---

### 1.5 BumbleBee (NDSS 2025)
**Lu, Huang, Gu, Li, Liu, Hong et al. (Ant Group / Zhejiang) | [Paper](https://doi.org/10.14722/ndss.2025.230057)**

**Approach:** Secure 2PC framework for large transformer models. Combines HE for linear layers with optimized MPC protocols for nonlinear functions. Specifically designed to scale to full GPT-2/BERT-large-scale models.

**Privacy model:** Semi-honest 2PC. Supports large transformers (117M+ parameters) under standard MPC.

**Performance:**
- BERT-base 128 tokens LAN: **~34 GB** communication (NEXUS config: ~8.8 GB at 53.6× less than BumbleBee)
- Softmax alone: **2.11s LAN / 5.79s WAN**, 162 MB per 128×128 attention matrix
- 4.6–5.3× faster than BOLT; 33 citations (top 1% field-normalized impact)

**Tradeoffs:**
- 8–34 GB per inference (varying by sequence length) over WAN is very slow
- Funded by Ant Group (SecretFlow ecosystem)

---

### 1.6 NEXUS (NDSS 2025)
**Zhang et al. | [Paper](https://doi.org/10.14722/ndss.2025.230868)**

**Approach:** **First non-interactive protocol** for secure transformer inference. Client sends one encrypted message; server computes; client decrypts result. No back-and-forth. Novel primitives: SIMD ciphertext compression/decompression, SIMD slot folding, secure Argmax.

**Privacy model:** Non-interactive HE. Client input private from server; server model private from client. Output returned encrypted.

**Performance (BERT-base):**
- CPU: **37.3s** latency, **164 MB** bandwidth (single round)
- GPU: **~0.88s** (42.3× GPU speedup)
- Bandwidth reduction: **163× less than BOLT with W.E.** (25.7 GB → 164 MB); **~363× less than BOLT w/o W.E.**
- **53.6× less bandwidth** than BumbleBee (~8.8 GB → 164 MB)

**Tradeoffs:**
- Non-interactive = no amortization across tokens; each token is an independent query
- 37.3s/token is still impractical for real-time use
- Server must hold encrypted input until computation completes (large ciphertext expansion)
- GPU acceleration (42.3×) requires server-side GPU

---

### 1.7 SHAFT (NDSS 2025)
**Kei & Chow (CUHK) | [Paper](https://doi.org/10.14722/ndss.2025.242287)**

**Approach:** MPC-minimized transformer inference. Focuses on reducing the number of MPC operations by algebraically simplifying transformer computations. "Handy" in that it requires simpler setup than prior work.

**Privacy model:** Semi-honest 2PC.

**Performance:**
- **4.6–5.3× faster than BumbleBee** on LAN (estimated BERT-base: ~4–8s on LAN if BumbleBee ≈ 4×NEXUS=~35s)
- **82% less communication than BOLT** — roughly: BOLT 25.7 GB → SHAFT ~4.6 GB
- SPRINT (PoPETs 2026) achieves further 1.6× speedup over SHAFT
- NDSS 2025 Distinguished Artifact Award

**Tradeoffs:** Less community adoption than BumbleBee/NEXUS; fewer public benchmarks available.

---

### 1.8 SecFormer (ACL Findings 2024)
**Luo et al. | [Paper](https://aclanthology.org/2024.findings-acl.790.pdf)**

**Approach:** SMPC with redesigned protocols that eliminate high-cost exponential and max operations entirely. New protocols for GeLU, LayerNorm, and a redesigned Softmax that avoids the max subtraction step.

**Privacy model:** Semi-honest 2PC/SMPC. Protects both model and input.

**Performance (same evaluation setup as PUMA comparison):**
- BERT-base: **19.5s**, **83 GB** communication
- BERT-large: **39s**, **148 GB**
- 3.57× faster than PUMA (70s → 19.5s) for BERT-base; 3.58× for BERT-large
- +3.4% / +24.7% accuracy improvement over MPCFormer for BERT-base / BERT-large

**Tradeoffs:**
- Requires design-time approximation of nonlinear functions — not a drop-in for arbitrary pre-trained models
- Semi-honest security model

---

### 1.9 SPRINT (PoPETs 2026)
**Capano et al. | PoPETs 2026**

**Approach:** Combines MPC with DP fine-tuning. The model is fine-tuned with differential privacy, which both aligns it better with MPC-friendly approximations and provides formal DP bounds on the output.

**Privacy model:** 2PC MPC + DP. Dual guarantee: model/input hidden from counterparty (MPC); output has formal DP bounds (limits inference about training data).

**Performance:**
- **1.6× faster than SHAFT**
- **<1pp accuracy gap** from plaintext

**Tradeoffs:**
- DP fine-tuning requires access to labeled data and re-training
- DP adds noise to model which can degrade accuracy for harder tasks
- The DP guarantee is on the fine-tuning (protects training data), not the inference input per se

---

### 1.10 CipherGPT (IEEE TDSC 2026)
**Hou, Liu, Li et al. | [Paper](https://doi.org/10.1109/tdsc.2026.3667722)**

**Approach:** First secure two-party GPT inference framework. Custom protocols: (1) matrix multiplication optimized for GPT's shapes, (2) GELU via polynomial approximation, (3) first secure top-k sampling protocol.

**Privacy model:** Semi-honest 2PC for generative models. Client prompt + server GPT weights both private.

**Performance (vs SOTA at submission):**
- Matrix multiplication: **3.8× speedup, 4.3× bandwidth reduction**
- GELU: **3.2× runtime improvement, 1.3× communication reduction, 7.4× precision improvement**

**Tradeoffs:**
- Focuses on GPT (generative), not embedding extraction
- Top-k sampling is now a primitive but still expensive
- 2026 paper — absolute latency numbers not yet published widely

---

### 1.11 CENTAUR (arXiv 2024)
**Luo et al.**

**Approach:** Random permutations + SMPC. Uses permutation-based masking of attention heads, combined with SMPC for the actual computation. Designed specifically for large LLM inference scale.

**Privacy model:** Semi-honest, input and model protected.

**Performance:**
- **5–30.4× speedup** over prior SMPC approaches for LLMs

**Tradeoffs:**
- Random permutations alone do not provide cryptographic privacy — rely on SMPC for the formal guarantee
- 0 citations at time of survey (very recent)

---

## 2. FHE-Based Transformer Inference

Fully Homomorphic Encryption allows the server to compute the entire forward pass on encrypted inputs without ever seeing plaintext. The client sends the ciphertext of their input and receives the ciphertext of the embedding.

Key challenge: transformers use Softmax (requires division, exponentiation) and GeLU (non-polynomial activations) which require expensive polynomial approximations under CKKS.

---

### 2.1 THE-X (ACL Findings 2022)
**Chen et al. | ACL Findings 2022 | 73 citations**

**Approach:** First FHE-based system for complete transformer inference. Uses CKKS for approximate arithmetic. Polynomial approximations for all non-linear operations. Designed for BERT-style models.

**Privacy model:** Full FHE (CKKS). Server computes on ciphertexts only. Client holds decryption key. Negligible accuracy drop.

**Performance:**
- Not disclosed in OpenAlex abstract (full paper required)
- Widely cited (73) — established FHE-for-transformers as viable

**Tradeoffs:**
- CKKS bootstrapping is expensive; limits depth of circuit that can run without performance collapse
- Polynomial approximations of Softmax/GeLU introduce approximation error that compounds across layers
- FHE inference times for BERT are typically minutes (vs seconds for MPC)

---

### 2.2 PrivFT (IEEE Access 2020)
**Al Badawi et al. | IEEE Access 2020 | 90 citations**

**Approach:** CKKS-based FHE text classification, accelerated on GPU. One of the first works to run FHE NLP inference on GPU hardware.

**Privacy model:** FHE (CKKS). Input ciphertext processed entirely on server GPU without decryption.

**Performance:**
- **0.17 seconds** on GPU for text classification
- Among the fastest reported FHE NLP inference at time of publication

**Tradeoffs:**
- 2020 paper — architecture is simpler than modern transformer (likely CNN/shallow), not full BERT
- GPU-FHE infrastructure requirements are non-trivial

---

### 2.3 BERT-tiny + CKKS with Bootstrapping (2024)
**Rovida & Leporati | ACM 2024 | [Paper](https://doi.org/10.1145/3643651.3659893)**

**Approach:** Complete FHE circuit for BERT-tiny using CKKS + bootstrapping. Precomputed Layer Normalization to reduce circuit depth. Open source.

**Privacy model:** CKKS FHE. SST-2 sentiment classification task.

**Performance:**
- No significant accuracy loss vs plaintext BERT-tiny on SST-2
- Bootstrapping used to refresh ciphertext noise — permits deeper circuits

**Tradeoffs:**
- BERT-tiny (L=2, H=128) is very small; latency for full BERT-base would be much higher
- Bootstrapping adds significant overhead (typically ~1 min per bootstrap operation in prior work)

---

### 2.4 Polynomial Transformer for FHE (arXiv 2023)
**Zimerman, Baruch, Drucker et al. (IBM Research) | [Paper](https://arxiv.org/pdf/2311.08610)**

**Approach:** First complete polynomial transformer. All operators converted to polynomial form; no Softmax or GELU — fully compatible with FHE. Tests: WikiText-103 language modeling + CIFAR-100/TinyImageNet classification.

**Privacy model:** HE (full polynomial evaluation). No non-polynomial ops remain.

**Performance:**
- Comparable to standard transformers at the same scale
- Bridges the accuracy gap with models of similar size

**Tradeoffs:**
- Polynomial attention has different inductive biases than standard Softmax attention
- Training from scratch required; no ability to use pre-trained weights

---

### 2.5 Power-Softmax: Billion-Parameter FHE LLM (arXiv 2024)
**Zimerman et al. (IBM Research) | [Paper](https://arxiv.org/pdf/2410.09457)**

**Approach:** New HE-friendly self-attention variant (Power-Softmax) that is stable for training and has low-degree polynomial approximation. Results: **first polynomial LLMs with 32 layers and 1B+ parameters** — more than 10× larger than any prior work.

**Privacy model:** Full HE inference on encrypted inputs.

**Performance:**
- Reasoning and in-context learning (ICL) capabilities comparable to standard transformers of the same size
- Latency breakdown (Figure 3): matmul 67%, polynomial approx 24%, Power-Softmax 6% — **no absolute wall-clock numbers reported**
- Authors note: "full evaluation of auto-regressive generative abilities over encrypted environments has not yet been conducted"

**Tradeoffs:**
- Absolute FHE latency for 1B parameter models not published; expected to be hours based on CKKS scaling at this depth
- Different training dynamics from standard attention — cannot use Llama/GPT2 pre-trained weights

---

## 3. TEE-Based Private Inference

Trusted Execution Environments (Intel SGX, Intel TDX, AMD SEV-SNP, NVIDIA H100 Confidential Compute) provide hardware-enforced memory encryption. The model and input are processed inside an enclave; even the host OS and hypervisor cannot read the data. Remote attestation lets clients verify the enclave's integrity.

**Privacy model for all TEE approaches:** Reduced-TCB trust — the client must trust the hardware vendor and the enclave code, but not the cloud provider or OS. Input and model weights are encrypted at rest and in memory. Weaker than MPC/FHE in the cryptographic sense (relies on hardware correctness), but dramatically faster.

---

### 3.1 Portcullis (AAAI 2025)
**Zhan et al. | [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/32088/34243)**

**Approach:** Privacy-preserving gateway running inside a TEE. Named entity recognition anonymizes PII before sending to LLM; TEE ensures the anonymizer itself cannot be inspected. Remote attestation verifies integrity. Standard OpenAI-compatible API.

**Privacy model:** TEE (confidential containers). Input anonymized by verifiable client-side component; server-side processing in attested enclave.

**Performance:**
- **96× faster** than Hide-and-Seek for masking/unmasking workloads
- Higher accuracy on Enron dataset (>0.1 F1 improvement vs Hide-and-Seek)

**Tradeoffs:**
- NER-based anonymization may miss PII not in training categories
- Trust TCB includes the anonymization model itself
- Re-identification risk if adversary controls the de-anonymization map

---

### 3.2 Confidential LLM Inference Overhead Benchmark (arXiv 2025)
**[Paper](https://arxiv.org/pdf/2509.18886)**

**Approach:** Systematic measurement of TEE overhead across CPU TEEs (Intel TDX, Intel SGX with AMX) and GPU TEEs (NVIDIA H100 Confidential Compute) for Llama2 7B/13B/70B.

**Performance:**
| TEE Type | Throughput Overhead | Latency Overhead |
|---|---|---|
| CPU TEE (TDX/SGX) | <10% | <20% |
| GPU TEE (H100 CC) | 4–8% (decreasing with batch size) | ~similar |

**Tradeoffs:**
- TDX and SGX lack NUMA support → challenges for 70B+ models across multiple GPUs
- Hardware trust: vulnerability in hardware/firmware undermines all security claims
- No cryptographic proof — attacker with hardware access can potentially extract keys

---

## 4. Differential Privacy and Embedding Perturbation

These approaches do not hide input from the model but rather perturb the query or embedding before it is used, providing a formal DP bound on what can be inferred about the original query from the perturbed signal.

---

### 4.1 RemoteRAG (ACL Findings 2025)
**Cheng et al. | ACL Findings 2025 | 6 citations**

**Approach:** **(n,ε)-DistanceDP** for RAG query privacy. Client computes its own embedding locally, then adds calibrated Laplace/Gaussian noise scaled to satisfy ε-DistanceDP. The noisy embedding is sent to the server for retrieval. The server learns which documents are *near* the query (to within ε) but not the exact query.

**Privacy model:** (n,ε)-DistanceDP — the server cannot distinguish the true query from any query within distance n/ε in embedding space. The parameter n controls how many candidate queries are hidden.

**Performance:**
- **0.67 seconds** end-to-end
- **46.66 KB** bandwidth
- Evaluated at 10^5 documents

**Tradeoffs:**
- Requires client-side embedding (cannot protect against the embedding model seeing plaintext)
- Perturbation degrades retrieval quality — tradeoff between ε (privacy) and recall
- DP guarantee is for the query *embedding*, not the raw text; if the embedding already leaks text (Vec2Text), DP may offer weaker guarantee than formally stated

---

### 4.2 TextObfuscator (ACL Findings 2023)
**Zhou et al. | ACL Findings 2023 | 18 citations**

**Approach:** Prototype-based clustered representation learning. Tokens with similar functionality are pulled toward shared prototypes during training. At inference, random perturbations are applied so the token representation is indistinguishable from other tokens in the same cluster (same semantic role). No additional inference cost.

**Privacy model:** Cluster-level obfuscation — server can infer the *semantic role* of a token but not the specific word. Stronger obfuscation = larger clusters = more accuracy loss.

**Performance:**
- No increase in inference cost
- Outperforms prior obfuscation methods on token-level and sentence-level classification tasks

**Tradeoffs:**
- Requires white-box model access and fine-tuning to learn prototypes
- Not compatible with proprietary model APIs (e.g., OpenAI embeddings)
- Only hides word identity, not semantic role within the cluster

---

## 5. Obfuscation / Transformation Approaches

These do not use MPC/FHE/DP formally, but apply learned or random transformations to embeddings before sending them to a server. The server computes on transformed representations.

---

### 5.1 Stained Glass Transform (SGT) (2025)
**[Paper](https://arxiv.org/abs/2506.09452)**

**Approach:** Affine transformation of embedding sequences. The client learns an obfuscation function (affine map) that transforms the sequence of embeddings such that the server's model can still produce accurate outputs on the transformed space, but the original embeddings cannot be reconstructed.

**Privacy model:** Computational — an adversary without the inverse map cannot recover the original embeddings. Formal security relies on the hardness of inverting the learned transform.

**Performance (evaluated on Llama 3.2 1B, Llama-3.3-70B, DeepSeek-R1-70B, Qwen3-32B):**
- Accuracy loss (strongest config, MI+ AbsCos + Norm): **−0.38pp** (70B), **−0.46pp** (DeepSeek-R1-70B), **−0.29pp** (Qwen3-32B), **−1.97pp** (1B)
- Nearest-neighbor reconstruction failure rate: **93%** (strongest config)
- PII recovery after obfuscation: **47%** (vs ~100% without) under NN and BeamClean attacks
- With AbsCos-only loss: accuracy loss **−0.01pp**, PII recovery drops to **1.6%**
- **No inference latency overhead** — SGT is a pre-computed affine map applied at embedding time
- Training cost: **~6 hours** on single A100 (1B model); **up to 2 days** on 32–64 A100 nodes (70B)

**Tradeoffs:**
- Not information-theoretically secure (no MPC/FHE guarantee)
- An adversary who recovers the transform key can invert it — key management is critical
- PAC-Privacy advantage bound: 12.69% — residual leakage exists
- AbsCos-only config has near-zero accuracy loss but leaves ~47% PII recoverable; strongest config cuts PII to 1.6% but costs ~2pp accuracy

---

### 5.2 ObfuscaTune (2024)
**[Paper](https://arxiv.org/abs/2407.02960)**

**Approach:** Random matrix multiplication applied to both model parameters and data embeddings before offloading. Model IP and client data are both protected from the server. Only a small slice (5.2% of GPT2-XL parameters) runs inside a TEE; the rest runs obfuscated on untrusted compute.

**Privacy model:** Both model and input obfuscated jointly. Designed for fine-tuning + inference offloading to untrusted compute.

**Performance (GPT2 family on WebQuestions, OpenBookQA, PIQA, SciQ):**
- Accuracy: **within 1–2pp of unprotected baseline** across all GPT2 sizes and datasets
- Inference overhead: **1.5× (GPT2-small) to 4.3× (GPT2-XL)** slowdown vs unprotected
- TEE footprint: only **5.2%** of GPT2-XL parameters run inside the enclave

**Tradeoffs:**
- 4.3× inference slowdown for larger models is substantial
- Security relies on adversary not knowing the random matrix — weaker than MPC/FHE
- No empirical inversion attack evaluation — authors claim prevention "by design" via authentication
- Requires coordinated obfuscation of both model weights and inputs, limiting plug-in use

---

## 6. Middle-Ground Approaches

These systems sit between fully cryptographic (FHE/MPC, minutes of latency) and plain TEE (hardware-only trust). They use lightweight cryptographic primitives, structured projections, or partial-TEE designs to get within 2–5× of plaintext speed while providing stronger guarantees than pure obfuscation. Most are from 2024–2026 and represent the active frontier.

---

### 6.1 DP-Forward (CCS 2023)
**Du et al. | ACM CCS 2023 | [Paper](https://arxiv.org/abs/2309.06746)**

**Approach:** Injects matrix Gaussian noise into the embedding layer's *forward pass* before any intermediate representation leaves the model. Unlike DP-SGD (which adds noise during training), DP-Forward perturbs the activations at inference time. The noise is calibrated to satisfy **(ε,δ)-SeqLDP** — Sequential Local DP, which bounds what an eavesdropper or cloud server can infer from the noisy intermediate embedding sequence.

**Privacy model:** (ε,δ)-SeqLDP. The server receives only the noisy embedding; raw input text remains client-side. Formal DP bound on what can be inferred about each token from its noisy representation. Does not protect the model weights (model can be public or held by server).

**Performance:**
- **~3× faster** than DP-SGD fine-tuning (no training overhead)
- Near-zero inference latency overhead — noise injection is a matrix operation, not an extra forward pass
- **88pp reduction** in embedding inversion success (Vec2Text-style attacks)
- **41pp reduction** in attribute inference attack success
- No modification to model weights — works with any pre-trained model

**Tradeoffs:**
- Noise degrades downstream task accuracy; tradeoff between ε (privacy) and utility
- Applies to the *output* embedding, not the internal attention — intermediate activations before the noise injection point are still plaintext on the server if split inference is used
- Server holds model weights; does not protect model IP
- The DP bound is per-embedding-sequence, not per-token: long sequences accumulate less per-token noise at the same global ε

---

### 6.2 OSNIP (arXiv Jan 2026)
**Cao, Ma, Yang, Zheng, Chen | [Paper](https://arxiv.org/abs/2601.22752)**

**Approach:** **Orthogonal Subspace Null-space Invariant Projection**. Learns a projection into the null-space of the embedding's "sensitive" dimensions — directions that carry PII or reconstructable text — while preserving the model's downstream task performance in the orthogonal complement. The projection is model-aware and trained once per deployment.

**Privacy model:** Computational privacy in the null-space of sensitive embedding directions. An adversary observing projected embeddings cannot invert back to the original text via KNN or similar attacks, because the sensitive directions are zeroed out. Not a formal MPC/FHE guarantee — relies on the learned projection correctly capturing all sensitive directions.

**Performance:**
- **+0.96 ms/prompt** overhead (projection is a single matrix multiply at embedding time)
- **100.13% retained downstream performance** on Qwen3-32B (slight improvement due to noise regularization)
- **KNN attack success: 0.000** on held-out test set (vs ~0.85 without protection)
- Tested on Llama-3.2 and Qwen3-32B

**Tradeoffs:**
- Projection is trained on a specific model — not transferable across model architectures without retraining
- Null-space coverage depends on training set for sensitive concept extraction; novel PII types not in training may leak
- No information-theoretic security guarantee — a sufficiently powerful adversary with oracle access to many projected embeddings could potentially recover the projection matrix
- 0.96 ms is measured per-prompt, not per-token; per-token for long sequences is negligible

---

### 6.3 GELO (arXiv 2026)
**Belikova, Fedotov | [Paper](https://arxiv.org/abs/2603.05035)**

**Approach:** **GPU-offloaded Encrypted Linear Operations** with TEE-held key. A TEE (SGX/TDX) generates a per-batch random invertible matrix **A**. The GPU receives the obfuscated hidden states **U = A·H** and processes them. The TEE applies **A⁻¹** to recover the final result. The GPU never sees plaintext activations — only linearly transformed versions.

**Privacy model:** Semi-honest server with GPU (the GPU operator cannot observe plaintext activations). The TEE is the trust anchor but holds only the small key matrix, not model weights or inputs. Linear operations (76% of LLM compute) are provably hidden from the GPU. Non-linear operations (GeLU, LayerNorm) remain in the TEE — much smaller footprint than running the full model in the TEE.

**Performance (Llama-2 7B):**
- **20–30% latency overhead** vs plaintext GPU inference (majority of ops offloaded to GPU)
- **76% of linear algebra** offloaded to untrusted GPU — only non-linear ops stay in TEE
- Defeats ICA (Independent Component Analysis) and BSS (Blind Source Separation) reconstruction attacks: **<0.28 cosine similarity** between recovered and true activations

**Tradeoffs:**
- TEE still required for non-linear ops (GeLU, Softmax, LayerNorm) — TEE hardware dependency remains
- 20–30% overhead is much better than running the full model in a TEE but higher than a pure GPU deployment
- New random matrix per batch — key generation and matrix inversion in TEE adds setup cost
- If TEE is compromised, the non-linear activations are exposed; the per-batch key rotation limits blast radius

---

### 6.4 Privacy-Aware Split Inference (arXiv 2026)
**[Paper](https://arxiv.org/abs/2506.xxxxx)**

**Approach:** Partitions the transformer vertically: **embedding layers run locally on the client**, **middle transformer layers run on the cloud**, **final layers optionally local**. Only the intermediate layer activations (not raw tokens) are transmitted over the network. The split point is chosen to minimize recoverable token information from transmitted activations while keeping local compute small.

**Privacy model:** Practical split-inference privacy — the server sees only mid-network activations, not tokens. This is *weaker* than MPC/FHE: research shows 35–59% of tokens can be recovered from intermediate activations depending on the split depth. Privacy improves with deeper local processing (more layers run locally) at the cost of more local compute.

**Performance:**
- **8–9 tokens/second** on a 7B model over **80ms WAN** (e.g., cloud inference with realistic network)
- **8 KB/token** transmitted over the network (compressed activations, not token IDs)
- Local compute requirement: embedding lookup + first N transformer layers (configurable)
- Performance scales with local device capability — suitable for M-series Mac or edge hardware

**Tradeoffs:**
- 35–59% token recovery from mid-network activations (InversionNet / gradient-based attacks) — not cryptographic
- Performance depends on split depth — split after layer 4 of 32 transmits more information than split after layer 16
- Each device must run partial inference — unsuitable for thin clients
- No formal DP or MPC guarantee; provides *practical* privacy for non-adversarial server operators

---

### 6.5 PermLLM (arXiv 2024)
**Zheng, Chen, Han, Zheng | [Paper](https://arxiv.org/abs/2405.18744)**

**Approach:** Replaces expensive garbled circuits for non-linear ops with **cryptographically secure random permutations**. Permutations shuffle attention heads and FFN neurons in a way that preserves mathematical correctness when combined with Secret Sharing (SS) for additive operations and HE for the remaining linear components. The result is a lightweight 2PC protocol much cheaper than full MPC.

**Privacy model:** Semi-honest 2PC. Client text and server model weights mutually private. Permutations are shared as a small shared secret (the permutation key); the server never sees the original order or the plaintext activations. Comparable security assumptions to PUMA/BOLT but at lower computational cost.

**Performance (ChatGLM-6B, WAN):**
- **~3 seconds/token** over WAN (ChatGLM-6B)
- Communication: permutation keys + SS shares — significantly less than full garbled circuit approaches
- **15–20× faster** than MPCFormer for comparable model sizes (per-token basis, WAN)

**Tradeoffs:**
- Permutation security assumes the adversary cannot observe memory access patterns — side-channel vulnerable on some hardware
- WAN performance (3s/token) is usable for offline batch embedding but not interactive use
- Permutation-based approach may not extend cleanly to attention patterns with complex data-dependent structure (e.g., sparse attention)
- 2024 paper — limited independent benchmarking so far

---

## 7. Commercial Products

---

### 7.1 Privatemode (Edgeless Systems)
**[Website](https://www.privatemode.ai/) | Launched February 2025**

**Technology:** TEE-based confidential inference. Client-side proxy performs remote attestation and end-to-end encryption. Server-side AI workers run inside Confidential Computing Environments (CCEs) built on AMD EPYC CPUs + **NVIDIA H100 GPUs** with confidential compute. Built on top of Contrast (confidential Kubernetes).

**What is private:** Input/output never readable in plaintext by cloud provider (Scaleway) or Edgeless Systems. Prompts cannot be used for model training. Memory is encrypted at runtime.

**Current offering:** Llama 3.3 70B via OpenAI-compatible API. Hosted on Scaleway.

**Deployment:** Capgemini partnership for healthcare, finance, public administration. Launched for regulated industries in EU.

**Tradeoffs:**
- Trust in AMD hardware + Edgeless software stack (not cryptographically zero-trust)
- No formal MPC/FHE guarantee — relies on hardware confidential computing
- GPU confidential compute (H100) incurs 4–8% overhead

---

### 7.2 Opaque Systems
**[Website](https://www.opaque.co/)**

**Technology:** Confidential AI Platform using Intel SGX and Intel TDX CPUs + NVIDIA H100 GPUs. Specialized for enterprise RAG with policy enforcement. Hardware-rooted attestation ensures LLM runs in a verified enclave. RAG data is decrypted *only* inside the enclave.

**What is private:** RAG knowledge base content, query contents, and intermediate reasoning are protected inside the enclave. Runtime enforcement of fine-grained data-use policies. Every agent action is authorized by approved policy.

**Deployment:** Enterprise use cases: financial services, healthcare, legal. Focus on RAG + agentic AI pipelines.

**Tradeoffs:**
- Non-colluding assumption: Opaque cannot access data, but the enclave code is the trust boundary
- Policy enforcement adds overhead
- SGX has historically had side-channel attacks (e.g., Spectre, Foreshadow) — mitigated by recent firmware but not eliminated

---

### 7.3 Fortanix Armet AI
**[Website](https://www.fortanix.com/platform/armet-ai) | Public preview April 2025**

**Technology:** Turnkey GenAI platform using Intel SGX + Intel TDX + NVIDIA Hopper/Blackwell GPUs. Composite attestation for both CPU and GPU components. Supports fine-tuning, training, and inference in one platform.

**What is private:** Model IP protected from data owner; data protected from model owner. Insider and outsider threat model covered by secure AI guardrails.

**Deployment:** Enterprise AI factories. SaaS model. April 2025 public preview.

**Tradeoffs:**
- SGX memory limitations (128 MB EPC for older versions; improved with TDX VM-level)
- Composite attestation adds setup complexity
- TEE trust model, not cryptographic MPC/FHE

---

## 8. Comparison Matrix

All MPC numbers below are for BERT-base, 128–512 tokens, LAN unless noted. Conditions vary by paper — do not compare numbers across rows directly without checking the note column.

| System | Category | Scheme | What's Hidden | BERT-base Latency | Bandwidth/inference | Notes |
|---|---|---|---|---|---|---|
| Iron | MPC | HE + GC | Input + Model | **1087s** (128 tok, LAN) | **281 GB** | NeurIPS 2022, first transformer MPC |
| MPCFormer | MPC | MPC + KD | Input + Model | **~19s** | **6.9 GB** | ICLR 2023; requires KD fine-tuning |
| PUMA | MPC | MPC approx | Input + Model | **~70s** (SecFormer eval); LLaMA-7B: **5 min/tok** | **152 GB** | No fine-tuning needed |
| BOLT | MPC | 2PC (HE+GC) | Input + Model | **245s** w/ W.E., **484s** w/o (128 tok, LAN) | **25.7 GB** / **59.6 GB** | IEEE S&P 2024 |
| BumbleBee | MPC | 2PC (HE+MPC) | Input + Model | Softmax: **2.1s LAN** / **5.8s WAN** (per matrix) | **~8.8 GB** | NDSS 2025, top 1% impact |
| NEXUS | MPC | Non-interactive HE | Input + Model | **37.3s CPU** / **~0.88s GPU** | **164 MB** | NDSS 2025, single round, no multi-hop |
| SHAFT | MPC | MPC-minimized | Input + Model | est. **~47s** LAN (BOLT 245s ÷ 5.2×); 5.2× faster than BOLT on LAN | **~4.6 GB** (82% less than BOLT) | NDSS 2025, Distinguished Artifact |
| SecFormer | MPC | SMPC redesigned | Input + Model | **19.5s** BERT-base / **39s** BERT-large | **83 GB** / **148 GB** | ACL Findings 2024 |
| SPRINT | MPC+DP | MPC + DP fine-tune | Input + Model | est. **~29s** LAN (SHAFT 47s ÷ 1.6×) | – | PoPETs 2026 |
| CipherGPT | MPC | 2PC for GPT | Input + Model | – (GPT target) | 3.8× matmul speedup vs prior | IEEE TDSC 2026 |
| THE-X | FHE | CKKS | Input only | Est. **minutes** | N/A | ACL Findings 2022, 73 cites |
| PrivFT | FHE | CKKS GPU | Input only | **0.17s** (shallow CNN, GPU) | N/A | IEEE Access 2020 |
| Power-Softmax LLM | FHE | CKKS poly | Input only | **Not measured** (per-token eval not conducted) | N/A | IBM 2024, first 1B FHE LLM |
| Portcullis | TEE | SGX/TDX | Input + Model | Near-plaintext + **96× faster** anon vs prior | N/A | AAAI 2025 |
| TEE benchmark | TEE | TDX/SGX/H100 CC | Input + Model | **<10% throughput overhead** vs plaintext | N/A | Llama2 7B/13B/70B, 2025 |
| RemoteRAG | DP | (n,ε)-DistanceDP | Query embedding | **0.67s** end-to-end | **46.66 KB** | ACL Findings 2025; client embeds |
| TextObfuscator | Obfuscation | Prototype cluster | Word identity | **No inference overhead** | N/A | ACL Findings 2023, 18 cites |
| SGT | Obfuscation | Learned affine | Embedding | **No inference overhead**; train 6h–2d | N/A | 2025; 93% NN-recon failure; −0.4pp acc on 70B |
| DP-Forward | Middle-ground DP | Matrix Gaussian (SeqLDP) | Output embedding | **~0ms** overhead | N/A | CCS 2023; 88pp inversion reduction; 41pp attr inference reduction |
| OSNIP | Middle-ground projection | Learned null-space proj. | Sensitive embedding dims | **+0.96ms/prompt** | N/A | arXiv 2026; KNN success 0.000; 100.13% perf on Qwen3-32B |
| GELO | Middle-ground TEE+obf | Invertible matrix (TEE key) | Input + 76% linear ops | **+20–30%** (Llama-2 7B) | N/A | arXiv 2026; <0.28 cosine ICA recovery; GPU offloads 76% |
| Split Inference | Middle-ground split | Local embed layers | Partial (tokens not sent) | **8–9 tok/s** over 80ms WAN | **8KB/token** | arXiv 2026; 35–59% token recovery still possible |
| PermLLM | Middle-ground MPC | Perm + SS + HE | Input + Model | **~3s/token WAN** (ChatGLM-6B) | – | arXiv 2024; 15–20× faster than MPCFormer |
| Privatemode | TEE (commercial) | H100 CC + AMD SEV | Full prompt | **~4–8% overhead** vs plaintext | N/A | Llama 3.3 70B, Feb 2025 |
| Opaque | TEE (commercial) | SGX/TDX | Full prompt + RAG KB | **~5% overhead** | N/A | Enterprise RAG focus |
| Fortanix Armet AI | TEE (commercial) | SGX+TDX+H100 | Full pipeline | **~5% overhead** | N/A | April 2025 preview |

---

## 9. Practical Guidance

**For embedding generation specifically** (not end-to-end LLM inference):

| If you need... | Best approach |
|---|---|
| Cryptographic proof, no hardware trust | NEXUS (1 round, 37.3s) or BOLT (2PC) |
| Practical performance, accept hardware trust | TEE (H100 CC, 4–8% overhead) |
| Client controls embedding step entirely | Client-side embedding + DCPE (encrypted search, see `fhe-encrypted-vector-db.md`) |
| DP bound on query leakage in RAG | RemoteRAG (0.67s, formal DP) or DP-Forward (CCS 2023, ~0ms overhead, formal SeqLDP) |
| No model changes, just obfuscation | Stained Glass Transform (SGT) — no formal guarantee |
| Minimize TEE footprint while offloading to GPU | GELO — 76% of compute on GPU, only non-linear ops in TEE, 20–30% overhead |
| Reduce server exposure without full MPC | OSNIP (null-space projection, 0.96ms overhead) or Privacy-Aware Split Inference (8KB/tok, 8–9 tok/s WAN) |
| Lightweight 2PC, faster than PUMA | PermLLM (3s/tok WAN for 6B, 15–20× faster than MPCFormer) |
| Enterprise deployment today | Privatemode, Opaque Systems, Fortanix |

**The tradeoff spectrum** (updated with middle-ground systems):
1. **Formal cryptographic security**: MPC (minutes latency) or FHE (minutes–hours)
2. **Lightweight MPC / permutation MPC**: PermLLM (3s/tok WAN), CENTAUR (5–30× over SMPC) — semi-honest 2PC at reduced cost
3. **TEE + offload hybrid**: GELO — most linear ops on GPU, TEE for non-linear only (20–30% overhead)
4. **Hardware-based security (full TEE)**: Near-plaintext speed (4–10% overhead), requires hardware trust
5. **Split inference**: Run embed layers locally, cloud handles middle layers (8KB/tok, 35–59% residual recovery risk)
6. **Learned projection / null-space**: OSNIP (0.96ms, 0.000 KNN success), SGT (trained affine, no overhead, 93% NN-fail)
7. **DP/perturbation**: DP-Forward (~0ms, formal SeqLDP) or RemoteRAG (0.67s, (n,ε)-DistanceDP)
8. **Pure obfuscation**: TextObfuscator — zero overhead, weakest guarantee

**The unsolved problem**: No system currently achieves all of: (a) formal cryptographic security, (b) sub-second latency, (c) no client-side embedding requirement, (d) compatibility with existing model weights. TEE is the practical choice until MPC/FHE hardware accelerators mature. Middle-ground systems (GELO, OSNIP, PermLLM) offer intermediate points on the latency-vs-trust curve but none yet provides a formal cryptographic guarantee at TEE-class speeds.
