# Private & Oblivious Information Retrieval for RAG

> Research date: April 2026. Covers systems that make the *retrieval step* of RAG private — hiding query content, access patterns, or both from an untrusted retrieval server.
>
> **Scope:** Private Information Retrieval (PIR), Oblivious RAM (ORAM), encrypted/functional-encryption vector search, secret sharing, DP perturbation, and ZK verifiability — for sparse (keyword/BM25), dense (embedding/ANN), and hybrid retrieval.

---

## Why Retrieval Privacy Matters

Standard RAG sends a query embedding to an untrusted vector store. The server learns:
1. **The query content** — directly (sparse) or via embedding inversion (dense; Vec2Text recovers 92% of tokens)
2. **The access pattern** — which documents were retrieved, in which order, how often

Even if the embedding is perturbed, a server observing access patterns over time can reconstruct topics, interests, and identities. PIR/OIR systems address one or both leakage channels.

---

## A. Dense / Semantic Retrieval Privacy

### 1. Tiptoe — Private Web Search
https://dl.acm.org/doi/10.1145/3600006.3613134
**Henzinger, Dauterman, Corrigan-Gibbs, Zeldovich | CCS 2023 | 23 citations**

**Target:** Dense (embedding/ANN) — semantic nearest-neighbor search over text, image, audio, and code embeddings.

**Approach:**
Reduces private full-text search to private nearest-neighbor search over semantic embeddings. The client computes a query embedding locally; a new high-throughput **linearly homomorphic encryption (LHE)** protocol — a single-server PIR primitive based on LWE — performs private ANN search. The server never learns which embedding or document was matched. Supports text, text-to-image, audio, and code search with the same underlying protocol.

**Privacy/security model:**
Cryptography-only. No hardware enclaves, no non-colluding server assumption. Single-server computational PIR via LWE-based LHE. Server learns nothing about the query vector or which document was retrieved. Standard computational security under LWE hardness.

**Performance:**
- Corpus: **360 million web pages**, 45-server cluster
- Server compute: **145 core-seconds** per query (amortizable across servers)
- Client–server communication: **56.9 MiB** — 74% sent *before* the client enters its query (offline-amortizable pre-fetch)
- End-to-end latency: **2.7 seconds**
- Search quality (MS MARCO): average rank **7.7** vs 2.3 for non-private neural search; 6.7 for classical TF-IDF

**Tradeoffs:**
- Works well on conceptual queries ("knee pain"); degrades on exact string matches ("123 Main Street")
- 45-server cluster required for practical throughput — not single-machine deployable today
- ~3× search quality gap vs non-private neural retrieval
- Corpus privacy not addressed — server holds plaintext document index; only *query* is hidden

---

### 2. Panther — Private ANN, Single-Server Setting
https://doi.org/10.1145/3719027.3765190
**Li, Huang, Zhang, Cheng, Liu, Tao | ACM 2025**

**Target:** Dense (embedding/ANN) — private approximate nearest-neighbor search in a single-server deployment with no non-colluding assumption.

**Approach:**
Private ANNS in the **single-server setting** via novel co-designs of **PIR, secret-sharing, garbled circuits, and homomorphic encryption**. Prior single-server private ANNS (Chen et al., USENIX Security 2020) suffered from high communication; prior efficient approaches (SANNS, SP 2022) required two non-colluding servers. Panther achieves competitive performance under the harder single-server assumption.

**Privacy/security model:**
Single-server computational PIR + MPC hybrid. Client query hidden from server. Server learns neither the query vector nor which documents were retrieved. No non-colluding server assumption — stronger than SANNS's 2-server model.

**Performance:**
- Corpus: **10 million points**, four public datasets (GIST, GISTM, and others)
- Latency: **18 seconds** per query
- Communication: **284 MB** per query
- vs Chen et al. (prior single-server SOTA): **7.8× faster**, **20× less communication**
- vs SANNS (two-server): competitive, with strictly stronger security assumption

**Tradeoffs:**
- 18s latency is still impractical for interactive RAG; suited for batch or offline retrieval
- Single-server assumption is stronger but heavier — 2-server (p²RAG, SANNS) remains faster
- 284 MB per query is high for cloud deployments; network becomes the bottleneck on slow links
- ANN approximation means top-k is not exact — recall depends on index configuration

---

### 3. PrivANN — Practical Private ANN via TEE + ORAM
https://doi.org/10.1109/trustcom66490.2025.00140
**Chen, Cui, Liu, Sun, Lai | IEEE TrustCom 2025**

**Target:** Dense (embedding/ANN) — fully oblivious vector similarity search in large-scale databases.

**Approach:**
Fully oblivious ANN search using **Trusted Execution Environments (TEEs)** with a read-optimized **Oblivious RAM (ORAM)** protocol to defend against memory access side-channel leakage. Three core contributions: (1) ORAM protocol for oblivious index traversal, (2) shuffling mechanism decoupling expensive offline preparation from fast online queries, (3) oblivious Top-k selection algorithm. Formally proven security guarantees.

**Privacy/security model:**
TEE (hardware trust) + ORAM (cryptographic access-pattern hiding). Defends against an honest-but-curious adversary with full memory read access to the untrusted device. Server learns neither query content nor which vectors are retrieved. Formal security proof provided.

**Performance:**
- Throughput: **2.4× over state-of-the-art FHE-based systems** (IEEE TrustCom 2025; no arXiv preprint)
- Client-side communication: **kilobytes** per query
- Absolute query latency: not available from public sources (IEEE-gated paper)

**Implications:**
- KB-scale comm vs GB for PIR-based approaches — practical for cloud deployment without bandwidth bottleneck
- No polynomial approximation degradation (unlike FHE): search quality is not compromised by approximation error
- Offline/online separation: expensive ORAM shuffling is done once offline; online query path is fast

**Tradeoffs:**
- TEE hardware dependency (Intel SGX/TDX or equivalent)
- ORAM offline preparation cost; must be re-run if corpus changes significantly
- Hardware side-channels (Spectre, Foreshadow) partially mitigated by firmware but not eliminated
- 2.4× throughput gain over FHE; still below plaintext TEE-only (no ORAM) throughput

---

### 4. RemoteRAG — (n,ε)-DistanceDP for Private RAG Query
https://arxiv.org/abs/2412.12775
**Cheng, Zhang, Wang, Yuan, Yao | ACL Findings 2025 | 6 citations**

**Target:** Dense (embedding/ANN) — private top-k document retrieval in cloud RAG services; requires client-side embedding model.

**Approach:**
Client embeds query locally, adds calibrated Gamma-distribution noise satisfying **(n,ε)-DistanceDP**, and sends the perturbed embedding to the cloud for retrieval. A range-limiting theorem (Theorem 1) guarantees that the true top-k documents are contained in a small candidate set retrieved by the perturbed query. Partially Homomorphic Encryption (PHE) + optional Oblivious Transfer (OT) protect the fine-grained retrieval within that candidate set.

**Privacy/security model:**
(n,ε)-DistanceDP: server cannot distinguish the true query from any query within distance n/ε in embedding space. PHE hides the exact query vector during candidate re-ranking. OT hides which k documents are selected from k' candidates when corpus embeddings are sensitive.

**Performance:**
- End-to-end latency: **0.67 seconds** at 10⁵ documents
- Client–server communication: **46.66 KB** (vs 1.43 GB for non-optimized full-PHE baseline)
- Speedup vs non-optimized: **~15,000×** latency, **~30,000×** bandwidth
- 100% recall@k across ε ∈ {0.03, 0.05, 0.07, 0.1} and corpus sizes 10⁴–10⁶
- Vec2Text attack SacreBLEU: drops from ~50 (no noise) to ~10 (ε = 0.2)

**Tradeoffs:**
- Requires locally-accessible embedding model — incompatible with proprietary cloud embedding APIs
- PHE supports only addition → cannot use Lp or Jaccard distance metrics
- DP bound is on the query embedding, not raw text; if the embedding itself leaks text, formal bound overstates protection
- Formal DP guarantee does not hide access patterns (which cluster of documents is retrieved)

---

## B. Full RAG System Privacy (Hybrid Retrieval)

### 5. RAGtime-PIANO — FHE-Based Private Hybrid RAG
*(local corpus; no arXiv preprint found)*

**Target:** Hybrid (dense + sparse) — FHE-protected embedding similarity search combined with keyword matching; full end-to-end RAG pipeline.

**Approach:**
End-to-end private RAG combining **dense and sparse retrieval** under **Fully Homomorphic Encryption**. Two-stage architecture to amortize FHE cost:
- *Query-independent pre-processing*: offline parallel phase computes encrypted cluster centroids and index structures
- *Stage 1*: Cluster-level top-N search via FHE — narrows search space from full corpus to N candidate clusters
- *Stage 2*: Local encrypted top-k search within identified clusters — fine-grained ANN within each cluster
- *Stage 3*: Augment & generate — retrieved context combined with query for LLM generation

**Privacy/security model:**
FHE (CKKS/BFV). Both the user query and the vector database contents remain encrypted on the cloud server at all times. No hardware trust required. Server processes only ciphertexts. Security sections cover formal security definition (§3.1) and threat/adversary models (§3.2).

**Performance:**
- Benchmarks reported: NDCG and Precision on IR datasets (§5.3); comparison with prior art (§5.4)
- Concrete absolute latency numbers not publicly available (paper not on arXiv; full PDF in local library)

**Implications:**
- Pre-query offline phase is designed to amortize per-query FHE overhead — but FHE cost still dominates at query time
- Likely seconds–minutes per query at research scale; no production deployment known

**Tradeoffs:**
- FHE overhead remains significant even with two-stage cluster reduction
- Offline pre-processing must be re-run if corpus changes
- Polynomial approximations of ranking functions introduce accuracy error that compounds across stages
- FHE-based hybrid retrieval is the most cryptographically ambitious system in this survey

---

### 6. prRAG + CAPRISE — Distance-Preserving Encrypted Vector RAG
https://arxiv.org/abs/2601.12331
**Ye et al. | 2026**

**Target:** Dense (embedding/ANN) — encrypted vector search for private RAG; both document corpus and query hidden from cloud.

**Approach:**
**CAPRISE** is a privacy-preserving framework for encrypted vector search that stores embeddings in a *distance-preserving encrypted* form in an untrusted cloud. **prRAG** builds a full RAG pipeline on CAPRISE:
- *Phase 1 (Upload)*: Client embeds documents locally, encrypts document content with AES, encrypts embeddings with CAPRISE, uploads both to untrusted cloud
- *Phase 2 (Retrieve)*: Client sends encrypted query; CAPRISE enables nearest-neighbor search over encrypted embeddings; results returned
- *Phase 3 (Augment & Generate)*: Retrieved ciphertexts decrypted locally; combined with query for LLM generation
An extended service model uses **Oblivious Transfer (OT)** for direct remote retrieval to hide which specific documents are selected.

**Privacy/security model:**
CAPRISE provides encrypted ANN search (functional encryption or inner-product preserving encryption variant). Server holds only ciphertexts of embeddings and document content. Client retains decryption keys. OT variant also hides access patterns.

**Performance:**
- Hardware: NVIDIA A100 GPU; dataset: MS MARCO; embedding model: gtr-t5-base
- CAPRISE encryption throughput: **2,339 vectors/second** at 768-dim
- Encryption overhead: **15 ms per 128 queries**
- Vec2Text attack resistance (attacker reconstruction quality — lower = better privacy): BLEU 83.0 → **12.4**; token-precision 0.947 → **0.482**; token-recall 0.950 → **0.498**
- Retrieval expansion required to guarantee true top-k at privacy radius r=0.033: k=5 → k'=258; k=20 → k'=928
- End-to-end query latency: not reported

**Implications:**
- 9× faster than homomorphic encryption baselines — encryption is not a bottleneck at indexing time
- <19% overhead over embedding generation — CAPRISE adds minimal cost to the upload pipeline
- At r=0.033, server returns 52× more candidates than needed; the true top-k is recovered by client-side re-ranking, but the expansion size itself is visible to the server
- Distance-preserving encryption preserves nearest-neighbor ordering; quality loss comes from the DP expansion layer, not from the encryption itself

**Tradeoffs:**
- Distance-preserving encryption leaks some information about relative embedding distances — an adversary can infer document clusters and relative topic similarity even without decryption
- Requires client-side embedding — incompatible with proprietary embedding APIs (OpenAI, Cohere)
- AES document encryption adds key management complexity; key rotation across corpus updates is non-trivial

---

### 7. PIR-RAG — Classical PIR Integrated into RAG
https://arxiv.org/abs/2509.21325
**Wang et al. | 2025**

**Target:** Dense (embedding/ANN) — classical PIR adapted to dense passage retrieval; benchmarked on MS MARCO.

**Approach:**
Systematic integration of classical single-server and multi-server **PIR protocols into the RAG retrieval pipeline**. Covers the full privacy gap analysis of standard RAG (§2.1), a survey of PIR advances applicable to retrieval (§2.2), and private search architectures (§2.3). Evaluates end-to-end query time vs search quality tradeoffs on MS MARCO passage retrieval (NDCG, Precision, Hit@10, MRR@10).

**Privacy/security model:**
PIR-based: server learns neither which document is requested nor (in oblivious variants) the access pattern across queries. Covers both query-privacy and access-pattern-hiding threat models. Distinguishes PIR (hides which record) from OIR (also hides query intent).

**Performance (MS MARCO, 5,000-document corpus):**
- Query latency — PIR-RAG: **16.84s**; Graph-PIR: **12.99s**; Tiptoe-style: **23.82s**
- Corpus setup — Graph-PIR: ~**20s**; PIR-RAG and Tiptoe-style: faster (not quantified)
- Uplink per query: **2.4 – 24 KB** (PIR-RAG); similar for others
- Downlink per query: ~**475 MB** (PIR-RAG); Graph-PIR / Tiptoe-style: "few hundred KB"
- NDCG@10: PIR-RAG **0.799**; Graph-PIR **0.901**; Tiptoe-style **0.513**
- Precision@10: PIR-RAG **0.710**

**Implications:**
- Graph-PIR has both best latency and best retrieval quality among the three; PIR-RAG trades some quality for simpler architecture; Tiptoe-style is slowest with weakest quality
- PIR-RAG's 475 MB downlink is ~1,000× worse than Graph-PIR's few-hundred-KB — single-server PIR must return O(N) data to hide which record was fetched; graph-based approaches only traverse a subgraph
- All three approaches (13–24s at 5K docs) are too slow for interactive use at current scale

**Tradeoffs:**
- Single-server PIR communication scales as O(√N)–O(N) with corpus size
- Two-server PIR avoids communication blowup but requires non-colluding infrastructure
- Characterizes quality–privacy tradeoff: tighter privacy → worse retrieval quality

---

### 8. p²RAG — Secret Sharing, Arbitrary Top-k
https://arxiv.org/abs/2603.14778
**Ming, Wang, Yang, Wang, Jia | arXiv 2026**

**Target:** Dense (embedding/ANN) — private top-k retrieval from a secret-shared vector corpus; supports arbitrary k at query time, suited for long-context LLMs.

**Approach:**
Two-server **secret sharing (SS)** RAG supporting **arbitrary top-k retrieval** without fixing k at deployment time. Prior systems using sorting required fixed k at index build time; p²RAG uses an **interactive bisection method** to determine the top-k set dynamically. Two semi-honest non-colluding servers hold SS shares of the document corpus and user query embedding. Includes restrictions and verification mechanisms to defend against malicious users and formally bound corpus leakage.

**Privacy/security model:**
Semi-honest 2-server non-colluding secret sharing. Database contents and query hidden from each individual server. Bounds on leakage of the database formally established. Malicious user resistance via verification layer.

**Performance (BEIR trec-covid, 171K docs, 1024-dim embeddings):**
- **3–300× faster than PRAG** for k = 16–1024
- User↔server communication: **2.168 MB** (N=2¹⁷, k'=16); **2.156 MB** (N=2¹⁷, k'=128); **16.86 MB** (N=2²⁰, k'=16)
- Intra-server communication: **35.65 MB** (N=2¹⁷, k'=16); **335.5 MB** (N=2²⁰, k'=16)
- Round trips: **14 RTTs** (N=2¹⁷, k'=16); **17 RTTs** (N=2²⁰, k'=16)
- Recall: **1.0** for all tested k' = 16–1024
- Relevance scores: >1.0 for k'≤300

**Implications:**
- Speedup grows with k — bisection scales better than sorting for large retrieval sets; the larger the k, the more p²RAG's approach dominates
- Perfect recall guarantees the bisection always finds the true top-k; no approximation sacrifice for privacy
- 14–17 RTTs per query means latency is dominated by network round-trip time; unsuitable for high-latency WAN without batching

**Tradeoffs:**
- Requires two non-colluding servers — real operational burden (independent cloud deployments)
- Interactive bisection requires multiple round-trips between client and servers per query
- Non-colluding assumption: if both servers are controlled by the same adversary, security fails
- Does not address query embedding privacy if the embedding model itself is cloud-hosted

---

## C. Graph-Based and Sparse Retrieval Privacy

### 9. GraSS — Graph-Based Similarity Search on Encrypted Query
*(local corpus; Kim et al.)*

**Target:** Dense (embedding/ANN) — private approximate nearest-neighbor search using a graph-based index (HNSW-style) where the query is encrypted.

**Approach:**
Graph-based ANN search (similar to HNSW) adapted for **encrypted queries**. Uses a graph-structured index over document embeddings to enable efficient similarity search while hiding the query vector from the server. The graph traversal is integrated with a PIR-style protocol to prevent the server from learning which nodes were visited.

**Privacy/security model:**
Encrypted query — server does not see the plaintext query vector. Access pattern during graph traversal is partially hidden via PIR-style masking. Server holds plaintext document index; corpus contents are not hidden.

**Performance (from PIR-RAG comparison, MS MARCO, 5,000 documents):**
- Corpus setup time: ~**20 seconds**
- Query latency: **12.99 seconds**
- Downlink per query: **few hundred kilobytes**
- NDCG@10: **0.901**
- Precision@10: **0.850**

**Implications:**
- Best latency and best retrieval quality of the three systems in PIR-RAG's comparison (vs PIR-RAG: 16.84s / 0.799; Tiptoe-style: 23.82s / 0.513)
- Few-hundred-KB downlink is ~1,000× more efficient than PIR-RAG's 475 MB — graph traversal only fetches nodes along the search path, not the full corpus response

**Tradeoffs:**
- Graph traversal leaks partial access pattern even with PIR masking — depth of traversal reveals approximate query region
- HNSW approximation means recall < 100% even without privacy constraints
- Downlink communication is low (KBs) — advantage over single-server PIR approaches
- 13s latency is impractical for interactive use; suited for batch or background retrieval

---

### 10. ZKIFV — Zero-Knowledge Inverted File Indexing
*(local corpus; no arXiv preprint found)*

**Target:** Sparse (keyword/inverted index) — verifiable correctness of keyword search results; does not hide the query, only proves result integrity.

**Approach:**
Zero-knowledge proof system over **inverted file indexes** for verifiable sparse search. Proves that the server returned the correct and complete set of keyword-matched documents from a committed corpus, without revealing which internal index entries the server accessed to generate the proof.

**Privacy/security model:**
Verifiability model (not query privacy): client receives a ZK proof that results are correct and complete w.r.t. a committed index snapshot. Server cannot return false negatives or tamper with results without detection. Query terms are known to the server (no query hiding).

**Performance:**
- Not available from public sources (paper not found on arXiv; full PDF in local library)

**Tradeoffs:**
- Addresses *verifiability* rather than query privacy — orthogonal to PIR
- ZK proof generation over large inverted indexes is expensive
- Commitment scheme requires the index to be immutable between proof windows

---

## D. Verifiable Vector Search

### 11. V3DB — Audit-on-Demand Verifiable Vector Search
https://arxiv.org/abs/2603.03065
**Qiu et al. | 2026**

**Target:** Dense (embedding/ANN) — verifiable correctness of ANN/vector search results against a committed corpus snapshot; audit-on-demand rather than per-query.

**Approach:**
ZK proof system for **verifiable ANN search over committed corpus snapshots**. The server commits to a database snapshot; on audit request, generates a ZK proof that the top-k results returned were correct and complete w.r.t. that commitment. "Audit-on-demand" design: proofs generated lazily on request rather than per-query, amortizing proof cost.

**Privacy/security model:**
Verifiability model: client can verify result correctness against a committed database state. Does not hide which document was retrieved (not PIR). Addresses server-side tampering, selective omission, and false negatives.

**Performance (SIFT1M and GIST1M benchmarks):**
- ZK proof generation: **up to 22× faster** than circuit-only baseline
- Peak memory during proving: **up to 40% lower** than circuit-only baseline
- Verification time: **millisecond-level** per proof
- ZK-friendly index construction vs standard FAISS:
  - SIFT1M (D=128, high-acc): 95s → **207s**
  - GIST1M (D=960, high-acc): 341s → **4,542s**
- Recall with ZK-friendly preprocessing vs standard:
  - SIFT1M Recall@1: 0.503 → 0.504; Recall@100: 0.953 → 0.957
  - GIST1M Recall@1: 0.190 → 0.188; Recall@100: 0.531 → 0.523

**Implications:**
- 2.2× index overhead on SIFT1M (D=128) is acceptable; 13.3× on GIST1M (D=960) is prohibitive — ZK preprocessing cost scales sharply with embedding dimension
- Recall impact is negligible on low-dim vectors; small but present on high-dim (GIST1M ~1.5% Recall@1 drop)
- Millisecond verification means proof checking is essentially free for the client

**Tradeoffs:**
- Verifiability ≠ privacy: server still observes query in plaintext
- ANN is non-deterministic (approximate); ZK proofs for approximate search are technically harder than for exact search — correctness notion must be carefully defined
- Corpus must be immutable between audits; rolling updates require re-commitment

---

## Comparison Matrix

All entries assume a semi-honest (honest-but-curious) server unless noted.

| System | Type | Scheme | Query hidden? | Access pattern hidden? | Corpus hidden? | Verifiable? | Latency | Comm |
|---|---|---|---|---|---|---|---|---|
| **Tiptoe** | Dense | LHE / LWE PIR | ✓ | ✓ | ✗ | ✗ | 2.7s / 360M pages | 56.9 MiB |
| **Panther** | Dense | PIR + SS + GC + HE | ✓ | ✓ | ✗ | ✗ | 18s / 10M points | 284 MB |
| **PrivANN** | Dense | TEE + ORAM | ✓ | ✓ | ✗ | ✗ | 2.4× over FHE (abs. n/a) | KB |
| **RemoteRAG** | Dense | (n,ε)-DistanceDP + PHE | ✓ (DP) | ✗ | ✗ | ✗ | 0.67s / 100K docs | 46.66 KB |
| **RAGtime-PIANO** | Hybrid | FHE (CKKS+BFV) | ✓ | ✓ | ✓ | ✗ | n/a (paper not public) | — |
| **prRAG / CAPRISE** | Dense | Dist-preserving enc + OT | ✓ | Partial (OT) | ✓ | ✗ | enc: 15ms/128 queries | enc: low |
| **PIR-RAG** | Dense | PIR | ✓ | ✓ | ✗ | ✗ | 16.84s / 5K docs | ↑2.4–24 KB ↓475 MB |
| **p²RAG** | Dense | 2-server SS | ✓ | ✓ | ✓ | Partial | 3–300× vs PRAG | 2.2–17 MB (N=131K–1M) |
| **GraSS** | Dense | Graph + PIR | ✓ | Partial | ✗ | ✗ | 12.99s / 5K docs | few hundred KB |
| **ZKIFV** | Sparse | ZK proofs | ✗ | ✗ | ✗ | ✓ | n/a (paper not public) | — |
| **V3DB** | Dense | ZK proofs (multiset) | ✗ | ✗ | ✗ | ✓ | verify: ms; prove: 22× speedup | — |

---

## Key Observations

**1. Dense retrieval is the better-studied problem.**
Tiptoe, PrivANN, Panther, p²RAG, and RemoteRAG all target private ANN/dense search. The cryptographic approaches (Tiptoe/LHE, Panther/PIR) provide the strongest guarantees. TEE + ORAM (PrivANN) is the current practical frontier: KB-scale communication, 2.4× over FHE, no polynomial approximation degradation.

**2. Sparse / keyword retrieval privacy is significantly understudied.**
Private BM25/TF-IDF at scale remains an open problem. Only ZKIFV addresses it (verifiability only); no system achieves practical private *keyword* retrieval at web scale. GraSS uses a graph-based index but targets dense vector search, not sparse keyword search. The structural leakage from inverted index access patterns (term frequency, co-occurrence) is harder to hide than ANN access patterns.

**3. Hybrid RAG with full cryptographic privacy is essentially unsolved.**
RAGtime-PIANO is the only paper attempting FHE-based hybrid dense+sparse retrieval. The two-stage cluster approach is architecturally sound but FHE overhead for hybrid workloads likely puts per-query latency in the seconds-to-minutes range. No production-ready system exists.

**4. The non-colluding two-server model dominates practical designs.**
p²RAG, SANNS, and others use this model — it is 10–100× more efficient than single-server PIR and avoids FHE, at the cost of requiring two independently operated servers. It is the go-to design for systems that need to ship.

**5. Verifiability and privacy are orthogonal and rarely combined.**
V3DB and ZKIFV address verifiability (no tampering); Tiptoe, PrivANN, and PIR-RAG address privacy (no query leakage). No system in this survey delivers both at scale. This is an open research gap.

**6. Communication overhead is the critical engineering constraint.**
Single-server PIR requires O(N) server-side work for databases of size N; naive approaches need GB-scale communication. Tiptoe's 56.9 MiB for 360M pages is a landmark result; PrivANN's KB-scale communication via TEE + ORAM is the best current result. For RAG specifically, where corpora are typically 10⁴–10⁶ documents (not billions), single-server PIR may become practical with current hardware.

**7. No system protects all three channels simultaneously.**
Query content, access pattern, and corpus contents — protecting all three requires either FHE (too slow) or TEE + ORAM (hardware trust). The practical spectrum is: DP (cheapest, weakest) → TEE-only (cheap, hardware trust) → TEE+ORAM (moderate, hardware trust + crypto) → PIR (no hardware, expensive) → FHE (strongest, very expensive).
