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

### 2. Painter / Fanther — Private ANN, Single-Server Setting
*(local corpus)*

**Approach:**
Private Approximate Nearest Neighbor Search (ANNS) in the **single-server setting** via a novel **shared-cloud framework**. Protocol P_privApprox uses a coarse IVF-style clustering index combined with a PIR layer. A comparison table with SANNS (two-server) covers data/computation and point retrieval capabilities.

**Privacy/security model:**
Single-server computational PIR. Client query hidden from server. Server learns neither the query vector nor the retrieved document index. Stronger assumption than two-server (no non-colluding requirement) but heavier computation.

**Performance:**
- Benchmarked on GIST and GISTM datasets across N_list and T_proc configurations
- Capability comparison with SANNS on data vs computation axes
- Concrete latency/communication figures not available in surveyed content

**Tradeoffs:**
- Single-server assumption avoids non-colluding deployment complexity but raises per-query cost
- ANN approximation means top-k is not exact — some false negatives vs exact PIR

---

### 3. PrivANN — Practical Private ANN via TEE + ORAM
https://doi.org/10.1109/trustcom66490.2025.00140
**Chen, Cui, Liu, Sun, Lai | IEEE TrustCom 2025**

**Approach:**
Fully oblivious ANN search using **Trusted Execution Environments (TEEs)** with a read-optimized **Oblivious RAM (ORAM)** protocol to defend against memory access side-channel leakage. Three core contributions: (1) ORAM protocol for oblivious index traversal, (2) shuffling mechanism decoupling expensive offline preparation from fast online queries, (3) oblivious Top-k selection algorithm. Formally proven security guarantees.

**Privacy/security model:**
TEE (hardware trust) + ORAM (cryptographic access-pattern hiding). Defends against an honest-but-curious adversary with full memory read access to the untrusted device. Server learns neither query content nor which vectors are retrieved. Formal security proof provided.

**Performance:**
- Throughput: **2.4× over state-of-the-art FHE-based systems**
- Client-side communication: **kilobytes** (vs gigabytes for PIR-based approaches)
- Search quality: superior to FHE-based systems — no polynomial approximation degradation
- Offline/online separation: expensive shuffling is offline; online queries are fast

**Tradeoffs:**
- TEE hardware dependency (Intel SGX/TDX or equivalent)
- ORAM offline preparation cost; must be re-run if corpus changes significantly
- Hardware side-channels (Spectre, Foreshadow) partially mitigated by firmware but not eliminated
- 2.4× throughput gain over FHE; still below plaintext TEE-only (no ORAM) throughput

---

### 4. RemoteRAG — (n,ε)-DistanceDP for Private RAG Query
https://arxiv.org/abs/2412.12775
**Cheng, Zhang, Wang, Yuan, Yao | ACL Findings 2025 | 6 citations**

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
*(local corpus, doc 8412e27d)*

**Approach:**
End-to-end private RAG combining **dense and sparse retrieval** under **Fully Homomorphic Encryption**. Two-stage architecture to amortize FHE cost:
- *Query-independent pre-processing*: offline parallel phase computes encrypted cluster centroids and index structures
- *Stage 1*: Cluster-level top-N search via FHE — narrows search space from full corpus to N candidate clusters
- *Stage 2*: Local encrypted top-k search within identified clusters — fine-grained ANN within each cluster
- *Stage 3*: Augment & generate — retrieved context combined with query for LLM generation

**Privacy/security model:**
FHE (CKKS/BFV). Both the user query and the vector database contents remain encrypted on the cloud server at all times. No hardware trust required. Server processes only ciphertexts. Security sections cover formal security definition (§3.1) and threat/adversary models (§3.2).

**Performance:**
- Dedicated accuracy/latency benchmark (§5.3) and comparison with prior art (§5.4)
- Hybrid dense+sparse retrieval — both embedding similarity and keyword matching handled privately
- NDCG and Precision benchmarks on IR datasets; pre-query phase amortizes per-query FHE overhead
- Concrete absolute latency numbers not available from surveyed content

**Tradeoffs:**
- FHE overhead remains significant even with two-stage cluster reduction
- Offline pre-processing must be re-run if corpus changes
- Polynomial approximations of ranking functions introduce accuracy error that compounds across stages
- FHE-based hybrid retrieval is the most cryptographically ambitious system in this survey — likely research-stage latency (seconds–minutes per query)

---

### 6. prRAG + CAPRISE — Distance-Preserving Encrypted Vector RAG
*(local corpus, doc e5a7cebf)*

**Approach:**
**CAPRISE** is a privacy-preserving framework for encrypted vector search that stores embeddings in a *distance-preserving encrypted* form in an untrusted cloud. **prRAG** builds a full RAG pipeline on CAPRISE:
- *Phase 1 (Upload)*: Client embeds documents locally, encrypts document content with AES, encrypts embeddings with CAPRISE, uploads both to untrusted cloud
- *Phase 2 (Retrieve)*: Client sends encrypted query; CAPRISE enables nearest-neighbor search over encrypted embeddings; results returned
- *Phase 3 (Augment & Generate)*: Retrieved ciphertexts decrypted locally; combined with query for LLM generation
An extended service model uses **Oblivious Transfer (OT)** for direct remote retrieval to hide which specific documents are selected.

**Privacy/security model:**
CAPRISE provides encrypted ANN search (functional encryption or inner-product preserving encryption variant). Server holds only ciphertexts of embeddings and document content. Client retains decryption keys. OT variant also hides access patterns.

**Performance:**
- Detailed performance data not available from surveyed content

**Tradeoffs:**
- Distance-preserving encryption leaks some information about relative embedding distances — an adversary can infer document clusters and relative topic similarity even without decryption
- Requires client-side embedding — incompatible with proprietary embedding APIs (OpenAI, Cohere)
- AES document encryption adds key management complexity; key rotation across corpus updates is non-trivial

---

### 7. PIR-RAG — Classical PIR Integrated into RAG
*(local corpus, doc 09517a6b)*

**Approach:**
Systematic integration of classical single-server and multi-server **PIR protocols into the RAG retrieval pipeline**. Covers the full privacy gap analysis of standard RAG (§2.1), a survey of PIR advances applicable to retrieval (§2.2), and private search architectures (§2.3). Evaluates end-to-end query time vs search quality tradeoffs on MS MARCO passage retrieval (NDCG, Precision, Hit@10, MRR@10).

**Privacy/security model:**
PIR-based: server learns neither which document is requested nor (in oblivious variants) the access pattern across queries. Covers both query-privacy and access-pattern-hiding threat models. Distinguishes PIR (hides which record) from OIR (also hides query intent).

**Performance:**
- Benchmarked on MS MARCO passage retrieval: NDCG, Precision, Hit@10, MRR@10
- End-to-end query time for retrieval phase measured and plotted
- Concrete numbers not available from surveyed content

**Tradeoffs:**
- Single-server PIR communication scales as O(√N)–O(N) with corpus size
- Two-server PIR avoids communication blowup but requires non-colluding infrastructure
- Characterizes quality–privacy tradeoff: tighter privacy → worse retrieval quality

---

### 8. p²RAG — Secret Sharing, Arbitrary Top-k
https://arxiv.org/abs/2603.14778
**Ming, Wang, Yang, Wang, Jia | arXiv 2026**

**Approach:**
Two-server **secret sharing (SS)** RAG supporting **arbitrary top-k retrieval** without fixing k at deployment time. Prior systems using sorting required fixed k at index build time; p²RAG uses an **interactive bisection method** to determine the top-k set dynamically. Two semi-honest non-colluding servers hold SS shares of the document corpus and user query embedding. Includes restrictions and verification mechanisms to defend against malicious users and formally bound corpus leakage.

**Privacy/security model:**
Semi-honest 2-server non-colluding secret sharing. Database contents and query hidden from each individual server. Bounds on leakage of the database formally established. Malicious user resistance via verification layer.

**Performance:**
- **3–300× faster than PRAG** (prior state-of-the-art) for k = 16–1024
- Speedup increases with k — bisection scales better than sorting-based approaches
- Enables modern long-context LLMs that benefit from large retrieval sets (k = 64–1024)

**Tradeoffs:**
- Requires two non-colluding servers — real operational burden (independent cloud deployments)
- Interactive bisection requires multiple round-trips between client and servers per query
- Non-colluding assumption: if both servers are controlled by the same adversary, security fails
- Does not address query embedding privacy if the embedding model itself is cloud-hosted

---

## C. Sparse / Keyword Retrieval Privacy

### 9. Graph-PIR — Private Keyword RAG (PACMANN-Inspired)
*(local corpus)*

**Approach:**
Private keyword/sparse retrieval for RAG inspired by **PACMANN**. Combines graph-based document indexing (HNSW-style) with a PIR layer for access-pattern hiding during sparse/keyword-based retrieval. Evaluated on search quality and performance in comparison tables.

**Privacy/security model:**
PIR-based access-pattern hiding for keyword retrieval. Query terms hidden from server. Server learns neither which keyword was searched nor which documents matched.

**Performance:**
- Comparison with PACMANN baseline on search quality and performance metrics
- Detailed numbers not available from surveyed content

**Tradeoffs:**
- Graph-based PIR is more efficient than naive inverted-index PIR but still requires O(depth × branching) PIR calls per query
- HNSW approximation compounds with PIR approximation — imperfect recall
- Keyword privacy is harder than vector privacy: inverted index leaks term frequency statistics even with PIR

---

### 10. ZKIFV — Zero-Knowledge Inverted File Indexing
*(local corpus)*

**Approach:**
Zero-knowledge proof system over **inverted file indexes** for verifiable sparse search. Proves that the server returned the correct and complete set of keyword-matched documents from a committed corpus, without revealing which internal index entries the server accessed to generate the proof.

**Privacy/security model:**
Verifiability model (not query privacy): client receives a ZK proof that results are correct and complete w.r.t. a committed index snapshot. Server cannot return false negatives or tamper with results without detection. Query terms are known to the server (no query hiding).

**Performance:**
- Not available from surveyed content

**Tradeoffs:**
- Addresses *verifiability* rather than query privacy — orthogonal to PIR
- ZK proof generation over large inverted indexes is expensive
- Commitment scheme requires the index to be immutable between proof windows

---

## D. Verifiable Vector Search

### 11. V3DB — Audit-on-Demand Verifiable Vector Search
*(local corpus)*

**Approach:**
ZK proof system for **verifiable ANN search over committed corpus snapshots**. The server commits to a database snapshot; on audit request, generates a ZK proof that the top-k results returned were correct and complete w.r.t. that commitment. "Audit-on-demand" design: proofs generated lazily on request rather than per-query, amortizing proof cost.

**Privacy/security model:**
Verifiability model: client can verify result correctness against a committed database state. Does not hide which document was retrieved (not PIR). Addresses server-side tampering, selective omission, and false negatives.

**Performance:**
- Audit-on-demand design amortizes proof cost over query batches
- Concrete numbers not available from surveyed content

**Tradeoffs:**
- Verifiability ≠ privacy: server still observes query in plaintext
- ANN is non-deterministic (approximate); ZK proofs for approximate search are technically harder than for exact search — correctness notion must be carefully defined
- Corpus must be immutable between audits; rolling updates require re-commitment

---

## Comparison Matrix

All entries assume a semi-honest (honest-but-curious) server unless noted.

| System | Type | Scheme | Query hidden? | Access pattern hidden? | Corpus hidden? | Verifiable? | Latency | Comm |
|---|---|---|---|---|---|---|---|---|
| **Tiptoe** | Dense | LHE / LWE PIR | ✓ | ✓ | ✗ | ✗ | 2.7s | 56.9 MiB |
| **Painter/Fanther** | Dense | Single-server PIR | ✓ | ✓ | ✗ | ✗ | — | — |
| **PrivANN** | Dense | TEE + ORAM | ✓ | ✓ | ✗ | ✗ | 2.4× FHE | KB |
| **RemoteRAG** | Dense | (n,ε)-DistanceDP + PHE | ✓ (DP) | ✗ | ✗ | ✗ | 0.67s | 46.66 KB |
| **RAGtime-PIANO** | Hybrid dense+sparse | FHE | ✓ | ✓ | ✓ | ✗ | — | — |
| **prRAG / CAPRISE** | Dense | Dist-preserving enc + OT | ✓ | Partial (OT variant) | ✓ | ✗ | — | — |
| **PIR-RAG** | Dense | PIR | ✓ | ✓ | ✗ | ✗ | — | — |
| **p²RAG** | Dense | 2-server SS | ✓ | ✓ | ✓ | Partial | 3–300× vs PRAG | — |
| **Graph-PIR** | Sparse | PIR + graph index | ✓ | ✓ | ✗ | ✗ | — | — |
| **ZKIFV** | Sparse | ZK proofs | ✗ | ✗ | ✗ | ✓ | — | — |
| **V3DB** | Dense | ZK proofs | ✗ | ✗ | ✗ | ✓ | audit-on-demand | — |

---

## Key Observations

**1. Dense retrieval is the better-studied problem.**
Tiptoe, PrivANN, Painter, p²RAG, and RemoteRAG all target private ANN/dense search. The cryptographic approaches (Tiptoe/LHE, Painter/PIR) provide the strongest guarantees. TEE + ORAM (PrivANN) is the current practical frontier: KB-scale communication, 2.4× over FHE, no polynomial approximation degradation.

**2. Sparse / keyword retrieval privacy is significantly understudied.**
Private BM25/TF-IDF at scale remains an open problem. Graph-PIR and ZKIFV address it but concrete performance data is absent. No system achieves practical private sparse retrieval at web scale. The structural leakage from inverted index access patterns (term frequency, co-occurrence) is harder to hide than ANN access patterns.

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
