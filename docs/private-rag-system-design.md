# Private RAG for Outsourced Confidential Knowledge Bases

> Research date: 2026-04-16. Sources: edgequake knowledge graph (25 indexed papers), local research docs, OpenAlex, web search.

---

## Goal Definition

**System goal:** Let clients outsource confidential / proprietary knowledge to a remote RAG service, query that data privately, and retrieve relevant context without exposing raw data or raw query contents to the service operator.

**Target security guarantee levels (three tiers):**

1. **Retrieval-only private RAG** — server never sees plaintext documents, queries, or which records match. Client generates answers locally. No hosted generation. Strongest coherent model.
2. **TEE-based full-service RAG** — all processing (embedding, retrieval, generation) inside hardware-isolated enclaves. Server operator cannot read memory. Full feature set including hosted generation. Trust assumption: hardware vendor.
3. **Hybrid crypto-retrieval + TEE-generation** — cryptographic privacy for storage and retrieval; TEE for generation only. Balances crypto strength on the tractable problem (vector search) with practical hardware trust on the hard problem (LLM inference).

**Assumptions:**
- Corpus size: 10^4 – 10^6 documents (enterprise knowledge base, not web scale)
- Client has moderate local compute (can run an embedding model; optionally a 7B–32B LLM)
- Interactive latency target: <5s for retrieval, <30s total with generation
- Semi-honest (honest-but-curious) server threat model as baseline
- Multi-tenant isolation required

**Success looks like:** A deployed system where a regulated enterprise (healthcare, finance, legal, government) can outsource a confidential document corpus to a cloud RAG service and query it, with a formally characterizable privacy guarantee at every stage, without degrading retrieval quality below 90% of plaintext baseline.

---

## Problem Space

### Why this needs to be solved

Standard RAG sends plaintext at every stage. The server sees:
1. **Raw documents** during ingestion and chunking
2. **Raw embeddings** that are invertible back to source text (Vec2Text: 92% exact recovery from ada-002; EDNN: near-100% model-agnostic)
3. **Plaintext queries** or query embeddings (equally invertible)
4. **Access patterns** — which documents are retrieved per query, revealing user intent over time
5. **Full assembled prompt** at generation time — query + retrieved context sent to the LLM

### Concrete threats

| Threat | Impact | Evidence |
|---|---|---|
| Embedding inversion | Cloud vector DB operator reconstructs document text from stored embeddings | Vec2Text (EMNLP 2023, 54 cites): 92% token recovery. EDNN (2024): near-100% without model weights |
| Access pattern leakage | Adversary infers query topics, popular documents, user interests from retrieval logs | Kellaris et al. (CCS 2016, 267 cites): formal proof that access pattern + volume leak is unavoidable without ORAM |
| Query content leakage | Server learns exactly what each user is asking about | Direct exposure in sparse search; embedding inversion in dense search |
| Generation-time exposure | Cloud LLM provider sees full assembled prompt including all retrieved confidential context | All non-TEE, non-MPC retrieval systems (RemoteRAG, prRAG, p²RAG) silently break privacy at this stage |
| Regulatory violation | GDPR, HIPAA, EU AI Act violations from processing personal/medical/legal data in cloud | EDPS TechSonar: embeddings derived from personal data are still personal data under GDPR. Cumulative GDPR fines: ~EUR 4.5B by 2024 |

### Deployment blockers without solving this

- Healthcare: cannot outsource patient record RAG (HIPAA BAA requirements)
- Financial services: cannot use cloud RAG over proprietary research (SOC 2, data residency)
- Legal: cannot outsource privileged document search (attorney-client privilege)
- Government: classified or sensitive data cannot leave controlled environments
- Cross-institutional: hospitals/banks cannot pool data for shared RAG (data clean room problem)

50% of enterprises already use RAG with data privacy as the stated motivation (Cloudera 2024). The market is projected $1.7B (2024) to $18–20B+ by 2030 (MarketsandMarkets).

---

## Design / Tradeoff Space

### Major design dimensions

**1. Trust model**
- Cryptographic only (FHE/MPC/PIR): no hardware trust; formal mathematical guarantees; 10–1000x overhead
- Hardware trust (TEE/CVM): Intel TDX, AMD SEV-SNP, NVIDIA H100 CC; 4–10% overhead; trust the silicon vendor
- Hybrid (TEE + crypto): GELO, ObfuscaTune, PrivANN; TEE for hard parts, crypto for tractable parts
- Statistical (DP): formal epsilon-delta bounds; fast; probabilistic, not absolute

**2. What is private vs. what is trusted**

| Component | Can be made private | Current best approach | Cost |
|---|---|---|---|
| Document content at rest | Yes | AES + CAPRISE/DCPE encrypted embeddings | ~0 overhead (DCPE) to 15ms/128q (CAPRISE) |
| Embedding generation | Yes (local) or partial (MPC/obfuscation) | Local model (FastEmbed, Ollama) | Zero API leakage; requires client GPU/CPU |
| Query content | Yes | DP perturbation (RemoteRAG) or FHE/PIR | 0.67s (DP) to 18s (PIR) |
| Access patterns | Yes | ORAM (PrivANN), PIR (Panther) | KB comm (ORAM+TEE) to 284MB (PIR) |
| Retrieval results | Yes | PIR document fetch, OT (RemoteRAG) | Included in retrieval overhead |
| LLM generation | Partial | TEE (4-8% overhead) or MPC (3s/tok best case) | TEE is practical; MPC is research-stage for >6B models |

**3. Retrieval type: dense vs sparse vs hybrid**
- Dense: PIR/FHE/ORAM solutions exist (Tiptoe, Panther, PrivANN, RemoteRAG)
- Sparse/BM25: **no practical private scheme exists** — SSE literature predates neural RAG; frequency-inference attacks unsolved
- Hybrid: **open research gap** — only RAGtime-PIANO attempts it, no shipped solution

**4. Client-side vs server-side processing**
- More client-side processing = stronger privacy, weaker product (client needs hardware)
- More server-side processing = weaker privacy, better UX
- Critical boundary: embedding and generation are the two most expensive steps

**5. Hosted generation vs client-side generation**
- Hosted generation caps the entire system's privacy at the generation step's trust model
- If generation requires TEE (weakest practical option for large LLMs), then overengineering retrieval with heavy FHE/PIR adds cost without improving end-to-end security
- Client-side generation eliminates the weakest link entirely, at the cost of requiring client hardware

**6. Performance wall**

| System | Best latency | Scale | Security model |
|---|---|---|---|
| DCPE (IronCore) | ~0ms overhead | Production | Leaks distance ordering |
| RemoteRAG | 0.67s | 10^5 docs | DP query only, no access pattern |
| PrivANN (TEE+ORAM) | 2.4x over FHE | Research | TEE + ORAM |
| Tiptoe | 2.7s | 360M pages | Crypto-only, 45 servers |
| GraSS | 13s | 5K docs | FHE graph search |
| PIR-RAG | 16.8s | 5K docs | PIR |
| Panther | 18s | 10M points | Single-server PIR+HE+GC+SS |

The gap between DCPE (~0ms, weaker) and full crypto (13–18s, stronger) is enormous. Nothing in the 2024–2026 literature closes it. TEE+ORAM (PrivANN) is the practical frontier: KB-scale communication, no quality degradation.

**7. Updateability / incremental indexing**
- ORAM requires periodic reshuffling when corpus changes
- PIR index structures must be rebuilt on corpus update
- DCPE/CAPRISE support incremental insertion (encrypt new vectors, append)
- FHE cluster structures (RAGtime-PIANO) require offline re-preprocessing

**8. Verifiability**
- V3DB: ZK proofs for ANN search correctness; 22x faster than circuit baseline; ms-level verification
- ZKIFV: ZK for inverted index; no performance numbers public
- No system combines privacy + verifiability at scale — **open gap**

---

## Differentiation Factor

### Where we can meaningfully innovate

**1. Coherent end-to-end security model with client-side generation**

Most existing work focuses on one pipeline stage (private retrieval OR private generation). The retrieval-to-generation handoff is the most common privacy gap — RemoteRAG, prRAG, p²RAG all do private retrieval then send plaintext to a cloud LLM. No shipped system delivers a coherent security model across all stages.

*Opportunity:* Design an end-to-end system where the privacy model is consistent and the weakest link is explicitly chosen, not accidentally introduced.

**2. Private retrieval at enterprise scale (10^4–10^6 docs)**

PIR/FHE research targets web scale (10^7–10^9) where the overhead is prohibitive. At enterprise scale (10^4–10^6 docs), the same primitives may become interactive. PIR-RAG at 5K docs already achieves 16.8s; p²RAG at 171K docs achieves perfect recall with 2.2MB communication. This is the sweet spot where crypto approaches transition from research to deployable.

*Opportunity:* Optimize PIR/SS protocols specifically for the 10^4–10^6 enterprise range rather than web scale.

**3. Practical private hybrid retrieval**

No system does private BM25 + dense retrieval. This is a genuine open gap with high practical value — hybrid retrieval is the production standard, and the DP-RAG line (Private-RAG, LPRAG) is the only pragmatic path.

*Opportunity:* First system to offer private hybrid retrieval, even with weaker (DP or TEE-based) guarantees.

**4. Client-side generation as a product feature, not a limitation**

With 7B–32B local models now matching or exceeding cloud API quality for many tasks (Llama 3.3 70B, Qwen3-32B, DeepSeek-R1-70B), client-side generation is no longer a sacrifice. Position it as a feature: "your data never leaves your control, and the model is better than what you'd get from a cloud API anyway."

*Opportunity:* Productize the retrieval-only private RAG model with excellent client-side generation as the default.

---

## Approaches

### Approach 1: Retrieval-Only Private RAG with Client-Side Generation

**Summary**

The server is a private retrieval service only. It stores encrypted document embeddings and encrypted document content. It performs private similarity search and returns encrypted results. The client handles embedding, decryption, and answer generation locally. No plaintext ever exists on the server. No hosted generation. The security model is consistent from ingestion through answer generation.

**Approaches used**

- **CAPRISE** (Ye et al. 2026) — distance-preserving encryption for vector storage
- **RemoteRAG** (Cheng et al. 2025) — (n,epsilon)-DistanceDP for query perturbation
- **p²RAG** (Ming et al. 2026) — 2-server secret sharing for access-pattern-hiding retrieval (alternative to DP)
- **IronCore Cloaked AI** — DCPE for zero-overhead encrypted vector search (weaker alternative)
- **Local embedding models** — Jina v2, nomic-embed-text, gte-Qwen2 via Ollama/FastEmbed
- **Local LLM** — Llama 3.3 70B, Qwen3-32B, DeepSeek-R1 via llama.cpp/Ollama

**Specification**

1. **Ingestion:** Client chunks documents locally (sentence/paragraph splitting, metadata extraction). All preprocessing is client-side — no privacy impact from chunking.

2. **Chunking / preprocessing:** Client-side. Standard chunking strategies (fixed-size with overlap, semantic, context-aware). Metadata extraction (headings, page numbers, entity tags) done locally. No server involvement.

3. **Embedding:** Client runs embedding model locally (e.g., Jina v2 768-dim, gte-Qwen2-7B). Zero API leakage. Embedding quality matches or exceeds cloud APIs on MTEB benchmarks.

4. **Storage:** Client encrypts embeddings with CAPRISE (distance-preserving encryption; 2,339 vec/s at 768-dim; Vec2Text BLEU drops from 83 to 12.4). Document content encrypted with AES-256. Both uploaded to server. Server stores only ciphertexts.
   - *Alternative (weaker, simpler):* IronCore DCPE — zero server overhead, drop-in to existing vector DBs (Qdrant, Pinecone), but leaks distance ordering.
   - *Alternative (stronger, slower):* p²RAG secret-shared storage across 2 non-colluding servers.

5. **Query submission:** Client embeds query locally. Applies DistanceDP noise (epsilon configurable; RemoteRAG: 100% recall at epsilon in {0.03, 0.05, 0.07, 0.1}). Sends perturbed encrypted query to server.
   - *Alternative:* For access-pattern hiding, use p²RAG's secret-sharing protocol instead of DP — each server sees only a share of the query.

6. **Retrieval:** Server performs similarity search over encrypted embeddings. Returns top-k' encrypted candidates (k' > k to account for DP expansion; at r=0.033: k=5 -> k'=258). Client decrypts, re-ranks to true top-k locally.
   - With CAPRISE: server computes on distance-preserving ciphertexts directly.
   - With p²RAG: two servers compute partial results on shares; client recombines. 2.168 MB comm, 14 RTTs, recall 1.0 at 171K docs.

7. **Reranking / filtering:** Client-side. Client has decrypted top-k' candidates; applies any reranking (cross-encoder, metadata filter, MMR diversity) locally. No server involvement.

8. **Answer generation:** Client assembles prompt (query + top-k retrieved chunks) and runs local LLM. Server never sees the assembled prompt or the response. Privacy model is end-to-end: no plaintext exists outside the client at any stage.

9. **Provenance / verification:** [OPEN GAP] No deployed system combines private retrieval with verifiable correctness. V3DB (ZK proofs for ANN) could be adapted to prove the server returned correct top-k from the committed encrypted index, but this has not been demonstrated with CAPRISE or p²RAG storage.

**Tradeoffs**

| Dimension | Assessment |
|---|---|
| Privacy guarantees | **Strongest coherent model.** No plaintext on server at any stage. DP or SS for query; DPE or SS for storage; local generation. |
| Trust assumptions | Minimal. Crypto only (no hardware trust) if using p²RAG. CAPRISE + DP: client must trust the DP noise is sufficient. p²RAG: non-colluding server assumption. |
| Performance / latency | Retrieval: 0.67s (DP path) or ~seconds (p²RAG at 171K docs). Generation: depends on client hardware (7B model: ~20 tok/s on M-series Mac; 70B: ~5 tok/s with quantization). |
| Retrieval quality | DP noise affects *which* chunks are returned, not their content — decrypted chunks are exact, no post-processing needed. Noise shifts the query vector; the server finds neighbors of `Q_noisy`, not `Q`. CAPRISE is distance-preserving, so encryption itself causes no quality loss. p²RAG: recall 1.0. **ε tradeoff:** smaller ε (stronger privacy) = larger noise = higher recall loss. RemoteRAG shows 100% recall at ε∈{0.03–0.1} on their benchmark, but high-dimensional spaces are robust to small perturbations. Main risk: rare/specific queries near cluster boundaries — noise can push the query into the wrong cluster, dropping relevant documents from top-k entirely. |
| Cost | Client needs hardware for embedding + LLM. Server cost is standard vector DB hosting on encrypted data. |
| Implementation complexity | Medium. Client SDK (embed + encrypt + generate); server is a modified vector DB accepting encrypted queries. |
| Deployment feasibility | **High.** CAPRISE and RemoteRAG are implementable with current libraries. Local LLMs are production-ready (Ollama, llama.cpp). IronCore DCPE is already a commercial product. |
| Product completeness | Partial. No hosted generation. Client must have capable hardware. Not suitable for thin clients (mobile, browser-only). |
| Major open risks | Client hardware requirement limits market. DP expansion (k' >> k) leaks approximate query region to server. CAPRISE leaks inter-embedding distance structure (cluster topology visible). |

**When this approach is the right choice**

- Client has local compute (developer workstation, on-prem server, M-series Mac)
- Regulatory requirement prohibits any plaintext data in cloud (HIPAA, classified)
- Highest privacy bar required — no hardware trust assumptions acceptable
- Corpus is moderate size (10^4–10^6 docs) where PIR/SS overhead is tolerable
- Generation quality from local 7B–32B models is sufficient for the use case

---

### Approach 2: Full-Service TEE RAG

**Summary**

Everything runs inside Trusted Execution Environments (Intel TDX VMs, AMD SEV-SNP, NVIDIA H100 Confidential Compute). Documents are encrypted at rest and in transit; decrypted only inside hardware enclaves. The server provides full RAG service including embedding, retrieval, and generation. The trust assumption is the TEE hardware vendor. Performance is near-plaintext (4–10% overhead). Full product feature set.

**Approaches used**

- **Petridish** (Li et al. 2024) — Confidential VM architecture for LLM serving; Secure Partitioned Decoding for multi-tenant isolation
- **Opal** (Kaviani et al. 2026) — ORAM-backed private memory inside TEE for access-pattern hiding; knowledge graph retrieval inside enclave
- **Privatemode.ai** (Edgeless Systems) — commercial confidential LLM inference on H100 TEE; Capgemini partnership
- **Opaque Systems** — confidential AI platform with policy enforcement; RAG + agentic AI
- **Fortanix Armet AI** — turnkey GenAI platform on SGX+TDX+Hopper; public preview April 2025
- **TEE overhead benchmarks** (2025): <10% throughput, <20% latency for CPU TEE; 4–8% for GPU TEE

**Specification**

1. **Ingestion:** Client encrypts documents with a client-controlled key and uploads to the TEE-hosted service. Documents are decrypted only inside the enclave. The enclave performs chunking, metadata extraction, and preprocessing on plaintext inside the hardware-isolated boundary.

2. **Chunking / preprocessing:** Inside TEE. Standard chunking, NER, metadata extraction all run on decrypted plaintext within the enclave. The cloud operator cannot observe this processing. Remote attestation verifies the enclave code is correct.

3. **Embedding:** Inside TEE. Embedding model (e.g., Jina v2, nomic-embed) runs inside the enclave. Resulting embeddings are stored encrypted at rest within the TEE's encrypted memory region. Cloud operator sees only ciphertexts.

4. **Storage:** Embeddings and chunks stored in enclave-managed encrypted storage. Option A: standard vector DB inside CVM (simple, no access-pattern hiding). Option B: ORAM-backed storage (Opal-style) for access-pattern hiding — adds 2.4x overhead but prevents the server from learning which records are accessed.

5. **Query submission:** Client sends encrypted query to TEE endpoint. TLS terminates inside the enclave. Query is decrypted only inside the enclave. Cloud operator cannot intercept.

6. **Retrieval:** Vector similarity search runs on plaintext embeddings inside the enclave. If ORAM (Opal): access pattern is hidden even from the enclave's host OS. Standard ANN algorithms (HNSW, IVF) work without modification — no quality degradation.

7. **Reranking / filtering:** Inside TEE. Cross-encoder reranking, metadata filtering, hybrid BM25+dense scoring all run on plaintext inside the enclave. No privacy limitation on retrieval sophistication — **this is the only approach that supports private hybrid retrieval today.**

8. **Answer generation:** LLM runs inside CVM (Petridish architecture) or on H100 Confidential Compute GPU. Full frontier-class models (Llama 3.3 70B) at 4–8% overhead. Generated response encrypted before leaving the enclave.

9. **Provenance / verification:** Remote attestation proves the enclave runs the expected code. Client can verify the software stack via attestation report before sending data. Does not cryptographically prove result correctness (unlike ZK approaches), but proves the correct code was executed.

**Tradeoffs**

| Dimension | Assessment |
|---|---|
| Privacy guarantees | **Hardware-level.** All data encrypted at rest and in transit; decrypted only inside enclave. Cloud operator cannot read memory. Weaker than cryptographic (FHE/MPC) — relies on hardware correctness. |
| Trust assumptions | Must trust TEE hardware vendor (Intel, AMD, NVIDIA). Historical TEE vulnerabilities: Spectre (2018), Foreshadow (2018), PLATYPUS (2020). Mitigated by firmware updates but not eliminated. |
| Performance / latency | **Near-plaintext.** CPU TEE: <10% throughput overhead. GPU TEE (H100 CC): 4–8%. Full LLaMA-70B inference: same quality, same speed. |
| Retrieval quality | **No degradation.** Standard algorithms run on plaintext inside enclave. HNSW, IVF, BM25, hybrid, cross-encoder reranking — all supported. |
| Cost | TEE-capable hardware (TDX VMs, H100 CC) costs 10–30% premium over standard instances. Opal ORAM adds ~$1.46M/yr at 1M users. |
| Implementation complexity | **Low to medium.** CVM deployment on Azure/GCP/Scaleway is productized. Opal adds ORAM complexity. Remote attestation setup is non-trivial but documented. |
| Deployment feasibility | **High today.** Privatemode.ai, Opaque Systems, Fortanix all ship TEE-based RAG products. Azure Confidential Computing and GCP Confidential VMs are GA. |
| Product completeness | **Full.** Hosted generation, hybrid retrieval, reranking, metadata filtering — everything works. Thin clients (mobile, browser) fully supported. |
| Major open risks | TEE hardware vulnerability = total compromise. Side-channel attacks on SGX are a real historical concern. No cryptographic fallback. Vendor lock-in to Intel/AMD/NVIDIA. |

**When this approach is the right choice**

- Full-service product required (thin clients, mobile, browser)
- Hardware trust assumption is acceptable (most enterprises accept this for cloud computing already)
- Need hosted generation with frontier-class models (70B+)
- Hybrid retrieval (BM25 + dense) required with no quality compromise
- Near-term deployment (can ship on existing commercial TEE platforms today)

---

### Approach 3: Hybrid Crypto-Retrieval + TEE-Generation

**Summary**

Cryptographic privacy for data storage and retrieval (the tractable problem). TEE-based privacy for LLM generation (the hard problem). The security model is explicit about where crypto ends and hardware trust begins. Stronger than pure TEE for storage/retrieval; more practical than pure crypto for generation. The trust boundary is: "the server cannot learn your data or your query from storage/retrieval; the server's TEE-hosted LLM sees the assembled prompt but the cloud operator cannot."

**Approaches used**

- **CAPRISE + DP** (Ye et al. 2026, Cheng et al. 2025) — distance-preserving encrypted storage + DP query perturbation for retrieval
- **p²RAG** (Ming et al. 2026) — 2-server secret sharing for retrieval as a stronger alternative
- **Petridish / Privatemode** — CVM-hosted LLM for generation
- **GELO** (Belikova et al. 2026) — obfuscation + TEE hybrid for generation; 20–30% overhead, 76% of compute offloaded to untrusted GPU
- **OSNIP** (Cao et al. 2026) — null-space projection of assembled prompt before sending to cloud LLM; 0.96ms overhead, no quality loss, works with existing APIs
- **Portcullis** (AAAI 2025) — TEE-attested PII anonymization gateway before cloud LLM

**Specification**

1. **Ingestion:** Client chunks and embeds locally (same as Approach 1). Encrypts embeddings with CAPRISE and document content with AES. Uploads to server.

2. **Chunking / preprocessing:** Client-side (same as Approach 1).

3. **Embedding:** Client-side local model (same as Approach 1).

4. **Storage:** CAPRISE-encrypted embeddings on server. AES-encrypted document chunks. Server holds only ciphertexts. No TEE needed for storage — crypto guarantees sufficient.

5. **Query submission:** Client embeds query locally, applies DistanceDP noise, encrypts, sends to retrieval server.

6. **Retrieval:** Server performs similarity search over CAPRISE-encrypted embeddings. Returns top-k' encrypted candidates. Client decrypts, re-ranks to true top-k. Retrieval is cryptographically private — no TEE dependency.
   - *Alternative:* p²RAG for access-pattern hiding (stronger, requires 2 servers).

7. **Reranking / filtering:** Client-side re-ranking on decrypted candidates (same as Approach 1).

8. **Answer generation:** Client assembles prompt (query + top-k chunks) and sends to **TEE-hosted LLM** for generation. Three sub-options with different trust/performance tradeoffs:
   - **Option A: CVM-hosted LLM (Petridish/Privatemode).** Assembled prompt enters CVM; cloud operator cannot read memory. 4–8% overhead. Full model quality. Trust: hardware vendor.
   - **Option B: Obfuscation gateway (OSNIP).** Client applies null-space projection to the assembled prompt before sending to any cloud LLM (including OpenAI/Anthropic). 0.96ms overhead. No quality loss on Qwen3-32B. KNN attack success: 0.000. Trust: learned projection covers all sensitive directions (no formal guarantee).
   - **Option C: PII gateway (Portcullis).** TEE-attested NER anonymization strips PII before sending to cloud LLM; de-anonymizes response. 96x faster than Hide-and-Seek. Hides PII only, not query semantics. Trust: NER model + TEE.

9. **Provenance / verification:** Remote attestation for the generation TEE. CAPRISE storage could be combined with V3DB ZK proofs for retrieval correctness (not yet demonstrated). The retrieval layer has crypto guarantees; the generation layer has hardware guarantees. The boundary is explicit.

**Tradeoffs**

| Dimension | Assessment |
|---|---|
| Privacy guarantees | **Layered.** Crypto for storage/retrieval (no hardware trust needed). TEE or obfuscation for generation (hardware trust or computational privacy). Explicit boundary between the two. |
| Trust assumptions | Storage/retrieval: crypto-only (CAPRISE) or non-colluding servers (p²RAG). Generation: TEE hardware trust (Option A), learned projection (Option B), or NER correctness (Option C). |
| Performance / latency | Retrieval: 0.67s (DP) + encryption overhead. Generation: 4–8% (CVM), 0.96ms (OSNIP), or near-zero (Portcullis). Total: <5s retrieval + standard LLM generation latency. |
| Retrieval quality | Same as Approach 1 — no degradation from CAPRISE/DP. |
| Cost | Client needs embedding hardware. Server needs TEE for generation (Option A) or none (Options B/C). Lower client hardware requirement than Approach 1 (no local LLM needed). |
| Implementation complexity | **Medium-high.** Two distinct privacy layers (crypto + TEE/obfuscation) that must be correctly composed. Key management for CAPRISE + attestation for TEE. |
| Deployment feasibility | **Medium.** Retrieval layer is implementable today (CAPRISE + RemoteRAG). Generation via CVM is available (Privatemode). OSNIP is research-stage. Portcullis is deployable. |
| Product completeness | **Full with hosted generation.** Supports thin clients. Client still needs embedding capability (lightweight: FastEmbed on CPU). |
| Major open risks | Composition risk: the boundary between crypto and TEE layers must be carefully implemented to avoid leakage at the handoff. OSNIP's non-formal guarantee may not cover novel attack vectors. CAPRISE leaks inter-embedding distance structure. |

**When this approach is the right choice**

- Want stronger-than-TEE privacy for stored data (crypto guarantees for the corpus at rest)
- Accept hardware trust only for generation (the part where crypto is impractical)
- Need hosted generation but with explicit, layered trust model
- Building a product where different customers have different trust requirements (offer crypto retrieval + choice of generation privacy level)
- Research-oriented: want to demonstrate a novel layered architecture

---

## Approach 4: Thin-Client Private RAG with Server-Side Private Embedding

**Security tier:** 3.5 — CAPRISE-based storage/retrieval; embedding outsourced to a trust-anchored TEE; generation via ObfuscaTune/GELO (deferred). The distinguishing characteristic: **no embedding capability required on the client.**

**Summary**

Approaches 1 and 3 require clients to run a local embedding model (150MB–1.5GB) and hold the CAPRISE key. This blocks thin clients (mobile, browser) and complicates multi-tenant SaaS deployment. Approach 4 routes all embedding through a lightweight **Embedding TEE** — a CVM (TDX/SEV-SNP) running the embedding model, verified by remote attestation before any data is sent. Three CAPRISE key management models are offered with different trust and forward-security tradeoffs.

---

**Approaches used**

- **Petridish / CVM** (Li et al. 2024, arXiv:2409.19134) — CVM architecture for hosting the embedding model; the Embedding TEE
- **CAPRISE** (Ye et al. 2026) — distance-preserving encryption; key model varies per storage option
- **GELO** (Belikova et al. 2026) — GELO-encoder split variant: client holds embedding lookup + obfuscation matrix A; TEE runs linear projections on obfuscated activations
- **RemoteRAG DistanceDP** (Cheng et al. 2025) — DP noise applied at query submission
- **ObfuscaTune** (Frikha et al. 2024) — generation layer: TEE holds input embedding layer + lm_head; GPU runs linear projections on obfuscated activations; input and output text stay inside TEE (deferred)
- **GELO + OSNIP** — alternative generation layer: GELO for activation protection, OSNIP for null-space projection before cloud LLM (deferred)

---

**Specification**

1. **Ingestion / Chunking:** Client-side. Client reads documents, applies chunking strategy, produces text chunks. Client AES-encrypts chunks with a per-tenant symmetric key (held only by client) and uploads to storage server. **No embedding model required on client.**

2. **Session setup / attestation:** Before sending any plaintext, client performs remote attestation of the Embedding TEE (Intel TDX DCAP or AMD SEV-SNP attestation report). Client verifies: (a) TEE runs the expected embedding model and version; (b) TEE runs the expected CAPRISE implementation; (c) CAPRISE key derivation matches the chosen key model (step 4). Client establishes RATLS channel to TEE. TLS 1.3 ECDHE ensures past sessions cannot be decrypted if the TLS certificate is later compromised.

3. **Embedding (TEE-side):** Two variants with the same privacy outcome and different TCB tradeoffs:

   **Primary — full embedding model in TEE:** Client sends plaintext text chunks over RATLS to Embedding TEE. TEE runs the full embedding model (e.g., Jina v2 768-dim, ~150MB; fits in TDX VM memory). TEE outputs the final embedding vector, applies CAPRISE encryption (key model per step 4), and sends ciphertext to storage server. Storage server receives only ciphertexts — never plaintext text or plaintext embeddings.

   **GELO-encoder variant (reduced TEE memory footprint):** Client holds the token embedding lookup table (vocab × D matrix, public model weights, ~90MB for 768-dim) and generates a per-session random invertible matrix A. Client tokenizes the chunk, runs the embedding lookup, and obfuscates: `H̃ = A · H`. Client sends `H̃` to TEE over RATLS. TEE holds `A⁻¹` (established at session setup), runs transformer linear projections on `H̃`, applies `A⁻¹` before each non-linear op (GELU, LayerNorm, Softmax), re-obfuscates after, and produces the final embedding. TEE applies CAPRISE encryption. This is GELO applied to an encoder model: the embedding lookup stays on client, shrinking enclave memory; TEE never sees raw token embeddings.

   *Why split-without-obfuscation is not viable:* sending unobfuscated intermediate activations to TEE does not protect privacy — early transformer layer activations can be inverted to recover input tokens via representation inversion attacks. The obfuscation matrix A is required.

4. **Storage — CAPRISE key management:**

   Three options with different trust models and forward-security properties:

   **Option 1 — TEE-held key (most performant):**
   TEE generates and seals the CAPRISE key inside the enclave. TEE CAPRISE-encrypts embeddings and sends ciphertexts to storage server. Minimum round trips; no key material on client.
   - *Forward security:* None — TEE compromise exposes the sealed CAPRISE key; all stored embeddings become decryptable.
   - *Trust required:* TEE trusted indefinitely.

   **Option 2 — Client-held key (best forward security):**
   TEE produces plaintext embedding, sends to client over RATLS (TLS 1.3 ECDHE, forward-secret). Client CAPRISE-encrypts with its own key and uploads to storage server.
   - *Forward security:* Full — future TEE compromise cannot recover the client's CAPRISE key (never stored in TEE). Past ingestion sessions remain private even after TEE is later broken.
   - *Cost:* One extra hop (TEE → client → storage). Client must be online during ingestion.
   - *Trust required:* TEE trusted only for the duration of each attested session.

   **Option 3 — Two-party KDF (HKDF; forward-secure against TEE-only compromise):**
   During the attested session, client transmits `user_x_sk` to TEE over the RATLS channel (TLS 1.3 ECDHE — recording traffic then later compromising the TLS certificate cannot decrypt past sessions). TEE holds `tee_user_x_sk` sealed to the enclave. TEE computes:
   ```
   caprise_key = HKDF-SHA256(user_x_sk ∥ tee_user_x_sk)
   ```
   TEE CAPRISE-encrypts embeddings, sends ciphertexts to storage server, then immediately discards `user_x_sk` (not persisted in enclave storage).
   - *Forward security:* Attacker who breaks the TEE seal recovers `tee_user_x_sk` but not `user_x_sk` — it was never persisted, and the TLS session that carried it is forward-secret. Recovering `caprise_key` requires compromising **both** the TEE seal **and** the client endpoint simultaneously.
   - *Retrieval:* Client re-sends `user_x_sk` over a fresh attested RATLS session. TEE re-derives the same `caprise_key` deterministically.
   - *Trust required:* TEE code must be auditable via attestation (must not log or persist `user_x_sk`). An actively malicious TEE exfiltrating `user_x_sk` during a live session defeats this — attestation binds code identity, not runtime behavior. Same caveat as all TEE-based schemes.
   - *Round trips:* Same as Option 1. Better security than Option 1 for the TEE-break-only scenario.

   | | Option 1 | Option 2 | Option 3 |
   |---|---|---|---|
   | TEE compromise alone | Breaks all storage | Safe | Safe |
   | TEE + client compromise | Breaks all | Breaks all | Breaks all |
   | Round trips (ingestion) | Minimum | +1 hop | Minimum |
   | Client online at ingestion | No | Yes | No |

5. **Query submission:** Client sends query text over RATLS to Embedding TEE. TEE embeds the query (same path as step 3). Applies DistanceDP noise (ε configurable). CAPRISE-encrypts noisy query embedding per the chosen key model. Sends encrypted query to storage server.

6. **Retrieval:** Storage server performs ANN similarity search over CAPRISE-encrypted embeddings. Returns top-k' CAPRISE-encrypted candidates + corresponding AES-encrypted document chunks. CAPRISE distance-preservation means ANN operates directly on ciphertexts — no server-side decryption required.

7. **Decryption / handoff:** Depends on storage key model:
   - **Option 1:** TEE CAPRISE-decrypts candidate embeddings and re-ranks. Client provisions AES key to TEE at session setup for chunk decryption, or retrieves AES-encrypted chunks directly from storage and decrypts locally.
   - **Option 2:** Client CAPRISE-decrypts retrieved embeddings and re-ranks locally. Client AES-decrypts document chunks directly from storage server.
   - **Option 3:** Client re-sends `user_x_sk` over fresh RATLS session; TEE re-derives `caprise_key`, CAPRISE-decrypts and re-ranks.

8. **DistanceDP — defense-in-depth:**
   Applied at query submission (step 5) only. Do not apply to ingestion — noise on stored embeddings permanently degrades all future retrieval with no compensating privacy benefit.

   Query noise: `q̃ = q + Lap(0, Δf/ε)` where `Δf = 2` for unit-normalized embeddings. Protects against access-pattern inference: an adversary observing which documents are retrieved per query cannot infer the true query direction even if the encrypted query vector is visible.

   | ε | Privacy strength | Recall impact (RemoteRAG benchmark) |
   |---|---|---|
   | 0.03–0.1 | Strong | ~0% degradation |
   | 1.0 | Moderate | ~15–20% drop at top-5 |
   | 5.0 | Weak | ~3–5% drop |

   *Interaction with CAPRISE:* Complementary. CAPRISE protects stored embedding values (prevents document content recovery from stored ciphertexts). DistanceDP masks query intent (prevents access-pattern inference from retrieval logs). Both should be applied together.

9. **Answer generation (research-deferred):** Private generation via ObfuscaTune or GELO+OSNIP. Core concern: cloud LLM returns plaintext — if the cloud operator can read LLM output, retrieved context leaks at generation time regardless of retrieval privacy.

   ObfuscaTune (Frikha et al. 2024) is the strongest viable option: TEE handles the input embedding layer and output lm_head (~5% of model parameters); GPU runs linear projections on obfuscated activations. Both input and output text remain inside TEE. RAG is explicitly deferred in the paper as future work. GELO protects intermediate activations only — plaintext output remains visible to cloud operator, requiring a full generation TEE (Petridish/CVM) for complete output privacy.

   **Marked research-deferred:** no production implementation of ObfuscaTune or GELO+OSNIP for RAG exists. For immediate deployment, generation falls back to local LLM (Approach 1 model) or CVM-hosted LLM (Approach 2 model).

---

**Tradeoffs**

| Dimension | Assessment |
|---|---|
| Privacy guarantees | **TEE-anchored crypto stack.** Embedding TEE: hardware trust for embedding and key management. Storage: CAPRISE crypto (same as Approaches 1/3). Generation: deferred (ObfuscaTune/GELO/TEE/local). Storage operator never sees plaintext text, embeddings, queries, or document content. |
| Trust assumptions | Embedding: hardware trust (TDX/SEV-SNP). CAPRISE key model determines ongoing trust scope: Option 1 requires indefinite TEE trust; Options 2/3 limit trust to session windows. Generation (deferred): hardware trust for output privacy. |
| Forward security | Option 1: none. Option 2: full (client key, TEE never holds it). Option 3: TEE-break-only safe; simultaneous TEE + client compromise collapses security. |
| Performance / latency | RATLS setup: ~10–50ms amortized per session. Embedding TEE: ~0% overhead (TDX VM is a full encrypted VM; no SGX EPC limit). Retrieval: 0.67s (DP path). Option 2 adds one extra round trip at ingestion. |
| Client requirements | **Lowest of all approaches.** Chunking only (primary path). No embedding model, no CAPRISE library, no local LLM. True thin-client: mobile SDK, browser WASM. GELO-encoder variant adds embedding lookup table (~90MB) and obfuscation model. |
| Retrieval quality | Same as Approaches 1/3: CAPRISE is distance-preserving (no encryption overhead on quality); DP recall 100% at ε∈{0.03–0.1} on benchmark. Decrypted chunks are exact plaintext. |
| Cost | Small CVM with TDX or SEV-SNP. 150MB embedding model fits in standard TDX VM memory. Fraction of the cost of a generation TEE. |
| Implementation complexity | **High.** RATLS attestation is the primary engineering challenge. Per-tenant key management inside TEE (Options 1/3) or client-side CAPRISE key lifecycle (Option 2). AES key handoff for chunk decryption must prevent TEE+storage-server collusion. |
| Deployment feasibility | **Medium.** CVM hosting available (Scaleway/Azure/GCP Confidential VMs). RATLS libraries available (Intel TDX DCAP, AMD SEV-SNP SDK). CAPRISE research-stage but implementable. Generation step deferred. |
| Product completeness | Retrieval and embedding: full thin-client support. Generation: requires local LLM or additional TEE infrastructure (deferred). |
| Major open risks | (1) Attestation correctness: incorrect RATLS implementation nullifies all TEE guarantees — the most common deployment pitfall. (2) Key model selection: wrong choice among Options 1/2/3 for the threat model leaves forward security gaps. (3) Generation output privacy: cloud LLM plaintext output requires ObfuscaTune TEE or on-premise generation for full privacy. (4) CAPRISE inter-embedding distance leakage (inherited from all CAPRISE-based approaches). |

**When this approach is the right choice**

- Target users are thin clients (mobile, browser) unable to run embedding models locally
- SaaS provider manages the embedding TEE on behalf of many tenants (single TEE, per-tenant CAPRISE keys)
- Wanting Approach 3 privacy guarantees without client embedding infrastructure
- Forward-secure key model required: use Option 2 (strongest) or Option 3 (performant)
- Willing to invest in a small embedding TEE; generation infrastructure deferred or client-side


## Cross-Approach Comparison

### Design factor summary

| Dimension | Approach 1 | Approach 2 | Approach 3 | Approach 4 |
|---|---|---|---|---|
| **Client requirement** | Embedding model + LLM | Embedding model | Embedding model | Chunking only |
| **Thin client support** | No | No | No | Yes |
| **Embedding trust** | None (fully local) | Full TEE | None (fully local) | TEE (hardware, TDX/SEV-SNP) |
| **Storage privacy** | CAPRISE (crypto) | AES + TEE memory | CAPRISE (crypto) | CAPRISE (crypto, key model Options 1/2/3) |
| **Retrieval privacy** | DistanceDP (formal) | TEE + ORAM | DistanceDP (formal) | DistanceDP (formal) |
| **Generation** | Local only | TEE-hosted | GELO / OSNIP / TEE | Deferred (ObfuscaTune / GELO / local) |
| **Hardware trust required** | None | Full pipeline | Generation only (small TEE) | Embedding TEE |
| **TEE enclave size** | — | Full model pipeline | Matrix + non-linear ops | Full embedding model (~150MB) primary; linear layers only (GELO variant) |
| **Crypto-only path** | Yes (the baseline) | No | No | 2PC sub-option (slow) |
| **Retrieval latency** | 0.67s | ~1s (TEE+ORAM) | 0.67s | 0.67s |
| **Generation latency** | Client hardware | 4–8% TEE overhead | 4–8% (CVM) / 0.96ms (OSNIP) | Deferred |
| **Forward security** | N/A (client key) | N/A | N/A | Option 1: none; Option 2: full; Option 3: TEE-break-only safe |
| **Implementation complexity** | Medium | Medium | Medium-high | High (RATLS + key mgmt) |
| **Strongest privacy stage** | Storage + retrieval | Full pipeline | Storage + retrieval | Storage + retrieval |
| **Weakest link** | Client endpoint | TEE hardware trust | Retrieval→generation handoff | Attestation correctness + generation output |
| **Market differentiation** | Coherent crypto-only RAG | TEE RAG (exists today) | Layered crypto+TEE | Thin-client SaaS, configurable forward security |

### Trust boundary progression

```
Approach 1: [Client: embed + encrypt + generate] → [Server: CAPRISE ciphertexts only]
                                                    ↑ No hardware trust anywhere

Approach 2: [Client: embed] → [Full TEE: retrieve + generate] → [Client: response]
                               ↑ Full pipeline hardware trust

Approach 3: [Client: embed + encrypt] → [Server: CAPRISE ciphertexts] → [Gen TEE: GELO/OSNIP]
                           ↑ Crypto for storage/retrieval    ↑ Small TEE for generation

Approach 4: [Client: chunk] → [Embed TEE: embed + CAPRISE-encrypt] → [Server: ciphertexts] → [Gen: deferred]
                               ↑ Hardware TEE (TDX/SEV-SNP)           ↑ Crypto only
            Key model: TEE-held (Opt 1) | client-held (Opt 2) | two-party KDF HKDF (Opt 3)
            GELO-encoder variant: [Client: lookup + A·H] → [TEE: linear layers + A⁻¹]
```

TEE scope progression (smallest → largest): 3 (GELO matrix, few KB) < 4/GELO-variant (linear layers only) < 4/primary (full embedding model, ~150MB) < 2 (full pipeline).

---

## Final Recommendation

**Pursue Approach 1 (Retrieval-Only Private RAG with Client-Side Generation) first.**

Rationale:

1. **Coherent security model.** It is the only approach where no plaintext exists on the server at any stage. No awkward weakest-link at generation. The privacy story is simple to explain and audit.

2. **Technical feasibility.** Every component exists and is implementable today:
   - Client-side embedding: Ollama/FastEmbed (production-ready)
   - Encrypted storage: CAPRISE (research but implementable; IronCore DCPE as commercial fallback)
   - Private retrieval: RemoteRAG DP (0.67s, 100% recall demonstrated) or p²RAG (perfect recall at 171K docs)
   - Local generation: Llama 3.3 70B / Qwen3-32B / DeepSeek-R1 via llama.cpp (production-ready, state-of-the-art quality)

3. **Near-term implementation.** Can build an MVP without TEE hardware procurement, without MPC infrastructure, without FHE expertise. The hardest engineering task is implementing CAPRISE correctly; the rest is integration.

4. **Differentiation.** No commercial product offers this end-to-end model. IronCore does encrypted storage only. Privatemode does TEE inference only. PrivateGPT does local-only. Nobody chains private encrypted retrieval with high-quality local generation as an outsourced service.

5. **Upgrade path.** Approach 1 naturally extends to Approach 3: add TEE-hosted generation as an optional feature for clients without local LLM capability. The retrieval infrastructure is identical.

**Second priority:** Add TEE-hosted generation (Approach 3, Option A or B) as an optional feature for clients with embedding capability but no local LLM. This extends the market without changing the retrieval architecture.

**Third priority (thin-client SaaS):** Approach 4 — move the embedding TEE server-side. Retrieval architecture stays identical; the new engineering investment is the RATLS attestation protocol and per-tenant CAPRISE key management inside the TEE. This unlocks mobile/browser clients and enables multi-tenant SaaS deployment.

**Do not pursue Approach 2 (full TEE) as the primary architecture.** While it's the easiest to deploy today (commercial products exist), it offers no technical differentiation — Privatemode, Opaque, and Fortanix already ship it. The value of our work is in the crypto layer.

**Explicit uncertainty:**
- [UNCLEAR] Whether CAPRISE's distance-structure leakage is acceptable for high-sensitivity corpora (medical records, classified docs). Need empirical evaluation of what an adversary can reconstruct from the encrypted distance ordering.
- [UNCLEAR] Whether p²RAG's 2-server non-colluding assumption is operationally achievable for most enterprise deployments (requires two independent cloud providers).
- [UNCLEAR] Whether local 7B–32B models are sufficient quality for all target use cases (legal reasoning, medical diagnosis may require frontier 70B+ models that need more client hardware than typical).

---

## Source Map

### By pipeline stage

| Stage | Primary sources | Key finding |
|---|---|---|
| **Ingestion** | No dedicated work | Must be client-side; no privacy-preserving outsourced chunking exists |
| **Embedding** | private-embedding-research.md; NEXUS, SHAFT, PermLLM, OSNIP, GELO, SGT, DP-Forward, RemoteRAG, Petridish | Local solves it for Approaches 1/3. TEE-anchored embedding (Approach 4): ~0% overhead for 150MB models in TDX. GELO-encoder variant reduces TEE footprint. 2PC alternative: 3–47s/doc. |
| **Storage** | fhe-encrypted-vector-db.md; CAPRISE, IronCore DCPE, Panther, RAGtime-PIANO, p²RAG, Compass, FRAG | DCPE deployed (IronCore). CAPRISE: 2339 vec/s. FHE: 18s at 10M (Panther). 18s gap between DCPE and full crypto |
| **Retrieval** | private-information-retrieval.md; PIR-RAG, Tiptoe, Panther, RemoteRAG, p²RAG, GraSS, PrivANN | RemoteRAG 0.67s DP. p²RAG perfect recall 171K docs. Panther 18s single-server. PrivANN TEE+ORAM KB comm |
| **Generation** | private-response-generation.md; PermLLM, PUMA, SIGMA, MERGE, BumbleBee, Petridish, GELO, OSNIP, SCX | MPC: 3s/tok (PermLLM 6B) to 200s/tok (PUMA 7B). TEE: 4-8%. GELO: 20-30%. OSNIP: 0.96ms |
| **Verification** | V3DB, ZKIFV, VLAH, ANNProof | V3DB: 22x faster ZK proofs. Privacy + verifiability combined: open gap |
| **End-to-end** | Opal, RAGtime-PIANO, prRAG+CAPRISE, SoK (Bodea 2026) | No shipped end-to-end private RAG. Opal closest (TEE+ORAM). RAGtime-PIANO (full FHE, research-only) |
| **Commercial** | IronCore, Privatemode, Opaque, Fortanix, CyborgDB, PrivateGPT | All TEE-based except IronCore (DCPE). No FHE/MPC product ships |

### Key papers by importance

| Paper | Year | Venue | Why it matters |
|---|---|---|---|
| Vec2Text (Morris et al.) | 2023 | EMNLP | Proves embeddings leak text. Motivates all private RAG work |
| Kellaris et al. | 2016 | CCS | Proves access patterns leak. Motivates ORAM/PIR |
| RemoteRAG (Cheng et al.) | 2025 | ACL Findings | Best practical private retrieval: 0.67s, 100% recall, DP |
| p²RAG (Ming et al.) | 2026 | arXiv | Best crypto retrieval: SS, arbitrary top-k, perfect recall, 2.2MB comm |
| CAPRISE / prRAG (Ye et al.) | 2026 | arXiv | Best encrypted vector storage: 2339 vec/s, Vec2Text BLEU 83->12 |
| Panther (Li et al.) | 2025 | CCS | Best single-server private ANN: 18s at 10M, 7.8x faster than prior |
| Opal (Kaviani et al.) | 2026 | arXiv | Most complete private RAG system: TEE+ORAM, knowledge graph |
| RAGtime-PIANO (Januszewicz et al.) | 2026 | ePrint | Only fully FHE RAG protocol; 40x faster, 323x less comm than PIR-RAG |
| GELO (Belikova et al.) | 2026 | arXiv | Best lightweight private generation: 20-30% overhead, BSS security |
| OSNIP (Cao et al.) | 2026 | arXiv | Best obfuscation for generation: 0.96ms, KNN attack 0.000 |
| PermLLM (Zheng et al.) | 2024 | arXiv | Fastest MPC LLM inference: 3s/tok for 6B model on WAN |
| Petridish (Li et al.) | 2024 | arXiv | CVM architecture for LLM serving; directly deployable |
| IronCore DCPE | 2023 | Commercial | Only deployed encrypted vector search product |
| Bodea et al. SoK | 2026 | arXiv | Systematization of privacy risks in RAG |
