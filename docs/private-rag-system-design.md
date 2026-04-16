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
| Retrieval quality | DP path: 100% recall demonstrated. CAPRISE: distance-preserving, no quality loss from encryption. p²RAG: recall 1.0. |
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

**Second priority:** Add TEE-hosted generation (Approach 3, Option A or B) as an optional feature for thin clients. This extends the market to mobile/browser users without changing the retrieval architecture.

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
| **Embedding** | private-embedding-research.md; NEXUS, SHAFT, PermLLM, OSNIP, GELO, SGT, DP-Forward, RemoteRAG | Local embedding solves it. MPC: 0.88s GPU (NEXUS) to 200s (PUMA). TEE: 4-8%. Obfuscation: 0.96ms (OSNIP) |
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
