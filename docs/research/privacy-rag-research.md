# Privacy-Enhanced RAG: Research & Market Landscape

> Researched 2026-04-14. Sources: OpenAlex (academic papers), SearXNG (projects/market data).

---

## Technical Directions

---

### 1. Private / Encrypted Vector DB Storage

**Key papers**

| Paper | Year | Cites | URL |
|---|---|---|---|
| SANNS: Scaling Up Secure Approximate k-NNS | 2019 | 12 | https://doi.org/10.48550/arxiv.1904.02033 |
| FRAG: Federated Vector DB for Collaborative Secure RAG | 2024 | 3 | https://doi.org/10.48550/arxiv.2410.13272 |
| Panther: Private ANNS, Single Server | 2025 | 0 | https://doi.org/10.1145/3719027.3765190 |
| PrivANN: Private ANN via TEE+ORAM | 2025 | 0 | https://doi.org/10.1109/trustcom66490.2025.00140 |
| RAGtime-PIANO: FHE+PIR RAG | 2026 | 0 | https://eprint.iacr.org/2026/231.pdf |
| p²RAG: Arbitrary Top-k with PIR | 2026 | 0 | https://arxiv.org/abs/2603.14778 |
| EDNN: Embedding inversion attack | 2024 | 2 | https://doi.org/10.18653/v1/2024.emnlp-main.126 |
| Eguard: Defending against embedding inversion | 2024 | 2 | https://doi.org/10.48550/arxiv.2411.05034 |
| STEER: Secure Transformed Embedding Retrieval | 2025 | 0 | https://doi.org/10.48550/arxiv.2507.18518 |

**Problem addressed**

Two distinct leakage surfaces:

1. **Embedding inversion** — raw vectors stored in any cloud DB can be inverted back to near-exact source text. Vec2Text (Morris et al. 2023) and EDNN (Lin et al. 2024) demonstrate near-100% token recovery from standard embeddings. Storing unencrypted embeddings in Pinecone/Weaviate/Qdrant Cloud leaks document content.
2. **Access pattern leakage** — even with encrypted data, which records get retrieved reveals query intent over time.

**State of approaches**

| Approach | What it hides | Overhead | Practicality |
|---|---|---|---|
| DCPE (IronCore Cloaked AI) | Vectors at rest; approximate distances computable | ~0 | Deployed product. Known leakage: distance ordering preserved — relative positions reconstructible by sophisticated attacker |
| FHE over vectors (FRAG, PIR-RAG, RAGtime-PIANO) | Query + which docs match | 10–100x; 100s MB comm | Low. 18s/query at 10M vectors even after 2025 improvements |
| Single-server PIR (Panther 2025) | Access patterns | 284 MB comm; 18s at 10M pts | Medium. 7.8x faster than prior work; still not interactive |
| TEE + ORAM (PrivANN 2025) | Query + access patterns from OS/hypervisor | 2.4x over FHE baselines; KB comm | Medium-high. Requires SGX hardware; side-channel exposure risk |
| Embedding obfuscation / STEER | Query text from provider | Near-zero | Medium. No crypto guarantees; weaker than above |
| Local self-hosted (PrivateGPT, llama.cpp) | Everything | None (hardware cost) | High. No cloud convenience |

**Products / projects**

- **IronCore Labs Cloaked AI** — commercial, DCPE, Nov 2023, Gartner Cool Vendor 2025, integrates with Qdrant/Pinecone. https://ironcorelabs.com/products/cloaked-ai
- **CyborgDB / Cyborg Inc.** — confidential vector DB, NVIDIA cuVS partnership, pre-revenue. https://www.cyborg.co
- **Javelin AI / Highflame** — HE for embeddings, Nov 2024 startup.
- **PrivateGPT** — open source, local-only, widely deployed. https://github.com/zylon-ai/private-gpt

**Importance / maturity**

**Underbuilt.** The embedding inversion attack surface is concrete and underappreciated in deployed RAG. Only one commercial product (Cloaked AI DCPE) addresses it, with a known residual leakage model. FHE/PIR solutions are academically rigorous but 18+ seconds per query at 10M vectors. TEE (PrivANN) is closest to practical but no product ships it. Full-privacy vector search at production scale does not exist as a shipped product.

---

### 2. Private Embedding Generation

**Key papers**

| Paper | Year | Venue | URL |
|---|---|---|---|
| Vec2Text: Text Embeddings Reveal Almost As Much As Text — Morris et al. | 2023 | EMNLP | https://arxiv.org/abs/2310.06816 |
| Multilingual embedding inversion — Chen et al. | 2024 | ACL | https://doi.org/10.18653/v1/2024.acl-long.422 |
| RemoteRAG: (ε,δ)-DistanceDP for cloud RAG | 2024/2025 | ACL Findings | https://arxiv.org/abs/2412.12775 |
| FedE4RAG: Federated embedding learning for local RAG | 2025 | arXiv | https://arxiv.org/abs/2504.19101 |
| SecFormer: 2PC secure inference for BERT/GPT | 2024 | ACL Findings | https://arxiv.org/abs/2401.00793 |
| Primer: FHE-based transformer inference | 2023 | DAC | https://arxiv.org/abs/2303.13679 |
| SHAFT: MPC-minimized transformer inference | 2025 | IACR ePrint | https://eprint.iacr.org/2025/2324 |
| Survey on Private Transformer Inference | 2024 | arXiv | https://arxiv.org/abs/2412.08145 |

**Problem addressed**

When using a cloud embedding API (OpenAI, Cohere, Jina):

1. Document/query content transmits in plaintext — provider can log, train on, or sell it.
2. Embeddings themselves are reversible — Vec2Text recovers near-exact original text including passwords and PII from ada-002/GTR.
3. Returned documents reveal query intent even if the query was perturbed.

Embeddings are not anonymized representations — they are high-fidelity proxies for source text.

**State of approaches**

- **Local model deployment** — run embedding model on own hardware (Ollama, FastEmbed, llama.cpp). Zero API leakage. State-of-the-art MTEB scores from local models (e5-large-v2, gte-Qwen2-7B) now match or exceed OpenAI text-embedding-3-large on many benchmarks. Most practical and widely adopted solution today.
- **DP query perturbation (RemoteRAG)** — add calibrated Laplace/Gaussian noise to embedding vector before sending. (ε,δ)-DistanceDP variant resists Vec2Text inversion while preserving retrieval quality. 0.67s / 46KB overhead at 10⁵ docs. Practical if local compute unavailable; 2025 ACL paper, no shipped library yet.
- **Federated learning (FedE4RAG)** — distributed clients train embedding models via knowledge distillation + HE aggregation. Addresses training-time privacy, not inference-time leakage.
- **Cryptographic secure inference (SecFormer, Primer, SHAFT)** — 2PC/FHE: client holds input, server holds model weights, neither learns the other's data. SecFormer achieves BERT-base inference in ~1–3 minutes. Still 10–1000x slower than plaintext. Not production-viable for interactive RAG today.

**Products / projects**

- **Ollama + nomic-embed-text / mxbai-embed-large** — most widely deployed practical solution; zero API leakage by design.
- **FastEmbed** (Qdrant) — lightweight local embedding library with ONNX models.
- **SecFormer** (research prototype) — https://github.com/jinglong696/SecFormer

No commercial product ships cryptographic private embedding inference.

**Importance / maturity**

**High importance, partially solved by local deployment.** Vec2Text (2023) is a watershed finding: all "Embeddings as a Service" integrations are retroactively a privacy risk. Local model deployment fully solves this. RemoteRAG's DistanceDP is the best known defense if cloud APIs must be used. Cryptographic secure inference is 2–5 years from practicality at embedding model scale.

---

### 3. Private Information Retrieval for RAG (Sparse / Dense / Hybrid)

**Key papers**

*Foundational PIR / SSE:*

| Paper | Year | Cites | URL |
|---|---|---|---|
| Chor, Kushilevitz et al.: Private Information Retrieval (PIR) | 1998 | 1,614 | https://doi.org/10.1145/293347.293350 |
| Wang et al.: Secure Ranked Keyword Search over Encrypted Cloud | 2010 | 736 | https://doi.org/10.1109/icdcs.2010.34 |
| Naveed et al.: Dynamic SSE via Blind Storage | 2014 | 312 | https://doi.org/10.1109/sp.2014.47 |
| Xu et al.: Hardening Database Padding for SSE | 2019 | 48 | https://doi.org/10.1109/infocom.2019.8737588 |
| Wu et al.: Survey on Secure Keyword Search | 2023 | 7 | https://doi.org/10.1145/3617824 |
| Chen et al.: MFSSE — Multi-Keyword Fuzzy Ranked SSE | 2024 | 14 | https://doi.org/10.1109/tcc.2024.3430237 |

*Dense / semantic RAG-specific:*

| Paper | Year | Cites | URL |
|---|---|---|---|
| Tiptoe: Private Web Search via embeddings (MIT/Berkeley, SOSP 2023) | 2023 | 23 | https://doi.org/10.1145/3600006.3613134 |
| Compass: Encrypted Semantic Search (OSDI 2025) | 2025 | 0 | https://eprint.iacr.org/2024/1255 |
| p²RAG: 2-server secret sharing, arbitrary top-k | 2026 | 0 | https://doi.org/10.48550/arxiv.2603.14778 |
| PIR-RAG: PIR applied to RAG pipeline | 2025 | 0 | https://arxiv.org/abs/2509.21325 |

*DP-based:*

| Paper | Year | Cites | URL |
|---|---|---|---|
| Private-RAG: Multi-Query DP-RAG (NeurIPS 2025) | 2025 | 0 | https://doi.org/10.48550/arxiv.2511.07637 |
| LPRAG: Local DP on entity perturbation | 2025 | 15 | https://doi.org/10.1016/j.ipm.2025.104150 |
| DP-SynRAG: DP synthetic corpus | 2025 | 0 | https://doi.org/10.48550/arxiv.2510.06719 |

**Problem addressed**

- **Sparse/BM25:** SSE hides keyword queries but leaks access patterns (which documents matched) and search patterns (whether two queries matched the same set). IDF/BM25 scoring is structurally hard to hide — ranking requires frequency statistics that are intrinsically revealing.
- **Dense/semantic:** Client query embedding reveals intent. Server learns both the query vector (inferrable text) and which documents were retrieved.
- **Hybrid:** No published scheme addresses private BM25+dense hybrid retrieval as a unified problem. **This is a genuine open research gap.**

**State of approaches**

**Sparse/BM25:** SSE literature is substantial (Wang 2010, 736 cites) but predates neural RAG. No dedicated "private BM25-for-RAG" paper exists. The only known implementation is `ebm25.rs` (https://github.com/slevental/ebm25.rs) — a hobby project with no formal security proof. Frequency-inference attacks on SSE (Islam et al. NDSS 2012, 496 cites) remain unsolved without prohibitive padding overhead (2–5x storage, Xu et al. 2019).

**Dense/semantic:** Active frontier. Key results:
- **Tiptoe** (SOSP 2023): crypto-only, 360M pages, 2.7s latency, 56.9 MiB comm, requires 45-server cluster, degrades quality (rank 7.7 vs. 2.3 non-private).
- **Panther** (CCS 2025): single-server PIR, 10M vectors in 18s, 284 MB comm, 7.8x faster than prior work.
- **PrivANN** (TrustCom 2025): TEE+ORAM, 2.4x faster than FHE, KB-scale comm.
- **Compass** (OSDI 2025): FHE+ORAM combined.

**Hybrid:** The DP-RAG line (Private-RAG, LPRAG, DP-SynRAG) is the only pragmatic path — statistical DP guarantees over standard retrieval backends, works with both dense and sparse components. Private-RAG achieves 100+ queries within ε≈10 budget. Trades information-theoretic PIR guarantees for practical deployability.

**Products / projects**

- **Tiptoe** (MIT, open source) — https://github.com/ahenzinger/tiptoe
- **ebm25.rs** (Rust, hobby project, no security proof) — https://github.com/slevental/ebm25.rs
- **IronCore Labs** — SSE variants in commercial product.

No production-grade product implements private hybrid retrieval for RAG at scale.

**Importance / maturity**

**Dense: active and accelerating. Sparse: thin. Hybrid: open gap.** The DP-RAG approach is the only practical path for hybrid retrieval today. PIR for dense search is advancing rapidly (Panther, Tiptoe, PIR-RAG) but remains too slow (18s/query) for interactive use. PIR-RAG (arXiv:2509.21325, Sep 2025) is the most directly relevant paper to RAG pipelines and worth reading in full.

---

### 4. Oblivious Retrieval (Query Hiding + Access Pattern Hiding)

**Key papers**

*Foundational:*

| Paper | Year | Cites | URL |
|---|---|---|---|
| Goldreich & Ostrovsky: ORAM (JACM 1996) | 1996 | ~3,000+ | https://dl.acm.org/doi/10.1145/233551.233553 |
| Chor et al.: PIR (JACM 1998) | 1998 | 1,614 | https://doi.org/10.1145/293347.293350 |
| Islam, Kuzu, Kantarcioglu: Access pattern disclosure attack (NDSS 2012) | 2012 | 496 | NDSS 2012 |
| Kellaris et al.: Generic attacks on secure outsourced databases (CCS 2016) | 2016 | 267 | https://doi.org/10.1145/2976749.2978386 |
| Mishra et al.: Oblix — doubly-oblivious search via SGX (S&P 2018) | 2018 | 160 | https://doi.org/10.1109/sp.2018.00045 |
| ZeroTrace: Oblivious memory from SGX (NDSS 2018) | 2018 | 182 | https://doi.org/10.14722/ndss.2018.23239 |
| Boldyreva & Tang: Privacy-Preserving Approx. kNN (PoPETs 2021) | 2021 | 14 | https://doi.org/10.2478/popets-2021-0084 |

*RAG-specific (2024–2026):*

| Paper | Year | URL | Approach |
|---|---|---|---|
| RemoteRAG: (n,ε)-DistanceDP (ACL Findings 2025) | 2025 | https://doi.org/10.18653/v1/2025.findings-acl.197 | DP embedding noise; query hiding only; 0.67s / 46KB at 10⁵ docs |
| Compass: Encrypted Semantic Search (OSDI 2025) | 2025 | https://eprint.iacr.org/2024/1255 | FHE + ORAM; both query and access pattern privacy |
| RAGtime-PIANO (ePrint 2026) | 2026 | https://eprint.iacr.org/2026/231 | FHE (coarse matching) + PIR (doc fetch); first fully secure RAG protocol |
| PIR-RAG (arXiv 2025) | 2025 | https://arxiv.org/abs/2509.21325 | Semantic clustering + lattice-based PIR; access pattern hiding |
| Opal: Private Memory for Personal AI (arXiv 2026) | 2026 | https://arxiv.org/abs/2604.02522 | ORAM + TEE for AI memory/RAG |

**Problem addressed**

Two distinct attack vectors:

- **Query hiding** — server learns plaintext query embedding → infers user intent, topics, sensitive conditions. Embedding inversion attacks make this concrete.
- **Access pattern hiding** — even with encrypted queries/data, which documents are retrieved per session reveals popular/sensitive documents, query-to-document correlations, and volume/frequency metadata. Kellaris et al. (CCS 2016, 267 cites) proved formally that access pattern + volume are unavoidable leakage in all practical outsourced DBs short of ORAM.

**State of approaches**

| Approach | Bandwidth overhead | Latency | Hides query | Hides access pattern |
|---|---|---|---|---|
| DP embedding (RemoteRAG) | ~1x | +0.67s | Partial (ε-DP) | No |
| PIR multi-server (lattice) | O(√n)–O(n^{1/3}) | 100ms–seconds | No (separate) | Yes |
| ORAM (tree-based) | O(log n) per access | 2–10x | No | Yes |
| FHE + PIR (RAGtime-PIANO) | High | Minutes at scale | Yes | Yes |
| TEE + oblivious structures (Compass, Opal) | O(log n) | 5–50x | Yes (inside enclave) | Yes (if doubly-oblivious) |

Key tension: full obliviousness (ORAM/PIR) gives provable security at O(log n) or worse overhead. DP is fast but probabilistic and doesn't hide access patterns. Side-channel attacks on TEEs are real — Xu et al. (S&P 2015, 708 cites) demonstrated full document reconstruction via page-fault side-channels from SGX; doubly-oblivious data structures (Oblix, Opal) are required to close this.

**Products / projects**

- **RemoteRAG** (Tsinghua, ACL 2025, practical DP) — https://doi.org/10.18653/v1/2025.findings-acl.197
- **Tiptoe** (MIT, open source) — https://github.com/ahenzinger/tiptoe
- **Opal** (personal AI memory, ORAM+TEE, 2026) — https://arxiv.org/abs/2604.02522
- **RAGtime-PIANO** (NSF-funded, Notre Dame, academic prototype) — https://eprint.iacr.org/2026/231

No commercial product ships full oblivious RAG.

**Importance / maturity**

**Early but accelerating; all RAG-specific papers from 2024–2026.** Remote/cloud RAG exposes every query and access pattern to the provider — this is a concrete, exploitable leakage today, not theoretical. Practical near-term option: DP query perturbation (RemoteRAG, easy to implement, 0.67s overhead, but query privacy only). Full obliviousness requires ORAM infrastructure or TEE deployment — significant engineering. Lattice-based PIR (SimplePIR, PIANO, Spiral) is getting faster and directly powers the newest RAG-specific schemes.

---

### 5. Trusted / Verifiable Embedding and Retrieval

> **Important distinction:** *Citation provenance* (metadata: who cited what) is trivially solved with signed metadata. *True verifiable provenance* = cryptographic proof that (a) a specific model computed a specific embedding, (b) retrieval was honest (correct top-k returned), (c) results weren't tampered with. These require entirely different machinery.

**Key papers**

*ZK / zkML:*

| Paper | Year | URL |
|---|---|---|
| Kang et al.: Scaling up Trustless DNN Inference with ZK Proofs | 2022 | https://arxiv.org/abs/2210.08674 |
| Keršič et al.: On-chain zkML — EZKL vs Orion comparison | 2024 | https://doi.org/10.1016/j.jksuci.2024.102207 |
| Anchuri et al.: Verifiable AI via Lightweight Cryptographic Proofs (SaTML 2026) | 2026 | https://arxiv.org/abs/2603.19025 |
| Jolt Atlas: ZK for embeddings/small LMs | 2026 | https://arxiv.org/abs/2602.17452 |
| Akor et al.: EZKL benchmarks — empirical costs | 2026 | https://doi.org/10.1109/icaiic68212.2026.11454315 |

*TEE / attested inference:*

| Paper | Year | URL |
|---|---|---|
| Steiakakis & Vasiliadis: Python ML inference inside SGX (ONNX) | 2026 | https://doi.org/10.3390/jcp6010023 |
| PrivANN: TEE+ORAM private ANN | 2025 | https://doi.org/10.1109/trustcom66490.2025.00140 |
| QShield: SGX-based SQL over encrypted data | 2020 | https://doi.org/10.1109/tpds.2020.3024880 |

*Verifiable ANN (result integrity):*

| Paper | Year | URL |
|---|---|---|
| VLAH: Verifiable ANN via LSH+HNSW+Merkle proofs | 2025 | https://doi.org/10.1109/trustcom66490.2025.00187 |
| ANNProof: Blockchain-anchored ANN verification | 2024 | https://doi.org/10.1016/j.future.2024.03.002 |
| VPIRL: LWE encryption + Merkle HT for image retrieval | 2025 | https://doi.org/10.1109/tdsc.2025.3649671 |

**Problem addressed**

An untrusted RAG server can: substitute the embedding model (weaker/biased model while claiming otherwise); return cherry-picked top-k results (omitting inconvenient documents); tamper with retrieved context before passing to the LLM. True verifiable provenance requires proving all three steps are honest.

**State of approaches**

| Approach | Proves model identity | Proves computation correctness | Proves result integrity (top-k) | Practical today |
|---|---|---|---|---|
| Full ZK circuit (EZKL/Halo2) | Yes (committed weights) | Yes, cryptographic | Yes (if index committed) | No — Conv2d alone requires >125 GB RAM for real embedding models |
| Hybrid statistical ZK (Anchuri 2026) | Partial | Probabilistic | Partial | Yes — milliseconds, LLM-scale, requires rational/incentivized prover |
| TEE attestation (SGX/TDX) | Yes (measurement hash) | Yes (trust hardware) | Yes (if index inside enclave) | Yes — ~17% overhead small models |
| TEE + ORAM (PrivANN) | Yes | Yes (trust hardware) | Yes + access pattern hiding | Research-stage 2025 |
| Verifiable ANN (VLAH, ANNProof) | No | Partial | Yes | Emerging |
| Blockchain document hash | No | No | No | Yes (trivial, but proves only pre-ingestion integrity) |

**Critical gaps:**

1. **Identity gap** — zkML proves computation correct given committed weights, but does NOT prove which model those weights belong to. Explicitly noted as "missing identity layer" in 2026 preprints.
2. **Scale gap** — EZKL empirically measured (Akor et al. 2026): Conv2d layers require >125 GB RAM in Halo2. Real embedding models (BERT-base, Jina v2) are currently impractical to prove in ZK.
3. **Index gap** — proving ANN search returned true top-k is a separate hard problem from proving embedding correctness. VLAH/ANNProof address this independently.
4. **Retrieval-to-answer gap** — nothing currently proves the LLM generation step used retrieved context honestly.

**Products / projects**

- **EZKL** — open source, production ZK-SNARK + TEE + CUDA backends, Microsoft+MIT partnerships. https://ezkl.xyz / https://github.com/zkonduit/ezkl
- **Jolt Atlas** — 2026 research, ZK for embeddings/small LMs, BlindFold technique. https://arxiv.org/abs/2602.17452
- **JSTprove** — 2025 CLI wrapper, Polyhedra Expander backend.
- **PrivANN** — 2025 research prototype.

**Importance / maturity**

**Underbuilt — fragmented and mostly research-stage.** No single system provides end-to-end proof that embeddings came from model X, top-k retrieval was honest, and generation used those documents — at production embedding scale. Near-term practical options: TEE attestation (~17% overhead, trusts Intel/AMD) or hybrid statistical proofs (millisecond verification at LLM scale, for staked/accountable providers). Full ZK for real embedding models: 2–5 years minimum. The scale gap is concrete and empirically quantified.

---

## Market Problems — Ranked by Economic / Practical Urgency

---

### 1. Regulatory Liability Exposure (HIPAA / GDPR / EU AI Act)

**Description:** Using cloud-hosted RAG over PHI, PII, or legally privileged data violates GDPR, HIPAA BAA requirements, and the EU AI Act's high-risk AI provisions. The EDPS TechSonar advisory explicitly addresses RAG and confirms embeddings derived from personal data are still personal data under GDPR. https://www.edps.europa.eu/data-protection/technology-monitoring/techsonar/retrieval-augmented-generation-rag

**Urgency and what it blocks:** GDPR cumulative fines reached ~€4.5B by 2024; OCR stepped up HIPAA enforcement with multi-million-dollar settlements throughout 2023–2025; EU AI Act in force with compliance obligations staging through 2026–2027. Active enforcement, active liability now. Blocks cloud RAG adoption in healthcare, financial services, legal, and government entirely. Protecto.ai 2025 survey: 40% of organizations experienced AI-related data privacy incidents.

**Companies / projects:**

- LogionOS — compliance runtime, 4,004 regulations, 6 jurisdictions. https://logionos.com
- Protecto.ai — PII redaction for RAG. https://www.protecto.ai
- Microsoft Azure Confidential Computing, GCP Assured Workloads
- John Snow Labs (healthcare NLP)
- Artezio, Intuz, TechCloudPro, Aiveda — private LLM deployment services

---

### 2. PII / PHI Leakage via RAG Retrieval

**Description:** RAG systems routinely index PDFs containing PII/PHI without redaction, then expose that data at query time to any authorized user — not just the data subject. A demonstrated end-to-end exploit (Feb 2026): a malicious web page embedded in a retrieval corpus caused a RAG agent to exfiltrate secrets. https://www.kiteworks.com/cybersecurity-risk-management/ai-agents-ungoverned-data-security-threat/

**Urgency and what it blocks:** Known exploitable attack surface today. GDPR data minimization and purpose limitation principles apply directly to what gets indexed and returned. Makes RAG over unredacted enterprise document repositories non-compliant by default. Forces manual redaction workflows or restricts RAG to sanitized corpora, degrading quality.

**Companies / projects:**

- Protecto.ai — https://www.protecto.ai
- Microsoft Presidio — open source PII redaction, standard component in enterprise RAG
- AWS Comprehend PII detection
- Nightfall AI

---

### 3. Third-Party Inference API Exposure (Data Egress / Sovereignty)

**Description:** Every call to a cloud embedding API (OpenAI, Cohere, Jina) sends document/query content in plaintext to a third party. Violates SOC 2, FedRAMP, most financial-sector data handling policies, and data residency requirements. Cloudera October 2024 survey: **50% of enterprises use RAG specifically to "bring models to the data"** — half the market is already sovereignty-motivated. Vec2Text (2023) additionally proves that transmitting embeddings effectively transmits the original text.

**Urgency and what it blocks:** Blocking force for financial services, defense, and EU-regulated enterprises. Prevents use of frontier embedding models (OpenAI, Cohere) for any sensitive corpus. Pushes enterprises to local models. Slows AI adoption in regulated industries.

**Companies / projects:**

- Edgeless Systems / Privatemode.ai — confidential LLM inference via NVIDIA H100 TEEs, Capgemini partnership Apr 2025. https://www.edgeless.systems / https://www.privatemode.ai
- enclaive / Garnet — GenAI firewall on confidential cloud with Qdrant. https://www.enclaive.io
- Private LLM deployment services market broadly (Artezio, Intuz, TechCloudPro)

---

### 4. Data-in-Use Exposure During Search (Encrypted-at-Rest Is Not Enough)

**Description:** Vector databases encrypt data at rest, but decryption during search means raw vectors — and by reconstruction, source documents — are visible to cloud operators, privileged insiders, and in breach scenarios. The confidential computing funding wave confirms real demand: Vaultree $12.8M Series A; DataKrypto €3M Mar 2024; Lattica $3.25M pre-seed Apr 2025; CyborgDB NVIDIA partnership.

**Urgency and what it blocks:** Enterprises in banking, healthcare, and government cannot accept operator-visible data. Blocks cloud vector DB adoption for the most sensitive corpora. Forces on-premises deployment with associated cost and operational burden.

**Companies / projects:**

- **IronCore Labs Cloaked AI** — DCPE, commercial, Gartner Cool Vendor 2025. https://ironcorelabs.com/products/cloaked-ai
- **CyborgDB / Cyborg Inc.** — first confidential vector DB. https://www.cyborg.co · NVIDIA blog: https://developer.nvidia.com/blog/bringing-confidentiality-to-vector-search-with-cyborg-and-nvidia-cuvs/
- **Fortanix** — confidential computing platform, ~$90M raised. https://www.fortanix.com
- **Anjuna Security** — Seaglass platform, $67M total, $25M Series B2 Aug 2024. https://www.anjuna.io
- **Vaultree** — data-in-use encryption, $12.8M Series A. https://www.vaultree.com
- **DataKrypto** — FHE for AI inference, €3M Mar 2024. https://datakrypto.ai
- **Lattica** — FHE platform for AI, $3.25M pre-seed Apr 2025. https://www.securityweek.com/lattica-emerges-from-stealth-with-fhe-platform-for-ai/
- **VectorX** — encrypted vector DB plugin. https://vectorxdb.ai

---

### 5. Multi-Party / Cross-Institution Data Sharing Blockages

**Description:** Hospitals cannot pool patient records for shared RAG; banks cannot join fraud-detection corpora; pharma companies cannot share clinical trial embeddings. The "data clean room" problem — multiple parties want joint RAG over private, siloed datasets without revealing data to each other or to any central party.

**Urgency and what it blocks:** Real but long sales cycles (18–36 months in regulated verticals). Prevents cross-institutional AI in healthcare (multi-hospital RAG), finance (cross-bank fraud detection), and pharma (multi-trial clinical search). Large TAM but slow-moving.

**Companies / projects:**

- **Opaque Systems** — ~$46M total raised ($22M Series A 2022 + $24M Series B @ $300M val Feb 2026), TEE-based confidential analytics + multi-party AI. https://www.opaque.co
- **DataKrypto** — FHE for AI inference. https://datakrypto.ai
- **Lattica** — FHE platform for AI models. https://www.securityweek.com/lattica-emerges-from-stealth-with-fhe-platform-for-ai/
- **FRAG** — academic, IND-CPA secure distributed vector DB. https://doi.org/10.48550/arxiv.2410.13272
- **Marblerun / Contrast** (Edgeless Systems) — confidential Kubernetes, open source.

---

## Summary

| Direction | Maturity | Key gap | Best near-term option |
|---|---|---|---|
| Encrypted vector DB storage | Early — 1 commercial product | FHE/PIR too slow; DCPE has residual leakage | IronCore Cloaked AI (DCPE) or local self-host |
| Private embedding generation | Partial — local solved, cloud unsolved | Cryptographic inference 10–1000x too slow | Local model (FastEmbed, Ollama) |
| Private PIR/retrieval (sparse/dense/hybrid) | Dense: active; Sparse: thin; Hybrid: open gap | No private hybrid retrieval scheme exists | DP-RAG (ε-differential privacy over standard backends) |
| Oblivious retrieval (query + access pattern) | Research-stage — all papers 2024–2026 | ORAM/PIR infeasible at interactive latency | RemoteRAG DistanceDP (query only, 0.67s) |
| Verifiable/attested retrieval | Fragmented — mostly research | Scale gap: zkML requires >125 GB RAM for real models | TEE attestation (SGX/TDX) or hybrid statistical ZK (Anchuri 2026) |

**Biggest economic opportunity, underserved:** Problems 1–3 are paying now. Problem 4 (data-in-use) has active venture investment. Problem 5 (multi-party) has the largest theoretical TAM but longest sales cycles. The technical literature is most active on dense private retrieval; private BM25 and private hybrid retrieval are concrete open gaps with no shipped solutions.

**RAG market context:** Projected growth from ~$1.7B (2024) to $18–20B+ by 2030 (MarketsandMarkets). Vector database startups raised >$350M by mid-2024 with privacy/compliance cited as the next differentiation wave. 50% of enterprises already use RAG with data privacy as the stated motivation (Cloudera 2024).
