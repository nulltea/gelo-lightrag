---
type: research
status: reference
created: 2026-05-11
updated: 2026-05-26
tags: [rag, privacy, navigation]
companion: [fhe-encrypted-vector-db, private-embedding-research, private-information-retrieval, private-graph-search]
archive_reason: "Originally a broad survey; the four narrower landscape docs in docs/research/ are now deeper and current. This doc was collapsed to a navigation spine + market-problem context."
---

# Privacy-Enhanced RAG: Research & Market Landscape

> Originally researched 2026-04-14. Collapsed 2026-05-26 — the per-direction
> technical content now lives in deeper docs; this file is the entry point
> + market context.

## Technical Directions

Each direction is covered in depth in a dedicated research doc. Jump to the
relevant one:

| Direction | Primary doc | Scope |
|---|---|---|
| Private / encrypted vector DB storage | [`fhe-encrypted-vector-db.md`](fhe-encrypted-vector-db.md) | FHE schemes (CKKS, BFV), DCPE, commercial landscape (IronCore Cloaked AI, CyborgDB, etc.). Tiptoe / Panther / RAGtime-PIANO / FRAG / p²RAG / Compass. |
| Private embedding generation | [`private-embedding-research.md`](private-embedding-research.md) | Embedding-layer threats (Vec2Text, EDNN inversion), in-TEE inference, DP perturbation, obfuscation, split-inference, MPC/FHE crypto encoders, comparison and recipes. |
| Private retrieval (PIR / dense / sparse / hybrid) | [`private-information-retrieval.md`](private-information-retrieval.md) | PIR/ORAM/secret-sharing/DP schemes for retrieval. Tiptoe, Panther, PrivANN, RemoteRAG, RAGtime-PIANO, p²RAG, GraSS, sparse-retrieval gaps. |
| Oblivious retrieval (query + access-pattern hiding) | [`private-information-retrieval.md`](private-information-retrieval.md) §B + DP-RAG section | RemoteRAG DistanceDP, Compass, Opal, ORAM-based options and their latency floors. |
| Verifiable / attested retrieval | [`private-information-retrieval.md`](private-information-retrieval.md) §D | zkML, TEE attestation, V3DB, ZKIFV, ANNProof. The scale gap (zkML needs >125 GB RAM for embedding-scale models). |
| Graph-RAG privacy (LightRAG-class systems) | [`private-graph-search.md`](private-graph-search.md) | Attack landscape (extraction, reconstruction, poisoning, logic-rewiring) + graph-specific retrieval primitives (GORAM, Graphiti, OblivGNN). |

For our own integrated system designs see `docs/dev/prototype/private-rag-system-design.md` and `docs/dev/prototype/private-graph-rag-design.md`.

## Market Problems — Ranked by Economic / Practical Urgency

> The market-context section below is the unique value of this doc: it
> frames *why* the technical work above matters in compliance / commercial
> terms. The technical surveys do not duplicate this.

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
