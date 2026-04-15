# FHE Approaches for Private / Encrypted Vector DB Storage

> Researched 2026-04-14. Sources: Edgequake (local corpus), OpenAlex (paper metadata), prior research doc.

---

## Overview

Scope: papers and projects applying Fully Homomorphic Encryption (or closely related schemes) to store and search vector embeddings without exposing query content or access patterns to the server. Ordered roughly by maturity / recency.

---

## 1. Tiptoe — Private Web Search via Linearly Homomorphic Encryption

**Paper:** Henzinger, Dauterman, Corrigan-Gibbs, Zeldovich — MIT / Berkeley, SOSP 2023. 23 citations.
**URL:** https://doi.org/10.1145/3600006.3613134 | Code: https://github.com/ahenzinger/tiptoe

### Approach

Scheme: **linearly homomorphic encryption (LHE)** over LWE lattices. Not full FHE — supports only linear operations (inner products). No bootstrapping required.

Pipeline:
1. Client compresses query embedding into a short LHE ciphertext
2. Server computes homomorphic dot products between encrypted query and all stored document embeddings to identify top-N clusters
3. Server returns encrypted cluster results; client decrypts and selects candidates
4. Client fetches documents from matching clusters

Key insight: semantic search reduces to private inner-product computation, which LHE handles efficiently without the expensive bootstrapping that full FHE requires. Query-independent pre-processing (74% of total bandwidth) is amortizable across sessions.

### Privacy Gains

- Query vector never seen in plaintext by the server
- Crypto-only guarantee — no TEE, no non-colluding server assumption
- Server learns nothing about query content or intent

### Performance

| Metric | Value |
|---|---|
| Corpus size | 360M web pages |
| Server cluster | 45 servers |
| Server compute | 145 core-seconds per query |
| Communication | **56.9 MiB** (74% pre-query, amortizable) |
| End-to-end latency | **2.7s** |
| MS MARCO avg rank | **7.7** (vs 2.3 non-private neural, 6.7 tf-idf) |

### Tradeoffs

- Requires 45-server cluster; not deployable on a single server
- Significant search quality degradation vs non-private neural retrieval (rank 7.7 vs 2.3)
- LHE supports only linear operations — no support for HNSW or IVF index structures without additional machinery
- Best on conceptual queries; poor on exact string matches
- 56.9 MiB per-query comm is high even with amortization

### Used By

Open source (MIT, NSF-funded). Foundation paper cited by Panther, PIR-RAG, RAGtime-PIANO. No commercial deployment.

---

## 2. Panther — Single-Server Private ANNS via Hybrid HE

**Paper:** Li, Huang, Zhang, Hong, Liu, Wei, Chen — Ant Group + Chinese Academy of Sciences + Zhejiang University, CCS 2025. 0 citations (very recent).
**URL:** https://doi.org/10.1145/3719027.3765190

### Approach

Scheme: **hybrid co-design** of PIR + secret sharing + garbled circuits + HE (BFV and CKKS). Called the "middle-private approach" — no single crypto primitive is efficient enough at this scale, so the protocol mixes them strategically:

- **HE (BFV/CKKS)** for encrypted distance computation over cluster centroids
- **Garbled circuits** for encrypted comparison operations (selecting top-k)
- **Secret sharing** for index traversal
- **PIR** for final document fetch

Operates in the **single-server setting** — a strictly stronger security guarantee than Tiptoe's 45-server cluster. Protocol formally named P_privApprox; uses a shared-cloud framework internally.

### Privacy Gains

- Query hidden from a single server (not just from non-colluding servers)
- Access pattern hidden (which documents retrieved)
- Single-server setting is strictly stronger than multi-server non-colluding

### Performance

Tested on 4 public ANNS benchmark datasets (Deep, GIST, SIFT, etc.):

| Metric | Value |
|---|---|
| Dataset size | 10M vectors |
| Latency | **18 seconds** |
| Communication | **284 MB** |
| vs Chen et al. 2020 | **7.8× faster, 20× more compact** |
| Online bandwidth (smaller scale) | ~10 MB / ~0.06s serving latency |

### Tradeoffs

- 18s/query at 10M vectors — not interactive
- 284 MB comm per query is high for production
- "Middle-private approach" hybrid protocol complexity is a security analysis burden
- No ablation showing individual component contribution to overhead
- Published 2025-11-19 — not yet cited or externally validated

### Used By

Built and funded by Ant Group (Alipay parent). Internal R&D — no public deployment announced.

---

## 3. RAGtime-PIANO — CKKS Coarse Search + PIR Document Fetch

**Paper:** Notre Dame, NSF-funded, IACR ePrint 2026/231. 0 citations.
**URL:** https://eprint.iacr.org/2026/231

### Approach

Scheme: **CKKS** (approximate FHE for floating-point arithmetic) for cluster-level search; **lattice-based PIR** for document fetch.

Three-stage pipeline:

| Stage | Who | What | Crypto |
|---|---|---|---|
| Stage 0 (offline) | Server + Client | Precompute cluster structures and key material | None (amortized) |
| Stage 1 (online) | Server | Homomorphic inner products over cluster centroids; return encrypted top-N cluster IDs | CKKS |
| Stage 2 (online) | Server | Fetch top-k documents from identified clusters | PIR |

CKKS is chosen specifically because it handles approximate floating-point inner products natively, appropriate for embedding similarity, and is 10–100× cheaper than BFV/TFHE for this operation. FHE is applied only at the cluster level (small N) to keep the expensive step cheap; the pre-processing stage amortizes setup costs across queries.

### Privacy Gains

- Query content hidden from server (CKKS FHE — server only sees ciphertext)
- Access pattern hidden (PIR on document fetch — server cannot tell which document was retrieved)
- Described as the "first fully secure RAG protocol" — no leakage from either search or retrieval phase
- Supports arbitrary top-k (unlike fixed-k schemes)

### Performance

Concrete benchmarks not publicly available (ePrint preprint). Based on the FHE/PIR category, estimated ~18s/query at 10M vectors. The two-stage design (coarse FHE over clusters + fine PIR for docs) is intended to reduce FHE surface to only the cheap cluster-level computation.

### Tradeoffs

- CKKS introduces approximation noise — may affect precision for embeddings that are geometrically close
- Requires CKKS bootstrapping if computation depth exceeds parameters
- PIR for document fetch still carries O(√n) communication in the worst case
- Two-stage design adds implementation and deployment complexity
- Preprint only, no external validation

### Used By

NSF-funded academic prototype (Notre Dame). No commercial deployment.

---

## 4. FRAG — Federated Vector DB with Single-Key HE

**Paper:** Zhao (single author), arXiv 2024. 3 citations.
**URL:** https://doi.org/10.48550/arxiv.2410.13272 | PDF: https://arxiv.org/pdf/2410.13272

### Approach

Scheme: **single-key homomorphic encryption** (likely BFV/BGV for integer arithmetic). Key novelty: **multiplicative caching** — precomputes and caches multiplicative factors for encrypting floating-point vector elements, significantly reducing per-element encryption cost in large federated environments.

Multiple mutually-distrusted parties each hold encrypted shards of a distributed vector DB. They collaboratively execute ANN search without any party learning the queries or data of others. Security proven via standard cryptographic reductions. Avoids the non-collusion assumption via a single shared key with careful protocol design.

### Privacy Gains

- IND-CPA security: encrypted query vectors + encrypted stored vectors
- No party learns plaintext of any other party's queries or data
- No non-collusion assumption needed (unusual for multi-party schemes)

### Performance

Claims "performance overheads comparable to traditional non-federated RAG systems." No specific latency or bandwidth numbers published. Validated on benchmark and real-world datasets per the paper.

### Tradeoffs

- Federated setting assumption — requires multiple parties; not applicable to single-cloud deployment
- Single-key design simplifies key management but is an unusual trust model
- 3 citations, single author — limited peer validation
- "Comparable to non-federated RAG" performance claim is extraordinary and unverified externally

### Used By

No deployment. Preprint only.

---

## 5. p²RAG — 2-Server Secret Sharing + PIR, Arbitrary Top-k

**Paper:** arXiv 2026. 0 citations.
**URL:** https://arxiv.org/abs/2603.14778

### Approach

Scheme: **2-server additive/Shamir secret sharing** combined with **PIR**.

- Vector DB shards are split across two non-colluding servers
- Query is secret-shared: each server sees only a share, computes partial ANN results on its shard
- Results are recombined client-side
- PIR handles final document fetch: neither server learns which document was retrieved

Key contribution over prior work: supports **arbitrary top-k** retrieval, achieved by merging the retrieval and fetching steps — PIR retrieves the entire target cluster, giving the client full context without revealing which k documents were selected.

### Privacy Gains

- Query content hidden (secret sharing — neither server sees full query)
- Access pattern hidden (PIR document fetch)
- Arbitrary k: no leakage of "how many results were retrieved"

### Performance

No concrete numbers extracted. 2-server PIR protocols (SimplePIR, PIANO) achieve O(√n) communication. Expected to be faster than single-server FHE approaches given the weaker server trust assumption.

### Tradeoffs

- 2-server non-colluding assumption — weaker security than Panther's single-server guarantee
- Requires operating two independent non-colluding cloud deployments
- PIR over full clusters = over-retrieval overhead (fetches more than needed)
- Preprint only, 0 citations

### Used By

No deployment. Preprint only.

---

## 6. Compass — FHE + ORAM for Encrypted Semantic Search

**Paper:** OSDI 2025, IACR ePrint 2024/1255. 0 citations.
**URL:** https://eprint.iacr.org/2024/1255

### Approach

Scheme: **FHE + ORAM combined**. FHE handles encrypted similarity computation (query hidden during computation); ORAM (Oblivious RAM) handles access pattern hiding during ANN index traversal (server cannot observe which memory locations are accessed).

This combination addresses both leakage surfaces simultaneously:
- FHE alone: computation is private but access pattern leaks
- ORAM alone: access pattern is hidden but query/computation leaks

Together they provide the strongest single-server cryptographic guarantee of any system in this list.

### Privacy Gains

- Query hidden (FHE — server computes on ciphertext)
- Access pattern hidden (ORAM — server cannot distinguish index traversal patterns)
- Both leakage surfaces addressed simultaneously
- Peer-reviewed (OSDI, top systems venue)

### Performance

Not publicly benchmarked. FHE + ORAM compositions typically carry multiplicative overhead: FHE adds O(poly) factor, ORAM adds O(log n) factor per access. Expected to be the slowest approach here.

### Tradeoffs

- ORAM requires periodic server-side database reshuffling
- FHE + ORAM together: highest implementation complexity
- No performance numbers available in indexed corpus
- Most expensive combination

### Used By

Academic prototype. No commercial deployment. OSDI acceptance provides stronger peer-review credibility than the preprints above.

---

## 7. Tiptoe Extension — Private Text-to-Image Search

Tiptoe also supports private text-to-image search and (with minor modifications) audio and code search via the same LHE mechanism. Same performance characteristics as above.

---

## Commercial Landscape

| Company | Approach | Actual scheme | Status |
|---|---|---|---|
| **IronCore Labs Cloaked AI** | DCPE — **not FHE**; approximate distance-comparison-preserving symmetric encryption; nearest-neighbor search runs natively on encrypted vectors in existing DBs | SAP (Scale-and-Perturb), see §below | **Deployed** — Gartner Cool Vendor 2025; Alloy SDK in Kotlin/Java/Python/Rust; integrates with Qdrant, Pinecone, Weaviate |
| **DataKrypto** | FHE for AI inference including vector operations | Unspecified (likely CKKS) | Pre-product — €3M seed, Mar 2024 |
| **Lattica** | FHE platform for AI models including retrieval | Unspecified | Pre-product — $3.25M pre-seed, Apr 2025 |
| **Javelin AI / Highflame** | HE for embeddings | Unspecified | Early-stage startup, Nov 2024 |
| **CyborgDB / Cyborg Inc.** | Confidential vector DB, NVIDIA cuVS partnership | TEE-based (not FHE) | Pre-revenue |

**No company currently ships true FHE-based vector search at production scale.** IronCore DCPE is the only deployed product, with formally weaker-than-FHE guarantees.

---

## Deep Dive: IronCore Labs Cloaked AI / DCPE

> Sources: IronCore docs (how-it-works, overview), IronCore blog (NIST standards post, embedding attack post), independent Hexens.io security analysis, academic basis: Fuchsbauer, Ghosal, Hauke, O'Neill — "Approximate Distance-Comparison-Preserving Symmetric Encryption" (SCN 2022).

### What It Is

**DCPE** = Distance-Comparison-Preserving Encryption, specifically the **β-DCPE** (approximate) variant. Not homomorphic encryption — it is a symmetric cipher applied once at write time. The server never performs any cryptographic computation; it runs standard ANN search on the transformed vectors as if they were plaintext.

### How It Works: The SAP Scheme (Scale-and-Perturb)

Three sequential steps applied to a plaintext vector **v**:

1. **Scale** — multiply each element by a secret factor `s`, mapping the vector to a different magnitude space: `v' = s·v`
2. **Perturb** — add pseudorandom uniform noise bounded within a sphere of radius `s·β/4`, where `β` is the tunable approximation factor: `v'' = v' + noise(PRF(key, nonce))`
3. **Shuffle** — deterministically reorder vector elements using a key-derived permutation: `v_enc = permute(v'', key)`

**Decryption** reverses the shuffle, subtracts the same noise (reconstructed via the same PRF + key + nonce), and descales.

**Key management:** One key per *segment* (e.g., per tenant). All vectors within a segment share the same scaling factor and permutation. Cross-segment comparison is cryptographically blocked.

**Supported distance metrics on encrypted vectors:** Euclidean, cosine, dot product — all preserved approximately.

### Formal Security Guarantee

**Security model: Real-or-Replaced (RoR)** — an adversary cannot distinguish a database containing record `x` from one where `x` is replaced by a nearby point. This formally proves resistance to membership inference attacks.

**The preserved property:** If `‖x−y‖ < ‖y−z‖ − β`, then after encryption the same distance comparison holds with high probability. Nearest-neighbor search is correct for all pairs separated by more than `β`.

**NOT IND-CPA.** The scheme is explicitly not semantically secure — it preserves enough distance structure to be useful, which is also what leaks.

### Performance

| Metric | Value |
|---|---|
| Server-side overhead | **~0** — vector DB runs unchanged on encrypted vectors |
| Client-side encryption latency | Near-zero — symmetric cipher (scale + noise + permute), no HE |
| Throughput | Not formally benchmarked; effectively bounded by memory bandwidth |
| Query overhead | None — client encrypts query vector with same key; DB searches normally |
| Infrastructure changes required | None — drop-in replacement at write/read time |

This is the core practical advantage over FHE: the server infrastructure (Qdrant, Pinecone, etc.) is completely unmodified.

### What It Protects Against

| Attack | Protected? | Notes |
|---|---|---|
| Embedding inversion (Vec2Text, EDNN) | **Yes** | Inverted output is "pure nonsense — doesn't produce words" |
| Membership inference | **Yes** | Formal RoR proof |
| Direct plaintext recovery from stored vectors | **Yes** | Without the key, no starting point for inversion |
| Frequency/statistical analysis | **Yes (mitigated)** | Normalization + shuffling defeats frequency attacks |
| PCA-based transform recovery (exact DCPE) | **Yes (mitigated)** | Perturbation destroys exact eigenvalue structure |

### Known Weaknesses and Leakage

**1. Distance ordering is preserved (fundamental, unavoidable)**

The nearest-neighbor structure of the corpus is visible in plaintext. An attacker with database access can:
- Run HDBSCAN or k-means on encrypted vectors to reconstruct the topic/semantic cluster map of the corpus
- Correlate access logs with the cluster map to infer what users are querying
- Determine which documents are "close" to each other without any key

This is not a flaw — it is the design. It is the price of zero server overhead.

**2. Chosen-plaintext partial inversion**

An attacker who obtains plaintext-ciphertext pairs under a given key can train a regression model to partially invert encrypted vectors. IronCore's own demo: "I live in Hawaii" → "I am in Florida" (partial, wrong but directionally correlated). Protection level depends on `β` (larger = noisier = harder to invert, but worse recall). Mitigated by keeping keys in KMS/HSM so pairing plaintext-ciphertext is difficult.

**3. No access pattern hiding**

The server knows exactly which encrypted vectors are retrieved per query. Combined with the preserved distance structure, this is a meaningful leakage channel over time. Kellaris et al. (CCS 2016) attacks apply directly.

**4. Approximation factor β is a user-tuned knob, not a cryptographic parameter**

β must be chosen relative to the distribution of inter-vector distances in the corpus. Too small → poor inversion protection; too large → ANN recall degrades. There is no formally safe default — it is dataset-dependent in practice even if the theoretical definition is dataset-independent.

### Comparison: DCPE vs FHE for Vector Storage

| Property | DCPE (IronCore) | FHE (Panther/RAGtime-PIANO) |
|---|---|---|
| Server infrastructure | **Unchanged** — existing vector DB | Custom or heavily modified |
| Query latency overhead | **~0ms** | 18s (Panther, 10M vecs) |
| Hides query from server | **No** — server sees encrypted query, can cluster queries | Yes |
| Hides which docs retrieved | **No** — full access pattern visible | Yes (with PIR) |
| Hides distance structure | **No** — preserved by design | Yes |
| Inversion attack resistance | Yes (with appropriate β) | Yes (unconditional) |
| Formal security model | β-DCPE / RoR | IND-CPA or stronger |
| Deployment complexity | **Minimal** — SDK, no infra change | High — custom server |
| Practical today | **Yes** | Research-stage (18s not interactive) |

### Verdict

DCPE is the right choice when the threat model is **"prevent document content recovery from a stolen vector DB"** — i.e., an attacker who exfiltrates the vector database and tries to reconstruct documents. It is not appropriate when the threat model includes **"prevent the server from learning what users are searching for"** or **"hide which documents are popular/accessed."** Those require FHE or ORAM, which are not yet interactive at production scale.

---

## Comparative Summary

| System | Scheme | Single server | Hides query | Hides access pattern | Latency (10M vecs) | Comm | Peer-reviewed |
|---|---|---|---|---|---|---|---|
| **Tiptoe** | LHE (LWE) | No (45 servers) | Yes | Partial | **2.7s** | 56.9 MiB | Yes (SOSP) |
| **Panther** | HE + GC + SS + PIR | **Yes** | Yes | Yes | 18s | 284 MB | Yes (CCS) |
| **RAGtime-PIANO** | CKKS + PIR | Yes | Yes | Yes | ~18s (est.) | High | No (ePrint) |
| **Compass** | FHE + ORAM | Yes | Yes | Yes | Unknown (high) | High | Yes (OSDI) |
| **p²RAG** | SS + PIR | No (2 servers) | Yes | Yes | Unknown | O(√n) | No (preprint) |
| **FRAG** | Single-key HE | Yes (federated) | Yes | Yes | Claimed ~1× | Unknown | No (preprint) |
| **IronCore DCPE** | DCPE (not FHE) | Yes | **No** | **No** | ~0ms | ~0 | No |

---

## Key Gaps and Observations

1. **CKKS dominates** academic proposals for vector similarity — it handles floating-point inner products natively and is 10–100× cheaper than BFV/TFHE for this operation. All serious RAG-specific FHE schemes use it.

2. **Hybrid protocols are the practical frontier.** Panther's PIR+GC+SS+HE co-design achieves 7.8× speedup over pure HE while keeping single-server guarantees. This "mix and match" approach will likely dominate near-term work.

3. **The performance wall is real.** 18s/query at 10M vectors (Panther, best-in-class single-server) is not interactive. The gap between DCPE (~0ms, weaker) and full FHE (18s, stronger) is enormous. Nothing in the 2024–2026 literature closes it.

4. **FHE + ORAM (Compass) is the theoretically complete combination** but the most expensive and least benchmarked. ORAM's O(log n) access pattern overhead compounds with FHE's computational overhead.

5. **No production FHE vector DB exists.** The entire space is research prototypes or pre-product startups. IronCore DCPE is the only deployed product in the proximity-preserving-encryption space.

6. **Verification gap.** FRAG's claim of "comparable to non-federated RAG" performance would be a major breakthrough if true — it has not been independently verified and comes from a single-author preprint.
