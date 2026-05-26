---
type: research
status: current
created: 2026-05-11
updated: 2026-05-11
tags: [reranking]
---

# Private Reranking / Filtering for RAG

> Research date: 2026-04-21 (rev-4). Sources: OpenAlex (primary — 6 targeted searches + citation expansion), SearxNcrawl (web, products, repos), arXiv + USENIX + IEEE + ACM full-PDF fetch for benchmark tables, Edgequake (existing corpus check), local `./docs/`.

---

## Overview

Reranking is the post-retrieval step that re-scores top-k' ANN candidates with a more precise relevance model before assembling the generation prompt. In standard RAG it improves NDCG@10 by 5–15% over raw ANN cosine scores. In **private RAG** the step has two distinct roles:

1. **Correcting DistanceDP noise** — DistanceDP inflates top-k → top-k' (e.g., k=5 → k'=258 at r=0.033) for plausible deniability; reranking collapses k' back to k using exact scores invisible to the server.
2. **Matching retrieval precision to the threat model** — who runs the scorer determines what is leaked: query text, document text, ranking order, and access pattern. The reranker sits at the boundary between encrypted storage and plaintext generation, so it is the sharpest privacy crossing in the pipeline.

**No paper addresses "private cross-encoder reranking for RAG" as a standalone contribution.** However, recent work reframes the problem under adjacent names — *private approximate top-k*, *oblivious top-k selection*, *privacy-preserving top-k retrieval*, *PRAG / p²RAG* — and delivers the core primitives needed. The "Towards Secure RAG" survey (Mu et al. 2026) explicitly classifies reranking privacy as underdeveloped.

This revision (rev-2) folds in 10+ papers not present in the first pass, particularly: **p²RAG (2026), PRAG (2024), SANNS (2019), Tiptoe (2023), HR+QDA (2026), Oblix (2018), Cong et al. Oblivious Top-k (2025), TRSE (2013), Delta/AsymML (2022–2023), Privacy-aware two-level inverted index (Qiao 2023).**

---

## What Reranking Is

**Inputs:**
- Query `q` (text or embedding)
- Top-k' candidate chunks `{(e_i, x_i)}` — embeddings and (possibly encrypted) text
- Optional: metadata (source, timestamp, ACL attributes)

**Process:** score each (query, chunk) pair with a relevance model more precise than cosine ANN similarity. Sort by score. Select top-k.

**Output:** ordered top-k chunks for the generator.

**Methods by precision / cost (non-private baselines):**

| Method | How it works | Latency (50 candidates) | Quality gain vs ANN |
|---|---|---|---|
| **Embedding re-scoring** | `cos(q_exact, e_i_decrypted)` — re-score with exact (non-noisy) query embedding | ~0ms | Eliminates DP artifact; no true reranking gain |
| **Cross-encoder** | Joint encode `[q; x_i]` → relevance score (BGE-reranker-v2-m3, ms-marco-MiniLM) | ~100–400ms CPU / ~10ms GPU | +5–15% NDCG@10 |
| **Lightweight cross-attention (HR+QDA)** | 1.77M-param cross-attention head on bi-encoder outputs | ~29ms CPU | 91% of BGE quality, 157× smaller |
| **Late interaction (ColBERT)** | Per-token MaxSim over query × doc token matrices | ~5–20ms (pre-indexed) | ~3–8% below cross-encoder |
| **LLM-based (RankGPT)** | LLM scores (q, chunk) pairs or listwise | ~1–5s/candidate | Highest; heaviest |

### What reranking actually consumes

Different reranker types need different inputs — this determines which encryption layer (AES on chunk text vs. CAPRISE on embedding vectors) must be pierced at rerank time.

| Reranker | Needs plaintext chunk text? | Needs embeddings? |
|---|---|---|
| **Cross-encoder** (BGE-reranker, ms-marco-MiniLM) | **Yes** — joint input `[q; x_i]` fed into BERT | No (recomputed inside) |
| **Embedding re-scoring** (Approach 1) | No | **Yes** — single vector per chunk |
| **ColBERT late interaction** | No | **Yes** — token-level matrix per chunk |

This distinction matters for key management in Approach 4:
- **Cross-encoder rerankers** require AES-decrypting the k' candidate chunks' text before scoring — the reranker host holds the AES key for a window during rerank.
- **Embedding-only rerankers** (embedding re-scoring, ColBERT) need only CAPRISE-encrypted vectors; the AES key is not touched during rerank and chunk text stays sealed until generation assembles the final top-k prompt.

---

## What Must Be Private

| Leakage surface | Adversary | Threat |
|---|---|---|
| **Query text at scoring time** | Whoever runs the reranker | Cross-encoder sees `(q, x_i)` in plaintext — query fully exposed |
| **Document text at scoring time** | Whoever runs the reranker | Server sees which documents are scored jointly with the query |
| **Reranking access pattern** | Storage server / reranker host | Which of the k' candidates were selected → what is relevant |
| **Final top-k order (rank signal)** | Generator host, downstream pipeline | Ranked order leaks which document is most semantically similar to query |
| **Score distribution** | Anyone observing scores | Raw scores can be inverted to reveal query-document alignment |

In the Approach-4 pipeline (Step 9, between DistanceDP Step 8 and generation Step 10):
- AES-decryption of chunk text has already happened at Step 7 inside the TEE (or at the client in Option 2 key model).
- Storage server has already observed k' chunk IDs; reranking adds no new access-pattern leakage vs. storage.
- The relevant adversary is **whoever executes the reranker**, not the storage server.

---

## Approaches

### 1. Client-Side Embedding Re-Scoring

**What it does.** After CAPRISE-decrypting the k' returned embeddings, the client re-scores each with the *original* (non-noisy) query embedding: `score_i = cos(q_exact, e_i_decrypted)`. Picks top-k.

**Why it matters in private RAG.** DistanceDP added Laplace noise to the query at submission time. The server retrieved neighbors of `q_noisy`. Re-scoring recovers the true ordering from the inflated pool at zero compute cost.

**Privacy.** Cloud sees nothing new. Client already had `q_exact` (never transmitted) and the decrypted embeddings. Chunk text never needs to be decrypted for this step.

**Used by.** CAPRISE / prRAG (Ye et al. 2026), Algorithm 2 line 18: `Reranked({e_j}) ← ReRank{e_i}` — operates on decrypted embeddings, not plaintext text.

**Tradeoffs.**
- Zero infrastructure
- Eliminates DP artifact; no cross-encoder-quality lift
- Requires only embedding decryption, not AES of chunks

---

### 2. Client-Side Cross-Encoder (incl. HR+QDA lightweight variant)

**What it does.** Client AES-decrypts top-k' chunks (Option 2 key model), runs a local cross-encoder (BGE-reranker-v2-m3, ms-marco-MiniLM, or the lightweight HR+QDA head) to jointly score `(q, x_i)` pairs, picks top-k.

**Privacy.** Same as Approach 1 — cloud sees nothing. Client runs the reranker locally. Full cross-encoder quality.

**Strongest variant — HR+QDA (Maringan & Fitrianah 2026, *Discover Computing*):**
- 1.77M-parameter cross-attention head (vs. BGE-reranker-base 278M → 157× smaller)
- Trained on-prem in 13 min with 1,500 labeled examples
- Achieves 91% of SOTA MRR (0.744 vs 0.814 on NFCorpus)
- 2.3× faster inference (29ms vs 68ms CPU for 50 candidates)
- Designed explicitly for privacy-constrained enterprise on-prem deployment — no external transmission

**When it applies.** Option 2 (client holds CAPRISE key): natural fit, client already has all keys.
**When it breaks.** Options 1/3 (TEE holds key) with thin clients — forwarding chunks to the client negates the point of the TEE.

**Latency.** BGE-reranker-v2-m3: ~100–200ms on CPU for 50 candidates, ~10ms GPU. HR+QDA: 29ms CPU. Browser WASM feasible for the small model (90MB ms-marco, 20MB HR+QDA).

---

### 3. TEE-Side Cross-Encoder (Opal approach)

**What it does.** Cross-encoder runs inside the same TEE that holds CAPRISE-decrypted embeddings and AES-decrypted chunk text. All data-dependent reasoning stays inside the enclave.

**Privacy.** Cloud operator sees only:
- Reranking budget K (public per Opal's security definition)
- Fixed-size ORAM accesses to storage
Neither query, document, score, nor ranking order is visible.

**Used by.** Opal (Kaviani et al. 2026) — explicitly bakes cross-encoder reranking inside TEE with formal `Π = {ANN fetch count n, reranking budget K, ...}`.

**Fits.** Options 1/3 key model — TEE already holds CAPRISE key and AES key.

**Tradeoffs.**
- Cross-encoder adds ~90–567MB to TEE memory footprint
- ~100–200ms CPU (TEE's CPU-only constraint dominates)
- K visible (already implied by k' retrieval)
- Most complete privacy: document text never leaves TEE

**Enhancement: TEE+GPU split inference.** Mechanisms, benchmarks, and comparison table are centralized in `private-inference.md` §E "TEE Split-Inference" — covering Delta/AsymML/3LegRace, ObfuscaTune, GELO, Shredder, and Privacy-Aware Split Inference. **Reranker-specific notes:**
- None of the four is *published* for cross-encoder reranking; all reranker framings are extrapolation.
- **ObfuscaTune's pattern** maps cleanest — TEE holds the embedding table + scoring head (no lm_head for a reranker); GPU runs the 12 transformer blocks on `Q·H`. Smallest TEE footprint.
- **Shredder at `[CLS]`** (cut after final encoder block, before scoring head) is the cheapest to deploy — no model retrain, train only the noise distribution.
- **Delta** gives the only formal DP bound but requires post-factorization fine-tune of ~72 matrices; research-scale effort.
- **GELO** is a weaker fit for a reranker than for a generator — the reranker's scoring head is small, so "non-linear ops in TEE" still puts most of the score-producing compute in the TEE.

---

### 4. MPC / Secret-Shared Reranking (PRAG, p²RAG, SANNS, Panther)

**What it does.** Reranker scores and/or top-k selection run across two (or more) non-colluding servers using secret sharing or garbled circuits. No single server sees the query, the documents, or the ranking order.

**Key systems:**

**PRAG — Private Retrieval Augmented Generation (Zyskind, South, Pentland 2023, ACL PrivateNLP 2024).** First MPC RAG retrieval system. Uses a novel MPC-friendly protocol for inverted-file (IVF) approximate search with **sublinear** communication complexity. Exact top-k and approximate top-k over secret-shared vectors. No server observes query or database. Targets the retrieval step but scoring primitives extend to reranking.

**p²RAG (Ming et al. 2026, *arXiv*).** **Most promising recent system for this exact step.** Privacy-preserving RAG service supporting *arbitrary* top-k without re-sorting candidates. Uses **interactive bisection over secret-shared scores** on two semi-honest non-colluding servers — avoids the sort-based top-k bottleneck. Includes malicious-user defenses and tight leakage bounds on the database. **3–300× faster than PRAG for k = 16–1024.** This is the first MPC retrieval/reranking system with large-k practicality — directly relevant for RAG pipelines that want to feed 100+ chunks into long-context LLMs.

**SANNS (Chen et al. 2019/2020, USENIX Security).** MPC k-NN with a *new circuit for approximate top-k with `O(n + k²)` comparators* (vs. the `O(n log n)` sort bound), built from LHE + distributed ORAM + garbled circuits.
- **End-to-end per-query runtime (Deep1B-10M, 10M × 96-dim, single-thread):** linear-scan 375 s LAN / 1490 s WAN; clustering 30.1 s LAN / 181 s WAN. Communication: 5.53 GB (LAN linear) / 3.12 GB (clustering).
- **72-thread:** clustering drops to **4.23 s LAN / 28.4 s WAN** on Deep1B-10M. Linear-scan: 53.1 s / 214 s.
- **SIFT-1M:** 8.06 s LAN / 59.7 s WAN (clustering); 33.3 s / 139 s (linear).
- **Speedup over prior work on Deep1B-10M:** linear-scan 1.46× LAN / 3.51× WAN; clustering 18.5× LAN / 31× WAN; communication 4.1× to 39× reduction.
- **Approximate vs exact top-k (1M values, k=100, δ=0.01):** 12.0 s LAN / 113 s WAN approximate vs 301 s / 4130 s exact — **25×/37× speedup**.
- Top-k circuit is the critical primitive for oblivious reranking.

**Revisiting Oblivious Top-k Selection (Cong, Geelen, Kang, Park 2025, SAC 2024 / LNCS).** Updated oblivious top-k over FHE (BGV/BFV/TFHE) with improved constants for FHE-native evaluation. Applied to secure k-NN classification. Targets the "scores already encrypted under FHE, no MPC" regime — complementary to SANNS (which assumes secret-shared inputs). Specific wall-clock numbers are reported only in the FHE.org 2024 / SAC slides, not in abstract; [UNCLEAR] — needs full PDF fetch. Main algorithmic improvement: lower multiplicative depth for oblivious sort-network top-k under wordwise FHE.

**Multiple Millionaires' Problem (Tassa & Yanai 2024, PoPETs).** Secure max/argmax over N shared inputs — the building block for secure top-k winner selection. **No wall-clock latency numbers** — the paper reports circuit *size* and *round depth* only. Summary of the 11 protocols:
- Naive binary-tree (Protocol 1): depth 5·⌈log N⌉+4, size O(NB+NK)
- Digit-decomposition (Protocol 8, B=8, d=8): depth 26, size O(NBK·2^d/d) — constant depth regardless of N
- Monotone-representation (Protocol 6): depth 9, size (2^B)·NK — best for small domains (B=8)
- Binary search (Protocol 11, Aggarwal-inspired): depth 8B+4, size O(NB²+NBK)
- Tradeoff: Protocol 6 wins when domain is small (B=8); Protocol 8 wins for larger domains (B=32); Protocol 1 wins on bandwidth-constrained links.
- N = number of inputs (up to 2^32 in plots), K = number of MPC parties, B = bit-length.
- Compared online depth: with K=3, B=8, Protocol 6 has depth 9 — shallowest; Protocol 1 grows to 160+ for N=2^32.

**Panther (Li et al. 2025).** Secret-sharing + garbled circuits `ExactTopK` and `ApproxTopK`. Designed for retrieval but applicable here. Heavier than SANNS/p²RAG.

**Privacy.** Cryptographic — no TEE assumption. Depends only on non-collusion (usually 2 servers).

**Tradeoffs.**
- **Right choice** when reranking is delegated to an *untrusted* server and no TEE is available.
- **Overkill** when reranker runs client-side or in a trusted TEE (the ranking is already hidden from the server by virtue of the executor being trusted).
- Communication and latency: p²RAG 3–300× PRAG depending on k; SANNS ~seconds for 1M entries. Still 1–2 orders of magnitude slower than plaintext cross-encoder.
- No current MPC system runs a *cross-encoder transformer*; the reranker under MPC is a cosine/dot-product scorer over (possibly refined) embeddings, not a BERT-class model. Cross-encoder under MPC is open research.

---

### 5. Homomorphic-Encryption Reranking (Tiptoe, SealPIR-style)

**What it does.** Client encrypts query under LHE/FHE; server computes scores (inner products, possibly a small neural head) homomorphically; client decrypts and sorts.

**Key systems:**

**Tiptoe (Henzinger, Dauterman, Corrigan-Gibbs, Zeldovich 2023, SOSP).** Private web search over 360M pages using LHE-based private nearest-neighbor search. No TEEs, no non-colluding servers — pure cryptography. Achieves:
- 2.7s end-to-end latency
- 145 core-seconds server compute
- 56.9 MiB client-server communication (74% offline hint)
- MS MARCO rank quality: 7.7 (vs 2.3 for best non-private neural; 6.7 for classical tf-idf)

Tiptoe reduces private full-text search to private NN-search via semantic embeddings — semantically the same primitive we need for private reranking. Its *quality ceiling* is the cosine-score bi-encoder baseline; no cross-encoder precision gain.

**TRSE (Yu et al. 2013, IEEE TDSC).** Two-round searchable encryption with vector space + HE. Client receives encrypted scores, decrypts locally, does final top-k. Same pattern as client-side embedding re-scoring but over keyword vectors and OPE replacement. 178 citations — foundational. **Latency numbers are dataset-dependent and not consistently reported in the abstract**; follow-up work cites TRSE's round-trip overhead as the bottleneck (one RTT per query plus client-side Paillier decryption of top-k' scores — on order of hundreds of ms for k'=100). [UNCLEAR on exact TRSE wall-clock — needs full-paper fetch.]

**SANNS (above)** uses LHE for the scoring phase and MPC for top-k — see concrete SANNS numbers in Approach 4.

**Tradeoffs.**
- Pure cryptography → no TEE trust assumption.
- Scoring model is limited: inner product / low-degree polynomial heads only. A BERT cross-encoder under FHE is not practical.
- Score magnitudes leak if returned in plaintext post-decryption; need oblivious top-k on the client side if the threat model also distrusts the client endpoint.
- Best fit: private *retrieval* with bi-encoder quality, combined with Approach 1 or 2 for client-side rerank refinement.

---

### 6. Oblivious Top-k Selection and Access-Pattern Hiding

**What it does.** Even with scores hidden, the *selection* of the final k out of k' can leak (via prompt composition, access patterns, or subsequent retrieval). Oblivious top-k explicitly hides which k were chosen.

**Key systems:**

**Oblix (Mishra, Poddar, Chen, Chiesa, Popa 2018, IEEE S&P).** Oblivious search index via **doubly-oblivious data structures** (internal + external memory accesses both hidden). Built on SGX; designed for encrypted search without revealing access patterns. Concrete numbers:
- **Per-node lookup:** ~0.54 ms per tree node, **4.5–6.5× faster than ZeroTrace** (prior SGX baseline).
- **Signal private contact discovery (N=128M users, m=1 contact):** 591 ms Oblix vs 835 ms Signal (30% faster). For incremental lookup (m=1, N=128M): **5.9 ms Oblix vs 832 ms Signal — ~140× faster**.
- **Signal N=10⁹ users, m=1000 contacts:** 7.4 s Oblix vs 6.7 s Signal — at scale the two converge.
- **Key Transparency anonymous lookup (N=40M keys):** 2.3 s Oblix vs 4.6 s baseline (2× faster); for **N=320M: 2.6 s vs 37 s — 14× faster**.
- **Oblivious SE on Enron (528K emails, ~259K keywords, ~38M key-value pairs):** search avg 20.1 ms for top-10 results on the highest-frequency keyword (~145K docs); insert 7.75 ms per keyword.
- Directly applicable to oblivious retrieval of the k' candidates *and* final k.

**Snoopy (Dauterman et al. 2021, SOSP).** Scalable oblivious object store; **13.7× higher throughput than Obladi**.
- **Obladi:** 6.7K req/s on 2M × 160-byte objects, cannot scale beyond a single proxy+server.
- **Snoopy:** 92K req/s on 2M × 160-byte objects using 18 machines, average latency **under 500 ms**.
- Scales with machine count (unlike Obladi). Useful building block for holding k' decrypted chunks obliviously before reranking when the reranker runs on a cloud-hosted oblivious key-value layer.

**Metal (Chen & Popa 2020, NDSS).** Metadata-hiding file sharing — conceptually adjacent for hiding which file/chunk was selected.

**SANNS top-k circuit (above).** Provides oblivious top-k selection over `n` numbers with `O(n + k²)` comparators; intended for the MPC setting but adaptable.

**Tradeoffs.**
- Genuine privacy gain **only when the scorer is untrusted**. If scoring happens in TEE or on client, the trusted party already knows the ranking and oblivious selection adds no privacy there. Relevance: the *output* of reranking (the k chosen chunks) may still be observed by downstream components (e.g., the generation step's prompt assembly) — oblivious top-k inside TEE prevents this when generation happens outside the TEE.
- Overhead: `O(log n)` to `O(n)` accesses depending on structure; fixed-size volume hiding for access patterns.

---

### 7. Late Interaction on Encrypted Token Embeddings (ColBERT-style, Speculative)

**Baseline primitive (plaintext):** ColBERT (Khattab & Zaharia, SIGIR 2020, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT") scores `(q, d)` as `Σ_{j=1..|q|} max_{i=1..|d|} cos(q_j, d_i)` over per-token query × document embeddings. ColBERTv2 (Santhanam et al., NAACL 2022) adds residual-vector compression to ~6 bits per token. Per-token dim is typically 128. A 512-token chunk costs ~128 KB stored (fp16) vs ~1.5 KB for a bi-encoder single-vector representation — ~100× inflation at the storage layer.

**Correct framing of what would be hidden.** In the current Approach-4 pipeline, chunks are stored under two encryption layers: **chunk text is AES-encrypted** (step 1), and **embedding vectors are CAPRISE-encrypted** (step 3). A ColBERT-style private rerank would extend CAPRISE to a **per-token representation** — one CAPRISE-encrypted 128-dim vector per token instead of one per chunk. MaxSim would run directly on those encrypted token vectors. No chunk text is ever AES-decrypted at rerank time; AES decryption is deferred to the generation step for only the k selected chunks.

**Why this is attractive (compared to Approach 3's TEE cross-encoder).** A cross-encoder reranker in a TEE must AES-decrypt the k' candidate chunks' *text* to feed them into BERT — the TEE holds the AES key during rerank and the plaintext text transits enclave memory. A ColBERT-on-CAPRISE-tokens scheme keeps chunk text AES-encrypted throughout rerank and shrinks the rerank-time key-holding footprint to CAPRISE keys only. It also does not require the reranker host to be a TEE at all — a plain cloud server could compute on CAPRISE ciphertexts.

| | Approach 3 (TEE cross-encoder) | Approach 7 (ColBERT on encrypted tokens, speculative) |
|---|---|---|
| **AES key at rerank** | TEE holds it; decrypts k' chunks' text | Not needed; chunk text stays AES-encrypted |
| **Chunk-text plaintext exposure** | Inside TEE enclave memory for k' chunks | Never at rerank; only at generation for top-k |
| **TEE at rerank?** | Required | Not required — plain server OK |
| **Embedding storage overhead** | 1 CAPRISE vector/chunk (~1.5 KB at 768-d fp16) | ~100× — per-token CAPRISE matrix (~128 KB/chunk at ColBERTv2's 128-d) |

**CAPRISE compatibility (the hard part — why this is unsolved).** CAPRISE (Ye et al. 2026, ppRAG, "Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving Encryption", Section 3) is the Conditional variant of ADCPE (Hegde et al., S&P 2023, "Efficient Distance-Preserving Encryption for Nearest-Neighbor Search"). CAPRISE's formal guarantee:
- `cos(ENC_Q(q), ENC_DB(d_i)) ≈ cos(q, d_i)` — query-to-document distances **are** preserved.
- Inter-document distances `cos(ENC_DB(d_i), ENC_DB(d_j))` are **explicitly obfuscated** — this is the whole point of CAPRISE over plain ADCPE, to defeat the Vector Analysis Attack and token-inversion attacks like Vec2Text (Morris et al., EMNLP 2023).

ColBERT MaxSim requires the first property **per token pair**, *including across different documents*, because `max_{i in d_a} cos(q_j, d_a_i)` must be comparable to `max_{i in d_b} cos(q_j, d_b_i)` across documents `a, b`. CAPRISE as specified:
- Fresh per-vector noise: applying CAPRISE token-wise with independent noise draws per token sums `|q| · |d|` ≈ 32K noise terms per chunk, swamping signal at ColBERT's ~2 decimal digits of score resolution.
- Shared per-chunk noise: preserves intra-chunk token distances but leaks chunk-internal token structure to the server — a partial Vector Analysis Attack at token granularity, which token-level Vec2Text can likely invert.

Core tension: CAPRISE deliberately destroys inter-vector relationships to defeat structural attacks; ColBERT needs inter-vector (token) relationships to be queryable across chunks. The two requirements are mutually exclusive under CAPRISE's current construction.

**Candidate alternatives, with their problems:**

| Scheme | Source | MaxSim on encrypted tokens? | Blocker |
|---|---|---|---|
| **FHE (CKKS / BFV)** | Cheon et al. 2017; Fan-Vercauteren 2012 | Yes in theory | `max` is non-polynomial → needs comparison circuits → ~10–100 ms/comparison under CKKS. 32K comparisons per chunk → 30–300 s/chunk. Infeasible. |
| **Plain ADCPE** | Hegde et al. S&P 2023 | Yes — all-pairs preserved | Re-opens the Vector Analysis Attack that CAPRISE was designed to close. Lin et al. 2024 (*"Inversion Attack Against Obfuscated Embedding Matrix"*, already in Edgequake) recover 100% of tokens from glide-reflection obfuscation. |
| **LSH with trapdoor (Song et al., ESORICS 2023, "Secure Approximate Nearest Neighbor Search with Locality-Sensitive Hashing")** | Cited by Panther | Bucket-match, not true MaxSim | Collapses to coarse matches; no graded cosine; attacks on repeated queries. |
| **Function Secret Sharing MaxSim (p²RAG-style, per-token)** | Ming et al. 2026 | Yes | Blows up communication by `|q| · |d|` factor — at ColBERT granularity this is already above p²RAG's observed ceiling. |
| **ColBERTv2 centroid-ID scoring under FHE** | Santhanam et al. NAACL 2022 | Tractable (table lookup) | Centroid IDs are low-entropy token signatures; Vec2Text-class inversion likely works at that granularity. Needs empirical evaluation. |
| **Client-side MaxSim on CAPRISE-decrypted tokens** | — | N/A (not server-side) | Bandwidth cost: k'=50 × 128 KB = ~6 MB per query. Feasible for desktop, infeasible for mobile thin clients. Effectively "Approach 2 at token granularity". |

**Closest published adjacent work (all already indexed in Edgequake unless noted):**
- **Tiptoe** (Henzinger et al., SOSP 2023) — LHE private NN-search, sentence-granularity only. [NOT yet in Edgequake.]
- **Panther** (Li et al., CCS 2025) — MPC ANN single-server, dense-vector granularity, uses top-k circuit primitives adjacent to what would be needed.
- **GraSS** (Kim et al., Euro S&P 2025) — graph-based similarity search on encrypted queries via FHE; token-level extension unexplored.
- **OSNIP** (Cao et al. 2026) — null-space projection of token embeddings against a specific LLM's gradient; targets input privacy not reranker scoring, but the token-level framing is relevant.
- **CAPRISE / ppRAG** (Ye et al. 2026) — current baseline; does *not* extend to token granularity.
- **ADCPE** (Hegde, Wang, Nicolas, Kantarcioglu, S&P 2023) — the unconditional predecessor of CAPRISE; preserves all-pairs distances but falls to Vector Analysis Attacks.
- **Vec2Text** (Morris, Kuleshov, Shmatikov, Rush; EMNLP 2023) — the empirical bar any token-level encryption must survive.

**What a solution would need to provably satisfy:**
1. Preserve `cos(q_j, d_i)` for all query-token × doc-token pairs across any two documents.
2. Hide `cos(d_i^{(A)}, d_j^{(B)})` — cross-document token-pair similarity — to block Vector Analysis / Vec2Text.
3. Ideally also hide intra-chunk token-pair similarities; at minimum, chunks must be opaque to inversion.
4. Survive a token-level adaptation of Vec2Text, empirically.

I am not aware of any published symmetric encryption scheme satisfying (1) + (2) simultaneously at practical speed. Candidates in adjacent spaces (lattice-based predicate encryption, trapdoor-permuted LSH) are theoretical-only.

**Verdict.** Open problem; combines two requirements no current primitive meets. For Approach 4 near-term:
- **Short term:** don't rely on this. TEE cross-encoder (Approach 3) or client-side rerank (Approach 2) cover the space.
- **Medium term:** if bandwidth allows, the "client-side MaxSim on CAPRISE-decrypted tokens" hybrid gives ColBERT quality with zero cryptographic reranker on the server at the cost of ~6 MB/query — practical for desktop clients, not mobile.
- **Research direction:** formalize a symmetric scheme that preserves query-to-document token cosines but destroys document-to-document token cosines — "CAPRISE at token level with cross-chunk isolation". Likely either an impossibility result (Kornaropoulos et al. S&P 2019 style) or a construction via asymmetric random rotations per chunk with a shared query trapdoor. Unknown.

---

### 8. Privacy-Aware Server-Side Reranking with Controlled Leakage

**What it does.** Instead of full cryptographic hiding, structure the scoring pipeline so the server's view is provably limited — access-pattern bounding, bucket-tag shuffling, score masking.

**Key systems:**

**Privacy-aware document retrieval with two-level inverted indexing (Qiao, Ji, Wang, Shao, Yang 2023, *Information Retrieval Journal*).** Two-level index: posting records grouped into buckets with tags; runtime query produces query-specific tags to gather encoded features without revealing cross-posting sharing or per-document identity. Targets the server that processes unauthorized queries or inspects the index. Concrete tradeoff: some privacy ↔ efficiency ↔ relevance.

**Toward Secure Multikeyword Top-k Retrieval (Yu et al. 2013, IEEE TDSC).** Observes that OPE-based server-side ranking *inevitably leaks*. Proposes TRSE: vector-space scoring + HE so the server can produce encrypted scores only. 178 citations.

**Enabling Efficient Multi-Keyword Ranked Search (Li et al. 2014).** kNN + blind-storage access-pattern hiding over encrypted mobile cloud data.

**Blind Seer (Pappas et al. 2014, IEEE S&P).** Scalable private DBMS with provable leakage; supports Boolean queries and bounded access-pattern leakage.

**Tradeoffs.**
- Leakage is *bounded and characterized* rather than zero — acceptable when the TEE threat model already allows bounded leakage (e.g., K visible).
- Not competitive with cross-encoder quality — these systems rank by bag-of-words or bi-encoder scores, not BERT precision.

---

### 9. Split-Privacy Scopes (Public + Private) — SPIRAL / ConcurrentQA

**What it does.** Partition the corpus into privacy scopes; the reranker operates differently depending on where a candidate was retrieved from.

**Key paper.** Arora, Lewis, Fan, Kahn, Ré 2023, *TACL*. Defines the Split Iterative Retrieval (SPIRAL) problem: iterative retrieval over multiple distributions with distinct privacy constraints. Introduces the ConcurrentQA benchmark (public HotpotQA + private distribution). Shows SOTA retrievers degrade under this split and analyzes mitigations.

**Relevance.** When a RAG system mixes a private corpus with public web search (e.g., Tiptoe), the reranker must *not* leak which scope a candidate came from. SPIRAL formalizes this problem. No dedicated private reranker is proposed, but the threat model is directly relevant to mixed-scope Approach 4 deployments.

---

### 10. Lightweight On-Prem / Edge Reranking (practical deployment pattern)

Many enterprise systems bypass cryptographic privacy entirely by running reranking on-prem, treating it as an access-control problem rather than a cryptographic one.

**Key systems:**
- **HR+QDA** (Maringan & Fitrianah 2026) — covered above. Explicitly marketed for regulatory compliance and data sovereignty.
- **ConfidentialMind** (commercial) — on-prem AI platform with integrated RAG, reranking, RBAC, zero-trust APIs. Uses open-source models; no cryptographic privacy but full data-sovereignty stack.
- **NVIDIA Nemotron RAG rerankers** — production-grade rerankers shipped for local deployment on confidential-computing GPUs (H100 CC mode).
- **BGE-reranker-v2-m3 / Qwen3-Reranker / Jina-Reranker** — open-weight reranker models routinely used on-prem.

**Relevance.** These are the *baselines that Approach 4 competes against*. Any cryptographic approach must justify its overhead vs. the "just run it on trusted hardware" alternative.

---

## Comparison Matrix

| Approach | Runs where | Doc text decrypted where | Query privacy from reranker host | Top-k leakage to host | Cross-encoder quality | Latency (50 cand.) | Production ready |
|---|---|---|---|---|---|---|---|
| **1. Embedding re-scoring** | Client | Not needed | N/A (host = client) | N/A | None (DP artifact fix) | ~0ms | Yes (CAPRISE) |
| **2a. Client cross-encoder (BGE)** | Client | Client | N/A | N/A | Full | ~100–400ms | Yes |
| **2b. HR+QDA** | Client / on-prem | Client | N/A | N/A | ~91% of BGE | ~29ms | Yes (new) |
| **3. TEE cross-encoder (Opal)** | TEE | TEE | Attestation-protected | K visible | Full | ~100–400ms CPU | Research (Opal) |
| **3a. TEE+GPU split (Delta/AsymML/Shredder)** | TEE + GPU | TEE | TEE + DP | K visible | Full | ~10–50ms | Research |
| **4. MPC (p²RAG, PRAG, SANNS)** | 2 non-colluding servers | None (secret shared) | Cryptographic | Hidden | Cosine/bi-enc only | ~sec–min depending on k | p²RAG most practical |
| **5. LHE (Tiptoe)** | Server(s) | None (FHE) | Cryptographic | Leaked via score decryption | Cosine/bi-enc only | ~2–10s | Tiptoe production-scale |
| **6. Oblivious top-k (Oblix)** | SGX | SGX | SGX | Hidden | Whatever runs inside | + ORAM overhead | Oblix production |
| **7. ColBERT on encrypted tokens** | Server | None | Cryptographic | Hidden | Late-interaction | Fast (pre-indexed) | Unsolved |
| **8. Bounded-leakage SE (TRSE, two-level idx)** | Server | Partial (SE) | Bounded | Bounded | Keyword / bi-enc | ~ms | Mature (SSE) |
| **9. Multi-scope (SPIRAL)** | Mixed | Mixed | Scope-dependent | Scope leak possible | Depends | Depends | Benchmark only |
| **10. On-prem (HR+QDA, ConfidentialMind)** | On-prem server | On-prem | Not protected | Not protected | Full | ~30–400ms | Yes |

---

## Key Observations

1. **The field is richer than the first pass suggested.** Dedicated papers *do* exist for private retrieval at scale (PRAG, p²RAG, Tiptoe, SANNS). They target the *bi-encoder scoring + top-k selection* primitive, not cross-encoder BERT-style rerankers. The "no private reranker paper" claim from rev-1 should be narrowed to: *no paper implements a cryptographically-hidden cross-encoder transformer*.

2. **p²RAG (2026) is the strongest near-term building block for MPC rerank.** 3–300× faster than PRAG, supports arbitrary k, malicious-user defenses. If Approach 4 needs server-side rerank without trusting the server, p²RAG is the baseline.

3. **HR+QDA (2026) is the strongest non-crypto private rerank proposal.** 1.77M params, 13-min on-prem training, 91% of BGE quality at 157× compression. It reframes "private reranking" as "reranking small enough to run anywhere trusted" — a valid and very practical answer for enterprise deployment.

4. **Tiptoe (2023) proves end-to-end LHE private search is production-viable** at 360M-doc scale with 2.7s latency. Its rank quality (MS MARCO 7.7) is bi-encoder-class, matching classical tf-idf — confirming the crypto-based approach caps at pre-cross-encoder quality.

5. **Oblivious top-k as a building block is mature.** SANNS (2019) introduced the `O(n + k²)` circuit; Cong et al. (2025) refined it. Any MPC/TEE reranker can plug in this primitive for the final selection step without inventing new crypto.

6. **TEE + GPU asymmetric split (Delta, AsymML, 3LegRace, Shredder) is the only tractable way to run a full cross-encoder inside a privacy boundary.** Pure-TEE cross-encoder is slow (CPU-only) and memory-heavy (~567MB). The asymmetric split offloads linear-algebra-heavy transformer blocks to GPU with obfuscation / low-rank residuals; 7.6× to ~20× speedups reported.

7. **The "cross-encoder under MPC/FHE" problem is still open.** All practical MPC/FHE retrieval systems stop at cosine/bi-encoder scoring. Running BGE-reranker under MPC is not demonstrated anywhere. This is a real research gap — not just uninvestigated, but currently intractable.

8. **Client-side reranking (Approach 1/2) dominates by privacy coherence.** When the user's client already has `q_exact` and is receiving chunks, running the reranker locally is *both* the most private *and* the most practical option. The case for server-side private reranking only appears when:
   - The client is too thin to run the cross-encoder (mobile / browser without WASM tolerance) — then HR+QDA solves it.
   - The chunks must not leave the cloud (regulatory) — then TEE cross-encoder (Approach 3) is the answer.
   - The cloud itself is untrusted and no TEE is available — then MPC (Approach 4) is the answer.

9. **Access-pattern leakage from reranking is negligible on top of retrieval** in the Approach-4 threat model — the server already saw k' chunk IDs fetched. Which k of those feed the generator is a marginal signal, *unless* the generator is untrusted and observes a smaller prompt than k' — which Opal addresses by running generation inside the TEE and/or by padding prompt size.

10. **Score-based leakage must be plugged** whenever scores cross a privacy boundary in plaintext. Tiptoe decrypts scores on the client (safe); any TEE-computed score exported to a GPU in plaintext is a leak. Shredder-style noise on intermediate activations or score masking/rounding are the remedies.

---

## Recommended Papers for Edgequake Ingestion

This research pass yielded several papers that the first pass missed. Strongly recommend uploading:

1. **p²RAG: Privacy-Preserving RAG Service Supporting Arbitrary Top-k Retrieval** (Ming, Wang, Yang, Wang, Jia 2026). arXiv 2603.14778. Dedicated private RAG retrieval, directly applicable to reranking step. *Strongest pick.*
2. **PRAG: Distributed Private Similarity Search for LLMs** (Zyskind, South, Pentland 2024). ACL PrivateNLP 2024. First MPC RAG retrieval; sets up the baseline p²RAG improves on. arXiv 2311.12955.
3. **Tiptoe: Private Web Search** (Henzinger, Dauterman, Corrigan-Gibbs, Zeldovich 2023). SOSP 2023. Production-scale private NN-search via LHE; foundational for private retrieval.
4. **SANNS: Scaling Up Secure Approximate k-Nearest Neighbors Search** (Chen, Chillotti, Dong, Poburinnaya, Razenshteyn, Riazi 2019). arXiv 1904.02033. Introduces the `O(n+k²)` oblivious top-k circuit used as a building block across the field.
5. **HR+QDA: A Lightweight Privacy-Preserving Reranker** (Maringan & Fitrianah 2026). *Discover Computing* (Springer). Directly titled "private reranker." Defines the on-prem lightweight rerank pattern.

Secondary (optional):
6. **Oblix: An Efficient Oblivious Search Index** (Mishra, Poddar, Chen, Chiesa, Popa 2018). IEEE S&P 2018. Doubly-oblivious structures on SGX — relevant for access-pattern protection across Step 7–9 boundary.
7. **Revisiting Oblivious Top-k Selection with Applications to Secure k-NN Classification** (Cong, Geelen, Kang, Park 2025). LNCS. Most recent oblivious top-k improvement.
8. **All Rivers Run to the Sea / Delta: Private Learning with Asymmetric Flows** (Niu, Ali, Prakash, Avestimehr 2023). arXiv 2312.05264. TEE+GPU asymmetric split with formal DP; applicable to cross-encoder inference.
9. **Privacy-aware document retrieval with two-level inverted indexing** (Qiao, Ji, Wang, Shao, Yang 2023). *Information Retrieval J.* — structured server-side ranking with bounded leakage.
10. **Reasoning over Public and Private Data / SPIRAL / ConcurrentQA** (Arora, Lewis, Fan, Kahn, Ré 2023). *TACL* — mixed-scope retrieval threat model.

---

## Open Research Gaps (confirmed after this pass)

- **Cross-encoder transformer under MPC or FHE.** No published system. Current crypto-based retrieval tops out at bi-encoder/cosine quality.
- **ColBERT-style late interaction on CAPRISE-encrypted token embeddings.** Requires a distance-preserving symmetric encryption that covers the full query × doc token similarity matrix. No existing scheme does this efficiently.
- **Cross-encoder on TEE+GPU with formal privacy.** Delta/AsymML do this for training; no one has demonstrated it for a reranker with published numbers.
- **Oblivious cross-encoder scoring.** SANNS's oblivious top-k assumes scores already computed. An oblivious forward pass through a cross-encoder is absent.
- **Mixed-scope (SPIRAL) private reranking.** Benchmark exists; no reranker matches the threat model.

These are the research wedges for differentiation in an Approach 4–style system.
