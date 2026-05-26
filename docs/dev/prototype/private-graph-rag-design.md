---
type: prototype-note
status: current
created: 2026-05-16
updated: 2026-05-16
tags: [graph-rag, lightrag]
companion: [private-graph-search, private-rag-system-design]
---

# Private Graph RAG (LightRAG) — Research Delta + Design

> Research+design date: 2026-05-16. Companion to [`../../research/private-graph-search.md`](../../research/private-graph-search.md) (research baseline, 2026-04-23) and [`private-rag-system-design.md`](private-rag-system-design.md) (vector-only system design, 2026-04-16). Sources: arXiv, OpenAlex, EdgeQuake corpus, USENIX/IACR/ICLR proceedings, direct read of the LightRAG reference implementation (`HKUDS/LightRAG` `lightrag/operate.py` @ main, 5197 LOC).
>
> Scope: graph **search** (retrieval) only. Out of scope per instruction: entity/relation extraction (which is an LLM-inference problem, covered separately by [`gelo-llm.md`](../prototype/gelo-llm.md), [`dp-forward.md`](../prototype/dp-forward.md), and ObfuscaTune).

---

## 0. Definitions

- **LightRAG** — graph-augmented RAG system (Guo et al., EMNLP 2025, arXiv 2410.05779). Indexes a corpus as a KG of entities and relations, plus the original text chunks; each entity/relation carries a list of source chunk IDs.
- **VDB** — vector index (cosine ANN, typically HNSW under the hood). LightRAG ships three: `entities_vdb`, `relationships_vdb`, `chunks_vdb`.
- **KG** — knowledge graph: nodes (entities), edges (relations), with property bags.
- **HL / LL keywords** — high-level (global, thematic) and low-level (local, specific entity) keywords extracted from the query by the LLM. Drive the relations VDB and entities VDB respectively.
- **Doubly oblivious** — both the server's host OS and any external observer see access patterns that are computationally indistinguishable from random. ORAM hides patterns from one of those layers; doubly-oblivious schemes hide from both (Oblix, Opal, H₂O₂RAM).
- **EMM** — encrypted multi-map (key → list-of-values). The natural SSE primitive for an adjacency list.
- **Volume-hiding EMM** — EMM where the response length is independent of the queried key (FLASH/XorMM/Veil/PRT).
- **PIR** — Private Information Retrieval. Client gets DB[i] without revealing i.
- **DOC** — Differential Obliviousness (Chan-Shi). Relaxes ORAM from indistinguishability to (ε,δ)-DP over access patterns.
- **AP / QP / SP** — Access Pattern / Query Pattern / Search Pattern leakage (SSE literature).
- **CCS-25 Panther**, **OSDI-25 Compass**, **ICLR-25 Pacmann** — three 2025 private-ANN systems referenced throughout.
- **Source-id field** — LightRAG stores per-entity and per-relation a `GRAPH_FIELD_SEP`-delimited *list* of chunk IDs in a single string property (`operate.py:4502`); fan-out from KG node → text chunks goes through this list.

---

## 1. The LightRAG retrieval surface (from the reference implementation)

LightRAG retrieval is dispatched in `operate.py:kg_query` (line 3164) and the core search runs in `_perform_kg_search` (line 3573). Direct read of the source establishes the following protocol — concrete and finite, very different from the marketing description.

### 1.1. Indexed artifacts

Three vector indexes and three KV-style stores:

| Storage | Role | Key | Value |
|---|---|---|---|
| `entities_vdb` | cosine ANN | (embedding of) entity name + description | `entity_name` → similarity score |
| `relationships_vdb` | cosine ANN | (embedding of) per-relation "key" — concatenation of src/tgt names + relation description + any LLM-added "global" keywords | `(src_id, tgt_id)` → similarity score |
| `chunks_vdb` | cosine ANN (optional; only used in `mix` and `naive` modes) | embedding of the chunk text itself | `chunk_id` → similarity |
| `knowledge_graph_inst` (graph) | adjacency + properties | `entity_name`, `(src,tgt)` | node-props, edge-props |
| `text_chunks_db` (KV) | chunk text | `chunk_id` | `{content, file_path, ...}` |
| node/edge degree counters | structural | name / pair | int |

Crucially, each entity-node prop bag contains a `source_id` string which is a ``-delimited *list* of all chunk IDs the entity appeared in. Same for each edge. This is the fan-out the search uses to pull text chunks back through the graph.

### 1.2. The query-time protocol

For each query `q`:

```
1. (LLM) Extract  hl_keywords, ll_keywords ← keywords_extract(q)            # operate.py:3406
2. Embed 1–3 strings in one batch:                                            # operate.py:3637
   ll_emb ← Embed(ll_keywords)   [for local/hybrid/mix]
   hl_emb ← Embed(hl_keywords)   [for global/hybrid/mix]
   q_emb  ← Embed(q)             [for mix/naive]
3. Three independent VDB calls (parallel):                                    # operate.py:3658
   Hits_E  ← entities_vdb.query(ll_emb, top_k)            # ~20 entity names
   Hits_R  ← relationships_vdb.query(hl_emb, top_k)       # ~20 (src,tgt) pairs
   Hits_C  ← chunks_vdb.query(q_emb, top_k)               # ~60 chunk_ids, mix only
4. KV reads on the graph:                                                     # operate.py:4381
   props_E ← get_nodes_batch(Hits_E)            # entity prop bags
   deg_E   ← node_degrees_batch(Hits_E)
   props_R ← get_edges_batch(Hits_R)            # edge prop bags
   deg_R   ← edge_degrees_batch(Hits_R)
5. 1-hop expansion (local-path only):                                         # operate.py:4419
   adj      ← get_nodes_edges_batch(Hits_E)     # list-of-edges per node
   props_R' ← get_edges_batch(adj)              # properties of those edges
   deg_R'   ← edge_degrees_batch(adj)
   sort by (degree desc, weight desc)
6. Fan-out to chunks via embedded source_id lists:                            # operate.py:4500
   For each entity, parse `source_id` → chunk_id list of variable length
   Two selection methods:
     WEIGHT: linear-gradient weighted polling by occurrence count
     VECTOR: re-rank candidate chunk_ids by cosine(q_emb, chunk_emb)
   Same path for relations → relation_chunks
7. Token-budget round-robin merge of {vector_chunks, entity_chunks, relation_chunks}.
8. Batch chunk read:                                                          # operate.py:4612
   chunks ← text_chunks_db.get_by_ids(selected_chunk_ids)
9. Assemble context = entities_str ⨁ relations_str ⨁ chunks_str.
10. (LLM) Generate answer ← LLM(sys_prompt(context) ⨁ q).
```

There is **no multi-hop BFS**. The "graph" part of LightRAG's graph-RAG is precisely *one* hop of edge expansion, run only in the local path. The remaining "graph-ness" is the entity→chunks and relation→chunks fan-out via `source_id`. From a private-search standpoint this is *good news*: we do not need a private k-hop oracle, only private 1-hop neighbor fetch + private multi-map lookup.

### 1.3. What the server sees, by component

In plaintext today, the server learns, per query:

- The query embedding (for chunks_vdb) and the two keyword embeddings (for entities/relations VDB) — all of which Vec2Text-class attacks invert to readable text.
- Each VDB returns the chosen IDs in rank order → the server sees the entire top-20 hit list.
- The graph store sees: which entity names were looked up, the *degree* of each (a strong fingerprint), which entity → which edges (i.e., which neighborhoods are read), and which chunk IDs are pulled.
- The KV store sees the final selected chunk IDs.

This is a denser leakage surface than vector-only RAG: every query exposes a small *subgraph* of the KG, not just a list of chunk IDs. AGEA (arXiv 2601.14662) operationalizes this — at 1000 queries it reconstructs 96.4% of LightRAG's node set on Medical, 88% on Agriculture, with >98% precision.

---

## 2. Threat model — refinement over `private-graph-search.md`

We carry forward the four LightRAG-specific attack classes from the existing survey (Liu 2025 / AGEA / GRAGPOISON / LogicPoison). Additions since 2026-04:

### 2.1. New attacks / sharper attacks

| Work | Date | Class | Concrete |
|---|---|---|---|
| **[NEW]** "Anonymization Along the RAG Pipeline" (arXiv 2604.15958, Apr 2026) | 2026-04 | Defense audit | Empirical: anonymization applied to *raw data* before indexing is the only stage that materially attenuates downstream extraction; late-stage anonymization (at retrieval or response) does not survive AGEA-style probing. |
| **[NEW]** "Securing RAG: Taxonomy" (arXiv 2604.08304, Apr 2026) | 2026-04 | Survey | Classifies graph-RAG-specific attacks. Names DCPE / distance-preserving encryption as the practical defense floor; no novel scheme proposed. |
| **[NEW]** "DP-RAG: Differentially Private RAG" (Tang, Flemings, Wang, Annavaram — USC, arXiv 2602.14374, Feb 2026) | 2026-02 | Defense | DP-KSA (DP keyword sketch + augment): output-side DP guarantee on generated text w.r.t. the corpus. Compatible with any underlying retrieval backend. No ε reported in abstract; relies on propose-test-release. |

### 2.2. Threat surfaces to cover (LightRAG-specific)

Concretely, a private LightRAG must defend against — at minimum — these distinct adversaries on the retrieval surface (entity-extraction defenses are out of scope, but we note the interaction):

1. **Plaintext-embedding inversion** of the per-query embeddings (`ll_emb`, `hl_emb`, `q_emb`). Existing crypto path: CAPRISE-DPE storage + DistanceDP query noise.
2. **Search-pattern repetition** — identical or near-identical query embeddings across sessions are linkable. Even with DistanceDP, the cluster of perturbed queries reveals the underlying topic.
3. **Access-pattern correlation over the KG** — which entity IDs got looked up, which edges expanded, which chunks fetched. The graph topology of the access trace is itself a fingerprint of the query topic.
4. **Degree / volume leakage** — every node has a distinctive degree; `get_nodes_edges_batch` returns variable-length edge lists that pin down which exact entity was queried even if the ID is hidden. This is the LightRAG-specific failure mode.
5. **AGEA-class extraction** — the *content* of returned entities/relations/chunks, once decrypted by the client, can be exfiltrated. This is a defense problem at the *generation/output* layer (DP-RAG, ARoG/PrivGemo-style anonymization) and at the *query-authorization* layer, not at the crypto-retrieval layer.

This document focuses on (1)–(4). For (5) we explicitly compose with PrivGemo-style dual-LLM (semantic+structural anonymization on the way into the generator) and/or DP-RAG-style output DP — neither of which is a crypto primitive and both of which are orthogonal to the choice of private retrieval mechanism.

---

## 3. New primitives since 2026-04 — delta over `private-graph-search.md`

The 2026-04-23 survey already covers GORAM, Graphiti, GraphSC, TOGES, GraSS, OblivGNN, PeGraph, FBSSE-BG, ARoG, PrivGemo, Opal. The delta below adds, in decreasing order of design impact:

### 3.1. Compass — Encrypted Semantic Search with High Accuracy (OSDI 2025) [HIGH IMPACT]

- **Authors:** Jinhao Zhu, Liana Patel, Matei Zaharia, Raluca Ada Popa (UC Berkeley + Stanford). OSDI 2025; full ePrint 2024/1255.
- **Reference implementation:** [github.com/Clive2312/compass](https://github.com/Clive2312/compass) (C++, no license declared, USENIX Artifact Available + Functional + Reproduced badges, last updated 2026-04). Built on Ring-ORAM with HNSW init; ships `compass_init`, `test_compass_ring` (the reference client/server), plus `test_compass_accuracy` (intentionally leaks AP, for fast recall sweeps) and `test_compass_tp` (throughput harness). Datasets, indices, and pre-built client/server states hosted in a public GCS bucket `compass_osdi`.
- **What it actually does:** HNSW index stored *encrypted* on the server; an ORAM controller fetches HNSW nodes during traversal; three co-design tricks — *Directional Neighbor Filtering*, *Speculative Neighbor Prefetch*, and *Graph-Traversal-Tailored ORAM* — bring per-query latency to **sub-second on LAN, orders of magnitude faster than baselines**. Vector embeddings *and* the graph index are encrypted at rest; the ORAM controller hides which node is visited at each hop.
- **Why it is the headline finding for our design:** LightRAG's `entities_vdb` and `relationships_vdb` are *already* HNSW (the default in nano-vectordb and most LightRAG backends). Compass slots directly under both, with no quality degradation versus plaintext HNSW. This is strictly better than GraSS for our use case — same query-hiding goal, real wall-clock numbers, plaintext-quality recall.
- **Limits:** Single-server-trust model still requires the ORAM controller to be co-located inside a TEE or trusted client. ORAM-style bandwidth is O(log n); large `n` still ships KB-MB per query.
- **Integration cost:** Replace any open-source HNSW backend (`nano-vectordb`, `faiss`, `usearch`) with Compass's encrypted HNSW. Storage encrypted with the same CAPRISE-class key already used elsewhere in our stack. ORAM controller runs in the Embedding TEE (Approach 4) or on the client (Approach 1/3).

### 3.2. Pacmann — Efficient Private Approximate Nearest Neighbor Search (ICLR 2025) [HIGH IMPACT]

- **Authors:** Mingxun Zhou, Elaine Shi, Giulia Fanti (HKUST + CMU). ICLR 2025; ePrint 2024/1600.
- **Reference implementation:** [github.com/wuwuz/Pacmann](https://github.com/wuwuz/Pacmann) (Go, MIT license, last updated 2026-04). Built on PianoPIR (ePrint 2023/452) for the offline-preprocessed PIR layer, NGT and hnswgo for the graph-ANN layer. SIFT-1M / SIFT-1B reference scripts (`run-private-search.sh`, `run-ngt-search.sh`, `run-cluster-search.sh`); requires ~230 GB disk for the SIFT-1B download.
- **What it does:** Client runs graph-based ANN traversal *locally*, but fetches each visited node's local subgraph via batched PIR. Trades offline preprocessing (server pre-computes PIR hint tables) for online efficiency (sublinear comm per hop).
- **Reported result:** **2.5× better search accuracy** than prior single-server private ANN at matched comm budget; **reaches 90 %** of the quality of a non-private graph ANN. Public release adds **up to 62 % computation-time reduction and 22 % overall latency reduction at the 100 M-vector scale.**
- **Where it sits in our design space:** Pacmann is the natural fallback when no TEE is available — pure crypto, client-side traversal. It is the cleanest single-server-trust private-ANN scheme published. Slower than Compass (1-2 RTTs per hop) but no hardware-trust assumption.
- **Limits:** Client must hold the high-level navigable graph structure (entry points + a small cached prefix). PIR hints regenerated periodically as the corpus changes (similar to Tiptoe's preprocessing).

### 3.3. PRAG — End-to-End Privacy-Preserving RAG (arXiv 2604.26525, Apr 2026)

- **Authors:** Li, Xu, Qi, Yu, Zhang, Zhang, Shang, Ma, Cheng (Shandong U + collaborators).
- **What it does:** HE-based ranked retrieval over encrypted vector indexes, with two modes — PRAG-I (non-interactive HE-only) and PRAG-II (with client assistance for the ranking step). Novelty: *Operation-Error Estimation* (OEE) — stabilizes ranking under accumulated CKKS noise, so the top-k order remains correct after several homomorphic multiplications.
- **Headline number from abstract:** **recall@10 of 72–74 %** at "practical retrieval latency"; the abstract also claims resilience to graph-reconstruction attacks but does not quantify.
- **Why interesting but not picked as primary:** OEE is a genuine contribution — ranking-order stability under FHE noise is the operationally hardest part of FHE-vector-search. But absolute latency / dataset scale are not verified in our verification fetch, and PRAG targets the vector-RAG setting (no graph-specific design). At best, PRAG-I supplements Compass for clients that cannot run an ORAM controller.

### 3.4. ZKGraph — Zero-Knowledge Verifiable Graph Query Evaluation (arXiv 2507.00427, Jul 2025) [LATER PHASE]

- **What it does:** PLONKish ZK circuits for *graph query correctness* — decomposes graph queries into expansion-centric operators (neighborhood, shortest-path-like), each a primitive ZK gadget. The prover is the database owner; the verifier (client) confirms the server returned the correct k-hop neighborhood given a committed graph snapshot.
- **Why it matters for us:** It is the first published ZK scheme that proves *graph traversal* correctness specifically. Provides the "verifiable" leg of the integrity story under GRAGPOISON / LogicPoison attacks: client can verify the server didn't substitute the corpus mid-query.
- **Caveat:** No proof-size / verification-time numbers in the abstract. Treat as research-stage. Track for a v2 integrity layer; do not adopt for the v1 design.

### 3.5. Volume-hiding EMMs — FLASH, XorMM, Veil [DIRECT FIT for adjacency-list fetch]

- **FLASH** (IEEE TDSC 2024) — conjunctive volume-hiding EMM. 2-3× storage overhead. Optimal asymptotic comm. Hides query-equality, access pattern, response-set size for multi-predicate adjacency queries.
- **XorMM** (CCS 2022) — non-lossy volume-hiding EMM. 1.5-2× storage. Optimal comm. The most direct candidate for `get_nodes_edges_batch` since it returns *exactly* the response set with no padding loss.
- **Veil** (SIGMOD 2023) — overlapping-bucket volume-hiding. 1.2-1.8× storage (tunable). Reduces dummy-record overhead at the cost of a small computational-indistinguishability gap vs. perfect volume hiding.

These solve LightRAG's distinctive failure mode: variable-length adjacency lists fingerprint entities even when entity IDs are encrypted. **All three are mature SSE constructions.** Default: XorMM for the adjacency-list and source-id multi-maps (small constant overhead, no truncation); FLASH if conjunctive filters (type=PERSON ∧ degree<k) are later required.

### 3.6. H₂O₂RAM — Doubly Oblivious RAM in TEE (Sep 2024) [DROP-IN UPGRADE for Opal]

- **What it does:** Doubly-oblivious RAM running inside a TEE; ~1000× faster than prior O₂RAM constructions, 5-44× less memory.
- **Why it matters:** Opal's TEE-backed access-pattern hiding is the closest production-shape blueprint for private GraphRAG. H₂O₂RAM is the natural under-the-hood substrate. If we adopt the Opal architecture, swap Path-ORAM-in-TEE for H₂O₂RAM.

### 3.7. CryptGNN — Secure GNN inference (CCS 2025) [PERIPHERAL]

- Removes the non-colluding-third-party assumption from MPC-GNN inference. Relevant if we later add a GNN-reasoner step (GNN-RAG-style). Not on the v1 path.

### 3.8. DP-KSA / DP-RAG (arXiv 2602.14374, Feb 2026) [ORTHOGONAL DEFENSE]

- Output-side DP. Composes orthogonally with any retrieval crypto. Adopt as the *answer-layer* defense against AGEA-style extraction. Does not interact with the choice of retrieval primitive.

### 3.9. Federated graph querying (Springer 2025, "Fast and Secure Multiparty Querying") [DEFERRED]

- MPC for vertically-partitioned KG queries. Out of scope for v1 single-tenant LightRAG; relevant if/when multi-tenant cross-silo KG-RAG becomes a target.

---

## 4. Per-step design — mapping LightRAG to private primitives

For each of the eight retrieval sub-operations identified in §1.2, the per-step protocol below shows: (a) what is private, (b) which primitive realizes it, and (c) the next-best alternative.

### Step 1 — Keyword extraction (LLM call)

Out of scope (LLM inference). Handled by the chosen generation-privacy layer (ObfuscaTune / GELO / TEE inference). For the retrieval layer this is a black-box that emits two *short* strings.

### Step 2 — Embedding (`Embed(...)`)

Inherited from `private-rag-system-design.md`:
- Approach 1: client-side local model. Zero leakage.
- Approach 3 / 4: TEE-side embedding (Embedding TEE per `private-rag-system-design.md` §Approach 4). Inputs over RATLS; output never leaves the trust boundary in plaintext.

LightRAG additionally embeds *two* short keyword strings per query. Same primitive, same trust boundary, batched in one call (`operate.py:3637`). No new design needed.

### Step 3 — Three parallel VDB calls (`entities_vdb`, `relationships_vdb`, `chunks_vdb`)

This is the largest design choice. Three options, ranked by recommendation:

**(A) Compass-style encrypted HNSW under an ORAM controller in TEE — recommended default.**

- Server stores three HNSW indexes encrypted with a CAPRISE-class symmetric key.
- ORAM controller (Path-ORAM or H₂O₂RAM if available) runs inside the Embedding TEE — co-locating the ANN traversal logic with the embedding endpoint avoids an extra round trip.
- Compass's *Directional Neighbor Filtering* + *Speculative Prefetch* are HNSW-traversal-specific; they survive transparently because LightRAG's vector store is HNSW under the hood.
- Per-query cost: ≈ 1 s LAN per VDB (so ≈ 3 s for the three indexes; if needed, the three Compass traversals run in parallel since they share nothing).
- What the server sees: random-looking ORAM accesses; no information about which HNSW nodes were visited.

**(B) Pacmann (PIR + client-side traversal) — for the no-TEE variant of Approach 1.**

- Client downloads the high-level HNSW graph skeleton (entry points + a small upper-level prefix) at session setup. Server holds the encrypted lower-level subgraphs.
- For each hop, client batches a PIR query for the subgraph of the current frontier; server returns oblivious responses.
- Trades ≈ 2× online latency vs. Compass for zero hardware-trust assumption.

**(C) CAPRISE + RemoteRAG DistanceDP — the *lightweight* path already in our stack.**

- The vector indexes remain CAPRISE-encrypted exactly as in `private-rag-system-design.md` Approaches 1/3/4. Query embeddings perturbed by `(ε, δ)`-DistanceDP at submission.
- This *does not* hide access patterns — it only hides the query content (probabilistically) and the precise embedding value. The server still sees the top-k IDs in plaintext.
- We adopt this as the **floor**: even if Compass/Pacmann are deferred to a later phase, CAPRISE+DistanceDP gives a coherent storage-layer privacy story today.

**Recommendation:** Compass under TEE (A) for the default, with CAPRISE+DistanceDP (C) as the v0 simplification. Pacmann (B) only if a no-TEE deployment is explicitly required.

### Step 4 — KV reads on the graph (`get_nodes_batch`, `get_edges_batch`, `*_degrees_batch`)

The hit lists from Step 3 are small (top-20 per VDB). What we need is a private batch-fetch from a key→value store *whose access pattern is also hidden*.

- **Primary:** Same ORAM controller as Step 3 backs the node and edge KV stores. With the controller already inside the TEE, this is just another oblivious read against a different encrypted dictionary. No extra crypto primitive.
- **Lighter alternative (no ORAM):** XorMM volume-hiding EMM for node-props and edge-props (the "key→value" case is just an EMM with single-element values + padding). 1.5-2× storage overhead. Search pattern still leaks (same key queried twice is observable); pair with a per-session pseudonymization layer (HMAC-keyed key transformation, rotated per session à la PrivGemo) to break cross-session linkability.

For the **degree counters**, the safest treatment is to *fold them into the node/edge property blob* (encrypted, fetched by the same primitive). Avoid exposing degrees as a separate keyed store — a separate degree-lookup table is a trivially-correlatable side channel.

### Step 5 — 1-hop expansion (`get_nodes_edges_batch`) — the LightRAG-specific failure mode

This is the primitive that does not exist cleanly in the vector-RAG stack: *given a set of entity IDs, return the edge list incident to each*. Naïve EMMs leak the degree of each node.

- **Primary:** XorMM volume-hiding EMM with the entity name as key and the *padded* edge list as value. Padding policy: bucket nodes by degree-quantile (e.g., five buckets at 50/75/90/95/99 percentile); within a bucket, pad every response to the bucket's max degree. Volume leakage reduces from "per-node degree" to "degree-bucket of the queried node."
- **Stronger alternative (recommended at small scale):** Run the adjacency-list fetch inside the same ORAM-on-TEE controller as Step 4. Bandwidth is then O(log |V|) per node regardless of degree. Comm cost moves from "max-degree padded" to "log-of-graph-size" — preferable on KGs where the degree distribution is heavy-tailed.
- **GORAM** (PVLDB 2025) is the obvious third candidate for this step (its raison d'être is exactly ego-centric queries). However, GORAM is a 3-party MPC scheme — it lifts the trust assumption from "one server + TEE" to "three non-colluding servers." For a single-tenant cloud RAG service this is operationally heavier than ORAM-on-TEE. Keep GORAM in reserve for the federated/multi-silo variant (cf. §3.9).

### Step 6 — Fan-out from KG node → chunk ID list (`source_id` multi-map)

LightRAG stores `source_id` as an inline string on the node/edge prop bag (`operate.py:4502`). Length leaks: a 5-chunk entity is distinguishable from a 50-chunk entity once the prop bag is decrypted client-side. Two fixes:

- **Server-side:** lift `source_id` out of the prop bag into its own EMM (entity_name → chunk_id list). Apply the same XorMM volume-hiding EMM treatment as the adjacency list. This *also* avoids re-parsing the GRAPH_FIELD_SEP-delimited string on the client.
- **Client-side (cheaper):** keep the inline `source_id` but pad it to a per-bucket max length inside the encrypted prop bag at indexing time. The client then sees a *fixed-length* chunk list per node, with sentinel chunk IDs at the tail. Less elegant than the EMM but a one-line change at indexing.

### Step 7 — Token-budget round-robin merge

Pure client-side computation. No server involvement. No new primitive.

### Step 8 — Final chunk batch read (`text_chunks_db.get_by_ids`)

Identical to vector-RAG; covered in `private-rag-system-design.md`:
- Approach 1/3: AES-encrypted chunks on storage server; client fetches by ID and decrypts.
- Approach 4: same, fetched into the TEE if the TEE is doing reranking.

The set of IDs fetched still leaks under naïve KV access. Either:
- Send the fetch through the same ORAM controller as Steps 3-6 (uniform story); or
- Use PIR for the final chunk fetch (this matches the PIR-RAG / RAGtime-PIANO / Panther tradition — chunk DB is the natural PIR target since chunks are large fixed-blob entries).

---

## 5. Trust-model variants

LightRAG-private fits inside two of the four trust models already enumerated in `private-rag-system-design.md`. We do not introduce a new top-level approach — graph search adds primitives *inside* the existing trust boundary.

### 5.0. Who plays Compass's "client" role

This is the design decision that distinguishes the two variants below. Compass's paper defines a protocol with a **client** that holds the position map, stash, treetop cache, and ORAM key, and an untrusted **server** that holds the encrypted ORAM tree. The paper's deployment puts the client on the **user's own device** (laptop / phone / browser) — explicitly avoiding TEEs because TEE side channels weaken the story.

What Compass actually proves is a property of the *protocol*: an adversary holding only the ORAM-tree storage cannot distinguish two access traces of equal public structure. That property is independent of which silicon hosts the client role. The role can be played by any party that holds the ORAM client state confidentially, generates path IDs from a private PRP, and is reachable by the user over an authenticated channel.

| Compass client lives on… | User's device (paper-faithful) | SEV-SNP CVM (this design) |
|---|---|---|
| Trust anchor | user owns the hardware | AMD SEV-SNP attestation + vendor |
| ORAM state location | device RAM | encrypted CVM RAM (hidden from host OS) |
| Side-channel exposure | device-local | SEV-SNP side channels in-scope |
| State portability | device-local; per-device sync | centralised; users switch freely |
| Multi-tenant | one user per device | many tenants per CVM, isolated by HKDF |
| User-facing client | thick (5–500 MB ORAM state, full protocol code) | thin (RATLS + plaintext requests only) |

Opal (cited in §3.5 and §C.3) already runs Compass inside a TEE and formally imports Compass's batched-access lemma into its `G_att`-hybrid security proof. The composition is recognised. We adopt it here for the same reason: the thin-client / multi-tenant SaaS shape of Approach 4 cannot put 5–500 MB of state and an ORAM controller on every user's browser.

The two variants below are not "primary vs fallback" — they are **two equally valid Compass deployments with different trust anchors**, chosen by who can or cannot accept SEV-SNP trust.

### 5.1. Variant A — Compass client inside a SEV-SNP CVM (extension of Approach 4)

The SEV-SNP CVM plays Compass's "client" role. The Embedding TEE that already holds the embedding model also hosts the Compass ORAM controller and the XorMM EMM endpoints. User's device is a thin client connected over RATLS; everything Compass calls "client work" runs inside the CVM.

```
[Client (thin)]──RATLS──▶[ TEE: Embed + Keyword-LLM + Compass ORAM + EMM endpoints ]
                              │
                              ├──encrypted ANN traversal──▶[Storage: encrypted HNSW]
                              ├──encrypted EMM lookups   ──▶[Storage: XorMM adjacency, source_id EMMs]
                              └──encrypted KV reads      ──▶[Storage: encrypted node/edge props, AES chunks]
```

What the storage server learns:
- That a query happened.
- ORAM-mediated random-looking access patterns. Computationally indistinguishable from random under standard tree-ORAM analysis.
- Volume-hiding EMM accesses with bucket-level degree leakage (Step 5/6) or O(log|V|) padded bandwidth if pushed through ORAM.

What the TEE learns:
- Plaintext query, plaintext intermediate context, plaintext final prompt. Bounded by attestation + RATLS lifecycle.

TCB: TEE hardware vendor + Compass+EMM+ORAM implementation + attestation correctness.

### 5.2. Variant B — Paper-faithful Compass on the user's device (extension of Approach 1)

Compass's "client" role runs on the user's own device exactly as the paper describes; no TEE anywhere. The right choice when SEV-SNP trust is not acceptable (regulated finance, defence, supply-chain-paranoid deployments) or when the user already has a thick client (developer workstation, on-prem appliance).

```
[Client: Embed + Keyword-LLM + Pacmann client + EMM client]
   │
   ├──Pacmann PIR for HNSW subgraphs ──▶[Storage: encrypted HNSW + PIR hints]
   ├──Volume-hiding EMM queries      ──▶[Storage: XorMM adjacency, source_id EMMs]
   └──PIR / SS chunk fetch           ──▶[Storage: encrypted chunks (PIR-friendly format)]
```

What the storage server learns:
- Per-query PIR access pattern over the encrypted HNSW lower-level subgraphs — information-theoretically indistinguishable under SimplePIR/PIANO.
- Volume-hiding EMM accesses — degree-bucket leakage at worst.
- PIR chunk fetches — single-DB-row access pattern hidden.

Latency: dominated by PIR; estimated 5-15 s per query at 10⁵-10⁶ entities under SimplePIR-class schemes (consistent with Tiptoe at much larger scale).

Client requirements: device must hold per-tenant ORAM state (5 MB for LAION-scale up to ~500 MB for MS-MARCO-scale, per Compass Tab. 4) plus run the embedder. Per-device state sync is the user's problem; switching devices requires re-attesting and re-downloading the ORAM client state.

TCB: no hardware trust. Crypto assumptions only.

### 5.3. What is *not* a separate variant

Per the existing system-design doc, the **all-TEE** Approach 2 trivially supports graph search: standard plaintext LightRAG inside a CVM. No private-retrieval design work needed. The interesting design space is exactly Variants A and B above.

---

## 6. End-to-end leakage profile (Variant A)

| Stage | What the storage server learns | What the TEE learns | What an external observer learns |
|---|---|---|---|
| Embed | nothing | plaintext query + keywords | encrypted RATLS traffic |
| `entities_vdb` traversal | ORAM-random reads on HNSW | top-20 entity_name candidates | nothing |
| `relationships_vdb` traversal | ORAM-random reads | top-20 (src,tgt) candidates | nothing |
| `chunks_vdb` traversal (mix mode) | ORAM-random reads | top-60 chunk_ids | nothing |
| node/edge KV reads | ORAM-random reads | plaintext prop bags | nothing |
| 1-hop adjacency fetch | volume-hidden EMM access (degree-bucket leak) **or** ORAM (no leak) | plaintext edge list | nothing |
| source_id fan-out | same as above | plaintext chunk_id list | nothing |
| chunk fetch | ORAM-random reads | plaintext chunk content | nothing |
| LLM generation | nothing (if local or TEE) | plaintext prompt + answer | encrypted RATLS to client |

Residual leakage relative to a perfect ideal (cleartext-only-inside-client):

1. **Degree-bucket leakage on the adjacency EMM** if the XorMM variant is chosen over ORAM. Quantifiable: equivalent to disclosing the degree quantile of each visited node. AGEA's reconstruction precision drops measurably when degrees are bucketed (no published number for this specific countermeasure; clearly a worthwhile follow-on experiment).
2. **Cross-session linkability** if no per-session pseudonymization is added to the EMM/KV keys. Mitigated by PrivGemo-style per-session HMAC rotation: `key' = HMAC(session_secret, entity_name)`.
3. **Query-volume leakage** — observer can count requests-per-session. Mitigated by request-padding at the RATLS layer (constant request rate per session).
4. **Generation-time extraction (AGEA)** — once the TEE assembles the plaintext context, an adversary running AGEA-class queries through the legitimate API still extracts the KG over many sessions. Defended in the *prompt* layer with PrivGemo's anonymized-subgraph approach and/or DP-KSA — orthogonal to this design.

---

## 7. Open gaps and known risks

1. **Compass + Embedding TEE co-location.** Compass's ORAM controller plus three HNSW indexes plus the embedding model must all fit in the same TEE; total memory is comfortably under a TDX VM cap (Compass ORAM state is O(|V| log |V|) bits of position-map state; embedding model ~150 MB; HNSW graphs are the bulk and live encrypted on storage), but the integration is not published. Engineering risk, not a design risk.
2. **XorMM at LightRAG scale.** XorMM and FLASH are evaluated at ≤ 10⁷ keys in the source papers. LightRAG KGs in our target range (10⁴-10⁶ entities, 10⁵-10⁷ chunks) fit comfortably; nothing larger has been benchmarked.
3. **Updates.** XorMM, Compass HNSW, and ORAM all need re-preprocessing when the corpus changes. LightRAG's incremental update path (`operate.py:rebuild_knowledge_from_chunks`, line 560) is natively incremental; preserving that under encryption requires either dynamic SSE (forward+backward secure) or batched re-builds. Match the LightRAG ingest cadence: rebuild EMM/ORAM once per N inserts; live additions buffered in a small "fresh" tier searched in parallel.
4. **Pacmann at corpus size.** Pacmann's quality numbers come from 1M-scale evaluations; LightRAG KGs are typically smaller, which is favorable. Pacmann's PIR hints must be regenerated on inserts; treat like Compass.
5. **The ZK-integrity gap remains open.** Variant A's TEE attestation proves the *code* is correct; ZKGraph could close the *result-integrity* gap (against malicious cloud + side-channel-compromised TEE). No production-grade implementation yet; track for v2.
6. **Volume-hiding under conjunctive metadata filters.** LightRAG occasionally supports query-time entity-type filters (e.g., only PERSON entities). Conjunctive volume-hiding (FLASH) handles this; XorMM does not. If the metadata-filter feature is needed, switch the EMM to FLASH.
7. **The generation-side defense gap.** AGEA still works against a *retrieval-private* system that emits plaintext context to a plaintext LLM. The PrivGemo dual-LLM + DP-RAG output-DP composition is the recommended out-of-scope companion; without it, retrieval privacy is necessary but not sufficient.

---

## 8. Recommended implementation sequence

Tied to the existing prototype roadmap in `docs/prototype/`:

**Phase 0 — Baseline private LightRAG (no new primitives, fastest to ship)**

- Run plain LightRAG inside the existing CAPRISE-encrypted storage + DistanceDP query path of Approach 1/3/4. Three VDBs, both KV stores, the chunk store — all CAPRISE/AES at rest, DistanceDP on `ll_emb`/`hl_emb`/`q_emb`.
- This gives storage-layer privacy and query-content privacy. Access patterns leak. Acceptable for low-sensitivity tenants.
- Effort: ~1 week. No new crypto primitives — just compose existing parts with LightRAG's `kg_query` entry point.

**Phase 1 — Access-pattern hiding for the three VDBs (Compass under TEE)**

- Drop in Compass as the HNSW backend for `entities_vdb`, `relationships_vdb`, `chunks_vdb`. ORAM controller in the Embedding TEE (same TDX/SEV-SNP VM that already runs embeddings in Approach 4).
- This closes the access-pattern leak on the VDB lookups, which is the single largest unaddressed surface after Phase 0.
- Effort: 4-6 weeks. Largest single integration in the roadmap; the Compass paper has a reference implementation we can fork.

**Phase 2 — Volume-hiding for the adjacency and source_id EMMs**

- XorMM EMM for `get_nodes_edges_batch` and for the `source_id` multi-map.
- Add per-session HMAC keying for cross-session unlinkability.
- Effort: 2-3 weeks. XorMM has reference implementations; the integration is in indexing time + query-time KV reads.

**Phase 3 — Compose with generation-side defenses**

- Wrap LightRAG output assembly with PrivGemo-style structural-anonymization (supernode clustering, per-session entity-ID pseudonymization).
- Add DP-KSA-style output DP at the generation step. Both are orthogonal to the retrieval crypto.
- Effort: 2-4 weeks. Requires aligning with the generation-privacy layer (GELO / ObfuscaTune / TEE-LLM).

**Phase 4 — Verifiability (optional, research-tracking)**

- ZKGraph-style proof of correct 1-hop expansion; lifted to multi-hop only if we add multi-hop expansion to LightRAG later.
- Effort: open. Tracking only until proof-size / verifier-time numbers appear in the literature.

---

## 9. Recommended primitives — one-line decision matrix

| LightRAG step | Variant A (TEE) | Variant B (crypto-only) | Notes |
|---|---|---|---|
| Embed | TEE | Client-local | unchanged from `private-rag-system-design.md` |
| `entities_vdb` ANN | Compass ORAM-HNSW in TEE | Pacmann PIR | |
| `relationships_vdb` ANN | Compass | Pacmann | same as above |
| `chunks_vdb` ANN | Compass | Pacmann | `mix` mode only |
| `get_nodes_batch` | Same ORAM controller | XorMM EMM | |
| `get_edges_batch` | Same ORAM controller | XorMM EMM | |
| `*_degrees_batch` | Fold into prop bag | Fold into prop bag | never expose as separate keyed store |
| `get_nodes_edges_batch` | ORAM (preferred) or XorMM | XorMM | LightRAG-specific volume hazard |
| `source_id` → chunk fan-out | ORAM or padded inline list | XorMM EMM or padded inline | |
| `text_chunks_db.get_by_ids` | ORAM | PIR (SimplePIR/PIANO) | |
| Token merge / rerank | client / TEE | client | no crypto |
| Generation | per `private-rag-system-design.md` §gen | per `private-rag-system-design.md` §gen | orthogonal; DP-KSA / PrivGemo composes here |

---

## 10. Paper additions for EdgeQuake ingestion

The 2026-04 survey already recommended ingesting AGEA, GRAGPOISON, LogicPoison, P-NGDB, GORAM, Graphiti, PrivGemo, TOGES, LINKTELLER, GraphSC, PeGraph. Net-new ingestion candidates from this round:

| # | Title | URL | Justification |
|---|---|---|---|
| 1 | Zhu, Patel, Zaharia, Popa — Compass: Encrypted Semantic Search (OSDI 2025) | https://eprint.iacr.org/2024/1255 (code: https://github.com/Clive2312/compass) | Primary VDB-private primitive; sub-second LAN; HNSW co-design. USENIX-AE Available + Functional + Reproduced. |
| 2 | Zhou, Shi, Fanti — Pacmann: Efficient Private ANN (ICLR 2025) | https://eprint.iacr.org/2024/1600 (code: https://github.com/wuwuz/Pacmann) | Crypto-only fallback for Variant B; 2.5× better accuracy vs prior private ANN; Go implementation. |
| 3 | Li et al. — PRAG: End-to-End Privacy-Preserving RAG (arXiv 2604.26525, Apr 2026) | https://arxiv.org/abs/2604.26525 | HE-only ranked retrieval with Operation-Error Estimation; auxiliary to Compass. |
| 4 | Tang, Flemings, Wang, Annavaram — DP-RAG (arXiv 2602.14374) | https://arxiv.org/abs/2602.14374 | Orthogonal output-side DP defense against extraction attacks. |
| 5 | Wu, Wei, Wang et al. — ZKGraph (arXiv 2507.00427, Jul 2025) | https://arxiv.org/abs/2507.00427 | Verifiable graph-query primitive; v2 integrity layer. |
| 6 | Wang et al. — XorMM volume-hiding EMM (CCS 2022) | https://dl.acm.org/doi/10.1145/3548606.3560593 | Volume-hiding EMM for adjacency / source_id multi-maps. |
| 7 | FLASH conjunctive volume-hiding EMM (IEEE TDSC 2024) | https://ieeexplore.ieee.org/document/11129933/ | Conjunctive variant when metadata filters required. |
| 8 | Veil overlapping-bucket volume-hiding (SIGMOD 2023) | (see search) | Tunable storage-vs-leakage knob. |
| 9 | Anonymization Along the RAG Pipeline (arXiv 2604.15958, Apr 2026) | https://arxiv.org/abs/2604.15958 | Empirical evidence that *early* anonymization beats late masking — guides PrivGemo composition. |
| 10 | Securing RAG: Taxonomy (arXiv 2604.08304, Apr 2026) | https://arxiv.org/abs/2604.08304 | Recent threat-taxonomy survey; useful for threat-model cross-referencing. |
| 11 | H₂O₂RAM: High-Performance Doubly Oblivious RAM (Sep 2024) | https://arxiv.org/abs/2409.07167 | Replaces Opal's Path-ORAM-in-TEE; ~1000× faster substrate. |
| 12 | CryptGNN: Secure GNN Inference (CCS 2025) | — | Deferred; for the GNN-RAG variant. |

---

## 11. Sources

**Primary code read:** `HKUDS/LightRAG`, `lightrag/operate.py` (5197 LOC), `kg_query` (3164), `_perform_kg_search` (3573), `_get_node_data` (4359), `_find_most_related_edges_from_entities` (4419), `_find_related_text_unit_from_entities` (4475), `_get_edge_data` (4634), `_find_related_text_unit_from_relations` (4726).

**Companion docs in this repo:** `private-graph-search.md` (research baseline 2026-04-23), `private-rag-system-design.md` (vector-only system design), `private-inference.md`, `private-information-retrieval.md`, `private-embedding-research.md`, `fhe-encrypted-vector-db.md`.

**External — anchor papers (new this round):**
- LightRAG (Guo et al., EMNLP 2025) — https://arxiv.org/abs/2410.05779
- GraSS (Kim et al., 2024) — https://eprint.iacr.org/2024/2012 (already in EdgeQuake)
- Compass (Zhu et al., OSDI 2025) — https://www.usenix.org/conference/osdi25/presentation/zhu-jinhao
- Pacmann (Zhou et al., ICLR 2025) — https://openreview.net/forum?id=yQcFniousM
- PRAG (Li et al., 2026) — https://arxiv.org/abs/2604.26525
- DP-RAG (Tang et al., 2026) — https://arxiv.org/abs/2602.14374
- ZKGraph (Wu et al., 2025) — https://arxiv.org/abs/2507.00427
- Anonymization Along RAG (2026) — https://arxiv.org/abs/2604.15958
- Securing RAG Taxonomy (2026) — https://arxiv.org/abs/2604.08304
- H₂O₂RAM (2024) — https://arxiv.org/abs/2409.07167
