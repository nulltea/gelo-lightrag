# Private LightRAG — Variant A (TEE-anchored): Detailed Implementation Plan

> Plan date: 2026-05-16. Targets the **Variant A** design from
> [`../research/private-graph-rag-design.md`](../research/private-graph-rag-design.md)
> §5.1 — Compass-style encrypted HNSW + XorMM-style volume-hiding EMM inside the
> existing Embedding-TEE / Two-Party-KDF infrastructure (`gelo-rag`,
> `gelo-tee-sev-snp`, `rag-core`).
>
> Source anchors:
> - Compass paper + reference impl indexed in EdgeQuake (paper
>   `7b372edb-1d69-44ca-a053-c64b49c9dc56`, repo
>   [`Clive2312/compass`](https://github.com/Clive2312/compass), USENIX-AE
>   Available + Functional + Reproduced).
> - LightRAG retrieval protocol from
>   [`HKUDS/LightRAG`](https://github.com/HKUDS/LightRAG) `lightrag/operate.py`
>   @ main (referenced by line numbers throughout — see design doc §1.2 for the
>   numbered 10-step protocol).
> - Existing private-RAG stack: `crates/core` (CAPRISE, AES, HKDF, embedder
>   trait), `crates/gelo-rag` (`GeloRagInMemoryService`,
>   `GeloRagTwoPartyService`), `crates/gelo-tee-sev-snp` (`SnpTrustedExecutor`,
>   SEV-SNP RATLS attestation), `crates/remote-rag` (HNSW reference usage with
>   `hnsw_rs`).

---

## 0. Definitions and acronyms used in this plan

- **Variant A** — the TEE-anchored deployment from the design doc: Embedding TEE
  also hosts the Compass ORAM controller and the XorMM EMM endpoints; storage
  server is fully untrusted.
- **Compass** — Zhu et al., OSDI 2025. HNSW-over-Ring-ORAM with three
  optimizations (Directional Neighbor Filtering, Speculative Neighbor Prefetch,
  Graph-Traversal-Tailored ORAM with multi-hop lazy eviction).
- **Ring ORAM** — Ren et al. — the ORAM construction Compass extends. Each
  bucket has `Z + S` slots (Z real + S dummy); reshuffle is *amortized* (every
  `A` accesses) rather than per-access like Path ORAM. Constant **online**
  bandwidth via the XOR trick — the eviction path is the heavy part, which
  Compass batches.
- **HNSW** — Hierarchical Navigable Small World graph. Tens of layers, top
  layer has very few nodes (entry point), bottom layer holds all data.
  LightRAG defaults to HNSW under the hood via `nano-vectordb` (or any of the
  pluggable backends).
- **`ef` / `ef_spec` / `ef_n` / `M`** — HNSW + Compass parameters. `ef` is
  the dynamic candidate-list width (beam width during greedy search); `M`
  is the degree bound; `ef_spec` is Compass's speculation-set size;
  `ef_n` is the directional-filter size. **The number of batched ORAM
  round-trips Compass issues per query is `n = ⌈ef / ef_spec⌉`** — so
  `ef` is publicly observable via RPC count. The paper explicitly
  declines to hide it.
- **`Z`, `S`, `A`** — Ring-ORAM bucket parameters: real-block slots,
  dummy slots, eviction rate. Tuned per Compass index.
- **EMM** — Encrypted Multi-Map. Key→list-of-values. The natural substrate for
  an adjacency list / source_id list.
- **XorMM** — Patel-Persiano-Yeo et al., CCS 2022. Non-lossy volume-hiding EMM
  with 1.5–2× storage. The recommended substrate for LightRAG's adjacency-list
  and `source_id` multi-maps.
- **PRP / PRF** — pseudo-random permutation / function. Used for path mapping
  (PRP) and search-key transformation (PRF) inside the ORAM controller and the
  EMM.
- **`InProcessTrustedExecutor`** — the existing TEE protocol engine
  (`gelo-protocol/src/sim.rs`). Wrapped by `SnpTrustedExecutor` for real
  SEV-SNP attestation. The natural host for the Compass controller and EMM
  endpoints.
- **CAPRISE** — `Caprise` in `crates/core/src/caprise.rs`. The distance-
  preserving symmetric-encryption scheme protecting embeddings at rest. **Not**
  replaced by Compass — *complements* it: Compass hides access pattern over the
  ORAM tree; CAPRISE renders the leaked ciphertext content useless if the
  storage server is later compromised (defense in depth).
- **TPM / VCEK** — Trusted Platform Module / Versioned Chip Endorsement Key.
  The latter is the SEV-SNP attestation root; `gelo-tee-sev-snp` already
  speaks the `/dev/sev-guest` `SNP_GET_REPORT` ioctl in the production
  feature.
- **RATLS** — Remote-Attested TLS. Client verifies the SEV-SNP report inside
  the TLS handshake before sending plaintext. Existing path in
  `gelo-rag::two_party_service` carries `user_x_sk` over this channel.
- **HKDF** — HMAC-based Key Derivation Function (RFC 5869). Used for the
  two-party `(user_x_sk, tee_user_x_sk) → per-tenant child key chain`.
  Variant A bumps the chain from 2 children to 8.
- **HMAC perturbation** — this plan's per-session deterministic
  embedding perturbation that randomises the HNSW start vertex per
  RATLS session. Breaks cross-session linkability of identical queries
  at the execution-pattern level (timing, RPC count, perf-counter side
  channels). Detailed in §8.6.
- **DistanceDP** — RemoteRAG's `(n, ε)`-DP construction
  (`remote-rag::planar_laplace`). Planar-Laplace noise on the embedding
  such that any two queries within radius `r` are formally
  indistinguishable. Composes on top of HMAC perturbation as an opt-in
  formal layer.
- **AP / QP / SP** — Access Pattern / Query Pattern / Search Pattern
  leakage (SSE terminology). Compass hides AP; XorMM hides volume / a
  slice of QP; HMAC perturbation hides SP across sessions.
- **Opal** — Kaviani et al. 2026. Runs Compass inside a TEE; formally
  imports Compass's batched-access lemma into a `G_att`-hybrid security
  proof. Recognised precedent for putting Compass's client role inside
  a CVM (see §2.0).
- **AGEA** — Agentic Graph Extraction Attack (arXiv 2601.14662). 96 %
  node-leak at T=1000 queries on LightRAG. Defended at the generation
  layer (PrivGemo / DP-RAG), not by retrieval crypto — out of scope
  here, called out where it bears on acceptance criteria.

---

## 1. What we are building, in one paragraph

A multi-tenant SEV-SNP-attested service that runs **LightRAG retrieval**
(operate.py `kg_query` → `_perform_kg_search`) end-to-end inside a single
CVM. The CVM holds: an HNSW index over entity embeddings, an HNSW index over
relation embeddings, an HNSW index over chunk embeddings, an EMM over the
graph adjacency lists, an EMM over per-node and per-edge `source_id` lists,
and an AES-GCM key for the chunk store. All four data structures live
**encrypted on a fully untrusted blob server**; the CVM only holds the
position maps, stash, treetop cache, quantized hints, and per-tenant keys.
A storage-server compromise leaks neither query content nor access pattern.

The client (thin) chunks documents, attests the CVM via RATLS,
ships plaintext text + tenant secrets into the CVM over the attested channel,
and receives the assembled LightRAG context (or the final LLM answer, if the
generation layer is also TEE-hosted via the existing `gelo-rag` path).

---

## 2. Trust model recap and what changes vs. today

### 2.0. Departure from Compass's paper: who plays the "client" role

Compass's paper defines a protocol with a **client** holding the position
map, stash, treetop, and ORAM key, and an untrusted **server** holding
the encrypted ORAM tree. The paper's deployment puts the client on the
**user's own device** — explicitly avoiding TEEs because hardware-enclave
side channels weaken the security story. The headline numbers in Tab. 3
(LAION 0.7s / SIFT1M 1.1s) are measured with the client running on a
commodity laptop or web browser.

This plan **deviates from the paper's deployment** in one specific way:
we put Compass's client role *inside the SEV-SNP CVM*, with the user's
device acting as a thin RATLS client to it. The Compass *protocol* is
unchanged; the *trust anchor* shifts from "user owns the hardware" to
"SEV-SNP attestation + AMD vendor."

This is a recognised composition. **Opal** (cited in the design doc
§3.5 and §C.3) already runs Compass inside a TEE and formally imports
Compass's batched-access lemma (Lemma 4 of the Compass paper) into its
`G_att`-hybrid security proof. The security property — "any two access
traces of equal public structure are computationally indistinguishable
to an adversary holding only the ORAM-tree storage" — is preserved
because it is a property of the protocol, not of which silicon hosts
the client.

| Aspect | Paper-faithful Compass | Variant A (this plan) |
|---|---|---|
| Trust anchor | user owns the hardware | SEV-SNP attestation + AMD vendor |
| Compass client state lives | device RAM | encrypted CVM RAM (hidden from host OS) |
| Side-channel exposure | device-local | SEV-SNP side channels in-scope (CVE-2023-20593, Hertzbleed, PSP firmware) |
| User-facing client | thick (5–500 MB ORAM state) | thin (RATLS only, plaintext requests) |
| State portability | per-device, sync is user's problem | centralised; switch devices freely |
| Multi-tenant | one user per device | many tenants per CVM, HKDF-isolated |
| Forward security | OS-process isolation | two-party KDF Option 3 (TEE-seal break alone does not recover past sessions) |

**Why this swap is acceptable in our deployment.** The system-design
doc's Approach 4 targets thin clients (mobile, browser, multi-tenant
SaaS). A thick Compass client with 5–500 MB of position-map state and
the full ORAM protocol code in JavaScript is not a viable target there
— *some* trust anchor other than device-ownership is required. SEV-SNP
+ attestation is the existing trust anchor in this codebase
(`gelo-tee-sev-snp::SnpTrustedExecutor`) and the same one CAPRISE-at-rest
already relies on; reusing it for Compass keeps the TCB unchanged.

**When this swap is not acceptable.** Deployments where SEV-SNP trust
is forbidden by policy (defence, regulated finance, supply-chain-paranoid
shops) should use **Variant B** from the design doc instead — that is
the paper-faithful Compass deployment. Variant A and Variant B are
parallel deployments of the same protocol with different trust anchors,
not primary-and-fallback.

### 2.1. State and key boundaries

Today (existing `GeloRagTwoPartyService`):

```
client ──RATLS──▶ [CVM]
                  │ HKDF(user_x_sk, tee_user_x_sk) ─▶ (caprise_seed, aes_chunk_key)
                  │ FastEmbedder | GeloBertEmbedder
                  │ Caprise.encrypt   (DPE)
                  │ InMemoryEncryptedIndex.search   (LINEAR over CAPRISE ciphertexts)
                  │ AesChunkCipher.decrypt
                  ▼
              RetrievalHit
```

What changes for Variant A LightRAG:

1. The single flat `InMemoryEncryptedIndex` becomes **three Compass-encrypted
   HNSW indexes** plus **two XorMM-encrypted EMMs** plus **two encrypted KV
   stores** plus the existing AES-GCM chunk store — six new storage
   structures, all encrypted-at-rest with keys derived from the same HKDF
   chain.
2. The retrieval entry point becomes `LightRagQuery::kg_query(text, mode,
   top_k, …)` mirroring `operate.py:3164` rather than the flat `query(text,
   top_k)`.
3. The TEE process now holds **Ring-ORAM position maps + stash + treetop
   caches** for each of the three Compass instances, plus **EMM client
   state** for each XorMM instance. All of this lives **inside** the
   `SnpTrustedExecutor` boundary.
4. Generation is **out of scope for this plan**; the deliverable is a
   `LightRagQuery::context(...)` API that returns the assembled
   entities-relations-chunks string (and, separately, the structured
   `QueryContextResult`). The existing `gelo-rag` ObfuscaTune/GELO+OSNIP
   path can plug in on top later.

What stays the same (and must keep working):

- The CVM is attested via SEV-SNP using `gelo-tee-sev-snp::SnpTrustedExecutor`;
  `model_identity` now also covers the LightRAG retrieval-protocol identity
  (a hash of the Compass + XorMM build manifest).
- Per-tenant key derivation via `HkdfPolicy` from `caprise-two-party-kdf.md`.
  We **extend** the derivation: from two children `(caprise_seed,
  aes_chunk_key)` to eight `(caprise_seed, aes_chunk_key, oram_keys × 3,
  emm_keys × 2, search_pattern_key)` — same two-party root, six more
  derived child keys. The `search_pattern_key` powers the per-session
  HMAC perturbation in §8.6.
- Forward-security property of Variant A (Option 3 in `private-rag-system-
  design.md` §Approach 4 step 4): a TEE-only seal break must still not
  recover any plaintext. Each child key is HKDF-derived from
  `(user_x_sk, tee_user_x_sk)` and zeroized per request.

---

## 3. Compass — concretely, the parts we need

The EdgeQuake-indexed paper section establishes the following invariants
(see Tab. 1, §4.1, §4.7 of the paper):

- **ORAM substrate:** Ring-ORAM with parameters `(Z, S, A)` — `Z` real
  block slots, `S` dummy slots, eviction every `A` accesses. Bucket has
  `Z + S` blocks total.
- **Block layout (critical):** each ORAM block stores `(embedding,
  neighbor_list)` for exactly one HNSW node. Storing both in the same block
  avoids a second ORAM round-trip per hop. Per-block size:
  `D · f32 + M · u32 ≈ 4D + 4M` bytes (typical: D=768, M=16 → ~3.1 KB).
- **Client-side state (per index):** stash, position map (block_id →
  path_id), per-bucket metadata cache, treetop cache (top `t` levels),
  HNSW metadata (number of layers, M, ef, ef_n, ef_spec), quantized hints
  for Directional Neighbor Filtering.
- **Server-side state:** the encrypted ORAM tree with `~2N` buckets (binary
  tree of ~`log N` levels), plus a Merkle tree of bucket-content hashes for
  malicious-server integrity.
- **Per-query algorithm (Alg. 1 of paper):**
  1. Greedy-search at each HNSW layer using *batched* ORAM ReadPath.
  2. At each step, extract top `ef_spec` candidates from the candidate list;
     fetch their neighbors (Speculative Prefetch).
  3. Use quantized hints to filter neighbors to top `ef_n` per candidate by
     direction (Directional Neighbor Filtering).
  4. Issue **one batched ORAM request** for the resulting `ef_spec · ef_n`
     blocks (padded if fewer).
  5. After `n = ⌈ef / ef_spec⌉` iterations at the bottom layer, return the
     top-K.
  6. **Lazy eviction:** queue eviction-path operations; flush after the
     query returns to the user. Bandwidth-heavy but off the critical path.
- **Throughput / latency on commodity hardware (Tab. 3 of paper):**

| Dataset | size | dim | Perceived (s) | Full (s) | Round-trips | Comm/query (MB) |
|---|---|---|---|---|---|---|
| LAION | 100K | 512 | **0.7** | 0.7 | 8 | 13.5 |
| SIFT1M | 1M | 128 | **1.1** | 1.1 | 8 | 12.2 |
| TripClick | 1.5M | 768 | 6.0 | 9.0+ | 9 | 132.8 |
| MS MARCO | 8.8M | 768 | 8.9 | ~14 | 9 | 226.4 |

The first two rows are exactly the regime we expect for LightRAG enterprise
KGs (10⁴–10⁶ entities / relations / chunks at ~768 dims). Sub-2-second
per-VDB query is realistic; three parallel VDBs ⇒ ~2 s total, dominated by
network RTT.

- **Memory budget (Tab. 4 of paper):** for LAION/SIFT1M, the client-side
  total (stash + position map + treetop + graph) is **5.5 MB / 35.8 MB**
  respectively. Three such instances inside one CVM = 16 MB / 108 MB —
  comfortably fits in TDX/SEV-SNP VM memory alongside the embedding model
  (~150 MB) and tenant KDF state. TripClick/MS-MARCO scale would push to
  ~250 MB / ~1.5 GB client-side, still acceptable in CVM.
- **Throughput:** 436 QPS at 26 % CPU on a single LAION-scale instance under
  32 Gbps and 25 concurrent client threads. Implies we can serve multiple
  tenants per CVM if KGs are small.

---

## 4. New and changed crates — proposed workspace layout

Add to the workspace (`Cargo.toml` `[workspace] members`):

```text
crates/
  ring-oram/         (NEW)  Ring-ORAM client+server protocol, generic over block size.
  compass-index/     (NEW)  HNSW-over-Ring-ORAM with the three Compass optimizations.
  xormm-emm/         (NEW)  XorMM volume-hiding encrypted multi-map.
  light-kg-store/    (NEW)  LightRAG-shaped storage facade: 3× compass-index +
                            2× xormm-emm + encrypted KV + chunk store. The
                            single thing the higher-level service depends on.
  lightrag-private/  (NEW)  LightRAG retrieval protocol (kg_query, _perform_kg_search,
                            _get_node_data, …) re-implemented over light-kg-store.
                            Inhabits the same shape as gelo-rag::service —
                            ingest_documents / kg_query / get_context.
  core/              (CHANGE)  Add HKDF derivation paths for oram/emm child keys.
                              Extend `EmbeddingEncryptionScheme` trait to allow
                              CAPRISE wrapping inside the Compass block payload.
  gelo-rag/          (CHANGE)  Add `lightrag_two_party_service.rs` mirroring
                              `two_party_service.rs` but with `LightRagQuery`
                              entry point and the new derived key set.
  gelo-tee-sev-snp/  (CHANGE)  `scheme_identity` now hashes (CAPRISE params,
                              Compass parameters Z/S/A, EMM parameters,
                              LightRAG protocol version) instead of just the
                              CAPRISE/GELO bytes. No structural changes.
  gelo-snp-runner/   (CHANGE)  Add `/lightrag/*` HTTP routes alongside the
                              existing CAPRISE routes. RATLS layer unchanged.
```

Not touched: `gelo-protocol`, `gelo-embedder`, `gelo-reranker`, `dp-forward`,
`remote-rag`. They are orthogonal — embedding remains an
`Embedder`-trait call inside the TEE; DP-Forward is a generation-side path.

### 4.1 `ring-oram` (new) — surface

```rust
pub struct RingOramParams { pub z: usize, pub s: usize, pub a: usize, pub block_bytes: usize }
pub struct RingOramClient<B: BlockBackend> { /* stash, position_map, treetop, metadata */ }
pub trait BlockBackend {              // implemented by storage adapter
    fn read_path(&self, path: PathId, bucket_offsets: &[BucketOffset]) -> Vec<EncBucket>;
    fn write_path(&self, path: PathId, updated: &[EncBucket]);
    fn write_batch(&self, writes: &[(BucketOffset, EncBucket)]);
}
impl<B: BlockBackend> RingOramClient<B> {
    pub fn read(&mut self, block_id: BlockId) -> Result<Block>;
    pub fn read_batch(&mut self, block_ids: &[BlockId]) -> Result<Vec<Block>>;   // critical for Compass
    pub fn evict(&mut self);            // can be deferred (lazy eviction)
    pub fn flush(&mut self);            // forces all pending evictions
}
```

The XOR-trick optimisation (server returns `XOR_i b_i` rather than the full
bucket on `ReadPath` of a known-target block) is in `BlockBackend::read_path`'s
contract.

The integrity (malicious-server) path returns Merkle proofs alongside
buckets; `RingOramClient` verifies them. Integrity is a `RingOramClient`
mode flag — `SemiHonest` vs `Malicious` — matching the paper's Tab. 3
column split.

Test surface (M1 acceptance — see §6):
- Property test: `read(read(b)) == read(b)` after eviction.
- Property test: server view of any two same-length query traces is
  computationally indistinguishable (statistical — sample 10⁴ traces,
  chi-squared over path-ID histograms).
- Concrete: at `N = 10⁵`, `Z = 4`, `S = 5`, `A = 3`, a 1 KB block, 1 read +
  full eviction must be `< 50 ms` on localhost; 1 read + lazy-deferred
  eviction `< 5 ms`.

### 4.2 `compass-index` (new) — surface

```rust
pub struct CompassParams {
    pub hnsw: HnswParams,                     // M, layers, ef, ef_n, ef_spec
    pub oram: RingOramParams,                 // Z, S, A
    pub treetop_levels: usize,
    pub directional_hint_bits: u8,            // 4-8 typically
}
pub struct CompassIndex<B: BlockBackend> {
    oram: RingOramClient<B>,
    hnsw_meta: HnswMeta,                      // layer count, entry node, M
    upper_layer_cache: UpperLayerCache,       // small-graph + embeddings, cleartext in TEE
    hints: DirectionalHints,                  // quantized
}
impl<B: BlockBackend> CompassIndex<B> {
    pub fn from_plaintext_hnsw(hnsw: &Hnsw<f32, DistCosine>, params: CompassParams,
                                key: &OramKey, backend: B) -> Result<Self>;        // compass_init analog
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<(BlockId, f32)>>;
    pub fn insert(&mut self, embedding: &[f32], external_id: ExternalId) -> Result<BlockId>;
    pub fn flush_evictions(&mut self);
}
```

`Block` payload format inside Compass: `(node_id u32, embedding [f32; D],
neighbor_list [u32; M], padding ..)`. The embedding is CAPRISE-encrypted
*inside* the block before ORAM encryption — this gives a second crypto layer
for the post-quantum scenario where ORAM's secret-key encryption (AES-GCM)
is later broken: the underlying embeddings remain DPE-protected.

### 4.3 `xormm-emm` (new) — surface

```rust
pub struct XorMmParams { pub volume_bound: usize, pub stash_size: usize }
pub struct XorMmClient<B: ByteStoreBackend> { /* hashtables, stash, key */ }
pub trait ByteStoreBackend {
    fn get_blocks(&self, keys: &[BucketKey]) -> Vec<Vec<u8>>;
    fn put_blocks(&self, writes: &[(BucketKey, Vec<u8>)]);
}
impl<B: ByteStoreBackend> XorMmClient<B> {
    pub fn build(entries: impl Iterator<Item = (LogicalKey, Vec<LogicalValue>)>,
                 params: XorMmParams, key: &EmmKey, backend: B) -> Result<Self>;
    pub fn get(&mut self, key: &LogicalKey) -> Result<Vec<LogicalValue>>;
    pub fn get_batch(&mut self, keys: &[LogicalKey]) -> Result<Vec<Vec<LogicalValue>>>;
}
```

XorMM is *static* — built once from the full multi-map. For incremental
ingest we re-build the EMM at every N-th insert (configurable; default
`N = 1024` or 5 % of corpus, whichever is smaller). Inserts in between are
written to a small auxiliary in-TEE buffer searched in parallel; this is
the same "fresh tier" pattern the design doc §7 risk-5 calls out.

### 4.4 `light-kg-store` (new) — surface

```rust
pub struct LightKgStore<B: StorageBackend> {
    entities_idx: CompassIndex<B>,
    relations_idx: CompassIndex<B>,
    chunks_idx: CompassIndex<B>,
    adjacency: XorMmClient<B>,          // entity_name → list<(src,tgt)>
    src_chunks: XorMmClient<B>,         // entity_name OR (src,tgt) → list<chunk_id>
    node_props: EncryptedKv<B>,         // entity_name → encrypted prop bag
    edge_props: EncryptedKv<B>,         // (src,tgt) → encrypted prop bag
    chunks: AesChunkStore<B>,           // chunk_id → AES-GCM ciphertext
}
```

`EncryptedKv` is a thin AES-GCM wrapper over the same `BlockBackend`; for
LightRAG we deliberately keep it KV-shape and route reads through the same
Ring-ORAM as Compass (see Step 4 in §5) to avoid an extra access-pattern
side channel. Concretely, `EncryptedKv` *is* a `RingOramClient` over fixed-
sized blocks.

### 4.5 `lightrag-private` (new) — surface

```rust
pub struct LightRagPrivateService<E, B> {
    store: LightKgStore<B>,
    embedder: E,
    params: LightRagParams,             // top_k, chunk_top_k, max_*_tokens, mode
}
impl<E: Embedder, B: StorageBackend> LightRagPrivateService<E, B> {
    pub fn ingest_documents(&mut self, docs: Vec<DocumentChunk>,
                            kg: ExtractedKg) -> Result<()>;
    pub fn kg_query(&mut self, query: &str, mode: QueryMode,
                    params: QueryParam) -> Result<QueryContextResult>;
}

pub enum QueryMode { Local, Global, Hybrid, Mix, Naive }
pub struct QueryContextResult {
    pub entities_context: Vec<EntityContext>,
    pub relations_context: Vec<RelationContext>,
    pub chunks_context: Vec<ChunkContext>,
    pub assembled: String,              // the rendered prompt-ready blob
}
```

`ExtractedKg` is whatever the entity-extraction path emits (out of scope
here, per the design doc). For development we hand-construct it from
LightRAG fixtures or call the upstream LightRAG Python implementation
offline and serialize the result.

`kg_query` is a faithful Rust port of `operate.py:_perform_kg_search` +
`_apply_token_truncation` + `_merge_all_chunks` + `_build_context_str`,
with each "ANN call / KV read / EMM get" replaced by the appropriate
private primitive.

---

## 5. Step-by-step retrieval — what each step calls into

Tied to the 10-step protocol in
[`../research/private-graph-rag-design.md`](../research/private-graph-rag-design.md) §1.2.
For each step, we list (a) the LightRAG source code line we're emulating,
(b) the new Rust call site, (c) the underlying primitive.

| # | LightRAG `operate.py` site | Variant A call | Primitive |
|---|---|---|---|
| 1 | `kw_prompt → use_model_func` (3406) | out of scope; future `gelo-rag` LLM-in-TEE | n/a |
| 2 | `actual_embedding_func(...)` (3637) | `self.embedder.embed(&[q, ll_kw, hl_kw])` | `gelo-embedder` (TEE) |
| 2½ | *(no upstream equivalent — new step)* | `search_perturb(emb, kind, session_key)` for each of `q_emb`, `ll_emb`, `hl_emb` | `lightrag-private::search_perturb` — HMAC + optional DistanceDP (§8.6) |
| 3a | `entities_vdb.query` (4370) | `store.entities_idx.search(ll_emb, top_k)` | `CompassIndex::search` |
| 3b | `relationships_vdb.query` (4645) | `store.relations_idx.search(hl_emb, top_k)` | `CompassIndex::search` |
| 3c | `chunks_vdb.query` (3542) | `store.chunks_idx.search(q_emb, top_k)` (mix only) | `CompassIndex::search` |
| 4a | `get_nodes_batch` (4382) | `store.node_props.get_batch(node_ids)` | `EncryptedKv` (Ring-ORAM) |
| 4b | `node_degrees_batch` (4383) | fold into prop bag (see §6 — Risk D) | — |
| 4c | `get_edges_batch` (4655) | `store.edge_props.get_batch(edge_ids)` | `EncryptedKv` (Ring-ORAM) |
| 4d | `edge_degrees_batch` (4447) | fold into prop bag | — |
| 5 | `get_nodes_edges_batch` (4425) | `store.adjacency.get_batch(node_ids)` | `XorMmClient::get_batch` |
| 6 | `split_string_by_multi_markers(entity["source_id"], …)` (4502) | `store.src_chunks.get_batch(...)` | `XorMmClient::get_batch` |
| 7 | client-side round-robin merge (4000) | pure Rust, in-TEE | — |
| 8 | `text_chunks_db.get_by_ids` (4612) | `store.chunks.get_batch(chunk_ids)` | `AesChunkStore` over Ring-ORAM |
| 9 | `_build_context_str` (4056) | `LightRagPrivateService::assemble_context` | pure Rust, in-TEE |
| 10 | `use_model_func(...)` (3316) | out of scope | future `gelo-rag` LLM-in-TEE |

Concrete consequence: every server-visible operation in the LightRAG
retrieval pipeline is mediated by **one of three primitives** — Ring-ORAM
read (Compass and EncryptedKv), XorMM lookup, AES-GCM blob fetch. The
plan reduces to building those three, wiring them through the existing
two-party-KDF and SnpTrustedExecutor scaffold, and porting LightRAG's
`_perform_kg_search` logic faithfully.

---

## 6. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **A** | Compass's Ring-ORAM client + Merkle-integrity is a ~3-month build by itself; no Rust port exists. | High | High | Phase 0 builds *plain* Ring-ORAM without Compass optimizations first; layers Directional/Speculative/Lazy-Eviction on top in sequenced milestones. Verifies behavioural parity with `Clive2312/compass` reference on the LAION fixture at each step. |
| **B** | XorMM is *static*; rebuild on every ingest hurts. | Medium | Medium | Buffer recent inserts in a fresh in-TEE tier, rebuild EMM at threshold (≥1024 new entries or ≥5 %). LightRAG itself batches ingest. |
| **C** | Three parallel Compass instances on one CVM may exceed TDX VM memory for large corpora (TripClick-scale). | Medium | High | (1) Start at SIFT1M-scale corpora (35 MB client state per index ⇒ 108 MB total — fine). (2) For larger, evict client-side caches per-index round-robin between queries. (3) Document hard ceiling at MS-MARCO-scale (~1.5 GB client) for v1. |
| **D** | LightRAG's `node_degrees_batch` / `edge_degrees_batch` look like an obvious side channel if exposed as a separate keyed store. | Medium | Medium | Fold degree into the encrypted prop bag at indexing; never expose as a separate read primitive. Risk reduced to "prop-bag size leak", which is uniform per-bucket if we pad to `MAX_PROP_SIZE`. |
| **E** | LightRAG `source_id` is stored *inline* in the prop bag in upstream; length leaks. | High | Medium | Lift `source_id` out of the prop bag entirely; store via the dedicated `src_chunks` XorMM. Verified by integration test: prop-bag byte length equal across all nodes after padding. |
| **F** | Search-pattern leakage — Compass + Ring-ORAM randomises storage-side path IDs, but two sessions issuing the same query still produce identical *execution* fingerprints (timing · RPC count · batch sizes · SEV-SNP-leaky perf counters). Compass's paper concedes this at the operation-type level (`n = ⌈ef / ef_spec⌉` ⇒ search-vs-insert distinguishable by RPC count); we close the finer content-level version. | Medium | High | First-class component (§8.6, not an afterthought) — per-session HMAC perturbation on each of `q_emb` / `ll_emb` / `hl_emb` via the new `search_pattern_key` HKDF child. Optional DistanceDP composition for a formal `(ε, δ)`-bound. ε tuned at 1–5 % to stay within the 5 % recall-parity budget. |
| **G** | SEV-SNP side-channels (CVE-2023-20593, Hertzbleed, SEV-ES PSP bugs). | Low | Critical | Inherits the existing `gelo-tee-sev-snp` risk register. Compass already proves AP-hiding *against* a malicious host OS (its Tab. 3 "Mal" column); the doubly-oblivious property survives a host-OS-level side channel on the ORAM access path. |
| **H** | Compass's reference is C++; Rust port may diverge in subtle ORAM parameters and weaken the security argument. | High | High | Maintain a `compass-index/tests/parity/` directory that vendors small LAION/SIFT1M fixtures and asserts bit-for-bit identical ORAM access-path traces against a `Clive2312/compass` binary harness. Treat any divergence as a release blocker. |
| **I** | The `compass_init` step requires a *plaintext* HNSW index as input. Building this inside the TEE means the embedder must be in the TEE too — consistent with Variant A but worth flagging. | Low | Low | Build path: client streams documents into the CVM over RATLS → embed inside CVM via `gelo-embedder` → build plain `hnsw_rs::Hnsw` in CVM memory → `CompassIndex::from_plaintext_hnsw` → push encrypted ORAM tree to storage. Plaintext HNSW exists only inside the TEE for the duration of the build. |
| **J** | LightRAG's "ranking + degree-sort" step on edges (`operate.py:4468`) uses *degree* as a sort key — once we hide degree (Risk D), the sort key is gone. | Medium | Medium | Replace degree with a deterministic sort by (weight, cosine_score) — i.e., drop the degree term. The paper's accuracy ablation for LightRAG removing degree drops top-1 by <2 %; acceptable. |

---

## 7. Milestones — sequenced, with effort + done-when

Effort is in "engineer-weeks at one head-down implementer with no other
duties"; multiply by 1.5× for the realistic case of context-switched work
on the existing prototype.

### M0 — Skeleton crates + plumbing (1 week)

- Add `ring-oram`, `compass-index`, `xormm-emm`, `light-kg-store`,
  `lightrag-private` to the workspace as empty crates with `lib.rs` stubs
  and `Cargo.toml`s wired to the existing `[workspace.dependencies]`.
- Extend `rag_core::HkdfPolicy` to derive **six additional child keys**
  per tenant: `oram_keys × 3`, `emm_keys × 2`, and `search_pattern_key`.
  Tests assert the derived keys are stable under fixed inputs and
  zeroized on drop. Each child uses a distinct info string
  (`"gelo-rag.v2.{role}"`).
- Add `gelo-tee-sev-snp::scheme_identity` extension covering the new
  parameters (a SHA-256 of `LightRagParams + CompassParams + XorMmParams`).
  Confirms a stale CVM build cannot impersonate a current one.
- **Done when:** `cargo build --workspace` is green; `cargo test
  -p rag-core` asserts the six new derived keys.

### M1 — `ring-oram` correctness + integrity (2-3 weeks)

- Implement Ring-ORAM client + server harness. Backend is an in-memory
  `BlockBackend` impl for testing; real cloud backend is M5.
- Ports: `ReadPath`, `EarlyReshuffle`, `EvictPath`, position-map ops, stash
  insertion/removal with the sorted-by-path-ID data structure the paper
  calls out for lazy-eviction efficiency.
- Tests:
  - **Functional:** insert/read 10⁴ blocks, verify content fidelity over
    10⁴ random reads with mixed evictions.
  - **Trace indistinguishability:** sample 10⁴ ordered access traces for
    two distinct workloads, run a chi-squared test on path-ID histograms,
    assert `p > 0.01`.
  - **Stash bound:** assert post-eviction stash size ≤ paper bound at
    1 σ over 10³ runs.
  - **Latency:** localhost in-memory backend at `N = 10⁵`, single
    `ReadPath + EvictPath` < 50 ms.
- Add malicious-server mode: each bucket carries a SHA-256 Merkle node;
  client verifies. Match the paper's overhead `~3 %` over LAION.
- **Done when:** all tests above pass; doc in
  `docs/prototype/ring-oram.md` describes the parameter selection and
  has the stash-bound proof sketch.

### M2 — `xormm-emm` correctness (1-2 weeks)

- Implement static XorMM build + `get` + `get_batch`. Reference: Patel,
  Persiano, Yeo, CCS 2022.
- Tests:
  - **Functional:** build EMM over 10⁴ keys with skewed value-list lengths
    (10..10⁴); verify all `get` returns are correct and order-preserving.
  - **Volume-hiding (statistical):** for two same-cardinality multi-maps
    with maximally different volume distributions, assert chi-squared
    indistinguishability on bucket-access traces (`p > 0.01`).
  - **Latency:** 10⁴ key build < 1 s in-TEE; per-`get` < 1 ms.
- Build + `get` over the same `BlockBackend` we use for `ring-oram`; no
  separate storage adapter.
- **Done when:** EMM passes the volume-hiding statistical test; size on
  disk is 1.5–2× the raw multi-map per the paper bound.

### M3 — `compass-index` plain Ring-ORAM HNSW (2-3 weeks)

- Wrap an `hnsw_rs::Hnsw` (already in `remote-rag` as a reference) with
  the Ring-ORAM client. Each HNSW node serialises into a fixed-size block.
- Implement `from_plaintext_hnsw` — the `compass_init` analog: walks the
  plain HNSW, encodes each node as a block, places blocks into the
  Ring-ORAM tree.
- Implement `search` *without* the three optimisations first — i.e., the
  paper's "strawman" §4.3. This is wasteful but correct, and gives us a
  baseline to A/B against in M4.
- **Done when:** `compass-index::tests::recall_matches_plaintext_hnsw`
  passes (top-K recall ≥ 99 % of `hnsw_rs` baseline) on a 10K-vector
  fixture, and per-query latency is recorded in CI.

### M4 — Compass optimisations + parity (3-4 weeks)

- Add Directional Neighbor Filtering: quantize per-node hints (4 or 8
  bits per dimension, parameter `directional_hint_bits`), cache in TEE,
  use to filter neighbors *before* the ORAM read.
- Add Speculative Neighbor Prefetch: batch `ef_spec` candidates per
  ORAM round; pad to `ef_spec · ef_n` for security.
- Add Graph-Traversal-Tailored ORAM tweaks: store `(embedding,
  neighbor_list)` in one block (already done in M3 block layout);
  multi-hop lazy eviction; treetop caching of top `t` levels.
- **Parity test:** vendor a small LAION fixture; run our
  `compass-index::search` and a thin Rust harness around the
  `Clive2312/compass` binary; assert (a) same top-K results, (b) same
  number of round trips, (c) same total bytes transferred to ±5 %.
- **Done when:** parity tests pass; on the SIFT1M fixture our per-query
  perceived latency is within 1.5× of the paper's 1.1 s.

### M5 — Cloud-backed `BlockBackend` (1 week)

- Implement a `BlockBackend` over an S3-compatible object store (the
  natural target — buckets are immutable from the client's view, the
  ORAM tree write phase is a multi-PUT + a small in-place
  bucket-rotation index). Alternatively, a thin REST server with the
  same `(read_path / write_path / write_batch)` API.
- Add the same backend to `xormm-emm`'s `ByteStoreBackend`.
- **Done when:** end-to-end Compass query over network round-trips out
  of the CVM to the cloud backend and back, hitting the paper's
  latency ranges within 2× on TripClick-scale fixtures.

### M6 — `light-kg-store` integration (2 weeks)

- Wire three `CompassIndex` instances + two `XorMmClient` instances +
  one `EncryptedKv` (over Ring-ORAM) + one `AesChunkStore` into the
  `LightKgStore` struct.
- HKDF derives one OramKey per index from `(user_x_sk, tee_user_x_sk,
  "gelo-rag.v2.oram-entities" / -relations / -chunks)`; one EmmKey per
  EMM from `(…, "gelo-rag.v2.emm-adjacency" / -src-chunks)`; one AES
  key from `(…, "gelo-rag.v2.aes-chunks")` — same shape as current
  `aes_chunk_key`. The 8th child `(…, "gelo-rag.v2.search-pattern")`
  is consumed by `lightrag-private::search_perturb` (see §8.6, M7).
- Add a `LightKgStore::build_from_kg(plaintext_kg)` constructor that
  drives M3's `from_plaintext_hnsw` for each index and runs the
  initial XorMM builds — used by the ingest path.
- **Done when:** `LightKgStore::build_from_kg` followed by 100 mixed
  reads/writes reproduces a reference plaintext-LightRAG retrieval
  trace bit-for-bit (modulo deterministic seed).

### M7 — `lightrag-private` retrieval port (2-3 weeks)

- Faithful Rust port of `_perform_kg_search`, `_apply_token_truncation`,
  `_merge_all_chunks`, `_build_context_str`. Each `vdb.query` / `kg.get_*`
  / `text_chunks_db.get_by_ids` site replaced with the appropriate
  `LightKgStore` call.
- Implement `lightrag-private::search_perturb` (§8.6) and insert
  between the embed call and the three Compass searches. Acceptance:
  cross-session linkability test — same query under two distinct
  `session_nonce`s produces statistically uncorrelated Compass
  execution traces (round-count variance + perturbed-direction
  divergence over 10² trials).
- `extract_keywords_only` (LightRAG `operate.py:3406`) — for v1, accept
  pre-extracted `(hl_keywords, ll_keywords)` as input parameters; the
  in-TEE LLM call is plumbed in M9 (out of scope here).
- Replace LightRAG's `rank + weight` edge sort with `(weight, cosine)`
  per Risk J.
- Match LightRAG's round-robin merge bit-for-bit; this is the part
  that determines whether downstream LLM accuracy survives the
  migration.
- **Done when:** on a 1K-doc fixture, the assembled context string
  produced by `LightRagPrivateService::kg_query` matches the upstream
  LightRAG Python implementation modulo (a) the entity-ID
  pseudonymisation step (Risk F) and (b) the degree-sort drop (Risk J).
  Bilingual eval: feed both to the same offline LLM; assert F1 over a
  gold answer set is within 5 %.

### M8 — Multi-tenant integration + `gelo-snp-runner` routes (1-2 weeks)

- Mirror `gelo-rag::two_party_service::GeloRagTwoPartyService` as
  `gelo-rag::lightrag_two_party_service::LightRagTwoPartyService`.
  Each tenant gets one `LightKgStore` instance, keys derived
  per-request as today.
- Add `/lightrag/ingest`, `/lightrag/query`, `/lightrag/attest` routes
  to `gelo-snp-runner`. RATLS unchanged; the request payload now
  carries `(query, mode, top_k, hl_keywords, ll_keywords)`.
- Add request-padding at the RATLS layer (constant request rate per
  session) — closes Risk-mitigation §6 "query-volume leakage" from the
  design doc.
- **Done when:** an end-to-end integration test runs:
  (1) start the `gelo-snp-runner` in mock mode;
  (2) thin client attests, derives keys, ingests 1K docs + an extracted KG;
  (3) thin client issues 50 mixed-mode queries;
  (4) server logs show *only* ORAM-shaped traces; no entity name or
      chunk text leaks via tracing or stderr.

### M9 — Optional follow-ons (deferred)

- LLM-in-TEE keyword extraction (close Step 1).
- Generation-side privacy via `ObfuscaTune` / `GELO+OSNIP` (already in
  `gelo-rag`).
- Volume-hiding under conjunctive filters (FLASH instead of XorMM) if
  metadata filters are needed.
- ZKGraph proof of traversal correctness (Phase 4 in the design doc).
- Per-tenant Compass parameter tuning (smaller corpora → smaller
  treetop, smaller Z; tune for memory).

---

## 8. Cross-cutting work items

### 8.1 Configuration surface

A new `LightRagParams` struct mirrors LightRAG's upstream `QueryParam`:

```rust
pub struct LightRagParams {
    pub top_k: usize,                         // entities / relations VDB top-k (default 20)
    pub chunk_top_k: usize,                   // chunks VDB top-k for mix (default 60)
    pub max_entity_tokens: usize,             // default 4000
    pub max_relation_tokens: usize,           // default 4000
    pub max_total_tokens: usize,              // default 32000
    pub kg_chunk_pick_method: ChunkPickMethod,// VECTOR | WEIGHT
    pub mode: QueryMode,
    pub session_secret: SessionSecret,        // per-session HMAC key (Risk F)
}
```

Compass parameters live separately in `CompassParams` (tied to the
*index*, not the query) — tuned once per tenant at build time.

### 8.2 Token-budget truncation

LightRAG's `truncate_list_by_token_size` (operate.py:3884) needs a
tokenizer. The TEE already loads the embedder's tokenizer (in `gelo-
embedder`); reuse it. Constant time over the candidate list — no
side-channel risk.

### 8.3 Ingest path

```
client (thin) ──RATLS──▶  CVM
                          1. embed → in-TEE plaintext embedding
                          2. CAPRISE-encrypt embeddings → blocks
                          3. build plain HNSW in TEE memory (uses hnsw_rs)
                          4. CompassIndex::from_plaintext_hnsw → encrypted ORAM tree
                          5. push to storage server
                          6. discard plain HNSW (zeroize)
                          7. XorMM builds for adjacency + src_chunks
                          8. push EMM bytes to storage server
```

Ingest is heavier than query — expect minutes per 100K docs. Acceptable;
the upstream LightRAG ingest is also batch-shaped.

### 8.4 Observability without leakage

The CVM must not log plaintext entity names, chunk text, or keyword
strings *even at DEBUG level*. Constrain `tracing::debug!` macros to
print only structural information (counts, byte lengths after padding,
ORAM operation IDs). Add a CI lint check that scans
`crates/lightrag-private/src/**.rs` for forbidden `format!("{}", text)`
patterns. The existing `gelo-rag` codebase has this discipline already;
extend it.

### 8.5 Benchmarks tied to the existing benches

The existing test discipline (memory'd in
`feedback_benches_use_gelo_gpu.md`: GELO+mask+GPU for all benches with
`BEIR_DOCS=500` for routine runs) extends to LightRAG-private:

- `benches/lightrag_compass_query.rs`: end-to-end query latency over a
  synthetic 500-doc / 5K-entity KG.
- `benches/lightrag_compass_ingest.rs`: build-time for the same KG.
- `tests/lightrag_parity/`: assertions against a vendored small LightRAG
  fixture (so we can detect regressions in the port at M7).

### 8.6 Per-session search-pattern perturbation

**The gap.** Compass + Ring-ORAM make path IDs over the storage server
indistinguishable, but the *execution* of the search is still
content-deterministic: the same query embedding produces the same
HNSW traversal, the same RPC count, the same batch sizes, the same
timing. The Compass paper acknowledges a coarse version explicitly —
`n = ⌈ef / ef_spec⌉`, so search-at-ef=64 takes 4 RPCs while
insert-at-ef=200 takes 13, and "an adversary can still distinguish
the operation type." The paper puts this out of scope. We close the
finer content-level version: same operation, same mode, but two
sessions issuing the same query produce different content-adaptive
execution patterns (directional-filter prune rate, speculative
prefetch hit rate, hops-to-convergence) — without perturbation,
those fingerprints link sessions.

**The construction.** Per-session deterministic, per-call cheap.

```text
s_search        = HKDF.Expand(prk, "gelo-rag.v2.search-pattern")  // 8th child key, per-tenant
session_nonce   = runner.fresh_nonce_16()                          // per RATLS session
session_key     = HMAC(s_search, session_nonce)                    // per session

fn perturb(e: &[f32; D], kind: &str) -> [f32; D] {
    // kind ∈ {"q", "ll", "hl"}
    let h         = HMAC(session_key, kind.as_bytes() ⊕ quantize(e));
    let direction = unit_vector_from_32_bytes(h, D);   // deterministic in session_key + e
    normalize(e + ε * direction)                       // ε ≈ 1–5 % of ‖e‖
}
```

- `kind` differentiates the three LightRAG embeddings so they don't
  collide internally.
- `quantize(e)` bins each f32 to ~16 bits so embeddings within the
  same HNSW neighbourhood hash to the same `direction` — keeps recall
  stable under tiny CAPRISE-DPE noise.
- ε tuned at index-build time on the parity bench; default 2 %.

**Session lifecycle.** First request: runner generates
`session_nonce`, stores it indexed by RATLS session ID, returns it in
the response. Follow-up requests: client echoes it. Session end:
runner zeroizes `session_key`, drops the nonce.

**Optional DistanceDP composition.** When formal `(ε, δ)`-DP is a
tenant requirement, wrap `perturb` with the existing
`remote-rag::planar_laplace` mechanism:

```text
e_final = distance_dp(perturb(e, kind), eps_dp, sensitivity)
```

| Mechanism | Per-session deterministic? | Cross-session distinguishable? | Formal bound? |
|---|---|---|---|
| Compass+ORAM only | yes | yes (linkable) | — |
| + HMAC perturbation (default on) | yes | no | none |
| + DistanceDP (opt-in) | no (random per call) | no | (ε, δ)-DP |

**Cost.** One HMAC-SHA256 + one 32-byte-to-unit-vector projection per
embedding. ~µs on AES-NI / SHA-NI. Invisible against the ~1 s Compass
critical path. DistanceDP adds another ~µs when enabled.

**What this does NOT defend against.**

- **AGEA-class extraction.** Generation-layer concern; orthogonal.
- **Embedder side-channels on input length.** Mitigated by fixed-length
  token padding inside `gelo-embedder`.
- **Within-session repeat-query linkability.** Deliberate — needed for
  caching + result consistency. Tenants wanting hidden intra-session
  repeats switch to DistanceDP-only at the cost of result stability.
- **SEV-SNP itself broken.** TCB ceiling; same as everything else in
  Variant A.

**Where it lives.**

- `rag_core::keying::HkdfPolicy::V2` — derive `search_pattern_key`
  alongside the other seven children.
- `lightrag-private::search_perturb` — the `perturb()` function.
- `gelo-snp-runner` — generate / store / echo `session_nonce` per
  RATLS session.
- `lightrag-private::LightRagPrivateService::kg_query` — call
  `perturb()` between the embed step and the three Compass searches.
  Matches Step 2½ in §5.

---

## 9. Open questions to resolve before M1

1. **Ring-ORAM block size.** Compass uses 1 KB-4 KB blocks. Our HNSW
   nodes at D=768, M=16 fit in ~3.2 KB raw + ~50 B metadata. Decide:
   pad up to 4 KB blocks (fits MS-MARCO-class) or use 2 KB blocks with
   larger M trimmed (cheaper for 10⁴-10⁵ corpora). Default proposal:
   2 KB blocks, M=12, D=768; tune per tenant.
2. **AES-GCM nonce reuse on bucket rewrites.** Ring-ORAM rewrites every
   accessed bucket; we cannot have a deterministic nonce or it leaks
   access pattern. The standard pattern is `nonce = bucket_id ‖
   write_counter`; write_counter is on the server, but the client
   verifies via Merkle. Confirm the Compass impl does exactly this;
   port it.
3. **Multi-tenant Compass-index sharing.** Today's `GeloRagTwoPartyService`
   shares one embedder across tenants and isolates only the
   CAPRISE/AES keys. Compass needs per-tenant position maps and stash
   *inside* the TEE; that's 16-108 MB per tenant. Set a per-CVM tenant
   cap; allow horizontal scale by sharding tenants across CVMs.
4. **What does the storage server actually look like?** Object store
   (S3-compatible) is the simplest match for Ring-ORAM's
   write-mostly-buckets pattern, but our `gelo-snp-runner` only
   currently speaks HTTP. Decide: a thin REST server in the same VM
   as `gelo-snp-runner`, or a real S3 client inside the CVM. Default
   proposal: a thin REST server, same hosting machine, different
   process boundary, so we keep the operational story simple.
5. **Where does the plaintext KG come from?** For v1 we accept it as
   an input to `ingest_documents` (i.e., entity extraction is done
   *before* the CVM call). Long-term, we want the extraction LLM to
   live in the CVM too — confirm this is the right phasing and
   document the boundary explicitly.

---

## 10. Effort summary

| Milestone | Engineer-weeks | Critical-path dependency |
|---|---|---|
| M0 skeleton + HKDF extension | 1 | — |
| M1 `ring-oram` correctness + integrity | 2-3 | M0 |
| M2 `xormm-emm` correctness | 1-2 | M0 |
| M3 `compass-index` plain | 2-3 | M1 |
| M4 Compass optimisations + parity | 3-4 | M3 |
| M5 Cloud-backed `BlockBackend` | 1 | M1, M2 |
| M6 `light-kg-store` integration | 2 | M4, M5 |
| M7 `lightrag-private` port | 2-3 | M6 |
| M8 Multi-tenant + runner routes | 1-2 | M7 |
| **Total** | **15-21** | — |

Realistic calendar at 1.5× context-switch overhead: **5-7 months**. M1
and M4 are the largest items; they parallelise with M2.

---

## 11. Acceptance — what "Variant A is shipped" means

A signed release of Private LightRAG that satisfies all of:

- A SEV-SNP CVM running `gelo-snp-runner` with attestation chain rooted
  in AMD VCEK; `scheme_identity` covers LightRAG retrieval params.
- A thin client (mobile/browser/CLI) ingests a 10K-doc corpus + extracted
  KG and issues queries in any of LightRAG's modes (local / global /
  hybrid / mix / naive).
- Storage server observes only Compass + XorMM + AES-GCM traffic.
  Tracing on the storage side cannot reveal queried entity names, chunk
  text, or keyword strings, and an offline adversary holding the
  ciphertext cannot recover plaintext.
- **Cross-session unlinkability:** the same query issued under two
  distinct `session_nonce`s produces statistically uncorrelated
  Compass execution traces (round-count variance + perturbed-direction
  divergence over 10² trials, M7 acceptance gate).
- Per-query perceived latency at 10⁴-10⁵-scale matches the paper's
  LAION/SIFT1M baseline within 1.5×: budget **3 s for 10⁵ entities,
  6 s for 10⁶**.
- F1 over a gold-answer set is within **5 %** of upstream LightRAG's
  plaintext baseline on the same KG and queries (M7 acceptance gate).
- Reproducible benchmarks under the existing
  `feedback_benches_use_gelo_gpu` discipline.

After Variant A is shipped, we can compose with PrivGemo-style
dual-LLM and DP-RAG-style output DP — but those are *orthogonal* and
do not block this plan.
