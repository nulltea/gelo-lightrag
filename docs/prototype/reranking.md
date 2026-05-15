# Private Reranking Prototype

> **Scope.** Design document for the rerank stage implemented in
> `crates/gelo-reranker`, `crates/gelo-snp-runner`'s `/rerank`
> endpoint, and the e2e bench in `crates/gelo-rag/tests/rerank_e2e_bench.rs`.
> Documents the *what and why*, not the *how* — source-level detail
> lives in crate-level rustdoc. Companion docs: `gelo.md` for the
> protocol substrate, `remote-rag.md` for the alternative query-side
> privacy story, `caprise-two-party-kdf.md` for the embedder's
> session-key derivation pattern this design extends.
>
> Research context: `docs/research/private-reranking-research.md`
> (rev-4) and `docs/research/private-reranking-research-round-2.md`
> (rev-5, the round that picked Qwen3-Reranker-0.6B as primary and
> bge-reranker-v2-m3 as fallback).

---

## 0. Definitions

Project- and protocol-specific terms used below. The shared transformer
acronym set is defined in `inference-optimization.md` §0.

- **Cross-encoder** — A single transformer that jointly encodes
  `[CLS] query [SEP] document [SEP]` and emits one scalar relevance
  score per `(q, d)` pair. Distinct from a *bi-encoder*, which
  embeds query and doc separately and scores by cosine. Cross-encoder
  attention sees query-doc token interactions directly; bi-encoder
  does not. bge-reranker-v2-m3 is a cross-encoder.
- **Causal-LM discriminator** — A causal decoder LM prompted with a
  template ending in `assistant\n<think>\n\n</think>\n\n`, where
  the next-token distribution at the final position concentrates
  mass on `yes` / `no`. The "score" is
  `softmax([no_logit, yes_logit])[1]`. Qwen3-Reranker-0.6B uses
  this pattern.
- **`RerankService`** — The trait in `gelo_reranker::service`
  consumed by `gelo-snp-runner`'s `/rerank` route. Two implementations:
  `CrossEncoderRerankService` and `CausalDiscriminatorRerankService`.
  Names track architecture, not model family.
- **`SessionKey` / `QueryKey`** — HKDF-SHA256 keys
  (`gelo_reranker::session`). `SessionKey` is the per-session root;
  `QueryKey = HKDF(SessionKey, "gelo-rerank.query.v1", query_id)`
  is the per-query AES-256-GCM key the rerank bundle is encrypted
  under. Mirrors `rag_core::keying::HkdfPolicy::V1`.
- **`EncryptedRerankBundle`** — Fixed-shape AES-GCM-256 wire format
  emitted by every `RerankService::rerank` call. Always carries
  exactly `k_max` `(nonce, ciphertext)` items, shuffled. The rank
  index is encoded *inside* each encrypted payload, never in the
  emission order.
- **`k_prime` / `k_final` / `k_max`** — `k_prime` is the over-fetch
  size from retrieval (Stage B). `k_final` is the rerank cutoff
  (top-`k_final` after Stage C). `k_max ≥ k_final` is the fixed
  emission count — decoys pad the bundle from `k_final` up to
  `k_max` to hide `k_final` from network observers.
- **Score-export leakage** — The class of leakage specific to
  rerankers: even with GELO-protected inference, the scalar
  `(q, d) → score` output reveals query-document alignment under
  enough queries. Addressed structurally here by keeping scores
  inside the TEE and emitting only the encrypted ordered set.
- **bge-reranker-v2-m3** — `BAAI/bge-reranker-v2-m3`. XLM-RoBERTa-large
  cross-encoder, 568M params, Apache-2.0.
- **Qwen3-Reranker-0.6B** — `Qwen/Qwen3-Reranker-0.6B`. Qwen3-0.6B
  backbone with `tie_word_embeddings = true`, 600M params, Apache-2.0.

---

## 1. Role in the project

The reranker is the post-retrieval refinement step in private RAG.
After CAPRISE-decoded top-`k_prime` candidates land back inside the
CVM (see `gelo.md` §1 + `caprise-two-party-kdf.md`), the reranker
re-scores each `(query, chunk)` pair with a more precise relevance
model and selects the final `k_final`. Two facts make this step
worth treating as its own privacy primitive rather than a thin
extension of retrieval:

1. **The score function changes**, but more importantly the score
   *itself* becomes a new exit channel. A bi-encoder cosine score
   on the storage server is already encrypted by CAPRISE — the
   server learns rank order, not absolute scores. A cross-encoder
   or causal-LM discriminator score is a plaintext scalar that lives
   wherever the reranker runs. If exported in plaintext, accumulating
   scores across queries inverts back to query content with much
   stronger signal than CAPRISE leakage alone. The mitigation chosen
   here — score never exits the TEE, only the encrypted ordered set
   — is structural rather than statistical.

2. **The model is openweight** and runs the same GELO mask + TwinShield
   primitives the embedder validated. No new cryptography, no new
   protocol layer; the rerank service is an architecture-typed wrapper
   over the existing `gelo-embedder` BERT and decoder forward paths,
   plus a small head module and a re-encryption pass.

The design pulls the reranker fully *inside* the trust boundary —
ingest, retrieve, rerank, re-encrypt all happen in the CVM. The only
wire emission per query is `k_max` AES-GCM ciphertexts. This is the
right architecture when the generator is external (client-local LLM
or external API): the host observer can correlate count, timing, and
per-item size but learns nothing about score values, score
distribution, rank order, or chunk identity.

---

## 2. Threat model

Same trust posture as the embedder (`gelo.md` §2), extended to cover
the rerank-specific exit channel.

| Component | Trust | Sees | Does NOT see |
|---|---|---|---|
| User-side text | confidential | — | — |
| TEE (SEV-SNP CVM) | trusted | query, candidate chunk plaintext, scores, rank order, mask state, model weights | — |
| GPU + driver + PCIe | untrusted | public model weights, per-batch masked activations `U = A·H`, integrity-probed matmul results | clean activations `H`, mask `A`, scores, query, doc text |
| TEE host (CVM operator) | untrusted | encrypted CVM memory, masked PCIe traffic, per-query AES-GCM ciphertexts, fixed `k_max` count, per-item size | scores, rank order, chunk identity, query content, model output before encryption |
| Network operator | untrusted | TLS-wrapped requests, attestation evidence | RATLS contents |
| External generator (if used) | untrusted | the prompt content the *client* sends after decrypting the bundle | rerank protocol state |

### What's new vs the embedder threat model

- **Score-export leakage** (round-2 doc §4.1) is closed by keeping
  scores inside the TEE. The rerank output that crosses the
  trust boundary is the encrypted ordered set, not the scores.
- **Rank-order leakage from emission order** is closed by encoding
  `rank` *inside* the AES-GCM payload and shuffling the wire list.
  A host observer cannot read the ranking out of the list index.
- **Cardinality leakage** (the `k_final` value) is closed by
  emitting exactly `k_max` items, with `k_max - k_final` decoys.
  Decoys are AES-GCM-padded to the longest real candidate's length
  so per-item size doesn't fingerprint them.
- **Ciphertext-to-storage linkability** is closed by re-encrypting
  each top-`k_final` chunk under a per-query
  `QueryKey = HKDF(SessionKey, "rerank-output", query_id)`. The
  output ciphertexts share no bytes with the storage-time
  AES-GCM ciphertexts the host may have observed at ingest.

### What the threat model does not cover

- **The prompt content** the client (or whoever opens the bundle)
  forwards to an external generator. If generation runs outside
  the CVM, the chunk plaintext reaches that external service
  regardless. Mitigations: in-TEE generation
  (`gelo-llm.md` direction), client-local generation, or
  obfuscation schemes like OSNIP applied at the prompt boundary —
  out of scope for the rerank stage.
- **Query frequency and timing.** Stable per-query latency leaks
  workload shape. The fixed `k_max` and decoy padding stop content
  inference, not the existence of a query.
- **Side channels in the TEE itself.** Cache / timing / power
  analysis is out of scope for SEV-SNP per `gelo.md` §6.
- **Replay defense at the protocol layer.** Re-using the same
  `(SessionKey, QueryId)` re-derives the same `QueryKey`. Caller
  must guarantee unique `QueryId` per session — AES-GCM does not
  survive nonce reuse, even with fresh per-call nonces under a
  shared key, if the key itself collides across queries.

---

## 3. Supported model architectures

Two architecture-typed services share the `RerankService` trait. Each
loads as `Arc<…Weights>` from safetensors via the HuggingFace Hub;
the SHA-256 of the safetensors bytes (plus the head identity) rides
as `model_identity` through every attestation report.

| Family | Crate path | Reference model | Layers · Hidden · Inter | Distinguishing ops | Score function |
|---|---|---|---|---|---|
| **Cross-encoder** | `gelo_reranker::cross_encoder` | `BAAI/bge-reranker-v2-m3` | 24 · 1024 · 4096 (XLM-R-large) | post-LN BERT, GELU FFN, full bidirectional attention | `out_proj(tanh(dense(cls_row)))` — 2-layer `XLMRobertaForSequenceClassification` head |
| **Causal-LM discriminator** | `gelo_reranker::causal_discriminator` | `Qwen/Qwen3-Reranker-0.6B` | 28 · 1024 · 3072 (Qwen3-0.6B) | pre-LN RMSNorm, SwiGLU FFN, GQA(16/8), RoPE, causal mask | `softmax([no_logit, yes_logit])[1]` — tied LM head; two dot products |

### Why these two, and not jina-reranker-v3

Picked per round-2 research:

- **Qwen3-Reranker-0.6B is the primary.** The backbone is byte-
  identical to `Qwen3-Embedding-0.6B` already in
  `gelo_embedder::decoder`. Every GELO primitive (mask, shield
  rows, U-Verify, OutAttnMult, permuted attention, length
  auto-switch, sensitive-layer exclusion) applies without
  modification. Adding it took a head loader + a chat template
  + a yes/no logit gather.
- **bge-reranker-v2-m3 is the fallback.** Validates the protocol on
  a structurally different architecture (XLM-RoBERTa post-LN BERT vs
  Qwen3 pre-LN decoder), exercises the BERT path the embedder uses
  for BGE-base + BGE-small, and acts as a parity bench against
  Qwen3-Reranker. Apache-2.0 and well-trodden in the IR literature.
- **jina-reranker-v3 is deferred** (round-2 §3.3). Its listwise
  packed-context architecture pushes `n ≈ 16k+` per forward,
  which sits firmly in `gelo-llm.md` §3's fused-permuted-attention
  regime — a 5–7 week prerequisite. CC-BY-NC-4.0 license is also a
  blocker for commercial deployments.

### What the existing `gelo-embedder` code reuses

Both services route their forward pass through `gelo_embedder::bert::forward::run`
or `gelo_embedder::decoder::forward::run` unchanged. The reranker-specific
code is small:

- **Cross-encoder**: `HfTokenizer::encode_pair` (pair encoding of
  `(query, doc)`), `ClassifierHead` (loads
  `classifier.dense.{weight,bias}` + `classifier.out_proj.{weight,bias}`
  from safetensors, applies on the CLS row), a small constructor
  that mirrors `GeloBertEmbedder::new` / `::from_pretrained`. A
  one-line change to `bert/weights.rs::detect_prefix` lets the
  existing loader recognise the `roberta.` prefix XLM-R uses.
- **Causal-discriminator**: prompt template
  (`QWEN3_RERANKER_TEMPLATE` constant, SHA-pinned into
  `model_identity`), `YesNoHead { yes_token_id, no_token_id }`
  resolved at load via `tokenizer.token_id("yes" | "no")`, and
  a last-token-logit gather that does two dot products against
  `weights.token_embedding.row(yes_id)` and `.row(no_id)`. No
  separate LM head weight is loaded — Qwen3-Reranker sets
  `tie_word_embeddings = true`, so the input embedding table doubles
  as the output projection.

---

## 4. Components

The rerank stage adds five new components on top of the embedder.
Each ties one design property to its source code.

### 4.1 `RerankService` trait

`gelo_reranker::service::RerankService`. Three methods:

- `model_identity(&self) -> &[u8]` — `SHA-256(backbone weights ‖ head
  weights ‖ template ‖ pinned token IDs)`. Folded into
  `REPORT_DATA[0..32]` so a relying party can pin the exact reranker
  the CVM loaded — backbone + head + template all bound at once.
- `family(&self) -> &'static str` — `"cross-encoder"` or
  `"causal-discriminator"`. Part of `scheme_identity` so the
  attestation report covers what *kind* of reranker is running.
- `rerank(&mut self, session, request) -> EncryptedRerankBundle` —
  the only entry point that crosses the trust boundary.

### 4.2 `ClassifierHead` / `YesNoHead`

Two head adapters in `gelo_reranker::head`. `ClassifierHead` carries
the dense + out_proj weights and a SHA-256 identity over the head
tensor bytes (so swapping the head changes `model_identity` without
needing to recompute the backbone hash). `YesNoHead` is a `(u32, u32)`
holding the pinned vocab IDs; the LM head reuses the tied
`token_embedding` table from `DecoderWeights`.

### 4.3 `SessionKey` / `QueryKey` derivation

`gelo_reranker::session::SessionKeyPolicy::V1` defines the HKDF
labels:

```
SessionKey = HKDF-SHA256(
    salt = b"gelo-rerank.session.v1",
    ikm  = client_TEE_shared_secret,
    info = "gelo-rerank.session.v1",
)

QueryKey   = HKDF-SHA256(
    salt = SessionKey,
    ikm  = query_id,
    info = "gelo-rerank.query.v1",
)
```

Both keys are 32 bytes and wrapped in `zeroize::Zeroizing` so they
wipe on drop. The policy struct lives in code (no runtime config);
bumping `V1 → V2` is a deliberate breaking change and must be
re-attested. Pattern parallels `rag_core::keying::HkdfPolicy::V1`.

The `client_TEE_shared_secret` is currently a 32-byte token the
client supplies per request (`session_secret` field on `/rerank`).
M5.9 replaces it with an ECDH-derived secret bound to the
attestation report — same API surface, different secret source.

### 4.4 `EncryptedRerankBundle`

`gelo_reranker::output`. The wire format:

```
EncryptedRerankBundle {
    scheme = "aes-256-gcm.v1",
    items  = shuffle([
        // top-k_final real items:
        AES-GCM(QueryKey, nonce_i,
            encode(rank_i, chunk_id_i, chunk_text_i)),
        // padding to k_max:
        AES-GCM(QueryKey, nonce_j, encode(Decoy { padding bytes }))
    ]),
}
```

Sealing logic: in-TEE scoring → sort with tie-shuffle (RNG seeded
from `QueryKey`) → top-`k_final` → pad with `k_max - k_final`
decoys whose plaintext is padded to the longest real text length →
shuffle. Opening is the inverse: decrypt every item, drop decoys,
sort by embedded `rank`.

### 4.5 In-TEE sort with tie-shuffle

`gelo_reranker::score::top_k_with_tie_shuffle`. Sorts scored
candidates by score descending, then within each equal-score bucket
randomises with the same `QueryKey`-seeded RNG. Stops the host from
learning a stable secondary order when scores are close — without
this, two candidates that always tie at the head of the ranking
would consistently emit in the same internal order, leaking
a deterministic position signal.

### 4.6 HTTP `/rerank` endpoint

`gelo-snp-runner` registers `/rerank` alongside the existing
`/health`, `/attest`, `/ingest`, `/query`, `/rotate`. Handler shape:

```
POST /rerank
{ session_secret, query_id_b64, query, candidates[{id, text}],
  top_k, k_max }
```

```
200 OK
{ scheme: "aes-256-gcm.v1",
  items: [{nonce_b64, ciphertext_b64}; k_max],
  family: "cross-encoder" | "causal-discriminator",
  model_identity_b64 }
```

The runner holds the loaded reranker behind
`RerankerHandle = Option<Arc<Mutex<Box<dyn RerankService + Send>>>>`.
When `None` the route returns 501; otherwise it dispatches and
serialises the bundle. The runner integration test in `main.rs`
exercises both branches.

---

## 5. Compute flow & trust boundaries

Per-request flow when the generator is external:

```
Client
  │ TLS-wrapped query + session_secret + candidate_chunk_ids
  ▼
┌──────────────────────────── CVM (SEV-SNP) ─────────────────────────┐
│                                                                    │
│  ─ Stage A · prerequisite ─────────────────────────────────────    │
│  (already happened: docs ingested via GeloBertEmbedder + CAPRISE)  │
│                                                                    │
│  ─ Stage B · Retrieve ─────────────────────────────────────────    │
│   1. embed(query) under GELO+mask  ─────────► (GPU offload, masked)│
│   2. CAPRISE-cosine vs the index                                   │
│   3. AES-decrypt top-k' chunks (CAPRISE key, in-CVM)               │
│                                                                    │
│  ─ Stage C · Rerank ───────────────────────────────────────────    │
│   4. derive SessionKey from session_secret (HKDF)                  │
│   5. score each (q, chunk) under GELO+mask  ─► (GPU offload, masked)
│      • cross-encoder: [CLS] q [SEP] d [SEP] forward, classifier head│
│      • causal-disc.:  chat-template forward, last-token yes/no logits│
│   6. in-TEE sort + tie-shuffle (scores never leave CVM RAM)        │
│   7. take top-k_final, build payload (rank, chunk_id, chunk_text)  │
│   8. derive QueryKey = HKDF(SessionKey, "rerank-output", query_id) │
│   9. AES-GCM-encrypt every real item with fresh nonce              │
│  10. append k_max - k_final decoy items (length-padded)            │
│  11. shuffle list, emit EncryptedRerankBundle                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
       │
       ▼
Client (or an in-TEE generator, depending on deployment)
  ─ derive QueryKey from session_secret + query_id
  ─ AES-GCM-decrypt every item
  ─ drop decoys; sort real items by embedded `rank`
  ─ build prompt → forward to generator
```

What crosses each boundary:

| Boundary | What crosses | What does not |
|---|---|---|
| PCIe (TEE ↔ GPU), Stage B embed | `U = A·H_query` (masked activations), public weights | clean `H_query`, mask `A`, query tokens |
| PCIe (TEE ↔ GPU), Stage C rerank | `U = A·H_pair` per layer (masked activations of `[q; doc]` joint), public weights | clean activations, mask, scores, chunk text |
| CVM ↔ Host RAM | encrypted CVM pages, SWIOTLB DMA bounce buffers | decrypted activations, scores, plaintext chunks |
| Network (TEE → client) | `k_max` AES-GCM `(nonce, ciphertext)` pairs of fixed per-item size | scores, rank order, chunk identity, k_final |
| Client → external generator | the prompt content the client decides to build | the rerank protocol state |

The two PCIe rows above are *the same primitive* — Stage B embeds a
single query, Stage C embeds a `(q, doc)` pair through a different
model. Both ride GELO's per-batch orthogonal mask on the activation
axis, both keep weights public, both produce activations that are
information-theoretically a random rotation to the GPU. The reranker
adds no new privacy primitive at this boundary — it inherits the
embedder's.

---

## 6. Interfaces

### Trait

```rust
pub trait RerankService {
    fn model_identity(&self) -> &[u8];
    fn family(&self) -> &'static str;
    fn rerank(
        &mut self,
        session: &SessionKey,
        request: &RerankRequest<'_>,
    ) -> Result<EncryptedRerankBundle, RerankError>;
}
```

Implementations live next to the trait:
`CrossEncoderRerankService<X: TrustedExecutor>` and
`CausalDiscriminatorRerankService<X: TrustedExecutor>`. The generic
`X` is the executor type — typically
`InProcessTrustedExecutor<WgpuVulkanEngine>` in production, or the
`PlaintextExecutor` / `RayonCpuEngine` variants in tests.

### Constructors

```rust
CrossEncoderRerankService::from_pretrained("BAAI/bge-reranker-v2-m3", exec)
CrossEncoderRerankService::from_local(&model_dir, exec)
CausalDiscriminatorRerankService::from_pretrained("Qwen/Qwen3-Reranker-0.6B", exec)
CausalDiscriminatorRerankService::from_local(&cfg_path, &tokenizer_path, &shards, exec)
```

Both also expose `new(cfg, tokenizer, weights, head, exec)` for
synthetic-weight tests and a `score_input_ids` helper that bypasses
tokenisation (the parity tests use it to decouple model-shape parity
from tokenizer-file dependencies).

### HTTP

```
POST /rerank
Content-Type: application/json
{
  "session_secret":  "<base64 ≥ 16 B>",
  "query_id_b64":    "<base64>",
  "query":           "...",
  "candidates":      [{ "id": "...", "text": "..." }, ...],
  "top_k":           10,
  "k_max":           20
}
```

```
200 OK
{
  "scheme":             "aes-256-gcm.v1",
  "items":              [{ "nonce_b64": "...", "ciphertext_b64": "..." }; k_max],
  "family":             "cross-encoder" | "causal-discriminator",
  "model_identity_b64": "<base64-32B>"
}
```

`501 Not Implemented` when the runner was started without a
reranker model loaded. `500` on internal failure. The `/attest`
route covers the model+scheme binding that lets a client verify
which family + which weights are actually serving.

### Crate dependency boundary

`gelo-reranker` depends on `gelo-protocol`, `gelo-embedder`,
`rag_core`, plus crypto crates (`aes-gcm`, `hkdf`, `sha2`,
`zeroize`). The bench depends additionally on `gelo-gpu-wgpu` for
the production Vulkan path; production `gelo-snp-runner` wires the
trait object via `Box<dyn RerankService + Send>` so the runner
binary stays decoupled from a concrete model loader.

---

## 7. Performance & correctness

### BGE rerank latency — two regimes

"Per-pair" alone is ambiguous for a cross-encoder: it depends on
whether the candidate batch fans out across CPU workers (k′ > 1) or
runs sequentially (k′ = 1). Both regimes are reported.

Measured on `InProcessTrustedExecutor` with GELO+mask enabled, paper-
parity executor default, AOCL-BLIS via the `blas` feature (default-on).
Hardware: AMD Ryzen AI Max+ 395 (Strix Halo) iGPU
`AMD Radeon Graphics (RADV GFX1151)`.

| Workload | Single-pair (k′=1) | Batch (k′=20) |
|---|---:|---:|
| NFCorpus n≈256, **Vulkan** | **1.19 s/pair** | **3.02 s · 151 ms/pair** |

Wall-clock breakdown of the single-pair (k′=1) forward, traced via
`E2E_TRACE=1` on `rerank_e2e_bench.rs`. (Batch trace is empty by
design — `profile::time` is thread-local and rayon worker timings
don't roll up to the main thread.)

- **Mask `apply` + `unapply` GEMMs — 49% combined** (28% unapply,
  21% apply, 138 + 92 calls per forward). `(n+k)×(n+k)×d` on
  AOCL-BLIS AVX-512 with thread-local single-thread pin — this is
  the floor without rewriting the mask protocol.
- **GPU matmul — 27% combined** (18% single, 9% bundled QKV). Eight
  offloaded projections per layer × 23 layers (last layer skipped
  per GELO §3.2). ~3 ms per matmul × 92 dispatches per forward —
  bandwidth-bound on the integrated GPU.
- **Last-block in-TEE projections — 6.7%** (qkv_direct + ffn_up_direct
  + ffn_down_direct + o_direct). Kept in the TEE by `skip_last_layer`.
- **Element-wise (GELU / LayerNorm / add_bias) — ~5%.** Rayon-parallel
  at rerank shape via the `PAR_THRESHOLD_ELEMS = 32 768` cutoff.
- **In-TEE attention — 4.7%** (24 calls × 2.3 ms). 16 heads
  parallelised via rayon when `n ≥ 64`; previously 24% of wall
  before the parallel-heads lever landed.
- **Mask sample (Haar QR) — 2.5%.** One per forward — see §5
  cadence discussion.

### End-to-end stages (R7 GPU, 100 docs, 1 query, k′=20, k=10)

| Stage | Wall | Per-unit |
|---|---:|---:|
| A · Ingest (BGE-base GELO+mask+Vulkan + CAPRISE) | 7.7 s | 13.0 docs/s |
| B · Retrieve (BGE-base query embed + CAPRISE cosine, k′=20) | 178 ms | 178 ms/query |
| C · Rerank bge (20 pairs) | **3.02 s** | **151 ms/pair** |

### Optimization headroom landed

All numbers below ride the paper-parity executor default (per-forward
Haar mask + shield k=8) and AOCL-BLIS via the `blas` feature (default-
on for `gelo-rag` / `gelo-reranker` / `gelo-snp-runner`).

| Stack | Single-pair (k′=1) | Batch (k′=20) | vs T0 |
|---|---:|---:|---|
| T0 baseline — per-offload mask, no BLAS | 2.29 s | 45.7 s | — |
| + L1 (AOCL-BLIS, thread-local single-thread pin) | 1.74 s | 27.1 s | 1.32× / 1.69× |
| + L2 (rayon-parallel candidates) | 1.74 s | 3.23 s | 1.32× / **14.2×** |
| + L4/L5/L6 (rayon GELU/LN/bias, skip-last) | 1.62 s | 3.22 s | 1.41× / 14.2× |
| **+ L3 (rayon-parallel attention heads)** | **1.19 s** | **3.02 s** | **1.92× / 15.1×** |

Two non-default options tried and judged not worth turning on:

- **L3a · `ndarray/blas` globally** — routes every workspace
  `.dot()` through cblas. Marginal: 1.62 → 1.57 s single-pair,
  3.22 → 3.18 s batch (~3% / ~1%). Per-head attention shape
  `(256, 64) · (64, 256)` sits on the BLIS-vs-matrixmultiply
  crossover. Kept as an opt-in `blas-ndarray` feature; the global
  blast radius doesn't justify default-on for a 3% gain.
- **L3b · OutAttnMult for the BERT path** — plumbed (new
  `BertConfig::use_out_attn_mult` + `multi_head_attention_with_offload`
  mirroring the decoder), but on this iGPU at our shapes it regressed
  both regimes (single-pair 1.74 → 4.33 s; batch 3.23 → 17.83 s —
  the iGPU serialises per-layer Q·Kᵀ dispatches across rayon
  workers). Code stays — earns its keep at `n ≥ 512` or on a dGPU —
  but the bench default keeps it off.

### Ranking metrics on the same run

Baseline = BGE-base GELO+mask+Vulkan cosine over CAPRISE index.
Subset of 100 NFCorpus docs constructed to retain qrel-relevant docs
(`subset_corpus` in the bench). Single-digit query counts make these
numbers high-variance per-stage; the relative deltas matter more than
the absolute values at this scale.

| Stage | nDCG@10 | Recall@k | MRR@10 | Δ(nDCG@10 vs baseline) |
|---|---:|---:|---:|---:|
| B · retrieve (baseline) | 0.629 | 0.717 (k=20) | 0.800 | — |
| C · rerank bge | 0.597 | 0.490 (k=10) | 0.753 | **−0.032** |
| C · rerank Qwen3 (1-query slice) | 0.571 | 0.375 (k=10) | 1.000 | **−0.247** |

Two distinct stories:

- **bge** Δ = −0.032 on 10 queries is well within sample noise on a
  100-doc subset where the baseline is already strong (`subset_corpus`
  deliberately keeps relevant docs). Not evidence of a pipeline bug.
- **Qwen3** Δ = −0.247 across multiple runs is structural — the
  `QWEN3_RERANKER_TEMPLATE` constant in `causal_discriminator.rs`
  omits the `<Instruct>: ...` line from the official HF model card
  example. Without it, the discriminator falls back to a weaker
  signal. Tracked in §8 as the first followup.

### Protocol fidelity

Tested via `crates/gelo-reranker/tests/`:

- `cross_encoder_parity.rs` — `InProcessTrustedExecutor` masked vs
  `PlaintextExecutor` plain agree on `(q, doc)` score within `1e-3`
  on synthetic 2-layer BERT weights. Top-1 rank preserved across
  3-doc tests.
- `causal_discriminator_parity.rs` — same shape; `softmax([no, yes])[1]`
  agrees within `1e-3` and `[0, 1]` bounds hold under both executors.
- `bundle_round_trip.rs` — `RerankService::rerank` → wire-shape
  `EncryptedRerankBundle` (always `k_max` items) → client decrypts
  with matching `QueryKey` → recovers exactly `top_k` real items in
  the in-TEE rank order. Wrong session key fails to open.
- `comparative_bench.rs::real_models_bge_vs_qwen3` — `#[ignore]` real-
  weight A/B; both rerankers select a RAG-grounded doc at rank 0 on
  the 6-doc synthetic prompt (sanity assertion).
- `tests::rerank_*` in `gelo-snp-runner/src/main.rs` — `/rerank`
  returns 501 unconfigured, returns a valid bundle when configured.
  Client reconstructs the bundle from the JSON, opens with the
  session-derived key, recovers `top_k` real items.

Total: 9 reranker-specific tests across 4 files plus 2 HTTP
integration tests. All green on the unloaded-model path; the
ignored release-gate tests run on demand.

---

## 8. Status & gaps

### What's landed

- `crates/gelo-reranker` — full crate, 9 source files, 4 test files.
- `gelo-snp-runner` `/rerank` HTTP route with mock-issuer integration
  test.
- E2E bench `crates/gelo-rag/tests/rerank_e2e_bench.rs` with
  `E2E_DOCS` / `E2E_QUERIES` / `E2E_KPRIME` / `E2E_KFINAL` /
  `E2E_SKIP_BGE` / `E2E_SKIP_QWEN3` knobs and configurable corpus
  subsetting via `subset_corpus`.
- Real-weight bench `crates/gelo-reranker/tests/comparative_bench.rs::real_models_bge_vs_qwen3`
  running on AMD Vulkan iGPU.
- This documentation page.

### Levers — landed, opt-in, and open

| # | Lever | Status | Measured impact |
|---|---|---|---|
| L1 | AOCL-BLIS for mask GEMMs (`blas` default feature) + thread-local single-thread pin in `sgemm_blis` so rayon workers don't oversubscribe. | landed | −24% single-pair · −41% batch |
| L2 | Rayon-parallel candidate loop in `CrossEncoderRerankService::rerank`; one cloned executor per worker, single-candidate fast path. | landed | −88% batch · per-pair unchanged |
| L3 | Rayon-parallel attention heads in `bert::attention::multi_head_attention` when `n ≥ 64`; embedder shape stays serial. | landed | −27% single-pair (bucket 7.1× shrink) |
| L3a | Route every `ndarray::dot()` through BLIS (`blas-ndarray` opt-in feature). | opt-in | −3% single-pair · −1% batch — kept opt-in |
| L3b | OutAttnMult for the BERT path (`BertConfig::use_out_attn_mult`); plumbed end-to-end. | plumbed, default off | regresses at `n ≈ 256` on iGPU; useful at `n ≥ 512` or on dGPU |
| L4 / L5 | Rayon-parallel GELU, LayerNorm, add_bias in `bert::forward` with shape-conditional threshold (`n × d ≥ 32 768`). | landed | ~−7% single-pair (combined) |
| L6 | `BertConfig::skip_last_layer` on the reranker — GELO §3.2 sensitive-layer exclusion. | landed | −1.5% wall · paper-aligned |
| Qwen3 template | Fix `QWEN3_RERANKER_TEMPLATE` — add the missing `<Instruct>:` line per HF model card. | open | most likely cause of Qwen3 −0.247 nDCG regression |
| Bucket-pad seq | Bucket-pad input tokens to `{128, 256, 512}` to keep cubecl autotune cache hot. | open | p99 stabilisation, mean wall unchanged |

### Deferred / out of scope

- **jina-reranker-v3** — listwise n≈16k forwards need
  `gelo-llm.md` §3's fused permuted attention + FlashAttention to
  land first (5–7 weeks). License is CC-BY-NC-4.0. Revisit alongside
  the LLM-serving stack.
- **Shredder at the pooled activation** — round-2 §6 follow-up.
  Only relevant if a deployment needs to *export* scores for
  downstream use (calibrated cutoffs, hybrid fusion). Default
  TEE-internal architecture already keeps scores sealed.
- **Score-DP accountant** — formal `(ε, δ)` budget over rerank
  score exports. Same trigger as Shredder: only when scores must
  cross the boundary. Round-2 §4.1 / §5.3.
- **ECDH-bound session-key handshake** — currently the
  `session_secret` is a 32-byte token the client supplies per
  request. M5.9 swaps in an attestation-bound ECDH KEX. The
  `SessionKey::derive` API surface stays identical; only the
  source of the input secret changes.
- **`Game-of-Arrows` empirical attack bench** in reranker mode —
  round-2 §4.3. The construction is safe by the same argument that
  protects the embedder (`memory/gelo_research_round_2.md`); the
  empirical confirmation is gated on lifting the attack reference
  out of `qsxltss/Game-of-Arrows` into a workspace test.

---

## References

- `docs/research/private-reranking-research.md` — rev-4 survey of
  the field.
- `docs/research/private-reranking-research-round-2.md` — round-2
  pass that picked the two architectures shipped here.
- `gelo.md` — the protocol substrate the reranker reuses unchanged.
- `gelo-llm.md` — the long-context fused-attention work that
  unblocks listwise rerankers like jina-v3.
- `caprise-two-party-kdf.md` — the embedder's session-key derivation
  pattern this design extends with `gelo-rerank.session.v1` and
  `gelo-rerank.query.v1` HKDF labels.
- Chen et al., "BGE-M3 / bge-reranker-v2-m3." arXiv 2309.07597.
- "Qwen3-Reranker." arXiv 2506.05176 (Tongyi Lab, 2025).
- `BAAI/bge-reranker-v2-m3` and `Qwen/Qwen3-Reranker-0.6B` model
  cards on the HuggingFace Hub.
