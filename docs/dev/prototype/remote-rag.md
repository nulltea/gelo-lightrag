---
type: prototype-note
status: current
created: 2026-05-12
updated: 2026-05-14
tags: [remote-rag, paillier]
---

# RemoteRAG Prototype

> **Scope.** Design document for the RemoteRAG (Recipe C) protocol in
> `crates/remote-rag`. Documents the *what and why*, not the *how*. For
> source-level details see crate-level rustdoc. Companion documents:
> [`dp-forward.md`](dp-forward.md) for the DP-Forward layer this consumes
> for optional document-side noise, [`gelo.md`](gelo.md) for the GELO +
> SEV-SNP substrate, and [`future-rnd.md`](future-rnd.md) for the CKKS
> alternative and other research directions.

---

## 1. Background

RemoteRAG (Cheng et al., ACL Findings 2025,
[arXiv 2412.12775](https://arxiv.org/abs/2412.12775)) is a two-stage
retrieval protocol with a **distance-relative** privacy notion called
`(n, ε)`-DistanceDP — the server cannot distinguish the true query from
any other query within radius `n/ε` in embedding space. Critically,
**`n` is the embedding dimension**, not the number of clusters or
candidates — this is the single most common misreading of the paper.

```
Stage 1 (client → server):
    r  ~ Gamma(n, 1/ε)           // radial magnitude
    v  ~ Uniform(S^{n−1})        // direction, Gaussian-normalised
    q' ← q + r · v               // noisy query
                                 // server: ANN over plaintext index;
                                 // returns top-k' candidate vectors

Stage 2 (client → server):
    enc(q_i) for each q_i        // Paillier-encrypt clean query
                                 // server: ∏ᵢ enc(qᵢ)^{e_d[i]_int}
                                 // returns one ciphertext per candidate
                                 // client: decrypt, sort, return top-k
```

Stage 1 alone has degraded recall because the noisy query may pull the
search away from the true neighbour. Stage 2's homomorphic dot-product
rerank against the **clean** query recovers ~100 % recall@k.

---

## 2. How the rerank actually works

A natural reading of "Paillier" is "encrypted-vs-encrypted dot product"
— but Paillier doesn't support that. It supports two operations:

```
Enc(a) · Enc(b)  =  Enc(a + b)        // additive HE: cipher × cipher
Enc(a)^k         =  Enc(a · k)        // cipher × plaintext-scalar
```

So Paillier cannot multiply two ciphertexts to get the product. The
RemoteRAG protocol works *precisely because* one operand stays plaintext
on the server side. The doc embedding is the plaintext operand.

### Stage 1 — plaintext-vs-plaintext ANN

```
Client                                      Server
------                                      ------
q  := embedder.embed(text)                  (holds plaintext doc embeddings e_d[1..N])
q' := q + r·v        (planar-Laplace)
                       --- q' ---->
                                            score(q', e_d[i]) for all i  ← plain cosine
                                            top_k' = sort by score
                       <--- (chunk_id, e_d) for k' candidates ---
```

The server already has the docs in plaintext, so this is just standard
cosine ANN. No crypto. The `(n, ε)`-DistanceDP guarantee comes from `q'`
being a noisy version of `q`, not from any encryption.

### Stage 2 — encrypted query × plaintext docs

```
Client encrypts every dim of q:
  q_ct[i] = Enc(qᵢ)        for i = 0..n-1

Server computes (e_d is PLAINTEXT integer-quantised):
  result = ∏ᵢ q_ct[i]^{e_d[i]_int}
         = ∏ᵢ Enc(qᵢ)^{e_d[i]}
         = ∏ᵢ Enc(qᵢ · e_d[i])
         = Enc( Σᵢ qᵢ · e_d[i] )
         = Enc(<q, e_d>)              ← exact dot product, server never saw q

Client decrypts:
  Dec(result) = <q, e_d>
```

The product of Paillier ciphertexts gives an encryption of the sum of
plaintexts; raising a Paillier ciphertext to a plaintext exponent
multiplies the plaintext by that exponent. The two compose into a dot
product **as long as one side is plaintext on the server.** This
plaintext-doc requirement is the structural shape of the protocol, not an
optimization choice — it dictates the threat-model concessions in §3.

### Note on terminology — "rerank" here is *not* cross-encoder rerank

The word **rerank** is overloaded. Standard RAG pipelines use it to mean
*cross-encoder rerank* — take the top-k′ candidates from a bi-encoder ANN,
then re-score them with a more powerful **cross-encoder** model that
attends jointly over `(query_text, doc_text)`. The improvement comes from
upgrading the scoring function: bi-encoder cosine → cross-encoder
relevance score.

RemoteRAG's "PHE rerank" is a different operation that happens to share
the word:

| | Cross-encoder rerank (standard RAG) | RemoteRAG PHE rerank |
|---|---|---|
| What changes between Stage 1 and Stage 2 | The scoring **model** | The privacy state of the **query operand** |
| Stage 1 scoring | bi-encoder cosine | bi-encoder cosine over *noisy* query |
| Stage 2 scoring | cross-encoder relevance score | bi-encoder cosine over *clean* query (recovered under Paillier) |
| Improvement source | More expressive model captures token interactions | Removing DP noise that was added in Stage 1 for privacy |
| Server-side cost | Run a transformer on each of k′ (query, doc) pairs | k′ homomorphic dot products |

RemoteRAG's Stage 2 uses the **same scoring function** as Stage 1
(bi-encoder cosine). The only thing that changes is whether the query
operand is the planar-Laplace-perturbed `q'` (Stage 1) or the
Paillier-encrypted clean `q` (Stage 2). So Stage 2 isn't upgrading the
*model* — it's *denoising the ranking* that Stage 1 jittered for privacy.
A more descriptive name would be "PHE-denoise" or "exact-cosine
correction", but the paper inherits "rerank" from the retrieval-systems
literature where it generically means "second-stage scoring on a
candidate set."

**They are orthogonal — a full private RAG pipeline can stack both.** The
server runs Stages 1 + 2 (noisy ANN + PHE-rerank); the client then
decrypts the top-k chunks (Stages 2 result + AES-GCM chunk text) and can
optionally run a *cross-encoder* rerank locally on the recovered
plaintext. Cross-encoder rerank cannot happen server-side in this
protocol — running a transformer under FHE is not practical, and the
server in RemoteRAG never sees plaintext query text by design.

---

## 3. Threat model and where RemoteRAG sits relative to CAPRISE

### What each layer protects

| Layer | Protects against | Failure mode when removed |
|---|---|---|
| AES-GCM chunks (`AesChunkCipher`) | Server / disk reader without the chunk key | Plaintext document text |
| **RemoteRAG planar-Laplace query** (Stage 1) | Server seeing the query side; gives `(n, ε)`-DistanceDP | Exact query → query log profiling, membership inference |
| **RemoteRAG Paillier rerank** (Stage 2) | Server seeing clean `q` while still ranking accurately | ~80 % recall under Stage-1 noise alone |

### CAPRISE vs Paillier — different beasts at different layers

CAPRISE and RemoteRAG/Paillier look like alternatives (both "encrypt the
embedding") but solve different problems and trade off in opposite
directions:

| Axis | CAPRISE / SAP | Paillier (RemoteRAG) |
|---|---|---|
| **Primitive** | Property-preserving "scale + perturb" with keyed PRF noise | IND-CPA additively-homomorphic public-key encryption with fresh randomness |
| **Security** | NOT IND-CPA. Distance-preserving by construction ⇒ leaks geometry. Known attacks: KNN-recovery, frequency analysis, sample-and-aggregate (the DCPE/PPE literature) | IND-CPA under Decisional Composite Residuosity. Two encryptions of the same plaintext are computationally indistinguishable |
| **What the server can compute on ciphertexts** | Cosine similarity directly over ciphertext vectors — that is the point | Additions of ciphertexts; multiplication by **plaintext** scalars. Cannot compute distance between two ciphertexts |
| **Server-side index storage** | Encrypted ciphertext vectors (same shape as plaintext) | **Plaintext** embedding vectors (chunk text still AES-GCM-encrypted) |
| **Latency overhead per query** | ~µs (one affine transform + cosine over plain floats) | ~hundreds of ms (CRT-encrypt: ~750 ms; rerank: ~47 ms × k' with rayon) |
| **Ciphertext size** | 1 × n floats (same as plaintext) | n × ~256 B (one Paillier ciphertext per dim) |
| **Composability with DP-Forward** | ✓ DP noise added pre-encryption, survives CAPRISE decryption | ✓ on the doc side at ingestion |
| **Composability with GELO** | ✓ GELO produces the embedding; CAPRISE encrypts it | ✓ same |
| **Composability with the *other* of these two** | ✗ Paillier rerank requires plaintext doc embeddings; CAPRISE-encrypted form destroys the dot-product structure | ✗ same issue from the other side |
| **What it doesn't defend against** | A key-holder who decrypts and runs Vec2Text. A passive observer doing frequency analysis on the stored vectors | A breach of the server's plaintext index. The DP query budget is exhausted after enough queries |

**The fundamental tradeoff.** CAPRISE keeps embeddings encrypted *at rest*
on a curious-but-honest server in exchange for leaking geometry — fine if
the server is your own infrastructure but you want defence in depth, and
CAPRISE keys are held client-side. RemoteRAG/Paillier flips it: server can
be fully untrusted *with respect to queries* (formal `(n, ε)`-DistanceDP),
but you give up at-rest embedding confidentiality in exchange.

**Picker:**
- Curious server, key-holder is trusted, want fast retrieval → **CAPRISE**.
- Untrusted retrieval server, willing to trade query latency for formal
  DP on the query content, doc embeddings are not the secret you're
  protecting → **RemoteRAG/Paillier**.
- Both → not directly stackable; see "mutual exclusion" below.

### Why CAPRISE-at-rest and RemoteRAG-PHE-rerank are mutually exclusive

Paillier's additive homomorphism evaluates

```
Enc(<q, e_d>) = ∏ᵢ Paillier(qᵢ)^{e_d[i]_int}
```

which requires the **server** to hold the integer-quantised `e_d[i]` as an
exponent. CAPRISE's stored form `s · e + r · u` is a different algebraic
object — the affine scale and the unit-sphere noise term destroy the
dot-product structure under Paillier composition, and CAPRISE ciphertexts
cannot directly serve as Stage-1 ANN operands against a planar-Laplace-
perturbed query either.

A deployment therefore chooses one of two storage models:

- **CAPRISE-at-rest + DP-Forward**: server-side ciphertexts confidential
  to non-key-holders; key-holder gets DP-bounded embeddings. No PHE
  rerank, so the planar-Laplace mechanism cannot be used efficiently.
- **RemoteRAG plaintext-index + PHE rerank**: server-side embeddings are
  plaintext (chunk text still AES-GCM-encrypted), Stage-1 ANN + Stage-2
  Paillier rerank give `(n, ε)`-DistanceDP on queries with ~100 % recall.

The prototype ships both as **parallel services**, not a switch on the
same service — they have genuinely different surface areas. See
[`future-rnd.md`](future-rnd.md) for why "encrypted-everywhere ANN"
cannot bridge them, even with a fully-homomorphic scheme.

---

## 4. Implementation scope

### Crate layout

```
crates/dp-forward/                DP-Forward paper primitives (see dp-forward.md)
                ▲
                │
crates/remote-rag/                this crate
                ▲
                │
(parallel to `gelo-rag`, not consumed by it)
```

`remote-rag` owns:

- The internal `planar_laplace` module (Stage-1 mechanism).
- The Paillier implementation (`paillier.rs`) — see §4.3 for provenance.
- `RemoteRagService` — the two-stage retrieval protocol against a
  plaintext-server-side index.
- AES-GCM chunk-payload encryption via `rag_core::AesChunkCipher`.

### What's covered

- Planar-Laplace mechanism with empirical Gamma moment checks.
- Paillier keygen (with full CRT factor table precomputed), Enc/Dec,
  homomorphic add, scalar-mul, dot-product via multi-exponentiation,
  fixed-point quantisation, signed-result decode.
- `RemoteRagService` with `ingest_chunks` / `query` mirroring approach-4.
  Includes a unit test where the Stage-2 rerank is *load-bearing*: under
  ε=4 (tight) the Stage-1 cosine order alone would miss the true top-1,
  and the Paillier-decrypted rerank restores it.

### Stage-1 ANN backend

The Stage-1 ANN auto-switches between two backends based on corpus size:

- **Below `LINEAR_THRESHOLD = 256` docs** — exact linear cosine sweep over
  `Vec<IndexEntry>`. Deterministic, ~µs overhead per doc, used by the
  unit tests (3–12-doc corpora).
- **At or above 256 docs** — `hnsw_rs 0.3` (MIT/Apache-2.0) `Hnsw<f32,
  DistDot>` index built lazily on first ingest past the threshold,
  with `M=16`, `ef_construction=200`, and `ef_search=max(64, k')` at
  query time. Inserts are incremental thereafter.

Since the embedders produce L2-normalised pooled embeddings
(`pool::{last,mean}_l2`), `DistDot` over those vectors equals `1 −
cosine_similarity` — exact-up-to-f32-round-off, no separate cosine
distance impl needed.

**Measured at 10k docs** (`tests/remote_rag_scale.rs`): ingest = 18.3 s
total (~1.83 ms/doc, dominated by HNSW build + AES-GCM + Paillier
keygen); mean end-to-end query latency = 23.5 ms (Stage 1 + Stage 2
PHE rerank with 256-bit Paillier on 15 over-fetched candidates); recall
vs linear-cosine ground truth at k'=15 = 96.0 % (well above HNSW's
typical 95 % at this `ef_search`). Linear-scan at 10k docs would take
seconds.

---

## 5. Key design choices

### 5.1 Planar-Laplace lives inside `remote-rag`, not `dp-forward`

The two privacy mechanisms come from different papers, target different
threats, and have different calibration semantics:

- DP-Forward (`dp-forward`): output-space `(ε, δ)`-SeqLDP, applied to the
  pooled embedding before release. `ε` is a standard DP budget (`ε ∈
  [1, 10]` is the recommended range).
- Planar-Laplace (`remote-rag::planar_laplace`): query-space
  `(n, ε)`-DistanceDP, applied to the query vector immediately before
  Stage-1 transmission. **`ε` is distance-relative** (`ε ≈ 10·n` to
  `50·n`), so for `n = 384` you get `ε ≈ 4 000`. Mixing it up with
  standard DP `ε` is a common pitfall flagged repeatedly in the
  doc-comments.

Co-locating them in one umbrella module would invite users to compare
their ε's against each other, which is meaningless. Keeping the
planar-Laplace module `pub(crate)`-scoped inside `remote-rag` enforces
the boundary by construction.

### 5.2 RemoteRAG is a separate service, not a flag on `GeloRagInMemoryService`

The mutual-exclusion property of §3 (CAPRISE-at-rest vs PHE-rerank) means
the two protocols have fundamentally different storage models. Trying to
unify them behind a single service shape would either:

- force every approach-4 deployment to depend on `fast-paillier` (or our
  equivalent) even when it doesn't use the PHE path, or
- introduce a runtime branch that's unreachable in production — the
  worst of both worlds, with no compile-time guarantee of which path the
  deployment is actually running.

Keeping `RemoteRagService` as a parallel struct means each deployment
picks one **at the type level** and a verifier reading the binary's
crate-graph can immediately tell which storage model it implements. The
compiler enforces the choice — you cannot pass a `CapriseScheme` into the
`RemoteRagService` constructor; the type doesn't accept it.

### 5.3 Paillier implementation — fast-paillier algorithms re-implemented on `num-bigint`

The Paillier hot path is critical to RemoteRAG perf. We use the
performance-critical algorithms from
[`fast-paillier 0.3.2`](https://github.com/LFDT-Lockness/fast-paillier)
(LFDT-Lockness, MIT/Apache-2.0) — CRT-based modpow in keygen, encryption,
decryption; precomputed `(p, q, p², q², μ_p, μ_q, p_inv_q,
p²_inv_q²)` factor table on `PaillierPrivateKey`; bit-by-bit
simultaneous multi-exponentiation in the homomorphic dot product —
**re-implemented on `num-bigint`** rather than depended on directly.

Why re-implemented:

- `fast-paillier 0.3.2`'s pure-Rust `backend-num-bigint` feature
  transitively pulls `glass_pumpkin`, whose 1.9.x releases depend on the
  yanked `core2 0.4` crate and whose 1.10 release uses a `rand_core` API
  incompatible with `fast-paillier`'s call site (see
  [`future-rnd.md`](future-rnd.md) for the migration path once upstream
  resolves this).
- The crate's `backend-rug` feature works but introduces LGPL/GMP
  transitives, which the design rules out.

Re-implementing the ~400 LOC of textbook Paillier math we need is easier
than maintaining a patched fork. Provenance comment at the top of
`paillier.rs` credits the upstream algorithms.

#### Performance

| Op | Naive | After CRT + multi-exp + rayon | Speedup |
|---|---|---|---|
| Standalone 1024-dim dot product (single core) | 705 ms | 155 ms | **4.5×** |
| Standalone 1024-dim query encrypt (single core, CRT vs public-key) | 1349 ms | 757 ms | 1.8× |
| End-to-end Bench B query (1024-d Qwen3, k' = 6) | 5724 ms | **348 ms** | **16.4×** |
| Per-candidate dot product (rayon across 6 candidates) | ~943 ms | **47 ms** | ~20× |

Optimizations applied:

1. **CRT factor table at keygen.** Precomputes `p, q, p², q², p⁻¹ mod q,
   (p²)⁻¹ mod q², μ_p, μ_q`. All derived once, used by every decrypt and
   private-side encrypt thereafter.
2. **CRT-based decrypt.** `c^λ mod n²` split into `c^{p−1} mod p²` and
   `c^{q−1} mod q²`, Garner-recombine. Both exponent and modulus halved
   ⇒ ~4× faster.
3. **CRT-aware private-side encrypt.** `r^n mod n²` split into halves
   over `p²` and `q²`. Used by RemoteRAG's client (which holds the
   keypair). The server-side `homomorphic_dot` cannot use this — only the
   public key is available there.
4. **Bit-by-bit simultaneous multi-exponentiation** in `homomorphic_dot`.
   Pippenger-style: one full-width squaring per exponent bit (17 total
   for 16-bit fixed-point), instead of `bit-width × d` squarings.
   Signed exponents handled by pre-inverting bases via Bezout (one
   modinverse per negative dim — cheap compared to a modpow).
5. **Rayon parallelization.** Per-thread `ChaCha20Rng` seeded from
   `OsRng`. The 1024-dim query encryption and the per-candidate dot
   products run across cores. Paillier nonces must be unique per
   ciphertext — no shared-seed determinism.

### 5.4 Document-side noise is also Recipe-B, not a separate scheme

When the user wants DP-bounded *documents* in the RemoteRAG plaintext
index (so even a server compromise doesn't reveal exact training-corpus
embeddings), the natural choice is the same DP-Forward mechanism applied
at ingestion. That's why `remote-rag` depends on `dp-forward` — it
reuses the aMGM primitives instead of inventing a separate doc-side
mechanism. The privacy accounting composes cleanly: query-side
`(n, ε_q)`-DistanceDP and doc-side `(ε_d, δ_d)`-SeqLDP are *independent
guarantees* over disjoint releases.

---

## 6. Verification and current results

### Tests landed (all green)

| Test | What it asserts |
|---|---|
| `remote-rag::planar_laplace::radius_gamma_mean_matches_n_over_epsilon` | Empirical mean within 0.1 of `n/ε` over 20k samples |
| `remote-rag::planar_laplace::radius_gamma_variance_matches_n_over_epsilon_squared` | Within 10 % of `n/ε²` |
| `remote-rag::paillier::end_to_end_paillier_dot_matches_plaintext` | Encrypted dot product equals plaintext dot product to 1e-3 |
| `remote-rag::paillier::homomorphic_scalar_mul_negative` | Signed exponents round-trip correctly (modinverse path) |
| `remote-rag::service::paillier_rerank_recovers_when_stage1_is_noisy` | Stage 1 alone misses top-1; Stage 1 + 2 surfaces it |
| `remote-rag::service::dimension_mismatch_errors` | `dp_cfg.n ≠ embedder dim` is rejected at the type-protocol boundary |

### End-to-end (obfuscation_bench, Qwen3 on Vulkan, k' = 6)

```
B: GeloQwen on Vulkan + RemoteRAG (ε≈10·n, k'=3·top_k)
   ingest = 587 ms (146.7 ms/doc)   query = 348 ms   top1 = python-asyncio
   one-time 1024-bit Paillier keygen: 14.7 ms
   per-candidate Paillier dot product ≈ 46.8 ms

Δ vs Bench A (GELO + CAPRISE baseline, same Qwen3 embedder cost):
   ingest:  -0.5 ms / 4 docs   (RemoteRAG saves CAPRISE work)
   query:  +213 ms             (Paillier rerank net cost)
```

The +213 ms net cost on the query is small enough to be a reasonable
tradeoff for the `(n, ε)`-DistanceDP guarantee on the query side. At
production-scale `k'` and corpus size, the rayon scaling becomes more
visible.

---

## 7. Risks and proposed fixes

### Risk: Paillier implementation is unaudited and not constant-time

The implementation passes all functional round-trip tests but has not
been reviewed for side-channel resistance. The `modpow` calls go directly
to `num-bigint`'s `BigUint::modpow`, which is not constant-time and is
sensitive to bit-pattern in both operand and exponent. CRT decryption
introduces an additional timing channel that depends on the bit-length
of intermediate `m_p` / `m_q` values.

**Fix.**

1. For the prototype, this is acceptable: the threat model assumes the
   client's machine is trusted (Paillier private key lives there) and the
   server side only sees ciphertexts.
2. For production, swap the implementation for a tuned library once
   `fast-paillier 0.3.2`'s upstream `glass_pumpkin` dep resolves, or
   accept the LGPL/GMP transitive via its `backend-rug` feature. See
   [`future-rnd.md`](future-rnd.md) for the migration plan and
   alternatives (CKKS via `openfhe-rs`).

### Risk: 1024-bit Paillier modulus is ~80-bit security

The default `PaillierPrivateKey::generate()` modulus — for prototype perf
— is well below the 112-bit NIST-recommended floor.

**Fix.** The API exposes `PaillierPrivateKey::generate_with_bits(n_bits)`;
production deployments should pass `2048` (or `3072` for forward-looking
secrecy at 128-bit equivalent). The unit-test suite uses 256-bit for
speed (~5 ms keygen vs ~500 ms at 1024-bit) and is explicitly documented
as not security-meaningful.

### Risk: Planar-Laplace `ε` is several orders of magnitude larger than standard DP `ε`

A naive reader sees `ε = 4 000` and concludes "this provides no privacy."
The `(n, ε)`-DistanceDP notion is *distance-relative*: the guarantee is
on distinguishing queries within `n/ε` of each other in embedding space,
not on the standard `(ε, δ)`-DP scale.

**Fix.** Prominent doc-comment warning in `PlanarLaplaceConfig`
("ε is **distance-relative**; not comparable to standard DP ε"). The
`config_panics_on_nonpositive_epsilon` test locks the constructor's
panicky validation so a deployment can't accidentally use `ε = 0`.

### Risk: CAPRISE-at-rest and RemoteRAG-PHE-rerank are mutually exclusive

A user who reads §3 in a hurry and tries to layer them will get incorrect
retrieval results without an obvious error message — Paillier cannot
homomorphic-dot against a CAPRISE-encrypted operand.

**Fix.** The prototype enforces the split *structurally* by making
`RemoteRagService` a separate type that does not accept an
`EmbeddingEncryptionScheme`; you cannot pass a `CapriseScheme` into the
RemoteRAG constructor. The compiler enforces the choice. Document
prominently in `remote-rag/README.md` and in §3 of this design doc.

### Risk: Doc-side DP-Forward over RemoteRAG-plaintext-index requires re-ingestion

If the operator changes the document-side DP cfg after ingest, the
plaintext-index entries are already noised under the old cfg. There's no
way to "rebound" them without re-running embedding over the entire
corpus.

**Fix.** `RemoteRagService` does not currently expose a doc-side DP cfg
— the ingest path is intentionally minimal. Operators who want doc-side
noise compose it at the embedder layer (the same `with_dp_forward`
builder on `GeloQwenEmbedder` applies whether the service is
`GeloRagInMemoryService` or `RemoteRagService`).

---

## 8. Forward-looking work

- ~~HNSW over the RemoteRAG plaintext index~~ — **shipped in M7.2**;
  see §4 *Stage-1 ANN backend*.
- **Multi-query batching on the PHE rerank.** Amortise the Stage-2
  homomorphic dot products across a batch of queries to a single corpus,
  cutting per-query cost in proportion to batch size.
- **Migration to a maintained Paillier library** once `fast-paillier`'s
  `glass_pumpkin` dep resolves upstream, OR a wholesale move to CKKS via
  `openfhe-rs` — see [`future-rnd.md`](future-rnd.md) for the
  comparison and the conditions under which each migration pays off.

---

## References

- Cheng, Y., Yao, W., Lin, H., et al. *RemoteRAG: A Privacy-Preserving
  LLM Cloud RAG Service.* ACL Findings 2025.
  [arXiv:2412.12775](https://arxiv.org/abs/2412.12775)
- Paillier, P. *Public-Key Cryptosystems Based on Composite Degree
  Residuosity Classes.* EUROCRYPT 1999.
- LFDT-Lockness. *fast-paillier 0.3.2.* MIT/Apache-2.0.
  <https://github.com/LFDT-Lockness/fast-paillier>
