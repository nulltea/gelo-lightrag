---
type: prototype-note
status: current
created: 2026-05-12
updated: 2026-05-22
tags: [rnd, ckks]
---

# Future R&D — Private RAG Prototype

> **Scope.** Research directions and migration paths that are not yet
> implemented in this prototype but inform the architecture. Each section
> includes the technical case, the expected performance / security
> tradeoff, and the engineering effort to land. Companion documents:
> [`dp-forward.md`](dp-forward.md), [`remote-rag.md`](remote-rag.md),
> [`gelo.md`](gelo.md).

---

## 1. CKKS as an alternative to Paillier for RemoteRAG's rerank

> **Status:** parked — awaiting (a) `openfhe-rs` stabilization for
> production-quality CKKS in Rust and (b) a deployment with a measured
> Paillier hotspot that justifies migration cost. Not abandoned.

The Paillier hot path in `RemoteRagService::query` is the single biggest
crypto cost in the prototype. CKKS (Cheon-Kim-Kim-Song) is the
cryptographically natural alternative for this exact workload.

### Why CKKS is "the natural fit" for this shape

The Stage-2 operation is *encrypted query × plaintext server-side
embedding → encrypted scalar*. This is the canonical CKKS workload:

- **SIMD batching.** One CKKS ciphertext encodes up to `N/2` floats in
  slots (typical ring dim `N = 8192`). A 1024-dim query fits in **one**
  ciphertext — not 1024 Paillier ciphertexts.
- **Native ciphertext × plaintext multiplication.** Paillier supports it
  via the `c^k` exponentiation trick (the slow path we optimized). CKKS
  does it as one polynomial multiplication.
- **Log-depth dot product.** Encrypted query × plaintext doc → pointwise
  multiply, then `log₂(n)` rotation-and-sum operations to fold across
  slots. ~10 rotations for `n = 1024`.

### Performance comparison

| Axis | Paillier (current, optimized) | CKKS (projected) |
|---|---|---|
| Per-dot-product latency (CPU, 1024-d, 128-bit security) | 47 ms (rayon across 6 candidates) / 155 ms single-core | **~10–30 ms**, no parallelism required |
| Query encrypt | 757 ms single-core CRT | ~10 ms (one ciphertext, not 1024) |
| Result precision | Exact (after fixed-point dequantization) | Approximate (errors at ~2⁻²⁰ relative — invisible for top-k cosine ranking) |
| Public-key size | ~256 B | tens of KB |
| Rotation / relinearisation keys | n/a | **10s–100s of MB** |
| Keygen time | ~15 ms at 1024-bit | seconds; non-trivial parameter selection |
| Bootstrapping needed? | Never (single-multiplicative-depth scheme) | Only if you chain many ops; for one dot product, no |
| Security model | Provable IND-CPA under DCR | Lattice-based; ROM IND-CPA. Has known CKKS-passive vs active issues for some workloads (mostly ML training, not retrieval) |
| Rust ecosystem | We have a clean impl on `num-bigint` | `openfhe-rs` 0.2.0 (BSD-2, FFI to OpenFHE C++). Alpha-ish bindings, requires C++ toolchain. No production-quality pure-Rust option |

**Concrete prediction for Bench B query, CKKS version:**

- Query encrypt: ~10 ms (vs 757 ms Paillier single-core)
- 6 candidates × ~20 ms = ~120 ms reranks (vs 280 ms Paillier with rayon)
- Total RemoteRAG overhead: ~130 ms (vs 213 ms current)
- Marginal win at `k' = 6`; **massive win** at production scale (`k' =
  50` or `k' = 100`): CKKS still ~20 ms/candidate, Paillier scales
  linearly with d on the cipher-per-dim encoding.

### Engineering cost

`openfhe-rs` adds C++ build complexity (cmake, OpenFHE 1.5 vendored or
system-installed), MB-sized keys to serialise across the client/server
boundary, and a parameter-selection step (ring dim, modulus chain, scale)
that can produce silently-wrong results if mis-tuned. The protocol shape
of RemoteRAG stays the same — server still holds plaintext doc
embeddings, Stage-1 ANN unchanged, only the Stage-2 rerank backend
changes.

### Decision criteria

Switch to CKKS when:

- `k' > ~20` per query, OR
- Query rate > ~10 QPS sustained, OR
- The Paillier dot product becomes a measurable bottleneck in observed
  end-to-end latency.

Stay on Paillier when:

- Prototype demo / single-digit QPS load.
- The C++ toolchain dependency is a blocker for the build environment.
- A future cleanup of `fast-paillier`'s `glass_pumpkin` dep wedge lands
  upstream and obviates the migration entirely (see §3 below).

---

## 2. The "encrypted ANN" problem — why CKKS doesn't unify RemoteRAG and CAPRISE

> **Status:** structural impossibility argument; not a roadmap item.
> Documented here so future "let's just use FHE" proposals can be
> short-circuited by reading this section.

A natural question after seeing CKKS is whether it lets us encrypt the
*docs* server-side too — getting CAPRISE-style at-rest confidentiality
*and* RemoteRAG-style formal DP on queries. **It doesn't, and the reason
is structural.**

### Why no FHE scheme gives you cheap encrypted ANN

CAPRISE's ANN works because the encryption is **distance-preserving by
construction**: `cosine(enc(a), enc(b)) ≈ cosine(a, b)`. The server can
sort ciphertexts without decrypting. The cost is that *this is also the
leakage* — the server learns the plaintext geometry. That's the
fundamental trade.

CKKS (and every IND-CPA FHE scheme) has the opposite property:
ciphertexts are random-looking lattice elements whose pairwise distances
have **no relationship** to the underlying plaintext distances. That's a
feature for security and a structural barrier to ANN.

If you tried "fully CKKS-encrypted index + CKKS-encrypted query," what
the server can do:

```
For each doc i:
    encrypted_dot_i = ct_q ⊙ ct_d_i      // encrypted × encrypted multiply
                                          // requires relinearisation, costs ~ms
                                          // — and gives you a *ciphertext*, not a rank
```

To get top-k you would need to **compare** these encrypted cosines.
CKKS doesn't natively support comparison; the options are:

1. **Send all `N` encrypted cosines back to the client** to decrypt + sort
   locally. That's `N` decryptions per query and `N × |ciphertext|` of
   network. For `N = 10⁵`, ~10⁵ decryptions × ~1 ms each ≈ 100 seconds
   per query. Doesn't scale.
2. **Implement comparison inside CKKS** via polynomial approximation of
   `sign(a − b)` — costs ~10–30 multiplicative levels per comparison,
   needs bootstrapping, top-k via a sorting network takes many minutes
   per query at `N = 10⁴`.
3. **Switch schemes mid-protocol** — TFHE has cheap comparison via
   programmable bootstrapping but doesn't pack 1024-d floats per
   ciphertext. Wrong fit.

None of these is "fast encrypted ANN." This is why CAPRISE / DCPE /
property-preserving-encryption is a whole research area distinct from
FHE — they're solving a problem FHE structurally can't solve cheaply.

### Could we mix CAPRISE for ANN + Paillier/CKKS for rerank?

The natural "have your cake" question. Sketch:

- Server stores docs in **two forms**: CAPRISE-encrypted (for fast ANN
  with geometry leakage) *and* plaintext (for PHE rerank).
- Stage 1: ANN over CAPRISE ciphertexts (fast).
- Stage 2: Paillier/CKKS dot product over plaintext docs.

It doesn't work for the *security* reason — a server breach now leaks
both the CAPRISE-leakable geometry *and* the plaintext embeddings. Worse
than either pure option.

The only honest hybrid keeps docs **encrypted only once**, and the
encryption has to be both:

- distance-preserving enough for ANN (= CAPRISE-style leakage), and
- operand-shape-compatible with PHE multiplication (= plaintext
  integers).

These requirements are mutually contradictory. The mutual exclusion is
structural, not an artifact of the current implementation.

### What actually solves "encrypted-everywhere ANN"

If the application genuinely needs both encrypted-at-rest docs and
sublinear retrieval, the real options are:

1. **TEE-based (GELO / SEV-SNP path).** Docs encrypted at rest in CVM
   memory; the CVM operates on them in plaintext under hardware isolation.
   This is what `GeloRagInMemoryService` + SEV-SNP gives you — the
   cleanest architectural fit for "fully private but searchable." See
   [`gelo.md`](gelo.md).
2. **Multi-server PIR / split-ANN.** Two or more non-colluding servers;
   queries are split so no single server learns either query or which
   docs match. Heavy protocol cost; libraries like SealPIR, SimplePIR,
   FrodoPIR are doing this for keyword PIR.
3. **FHE-friendly LSH.** See §3 below — an active research area, not yet
   productionable.
4. **Accept CAPRISE-style leakage.** Document it formally as a property
   of the scheme. The DCPE / ORE / SPE literature bounds the leakage
   rather than eliminating it.

The three approaches in this codebase (CAPRISE / RemoteRAG / GELO + SEV-
SNP) are a fairly complete partition of the design space:

- *Accept leaky geometry in exchange for fast server-side ANN* →
  CAPRISE.
- *Accept plaintext doc embeddings in exchange for IND-CPA queries with
  formal DP* → RemoteRAG.
- *Accept hardware trust in exchange for "everything encrypted at rest,
  all crypto happens inside the TEE"* → GELO + SEV-SNP.

CKKS doesn't add a fourth quadrant. It would just make RemoteRAG's
rerank faster within the same threat model.

---

## 3. FHE-friendly LSH — bucketing-with-bounded-leakage research direction

> **Status:** research-stage survey; no implementation budget. Track
> upstream papers; pick up when a candidate construction matures
> enough to bound the bucket-leakage rigorously.

Recent research tries to thread the needle: filter candidates without
revealing geometry, then exact rerank inside FHE. Examples:

- **PRIVANN** — privacy-preserving approximate nearest-neighbour using
  LSH bucketing under HE-friendly hash functions. The bucketing leaks
  only the bucket assignment (not the embedding); within a bucket, exact
  distance is computed in FHE.
- **CHEX-MIX** — hybrid scheme using cleartext LSH on
  pseudonymised/blinded embeddings for the initial filter, then
  CKKS/BFV for rerank on the small candidate set.

These constructions are conceptually closer to what a unified
RemoteRAG+CAPRISE would look like — bounded geometry leakage from the
bucketing step (much less than CAPRISE's full distance-preservation),
exact rerank from the FHE step. They sidestep the "compare on encrypted"
cost by reducing the candidate set first.

**Status as of 2026-Q2:**

- Research-grade. Latencies reported in papers are typically **~seconds
  per query** even at modest corpus sizes.
- **No production-quality Rust implementation.** Reference codebases are
  C++ / Python prototypes tied to specific FHE libraries (SEAL, OpenFHE,
  HElib).
- Parameter selection, security analysis of the LSH-leakage budget, and
  composability with existing DP layers are all open questions.

**Where this could land in our codebase.** A future `crates/private-lsh/`
crate could host an FHE-friendly bucketing layer that sits between the
CAPRISE / RemoteRAG paths, offering a middle quadrant:

- Bucket assignments leak (less than CAPRISE, more than RemoteRAG).
- Within-bucket rerank uses CKKS over encrypted docs (no plaintext
  embedding on server, unlike RemoteRAG).
- The DP-Forward and GELO layers compose unchanged.

This would be the natural follow-on once `openfhe-rs` stabilises, or
when a pure-Rust CKKS implementation reaches production quality.

---

## 4. Migration path for Paillier

Even if we don't switch to CKKS, there is a cleaner migration target than
the current hand-ported implementation:

- **`fast-paillier 0.3.2` upstream cleanup.** The crate is well-maintained
  but currently wedged on `glass_pumpkin 1.9` (transitively pulls yanked
  `core2 0.4`) / `1.10` (incompatible `rand_core` API). Once the upstream
  maintainer bumps `fast-paillier`'s `rand_core` call sites to match
  `glass_pumpkin 1.10`, we can swap our implementation for the crate
  with ~50 LOC of glue.
- **`fast-paillier --features backend-rug`** is a working short-cut if
  LGPL/GMP transitive deps are acceptable for the deployment target.
- **`libpaillier 0.7.0-rc0`** (mikelodder7, Apache-2.0, built on
  `crypto-bigint` + `crypto-primes` + `unknown_order` — clean RustCrypto
  stack, no `glass_pumpkin`) is the only modern pure-Rust alternative
  without the dep wedge, but lacks multi-exp.

The migration is "swap the backend, keep the protocol." `RemoteRagService`
and the `homomorphic_dot` interface stay the same.

---

## 5. Other forward-looking items

- **Shared-`A` multi-text batching for the GELO mask.** Today each text
  in a corpus-ingest batch gets its own session mask `A`. A single
  shared `A` across `B` texts would let one BLIS `cblas_sgemm` call
  handle the whole batch — amortising the ~100 µs thread-launch floor
  across `B·n` rows instead of `n`. **Protocol concern:** with the same
  `A`, the GPU sees `U_i · U_jᵀ = A · H_i H_jᵀ · Aᵀ` for any pair of
  texts in the batch, leaking pairwise text-similarity to the
  untrusted side. For RAG retrieval this is the quantity the public
  index reveals *eventually*, but exposing it at embed time is a new
  threat surface. TwinShield's shield-vector defence may or may not
  extend cross-text under shared `A` — needs a proof. Alternative:
  block-diagonal `A` of `B` independent `(n+k)×(n+k)` blocks, which
  collapses back to `B` separate GEMMs (no amortisation win). The
  practical mid-point — *Approach 2A: embedder-level rayon parallelism
  with independent per-thread `A`* — is already shipped, see
  `gelo-embedder/src/{bert,decoder}/embedder.rs::embed_many`.
- **HNSW over the RemoteRAG plaintext index.** Drop in `hnsw_rs` behind
  the same `Vec<IndexEntry>` interface. Linear sweep works correctly but
  does not scale past ~10⁴ docs.
- **Multi-query batching on the PHE rerank.** Amortise the Stage-2
  homomorphic dot products across a batch of queries to a single corpus,
  cutting per-query cost in proportion to batch size.
- **M5.9 hardware bring-up.** Real SEV-SNP attestation on a Hetzner EPYC
  server with VFIO-passthrough GPU. Once this lands, the DP-Forward
  identity binding (see [`dp-forward.md`](dp-forward.md) §4.2) becomes a
  real production gate.
- **Tighter `δ`** for DP-Forward. The Balle–Wang bisection at `δ = 1e-5`
  is the paper's tested value; moderate corpora warrant `δ ≪ 1/N²`.
  Cost: ~1.5× larger σ.
- **Vec2Text empirical ablation** as a release-gate task, testing both
  the standalone DP path and the GELO + DP defence-in-depth composition.
- **Encrypted-KV-cache on GPU.** Under the current GELO §3 threat
  model the autoregressive KV cache stays host-resident in TEE RAM
  because post-RoPE K/V encode the prompt content the protocol is
  built to hide; moving it to the untrusted GPU would expose
  prompt-derived structure across decode steps. At long context this
  is the binding host-RAM cost (B · max_cache_len · 0.29 GiB on
  Qwen3-4B — e.g. ~5 GiB at B=8 max_cache_len=2052, ~10 GiB at
  B=8 max_cache_len=4100). Two research lines could relax the
  constraint and put the cache in VRAM:
  - *SCX-style encoded-KV* (SIGCOMM '25, has code) — apply a fresh
    per-step orthogonal cover to cached K/V before they leave the
    TEE, similar to GELO's mask but on the cache axis. Adds a
    decode-step round-trip cost in exchange for VRAM residency.
    Estimated 12-month research effort: needs a security argument
    that the per-step refresh closes the cross-step correlation
    leak under Amulet-class equivariance attacks, plus engineering
    integration with the existing `KvCache` API.
  - *GELO §3 re-validation for ciphertext-KV* — directly extend the
    mask-protected-PCIe argument to cache-line storage on GPU.
    Cheaper but riskier: the cache is read repeatedly across all
    decode steps (unlike per-offload activations which are
    masked-and-stripped within one forward), so a Gram-leak on the
    cache reuses the same `A` across many reads. Needs a freshness
    primitive (rolling shield rows, period-π refresh, etc.) and
    the corresponding shield-energy proof on the cache axis.
    Smaller scope (~3-month research spike) but may not yield a
    deployable design.

  Parked because (a) host RAM at 64 GiB is not yet the binding
  constraint on Strix Halo for our production B/max_cache_len; (b)
  the GPU-side win is bounded — the cache is read repeatedly but
  the read is in-TEE-attention (not GPU matmul) under the current
  M1.11 design. The unlock is only meaningful if R1.4
  batched-attention-on-GPU also lands, because then the GPU would
  consume the cache directly and host-VRAM transit per decode step
  becomes the bottleneck. See M1.11 plan §3.1 and the
  2026-05-22 handoff for the residency table that frames this.

---

## References

- Cheon, J.H., Kim, A., Kim, M., Song, Y. *Homomorphic Encryption for
  Arithmetic of Approximate Numbers (CKKS).* ASIACRYPT 2017.
- OpenFHE-rs: <https://github.com/openfheorg/openfhe-rs>
- OpenFHE: <https://github.com/openfheorg/openfhe-development>
- LFDT-Lockness. *fast-paillier 0.3.2.*
  <https://github.com/LFDT-Lockness/fast-paillier>
- mikelodder7. *libpaillier 0.7.0-rc0.*
  <https://github.com/mikelodder7/paillier-rs>
- Boemer, F. et al. *CHEX-MIX: Combining Homomorphic Encryption with
  Trusted Execution Environments for Two-Party Oblivious Inference in
  the Cloud.* arXiv:2104.03742.
- PRIVANN line: see e.g. Kim, M. et al. *Secure Approximate Nearest
  Neighbor Search with Locality-Sensitive Hashing under FHE.* (various
  workshop papers 2022–2024).
- Property-preserving / DCPE / ORE literature surveys for the leakage
  framework underlying CAPRISE-style schemes.
