# remote-rag

Rust implementation of the [RemoteRAG](https://arxiv.org/abs/2412.12775)
two-stage retrieval protocol — Cheng et al., ACL Findings 2025.

> 📝 **License note**: this crate depends on `fast-paillier` 0.3.2 (MIT OR
> Apache-2.0) configured with its default `backend-num-bigint` backend —
> the pure-Rust path. The crate's *optional* `backend-rug` feature pulls in
> LGPL `rug`/GMP, but we do not enable it; closed-source static linking is
> unblocked.

## Threat model

Different from CAPRISE-at-rest / DP-Forward / GELO — see
`docs/prototype/gelo.md` and the design doc at
`docs/prototype/dp-forward-remote-rag.md` (M6.8) for the full table.

In one sentence: the server holds **plaintext** document embeddings (chunk
text is still AES-GCM-encrypted); the client's *query* gets `(n, ε)`-
DistanceDP via planar-Laplace noise (Stage 1) plus exact rerank against a
Paillier-encrypted clean query (Stage 2). PHE rerank is what recovers
~100 % recall@k under the noise.

This is **mutually exclusive with CAPRISE-encrypted-index** — Paillier's
additive homomorphism evaluates over plaintext exponents, and CAPRISE's
`s·e + r·u` form destroys that structure. A deployment chooses either
RemoteRAG-PHE-rerank *or* CAPRISE-at-rest, not both.

## Modules

- `planar_laplace` (M6.3) — Stage-1 query perturbation: `r ~ Gamma(n, 1/ε)`,
  direction uniform on `S^{n-1}`. **`n` is the embedding dimension**, not
  a cluster count — a famously common misreading.
- `paillier` (M6.4 — pending) — fixed-point quantization, Enc/Dec, and
  homomorphic dot-product `Σᵢ Enc(qᵢ)^{e_d[i]_q}`.
- `service` (M6.5 — pending) — `RemoteRagService`, a parallel service to
  `Approach4InMemoryService` that runs the two-stage protocol.
