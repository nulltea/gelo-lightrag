# dp-forward

Rust implementation of the [DP-Forward](https://arxiv.org/abs/2309.06746)
analytic Matrix Gaussian Mechanism (aMGM) — Yue et al., CCS 2023.

Scope intentionally narrow: this crate provides only the primitives from
the DP-Forward paper.

- `amgm::clip_l2_in_place(v, C)` — per-row L2 clipping.
- `amgm::calibrate_sigma(ε, δ, Δ)` — Balle–Wang analytic-Gaussian σ.
- `amgm::add_gaussian_noise(v, σ, rng)` — isotropic `N(0, σ²I)` noise.
- `DpForwardConfig::{calibrate, config_digest}` — calibrated config + the
  32-byte digest folded into `Embedder::model_identity` for SEV-SNP
  attestation binding.

The crate has **no dependency on `rag-core`** and does not wrap any
`Embedder`. Two callers consume it:

- `gelo-embedder` (feature `dp-forward`) — defence-in-depth path: applies
  the mechanism inside the attested CVM after pooling, before returning.
- `remote-rag` — optional document-side aMGM noise at ingestion, alongside
  its own internal planar-Laplace mechanism on the query side.

Why no `DpForwardEmbedder<E>` wrapper here: an external wrapper cannot be
attested. Baking DP into `gelo-embedder` lets the CVM commit to the DP
parameters in its SEV-SNP report — a strictly stronger statement.
