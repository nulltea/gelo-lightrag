# DP-Forward Prototype

> **Scope.** Design document for the DP-Forward (Recipe B) layer in
> `crates/dp-forward` and the `dp-forward` Cargo feature of
> `crates/gelo-embedder`. Documents the *what and why*, not the *how* — for
> source-level details see crate-level rustdoc. Companion documents:
> [`remote-rag.md`](remote-rag.md) for the RemoteRAG protocol that consumes
> these primitives doc-side, [`gelo.md`](gelo.md) for the GELO + SEV-SNP
> substrate this composes with, and [`future-rnd.md`](future-rnd.md) for
> research directions.

---

## 1. Background

DP-Forward (Yue et al., CCS 2023, [arXiv 2309.06746](https://arxiv.org/abs/2309.06746))
adds calibrated Gaussian noise to the output of a transformer layer to give
a formal **`(ε, δ)`-Sequence-LDP** guarantee — every embedding the system
releases is "noisier than this measurable amount", and any adversary who
recovers the noisy embedding cannot distinguish the original input beyond
the privacy budget. Concretely, for a vector `x`:

```
clip   :  x ← x · min(1, C / ‖x‖₂)        (per-row L2 bound C)
sigma  :  σ = Δ₂ / R(ε)                    (Balle–Wang analytic Gaussian)
release:  x̃ = x + N(0, σ² · I)             (irreversible noise)
```

where `Δ₂ = 2·C` is the post-clip sensitivity and `R(ε)` is the solution of
the Balle–Wang delta-balancing equation found by bisection. Yue et al.
report that DP-Forward at `(ε, δ) = (4, 10⁻⁵)` reduces Vec2Text
embedding-inversion BLEU by ~88 percentage points with under 2 pp utility
loss on SST-2 / QNLI.

The mechanism is **irreversible by construction**: unlike keyed
distance-preserving schemes (CAPRISE / DCPE / SAP), the Gaussian noise has
no key that subtracts it back out. That is what gives the formal DP
guarantee against a key-holder adversary, and it is what makes DP-Forward
compose cleanly with any downstream encryption.

---

## 2. Threat model and what DP-Forward protects

| Layer | Protects against | Failure mode when removed |
|---|---|---|
| AES-GCM chunks (`AesChunkCipher`) | Server / disk reader without the chunk key | Plaintext document text |
| CAPRISE / SAP (key-removable noise, distance-preserving) | Non-key-holder doing inversion on stored vectors | Clipped clean embedding `e` → Vec2Text-style text reconstruction |
| **DP-Forward** — irreversible Gaussian on pooled output | The *key-holder itself*; bounds `(ε, δ)`-SeqLDP | Tight numerical embedding inversion |
| GELO + SEV-SNP CVM | GPU / host observer of intermediate activations | Per-token activation streams |

DP-Forward is **strictly additive** to every other layer. It fills the gap
that CAPRISE structurally cannot: a key-holding adversary who decrypts the
index recovers `e + DP_noise`, not the clean `e`.

### Composition with CAPRISE

CAPRISE's noise is *keyed*: the key-holder can subtract it out and recover
`e`. That's correct behaviour against a server with no key. But if the
key is ever exposed (TEE breach, internal compromise, legal subpoena,
key-rotation bug), the attacker recovers `e` exactly and can run Vec2Text
on it. DP-Forward adds an *irreversible* term that survives CAPRISE
decryption:

```
DP-Forward emits         e + DP_noise                           (released by embedder)
CAPRISE encrypts         scale · (e + DP_noise) + crypto_noise  (in the index)
CAPRISE-key holder gets  (e + DP_noise)                         (post-decryption)
```

The key-holder never recovers `e` exactly — `e + DP_noise` is the only
thing that ever leaves the embedder. That residual is the formal DP
guarantee, intact through any number of cryptographic transforms.

### Composition with GELO

GELO protects activations *during inference* against an untrusted
GPU/host. DP-Forward applies *after* the pooled embedding is computed,
inside the same CVM. Both protections compose:

- GELO ensures intermediate Q/K/V/O/FFN tensors never leak to the GPU.
- DP-Forward ensures the pooled embedding the CVM releases is bounded in
  what it can reveal about the input.
- SEV-SNP attestation (M5) binds the report to the DP parameters — see §4.2.

---

## 3. Implementation scope

### Crate layout

```
crates/dp-forward/                DP-Forward paper primitives only (aMGM)
       ▲              ▲
       │              │
crates/gelo-embedder  crates/remote-rag
(feat: "dp-forward")  (consumed for doc-side noise at ingestion;
   defence-in-depth    see remote-rag.md)
```

- **`dp-forward`** is an independent crate that implements *only* the
  Yue-et-al. paper primitives: clip, calibrate σ via Balle–Wang bisection,
  add Gaussian noise, plus a `DpForwardConfig` value type whose
  `config_digest()` is the 32-byte SHA-256 over `(ε, δ, C, σ)` used for
  attestation binding. It does not depend on `rag_core`, has no notion of
  an `Embedder`, and lives at the bottom of the dependency graph so both
  consumers reuse the same math.
- **`gelo-embedder`** consumes `dp-forward` behind the optional Cargo
  feature `dp-forward`. When enabled, `GeloBertEmbedder` and
  `GeloQwenEmbedder` gain a `with_dp_forward(cfg)` builder; every call to
  `embed()` then applies clip + Gaussian noise to the pooled embedding
  before returning, using an `OsRng`-seeded ChaCha20 RNG dedicated to the
  DP path. The embedder's `model_identity()` rebinds to
  `hex(sha256(weights_id ‖ cfg.config_digest()))` so the SEV-SNP
  attestation report's `REPORT_DATA[0..32]` commits to the DP parameters.

### What's covered

- `DpForwardConfig::calibrate(ε, δ, C)` → memoised σ. Golden-value test
  locked to Balle–Wang's Table-1 entry σ ≈ 1.081 at `(ε=4, δ=1e-5, Δ=1)`.
- `GeloQwenEmbedder` / `GeloBertEmbedder` with `with_dp_forward` builder.
- Identity rebinding test against the SEV-SNP mock issuer + verifier.

---

## 4. Key design choices

### 4.1 DP-Forward primitives live in their own crate, scoped narrowly

`crates/dp-forward` implements *only* the Yue-et-al. paper primitives —
clip, Balle–Wang σ, Gaussian noise, config digest. Three deliberate
non-inclusions:

- **No planar-Laplace.** That mechanism is from the RemoteRAG paper and
  belongs in `crates/remote-rag`; combining the two under a `gelo-dp`
  umbrella conflates two different research lines and confuses callers
  about which crate owns which math. See [`remote-rag.md`](remote-rag.md) §3.
- **No `DpForwardEmbedder<E>` wrapper.** An external wrapper cannot be
  attested — a malicious operator could replace the wrapper with an
  identity transform between the embedder and the relying party. Baking
  DP into `gelo-embedder` lets the CVM commit to the DP parameters in
  its SEV-SNP report.
- **No `rag_core` dependency.** The crate's only callers are
  `gelo-embedder` (feature-gated) and `remote-rag` (for document-side
  noise at ingestion); forcing `statrs` into `rag_core` consumers who only
  want CAPRISE would be wrong.

### 4.2 DP-Forward folds into `Embedder::model_identity` for attestation binding

The `Embedder` trait has long had `model_identity(&self) -> &[u8]` for the
weights-only case. When the `dp-forward` feature is on and
`with_dp_forward(cfg)` was called, the embedder's `model_identity` becomes

```
hex(sha256(weights_identity || cfg.config_digest()))
```

Because SEV-SNP's `REPORT_DATA[0..32] = sha256(model_identity.as_bytes())`,
the attestation report commits to the DP parameters automatically. A
relying party who pins `expected_model_id` for a specific `(weights, ε,
δ, C, σ)` tuple immediately catches both:

- **Parameter substitution** (different `ε` or `δ`) — the digest is
  defined over all four fields.
- **Calibration substitution** (matching `ε, δ` but a manipulated `σ`) —
  `sigma` is included in the digest explicitly, so a CVM that misreads
  the bisection result and runs with the wrong noise scale also fails
  the report-data check.

The locking test (`crates/gelo-embedder/tests/dp_forward_attestation.rs`)
mints a mock report under `cfg_a` and verifies that a verifier expecting
`cfg_b`'s digest rejects it cleanly.

### 4.3 The DP RNG is *not* deterministic across runs

`gelo-embedder` seeds the DP-noise RNG from `OsRng` at construction time
and exposes no `with_seed` override on this path. That's intentional:
**deterministic DP noise voids the DP guarantee.** An adversary who can
guess (or observe) the RNG state can subtract out the noise term and
recover the clean embedding. The mask / shield / U-Verify RNGs that GELO
uses for activation masking are seedable because their properties allow it
— those are *per-batch fresh* anyway. The DP noise is the long-lived
release that has to look statistically unique forever.

(`dp-forward::amgm` itself accepts any `RngCore`, so tests can pass a
seeded ChaCha for property-level checks — but the integrated
`embed()` path uses `OsRng`.)

---

## 5. Verification and current results

| Test | What it asserts |
|---|---|
| `dp-forward::amgm::calibrate_sigma_at_ref_config` | σ ≈ 2.1623 at `(ε=4, δ=1e-5, Δ=2)`; σ ≈ 1.0811 at `Δ=1` — matches Balle–Wang Table 1 |
| `dp-forward::amgm::calibrate_sigma_scales_linearly_with_sensitivity` | `σ(2Δ) = 2·σ(Δ)` to f64 precision |
| `dp-forward::amgm::noise_empirical_std_matches_sigma` | 10⁴-sample empirical σ within ±0.02 of nominal |
| `dp-forward::config::digest_differs_when_epsilon_differs` | DP digest covers the privacy budget |
| `gelo-embedder::dp_forward_attestation::dp_config_rebinds_model_identity` | Different ε ⇒ different `model_identity` bytes |
| `gelo-embedder::dp_forward_attestation::mock_report_with_dp_binding_round_trips` | Real SEV-SNP mock issuer + verifier path accepts a matched DP binding |
| `gelo-embedder::dp_forward_attestation::mock_report_is_rejected_under_mismatched_dp_config` | Verifier with `expected_model_id` from `cfg_b` rejects a report issued under `cfg_a` |

### Measured overhead (`obfuscation_bench` `--release`, Qwen3 on Vulkan)

On the apples-to-apples bench against `GeloQwenEmbedder` + CAPRISE
baseline:

| Metric | GELO + CAPRISE | GELO + DP-Forward + CAPRISE | Δ |
|---|---|---|---|
| Ingest (4 docs) | 587 ms | 591 ms | **+3 ms** |
| Per-doc | 146.9 ms | 147.7 ms | +0.8 ms |
| Query | 134.8 ms | 131.3 ms | within noise |

DP-Forward overhead is **sub-1 %** on any real workload. At d=1024 the clip
+ Gaussian sample is single-digit microseconds per embedding; inference
dominates.

---

## 6. Risks and proposed fixes

### Risk: Sensitivity bound `C` is a hyperparameter

Too small ⇒ clipped embeddings cluster on the boundary of the L2 ball and
retrieval utility tanks (because all unit-norm BGE / Qwen3 embeddings are
on a sphere; clipping below 1 *moves* them). Too large ⇒ sensitivity goes
up and σ goes up, washing the signal out.

**Fix.** Default `C = 1.0` (correct for L2-normalised embedders like
Qwen3-Embedding and BGE). Document in `DpForwardConfig` that callers using
non-normalised embedders should set `C ≈ max‖e‖₂` on a calibration corpus.
Operationally: log the post-clip rate (fraction of embeddings whose norm
exceeded `C` and got scaled down) so a deployment can spot when its `C` is
too tight.

### Risk: Attestation binding only catches *recorded* CVMs

If a relying party never pins `expected_model_id`, the attestation report
still contains the DP-bound identity, but nothing forces the verifier to
check it. A misconfigured deployment could ship a CVM with the right
weights but a vacuous DP config and the verifier would happily accept.

**Fix.** The `Approach4InMemoryService::with_snp_verifier` constructor
(M5) accepts a fully-configured `SnpAttestationVerifier`. The intended
operator workflow is to compute `expected_model_id` *offline* from the
pinned `(weights manifest hash, DpForwardConfig)` pair using the same
hash chain the embedder uses (see
`crates/gelo-embedder/tests/dp_forward_attestation.rs::dp_bound_model_id_bytes`),
then load it into the verifier. The test file's helper is intentionally
copy-pastable as a deployment script.

### Risk: Vec2Text-attack regression coverage is not automated

A regression that breaks DP-Forward's defence empirically without breaking
the σ-calibration tests is possible (e.g. if the noise is added in the
wrong axis).

**Fix.** The existing tests assert `output ≠ no-DP output` and `output
mean ≈ no-DP output to O(σ/√N)`, which catches the "noise is being
applied" and "noise has the right scale" failure modes. A real Vec2Text
ablation belongs in a separate release-gate workflow; we explicitly do
not pay its cost on every PR. See [`future-rnd.md`](future-rnd.md) for the
planned release-gate addition.

---

## 7. Forward-looking work

- **Tighter `δ`.** The Balle–Wang bisection at `δ = 1e-5` is the paper's
  tested value, but moderate-sized embedding corpora (10⁴ docs) warrant
  `δ ≪ 1/N²`, i.e. `δ ≤ 1e-9`. Cost is ~1.5× larger σ — manageable.
- **Vec2Text empirical ablation** as a release-gate task, testing both the
  standalone DP path and the GELO + DP defence-in-depth composition.
- **M5.9 hardware bring-up** lands real SEV-SNP attestation on a Hetzner
  EPYC server. The DP-Forward identity binding then becomes a real
  production gate: a CVM running on bare-metal EPYC with a fresh-VCEK
  report can prove to any relying party that it loaded the attested
  weights with the attested DP parameters, no third-party trust required.

See [`future-rnd.md`](future-rnd.md) for the broader research roadmap.

---

## References

- Yue, X., Du, M., Wang, T., et al. *DP-Forward: Fine-tuning and Inference
  on Language Models with Differential Privacy in Forward Pass.* CCS 2023.
  [arXiv:2309.06746](https://arxiv.org/abs/2309.06746)
- Balle, B., Wang, Y.-X. *Improving the Gaussian Mechanism for
  Differential Privacy: Analytical Calibration and Optimal Denoising.*
  ICML 2018. [arXiv:1805.06530](https://arxiv.org/abs/1805.06530)
- xiangyue9607/DP-Forward — reference implementation:
  <https://github.com/xiangyue9607/DP-Forward>
