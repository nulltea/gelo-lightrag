# GELO Split-Inference Prototype

> **Scope.** Design document for the private-embedding prototype implemented in
> `crates/gelo-protocol`, `crates/gelo-embedder`, `crates/gelo-gpu-wgpu`,
> `crates/gelo-tee-sev-snp`, and `crates/gelo-snp-runner`. Documents the
> *what and why*, not the *how*. For source-level details see crate-level
> rustdoc; for the research context that motivated this design see
> `docs/research/private-embedding-research.md` §D Recipe D.

---

## 1. Background

The prototype implements a **TEE-anchored, GPU-accelerated** embedding model
under a threat model where the GPU is untrusted but the TEE is. The two
academic ingredients are GELO and TwinShield.

### GELO (Belikov & Fedotov, arXiv 2603.05035)

GELO is a single-batch obfuscation protocol for linear layers in transformer
inference. The trusted side holds the activations `H` and a public weight `W`
known to both parties; the goal is to compute `H · W` with the heavy
matrix-multiply on an untrusted accelerator without revealing `H`.

The construction is one line of linear algebra:

```
A  ← orthogonal random matrix, fresh each batch     (TEE)
U  ← A · H                                          (TEE; ship to GPU)
V  ← U · W                                          (GPU)
HW ← Aᵀ · V          // since Aᵀ·A·H·W = H·W        (TEE; recover)
```

Three properties make this work as a privacy primitive:

1. **`A` never touches the GPU.** The accelerator sees only `U = A·H` and the
   public `W`. For a fresh-per-batch orthogonal `A` drawn uniformly from `O(n)`,
   `U` is information-theoretically a random rotation of `H` — every `H` is
   equally consistent with the observed `U`.
2. **`W` may be public.** The protocol is designed for **openweight** model
   deployment. Attacks like ArrowMatch that exploit knowledge of `W` against
   masking schemes that combine `W` with the mask (e.g. STIP, ObfuscaTune) do
   not apply here — `A` is on the activation axis, never mixed with weights.
3. **Bit-exact recovery in IEEE-754** (modulo floating-point associativity).
   `Aᵀ = A⁻¹` for orthogonal `A`, so the recovery is one matmul and the round-trip
   error is at the f32 cancellation floor, not protocol-induced drift.

GELO doesn't cover Q·Kᵀ (both operands are runtime values, no public weight)
and doesn't address two leak channels that follow from the bare construction:

- **Gram leak**: `UᵀU = HᵀA·Aᵀ·H = HᵀH`. The masked output's row-Gram matrix
  exactly equals the cleartext's. An adversary with multiple batches can chain
  Gram matrices to mount a BSS-style ICA reconstruction.
- **Engine tampering**: the GPU could return garbage and the TEE would not know.

### TwinShield (Xue et al. 2025)

TwinShield extends GELO with three primitives that close the gaps:

- **Shield rows.** Splice `k` random rows of energy `σ ≈ 4–8 ·  mean‖h‖` into `H`
  before masking, strip them after recovery. The Gram leak becomes
  `UᵀU = HᵀH + SᵀS` where `S` is the per-batch shield — `HᵀH` is no longer
  isolated and a multi-batch BSS attack cannot reduce to ICA on `H`.
- **OutAttnMult.** A 4-partition embedding that lets the GPU compute Q·Kᵀ
  without recovering either operand. Both Q and Kᵀ are masked additively
  with fresh random matrices `R_Q`, `R_Kt` and scaled by random scalars `a, b`;
  they are then stacked into `2n × 2n` operands plus row/column permutations.
  The product decomposes into four partitions that the TEE recombines into
  `Q·Kᵀ` using the secret `(R_Q, R_Kt, a, b, λ_Q, λ_K)`.
- **U-Verify.** A Freivalds-style integrity probe: for asserted `Z = A·B`,
  pick random `r ∈ {−L..L}^p`, check `A · (B·r) ≈ Z · r`. Probability of
  missing a tamper per probe is `≤ 1/(2L)`; running `k` independent probes
  brings it to `(2L)^-k`. At `L=3, k=8` that is `≈ 2.4·10⁻⁷`.

The combination — fresh orthogonal mask + shield rows + U-Verify for the
public-weight matmuls, plus OutAttnMult for Q·Kᵀ — is what this prototype
implements.

---

## 2. Threat model

The prototype targets the **openweight embedding** scenario:

| Component | Trust | Visible to it |
|---|---|---|
| User-side text | Confidential | — |
| TEE (SEV-SNP CVM) | Trusted | text, activations, mask state, model weights, embeddings |
| GPU + driver + PCIe | Untrusted | public model weights, per-batch masked activations, integrity-probed matmul results |
| TEE host (CVM operator) | Untrusted | encrypted CVM memory, masked PCIe traffic |
| Network operator | Untrusted | TLS-wrapped requests + attestation evidence |
| Vector store | Untrusted | per-vector ciphertext (handled by approach4's SAP/CAPRISE schemes — out of scope here) |

**The asymmetry we lean on**: model weights are public. The user pulls a
specific openweight model (e.g. `Qwen/Qwen3-Embedding-0.6B`) by name + revision;
the SHA-256 of the safetensors bytes is published. There is no weight-side
secret to protect. The threat is **user-activation confidentiality flowing
through publicly-known weights**, not the converse.

This single assumption is load-bearing. Two consequences:

1. **Mask design is constrained but tractable.** With public `W`, mixing `W`
   into the mask is unsafe (ArrowMatch class of attacks). GELO's
   activation-axis-only mask is the only construction in the literature that
   stays sound under this assumption — and our implementation relies on it.
2. **The "trusted GPU" requirement collapses.** Confidential-compute GPUs
   (H100 CC, MIG CC) exist but cost an order of magnitude more than commodity
   silicon. Because the GPU only ever sees masked activations and public
   weights, an attested confidential GPU is not necessary. **Consumer GPU
   passthrough into a SEV-SNP CVM is sufficient.**

---

## 3. Protocol design

### 3.1 Substrate abstraction

Two traits define the protocol boundary (`crates/gelo-protocol/src/substrate.rs`):

```rust
trait GpuOffloadEngine: Send {
    fn register_weight(&mut self, h: WeightHandle, w: ArrayView2<f32>) -> Result<()>;
    fn matmul(&self, h: WeightHandle, input: ArrayView2<f32>) -> Result<Array2<f32>>;
    fn matmul_dynamic(&self, lhs: ArrayView2<f32>, rhs: ArrayView2<f32>) -> Result<Array2<f32>>;
    fn matmul_dynamic_batched(&self, lhs: ArrayView3<f32>, rhs: ArrayView3<f32>) -> Result<Array3<f32>>;
}

trait TrustedExecutor {
    fn provision_weight(&mut self, h: WeightHandle, w: ArrayView2<f32>) -> Result<()>;
    fn provision_weight_shared(&mut self, h: WeightHandle, w: Arc<Array2<f32>>) -> Result<()>;
    fn offload_linear(&mut self, h: WeightHandle, hidden: ArrayView2<f32>) -> Result<Array2<f32>>;
    fn offload_qkv(&mut self, layer: u16, hidden: ArrayView2<f32>) -> Result<(_, _, _)>;
    fn offload_attention_qkt_batched(&mut self, q: ArrayView3<f32>, kt: ArrayView3<f32>) -> Result<Array3<f32>>;
}
```

The split is deliberately narrow: the engine knows nothing about masking,
verification, or even what it's computing. The executor owns all secret state
(mask RNG, shield config, scheme identity, U-Verify weight cache) and decides
per-call what to ship.

Three impls cover the matrix of real deployments and test substrates:

| Impl | Where | Role |
|---|---|---|
| `RayonCpuEngine` | `gelo-protocol::sim` | Reference offload backend on CPU; used as parity baseline and for the in-TEE attention fallback. |
| `WgpuVulkanEngine` | `gelo-gpu-wgpu` | Real Vulkan compute via `wgpu` + `cubecl-matmul`. The production GPU backend. Vendor-agnostic — works on AMD, Intel, Nvidia. |
| `InProcessTrustedExecutor` | `gelo-protocol::sim` | The protocol engine. Owns the mask RNG + shield + U-Verify; delegates the underlying GEMM to *any* `GpuOffloadEngine`. |

The `SnpTrustedExecutor` (next section) wraps `InProcessTrustedExecutor` and
adds the attestation boundary — every protocol method is a single forward
to the inner executor. The math is identical at every tier.

### 3.2 GELO mask layer

`crates/gelo-protocol/src/mask.rs` samples a fresh orthogonal `A ∈ ℝ^(n×n)`
each batch via Householder reflectors from a Gaussian seed, seeded from a
`ChaCha20Rng`. Cost is `O(n²·d)` for the apply/unapply, `O(n²)` for sampling
— negligible against the offloaded matmul.

Each block of the encoder/decoder forward pass spends mask machinery on:

```
6 offloaded matmuls per layer (Q, K, V, O, FfnUp/Gate, FfnDown)
× 28 layers (Qwen3-0.6B) or 12 (BGE-small)
× (one mask-sample + mask-apply per offload group + mask-unapply per result)
```

A single `A` is reused across Q, K, V within a block (they read the same `H`),
saving two samples per block. O, FfnUp, FfnGate (where present), FfnDown each
get their own fresh `A`.

### 3.3 Shield rows

`crates/gelo-protocol/src/shield.rs` realises TwinShield's defence against
the Gram leak. Default config in production: `k = 8` rows at `energy = 6 ·
mean‖h‖`. The bench `tests/bss_recovery.rs` confirms FastICA cannot recover
the masked Gram with shielding on; without it, recovery is direct (test
asserts this as a regression guard, not a feature).

### 3.4 U-Verify

`crates/gelo-protocol/src/integrity.rs` implements Freivalds probing. Runs
*after* each offload, against the cached weight (or the runtime operand, for
OutAttnMult). Soundness scales as `(2L)^-k`; we ship `L = 3` and let the
deployment pick `k`. Tested settings:

- `k = 0`: off (default in benches that aren't testing tamper detection).
- `k = 2`: `≈ 2.8%` undetected-tamper rate. Used in the long-running
  `qwen3_overhead_bench` to amortise the integrity-cache cost.
- `k = 8`: `≈ 2.4·10⁻⁷`. Production setting per TwinShield §V-C.

The integrity check requires the TEE to keep a copy of every offloaded weight
for the `B·r` step. This is a memory cost (§5.2 covers the Arc-share design
choice that recovers it).

### 3.5 OutAttnMult and the length-based auto-switch

OutAttnMult handles `Q · Kᵀ` where both operands are runtime values. Q is
masked additively `(Q + R_Q)` and `a·R_Q`; Kᵀ likewise. The TEE recombines
the four partitions of the engine's `(2n × 2n)` output into `Q·Kᵀ`.

Cost decomposition (per layer, per head):

- **GPU work**: one `(2n, d) × (d, 2n) → (2n, 2n)` matmul = **4× the FLOPs**
  of the unprotected `(n, d) × (d, n) → (n, n)` attention matmul.
- **TEE work**: sample masks, stack the 2n-wide operands, apply permutations,
  recover the four partitions. CPU-side, no GPU dispatches.

**Key design choice — length-based auto-switch.** The 4× FLOP widening and
the CPU stacking overhead are a net loss at short sequence lengths and a
net win at long ones. The crossover follows from a simple FLOP-balance
argument:

```
Attention compute per layer  ≈  num_heads · n² · head_dim    (O(n²·d))
One linear projection        ≈  n · hidden² ≈ n · d²         (O(n·d²))
```

Attention starts matching one projection's work at `n ≈ d`. Below that,
attention is a small share of the per-layer compute and CPU-side `(n²·d)` is
faster than shipping a 4× widened matmul to the GPU and paying its dispatch
overhead. Above it, attention's quadratic dominates and GPU offload at 4× wins.

Implementation (`DecoderConfig`):

```rust
pub use_out_attn_mult: bool,                     // master switch (default true)
pub out_attn_mult_min_seq_len: Option<usize>,    // threshold (None → hidden_size)

fn out_attn_mult_enabled_for(&self, n: usize) -> bool {
    self.use_out_attn_mult && n >= self.out_attn_mult_threshold()
}
```

The dispatch site reads `n` per forward pass and picks the path. For
`Qwen3-Embedding-0.6B` (d=1024) on embedding inputs (n in the tens), the
auto-switch resolves to "off" — attention runs in-TEE, Q and K never cross
PCIe. For a long-context decoder LLM with n in the thousands, it engages
automatically.

This is the **only structural deviation from TwinShield's paper**. The paper
recommends OutAttnMult unconditionally; our measurements at embedding shape
make the auto-switch the right default. The privacy story is identical at
both ends: in-TEE attention is strictly more confidential (Q, K never visible
to the GPU at all), and OutAttnMult is a performance lever for the regime
where in-TEE attention becomes the bottleneck.

### 3.5b Permutation-shielded attention (Tier 1, Amulet-inspired)

A third attention path lives at `causal_gqa_attention_permuted` and the
protocol-level `TrustedExecutor::offload_attention_permuted`. It exploits the
softmax-permutation equivariance identity from Amulet (arXiv 2512.07495):

```
softmax(π·Q·Kᵀ·πᵀ / √d) · π·V  =  π · softmax(Q·Kᵀ / √d) · V
```

so a fresh-per-batch row permutation `π_b ∈ S_n` lets softmax, both attention
matmuls, and the optional Gaussian σ-noise (Hidden No More mitigation, arXiv
2505.18332) all run on the GPU under obfuscation. Causal mask is transformed
to `mask'[i,j] = -inf if perm[j] > perm[i] else 0` and added to scores TEE-side
between the engine matmul and engine softmax dispatch.

Phase-by-phase work logged in commits 3b5b587 (math), 3a47056 (substrate),
0f7f239 (engine routing), d6fbade (causal mask), 51a4cc3 (decoder wiring).
Math verified by `tests/permutation_attention.rs` (12 tests: bit-exact
σ=0 equivariance, σ=0.01 bounded drift, causal mask parity, σ-scaling
sanity, Gram-leak documentation, shield-row negative result). Engine
parity tests in `crates/gelo-gpu-wgpu/tests/parity.rs`.

**Measured wall-clock at embedding shape (Qwen3-Embedding-0.6B, n≈400,
NFCorpus 100-doc bench, parallel rayon, AOCL-BLIS):**

| Configuration | Qwen3 + mask wall | vs plain | Notes |
|---|---:|---:|---|
| In-TEE attention (default) | 153 ms/text | 1.27× | Phase 3 baseline |
| Permuted attention (`BEIR_PERM_ATTN=1`) | 300 ms/text | 2.49× | **regresses** at this shape |

**Why permuted attention regresses on this workload**: each forward
adds 3 extra engine dispatches per layer (matmul + softmax + matmul) ×
28 layers = 84 GPU sync points per text. On the integrated GPU (no
discrete PCIe), each sync is small but adds up to ~80 ms/text of pure
dispatch overhead. Meanwhile the in-TEE attention bucket is already
only ~60 ms aggregate across all texts (~0.3 ms/text amortized). So
moving attention off-TEE pays dispatch cost much greater than the
in-TEE compute it replaces.

This is structurally the same trade-off OutAttnMult has (4× FLOP
widening loses at short n); the permuted-attention threshold knob
defaults to `Some(64)` so the path engages at n ≥ 64 when its master
switch is on, leaving the user to opt in via
`DecoderEmbedder::with_perm_attention(true)`. The default keeps it off.

**Where it does win**: long-context decoder workloads (n in the
thousands), where attention is O(n²·d) and dominates per-layer cost.
At that regime the in-TEE attention bucket grows superlinearly while
the engine dispatch cost stays constant — the breakeven point is
~n = 1000-2000 on this hardware, beyond what the embedding benchmark
exercises. Long-context deployments should turn the switch on.

The protocol is correct and the engineering is reusable; it's a tool
in the toolbox rather than the new default.

### 3.6 Sensitive-layer exclusion

Per GELO §3.2, the embedding lookup, final pooling head, and (configurably)
the first / last transformer layers run entirely in the TEE — no offload.
The rationale is empirical: the first and last layers are the easiest
targets for inversion attacks because they sit closest to the cleartext.
Skipping their GEMMs costs a small fixed amount of TEE compute and removes
the leakiest matmuls from the GPU's view.

`DecoderConfig::skip_first_layers` / `skip_last_layer` knob it; default is
no skipping (modern shielding + per-batch fresh mask make the case for
skipping weaker, but the lever is kept for higher-assurance configs).

---

## 4. TEE substrate

### 4.1 Choosing SEV-SNP

Intel TDX and AMD SEV-SNP are the two production confidential-VM technologies
the prototype could target. We picked SEV-SNP for four reasons:

1. **Ecosystem.** The Rust SEV ecosystem is mature: `virtee/sev` (Apache-2.0)
   handles the full report ABI, ioctls, and signature verification. TDX's
   `tdx-quote` is AGPL — usable, but a license footgun for an Apache-2.0
   workspace. The `sev` crate is AMD-blessed and audited.
2. **Attestation chain.** SEV-SNP uses a single chain `ARK → ASK → VCEK →
   report`, with VCEKs fetched from AMD's KDS endpoint. TDX uses DCAP with
   multi-component collateral on quarterly rotation. Half the moving parts.
3. **Hardware path.** Development is on an AMD Strix Halo box (no SEV-SNP
   silicon, but same vendor). Production target is a Hetzner EPYC Genoa /
   Turin dedicated server — readily available at ~$50–150/mo vs. ~$5800/mo
   for managed-cloud confidential-GPU SKUs.
4. **Performance, security, managed-cloud availability** are comparable
   between the two. The differentiators above were the deciders.

### 4.2 Attestation flow

`crates/gelo-tee-sev-snp` implements both sides:

- **Issuer side** (inside the CVM). `SnpTrustedExecutor::evidence(nonce)`
  builds a 64-byte `REPORT_DATA`:

  ```
  REPORT_DATA[0..32]  = SHA-256(model_identity)
  REPORT_DATA[32..64] = SHA-256(scheme_identity || optional_nonce)
  ```

  `model_identity` is the hex-encoded SHA-256 of the loaded safetensors —
  binds the running CVM to specific publicly-known weights.
  `scheme_identity` covers protocol-secret state (mask seed config, shield
  config, OutAttnMult policy). The nonce binds the report to a session
  challenge.

  The 1184-byte SEV-SNP attestation report is then obtained via
  `SNP_GET_EXT_REPORT` (real silicon: `HardwareReportIssuer` opens
  `/dev/sev-guest`; mock: `MockReportIssuer` signs against a bundled
  test PKI). The VCEK certificate rides alongside.

- **Verifier side** (relying party).
  `SnpAttestationVerifier::verify(report, vcek, expected_binding)`:
  1. Parse report via `virtee/sev`.
  2. Validate the `ARK → ASK → VCEK` chain (mock: ECDSA-P-384; production:
     RSA-PSS, deferred to M5.9 in the implementation plan).
  3. Verify the report's ECDSA-P-384 signature against the VCEK.
  4. Recompute the expected `REPORT_DATA` from the binding and compare.
  5. Optionally pin `MEASUREMENT`, `POLICY`, and `expected_model_id`.

The verifier code is the **same path for mock-issued and real reports**,
because the on-the-wire signature shape is identical. Only the cert chain
validator differs.

### 4.3 Three simulation tiers

The CVM image and binary are bit-identical across tiers; only the host
environment and `SNP_MODE` env var change.

| Tier | Host | Binary mode | Validates |
|---|---|---|---|
| **T1** in-process | any x86_64 Linux | `cargo test --features mock` | protocol math, report byte format, parser/verifier round-trip, tamper rejection |
| **T2** VM-sim CVM | regular QEMU/KVM | `SNP_MODE=mock` in the production CVM image | OS boundary, systemd unit lifecycle, weight loading, full HTTP service end-to-end |
| **T3** real silicon | Hetzner EPYC + consumer GPU via VFIO | `SNP_MODE=production` | real `/dev/sev-guest`, real ARK chain, real CVM memory encryption, SWIOTLB DMA, GPU passthrough |

Mode selection is **fail-closed and explicit**: the runner parses `SNP_MODE`
at startup; `production` aborts if `/dev/sev-guest` is absent, `mock` aborts
if the `mock` feature wasn't linked. No autodetection — the operator must
opt into mock.

T1 + T2 give deployment confidence from a non-EPYC dev box: the same image
and binary that production runs are exercised through their full systemd
lifecycle. T3 covers hardware-specific behaviour once per release.

---

## 5. Key design choices

This section calls out the non-obvious decisions and why we made them.

### 5.1 Wrap, don't inherit: `SnpTrustedExecutor` over `InProcessTrustedExecutor`

The SEV-SNP boundary is the **environment**, not the protocol. Every
`TrustedExecutor` method on `SnpTrustedExecutor` is a single forward to the
inner `InProcessTrustedExecutor`. The wrapper adds two things: the identity
pair (`model_identity`, `scheme_identity`) and an `evidence(nonce)` method
that issues a fresh report.

This composition is why the same per-text overhead numbers hold for the
in-process simulator and the SEV-SNP CVM. It also means the same protocol
tests apply at every tier.

### 5.2 Arc-shared weight cache

U-Verify requires the TEE to keep a copy of every offloaded weight for the
`B·r` step. The naive implementation clones every weight in
`provision_weight`, producing a second ~2.4 GB f32 buffer for Qwen3-class
models on top of the embedder's existing `Arc<DecoderWeights>` shards.

Under the **openweight assumption**, the weights are public — they don't
need *confidentiality* on the TEE side, only *integrity* (the host must not
be able to mutate weight bytes mid-computation, but SEV-SNP's RMP catches
that). So the executor can share storage with the embedder rather than
clone:

```rust
weights: HashMap<WeightHandle, Arc<Array2<f32>>>,   // was Array2<f32>
```

`TrustedExecutor::provision_weight_shared(_, Arc<Array2<f32>>)` is the
sharing-friendly path; `provision_weight(_, ArrayView2)` still clones for
callers that don't have an Arc. **Net result: −2.4 GB encrypted CVM RAM on
Qwen3-0.6B**, which is the difference between a 32 GB and 64 GB Hetzner SKU.

This optimisation is **only safe because weights are public**. For a
private-model deployment it would not apply — the executor would need its
own encrypted-CVM copy.

### 5.3 Length-based OutAttnMult auto-switch

Covered in §3.5. The deviation from TwinShield-as-written is the only
configuration decision in the protocol layer that's measurement-driven
rather than paper-driven.

### 5.4 Consumer GPU passthrough over Confidential Compute GPU

GELO masks make the GPU information-theoretically blind. There is no
material confidentiality difference between running these matmuls on an
H100-CC and an RX 7900 XTX. The integrity story is handled by U-Verify,
not by the GPU's attestation. So:

- **Production GPU target**: commodity passthrough via VFIO PCIe.
  Anything Vulkan-capable: RTX 4090, RX 7900 XTX, Intel Arc A770.
- **Reason**: ~10× cost reduction over H100-CC, and our Vulkan backend
  (`gelo-gpu-wgpu`) works unchanged. A CUDA backend (M4 in the plan) is
  optional, not on the critical path to TEE deployment.

The trade-off is that we pay **SWIOTLB DMA cost** for every cross-boundary
transfer (~30 ns/KB), since SEV-SNP guests use bounce buffers for device
DMA. For Qwen3 per-text traffic (~480 MB cross-boundary) this is ~15 ms/text
overhead — visible but manageable. TDISP would remove it; we defer that
to when AMD/PCI-SIG ships the kernel pieces.

### 5.5 Thin CVM image + first-boot weight fetch

The deploy image is intentionally **thin** (~50 MB, no weights baked in).
A `gelo-fetch-weights` systemd one-shot downloads model bytes from a
configured URL (HuggingFace or an offline mirror) on first boot and
SHA-256-validates against the expected hash in `runner.env`. Mismatch ⇒
unit fails ⇒ runner service refuses to start.

The image hash is bit-reproducible across rebuilds; CI gates on it.
Rebuilding for a new model revision is a config change, not an image rebuild.

This is the conventional confidential-CVM pattern (canonical/tdx, AMDESE,
RH OpenShift CoCo all do similar) — what's specific to openweight is that
the weight blob's hash itself is a **public, attestable identifier**:
the relying party knows what to expect and can pin
`expected_model_id = SHA-256("Qwen/Qwen3-Embedding-0.6B@<rev>")`.

### 5.6 Fail-closed `SNP_MODE`

The runner refuses to start unless `SNP_MODE` is one of `production` or
`mock`. There is no autodetection (e.g. "do we see `/dev/sev-guest`?"),
because every silent-fallback we considered was a worse failure mode:

- A production binary booting into mock because the device was missing
  would silently emit reports a real verifier rejects.
- A mock binary deployed to production wouldn't be caught at all.

Operators must opt into either mode explicitly. The runner prints a loud
warning when mock is active: *"Reports from this issuer will NOT verify
against AMD's production ARK."*

---

## 6. Security & privacy model

### What the protocol protects

| Channel | Mechanism | Soundness argument |
|---|---|---|
| Per-batch activations crossing PCIe | GELO mask + shield rows | Fresh orthogonal `A` per batch makes `U = A·H` information-theoretically a random rotation; shield rows close the multi-batch Gram leak. |
| Q, K never seen by GPU (default) | In-TEE attention via auto-switch | Q, K, V do not cross PCIe at all for short-input embedding workloads. |
| Q, K under OutAttnMult (long context) | 4-partition embedding | GPU sees a `(2n, 2n)` masked permuted matmul without `(R_Q, R_Kt, a, b, λ_Q, λ_K)`. |
| Integrity of every offloaded matmul | U-Verify | At `k=8, L=3`: undetected-tamper rate `≈ 2.4·10⁻⁷` per offload. |
| Activations and mask state at rest | SEV-SNP memory encryption | CVM RAM encrypted with per-CVM key; host cannot read. RMP prevents tampering. |
| Attestation key material | SEV-SNP CVM isolation + AMD-SP | Report-signing key is per-chip, never leaves the AMD Secure Processor. |
| TEE → relying party binding | Attestation: `(model_identity, scheme_identity, nonce)` baked into REPORT_DATA | Relying party verifies it loaded the expected publicly-known weights with the expected protocol scheme. |

### What it does not protect

- **Model weights from the GPU.** Weights are public — uploaded to the GPU in
  cleartext at startup. This is by design (openweight).
- **PCIe traffic for the one-time weight upload.** ~2.4 GB once at startup;
  not per-text.
- **Side-channels in the TEE itself.** SEV-SNP isolates memory but not
  cache / timing / power. The mask Householder sampler does a data-dependent
  sqrt; for a future-revision hardening pass the `subtle` crate's
  constant-time primitives would close this.
- **Workload identity / volume.** The number and timing of offload calls is
  visible to the GPU. A persistent attacker can learn that the workload is
  *some* transformer of *some* shape from the dispatch pattern.

### What attack classes don't apply (and why)

Two 2025–2026 attack papers target schemes structurally related to GELO. Neither
applies to this prototype, because GELO's per-batch full-rank sampling is
exactly the architectural choice that closes both attack classes.

- **Precomputed-basis recovery** (Wang et al., "Vulnerabilities in Partial
  TEE-Shielded LLM Inference with Precomputed Noise", arXiv 2602.11088).
  Recovers a LLaMA-3 8B layer's secrets in ~6 minutes from SOTER, TSQP, and
  TransLinkGuard by exploiting their use of a precomputed K-dimensional static
  basis — the noise lives in that subspace forever, regardless of how
  coefficients are freshly resampled per query. The authors empirically show
  that random subset sampling of basis vectors per query provides no
  meaningful defense; the attack just costs more queries.

  GELO's `A` is sampled per batch via Householder reflectors from a fresh
  ChaCha20-seeded Gaussian — Haar-uniform over the full orthogonal group
  `O(n)`. There is no static low-dimensional subspace to attack; the entire
  orthogonal group is the support of the mask distribution.

- **Sequential vocabulary matching against fixed permutations** (Wang et al.,
  "Hidden No More: Attacking and Defending Private Third-Party LLM Inference",
  ICML 2025, arXiv 2505.18332). 99%+ recovery from PermLLM, STIP, and Centaur
  by exploiting decoder-only LLMs' non-collision property: the attacker tries
  each vocabulary token at each position, runs a forward pass, and matches
  the resulting hidden state against observed obfuscated states. The attack's
  declared scope is fixed/precomputed permutations only.

  GELO's `A` is not a permutation (it's a full-rank orthogonal rotation) and
  is fresh per batch. Two independent reasons the attack misses by
  construction.

Both attacks land squarely on schemes with a static mask basis — including
schemes that *look* fresh because their coefficients are resampled per query
while the basis stays fixed. GELO's full-rank per-batch sampling closes both
attack classes by design.

### When the threat model breaks

If anyone later wants **private-model** deployment (e.g. a fine-tuned
proprietary model), GELO as-implemented does not target this. Two recovery
paths:

1. Treat the entire CVM image as the private artifact: bake weights into
   the image, measure them into the launch policy, never download. Loses the
   thin-image / first-boot-fetch convenience but keeps the GELO masks
   for activations.
2. Switch to an STIP-style protocol that masks weights as well. Out of scope
   for this prototype.

---

## 7. Tradeoffs

### Per-text overhead (Qwen3-Embedding-0.6B, RADV Vulkan, 3 short texts)

| Configuration | Wall-clock | vs `gpu_plain` | What's enabled |
|---|---|---|---|
| `gpu_plain` | 371 ms | baseline | unprotected — GPU sees raw activations |
| `gpu + GELO` (in-TEE attention, auto-switch off) | 395 ms | **+6.4%** | GELO mask on Q/K/V/O/Up/Gate/Down; attention in TEE |
| `gpu + GELO + OutAttnMult` (forced on) | 460 ms | +24.0% | Same as above plus OutAttnMult for Q·Kᵀ |

The production default at embedding shape is the middle row: ~6% wall-clock
for openweight confidentiality on the public-weight matmuls. The
OutAttnMult row is what a long-context decoder would land at; the +24% is
the regime-mismatched cost (n ≪ hidden_size) and not representative of
where OutAttnMult is meant to operate.

**Steady-state cost decomposition** (`gpu + GELO`):

- GELO mask machinery (`mask_apply` + `mask_unapply` + `mask_sample`):
  ~17 ms across 168 offloaded linears per text — the irreducible mask cost.
- In-TEE attention (`tee:attn_inplace`): ~5 ms — negligible at this n.
- Everything else: within run-to-run variance.

### Per-text overhead on a 100-doc NFCorpus batch (Qwen3, AMD Ryzen AI Max+ 395)

The above is per-text wall-clock with a small, 3-text micro-benchmark.
At realistic corpus-ingest sizes the bench in
`crates/approach4/tests/beir_accuracy.rs` runs with **`BEIR_PAPER_PARITY=1`**
(one Haar `A` per forward, paired with shield rows — see §3.2) and
the `blas` cargo feature (CBLAS-direct in `mask::apply`/`unapply` via
BLIS), and parallel-fan-out `embed()` via rayon (one cloned executor
per worker, each with its own ChaCha20 stream so cross-text `A` stays
independent — see also `future-rnd.md` §5):

| Stage | Vanilla BLIS (AVX2 dispatch) | **AOCL-BLIS (AVX-512 via `skx_asm`)** |
|---|---:|---:|
| Qwen3 plain (`PlaintextExecutor`, no mask) | 123 ms | 121 ms |
| Qwen3 + GELO mask + CAPRISE | 281 ms (2.28× plain) | **153 ms (1.27× plain)** |
| `gelo:mask_apply` aggregate (over 100-doc bench) | 326.6 ms | **62.6 ms (5.22×)** |
| `gelo:mask_unapply` aggregate | 586.7 ms | **121.4 ms (4.83×)** |
| Total bench wall (100 docs + 100 queries) | 129.6 s | **88.7 s (−31.5%)** |

Both columns use **`BEIR_PAPER_PARITY=1`**, parallel-fan-out `embed()` via
rayon, `BLIS_NUM_THREADS=1` (rayon owns the parallelism), and the
`blas` cargo feature.

For a 100-doc batch ingest this lands at **~15 s/100 texts** with
AOCL-BLIS (vs ~30 s with vanilla BLIS, and ~85 s with the older
sequential `BLIS=16` configuration). The parallel path is gated on
`texts.len() > 1` so the single-query online path is unchanged
(single-text embed clones one executor and runs serially without the
rayon scope overhead).

**Why AOCL-BLIS wins.** Vanilla BLIS's `bli_sgemm` dispatcher on Zen 4/5
falls back to `bli_sgemm_haswell_asm_16x6` (AVX2 — 8 floats per
zmm-equivalent ymm register). AOCL-BLIS's `config/zen5/bli_cntx_init_zen5.c`
explicitly assigns SGEMM to `bli_sgemm_skx_asm_32x12_l2` (Intel SKX AVX-512
— 16 floats per zmm register). The skx kernel is pure AVX-512 instructions,
not Intel-specific, so it runs natively on Zen 4/5 with full per-clock
throughput. Hand-tuned Zen-specific SGEMM ASM doesn't actually exist
upstream — but Intel's SKX AVX-512 SGEMM is what we get instead, and AOCL
just makes the dispatcher pick it. (DGEMM/CGEMM/ZGEMM *do* have hand-tuned
Zen 4/5 kernels in AOCL-BLIS — see `kernels/zen4/3/bli_dgemm_zen4_asm_*` —
but our mask is f32 SGEMM, so we route through skx_asm.)

The 4–5× improvement at the bucket level is bigger than the naive "AVX-512
is 2× AVX2" estimate because (a) the skx kernel uses 32x12 tiling that
fits Zen 5's L2 cache better than haswell's 16x6, and (b) under rayon
parallelism each worker's CPU-bound time shrinks, leaving the GPU as the
real bottleneck — so wall-clock per text drops further than the BLAS
in-isolation speedup would predict.

**Install reproducibility.** AOCL-BLIS is built from `github.com/amd/blis`
into `vendor/aocl-install/` via the `scripts/install-aocl-blis.sh` script
(idempotent, no sudo). The `blis-src` crate's `system` feature picks it
up at link time given `LIBRARY_PATH` and `LD_LIBRARY_PATH` env vars set
to `vendor/aocl-install/lib`. The CVM build image needs `libblis-mt.so`
on `LD_LIBRARY_PATH` at runtime.

### Attestation cost

| Step | Cost | Frequency |
|---|---|---|
| `SnpTrustedExecutor::evidence(nonce)` (mock issuer) | 0.39 ms | once per session |
| `SnpAttestationVerifier::verify(...)` (mock chain) | 2.70 ms | once per session |
| Report size on wire | 1184 B (SEV-SNP ABI) + ~733 B VCEK PEM | once per session |

Production-silicon equivalents will be in the same order of magnitude.
Attestation is **never on the per-text path**; the `SnpTrustedExecutor`
wrapper adds zero overhead to the embed loop (forwarding-only `TrustedExecutor`
impl).

### Memory budget (Qwen3-Embedding-0.6B inside the CVM, with Arc-share)

| Component | Encrypted CVM RAM | Shared (SWIOTLB) |
|---|---|---|
| Model weights (one Arc, shared embedder + executor U-Verify cache) | 2.4 GB | — |
| Working-set activations across 28 layers | ~0.4 GB | — |
| Mask state, shield rows, RNG, scheme_identity, attestation private state | ~0.05 GB | — |
| OutAttnMult scratch (only when engaged at long n) | ~0.05 GB | — |
| GPU-DMA bounce buffers | — | ~0.5 GB |
| Misc systemd / kernel / Vulkan userspace | ~0.3 GB | — |
| **Total** | **~3.2 GB** | **~0.5 GB** |

Fits comfortably in a 32 GB Hetzner AX42. Without the Arc-share refactor
the budget would be ~7 GB and force a 64 GB SKU.

### Trade-off summary

| What we gave up | What we got |
|---|---|
| Bit-exact privacy for model weights | Openweight scope → simpler attacks model, smaller TEE footprint, ~10× cheaper GPU |
| Universal applicability across model classes | Sharp focus on openweight transformer inference → straightforward Arc-share, fail-closed runtime mode |
| Unconditional OutAttnMult (paper recipe) | Length-based auto-switch → +6% overhead at embedding shape, OutAttnMult kept available for long-context deployments |
| Confidential-GPU attested DMA | Per-PCIe-batch SWIOTLB cost (~15 ms/text on Qwen3) → consumer GPU compatibility |

---

## 8. Current results and forward-looking work

### Where the prototype stands

- **T1 in-process**: protocol math, mask, shield, U-Verify, OutAttnMult,
  attestation report parse/verify, tamper rejection — all green.
- **T2 VM-simulated CVM**: same binary boots in regular QEMU under
  `SNP_MODE=mock`, HTTP smoke green (`/health`, `/attest`, `/ingest`, `/query`).
- **T3 real silicon**: deferred to M5.9 (Hetzner EPYC provisioning).
  Hardware-only behaviours — real PSP, real ARK chain, real RMP, SWIOTLB on
  passthrough GPU — validated once per release on the dedicated server.

### Per-operation runtime breakdown (Qwen3+mask, paper-parity)

Measurement context: BEIR/NFCorpus, 100 docs + 100 queries (200 texts
total), Qwen3-Embedding-0.6B with 28 decoder blocks, n≈400 tokens
average, hidden=1024, intermediate=3072. Paper-parity mode (one Haar
`A` per forward + 8 shield rows), `--features blas` (CBLAS-direct in
`mask::apply`/`unapply` via BLIS), AMD Ryzen AI Max+ 395.

The wall-clock numbers below are from a **sequential** run
(`BLIS_NUM_THREADS=16`, single-threaded `embed`) because the
profiling aggregator is thread-local and the parallel run splits
samples across rayon workers. The per-text cost is identical between
sequential and parallel modes; parallel just runs N texts on N CPU
workers concurrently to drop the wall-clock — see §7's "100-doc
NFCorpus batch" table for the parallel-mode end-to-end numbers.

**Model compute (would happen without GELO too):**

| op | per-text | per-call | calls/text | what it is |
|---|---:|---:|---:|---|
| `tee:attn_inplace` | 237.8 ms | 8.50 ms | 28 | Causal GQA attention in TEE (Q/K/V never cross PCIe) |
| `engine:matmul_many` | 142.1 ms | 2.54 ms | 56 | Batched GPU matmuls (QKV-bundle: 28; gate+up-bundle: 28) |
| `engine:matmul` | 105.2 ms | 1.88 ms | 56 | Single GPU matmuls (O: 28; Down: 28) |
| `tee:swiglu_activate` | 25.5 ms | 0.91 ms | 28 | SiLU(gate)·up element-wise |
| `tee:rmsnorm` | 10.9 ms | 0.19 ms | 57 | Pre-attn + pre-FFN norm per block, + final norm |
| `tee:residual` | 3.9 ms | 0.07 ms | 56 | h + attn_out, h + ffn_out per block |
| `tee:rope` | 2.1 ms | 0.076 ms | 28 | Rotary embedding on Q, K per block |
| `tee:embed_lookup` | 0.08 ms | 0.075 ms | 1 | Token-id → embedding row |
| **subtotal — model compute** | **527.6 ms** | | | matches Qwen3 plain wall-clock (~511 ms) |

**GELO mask machinery (overhead added by the protocol):**

| op | per-text | per-call | calls/text | what it is |
|---|---:|---:|---:|---|
| `gelo:mask_unapply` | 173.1 ms | 0.88 ms | 196 | `Aᵀ · V` via direct `cblas_sgemm` (BLIS) |
| `gelo:mask_apply` | 100.9 ms | 0.90 ms | 112 | `A · stacked_H` via direct `cblas_sgemm` (BLIS) |
| `gelo:shield_stack` | 28.1 ms | 0.25 ms | 112 | Write data rows + 8 fresh Gaussian shield rows into scratch |
| `gelo:strip_shield` | 14.5 ms | 0.13 ms | 112 | Slice off shield rows, `to_owned()` the data block |
| `gelo:mask_sample` | 8.8 ms | 8.76 ms | 1 | Haar-uniform QR over (n+k)×(n+k) Gaussian (one per forward) |
| **subtotal — GELO overhead** | **325.4 ms** | | | |

**Totals:**

| | per-text | share |
|---|---:|---:|
| Model compute | 527.6 ms | 61.9% |
| GELO mask machinery | 325.4 ms | 38.1% |
| **Total Qwen3 + GELO mask, sequential** | **853 ms** | 100% |
| **Total Qwen3 + GELO mask, parallel (BLIS=1, rayon)** | **302 ms** | (4.5× throughput vs sequential) |
| Qwen3 plain reference | 132–511 ms | depends on parallel vs sequential |

**Call-count tally** (Qwen3 = 28 decoder blocks, paper-parity, Q2
gate+up bundling on):

| group | apply/block | unapply/block | per forward (×28) |
|---|---:|---:|---:|
| QKV (bundled via `offload_qkv`, shared `H_norm`) | 1 | 3 | 28 apply, 84 unapply |
| O (attention output) | 1 | 1 | 28 apply, 28 unapply |
| FFN gate+up (bundled via `offload_linear_many`, shared `H_norm_ffn`) | 1 | 2 | 28 apply, 56 unapply |
| FFN down | 1 | 1 | 28 apply, 28 unapply |
| **per forward** | **4** | **7** | **112 apply, 196 unapply** |

Without Q2's gate+up bundling this would be 5 apply / 7 unapply per
block (140 apply / 196 unapply per forward — Q2 saves 28 redundant
applies / forward at ~0.9 ms each).

**Where the time really goes:**

- **Mask GEMMs are 32% of total with vanilla BLIS** (274 ms/text apply+unapply).
  Each is a `(n+k)² × d` CPU matmul. With AOCL-BLIS swapped in (lever #4
  below — done), this drops to **2.2% of total (~184 ms across 308 GEMMs
  per text) — a 5× per-bucket reduction**. The wins come from the
  dispatcher selecting `bli_sgemm_skx_asm_32x12_l2` (AVX-512) over
  `bli_sgemm_haswell_asm_16x6` (AVX2 fallback).
- **In-TEE attention is 28% of total** (238 ms/text). Eight Q·Kᵀ
  matmuls per layer × 28 layers, each tiny — `ndarray::dot`
  (matrixmultiply, single-thread). OutAttnMult would move this to GPU
  but adds 4× FLOPs; the auto-switch (§3.5) keeps it in-TEE at our
  short n.
- **GPU GEMMs are 29% of total** (247 ms/text). Eight offloaded
  projections per layer × 28 layers, dispatched through `wgpu` /
  burn-cubecl. Mostly bandwidth-bound at the small (n+k, d) shape.
- **Everything else** (mask sample, shield stack/strip, RMSNorm,
  SwiGLU, residual, RoPE, lookup) totals **~94 ms (11%)** — none of it
  a single dominant bucket.

**Apples-to-apples with the GELO paper.** The paper's headline 20%
overhead is computed on Llama-2 7B at n=512 against a baseline that
includes ~14 ms/call of socket-IPC overhead between the SGX trusted
process and the GPU process. Our in-process TEE has no such IPC.
Comparing on the right metric — per-offload mask cost / per-offload
GPU GEMM cost — we land at **1.36×** (mask / GEMM ratio) vs the
paper's **1.07×**, within hardware-tuning distance. Comparing on
percentage overhead is misleading because the denominators differ:
the same absolute mask cost is a much larger fraction of our cleaner
~511 ms baseline than of the paper's ~1.9 s IPC-inflated baseline.
The full reasoning lives in the commit history for this section; the
take-away is **we are not doing redundant work relative to the paper
within attention scope** — we extend mask coverage to the FFN
projections too, which the paper omits, because the alternative is
running FFN in TEE at ~3× the wall-clock.

### Highest-impact next levers

1. **GPU-side OutAttnMult stacking.** `outattn:setup_stack_batched` is 42% of
   the protected-path overhead when OutAttnMult is engaged. Moving the
   2n-wide operand packing to a fused WGSL kernel (or SIMD on the CPU side)
   would compress the long-context path's overhead from +24% toward +12–15%.
2. **TDISP for DMA cost.** Once kernel + firmware support lands, the SWIOTLB
   bounce-buffer cost disappears. ~15 ms/text on Qwen3 today.
3. **Real-VCEK CI fixture.** After T3 boot, capture one VCEK + sample report
   into the repo; CI offline-verifies it against AMD's published ARK on
   every run, catching report-format regressions without per-run silicon
   cost.
4. **AOCL-BLIS swap for the `blas` cargo feature — DONE 2026-05-14.**
   Vanilla `blis-src 0.2.2` builds upstream BLIS from source; on Zen 4/5
   hosts the runtime dispatcher falls back to `bli_sgemm_haswell_asm_16x6`
   (AVX2) because no `zen4/5_asm` SGEMM kernels exist upstream (`nm libblis.a`
   shows only `zen*_ref`). AMD's AOCL-BLIS fork (`github.com/amd/blis`)
   does **not** add hand-tuned Zen SGEMM kernels either — but its
   `config/zen5/bli_cntx_init_zen5.c` explicitly maps SGEMM dispatch to
   `bli_sgemm_skx_asm_32x12_l2` (Intel SKX AVX-512 — pure AVX-512
   instructions, runs natively on Zen). That single dispatch-table change
   is the win. Implemented via `blis-src` features `["system", "cblas"]`
   pointing at `vendor/aocl-install/lib/libblis-mt.so` (see
   `scripts/install-aocl-blis.sh`). **Measured 4.96× speedup on mask GEMMs**
   (913 → 184 ms aggregate per 100-doc bench), taking Qwen3+mask
   from 281 → 153 ms/text (2.28× → **1.27× plain**). License: BSD-3,
   compatible.

5. **Softmax-equivariant attention offload (research lever).** Amulet
   (Wang et al., "Fast TEE-Shielded Inference for On-Device Model
   Protection", arXiv 2512.07495, Dec 2025) observes that
   `softmax(πQKᵀπᵀ/√d) = π · softmax(QKᵀ/√d) · πᵀ` — permutation matrices
   commute through softmax. If we composed a fresh-per-batch permutation π
   onto the attention block (in addition to GELO's orthogonal mask + shield
   rows + small Gaussian noise per Hidden No More's mitigation), softmax
   itself could run on the GPU. This would address the **in-TEE attention
   cost** (28% of total per-text, ~238 ms on Qwen3 at n≈400) — the single
   largest bucket after mask GEMMs. The construction differs from
   OutAttnMult (which masks Q·Kᵀ additively but still runs softmax in the
   TEE) by moving softmax itself off the trusted side. Risk: must be
   combined with shield rows + σ≈0.01 Gaussian noise to survive sequential
   vocabulary matching; for embedding shapes the threat is weaker than
   for decoder generation, but the security argument still needs an
   end-to-end re-derivation under our threat model. Amulet's own threat
   model is on-device weight protection, so the security proof doesn't
   port directly. No public Amulet code as of 2026-05-14. Effort: 1–2
   week spike, including an empirical attack-resistance benchmark using
   `qsxltss/Game-of-Arrows` as the attack-side reference.

### Out of scope (and why)

- **Private model weights** — different threat model; would require an
  STIP-style mask. The whole stack is built around the openweight assumption.
- **MPC / FHE inference** — addressed in `docs/research/private-inference.md`.
  Orders of magnitude slower than this for the embedding workload we target.
- **DP-perturbed embeddings** — a complementary mechanism for the output
  side; not needed when the embedding ciphertext itself is encrypted at rest
  (approach4's SAP / CAPRISE schemes).

---

## Appendix: GELO masking does not corrupt retrieval ranking

A confounded observation in an early accuracy bench attributed
ranking-corruption to "decoder-LLM anisotropy under GELO masking." A
follow-up controlled experiment
(`crates/approach4/tests/gelo_embedder_accuracy.rs`) falsifies that:

| Config | top1_grp | top1_vs_plain | rec3_vs_plain |
|---|---|---|---|
| FastEmbed MiniLM-L6 (control) | 1.00 | — | — |
| BGE-small plain | 1.00 | — | — |
| BGE-small + GELO masking | **1.00** | **1.00** | **1.00** |
| Qwen3-0.6B plain | 1.00 | — | — |
| Qwen3-0.6B + GELO masking | **1.00** | **1.00** | **1.00** |
| Qwen3-0.6B + GELO + OutAttnMult forced at any n | **1.00** | **1.00** | **1.00** |
| Qwen3-0.6B + GELO on Vulkan engine | **1.00** | **1.00** | **1.00** |
| Qwen3-0.6B plain + `"Instruct: …"` prefix | 0.50 | **0.25** | **0.25** |

GELO masking, the OutAttnMult 4-partition Q·Kᵀ path forced on at short
sequences, and the Vulkan engine all preserve ranking **bit-for-bit**
against their plain counterparts (`top1_vs_plain = 1.00`). The actual
corruption source identified by the bench was the
model-card-recommended `"Instruct: …\nQuery: …"` prefix interacting
badly with **last-token pooling**: ~12 instruction tokens prepended to a
short query means the pooled embedding sits in instruction-context
space, not query-content space, collapsing semantic separation across
queries.

Practical guidance: do not apply Qwen3-Embedding-0.6B's HF instruction
prefix to short-query zero-shot retrieval. The parity tests in
`crates/gelo-embedder/tests/{decoder_parity.rs, qwen3_e2e.rs}` already
confirm GELO masking preserves embeddings within `max_abs < 1e-2` per
component, and the accuracy bench confirms that level of drift is below
the threshold needed to flip cosine ranks on a 12-doc corpus.

---

## References

- Belikov & Fedotov, "GELO: Activation-Mask Split-Inference for Open-Weight
  Transformers." arXiv 2603.05035.
- Xue, Liu, Cao et al., "TwinShield: Defending Split-Inference Against the
  Gram Leak and Engine Tampering." 2025.
- AMD, "SEV Secure Nested Paging Firmware ABI Specification." Document 56860.
- Morris, Kuleshov, Shmatikov, Rush, "Text Embeddings Reveal (Almost) As
  Much As Text." EMNLP 2023 (Vec2Text — the embedding-inversion threat that
  motivates this prototype).
- Wang et al., "Vulnerabilities in Partial TEE-Shielded LLM Inference with
  Precomputed Noise." arXiv 2602.11088 — precomputed-basis recovery attack on
  SOTER / TSQP / TransLinkGuard; surveyed in §6 "What attack classes don't
  apply."
- Wang et al., "Hidden No More: Attacking and Defending Private Third-Party
  LLM Inference." ICML 2025, arXiv 2505.18332 — sequential vocabulary
  matching attack on fixed permutations (PermLLM / STIP / Centaur); surveyed
  in §6.
- Wang et al., "Amulet: Fast TEE-Shielded Inference for On-Device Model
  Protection." arXiv 2512.07495 — source of the softmax-permutation
  equivariance technique referenced in §8 "Highest-impact next levers."
- `docs/research/private-embedding-research.md` §D Recipe D — the survey
  that landed on this design.
