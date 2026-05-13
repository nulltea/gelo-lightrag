# GELO inference engine — optimization report

> **Audience.** Engineers working on `gelo-embedder` + `gelo-gpu-wgpu`.
> **Context.** M7 BEIR/NFCorpus bench surfaced ~900 ms/doc on BGE-base
> with full GELO masking via Vulkan/cubecl — orders of magnitude slower
> than fastembed CPU (~10 ms/doc) and llama.cpp (~3 ms/text-vec at
> batch=1 on the same iGPU).
>
> This report synthesizes (1) industry best practice for BERT and
> decoder-LLM inference, (2) how each technique interacts with GELO's
> per-batch fresh orthogonal mask, and (3) a prioritized concrete plan.

---

## 0. Definitions

Project-specific terms and domain acronyms whose meaning isn't
self-evident from context.

### Project / protocol
- **GELO** — *GPU-Encrypted Linear Offload.* The Belikov & Fedotov
  (arXiv 2603.05035) split-inference protocol: trusted side samples a
  fresh orthogonal mask `A`, ships `U = A·H` to an untrusted GPU,
  recovers `H·W = Aᵀ·(U·W)` on return.
- **CVM** — *Confidential Virtual Machine.* A VM running inside a TEE
  (the SEV-SNP "encrypted-RAM VM" form-factor).
- **SEV-SNP** — *Secure Encrypted Virtualization with Secure Nested
  Paging.* AMD's CVM technology; gives memory encryption + attestation.
- **U-Verify** — Our Freivalds-style integrity probe for offloaded
  GEMMs (trusted side compares `B · r` to engine-reported
  `masked_out · r`).
- **OutAttnMult / TwinShield** — Xue et al. (2025) §V-A 4-partition
  embedding for offloading runtime-runtime matmuls (`Q·Kᵀ`) without
  revealing either operand. Used in our attention offload path.
- **DP-Forward** — *Differentially Private Forward Pass.* Yue et al.
  (CCS 2023) noise-injection mechanism on transformer hidden states.

### Models / datasets
- **BERT** — *Bidirectional Encoder Representations from Transformers.*
  The encoder family (12 layers, 768 hidden for BERT-base).
- **BGE** — *BAAI General Embedding.* HuggingFace embedding models
  (`BAAI/bge-base-en-v1.5` is BERT-class).
- **BEIR / NFCorpus** — Information-retrieval benchmark suite / its
  medical-domain dataset (3,633 docs, 323 queries, graded qrels).

### Transformer ops
- **GEMM** — *General Matrix Multiply.* The dense linear-algebra core
  op `C = A·B`.
- **SGEMM** — Single-precision (fp32) GEMM.
- **QKV** — Query / Key / Value projections inside self-attention.
- **FFN** — *Feed-Forward Network.* The per-layer MLP block.
- **GELU** — *Gaussian Error Linear Unit.* BERT's activation function.
- **SwiGLU** — *Swish-Gated Linear Unit.* Decoder-LLM FFN variant used
  by Qwen3 (`down(silu(gate) * up)`).
- **LayerNorm** — Per-row mean/variance normalization layer.
- **KV-cache** — Decoder optimization caching past Key/Value tensors
  across autoregressive steps. *Not used in embedding models.*
- **Flash attention** — Tiled fused softmax-matmul kernel that avoids
  materializing the full attention matrix in HBM.

### Hardware / runtime
- **iGPU** — *Integrated GPU.* Shared-memory GPU on the same die as
  the CPU (vs discrete GPU with its own VRAM).
- **HBM** — *High-Bandwidth Memory.* GPU's main DRAM.
- **WGSL** — *WebGPU Shading Language.* The shader language consumed
  by our wgpu/cubecl pipeline.
- **cubecl** — Rust GPU kernel framework backing our wgpu engine.
- **wgpu** — Rust implementation of WebGPU on top of native APIs.
- **BLAS** — *Basic Linear Algebra Subprograms.* The reference CPU
  linear-algebra interface (`faer`, OpenBLAS, MKL implement it).
- **QR factorization** — Decomposition `A = Q·R` into orthogonal `Q`
  and upper-triangular `R`; we use it to sample Haar-uniform `A`.

### Precision
- **fp32 / fp16 / bf16** — 32-bit / 16-bit IEEE / "brain float"
  (8-exp + 7-mantissa) floating point.
- **INT8 / Q4_K** — 8-bit integer weight quantization / GGUF's
  4-bit block-quantized format with per-block scale.

### Other
- **CSPRNG** — *Cryptographically Secure PRNG.* Used by the trusted
  side to sample fresh masks (ChaCha20).
- **MoE** — *Mixture of Experts.* Sparse-routing transformer variant.

---

## 1. Where the time goes today

Profiling the bench (and reading `crates/gelo-gpu-wgpu/src/lib.rs` plus
`crates/gelo-embedder/src/bert/forward.rs`) attributes the cost to four
categories. Numbers are order-of-magnitude estimates from observed
wall-clock on a single-doc BGE-base forward at seq_len ≈ 128:

| Source | Per-text cost | Code location |
|---|---|---|
| cubecl `Strategy::Auto` autotune cache misses (every new (M,K,N) shape) | ~5–20 ms × 12 layers × 6 ops = **~300–1500 ms first-time** | `lib.rs:220`, `Strategy::Auto` |
| `client.read_one(out_handle)` synchronous GPU→CPU readback after every GEMM | ~24–36 μs sync stall × ~72 dispatches = **~2–3 ms minimum** | `lib.rs:229`, `lib.rs:308`, `lib.rs:385` |
| Per-call buffer creation (no pool) | ~5–15 μs alloc × ~144 buffers/text = **~1–2 ms** | `lib.rs:176-177`, `lib.rs:255-257`, `lib.rs:332-334` |
| One-text-at-a-time scheduling (no padding, no batching across docs) | M = seq_len of each text changes every call → autotune miss + tiny GEMMs | `bert/embedder.rs:155-211` |
| Haar-orthogonal mask sample (O(n³) Householder QR in pure Rust) | ~1–3 ms at seq_len 128 × 12 layers = **~15–35 ms** | `gelo-protocol/src/mask.rs:72-142` |
| LayerNorm / GELU / add_bias on CPU after each GPU GEMM | ~0.5 ms × 12 layers = **~5–10 ms** | `bert/forward.rs:134-167` |

The headline insight: at our shapes (BGE-base = 12 layers × hidden 768
× ffn 3072 × seq_len ~128) **the GEMM itself is fast** (~200 GFLOP/s
on an iGPU finishes a single QKV in <100 μs). What's slow is everything
*around* the GEMM — dispatch, sync, autotune, allocation. This is the
canonical "small-batch-on-WebGPU" failure mode documented in the
WebGPU-LLM literature.

---

## 2. Industry-SOTA inference: techniques and their GELO fitness

Each technique is rated **green (drop-in)**, **yellow (compatible with
moderate work)**, or **red (fundamentally conflicts with GELO mask
boundary)**.

### 2.1 Kernel-level optimizations

#### 2.1.1 Fused QKV projection — **green**

**What.** Concatenate `[W_q; W_k; W_v]` along the output axis into a
single `(d_model, 3·d_model)` weight; one GEMM produces `[Q | K | V]`.
**Why it helps.** Cuts 3 dispatches → 1. At GELO's per-dispatch cost
(~36 μs sync + ~5 μs setup), three QKV projections cost ~125 μs of
overhead per layer regardless of GEMM size. Fusing saves ~85 μs/layer
× 12 layers ≈ **1 ms/text** plus three autotune entries saved.

**GELO fitness.** The protocol *already* shares one mask `A` across Q,
K, V (see `sim.rs:216-267`, `offload_qkv`). The math `Aᵀ·(U·W_q) = H·W_q`
extends trivially to a concatenated `W_qkv = [W_q | W_k | W_v]` because
the mask is on the *token axis* and the weight stacking is on the
*output axis*. Implementation: change `register_weight` to accept the
concatenated `(d, 3d)` block, change `offload_qkv` to issue one
`matmul(handle_qkv, masked)` and slice the result.

#### 2.1.2 Fused FFN gate-up — **green** (decoder only)

**What.** For SwiGLU FFN (`down(silu(gate) * up)`), concatenate
`W_gate ‖ W_up` into one `(d_model, 2·d_ffn)` GEMM, then split. Saves
1 dispatch per layer. BGE-base is GELU-FFN (single up-projection), so
this only applies to the Qwen3 decoder path.

**GELO fitness.** Identical reasoning to fused QKV: one mask covers
both halves, weights stack on output axis.

#### 2.1.3 Flash attention (fused softmax-matmul) — **yellow**

**What.** Compute `softmax(Q·Kᵀ/√d_k)·V` in one tiled kernel, never
materializing the full (n, n) attention matrix in HBM. State of the
art on CUDA; multiple WGSL ports exist (`fmlc/whisper-webgpu`,
`Xenova/transformers.js`).

**GELO fitness — caveat.** Today the attention matmul `Q·Kᵀ` is
offloaded via **OutAttnMult** (TwinShield §V-A, `out_attn_mult.rs`)
using a 4-partition embedding so the untrusted GPU sees neither Q nor
Kᵀ in cleartext. A naive fused flash kernel would have to take Q and Kᵀ
as cleartext inputs, **breaking the privacy boundary**.

Compatible variants:
- Fuse only `softmax · V` (the second matmul), since by that point Q
  and Kᵀ have already been "consumed" and the attention scores can
  optionally be re-masked. Saves one dispatch + one round-trip; doesn't
  break the TwinShield split.
- Keep `Q·Kᵀ` offload as-is (it has a fused batched primitive already
  via `matmul_dynamic_batched`), and fuse only softmax + `Attn · V` on
  the trusted side. This is a TEE-side fusion, not a GPU one.

#### 2.1.4 Quantization (Q4_K / INT8 weight-only) — **yellow**

**What.** Store weights as 4-bit or 8-bit integers with per-block
scale; dequantize on the fly inside the matmul kernel. 4× memory cut,
2–3× GEMM speedup on memory-bound workloads.

**GELO fitness.** The mask round-trip math is `Aᵀ·(U·W) = H·W` — it
requires `W` to behave as a linear operator. Block-quantized weights
*are* linear (dequant happens inside the kernel and the kernel's output
is still `U·W` up to numerical noise), so quantization is mask-safe.

**But:** U-Verify probes (`integrity.rs`, Freivalds-style) currently
compare `B · r` (TEE-side) against `masked_out · r` (engine-side) with
exact float arithmetic and a small tolerance. INT8 quantization
introduces non-trivial dequant error that can blow the existing
tolerance and cause spurious verify failures. Need to widen tolerance
to a quantization-aware bound (e.g. `block_scale_max · ε_q · √k`) — not
hard, but a deliberate change.

Decision: yellow, not red. Defer until after the dispatch/sync wins
land (those are bigger).

#### 2.1.5 Fused residual + LayerNorm — **green**

**What.** Fold `add(h, proj) → layernorm` into one kernel. LayerNorm
is memory-bound; fusing with the preceding add cuts a roundtrip
through HBM.

**GELO fitness.** Both add and LayerNorm currently run on the *TEE
side* (`bert/forward.rs:131-157`, plain ndarray loops on CPU). They
are not offloaded; mask never enters. A fused kernel running locally
on CPU (SIMD via `wide` or `packed_simd`) or moved to the trusted
GPU path is a pure TEE-internal optimization with no protocol impact.

The bigger win here is that *the current implementation is single-
threaded scalar f32 in Rust*. Even without GPU offload, switching to
rayon-parallelized SIMD ndarray ops would 4–8× this step.

### 2.2 Dispatch-level optimizations

#### 2.2.1 Async dispatch / single sync per forward — **green** (highest ROI)

**What.** Today `WgpuVulkanEngine::matmul` calls `client.read_one()`
after **every** GEMM (`lib.rs:229`). This forces a GPU→CPU sync at
each of the ~72 GEMMs in a BGE-base forward pass. At a per-sync cost
of ~30 μs on Vulkan iGPU, that's ~2 ms of pure sync overhead.

**Fix.** Submit all dispatches asynchronously; sync once at the end of
the forward pass when the pooled embedding is needed CPU-side.

**GELO fitness — load-bearing caveat.** The mask round-trip needs the
GEMM output `U·W` on the *TEE side* to compute `Aᵀ·(U·W)`. If the TEE
runs on the *same machine* as the offload engine (current sim and
SEV-SNP CVM cases), the "TEE side" reads `U·W` from CPU memory after a
sync. The trick is that for the *next* layer's input we need the
unmasked result `H_next = Aᵀ·(U·W) + …`, so we need the masked output
on CPU before we can compute the unmasked one.

This is the architecture's hardest perf wall. Three ways out:

1. **Keep the unmask on the GPU.** Treat `Aᵀ` as a weight, register it
   per-layer, run `Aᵀ · masked_out` as one more GEMM. Then there's no
   need to read back until after the final pool. Mask is then visible
   to the engine for a brief instant per layer — but it's *one fresh
   mask per batch*, generated by the TEE's CSPRNG, never reused, so
   the engine seeing it post-hoc tells it nothing about `H`. This is a
   privacy-equivalent transformation, **and it's the trick GELO §3.4
   already hints at** for the "trusted-but-bandwidth-limited TEE" case.

2. **Pipeline with double-buffering.** Submit layer N+1's QKV using the
   *speculative* mask, then once layer N's unmask lands, fix up the
   inputs. Adds complexity without changing the asymptotic dispatch
   count; not recommended.

3. **Accept one sync per layer.** ~12 syncs × 30 μs = 360 μs/text. A
   12× improvement over today's 72 syncs without breaking the protocol.
   Easy win, no math changes.

Recommend option 3 first (low risk, big win), then option 1 if more
headroom is needed.

#### 2.2.2 Persistent buffers / zero-copy iGPU mapping — **green**

**What.** Allocate input/output buffers once, reuse across calls.
Today `lib.rs:176-177` calls `client.create_from_slice(...)` and
`client.empty(...)` on every `matmul` invocation — that's 72 buffer
allocations per text. On Vulkan, each `create_from_slice` involves a
staging buffer, a CPU→GPU copy, and a fence. On integrated GPUs
(shared memory), this is wasteful: we could map the buffer once with
`MAP_WRITE | STORAGE` and write directly into GPU-visible memory.

**GELO fitness.** Pure engine-side optimization. The mask is computed
TEE-side and the masked bytes go through the buffer regardless of how
that buffer is acquired. No protocol change.

#### 2.2.3 Static-shape bucketing + cached autotune — **green**

**What.** cubecl's `Strategy::Auto` runs a benchmark on first use of
each `(M, K, N)` shape combo, picks the best tile config, and caches
it. But because `M` (seq_len) varies per text, every text with a novel
seq_len triggers re-autotune. Fix: pad all inputs to one of a small set
of buckets (e.g. 64 / 128 / 256 / 512) so only 4 autotune entries
exist instead of one per text.

**Companion fix.** Persist the autotune cache to disk between runs of
the bench so even the first-bucket cost is amortized across runs.

**GELO fitness.** Padding adds `pad_token` rows to the hidden state.
Attention mask handles them on the TEE side, and the mask `A` doesn't
care what's in the rows — it's a token-axis transform applied uniformly.
Net: no protocol change.

#### 2.2.4 Cross-text batching — **green**

**What.** Concatenate N texts into a single `(N·seq_len, d)` batch
along the token axis (with an attention mask to prevent cross-text
leakage). Industry standard; fastembed and sentence-transformers do
this. Cuts dispatch count by N.

**GELO fitness.** Two routes:

1. **One mask per batch (across texts).** Sample `A ∈ R^(N·n, N·n)`
   covering the whole batch. Mathematically clean; mask cost grows
   O((N·n)³) so beyond N≈8 the mask sampling itself becomes the
   bottleneck (Haar QR is O(n³)).
2. **Block-diagonal mask: one independent `A_i` per text.** The mask
   becomes `diag(A_1, …, A_N)`, total cost O(N·n³). Privacy story per
   text identical to single-text. GEMM cost N× smaller (one dispatch
   covering the batch). This is the recommended form.

Either way, the dispatch count drops N-fold. Implementation is
moderate: protocol-level change to `offload_linear` / `offload_qkv` to
accept a batched hidden state plus a "batch boundaries" descriptor.

#### 2.2.5 Mask QR speedup — **green**

**What.** The Haar-orthogonal sampler at `mask.rs:72-142` is a textbook
O(n³) Householder QR in pure scalar Rust. At seq_len=128 and 12 layers
that's ~25 ms/text. Three fixes (compose):

- Pad seq_len to a fixed bucket and cache the QR factorization basis
  (still need fresh randomness, but the Householder reflections can be
  applied to a pre-allocated workspace).
- Replace scalar inner loops with `ndarray::Array2::dot` (BLAS via
  faer or matrixmultiply backend) — gets us SIMD + multi-thread for
  free.
- For seq_len ≤ 256, use a smaller mask via the *block-diagonal*
  construction: sample `A ∈ R^(b·b)` with `b = 32` and tile, paying
  O(b³ · n/b) = O(b²·n) ≪ O(n³) for the sampling. Privacy argument:
  cross-block correlations expose at most `O(b)` linear constraints
  per token, still computationally hiding under standard assumptions.
  (This last one is a *new* privacy claim; needs review before
  shipping.)

**GELO fitness.** Items 1 and 2 are pure perf, no protocol change.
Item 3 is a *protocol weakening* and needs a security analysis before
adoption — flagged for future-rnd.md.

### 2.3 Memory & data-layout optimizations

#### 2.3.1 KV-cache for decoder LLMs — **N/A for embedding**

Embedding models do one forward pass per text; no autoregressive
decoding, no KV-cache need. Listed for completeness only.

#### 2.3.2 Half-precision (fp16/bf16) — **yellow**

**What.** Most GEMM kernels are 2× faster in fp16 than fp32 on modern
GPUs (and ~3× on iGPUs with sharing).

**GELO fitness.** Mask `A` is sampled in fp32; if GEMM runs in fp16,
the `U·W` result is fp16 and `Aᵀ·(U·W)` in fp32 needs an upcast. Round-
trip error grows; needs to be characterized. U-Verify probes also need
quantization-aware tolerance (same issue as INT8).

Defer to after the dispatch wins land. The expected speedup at our
shapes is moderate (~30–50%) compared to the dispatch/sync fixes (~5–
10×).

#### 2.3.3 SwiGLU / GELU on TEE side — **green** (decoder only)

**What.** Move pointwise ops (bias add, GELU, residual, LayerNorm) off
the slow `for v in m.iter_mut()` scalar Rust path. Use `rayon` +
`ndarray-rayon` or hand-SIMD (`wide` crate). Net: ~3–5 ms/text recovered.

**GELO fitness.** These ops run TEE-side only; no protocol impact.

---

## 3. The actual implementation order

Ordered by **(impact) ÷ (engineering effort)**, biased toward
preserving the GELO privacy boundary unchanged.

### Tier 1 — cubecl-runtime built-ins + protocol-preserving wins (~1 week)

The headline finding from the framework-landscape survey: **cubecl
already ships solutions for two of our biggest bottlenecks** in
`cubecl-runtime` 0.9.0-pre.5 — we just haven't wired them up. Those
go first because they cost less than a day combined and need no new
dependency.

| Step | Change | Where | Effort |
|---|---|---|---|
| 1.1 | Wire `cubecl-runtime::TuneCache` with disk persistence at `target/cubecl-cache/` | `gelo-gpu-wgpu/src/lib.rs` (`RuntimeOptions`), `cubecl-runtime/src/tune/tune_cache.rs` | ½ day |
| 1.2 | Wire `cubecl-runtime::PersistentPool` for input/output buffer reuse | `gelo-gpu-wgpu/src/lib.rs:176-177`, `cubecl-runtime/src/memory_management/memory_pool/persistent_pool.rs` | ½ day |
| 1.3 | Cherry-pick `burn-cubecl-fusion` fused-epilogue kernels for LayerNorm / GELU / add_bias (MIT attribution); fall back to SIMD/rayon CPU if GPU-side fusion turns out to need a burn-graph context we don't have | `bert/forward.rs:131-167`, `burn-cubecl-fusion` upstream | 2 days |
| 1.4 | Pad `seq_len` to {64, 128, 256, 512} buckets — collapses autotune-entry universe from "one per text" to 4, even with `TuneCache` enabled | `bert/embedder.rs:155-211`, `decoder/embedder.rs` | 1 day |
| 1.5 | Async dispatch / single sync per layer: drop per-GEMM `client.read_one()`; sync only when the unmask `Aᵀ·(U·W)` needs the result CPU-side | `gelo-gpu-wgpu/src/lib.rs:229,308,385` | 2 days |
| 1.6 | Mask QR speedup: replace scalar Rust Householder with `ndarray::dot` + faer/matrixmultiply backend | `gelo-protocol/src/mask.rs:72-142` | 1 day |

#### Bench checkpoint policy

- **After 1.1 + 1.2 (the half-day wins):** quick A/B on a **4–5 doc
  NFCorpus subset**. If TuneCache+PersistentPool alone visibly improve
  per-text wall time (target: ≥3× cold-start reduction, given that
  autotune was our biggest single hit), continue to 1.3–1.6.
- **If 1.1+1.2 don't move the needle:** pause. That points to a
  different bottleneck than autotune/buffer-alloc — likely dispatch
  sync — and chasing 1.3+ before understanding it wastes effort.
- **After all of Tier 1 lands:** full **1k-doc NFCorpus subset** as
  the next checkpoint. Compare to the pre-Tier-1 baseline + to the
  in-bench fastembed-CPU reference.

**Expected outcome after Tier 1:** ~900 ms/doc → ~100–150 ms/doc on
BGE-base. Still ~10–15× slower than fastembed CPU at single-text
batch, but in the right order of magnitude and protocol-unchanged.

### Tier 2 — Corpus-scale bottlenecks (post-migration) (~1 week)

The burn-cubecl migration (Tier 1) revealed that two categories of
cost dominate at IR-corpus scale (1k+ docs, long seq_len) that
weren't visible in the 5-doc microbench:

1. **CPU mask sampling** at long seq_len (Haar QR O(n³) — added below
   as §2.1; the headline new bottleneck).
2. **Shape-variability cost** outside autotune (kernel specialisation,
   pool reuse, buffer alloc) when M = seq_len varies per text.

Tier 2 closes both. Tier 3 (separate task) is the trait change that
unblocks the bigger architectural win (GPU utilization >> 10%).

#### 2.1 BLAS-accelerated Haar QR mask sampler (**top priority**)

**Where.** `crates/gelo-protocol/src/mask.rs:72-142` —
`sample_haar_orthogonal`.

**Today.** Hand-written scalar Householder QR with Mezzadri-2007 sign
correction. O(n³) work, all scalar f32 in inner loops, no SIMD, no
threading. At seq_len=128 it's ~25 μs/sample × 240 samples/text =
~6 ms/text. At seq_len=400 (NFCorpus medical abstract) it's
(400/128)³ × 6 ms ≈ **180 ms/text** — easily 60–80% of wall-clock
under the GELO+mask path at corpus scale.

**Algorithm choice.** Mezzadri 2007 stays — it's the canonical
Haar-uniform sampler and the security premise of GELO requires the
output to be Haar-distributed on O(n). We only change *how* we
compute QR.

**Two-step replacement** (both standard, no algorithmic change):

| Sub-step | Change | Expected wall |
|---|---|---|
| 2.1.a | Replace the inner Householder loops with BLAS-3 `ndarray::dot` calls (uses `matrixmultiply` under the hood — SIMD-vectorised, cache-tiled). The Householder step `A[k:, k:] -= 2 v vᵀ A[k:, k:]` is a rank-1 update — expressible as `α A + β vvᵀA`. | 5–10× faster QR at our n |
| 2.1.b | Use `faer::linalg::qr::no_pivoting::compute::QrDecomposition` directly — fully tuned BLAS-equivalent QR. Skip the manual Householder. Apply Mezzadri sign correction after. | Another 1.5–2× over 2.1.a |

**Why both.** 2.1.a is a 2-hour change with no new dep; 2.1.b adds
the `faer` workspace dep (~2 MB compiled) and ~half-day integration.
Land 2.1.a first; A/B the 5-doc and 1k-doc benches; only do 2.1.b if
2.1.a doesn't bring the 1k-doc bench back to under 5 min.

**Files.**
- `crates/gelo-protocol/src/mask.rs:72-142` — function body rewrite.
- `crates/gelo-protocol/Cargo.toml` — possibly add `faer = "0.21"`.
- Add a unit-test that asserts Haar-uniformity properties survive:
  the existing `orthogonality()` test in `mask.rs` checks `AᵀA = I`;
  add `det(A) ∈ {±1}` and `mask.matrix().mean()` close to 0 spot
  checks. Mezzadri sign correction is the load-bearing piece — if we
  drop it the distribution is no longer Haar-uniform.

**Privacy invariant.** The output `A` must remain Haar-uniform on
O(n). Any drift toward signed-permutation, block-diagonal, or biased
distributions would weaken the GELO security argument. Catch with
the test above.

**Expected impact at 1k-doc NFCorpus:** GELO+mask config wall-clock
from ~10 min → ~2 min (sub-step 2.1.a) → ~1 min (sub-step 2.1.b if
needed).

#### 2.2 Shape bucketing — pad `seq_len` to {64, 128, 256, 512}

**Where.** `crates/gelo-embedder/src/bert/embedder.rs` (encode +
forward call site); analogous in `decoder/embedder.rs`.

**Today.** The tokenizer produces variable-length token sequences
(typically 16–512 for BEIR docs). Each unique length seeds a new
matmul shape inside the engine. burn-cubecl's autotune anchors to
power-of-two-ish buckets internally so this is less catastrophic
than under raw cubecl, but the **buffer pool** still partitions by
exact size and the mask QR is O(seq_len³) regardless of autotune.

**Change.** After tokenize + truncate to model max, pad up to the
nearest of `{64, 128, 256, 512}` with the tokenizer's `[PAD]` token.
Attention mask must mark padding positions so they don't influence
attention scores (already supported in `bert/attention.rs`'s mask
handling — verify wiring).

**Why this matters with 2.1.** At seq_len=512 the mask QR is still
expensive (~30–50 ms even with BLAS). Bucketing caps the worst case
and means 2.1's speedup applies uniformly. Without bucketing, a
single 512-token doc dominates the bench wall-clock.

**Files.**
- `crates/gelo-embedder/src/common/tokenizer.rs` — add `pad_to_bucket`.
- `crates/gelo-embedder/src/bert/embedder.rs` — call pad-to-bucket
  after `tokenize_truncate`.
- `crates/gelo-embedder/src/decoder/embedder.rs` — same.
- `crates/gelo-embedder/src/bert/attention.rs` — verify pad-mask path
  is exercised; if not, wire it (already correct for the BGE forward).

**Trade-off.** Padding inflates compute for short documents — a
17-token doc becomes 64 tokens, ~4× the matmul work + ~64× the mask
QR work (n³). For NFCorpus medians around 150–250 tokens this is
acceptable. For a short-query workload it's not — gate via a
`with_seq_len_bucketing(bool)` builder so query-only paths can skip.

**Expected impact:** ~1.5× on the matmul side (less per-text
variance keeps autotune cache hot); ~3–5× on the mask QR side at
typical doc lengths (we stop hitting n=400+ samples).

#### 2.3 Attention-path CPU/GPU dance

**Where.** `crates/gelo-embedder/src/bert/forward.rs:72-92`,
`bert/attention.rs`.

**Today.** Per layer the current flow is:
- engine.offload_qkv → 3 matmuls + 3 mask round-trips (GPU)
- add_bias → CPU
- `multi_head_attention(q, k, v)` → CPU (softmax(Q·Kᵀ/√d)·V)
- engine.offload_linear(O) → 1 matmul + 1 mask round-trip (GPU)

The CPU attention compute reads 3 large tensors back from GPU then
re-uploads the context for the O projection. Each CPU↔GPU bounce
inflates wall-clock.

**Two options:**

| Option | What | Privacy impact |
|---|---|---|
| 2.3.a | Move multi-head attention to the trusted side ndarray-rayon path with batched `Q·Kᵀ` and `softmax(...)·V` via ndarray::dot. Stays on CPU but vectorised. | None — already TEE-side; just faster. |
| 2.3.b | Use the existing `offload_attention_qkt` (TwinShield OutAttnMult) for Q·Kᵀ on GPU. Softmax + `attn·V` either stay on CPU or go through `offload_attention_qkt_batched`. | Requires OutAttnMult correctness — the implementation exists at `out_attn_mult.rs` but wasn't fully exercised in BERT path. |

**Recommended.** 2.3.a first — pure CPU speedup, no protocol surface
change. Probably ~3× faster attention at our seq_len. Defer 2.3.b
until Tier 3 (on-device tensor handle trait) lands; then it becomes
much easier to integrate without per-call sync.

**Files.**
- `crates/gelo-embedder/src/bert/attention.rs` — rewrite
  `multi_head_attention` to use `ndarray::dot` for `Q·Kᵀ` (currently
  manual loops) and to vectorise softmax.

**Expected impact:** 1.5–2× wall-clock on the attention slice
(currently ~10% of per-text time at NFCorpus seq_len).

#### 2.4 Shorter-text fastpath gate

**Where.** `crates/gelo-embedder/src/bert/embedder.rs::embed`.

**Today.** Every text — query or doc — runs the full 12-layer mask
+ matmul pipeline. Queries are typically 5–20 tokens, where the
mask QR and shape-bucketing-padding overheads dominate the actual
inference.

**Change.** When `seq_len ≤ 16`, skip bucketing (pad only to next
power of 2) and bypass the mask round-trip entirely for the
*public-query* case — queries are not generally confidential under
GELO's threat model (the privacy target is the doc embeddings, not
the query token IDs). Doc ingest still uses the full path.

**Caveat.** This is a *threat-model decision*, not a perf-only
change. `docs/prototype/gelo.md` lists query confidentiality as a
secondary goal. Confirm with you before flipping the default.

**Files.**
- `crates/gelo-embedder/src/bert/embedder.rs` — add
  `with_query_fastpath(bool)` and a per-call `is_query` flag plumbed
  through `embed`.
- `docs/prototype/gelo.md` — document the query-confidentiality
  exception.

**Expected impact:** ~3–5× on query-phase wall-clock; zero impact
on doc ingest.

#### Tier 2 outcomes (2026-05-13)

Two of four planned sub-items landed and **crushed the target**.
Headline: **500-doc NFCorpus GELO+mask config went from a >5 min
projection to 11.4 s** — a ~30× whole-system speedup vs the
post-Tier-1 baseline, on top of Tier 1's ~25× over the
pre-migration cubecl-direct path.

| Step | Status | Commit | 500-doc bench wall-clock |
|---|---|---|---|
| Tier 1 baseline (post-burn-cubecl migration) | ✓ | `45ff345` | ~5 min projection (1k was timing out at 22 min) |
| 2.1.a BLAS-rewrite Haar QR | ✓ | `ac28462` | 80 s |
| 2.3.a Vectorize attention softmax | ✓ | `86db002` | **11.4 s** |
| 2.1.b faer QR | deferred | — | — |
| 2.2 Shape bucketing + attention mask | deferred | — | — |
| 2.4 Query fastpath (threat-model gated) | deferred | — | — |

Per-text wall-clock at 500-doc NFCorpus (avg seq_len ~150): **~20 ms**
including the full GELO mask round-trip. Better than the 5-doc
microbench's 26 ms/text — the long-tail mask cost from O(n³) Haar QR
and from softmax's O(n²) per-head-per-layer hot loop has been
largely eliminated.

#### Tier 2 — deferred sub-items (status as of 2026-05-13)

**2.1.b faer-backed QR** — deferred. Mask QR is now ~3–5% of wall-
clock; an additional 1.5–2× on it would save <2% wall. Revisit only
if a future change re-elevates mask cost (e.g., much longer corpora).

**2.2 Shape bucketing + attention mask plumbing** — deferred. The
original motivation was (a) cap mask cost on long-tail docs and (b)
keep autotune cache / persistent pool hot. 2.1.a addressed (a)
directly; (b) is less load-bearing now that the autotune cache
persists to disk per Tier 1.4. The wiring cost (attention mask
plumbing through `bert/attention.rs` + `multi_head_attention` + the
mean-pool) is real (~1–2 days) and the upside is small (CI-cold-
start fairness, marginal warm-state improvement). **Re-evaluate
after Tier 3** — bucketing pairs naturally with the on-device
tensor handle if shape variance shows up as a bottleneck there.

**2.4 Query fastpath** — deferred pending threat-model decision on
query confidentiality. Filed in task list (#78).

#### Tier 2 — *not* doing

- **Caching the mask across calls** — breaks GELO's fresh-per-batch
  property. Reject.
- **Block-diagonal mask construction** — privacy weakening that
  needs a separate security analysis. Filed in §Tier 5.
- **Random signed-permutation masks** — leaves H's sparsity pattern
  exposed; not Haar.
- **Streaming embedding (yield per-text)** — orthogonal concern.

#### Constraints on Tier 3 learned from Tier 2

Building Tier 3 on top of these results forces an honest accounting
of what's still expensive:

- **Mask round-trip per layer is intrinsic.** GELO requires the mask
  `A` to stay on the trusted side. Sending `A` (or `Aᵀ`) to the GPU
  to do the unmask there would let the untrusted side recover `H`
  from `(Aᵀ · (U·W))` and `U·W` — privacy gone. So **at minimum
  we keep one CPU↔GPU sync per layer** even with aggressive
  on-device fusion. 12 syncs per BGE-base forward stay.
- **Engine matmul still dominates (~91% of per-text wall).** The
  remaining win must come from (1) cutting redundant syncs inside
  `offload_qkv` (3 → 1 per layer), (2) enabling pointwise epilogue
  fusion (`burn-cubecl-fusion` is feature-enabled but currently
  dead because of the per-call sync), and (3) amortising input
  upload across ops that share the masked H input.

Realistic Tier 3 ceiling: 3–5× additional end-to-end speedup over
Tier 2, NOT the 10–20× I optimistically projected pre-migration.

### Tier 3 — protocol-aware fusion (target: another 2–3×, ~1.5 weeks)

Applies to whichever engine is the default after Tier 2.

| Step | Change | Where | Effort |
|---|---|---|---|
| 3.1 | Fused QKV: concatenate `[W_q ‖ W_k ‖ W_v]` at provision time; one dispatch in `offload_qkv` | `bert/weights.rs`, `bert/forward.rs:72-83`, `decoder/forward.rs` | 3 days |
| 3.2 | Fused gate-up (decoder only): `[W_gate ‖ W_up]` | `decoder/forward.rs` | 2 days |
| 3.3 | Cross-text batching: block-diagonal mask, one batched dispatch per layer | `gelo-protocol/src/sim.rs`, engine impls | 4 days |
| 3.4 | On-GPU unmask (`Aᵀ` as a fresh per-call weight, one extra GEMM, drop layer-boundary readback) | `sim.rs:186-214`, engine impls | 3 days |

**Expected outcome after Tier 3:** ~40–60 ms/doc on BGE-base. Within
striking distance of fastembed CPU at single-text, competitive at
batch.

### Tier 4 — kernel-level (target: 1.5–2× more, ~3–4 weeks; only if needed)

| Step | Change | Risk |
|---|---|---|
| 4.1 | fp16 GEMM with fp32 unmask + widened U-Verify tolerance | error analysis required |
| 4.2 | Fused softmax+V WGSL kernel (keep `Q·Kᵀ` via OutAttnMult, fuse only the post-softmax matmul) | medium, mostly mechanical |
| 4.3 | Q4_K weight-only quantization + dequant-aware U-Verify (ggml engine has this for free; cubecl engine doesn't) | error analysis + new bench |

### Tier Q — Decoder-LLM (Qwen3-Embedding-0.6B) analysis + optimizations

Most of the BGE-base optimization work doubles as Qwen3-Embedding
work because the two share the same engine + GELO+mask pipeline. The
notable structural differences in Qwen3:

- **28 layers vs BGE's 12** (~2.3× more matmul calls per text)
- **Hidden 1024, FFN 3072, 16 Q heads + 8 KV heads via GQA**
- **SwiGLU FFN (gate + up + down)** instead of BERT's single up+down
- **RMSNorm pre-LN** instead of post-LayerNorm
- **RoPE positional embedding** computed per layer
- **OutAttnMult opt-in** for Q·Kᵀ offload (large GQA heads make this
  more attractive than for BGE — Q heads contribute 16 separate
  (n, d) tensors)

#### Empirical bottleneck breakdown (2026-05-13, 3-text micro-bench)

Captured by `crates/gelo-gpu-wgpu/tests/qwen3_overhead_breakdown.rs`.
3 short prompts on AMD Radeon Graphics (RADV GFX1151) via burn-cubecl
+ Vulkan. Post-Tier-2 baseline (BLAS QR, vectorised softmax in BERT —
note: decoder paths share the same gelo-protocol mask code).

Per-text wall-clock:

| Config | ms/text | Δ vs BGE-base steady-state (~26 ms) |
|---|---:|---:|
| gpu_plain (no privacy) | 82 | 3.2× slower |
| gpu + GELO (TEE attention) | 85 | 3.3× slower |
| gpu + GELO + OutAttnMult + SEV-SNP | 108 | 4.2× slower |

The 3-4× factor tracks model size (28 layers × 1024² vs 12 × 768² ≈ 4×).

Cost shares — `gpu + GELO` (the practical default), 3-text bench:

| Bucket | ms | Share |
|---|---:|---:|
| **engine:matmul** (non-QKV: O, FFN-up, FFN-down, FFN-gate) | 156 | 61.5% |
| **engine:matmul_many** (QKV batches, 28 layers × 3 texts = 84 calls) | 59 | 23.3% |
| tee:swiglu_activate (84 calls, ~130 μs/call) | 10.9 | 4.3% |
| gelo:mask_unapply | 8.8 | 3.5% |
| gelo:mask_apply | 5.6 | 2.2% |
| tee:attn_inplace (CPU softmax+V·Attn) | 5.0 | 2.0% |
| tee:rmsnorm (171 calls = 28×2+1 per text) | 3.7 | 1.5% |
| gelo:mask_sample | 2.4 | 0.9% |
| tee:rope + residual + shield bookkeeping | 2.4 | 1.0% |

**Engine matmul = ~85% of wall** — same structural shape as BGE.

OutAttnMult cost (only when enabled):

| Δ Bucket | Δ ms (added) | Δ ms/text |
|---|---:|---:|
| outattn:setup_stack_batched (CPU: sample fillers, build 2n×d, permute) | +43.0 | +14.3 |
| engine:matmul_dynamic_batched (offloaded Q·Kᵀ) | +20.8 | +6.9 |
| outattn:recover_batched | +2.1 | +0.7 |
| tee:softmax_av (− tee:attn_inplace) | −1.8 | −0.6 |
| gelo:* delta (slight uptick) | +5 | +1.7 |
| **Net** | **+69** | **+23** (+31% over GELO-only) |

#### Q-series tasks

Following the same logic as BGE Tier 2: tackle CPU-side wins, leave
the engine-kernel ceiling alone (already at burn-cubecl best).

**Q1 — Vectorize decoder CPU epilogues** (~½ day, low risk):
- `tee:swiglu_activate` — 10.8 ms over 84 calls; scalar inner loop with
  `silu(x) = x / (1 + (-x).exp())` per element. Tighten to slice-iter
  contiguous &mut [f32] passes, fuse silu computation with the
  element-wise multiplication against `up`. Expected ~2-3 ms/text
  recovered.
- `tee:attn_inplace` — softmax + Attn·V CPU code in the decoder
  attention path. Verify it has the same fused max+exp+sum +
  reciprocal-multiply pattern that fixed BERT's softmax (Tier 2.3.a).
  Expected ~1-2 ms/text recovered.
- `tee:rmsnorm` — single-pass sum-of-squares is already in place per
  Tier 2.3.a / R3; verify no regression at decoder shape (n=20, d=1024).
- **Total expected:** ~3-5 ms/text reduction = ~4-6% wall improvement.

**Q2 — `offload_linear_many` for FFN gate+up** (~½ day, abstraction
parity with `offload_qkv`):
- Decoder's `forward.rs:148-160` issues two separate `offload_linear`
  calls for FFN-gate and FFN-up, both with the same input `h_norm`.
- Mirror the `offload_qkv` pattern via a new
  `TrustedExecutor::offload_linear_many(&[handles], hidden)` that
  delegates to `engine.matmul_many`. Saves one upload + one sync per
  layer × 28 layers per text.
- Same empirical caveat as Tier 3 step 1: at our shape sizes the
  observable wall-clock win is likely in the noise. Worth doing for
  abstraction parity.

**Q3 — Vectorize OutAttnMult setup** (~1 day, only if we want
OutAttnMult on by default):
- `outattn:setup_stack_batched` is 511 μs/call × 84 = 43 ms over the
  3-text bench. The per-call work is:
  1. Sample two Gaussian filler arrays (`r_q` and `r_kt`)
  2. Build doubled stacked operands (2n × d) and (d × 2n)
  3. Random permutation per axis (Fisher-Yates O(n))
  4. Apply row/col permutation
- Optimizations:
  - Batch the Gaussian sample across all heads via `ndarray::Array::from_shape_fn`
    with one RNG (already StandardNormal-distrib via rand_distr)
  - Permute index buffer reuse across heads
  - SIMD-friendly slice copies for stacked-operand build
- Expected: cut ~14 ms/text → ~5 ms/text on OutAttnMult config.

**Q4 — Wire Qwen3 into BEIR/NFCorpus bench** (~½ day, gates Q1/Q2/Q3):
- Currently the BEIR bench (`crates/approach4/tests/beir_accuracy.rs`)
  only has BGE-base configs. Add a `BEIR_QWEN3=1` env gate that runs
  a `GeloQwenEmbedder + GELO+mask` config alongside the BGE+mask one.
- Validates retrieval correctness at corpus scale (the breakdown test
  is only 3 short prompts; nothing tells us Qwen3 holds protocol
  fidelity at NFCorpus seq_len).
- Gives a corpus-scale baseline number to optimize against.
- Defer Q1-Q3 prioritization until Q4 reveals the actual corpus-scale
  bottleneck split (long-tail NFCorpus seq_len may shift the
  relative weight of CPU epilogues vs engine matmul).

#### Tier Q — *not* doing

- **fp16 Qwen3 engine** — already validated on BGE that cubecl-wgpu's
  f16 path doesn't win on AMD RDNA3. Qwen3's 28 layers would only
  amplify the cold-start autotune problem.
- **Q4_K weight quantization** — needs cubecl quantized-matmul kernels
  we don't have. Filed in Tier 5 (deferred).
- **Replacing Qwen3 with a different model** — outside scope; this is
  perf work, not model-selection work.

### Tier 5 — research items (deferred, future-rnd.md)

- **Alternative engine: `gelo-gpu-ggml`** — vendored ggml/Vulkan
  build as a second `GpuOffloadEngine`. Deferred because the
  burn-cubecl migration (Tier 1) covered the practical win that
  motivated this exploration. Re-evaluate if Tier 3 (on-device
  tensor handle) doesn't close the GPU-utilization gap, or if we
  need Q4_K quantized weights (ggml has this for free). Source
  spike investigation recorded in `docs/prototype/` (this file's
  earlier revisions) and the agent transcripts.
- Block-diagonal mask sampling (privacy weakening, needs analysis).
- Full flash attention with TwinShield-compatible Q·Kᵀ embedding —
  open research problem.
- INT8 / Q4 weights with constant-time decoding for side-channel safety
  (production-only concern).
- Revisit IREE/iree-rs in ~12 months if the Rust bindings harden — its
  Vulkan AOT codegen is what cubecl will compete with long-term.

---

## 4. What to *not* do

- **Don't replace cubecl with a hand-rolled WGSL kernel.** cubecl's
  autotuned SGEMM is fine; the bottleneck is around the GEMM, not in
  it. Replacing the kernel is high-effort, low-yield.
- **Don't switch backends to CUDA-only / Metal-only.** GELO's threat
  model wants OEM-agnostic operation; locking the offload engine to one
  vendor's stack would close that door.
- **Don't push the mask onto the GPU naively.** The mask must come from
  TEE-side randomness for the privacy argument to hold. The "on-GPU
  unmask" trick in step 2.4 keeps the mask sampled in the TEE and only
  uploads `Aᵀ` as a per-call weight; this is mask-equivalent, not
  mask-on-untrusted-GPU.
- **Don't add MoE / sparse-expert routing to gain headroom.** Embedding
  models are dense; nothing to gain.

---

## 5. Validation checklist for each tier

Every step in §3 must preserve:

1. **Functional parity.** `cargo test -p gelo-embedder --release` green.
   Specifically: `masked_and_plaintext_executors_agree`,
   `qkv_shares_one_mask`, `mock_report_is_rejected_under_mismatched_dp_config`.
2. **Protocol fidelity.** `cargo test -p approach4 --release --test
   beir_accuracy -- --ignored` green; `top1_vs_plain ≥ 0.95` and
   `rec@10_vs_plain ≥ 0.95` for GELO+mask configs at 1k-doc subset.
3. **U-Verify probes green** at `verify_probes = 4`. Tolerance widening
   for quantized variants (Tier 3) needs a new tolerance bound test.
4. **Attestation rebinding** if any protocol surface changes (fused
   QKV registers a new weight handle kind → `config_digest` must
   include it).

### Bench checkpoint targets

| Tier | BGE-base ms/doc (single-text) | BGE-base ms/doc (batch=32) |
|---|---|---|
| baseline (today) | ~900 | not measured |
| post-Tier 1 mid-checkpoint (1.1 + 1.2 only, 4–5 doc subset) | ≤ ~300 (≥3× cold-start cut) | n/a |
| post-Tier 1 full (1k-doc subset) | ~100–150 | ~60 |
| post-Tier 2 (ggml engine, if it wins the A/B) | ~70–100 | ~40 |
| post-Tier 3 | ~40–60 | ~20 |
| post-Tier 4 | ~25 | ~10 |
| fastembed CPU reference | ~10 | ~3 |
| llama.cpp Vulkan reference (`llama-cpp-rs --features vulkan`) | ~5 | ~2 |

We will not match llama.cpp at small batch on iGPU — the GELO mask
round-trip imposes a structural overhead we can't optimize away. But
within 3–5× of fastembed CPU is achievable and is the target.

---

## 6. Open questions / decisions needed

1. **Tier 1 step 1.5: one-sync-per-layer vs on-GPU unmask?** The latter
   is cleaner but uploads `Aᵀ` per layer. At seq_len=128 that's a
   128×128 = 64 KB upload per layer × 12 = 768 KB/text — small. I lean
   toward on-GPU unmask if the cubecl `create_from_slice` cost is low
   enough; will measure during implementation. (On-GPU unmask formally
   moved to Tier 3 step 3.4.)

2. **Tier 3 step 3.3: block-diagonal mask within a batch?** The
   privacy argument is that adversary sees `A_i · h_i` for each text
   independently, identical to today's per-text mask. I believe this is
   sound but should write it up in `gelo.md` appendix.

3. **Tier 4 step 4.1 / fp16:** does cubecl support fp16 on Vulkan
   today? Last I checked the `bf16` path exists for CUDA only. Worth
   confirming before scheduling Tier 4. (ggml-vulkan has F16/BF16
   pipelines either way, so on the ggml engine this is "free".)

4. **Tier 1 step 1.3 / burn-cubecl-fusion integration shape:** burn's
   fused epilogues live inside `burn-cubecl-fusion` and assume a burn
   graph context. If lifting the kernels standalone proves to require
   reimplementing more of the burn runtime than the win is worth, fall
   back to a SIMD/rayon CPU LayerNorm/GELU/add_bias (the original
   plan) — same speedup target, easier integration.

---

## 7. Pointers

- `crates/gelo-gpu-wgpu/src/lib.rs` — engine, where most Tier 1 work lands.
- `crates/gelo-embedder/src/bert/{forward,embedder}.rs` — BERT path,
  Tier 1 step 1.3/1.4 + Tier 3 fused-QKV.
- `crates/gelo-embedder/src/decoder/{forward,embedder}.rs` — decoder
  path, Tier 3 fused gate-up.
- `crates/gelo-protocol/src/{sim,mask,substrate}.rs` — TEE side, Tier 1
  step 1.6 + Tier 3 step 3.3 + 3.4.
- `~/.cargo/registry/src/.../cubecl-runtime-0.9.0-pre.5/src/tune/tune_cache.rs`,
  `.../memory_management/memory_pool/persistent_pool.rs` — the already-on-disk
  facilities that Tier 1 steps 1.1 + 1.2 wire up.
- `https://github.com/ggml-org/llama.cpp` (`ggml/`) — vendoring target for Tier 2.
- `docs/prototype/gelo.md` §3.4 — the GELO "bandwidth-limited TEE"
  variant that motivates on-GPU unmask.
- `docs/prototype/future-rnd.md` — where block-diagonal mask sampling
  + full flash attention will be filed if/when they become real.
