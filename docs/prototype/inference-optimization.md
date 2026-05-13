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

### Tier 2 — bottlenecks at corpus scale (post-migration)

#### 2.0 Mask-QR sample at long-seq_len corpora (**new bottleneck — discovered 2026-05-13**)

**Symptom.** On the 1k-doc NFCorpus run, the bench wall-clock grew far
beyond the per-text projection. Per-text steady-state at 5 short docs
showed `gelo:mask_sample` at only 1.7% of wall (~0.4 ms/text). On real
NFCorpus medical abstracts the texts are ~150–400 tokens (sometimes
hitting the seq_len cap), and the Haar QR is O(n³).

**Mechanism.** `gelo-protocol/src/mask.rs:72-142` implements Mezzadri
2007 Householder QR with sign correction — the canonical Haar-uniform
sampler. At seq_len=128: ~2 M ops × 240 mask samples/text × 1k texts
≈ 500 G-ops over the bench just for mask sampling, dominantly scalar
single-thread CPU. The cost is GELO-mandated only to the extent that
**Haar-uniform** orthogonal masking is required; the *specific
sampler* is our implementation choice.

**Privacy boundary.** GELO requires `A ∈ O(n)` drawn from the Haar
measure on each batch — uniform among all orthogonal matrices. The
mask structure is the load-bearing security premise; the sampler is
not. Any cheaper construction that preserves Haar-uniformity is fine;
anything that biases the distribution weakens the privacy argument and
needs analysis.

**Candidate fixes** (ranked by impact ÷ engineering effort):

| Approach | Cost | Notes |
|---|---|---|
| 2.0.a | Replace scalar Householder with BLAS-accelerated QR (faer's `Qr::new` or matrixmultiply via ndarray::dot) | Same algorithm, SIMD + cache-friendly — expect 5–10× on the QR itself. No protocol change. |
| 2.0.b | Cap seq_len at our embedder bucket (e.g. 128) and pad — keeps Haar dim small and uniform | Already a hot Tier 1 item (1.4 shape bucketing); doubles as a privacy-neutral mask cost cap. |
| 2.0.c | Sample one mask per batch covering N texts: `A ∈ O(N·n)`. Cost grows as O((N·n)³), so only wins if we're already paying it; doesn't help here. | Skip. |
| 2.0.d | Block-diagonal mask `diag(A_1, …, A_k)`: small blocks, O(k·b³) instead of O(n³) | **Privacy weakening**; flagged in §Tier 5 — needs analysis before adoption. |
| 2.0.e | Cache the mask across calls (same `A` for multiple batches) | **Privacy weakening**; the GELO guarantee is fresh-per-batch — breaks the security argument. Reject. |

**Plan.** Do 2.0.a (BLAS-accel QR) and 2.0.b (shape bucketing) together
as the entry to Tier 2. Expected per-text wall-clock at 1k-doc scale:
~150 → ~60 ms (post-bucketing) → ~30 ms (post-BLAS QR), bringing the
1k-doc run from minutes back to ~tens of seconds.

### Tier 2 — Alternative engine: `gelo-gpu-ggml` (~1 week)

Build a second `GpuOffloadEngine` implementation backed by ggml's
Vulkan backend, behind `--features ggml-engine`. Default stays cubecl
until benchmarks justify the switch.

**Why a second engine, not a replacement:** ggml-vulkan ships
hand-tuned SGEMM kernels with three warptile presets selected by
static heuristic (no autotune cold-start), a real buffer pool, and
two years of AMD/Intel/NVIDIA Vulkan production hardening. Expected
warm-vs-warm speedup over cubecl is **1.3–2.0×**, but the bigger
qualitative win is no per-shape autotune and no per-dispatch sync.
The risk is that we add a C/C++ build dep and a second engine to
maintain, so we feature-gate and decide on the merge based on the
A/B numbers.

`llama-cpp-rs --features vulkan` is the **reference benchmark**, not
a dependency. Its `wrapper.h` doesn't expose `ggml.h`, so it gives us
nothing usable for per-GEMM offload. We build our own slim FFI.

| Step | Change | Effort |
|---|---|---|
| 2.1 | New `crates/gelo-ggml-sys`: vendored `ggml/` subtree from upstream `ggml-org/llama.cpp`, built `GGML_VULKAN=ON` (no llama runtime, no other backends) via cc/CMake build script | 2 days |
| 2.2 | Hand-written C shim exposing `engine_init`, `register_weight`, `matmul`, `matmul_dynamic`, `free`. Internals: one persistent `ggml_backend` + `ggml_gallocr`; per-call 1-node `cgraph` around `ggml_mul_mat`, `ggml_backend_tensor_set/get` for input/output | 1 day |
| 2.3 | New `crates/gelo-gpu-ggml`: Rust wrapper implementing `GpuOffloadEngine` over the FFI; mirror trait surface of `gelo-gpu-wgpu` so the rest of the stack is engine-agnostic | 1 day |
| 2.4 | Wire `--features ggml-engine` into `beir_accuracy` bench; A/B against post-Tier-1 cubecl on 1k-doc NFCorpus subset. Concurrently run `llama-cpp-rs --features vulkan` standalone (whole-model forward, plaintext) as the hand-tuned reference | 1 day |

#### Decision criterion

If `gelo-gpu-ggml` beats post-Tier-1 cubecl by **≥1.5× warm AND ≥10×
cold-start**, flip default and consider deprecating `gelo-gpu-wgpu`.
Otherwise keep cubecl as default and ggml as fallback/reference.

#### Known caveats

- ggml-vulkan has `graph_plan_create = NULL` — every call rebuilds
  the Vulkan command buffer. For 1-node graphs this is ~150 μs and
  amortizes fine at our shapes, but it's not a CUDA-Graph-style
  "record once, replay N" pattern.
- Intel iGPUs hard-code `support_async = false` in ggml-vulkan: every
  `graph_compute` synchronously waits. AMD/NVIDIA pipeline transfers
  but still submit per graph. We target AMD primarily.
- Build complexity rises (CMake, glslc/glslang for SPIR-V, vendored
  ~17 kLOC of C++ in our tree). TCB-irrelevant since the engine runs
  untrusted, but build/reproducibility story thickens.

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

### Tier 5 — research items (deferred, future-rnd.md)

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
