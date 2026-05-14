# GELO for LLM Inference (forward-looking)

> **Scope.** Design notes for extending the GELO/TwinShield protocol from
> the embedding workload (covered in `gelo.md`) to **LLM answer generation**.
> Nothing in this document is implemented; it's a forward-looking plan that
> sits behind a workload trigger.
>
> The protocol primitives already exist in the codebase
> (`crates/gelo-protocol/src/attention.rs` — Tier 1 phases 1-6 landed
> 2026-05-14, see commits `3b5b587..fffce6e`). What's missing is (a) the
> LLM-serving harness, (b) the upstream burn-cubecl gap that gates the
> fused-attention path, and (c) the decode-phase KV-cache primitives.

---

## 1. Why this is separate from `gelo.md`

`gelo.md` is the embedding prototype: single forward pass, last-token pool,
no autoregressive generation, no KV cache. At Qwen3-Embedding-0.6B's
typical sequence lengths (n ≈ 100-500), in-TEE attention is already cheap
enough that the permutation-shielded attention path **regresses** wall-clock
(measured in Phase 6: 153 → 300 ms/text). The path lives in the codebase
but defaults to off.

LLM answer generation is a different regime. The protocol gains we shelved
for embedding become real once `n` is in the thousands (RAG prefill
context) and once attention cost dominates per-layer compute. The
deferred-fused-attention work re-emerges as the right next move when
this workload lands.

---

## 2. Workload characterization: prefill vs decode

Answer generation has two distinct phases with very different attention
shapes. The protocol's privacy story has to handle both.

### 2.1 Prefill

Process the full prompt (system message + retrieved RAG context + user
question) in one forward pass to populate the KV cache. Typical sizes for
RAG-served LLMs:

| Workload | n at prefill | dominant cost |
|---|---|---|
| Short Q&A | 500-1k tokens | linear projections |
| Standard RAG | 2-8k tokens | attention `O(n²)` |
| Long-context RAG (16k+) | 16-128k tokens | attention dominates by 10×+ |

Compute at n = 4096 (Qwen3-0.6B class):
- One attention block: 16 heads × 4096² × 128 ≈ **34 GFLOPs**
- 28 layers: **~950 GFLOPs per prefill**
- On integrated GPU @ ~10 TFLOPS: ~95 ms compute-bound
- Score-tensor memory traffic (heads, n, n) = **1 GB** per layer, traversed 3× in our 3-dispatch path = **~3 GB/layer × 28 = ~90 GB bandwidth-bound**

At this regime FlashAttention's wins materialize (analysis in `gelo.md`
§3.5b extends here). Wall-clock for permuted attention drops from ~3 s
(3-dispatch with materialized scores) to ~150-200 ms (fused, streaming
softmax) — a ~20× wall-clock improvement specifically at long-context
prefill.

### 2.2 Decode

Generate one token at a time, attending the new token's Q to all previous
K, V (which sit in the KV cache). Per-token cost:

```
Per token:
  - Project (1, d) → Q, K, V                     # tiny
  - Append new K, V to KV cache (n_cache grows)
  - Attention: Q · K_cacheᵀ shape (1, n_cache)   # small
  - softmax · V_cache → output (1, d)
  - Project, FFN, normal stuff
```

Per-step shape: `n_q = 1`, `n_kv = n_cache` (growing). Per-step compute is
much smaller than prefill — `O(n_cache · d_head)` rather than `O(n² · d_head)`.

The KV cache itself is the heavy state: 28 layers × 2 (K + V) × n_cache × d_total
per head, packed across heads. For Qwen3-0.6B at n_cache = 4k: ~440 MB of
KV cache to manage across decode steps.

**Decode is where the protocol gets architecturally awkward**, see §4.

---

## 3. Prefill: fused permuted attention

This is the deferred work from `gelo.md` §3.5b. The audit (committed as
`fffce6e`) identified the engineering shape:

### 3.1 What's already built

| Piece | Status | Location |
|---|---|---|
| Math: `softmax(πAπᵀ) = π·softmax(A)·πᵀ` | ✅ proven, 13 tests | `crates/gelo-protocol/tests/permutation_attention.rs` |
| Permuted causal mask | ✅ `AttentionMask::Causal`, parity-tested | `crates/gelo-protocol/src/attention.rs` |
| Substrate trait method | ✅ `TrustedExecutor::offload_attention_permuted` | `crates/gelo-protocol/src/substrate.rs` |
| 3-dispatch GPU path | ✅ `matmul + softmax + matmul` via engine | `crates/gelo-protocol/src/attention.rs:permuted_attention` |
| Decoder wrapper | ✅ `causal_gqa_attention_permuted` | `crates/gelo-embedder/src/decoder/attention.rs` |
| 3-way autoswitch | ✅ in-TEE / permuted / OutAttnMult | `crates/gelo-embedder/src/decoder/forward.rs` |
| Fused-attention engine method | ❌ NEW | would live in `gelo-gpu-wgpu` |
| `causal: bool` parameter on burn-cubecl flash | ❌ NEW upstream | `burn_cubecl::kernel::attention::flash_attention` |

### 3.2 What needs to land

**Engine trait extension**:
```rust
fn fused_attention_batched(
    &self,
    q: ArrayView3<f32>,
    k: ArrayView3<f32>,
    v: ArrayView3<f32>,
    scale: f32,
    mask: Option<ArrayView3<f32>>,  // for permuted causal
) -> Result<Array3<f32>>;
```

Default impl: composed (3-dispatch) — what `permuted_attention` does today.

Wgpu override: calls `cubek::attention::launch::launch_ref` directly with
`AttentionOptions { causal: false, ... }`, passing our additive permuted mask
in the `Materialized` slot. CubeTensor handle plumbing via
`tensor.into_primitive()` matched on `TensorPrimitive::Float`. Estimated
~150 LOC including the dtype/precision setup.

**Protocol switch**: `attention::permuted_attention` checks for the new
engine capability and prefers `fused_attention_batched` when available.
Falls back to the composed path otherwise.

### 3.3 Why this is deferred, not in flight

The upstream gap: `burn_cubecl::kernel::attention::flash_attention`
hardcodes `causal: true` (see `burn-cubecl-0.20.1/src/kernel/attention/base.rs:46`).
For our case we need `causal: false` plus our permuted-causal mask as the
sole mask. Options:

1. **Upstream PR** to parameterize `causal: bool`. Lowest maintenance, blocks
   on tracel-ai/burn merge cycle.
2. **Fork the wrapper into `gelo-gpu-wgpu`** and call `cubek::attention::launch::launch_ref`
   directly with `causal: false`. Adds `cubek` and `cubek-attention` as
   direct deps. `cubek-attention` is v0.1.1 (June 2025) — young and
   likely API-unstable.
3. **Write a custom WGSL fused-attention kernel.** ~500 LOC, FlashAttention-style.
   Lowest dependency surface; highest implementation risk and ongoing
   maintenance.

When this work starts, the right decision likely depends on cubek's
maturity at that time. If cubek-attention has reached v0.5+ by then,
option (2) becomes the default. If burn-cubecl has already parameterized
causal upstream, option (1) is free.

### 3.4 Expected payoff (rough, at the workload above)

| Configuration | n=4096 prefill wall (Qwen3-0.6B est.) |
|---|---:|
| In-TEE attention | ~3-5 s (CPU bandwidth-bound on `n²·d`) |
| Permuted + 3-dispatch GPU | ~500 ms (1 GB score tensor traffic per layer) |
| **Permuted + fused FlashAttention** | **~150-200 ms** (no `n²` materialization) |
| Plain (no privacy) baseline | ~100 ms |

So the fused permuted path lands within ~2× of the unprotected baseline at
long-context prefill, vs the in-TEE path being effectively unusable.

---

## 4. Decode: a separate primitive entirely

Permuted attention as designed doesn't smoothly extend to autoregressive
decoding. Three issues:

### 4.1 Fresh per-batch π conflicts with cached state

The protocol's security argument depends on fresh π_b per forward pass.
In decode, each new token IS a new forward pass — but the KV cache from
previous tokens was permuted under the *previous* batch's π. If we sample
a fresh π for the current decode step, the cached K, V are in the wrong
permuted frame.

Options:
- **Carry π forward across decode steps.** Weakens security: the engine
  sees one π reused across N decode steps for the same generation, which is
  the multi-batch-shared-π attack surface we deliberately avoided in embed
  by per-batch sampling. Possibly OK if the user's generation is treated
  as one "session" with one persistent π, but needs a fresh analysis.
- **Re-permute the KV cache each decode step.** The KV cache is hundreds
  of MB for Qwen3-0.6B at n_cache=4k. Permuting it costs O(n_cache · d_total)
  per layer per step — at 28 layers and 440 MB total, that's a 12 GB
  permute-write per token. Dwarfs any other decode cost. Not viable.

### 4.2 Per-token dispatch overhead reappears

Decode is the regime where dispatch count starts mattering again. Each
generated token = 28 layers × per-layer GPU calls = lots of small ops on
small tensors. At `n_q = 1` the attention compute is `O(n_cache · d_head)`
≈ ~50K ops per head — completely dwarfed by the ~3 ms dispatch overhead on
integrated GPU.

This makes decode look more like the embedding workload than like prefill —
the wins from fused attention shrink, and the protocol's overhead grows
proportionally.

### 4.3 The right primitive: KV-cache encoding (SCX-style)

The SIGCOMM '25 SCX paper (`yuanmu97/scx`) targets exactly this regime:
**stateless KV-cache encoding** via per-user keys, optimised for the
decode loop. Their construction:

- Encode K, V at write time with a key derived from `(session_id, layer_id,
  position)`
- Decode-step attention uses encoded K, V directly without per-step
  re-encoding
- Each decode step pays only per-token encoding overhead, not full-cache
  permutation

SCX is the natural complement to permuted attention for prefill: use perm
attention to populate the KV cache; from that point on, switch to SCX-style
encoded-KV reads.

This is not a paper we've adopted yet. When LLM serving lands, evaluating
SCX as the decode-phase primitive is the next research spike.

### 4.4 What's documented elsewhere

- `private-rag/memory/gelo_research_round_2.md` — SCX classified as "highest
  relevance" published project. Github at `yuanmu97/scx`, 36 ms LLaMA-7B
  latency claimed.
- `gelo.md` §3.5b — explains why permuted attention regresses at embedding
  shape, with the bandwidth math that extends here for prefill.

---

## 5. End-to-end deployment shape (target picture)

When the full LLM serving path is built, the per-request flow looks like:

```
Client → TLS → CVM (SEV-SNP) ─── attest ───► relying party
                  │
                  ▼
            ┌───────────────────────────────────┐
            │  Trusted Executor (in CVM)        │
            │  • Tokenize prompt                │
            │  • Sample session π_b, A_b        │
            │  • For each layer:                │
            │     ┌─ Linear projections ──┐     │
            │     │  GELO mask + offload  │ → GPU (untrusted, attested)
            │     │  (existing code path) │     │
            │     └────────────────────────┘    │
            │     ┌─ Attention ───────────┐     │
            │     │  PREFILL → fused perm │ → GPU
            │     │           attention   │     │  ← NEW
            │     │  DECODE  → SCX KV-enc │ → GPU
            │     │           attention   │     │  ← NEW
            │     └────────────────────────┘    │
            │  • Sample next token              │
            │  • Append to KV cache             │
            │  • Loop until EOS                 │
            └───────────────────────────────────┘
                  │
                  ▼
                Client (decoded text)
```

The existing GELO mask code path covers all the linear projections
(per-layer Q, K, V, O, FFN gate, up, down) without modification — the
same `provision_weight_shared`, `offload_qkv`, `offload_linear_many`
machinery from the embedder. The new code is the attention paths.

---

## 6. Engineering plan when prioritised

A pragmatic ordering:

| Step | Effort | Gate |
|---|---|---|
| 1. LLM-serving harness (load decoder for generation, sampling loop, KV cache) | 1-2 weeks | None — pure engineering |
| 2. Profile prefill with current 3-dispatch permuted attention at n ≥ 1024 | 2 days | Step 1 |
| 3. Audit `burn-cubecl` flash_attention for `causal: bool` parameter status | 1 day | None |
| 4. Either upstream PR (option 1) or `cubek`-direct wrapper (option 2) | 1-3 days | Step 3 |
| 5. Wire `fused_attention_batched` engine trait, override on wgpu | 2-3 days | Step 4 |
| 6. Bench prefill: 3-dispatch vs fused | 1 day | Step 5 |
| 7. Decode primitive selection — port SCX-style encoded KV-cache or evaluate alternatives | 1-2 weeks | Step 2 |
| 8. End-to-end answer-generation bench under full protocol | 1 week | Steps 5 + 7 |

Total: roughly **5-7 weeks** of engineering from the day the workload is
prioritised. The work has no shared dependencies with current embedding
work — running it in parallel is feasible.

---

## 7. Trade-off summary (when this lands)

| What we'd give up | What we'd get |
|---|---|
| Simplicity of single-forward-pass embedding inference | Answer-generation capability under the GELO+TwinShield+perm protocol |
| The "no auto-regressive complications" property of embedding | Real users can run RAG end-to-end privately, not just retrieve |
| Additional dependency surface (`cubek` direct dep) | Long-context prefill within ~2× of unprotected baseline |
| Engineering attention currently focused on `gelo-snp-runner` T3 | A second concrete deployment shape for the protocol |
| Decode-phase complexity | Encoded-KV path validated as a separate primitive (useful even outside LLM contexts) |

---

## 8. Decisions deferred to time-of-implementation

These are notes for the future implementer, not commitments:

- **Decode-phase π**: persistent across a generation session, or
  per-step refresh? Requires fresh security analysis vs Hidden No More /
  ARROWMATCH attack class extensions to autoregressive settings.
- **KV-cache memory footprint** in the CVM: a 7B-class decoder with
  long context can have multi-GB KV cache. Whether that lives in
  encrypted CVM RAM (cheap to reach, expensive to allocate at scale)
  or in encoded form in shared (SWIOTLB) memory is a per-deployment
  decision.
- **Speculative decoding**: many production LLM serving stacks use
  speculative decoding for throughput. The protocol implications of a
  draft-model + verify-step pattern are completely unexplored.
- **Streaming output**: whether tokens stream out as they're generated,
  or batch at end-of-generation. Streaming requires the CVM to handle
  per-token TLS writes, which complicates the runner's request lifecycle.

---

## References

- `gelo.md` — the embedding prototype that this document extends.
- Commits `3b5b587..fffce6e` — the Tier 1 phase 1-6 work that built the
  permuted attention protocol path.
- Wang et al., "Amulet: Fast TEE-Shielded Inference for On-Device Model
  Protection," arXiv 2512.07495 — source of the softmax-permutation
  equivariance technique.
- Wang et al., "Hidden No More," arXiv 2505.18332 — σ-noise mitigation
  for permutation-based schemes.
- Yuan et al., "SCX: Stateless KV-Cache Encoding for Cloud-Scale
  Confidential Transformer Serving," SIGCOMM 2025 (`yuanmu97/scx`).
  Candidate decode-phase primitive.
- Dao et al., FlashAttention v1-v4 — the tiling/online-softmax algorithm
  that makes long-context attention bandwidth-tractable.
- `private-rag/memory/gelo_research_round_2.md` — research spike that
  identified SCX, Amulet, and the related attack literature.
