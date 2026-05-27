---
type: plan
status: current
created: 2026-05-21
updated: 2026-05-27
tags: [m1.11, decode]
---

# M1.11 — Optimal Batched Decode + Batched Prefill

> **Parent context:**
> - Handoff: [`2026-05-21-attn-offload-spike.md`](../archive/handoffs/2026-05-21-attn-offload-spike.md) (cubek-attention spike, GPU offload at decode m=1 confirmed non-viable on Strix Halo)
> - Handoff: [`2026-05-21-gelo-perf-shield-attn-batched.md`](../archive/handoffs/2026-05-21-gelo-perf-shield-attn-batched.md) §C (initial batched-decode scoping)
> - Parent plan: [`m1-10-fused-permuted-attention.md`](m1-10-fused-permuted-attention.md)
>
> **Status:** D2 orchestrator rewire shipped 2026-05-27 (commit `d241a7a`). D1 substrate + D2 orchestrator landed; D3 perf validation captured on v7 fixture (§8 #12). c5 AloePri shared-A gate still open.
> **Owner:** open.
> **Author date:** 2026-05-21.
> **Revision:** post-grilling decisions log added 2026-05-21 (§8); D2 measurement appended 2026-05-27.

---

## 0. TL;DR

The 89 s `tee:attn_cached` bottleneck cannot be moved off the CPU at
B=1, m=1 on this hardware — kernel-launch latency dominates compute
for every GPU strategy we tried (cubek-attention Unit, burn-tensor
chain, blackbox-accelerated). The structural fix is **batched
forwards**: amortise dispatch overhead, GELO mask, and shield-stack
across B sequences in flight.

But "batched decode" is two distinct workloads, and the GPU kernel
that wins is different for each:

| Workload | Shape per step | Right kernel | First user |
|---|---|---|---|
| **Batched prefill** (reranker, embedder) | `(B, n_prompt, d)`, `n_prompt ≥ 32` | **cubek-attention `Strategy::Unit`** (`1.24 ms` at n=64; full stage utilisation) | `CausalDiscriminatorRerankService` |
| **Batched decode** (LM generation) | `(B, 1, d)`, B ≥ ~12 | **burn-tensor `fused_attention_batched`** chain (already wired) | `decoder::generation::generate_batched` |
| Single-stream decode m=1 | `(1, 1, d)` | **in-TEE GEMV** (current; ~2 ms) | unchanged |

**Recommended landing order:** rerank first (R1–R3), then decoder (D1–D3).
Rerank is structurally batched-prefill (no autoregressive loop), gives
the largest measured win against existing code, and exercises every
new substrate primitive before the harder decoder migration.

---

## 1. Why decode-m=1 GPU offload doesn't work (recap, in design terms)

The cubek-attention `Strategy::Unit` kernel parallelises across the
**`seq_q` axis** with `plane_dim ≈ 32–64` lanes per stage. At decode
`n_q = 1` the kernel pays full workgroup launch overhead for a single
useful lane — the remaining 31–63 lanes idle. Batching across
sequences puts B in the *batch* axis, not `seq_q`, so each batch
element still wastes lanes the same way. **cubek-attention is the
right kernel for prefill, not for decode.**

The burn-tensor `matmul → mul_scalar → softmax → matmul` chain in
`WgpuVulkanEngine::fused_attention_batched` (lib.rs:535+) **does**
scale with B in the way we want: each of the 3 underlying GPU
dispatches becomes a single batched GEMM/softmax where B amortises
launch overhead. At B=1, m=1: ~22 ms. At B≈16: extrapolated 25–30 ms
(launch fixed, compute grows linearly with B). Crossover vs in-TEE
GEMV (~2 ms × B = 32 ms at B=16) sits around **B ≈ 12–16**. This must
be measured before flipping defaults — the extrapolation is from the
A1+A2 microbench at B=1.

So the two-kernel split is non-negotiable for this hardware. The
public design surface should hide it behind one
`forward::run_batched` / `generate_batched` API and let the engine
choose per shape.

---

## 2. What "best practice" looks like for this codebase

Standard LLM-serving optimisations — vLLM continuous batching, PagedAttention,
FlashAttention-2/-3, speculative decoding — were designed for **non-private**
inference. Most don't carry over without adjustment:

| Technique | Carries over? | Reason |
|---|---|---|
| Static batched prefill + EOS-padding decode | **Yes** | Pure shape work; orthogonal to GELO mask |
| Right-padding with attention mask | **Yes** | Mask just adds another `(B, n_q, n_kv)` additive tensor |
| Continuous batching (in-flight insertion) | **Defer** | Needs scheduler; works against GELO's per-forward-pass mask `A` (a new sequence joining mid-batch invalidates the mask seed) |
| PagedAttention / block KV cache | **Defer** | Win is host memory pressure on dGPU; iGPU shares system RAM. Not useful here. |
| FlashAttention-2/-3 (single fused kernel) | **Yes for prefill only** | cubek-attention is our FlashAttention-class kernel; ships with the dependency already |
| Speculative decoding | **No (yet)** | Cross-cuts with sampling determinism (greedy parity is load-bearing for byte-identical extraction) |
| KV quantisation (Q8 / Q4) | **Yes, separately** | Tracked under `q4-gpu-weights.md`; orthogonal to batching |

The two best-practice patterns we DO adopt:

1. **Static batched prefill** with right-padding + additive attention mask.
   All sequences enter together, mask out padding columns. Composes
   cleanly with GELO mask sampling: B per-sequence `A_b` sampled
   from sub-streams of one batched-forward seed (per §3.4).
2. **EOS-padding decode**. After a sequence emits its EOS, it keeps
   contributing a dummy `<pad>` token (its KV/attention work runs but
   is discarded). Wastes some compute on finished sequences, but
   removes the need for a compaction step. The break-even vs
   compaction is at ~30 % EOS divergence; we expect <10 % at typical
   extraction prompts. **Measure before optimising.**

Continuous batching, paged KV, and speculative decoding are
explicitly out of scope for M1.11 and tracked as M1.12+ research.

---

## 3. Substrate-level changes

### 3.1 KV cache → `(B, layers, max_cache_len, kv_dim)`

`crates/gelo-embedder/src/decoder/kv_cache.rs` becomes batch-aware.
Concretely:

- `LayerKvStore::Separate { k: Array3<f32>, v: Array3<f32> }` with
  shape `(B, max_cache_len, kv_dim)`. Per-sequence `len: Vec<usize>`
  replacing the scalar `len`.
- `KvCache::new_batched(batch_size: usize, num_layers, max_cache_len, kv_dim)`.
- `append(li, batch_idx, new_k, new_v)` for the prefill phase (each
  sequence gets its own append at its own offset).
- A new `append_decode(li, new_k: ArrayView2<(B, kv_dim)>,
  new_v: ArrayView2<(B, kv_dim)>)` for the decode phase that appends
  one row per batch element at each sequence's current `len[b]`.
- `view_batched(li) -> ArrayView3<(B, max_len, kv_dim)>` returns the
  whole reserved tensor; **per-batch valid length is encoded in the
  attention mask, not by trimming the view** (otherwise we'd need
  B separate strided views, breaking the GPU upload path).

Migration: the single-sequence `KvCache::new` stays as a thin wrapper
(`new_batched(1, ...)`). All existing tests keep passing.

Gemma 4 K=V sharing: orthogonal — Shared variant becomes
`Array3<f32>` of `(B, max_len, kv_dim)`. No semantic change.

### 3.2 Attention call site

`causal_gqa_attention_cached` and friends in
`gelo-embedder/src/decoder/attention.rs:239+` gain a `_batched`
variant that takes `(B, n_q, num_heads × d_head)` Q/K/V and an
additive attention mask `(B, n_q, n_kv)` (right-padding mask + causal
mask folded together).

The per-head reshape stays — the engine surface
(`fused_attention_batched`) already takes `(B', n_q, d_head)` where
`B' = B · num_q_heads`. So the call becomes:

```rust
// Reshape (B, n_q, num_heads · d_head) → (B · num_heads, n_q, d_head)
//                      (and similar for K, V with KV-head replication
//                       for GQA).
let q3 = q.into_shape((B * num_q_heads, n_q, d_head))?;
// ... K, V similarly with KV-head replication into B' rows
let mask3 = build_combined_mask(q_pos_offsets, seq_lens, n_kv, B, num_q_heads);
engine.fused_attention_batched(q3.view(), k3.view(), v3.view(), scale, Some(mask3.view()))
```

GQA group expansion: K and V each only have `num_kv_heads` per
sequence in the KV cache. The replication to `num_q_heads` happens
when packing the engine input. This is the same `(num_heads, n,
d_head)` → `(num_heads * group, n, d_head)` reshape that
`causal_gqa_attention_with_offload` already does, just with a B
prefix.

**Why this kernel choice at decode**: at `n_q = 1`, B=16, num_heads=16,
n_kv=2048, the engine call shape is `(256, 1, 128) × (256, 1, 2048)
× (256, 2048, 128)`. burn-tensor's batched GEMM kernels (via
burn-cubecl-fusion) handle the 256-batch dimension as concurrent
workgroups — full GPU utilisation. **This is the regime where
batched decode wins.**

### 3.3 Shield-k formula under batching

Variable shield-k that drives `stacked_n` to a power of two at every
B. Formula:

```rust
fn shield_k_for_batch(b: usize, k_base: usize) -> usize {
    // k_base = 8 (paper minimum). For B ≥ k_base, this returns
    // k ∈ [k_base, 2·k_base − 1]; for small B with overlay-friendly
    // n it returns k_base + (pad amount). Always ≥ k_base.
    (b + k_base).next_power_of_two().saturating_sub(b).max(k_base)
}
```

Concrete values (k_base = 8):

| B | shield_k | stacked_n | HD₃ pow2 | Auto resolves to |
|---:|---:|---:|---:|---|
| 1 | 15 | 16 | ✓ | HD₃ |
| 8 | 8 | 16 | ✓ | HD₃ |
| 12 | 20 | 32 | ✓ | HD₃ |
| 16 | 16 | 32 | ✓ | HD₃ |
| 24 | 8 | 32 | ✓ | HD₃ |
| 32 | 32 | 64 | ✓ | HD₃ |
| 48 | 16 | 64 | ✓ | HD₃ |
| 56 | 8 | 64 | ✓ | HD₃ |
| 64 | 64 | 128 | ✓ | HD₃ |

Every B lands on HD₃ pow2. `k` floor stays at `k_base = 8` (paper
minimum); excess `k` only adds shield rows, which is monotonically
safer per the paper's shield-energy argument. Cost is bounded:
worst-case `k = 2·k_base − 1 = 15` shield rows per offload.

Lives as `gelo_protocol::sim::shield_k_for_batch(b, k_base)`. Called
from `begin_decode_pass(B)` and `begin_prefill_pass(B, n_max)` (see
§3.5); callers that want a pinned k chain `.with_fixed_shield_k(k)`.

### 3.4 Mask topology under batching

Two distinct mask shapes, governed by phase and a feature flag.

**Prefill (R / batched extraction):** per-sequence `A_b` of size
`(n_max + shield_k, n_max + shield_k)`, B masks total, derived from B
sub-streams of the batched-forward seed. Mathematically identical to
today's per-Rayon-worker model — the batching is purely
organisational, mask-apply work runs rayon-parallel across `b`. No
new AloePri argument needed; the per-candidate security
proof carries over byte-for-byte.

**Decode (D, default):** **per-sequence `A_b` of size `(1 + shield_k,
1 + shield_k)`, block-diagonal in spirit.** Same per-sequence
mask-vec data structure as prefill, just with `n_max = 1`. Existing
single-stream security argument applies per-row; no new gate
required to ship the default path.

**Decode (D, opt-in `BATCHED_DECODE_SHARED_A=1`):** one shared dense
`A` of size `(B + shield_k, B + shield_k)`, mixing B current-token
rows + shield rows. **HD₃ fires cleanly at every B per §3.3** and
the mask-apply cost is O(stacked_n · log stacked_n · hidden), one
call per offload — the headline batched-decode perf path. Gated on
AloePri `c5_batched_decode_shared_a` passing (see §7). If the gate
fails, this branch is never enabled in production; substrate keeps
the block-diagonal path as the only live default.

This split keeps the substrate simple — `SessionMask` is always
`PerSequence(Vec<MaskFamily>)` in the default — with the shared-A
case as a special `Single(MaskFamily)` variant only entered behind
the env-var gate (see §3.5).

### 3.5 SessionMask enum + lifecycle

Today's `session: Option<SessionMask>` becomes:

```rust
enum SessionKind {
    /// Non-batched OR shared-A batched decode (feature-flagged).
    /// One mask covers all rows of the stacked operand.
    Single { mask: MaskFamily, data_n: usize },
    /// Default batched mode at prefill AND default batched decode.
    /// One mask per sequence; mask-apply rayon-parallel across b.
    PerSequence { masks: Vec<MaskFamily>, data_n: usize, batch_size: usize },
}

session: Option<SessionKind>;
```

Three new bracket APIs on `InProcessTrustedExecutor`:

```rust
fn begin_prefill_pass(&mut self, batch_size: usize, n_max: usize) -> Result<()>;
fn begin_decode_pass(&mut self, batch_size: usize) -> Result<()>;
// existing begin_forward_pass(n) stays as a thin wrapper for
// non-batched callers (delegates to Single mode at batch_size = 1).
fn end_pass(&mut self) -> Result<()>;
```

`offload_linear(handle, hidden)` branches internally on session kind:

- `Single`: existing path — one mask apply over all rows of `hidden`.
- `PerSequence`: hidden must be `(batch_size, n, d_in)` reshape-able
  (the trait-level signature stays `ArrayView2` but the call site is
  expected to pass `(B*n, d_in)` with contiguous B-blocks). The
  executor rayon-iterates over b, applies `masks[b]` to slice
  `[b*n..(b+1)*n, :]`, calls one batched engine matmul, unmasks
  per-block.

The engine surface (`GpuOffloadEngine::matmul`, `matmul_many`,
`fused_attention_batched`) stays unchanged — the substrate stacks B
masked operands into one contiguous tensor before dispatching, so
the engine sees a single `(B*(n+k), d_in)` GEMM input.

Lifecycle in `generate_batched`:

```
begin_prefill_pass(B, n_max)
  forward::run_prefill_batched(...)  // multiple offload_* under PerSequence
end_pass()

loop {
    begin_decode_pass(B)              // PerSequence default, Single under flag
    forward::run_decode_step_batched(...)
    end_pass()
    sample + EOS check
}
```

---

## 4. Rerank-first rollout (Phases R1–R3)

**Why rerank first.** `CausalDiscriminatorRerankService::rerank` is
**pure prefill** — no autoregressive decode, just one forward per
(query, doc) pair followed by a 2-element yes/no projection. Batched
prefill drops cleanly in as `forward::run_batched(input_ids: &[Vec<u32>])`.
No KV-cache batching needed. No EOS handling. No sampling parity.

The existing Rayon fan-out at `score_candidates_parallel` is the
seam: replace the parallel iterator with a single batched call.

### R1 — Batched forward primitive (`forward::run_batched`)

New entry point in `gelo-embedder/src/decoder/forward.rs`:

```rust
pub fn run_batched(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[Vec<u32>],          // B sequences, may differ in length
) -> Result<(Array3<f32>, Vec<usize>)> { /* (B, n_max, hidden), valid_lens */ }
```

Right-pads each sequence to `n_max = max(input_ids[b].len())`. Builds
an additive attention mask `(B, n_max, n_max)` where padding columns
get `-cfg.causal_mask_neg` (matches the F1+ resolution from
`m1-10-security-review.md`). Otherwise the forward is structurally
identical to `forward::run` — `embedding_lookup` becomes per-batch,
`decoder_block` calls the new batched attention kernel, RMSNorm and
residuals all run per-row (no cross-row dependency).

One `begin_forward_pass(n_max)` bracket covers the whole batched
forward. Shield + mask are sampled once for `stacked_n = B + shield_k`.

Kernel choice inside `decoder_block_batched`: routed through
`engine.fused_attention_batched`, which **shape-keys internally**
between cubek-attention `Strategy::Unit` and the burn-tensor chain
(see §3 of the [`2026-05-21-attn-offload-spike`](../archive/handoffs/2026-05-21-attn-offload-spike.md)
handoff for why the kernels are non-substitutable). The provisional
threshold is `n_q ≥ N_CUBEK_MIN`, default 32; **R3 must measure the
real crossover** — at B=16, num_heads=14 (Qwen3-Reranker-0.6B), the
burn chain may stay competitive past n_q=256 because its GEMM kernels
amortise across `B · num_heads` workgroups. The threshold is a
`WgpuVulkanEngine::with_cubek_min_n_q(usize)` builder knob so R3 can
bisect without code edits.

Fallback: if `cubek_attention::launch::launch` returns Err (JIT compile
failure, shape unsupported, etc.), engine falls back to the burn chain
with a single `tracing::warn!` per shape — never a silent fall-through.

### R2 — Reranker uses it

`CausalDiscriminatorRerankService::rerank` replaces the Rayon fan-out
with one call:

```rust
let prompts: Vec<Vec<u32>> = candidates.iter()
    .map(|c| build_prompt(query, &c.text))
    .collect();
let (hidden_b, lens) = forward::run_batched(&cfg, &weights, &rope, &mut exec, &prompts)?;
let scores: Vec<f32> = (0..prompts.len())
    .map(|b| yesno_score(hidden_b.slice(s![b, lens[b]-1, ..]).view(), &weights, &head))
    .collect();
```

The Rayon path is **deprecated, not deleted** — marked `#[deprecated]`,
left wired up for migration confidence and the `candidates.len() == 1`
fast path. All new code uses `run_batched`.

### R3 — Validation gates

**Functional:**
- Parity: for any candidate set, sequential `score_pair` × N must
  produce the same scalar scores as `run_batched + per-b yesno` **to
  f32 floor**. Per-sequence mask topology (§3.4) preserves byte-level
  determinism because each A_b is derived from the same sub-stream
  the Rayon worker would have used; numerical noise only enters via
  GEMM order. Argmax may flip at tied logits — acceptance test asserts
  on the final reranked-id list, not on score bit-equality.
- Tokenizer round-trip: `build_prompt` then back-tokenize must hit
  identical input_ids vs the single-pair path.
- The existing `causal_discriminator_parity` test extends with a
  batched variant.

**Performance:**
- `crates/gelo-reranker/tests/comparative_bench.rs` gains a
  `batched_vs_rayon` variant. Target: ≥ 4× wall-time at B=16 on
  Qwen3-Reranker-0.6B (extrapolating from current Rayon-per-candidate
  measurements — see [[private_reranking_round_2]] for the ~155 ms/pair
  baseline). Bench also bisects `with_cubek_min_n_q` over `{32, 64,
  128, 256, 512}` to settle the §3.2 threshold.

**Security:**
- AloePri attack-suite condition `c4_batched_rerank` — re-runs §08
  attacks (HNM vocab-matching, JADE, JD, Gram-error) at **B = 16** only
  (spot-check methodology — full sweep escalates only on flagged
  attacks). The prefill mask topology is per-sequence A_b, so the
  per-candidate security proof is structurally unchanged from
  today's per-Rayon-worker model. c4 is paranoia gate, not a re-derivation.
- See [[aloepri_hd3_gate_phase_a_b.md]] for the C3 precedent shape.

Effort estimate: **R1 ~4 days** (substrate SessionKind refactor adds
~1 day to the original 3), **R2 ~1 day**, **R3 ~2 days** (security
spot-check is fast at one B). **Total: ~1.5 weeks** for the rerank path.

---

## 5. Decoder rollout (Phases D1–D3)

After the rerank path lands and the AloePri gate is clean.

### D1 — Batched KV cache + generation loop

`KvCache` migration per §3.1. New `generate_batched`:

```rust
pub fn generate_batched(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    prompts: &[Vec<u32>],
    gen_cfg: &GenerationConfig,
) -> Result<Vec<GenerationOutput>>;
```

Inside:

1. **Batched prefill** via `forward::run_batched` (R1 primitive). Each
   sequence's last-non-pad row becomes its first `h_last`.
2. **Decode loop**:
   - At each step, run `forward::run_decode_step_batched(token_ids: &[u32], kv_cache)`.
   - `token_ids[b]` is the most-recent sampled token for sequence b
     (`<pad>` for sequences that have already emitted EOS).
   - Per-batch position offset comes from `kv_cache.len(b)`.
   - The combined mask is right-padding-only (causal mask is a no-op
     at `n_q = 1, q_pos_offset = len(b) − 1`); for padding columns
     beyond `len(b)` apply `-causal_mask_neg`.
3. **Sampling + EOS tracking**: per-sequence argmax (or whatever
   `SamplerConfig`), record EOS hit, freeze that sequence's output
   afterward. The forward keeps running on all B until either all
   sequences hit EOS or any sequence hits `max_tokens`.
4. **Compaction** (deferred): once >30% of sequences are EOS,
   compact the KV cache to drop padded sequences. Skip for v1 — the
   overhead is small at the expected EOS divergence and the
   complexity isn't justified yet.

### D2 — Plumb through `DecoderRuntime`

`gelo-snp-runner::DecoderRuntime` gains
`generate_extraction_batched(prompts: &[String])` that does the
chat-template wrap + tokenise + `generate_batched` + post-strip per
b. The single-prompt path delegates: `generate_extraction(p)` becomes
`generate_extraction_batched(&[p])[0]`.

This unblocks the **cross-chunk scheduler** for the extraction bench:
gather all chunks' prompts up-front, hand them to
`generate_extraction_batched` as one call. Per the [[private_reranking_round_2]]
and existing extraction performance numbers, the v7 fixture has 7
chunks. Batching the 7 chunks' extraction passes turns ~40 min
sequential wall into something we can actually iterate on.

### D3 — Validation gates

**Functional:**
- Per-sequence determinism: `generate_batched(&[p])[0]` is **NOT**
  byte-identical to `generate(&p)` because mask shapes and RNG
  consumption differ. Contract: per-sequence outputs match
  semantically (entity-count parity on the extraction bench;
  ranked-id parity on rerank) to **f32 floor**. Argmax may flip at
  tied logits; the test asserts on `extract_entity_set` equivalence,
  not on token-id sequences.
- N-of-N parity: `generate_batched(&prompts)` produces semantically-
  equivalent output per b as `generate(prompts[b])` independently.
- Existing greedy-determinism tests in
  `gelo-embedder/tests/qwen3_generation_*` extend with batched-B
  variants gated on the per-sequence default (the
  `BATCHED_DECODE_SHARED_A=1` opt-in path has its own divergent
  fixtures).

**Performance:**
- Crossover measurement (the missing data point): bench
  `generate_batched` at B ∈ {1, 2, 4, 8, 16, 24, 32, 48, 56, 64} on
  the v7 fixture's 7 chunks, **default path (per-sequence A_b at
  decode)**. Expected: 3–6× wall reduction at B=7-8 from GPU dispatch
  amortisation alone; the bigger 6–12× wall headline requires the
  shared-A path post-c5 gate.
- `tee:attn_cached_batched` instrumentation: confirm per-call cost
  drops as B grows. **This is the primary metric** — the bucket that
  was 39% at B=1 should fall to <10% at B=16 even under per-sequence
  masks (the win comes from GPU dispatch amortisation, not from
  mask sharing).
- Side bench: with `BATCHED_DECODE_SHARED_A=1`, measure the headline
  shared-A win. Only ship this number alongside the c5 gate result.

**Security:**
- Run AloePri `c5_batched_decode_shared_a` suite at **B = 8** only
  (spot-check). Gate: flipping `BATCHED_DECODE_SHARED_A=1` default-on.
  Methodology mirrors C3 (see [[aloepri_hd3_gate_phase_a_b.md]]).
- Per-sequence default needs no new gate — the per-row security
  argument carries over from today's single-stream proof.
- The Phase 1b (`PHASE_1B_DECODE_AMULET=1`) path — currently a 2.6×
  regression at B=1 — should hit crossover at B ≈ 12. Confirm
  measurement before re-flipping the env var's default. Phase 1b
  composes with batched-decode but each combination needs its own
  measurement.

Effort estimate: **D1 ~5 days** (KV cache migration is the largest
piece), **D2 ~2 days**, **D3 ~3 days**. **Total: ~2 weeks** for the
decoder path. Combined with R: **~3 weeks** end-to-end.

---

## 6. Performance model (extrapolated; needs measurement)

### Rerank (post-R)

Single-pair baseline (from `comparative_bench`, Qwen3-Reranker-0.6B,
post-Tier-2 on AOCL-BLIS): ~155 ms/pair.

| Config | Pairs | Wall (extrapolated) | Speedup vs sequential |
|---|---:|---:|---:|
| Sequential | 16 | 16 × 155 ms = 2.48 s | 1.0× |
| Rayon (today) | 16 | ~620 ms (4 CPU cores) | 4.0× |
| **Batched (R)** | 16 | **~280 ms** | **8.9×** |

Win mechanism: mask amortisation (16× less mask round-trip work) +
single GPU dispatch chain per layer + cubek-attention firing at
n_prompt ≥ 32 (rerank prompts are typically 300–500 tokens).

### Decoder generate (post-D)

Single-stream baseline (v7 fixture, BENCH_MAX_CHUNKS=1, post-shield
SIMD): ~343 s wall, 89 s `tee:attn_cached`. Extrapolated 7-chunk
sequential: 7 × ~343 s = 2 401 s.

**Measured D2 (2026-05-27, commit `d241a7a`, post-DCT-IV-cascade,
v7 fixture, Qwen3-4B, BENCH_EXTRACTION_BATCH_SIZE=8 — one batch of
B=7):**

| Config | Sequences (chunks) | Wall |
|---|---:|---:|
| Sequential extrapolation (pre-rewire 2026-05-21) | 7 chunks | ~2 401 s (extrapolated) |
| Batched, B=7, default per-sequence A_b | 7 chunks | §6 projected ~700-900 s |
| **Batched, B=7, default per-sequence A_b** | **7 chunks** | **578.83 s (measured)** ← **4.15× vs 2 401 s extrap** |
| Batched, B=7, shared-A (post-c5, not shipped) | 7 chunks | §6 projected ~480 s |

The measured D2 wall **beats the per-seq-A_b projection** (700-900 s)
by 14-28 % at the v7 fixture shape. The 5× headline target (~480 s)
was contingent on the c5-gated shared-A path, which D2 does not ship
(default is per-sequence A_b).

Bucket mix at v7 shape (post-D2, from
`bench-results/d2-extract-bench-2026-05-27_11-55-00.log`):

| Bucket | % of forward profile wall |
|---|---:|
| engine:registered_linear (GPU matmul) | 55.2 |
| tee:attn_cached_inplace_many (in-TEE attention) | 26.1 |
| tee:compute_logits | 5.3 |
| gelo:mask_unapply:hd3 | 5.0 |
| gelo:shield_stack | 4.3 |
| gelo:mask_apply:hd3 | 2.2 |

GPU matmul is the dominant bucket at v7's short-n (~750-prompt) shape
— different from the m1-12 sweep cells (long-n n=2048-2400) where
mask is co-dominant. Auto picks HD₃ at pad ratio 1.34 (s=765 →
pow2=1024).

Per-sequence A_b at decode (default): the win is from GPU dispatch
amortisation alone (one batched matmul per layer covers B sequences).
Mask cost grows linearly with B but per-sequence-cost stays constant.

Shared dense A at decode (opt-in, behind c5 gate): adds mask
amortisation — one HD₃ apply per offload covers all B sequences. The
headline ~5× wall reduction requires this path.

Crossover for in-TEE vs GPU attention (the load-bearing assumption):
- in-TEE per-step: ~2 ms × B (linear)
- GPU chain per-step: ~22 ms + ε × B (launch-dominated until B ~16)
- Crossover: B ≈ 11–16. **Must be measured at D1 before flipping the
  Phase 1b default.**

---

## 7. Security checklist

| Concern | Mitigation | Where it lands |
|---|---|---|
| Per-sequence A_b at prefill/decode default | Per-row security argument carries over from today's per-Rayon-worker model; no new gate | R1 / D1 default-on |
| Shared dense A at decode (opt-in) leaks cross-row info | `BATCHED_DECODE_SHARED_A=1` ships disabled; AloePri `c5_batched_decode_shared_a` at B=8 spot-check gates default-on | D3 |
| Right-padding mask reveals per-sequence length to GPU | Padding columns get `-cfg.causal_mask_neg` (F1+ pattern); the engine never sees the mask matrix directly under in-TEE softmax. Padding length is also encoded in the engine's view via the (B, n_max, n_kv) shape but is not deduplicated; this is acceptable per the existing F1+ argument. | R1 (combined-mask construction) |
| EOS-padding wastes GPU work on finished sequences | Performance concern, not security — but if EOS arrival times leak via timing, mask by inserting decoy tokens. Defer until measured. | D3 follow-up |
| Phase 1b GPU softmax at decode + batching | Phase 1b's `c5_perm_attn` and the new `c5_batched_decode_shared_a` are independent — both must clear before the combined config defaults on. | aloepri_hd3_gate_phase_a_b precedent |

**F1+ extension.** `docs/plans/m1-10-security-review.md` F1+ resolution
(in-TEE softmax under causal mask, soft `-30` penalty) extends
naturally — the new combined mask (right-padding + causal) keeps the
same `-cfg.causal_mask_neg` treatment for both padding and
future-position blocks.

### 7.1 Fallback ladder for c5

If `c5_batched_decode_shared_a` (B=8 shared dense A) flags above
the c1–c3 baseline:

1. `BATCHED_DECODE_SHARED_A` never flips default-on. Production stays
   at per-sequence A_b at decode — already the default behaviour.
2. The shared-A code path is left in the substrate, marked
   `#[doc(hidden)]` with a comment citing the failed gate, until a
   future protocol revision addresses the leakage.
3. No further fallback levels are needed — the per-sequence default
   already exists and ships unconditionally.

The substrate development cost for shared-A is born up-front; if c5
fails, the engineering investment buys nothing immediately
deployable. **This is the asymmetric risk of the opt-in path** and
the reason the gate's pass/fail decision is load-bearing for the
performance story.

---

## 8. Resolved decisions log (from 2026-05-21 grilling session)

| # | Decision | Picked | Rationale |
|---|---|---|---|
| 1 | First target | **Rerank (R1–R3 first)** | Pure prefill, no KV cache batching; de-risks substrate before decoder migration |
| 2 | Decode mask topology | **Shared dense A of size (B+k, B+k), behind feature flag** (default = per-sequence) | HD₃ at every B (§3.3); opt-in pending c5 gate (§7.1) |
| 3 | Prefill mask topology | **Per-sequence A_b, batched engine call** | B² cost of shared A makes shared-A unworkable at prefill shapes |
| 4 | Substrate API | **Extend SessionMask → SessionKind enum; new begin_prefill/begin_decode brackets** | Existing callers untouched; one executor type |
| 5 | Kernel routing | **Shape-keyed inside `fused_attention_batched`** | Threshold tunable via `with_cubek_min_n_q`; fallback on cubek launch error |
| 6 | EOS handling | **Pad-with-EOS** | Break-even with compaction at ~30% divergence; expected <10% |
| 7 | Padding length | **Right-pad to max(n_b)** | Simple; HD₃ does internal pow2 padding |
| 8 | shield_k formula | **Variable, `k = next_pow2(B+8) − B`** | Always ≥ 8 (paper minimum); HD₃ at every B |
| 9 | Determinism contract | **Per-sequence f32-floor parity, not byte-identical** | Bytewise would force per-sequence A_b at decode, contradicting #2 |
| 10 | AloePri gate scope | **Spot-check: c4 at B=16 prefill, c5 at B=8 decode** | Full sweep escalates only on flagged attacks |
| 11 | Legacy paths | **Deprecate-but-keep** | `#[deprecated]` hints; no migration deadline |
| 12 | D2 scope | ✅ **shipped 2026-05-27** (commit `d241a7a`) | Adaptive sub-batch dispatch in `extract_kg_from_chunks`; `ExtractionConfig.extraction_batch_size` default 8; B=1 fast-path falls back to single-prompt `generate_extraction`. v7 fixture measured: **578.83 s extract_kg wall** vs ~2 401 s pre-rewire extrapolation = **4.15×**, vs ~700-900 s §6 per-seq A_b projection beats by 14-28 %. Bench log: `bench-results/d2-extract-bench-2026-05-27_11-55-00.log`. |
| 13 | c5 failure fallback | **Defer-and-flag** | `BATCHED_DECODE_SHARED_A=1` stays disabled; substrate keeps both paths |

**Out of scope** (explicitly raised and deferred):

- Continuous batching (vLLM-style in-flight insertion). Conflicts
  with per-forward-pass mask `A`. M1.12+ research.
- PagedAttention. iGPU shares system RAM; the HBM-fragmentation
  problem doesn't exist here.
- Speculative decoding. Greedy-parity contract is load-bearing.

---

## 9. Non-goals for M1.11

- PagedAttention / block KV. iGPU shares system RAM with the host;
  the HBM-fragmentation problem PagedAttention solves doesn't exist
  here.
- KV quantisation. Tracked under [`q4-gpu-weights.md`](q4-gpu-weights.md);
  composes with batching but is separate work.
- Speculative decoding. Greedy parity is load-bearing for the
  extraction bench's determinism; speculative would break that
  contract without significant additional gating.
- bf16 mask GEMM. Per [[bf16_mask_gemm_skipped]] the win is ~10 %
  TTFT, dwarfed by what batching unlocks. Deferred.
- `tee_matmul_bf16` direct path optimisation (m=1 GEMV). Per
  [[tee_direct_m1_gemv_slowness]], blocks paper §3.2 sensitive-layer
  exclusion as a perf-positive default; orthogonal to batching.

---

## 10. Reproducing the baseline before R1 starts

```bash
# Single-pair rerank (today's baseline for R3 comparison)
cargo test -p gelo-reranker --release --test comparative_bench \
  -- --nocapture causal_discriminator

# Single-stream extraction (today's baseline for D3 comparison)
BENCH_MAX_CHUNKS=1 cargo run -p gelo-snp-runner --release \
  --example extract_and_query_bench

# cubek-attention microbench (validates the prefill kernel choice)
cargo bench -p gelo-gpu-wgpu --bench amulet_attention
```

The numbers in §6 are extrapolated from these; R3/D3 will replace
them with measured values.
