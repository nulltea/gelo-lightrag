# M1.10 — Fused Permuted Attention

> **Worktree:** original (this one).
> **Parent plan:** [`path-1-gelo-gemma.md`](path-1-gelo-gemma.md) §M1.10.
> **Status:** scaffold landed (substrate trait seam + composed
> default-impl). Engine kernel pending. **Causal-mask leak in the
> existing `permuted_attention` must be resolved before this lands
> in the cached generation path.**

---

## 0. Status (2026-05-18)

| Piece | Where | State |
|---|---|---|
| Trait seam `GpuOffloadEngine::fused_attention_batched` | `crates/gelo-protocol/src/substrate.rs:160-219` | landed · default impl composes 3 dispatches (matmul · scale+mask · softmax · matmul); regression test in `fused_attention_tests` |
| TEE-side protocol wrapper `permuted_attention` (Amulet softmax-equivariance + Hidden-No-More noise) | `crates/gelo-protocol/src/attention.rs:99-197` | landed · drives the embedder/reranker "Tier 1" path · uses 3 separate engine calls (`matmul_dynamic_batched` × 2 + `softmax_batched`) |
| Embedder dispatch into permuted path | `crates/gelo-embedder/src/decoder/forward.rs:491-498` (`decoder_block`) | landed · engages when `cfg.use_perm_attention && n ≥ perm_attention_min_seq_len` |
| Generation dispatch into permuted path | `crates/gelo-embedder/src/decoder/forward.rs:355-370` (`decoder_block_cached`) | **NOT WIRED** — global attention stays in-TEE at all `n` per the locked M1.3 design decision; M1.10 lifts this |
| Fused engine kernel (the actual win) | `crates/gelo-gpu-wgpu/src/lib.rs` | **PENDING** — no override of `fused_attention_batched`; runs composed default at 3-dispatch wall-clock |
| Auto-switch threshold for fused path | `crates/gelo-embedder/src/decoder/config.rs::perm_attention_threshold` | already exists (`= 64`); needs re-tuning with kernel measurements |

## 1. Why this is the structural answer for long context

Measured 2026-05-18 (`crates/gelo-gpu-wgpu/tests/qwen3_long_context_bench.rs`,
Qwen3-1.7B, AMD RADV GFX1151 iGPU):

| n_prompt | gpu_plain TTFT | gpu_gelo TTFT | gpu_gelo overhead |
|---:|---:|---:|---:|
| 64   | 231 ms | 372 ms | +44.5 % |
| 512  | 4 064 ms | 7 074 ms | +55.9 % |
| 2048 | 9 344 ms | **75 325 ms** | **+480.2 %** |

**The +480 % overhead at n=2048 is not attention compute** — both
branches run attention in-TEE on CPU at that shape. It is the
**GELO mask round-trip on the four linear-offload batches per layer
× 28 layers**: ~7.6 TFLOPs of CPU BLIS matmul (`A·h_padded` apply,
`Aᵀ·masked_out` unapply, both shape `(n+k, n+k) × (n+k, d)`).

The bench also shows that gpu_plain attention compute (running
in-TEE in both branches at this `n`) is ~7 s of the 9.3 s baseline —
it is the **next** bottleneck once the mask round-trip is dealt with.

M1.10 attacks both problems at once. It moves global attention onto
the GPU under a permutation cover (Amulet softmax-equivariance +
Hidden-No-More noise), so:

1. Attention compute moves from CPU BLIS to GPU at iGPU throughput
   (~10-20 TFLOPs/sec) — directly cuts the in-TEE attention cost
   that dominates `gpu_plain` at long n.
2. The four linear projections per layer are unaffected; the
   per-layer GELO mask round-trip remains. (Reducing **that** cost
   is a separate concern — see §10.)
3. The fused kernel avoids materializing the `(B, n_q, n_kv)` score
   tensor in HBM (`gelo-llm.html` §04 lede: ~3.2 GB/layer at n=4k on
   the composed 3-dispatch path → ~130 MB/layer fused).

## 2. Protocol primitive — disambiguate

Three privacy primitives appear in the docs; **M1.10 is option (c)**:

| Primitive | Hides | Where in code | When used |
|---|---|---|---|
| (a) **GELO mask** `A` on activations | `H` from GPU on linear-offload boundaries | `gelo-protocol::sim::offload_linear*` | every QKV / O / gate-up / FfnDown projection |
| (b) **OutAttnMult** 4-partition (TwinShield Xue '25 §V-A) | `Q` and `K` from GPU on `Q·Kᵀ` | `gelo-protocol::out_attn_mult` | embedder/reranker attention when `out_attn_mult_min_seq_len ≤ n` |
| (c) **Permuted attention** (Amulet softmax-equivariance + Hidden-No-More) | `Q`, `K`, `V` row-order on full softmax(QKᵀ/√d)·V | `gelo-protocol::attention::permuted_attention` | **embedder** dispatch only (today); **M1.10 wires it into cached generation + replaces the 3-dispatch impl with a fused FlashAttention-style kernel** |

(a) and (c) compose: (a) protects the linear projections, (c)
protects the attention subblock. They use independent privacy
arguments and independent state (mask `A` lives in the executor;
permutation `π` is sampled inside `permuted_attention` per call).
**Mask `A` never crosses PCIe** — that's the property that keeps the
on-GPU unmask idea unsafe (see `inference-optimization.md` §2.2.1).
Permutation `π` is also kept on the TEE side: the GPU sees
`π·Q + η_Q`, `π·K + η_K`, `π·V` and the raw permuted score tensor,
never `π` itself in the clear.

## 3. ⚠ Open security issue — causal mask leaks π

**Before M1.10 ships into the cached generation path, this must be
fixed.**

The current `permuted_attention` adds the permuted causal mask
`M_π[i,j] = -inf if perm[j] > perm[i] else 0` to the score tensor
**before** invoking `softmax_batched` on the GPU (see
`attention.rs:161-180` + `attention.rs:183`). The GPU therefore
sees the `(n, n)` score-plus-mask tensor with `-inf` exactly at the
blocked positions of the permuted causal pattern.

**The leak.** Counting `-inf` entries in row `i` gives
`n - 1 - perm[i]` exactly, so the GPU recovers `π` row-by-row from
a single softmax_batched call. With `π` known, the row-permuted Q,
K, V come back into canonical order (modulo the per-element
Gaussian noise η at σ = 0.01, which is much smaller than activation
magnitudes and provides little protection on its own). The
Hidden-No-More mitigation degrades to its no-permutation baseline,
and standard activation-inversion attacks become tractable.

**Why this hasn't been a problem yet.** The current dispatch into
`permuted_attention` is from the embedder/reranker (`decoder_block`,
not `decoder_block_cached`), where BGE-base and Qwen3-Embedding-0.6B
both run BIDIRECTIONAL attention (`AttentionMask::None`). The leak
only fires when `AttentionMask::Causal` is used, which today is
exercised only by the per-batch synthetic tests
(`crates/gelo-protocol/tests/permutation_attention.rs`).

**Wiring `permuted_attention` into the cached generation path
unconditionally activates this leak.** M1.10 cannot land without
addressing it.

### Candidate fixes (decision deferred to Phase 0 of implementation)

| Approach | Idea | Cost | Risk |
|---|---|---|---|
| **F1. In-TEE softmax** | Bring scores back to TEE after `Q·Kᵀ`, apply mask + softmax on CPU, send `probs` back to GPU for `probs·V`. Eliminates the score-tensor exposure to GPU. | +1 PCIe round-trip per layer with an `(n_q, n_kv)` tensor — at n=2048 that's 16 MB/head × 16 heads = 256 MB per layer. At 28 layers = 7 GB of round-trip per forward. Borderline; needs benchmarking. | Low — pure protocol change, math unchanged |
| **F2. Per-row scaled mask** | Replace `-inf` with a large but finite negative value `-C` (e.g. `-1e4 · max\|score\|` per row) so all rows have the same finite range. Then count-of-large-magnitudes per row doesn't trivially reveal `perm[i]`. | Cheap. | **Doesn't work** — softmax still has perfect numerical resolution to identify which positions were blocked vs which weren't, even at finite `-C`; the count attack still works on per-row patterns of "above threshold" vs "below". Reject. |
| **F3. Permute the mask pattern** | Sample a second independent permutation `π_M ≠ π` and apply it to the causal mask too, sending the mask through to GPU under `π_M`. | Cheap. | Breaks the equivariance identity unless `π_M = π` — wrong math. Reject. |
| **F4. Causal-aware fused kernel without explicit mask** | The fused kernel takes `q`, `k`, `v` and computes attention with implicit causal masking on the **permuted** axis (i.e., kernel knows positions are causally ordered after permutation). Requires the GPU to know `π`. | — | **Breaks the protocol.** Same problem as on-GPU unmask: GPU with `π` recovers Q/K/V order, defeating Amulet. Reject. |
| **F5. Sample-dependent mask noise** | Add fresh Gaussian noise to mask values per row, calibrated to overwhelm the discrete `-inf` vs `0` distinction below softmax saturation. | Cheap; small numerical impact on attention output. | **Investigate.** Likely insufficient — softmax has 24-bit mantissa, very large noise needed to hide block pattern. |
| **F6. Block-randomised masking** | Replace per-position causal blocking with block-randomised blocking inside the permutation — fewer distinct "block counts" per row, no exact `perm[i]` recovery. Privacy is bounded above by the block size. | Moderate impl effort. | Privacy weakening — needs separate security analysis. Filed under Tier 5 / future-rnd. Not v1. |

**Phase-0 task for M1.10:** complete a written security review of
options F1–F6 (and any further alternatives discovered during
analysis), pick a fix, validate it preserves the Amulet equivariance
identity, and add a regression test that asserts the engine cannot
recover `π` from a single forward pass.

**Decision (2026-05-18, post security review): F1+ ✓ — in-TEE
softmax with a soft-saturating causal mask** (`-C` at blocked
positions, `C ≈ 30`). Combines F1's mask-never-leaves-TEE property
with a fix for the residual zero-pattern leak (softmax(`-∞`)
produces exact zeros that still count-attack into π; softmax of
`-30` is ~1e-13, indistinguishable from f32 noise floor).

Full survey and security argument in
[`m1-10-security-review.md`](m1-10-security-review.md). Headline
finding: **none of the public TEE-GPU split-inference schemes
solves this gap for single-server commodity-GPU.** Amulet, KV-
Shield don't address causal masks; TwinShield (Liu '25) uses HE-
softmax we don't need; Cascade needs multi-party. F1+ appears to
be a novel construction for this threat model.

Estimated cost at n=2048 on Qwen3-1.7B / RADV iGPU: ~40 ms per
layer × 28 layers = ~1.1 s for the attention slice (vs ~7 s
in-TEE-attention today). Score-tensor PCIe round-trip is the
overhead; ~64 MB per layer per direction. **~6× speedup on
attention** before the M1.10 fused-output-matmul kernel (F7 in
the security review) lands.

## 4. Architecture (already 80 % in place)

```
┌─────────────────────────────────────────────────────────────────┐
│ decoder::forward::decoder_block_cached  (TEE)                   │
│                                                                 │
│   if n_q + n_kv ≥ FUSED_THRESHOLD && layer is Global:           │
│       → permuted_attention(exec, q, k, v, scale, Causal, cfg) ──┼─┐
│   else:                                                         │ │
│       → causal_gqa_attention_cached(..., q_pos_offset)          │ │
└─────────────────────────────────────────────────────────────────┘ │
                                                                    │
┌───────────────────────────────────────────────────────────────────┘
│ gelo-protocol::attention::permuted_attention  (TEE)
│
│   sample π_b ∈ S_n
│   q_perm = π·q;  k_perm = π·k;  v_perm = π·v   (+ optional N(0,σ²))
│   ┌─────────────────────────────────────────────────────────────┐
│   │ engine.fused_attention_batched(                             │
│   │     q_perm, k_perm, v_perm, scale, mask_π                   │
│   │ ) -> attn_perm  (B, n_q, d_head)                            │
│   └────────────────────────────────────┬────────────────────────┘
│                                        │
│   attn = π⁻¹ · attn_perm  ←─────────── ┘
│
└─ engine: GpuOffloadEngine
      │
      ├── default impl (substrate.rs:160-219, today)
      │    └─ 3 dispatches: matmul_dynamic_batched + softmax_batched + matmul_dynamic_batched
      │
      └── override (WgpuVulkanEngine, M1.10 to add)
           └─ 1 dispatch: FlashAttention-style fused kernel
```

What M1.10 changes:

1. **`fused_attention_batched` override** in `gelo-gpu-wgpu` — the
   actual kernel (Option A/B/C in §6).
2. **`permuted_attention` rewires** from `matmul/softmax/matmul`
   triplet to a single `fused_attention_batched` call — once the
   §3 causal-mask leak is resolved.
3. **`decoder_block_cached` dispatch** consults the auto-switch
   threshold and routes to `permuted_attention` (cached variant)
   when `n_q + n_kv` clears the threshold.

The shape contract is in `substrate.rs:160-219` and is already
locked by `fused_attention_tests` — fused override must produce the
same result as composed default within `1e-4`.

## 5. Phases & milestones

### Phase 0 — Causal-mask leak resolution (1 week)

**Mandatory before any kernel work.**

- M1.10.0.1 ✓ Security analysis written: [`m1-10-security-review.md`](m1-10-security-review.md)
  — surveyed Amulet, KV-Shield, TwinShield Liu '25, SCX, Hidden
  No More, PermLLM, Cascade. **Recommendation: F1+ (in-TEE softmax
  with soft causal mask, `C = 30`).** None of the surveyed papers
  solves this gap for our threat model — F1+ appears novel.
- M1.10.0.2 Implement F1+ in `attention.rs::permuted_attention`:
  move softmax to TEE, replace `-∞` mask with `-C` mask, add
  `causal_mask_neg: f32` to `PermAttnConfig` (default 30.0).
- M1.10.0.3 Add regression test in `tests/permutation_attention.rs`:
  `engine_cannot_recover_pi_from_single_forward` — SpyEngine
  captures `(matmul_dynamic_batched, softmax_batched)` inputs;
  attempt three recovery attacks (count-exact-zeros,
  count-below-1e-12, sort-by-row-magnitude); assert Spearman(π̂, π)
  < 0.1 across n ∈ {64, 256, 1024} over 1000 trials each.
- M1.10.0.4 Re-run existing parity tests at `AttentionMask::Causal`
  with `C = 30` — must match in-TEE reference within 1e-4.
- M1.10.0.5 Empirical σ re-tuning at Qwen3-1.7B activation
  magnitudes (Hidden No More uses GPT-2-class shapes; ours
  differ). Adjust `PermAttnConfig::HIDDEN_NO_MORE` default if
  σ=0.01 is insufficient.

### Phase 1 — Generation cached path wiring (~3 days)

- M1.10.1.1 Add `permuted_attention_cached(q_new, k_cached, v_cached,
  q_pos_offset, ...)` — asymmetric Q-vs-KV variant for decode
  shapes. Reuses `permuted_attention` internally; just handles the
  Q-positions-offset within the permuted causal mask.
- M1.10.1.2 Wire `decoder_block_cached` to dispatch to
  `permuted_attention_cached` for Global layers when
  `cfg.use_perm_attention && (n_q + n_kv) ≥ threshold`.
- M1.10.1.3 Parity tests:
  `permuted_attention_cached_matches_causal_gqa_attention_cached` at
  `n_q ∈ {1, 32}`, `n_kv ∈ {64, 1024, 2048}`.
- M1.10.1.4 Re-run `qwen3_generation_e2e.rs` token-parity test with
  `use_perm_attention = true` enabled — expect bit-identical greedy
  tokens vs the in-TEE path.

This phase lands the dispatcher and validates correctness while the
fused-kernel work in Phase 2 proceeds in parallel.

### Phase 2 — ~~Engine kernel~~ **DEPRECATED 2026-05-18 post-F1+**

> **Structural conflict with F1+.** Phase 2's `fused_attention_batched`
> override takes a mask tensor and runs softmax internally on the GPU.
> Under F1+ the causal mask **must not** be sent to the GPU and softmax
> **must** run in-TEE — otherwise the score-input pattern reconstructs
> π. Any fused-flash kernel that conforms to the existing trait
> signature re-introduces exactly the leak F1+ closes. The four
> candidate work-arounds (drop the mask argument and let the kernel
> infer causality from `q_pos_offset` / pre-noise scores inside the
> kernel / HE-mask under encrypted softmax / pattern-invariant mask
> shapes) are all research-level — they either weaken the threat
> model, require crypto we don't carry, or change model semantics.
>
> Under F1+ the GPU-side dispatch is **already optimal for our threat
> model**: two `matmul_dynamic_batched` calls with in-TEE softmax
> between them. There is no further "fusion" available without
> giving back the security argument.
>
> Any future engine-side perf work on the M1.10 path lives outside
> this milestone — likely re-scoped as "matmul kernel tuning"
> (better cubek autotune entries for the shapes we actually
> dispatch) rather than "fused attention." That re-scoping is
> separate from M1.10 and tracked when the long-context bench
> surfaces a matmul-perf bottleneck post-F1+.

The original Option A / B / C analysis in §6 below is preserved for
historical reference but no longer load-bearing.

### Phase 3 — Auto-switch threshold tuning (~2 days)

- M1.10.3.1 Sweep `perm_attention_threshold ∈ {128, 256, 512, 1024,
  2048}` on Qwen3-1.7B prefill across n ∈ {64, 256, 512, 1024, 2048,
  4096}. Pick the crossover where fused wall-clock beats in-TEE
  wall-clock.
- M1.10.3.2 Update `DecoderConfig::perm_attention_min_seq_len`
  default in `config.rs:142`.

### Phase 4 — Long-context bench validation (~1 day)

- M1.10.4.1 Extend `qwen3_long_context_bench.rs` to include a
  `gpu_gelo_fused` cell that opts into `use_perm_attention = true`
  with auto-switch.
- M1.10.4.2 Acceptance gate (§7).

## 6. ~~Engine kernel — three options~~ (historical, Phase 2 deprecated)

Same A/B/C breakdown as `path-1-gelo-gemma.md` §M1.10, refreshed:

### Option A — Cubek-direct (preferred if API is stable)

Direct `cubek::attention::launch::launch_ref` call from
`gelo-gpu-wgpu` with `causal: false` and our permuted causal mask
passed via the `Materialized` mask slot.

- **Pros:** ~150 LOC; reuses the well-tuned cubek-attention WGSL
  kernel; no custom shader maintenance.
- **Cons:** `cubek-attention` v0.1.1 was young at last check (Apr
  2026); API may have moved. **Phase 2 must start with a 1-day
  API-stability check** before committing.
- **Risk:** medium. Mitigated by Option B fallback.

### Option B — Upstream PR to burn-cubecl

Submit a PR to parameterise
`burn_cubecl::kernel::attention::flash_attention(causal: bool)` —
today it's hardcoded `causal: true`. Then we drive it from
`gelo-gpu-wgpu` via the standard burn-cubecl entry.

- **Pros:** lowest long-term maintenance; aligned with the rest of
  the burn-cubecl integration.
- **Cons:** **unbounded on tracel-ai merge cycle.** Cannot be relied
  on for v1 timing.

### Option C — Custom WGSL (fallback only)

Hand-rolled FlashAttention-style fused kernel in WGSL, ~500 LOC,
FLASH-D online softmax pattern. Engine override calls this directly.

- **Pros:** zero dependency on cubek/burn API stability; full
  control of the mask shape.
- **Cons:** highest implementation risk (numerical correctness,
  performance tuning, autotune entries per shape).
- **When pursued:** only if A and B are both unworkable at start of
  Phase 2.

**Default ordering:** start Phase 2 with a 1-day Option-A spike.
If cubek's API is stable, go A. If not, file the Option-B PR (no
blocker — it's just a tracking item) and execute Option C as the
critical path. The 3-week effort estimate assumes Option A; +2
weeks if forced to Option C.

## 7. Acceptance

### Performance

| Metric | n=2048 prefill TTFT (Qwen3-1.7B, RADV iGPU) | Source |
|---|---|---|
| gpu_plain (in-TEE attention) | 9 344 ms | observed 2026-05-18 |
| gpu_gelo (in-TEE attention) | 75 325 ms | observed 2026-05-18 |
| **gpu_gelo_fused (M1.10 target)** | **≤ 20 000 ms** | acceptance gate |

The gate's looseness reflects two unknowns: (i) how much of the
75 s comes from in-TEE attention compute (probably ~7 s, since
plain-vs-gelo delta is dominated by mask round-trip, not attention)
versus the mask round-trip (~66 s, untouched by M1.10), and
(ii) iGPU throughput on the fused kernel at this shape. The mask
round-trip remains the bigger gap after M1.10; closing it is the
next-tier work in §10.

A weaker but more directly attributable gate is on the
**attention-only** wall-clock at n=2048 in isolation: in-TEE BLIS
attention on the 28-layer Qwen3 model takes ~7 s; fused-on-GPU
should run that compute in ≤ 1 s. Add an attention-isolated bench
to validate this independently of the linear-projection mask cost.

### Correctness

- M1.10.0.3 — engine cannot recover `π` (§3 regression).
- M1.10.1.3 — `permuted_attention_cached` matches
  `causal_gqa_attention_cached` to 1e-4 across n_q ∈ {1, 32} and
  n_kv ∈ {64, 1024, 2048}.
- M1.10.1.4 — `qwen3_generation_e2e.rs` emits bit-identical greedy
  tokens with `use_perm_attention = true` vs the existing in-TEE
  path on `"The quick brown fox"` prompt.
- M1.10.2.2 — `fused_attention_tests` passes against the engine
  override.

## 8. Security argument summary

(For the security review doc to be written in Phase 0.)

What the GPU sees on each fused-attention call:

| Input | Shape | What it reveals |
|---|---|---|
| `q_perm + η_Q` | (B, n_q, d_head) | Row-permuted noisy Q. Adversary cannot identify which permuted row corresponds to which original Q position without `π`. |
| `k_perm + η_K` | (B, n_kv, d_head) | Same as above for K. |
| `v_perm` | (B, n_kv, d_head) | Same. (V is not noised in the current impl — sensitivity analysis needed: should V be noised too?) |
| `scale` | scalar | Public (`1/√d_head`). |
| `mask` | (B, n_q, n_kv) | **The piece §3 addresses.** Must not reveal `π`. |
| `attn_perm` | (B, n_q, d_head) | Output; row-permuted result. Adversary knows it's `softmax(...)·V_perm` but the row order matches `q_perm`, so it adds no information beyond what the inputs leaked. |

Open: V-noise. Should `v_perm` also get `+ η_V`? Hidden No More is
about preventing Q/K-based inversion attacks; V plays a different
role (it's what gets averaged after softmax). The current impl
doesn't noise V. Worth confirming during Phase 0 review whether the
Amulet/HNM threat model covers V-recovery.

## 9. Files touched

| File | Phase | Change |
|---|---|---|
| `docs/plans/m1-10-fused-permuted-attention.md` | 0 | this document |
| `docs/plans/m1-10-security-review.md` | 0 | written analysis of F1–F6, picks fix |
| `crates/gelo-protocol/src/attention.rs` | 0 | causal-mask-leak fix (whichever F1–F6 wins) |
| `crates/gelo-protocol/tests/permutation_attention.rs` | 0 | `engine_cannot_recover_pi_from_single_forward` |
| `crates/gelo-protocol/src/attention.rs` | 1 | `permuted_attention_cached(q_new, k_cached, v_cached, q_pos_offset, ...)` |
| `crates/gelo-embedder/src/decoder/attention.rs` | 1 | `causal_gqa_attention_permuted_cached` wrapper (Qwen3-shape dispatch) |
| `crates/gelo-embedder/src/decoder/forward.rs:355-370` | 1 | dispatch into permuted-cached path on Global layers past threshold |
| `crates/gelo-embedder/tests/qwen3_generation_e2e.rs` | 1 | add masked-with-perm cell; assert token parity |
| `crates/gelo-gpu-wgpu/src/lib.rs` | 2 | `fused_attention_batched` override (Option A/B/C) |
| `crates/gelo-gpu-wgpu/Cargo.toml` | 2 | (Option A) add `cubek-attention` dep |
| `crates/gelo-gpu-wgpu/src/kernels/flash.wgsl` | 2 | (Option C only) custom WGSL kernel |
| `crates/gelo-embedder/src/decoder/config.rs:142` | 3 | re-tune `perm_attention_min_seq_len` default after sweep |
| `crates/gelo-gpu-wgpu/tests/qwen3_long_context_bench.rs` | 4 | add `gpu_gelo_fused` cell |

## 10. What M1.10 does NOT fix

The +480 % overhead at n=2048 has two components:

- **Attention compute** (~7 s of the 9.3 s plain baseline) — M1.10
  fixes this.
- **GELO mask round-trip on linear projections** (~66 s of the 75 s
  gpu_gelo TTFT) — **not touched by M1.10**.

After M1.10 the n=2048 gpu_gelo TTFT is likely to be in the
**60-70 s** range (mask round-trip dominating, attention now cheap),
which is still much worse than gpu_plain. The remaining mask cost
needs separate attacks:

- Faster CPU BLIS for mask matmul (audit AOCL-BLIS thread-count and
  faer parallel back end — quick win, limited headroom)
- Block-diagonal mask `A = diag(A_1, ..., A_B)` — privacy weakening,
  needs separate analysis; filed in `future-rnd.md`
- Mask reuse across decode steps via HKDF-derived material — partly
  decode-only

Document those follow-ons in `gelo-llm.html` §09 when their plans
firm up. M1.10 is the structural answer for the **attention**
slice, not for the mask round-trip on linears.

## 11. Aggregate effort & dependencies

| Phase | Effort |
|---|---:|
| Phase 0 (causal-mask leak resolution) | 1 week |
| Phase 1 (cached-path wiring) | ~3 days |
| Phase 2 (engine kernel) | 3 weeks (Option A baseline) · +2 if Option C forced |
| Phase 3 (threshold tuning) | ~2 days |
| Phase 4 (long-context bench) | ~1 day |
| **Total** | **~5 weeks** (Option A); ~7 weeks if Option C forced |

Net delta vs the path-1 §M1.10 estimate (3 weeks): +2 weeks for
Phase 0 (security review + leak fix), which path-1 didn't budget.

**Critical-path dependency:** Phase 0 is mandatory before Phase 1
exposes the leak in production. Phase 2 can proceed in parallel
with Phase 0–1 because it depends only on the trait-seam shape,
which is already locked.

## 12. References

- Amulet softmax-equivariance: arXiv 2512.07495 (the protocol that
  motivates permuted attention)
- Hidden No More: arXiv 2505.18332 (Gaussian noise mitigation against
  sequential-vocabulary-matching attacks on fixed permutations)
- `gelo-llm.html` §07 — compute flow under the permuted protocol
- `path-1-gelo-gemma.md` §M1.10 — parent milestone bullet
- `inference-optimization.md` §2.1.3, §4.2 — related (and partly
  superseded — see Tier 3.4 strike) discussion
- `crates/gelo-protocol/src/attention.rs` — TEE-side wrapper
- `crates/gelo-protocol/src/substrate.rs:160-219` — trait seam
- `crates/gelo-gpu-wgpu/tests/qwen3_long_context_bench.rs` — the
  bench the M1.10 acceptance gate is measured against
