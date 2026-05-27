---
type: handoff
status: current
created: 2026-05-21
updated: 2026-05-21
tags: [gelo, perf, attention]
companion: [2026-05-21-attn-offload-spike]
---

# Handoff — 2026-05-21 — GELO perf: shield_stack, in-TEE attention, batched decode

## What this session shipped

Five commits on `master`, all on top of `c63e676`:

| Commit | Title |
|---|---|
| `7ccb81a` | feat(private-graphrap): bf16-native loader + GPU engine + take-after-upload |
| `2838ae5` | perf(gelo-protocol): default InProcessTrustedExecutor mask to Auto |
| `5bb16ea` | fix(extraction): Qwen3 chat template + disable thinking mode |
| `b49ba7a` | perf(gelo-protocol): split mask_apply/unapply profile by family + tune Auto threshold |
| `0cbe858` | perf(gelo-protocol): shape-adaptive shield_k for HD₃ alignment at decode |

Plus the prior plan-mode work for the GraphRAP route + bench harness (`feat(private-graphrap)` covers the bulk: chunker, extraction module, runner refactor, bench example, integration test).

**Substrate changes the next session inherits:**

- `gelo-protocol`: `MaskKind::Auto` is the default executor mask; HD₃ threshold is `7/5` (1.4); per-family profile categories `gelo:mask_apply:{haar,hd3,dct4}` + `gelo:mask_unapply:*`; shape-adaptive shield (`shield_default=k8`, `shield_small_n=Some(k15)` at `n≤1`); `MaskFamily::{apply,unapply}_profile_category` helpers. `RayonCpuEngine` is `#[deprecated]`.
- `gelo-protocol::substrate`: `GpuOffloadEngine::register_weight_bf16` + `register_weight_bf16_shared`; `TrustedExecutor::provision_weight_bf16` + `provision_weight_bf16_shared`.
- `gelo-gpu-wgpu`: `array2_bf16_to_tensor_f16` / `array2_bf16_to_tensor_f32` direct-upload paths; `WgpuVulkanEngine` overrides `register_weight_bf16` (no host f32 intermediate).
- `gelo-embedder::decoder::weights`: `DecoderLayerWeights.{wq,wk,wv,wo,w_gate,w_up,w_down}` is `Option<Arc<Array2<bf16>>>`; `token_embedding` is `Array2<bf16>`; loader uses `view_to_bf16` (no widening); `DecoderWeights` + `DecoderLayerWeights` derive `Clone` for parity tests.
- `gelo-embedder::GeloQwenEmbedder`: `new()` takes **owned** `DecoderWeights` and `take()`s the offloadable Arcs into the engine, dropping host bytes once the wgpu engine consumes them. `with_shared_weights(Arc<DecoderWeights>)` is the test-only parallel constructor that keeps host bytes alive.
- `gelo-reranker::CausalDiscriminatorRerankService`: same take-on-provision pattern; yes/no-head widens bf16→f32 per-element.
- `gelo-snp-runner`: split into lib + thin bin. `DecoderRuntime<E>` / `GeloDescriptionEmbedder<E>` are engine-generic; production concretises on `WgpuVulkanEngine` (= `RunnerEngine`). `POST /lightrag/extract_and_build` wraps chunker → extract → embed → ingest. `DecoderRuntime::generate_extraction` applies Qwen3 chat template with `<|im_start|>system ... /no_think ... <|im_start|>assistant\n<think>\n\n</think>\n\n` (disables reasoning mode — critical, without this Qwen3-4B emits 0 tuples at max_tokens=512).
- `crates/lightrag-private/src/extract/`: prompt builder + tuple parser + orchestrator. Per-chunk profile dump via `gelo_protocol::profile::snapshot().dump(...)` at chunk end. Decoded-text log when 0 entities parsed.

Memory notes added: `feedback_benches_use_gelo_gpu.md` (sharpened), `feedback_memory_efficiency_priority.md` (new P0 rule), `feedback_no_rayon_cpu_engine.md` (new).

## Current baseline (bench v7, 2026-05-21)

`cargo run -p gelo-snp-runner --release --example extract_and_query_bench` with `BENCH_MAX_CHUNKS=1`, Qwen3-4B (extraction) + Qwen3-Embedding-0.6B (embeddings), wgpu/fp16, paper-parity-default executor (Auto mask, shape-adaptive shield, per-forward-pass A).

One chunk (633 chars, prompt 745 tokens, output capped at 512 — model hits max_tokens before `<|im_end|>`):

| Bench | Wall (s) | Notes |
|---|---:|---|
| v6 (k=8, DCT-IV decode) | 359.5 | baseline post-Auto-threshold-tune |
| v7 (shape-adaptive k=8/k=15, HD₃ everywhere) | 361.2 | mask win cancelled by shield_stack cost — see profile below |

**Bottleneck profile (v7, ms / share of measured = 242.2 s; wall = 361.2 s; ~120 s unaccounted):**

| Stage | Time (ms) | Share | Calls | Class |
|---|---:|---:|---:|---|
| `tee:attn_cached` | 77 525 | 32.0 % | 18 468 | CPU/TEE — in-TEE attention over cached prefix |
| `engine:matmul` | 48 407 | 20.0 % | 36 936 | **GPU** |
| `engine:matmul_many` | 40 028 | 16.5 % | 36 936 | **GPU** |
| `gelo:shield_stack` | 35 871 | 14.8 % | 73 872 | CPU — RNG + memcpy of k shield rows per offload |
| `gelo:mask_unapply:hd3` | 24 688 | 10.2 % | 129 276 | CPU |
| `gelo:mask_apply:hd3` | 14 212 | 5.9 % | 73 872 | CPU |
| (norms, RoPE, swiglu, residual, embed, sample, strip) | ~1 460 | 0.6 % | — | CPU |

Per-call timings (v6 → v7 comparison):
- `mask_apply`: DCT-IV 358 µs/call → HD₃ 192 µs/call (**−46 %**)
- `mask_unapply`: DCT-IV 290 µs/call → HD₃ 191 µs/call (**−34 %**)
- `shield_stack`: 257 µs/call (k=8) → 486 µs/call (k=15) (**+89 %**)

GPU vs CPU split: GPU ≈ 21.5 % of wall (matches user-observed nvtop reading 15-21 %). CPU stages are **structurally in-series with GPU** by protocol design — every decode step is `[mask CPU] → [GPU matmul] → [mask CPU] → [in-TEE attention CPU] → [next token]`.

## Bottlenecks ranked

1. **`tee:attn_cached`: 32 % of wall, 77.5 s.** In-TEE attention (Q·Kᵀ + softmax + scores·V) runs per decode step on the full cached prefix. Per GELO §3 the softmax non-linearity prevents masked offload. Memory note `gelo_research_round_2.md` flags Amulet's softmax-equivariance as the candidate research surface. **Biggest single non-architectural lever.**

2. **`gelo:shield_stack`: 14.8 % of wall, 35.9 s.** RNG-and-memcpy-bound per-call cost; scales linearly with `k`. Currently 486 µs/call at k=15 generating 38 400 `StandardNormal` samples. RNG is `ChaCha20Rng` via `rand_distr::StandardNormal` (scalar Box-Muller). Optimization candidates below.

3. **`gelo:mask_*:hd3` combined: 16 % of wall, 38.9 s.** Already at the HD₃ radix-8 FWHT SIMD path. Further wins are kernel-tuning territory.

4. **~120 s unaccounted (33 % of wall).** Profile measured 242 s of 361 s wall. Primary suspect: `compute_logits` runs a 151 936-vocab × 2 560-d row-dot per decoded token in-TEE (single thread, bf16 widening per element). At 512 decode tokens × 388 M multiplies = 199 B widening-multiplies on a single CPU core ≈ ~120 s estimate. **Not yet instrumented; add `profile::time("tee:compute_logits", …)` to confirm.**

## Per-query baseline (unaffected by extraction work)

Mean across 5 hybrid queries against the (10-entity, 4-relation, 1-chunk) LightKgStore built from the chunk:

| Stage | Mean ms |
|---|---:|
| embed (LL+HL, two Qwen3-Embedding-0.6B forwards) | 148.6 |
| entities_search (Compass over Ring-ORAM) | 17.0 |
| relations_search | 4.4 |
| adjacency (XorMM) | ~0 |
| src_chunks (XorMM) | ~0 |
| chunk_decrypt (AES-GCM) | ~0 |
| **total (excl. embed)** | **21.5** |

Compass+XorMM+AES retrieval is sub-25 ms at this corpus size. Query-time cost is dominated by the embedder forward pass.

## Next-session focus (per /handoff args)

### A. `gelo:shield_stack` optimisations

**Current state:** `crates/gelo-protocol/src/shield.rs::stack_shield` + the inline scratch-reuse variant at `sim.rs:640-694`. Generates `k` rows of `N(0, σ²)` via `rand_distr::StandardNormal::sample()` per call. Scalar per-element Box-Muller.

**Why it's 486 µs/call at k=15, d=2560:** 38 400 calls into `StandardNormal::sample`, each producing one f32. The inner loop is `let v = ((-2.0 * log(u1)).sqrt()) * cos(2π·u2)`. No vectorisation. Plus the `.assign()` row-copy of the data block.

**Levers, ranked by likely return:**

1. **SIMD batched Box-Muller** — process 4 or 8 f32 lanes per iteration. `wide` crate or hand-rolled AVX2/AVX-512. Estimated 4-8× speedup on the RNG inner loop. Touches `shield.rs` only.
2. **Faster RNG than ChaCha20** — for the shield rows specifically, `Xoshiro256++` is ~3× faster than ChaCha20 and still passes BigCrush. Security: the shield rows aren't the cryptographic randomness — the mask `A` is. Shield is "data confusion noise", not key material. Could be a separate RNG instance.
3. **Skip per-row energy scaling** — currently sigma is computed per call from `mean_row_norm(hidden)` then scaled to each draw. If we precompute σ at `begin_forward_pass` we skip per-call work, but only saves a tiny fraction. Low return.
4. **Cap k=8 even at decode** — i.e. revert the shape-adaptive overlay. Trades the HD₃-at-decode win for the smaller shield_stack cost. Net wall is the same per v6/v7 comparison; only the bucket distribution changes. Already implemented as `with_small_n_shield(None)`. **Could be the cleanest "no-op" if the SIMD path doesn't materialise quickly.**

Quick A/B candidates for the bench harness: `cargo run -p gelo-snp-runner --release --example extract_and_query_bench` with and without `with_small_n_shield(None)` plumbed through `DecoderRuntime::from_config_and_dir` (currently the runtime always uses the executor's default; would need a constructor param).

### B. `tee:attn_cached` GPU offload

**Current state:** `gelo-embedder/src/decoder/attention.rs::causal_gqa_attention_cached` (and its `_swa`/`_permuted` variants). Runs entirely in-TEE on f32 — Q·Kᵀ → softmax → scores·V — at every decode step over the full cached K/V. Per memory `gelo_research_round_2.md` the candidate research direction is **Amulet's softmax-equivariance**, which lets the masked Q · Kᵀ be offloaded provided the softmax is rearranged to commute with the orthogonal action.

**Existing scaffolding:**
- `gelo_protocol::substrate::GpuOffloadEngine::softmax_batched` already exists with a default in-process impl + a wgpu override (`burn_tensor::activation::softmax`).
- `gelo_protocol::substrate::GpuOffloadEngine::matmul_dynamic_batched` exists with a wgpu override that fuses per-head Q·Kᵀ across heads.
- `gelo_protocol::out_attn_mult` is the TwinShield OutAttnMult path — already offloads the `Q · Kᵀ` matmul under a permutation shield, but **only at prefill** (auto-switch threshold `out_attn_mult_min_seq_len = hidden_size`). Decode at m=1 falls through to `causal_gqa_attention_cached`.
- `gelo_protocol::attention::PermAttnConfig` is the Tier 1 permutation-shielded attention; `with_perm_attention(true)` enables it. Default off — see `gelo-embedder/src/decoder/embedder.rs::with_perm_attention`.

**Suggested first probe:** enable `with_perm_attention(true)` on the bench's `DecoderRuntime` executor and re-run. If perm attention is correctness-stable on real Qwen3-4B weights at the decode shape (m=1, full cached prefix), it should offload Q·Kᵀ to GPU. The `with_perm_attention_min_seq_len` threshold (default 64, see `embedder.rs:159`) may need to be lowered to allow firing at decode.

**Risk:** perm attention security has caveats — see `feedback_aloepri_capture_resource_reuse.md` and the AloePri capture pipeline that captures snapshots under `c1_mask_only` / `c2_default` / `c3_hd3`. A new `c5_perm_attn` condition might be needed before flipping defaults. Don't enable in production without re-running the attack-suite gate.

### C. Batched decode (m=N decode steps)

**Architectural change.** Currently the generation loop in `gelo-embedder/src/decoder/generation.rs::generate` processes one sequence at a time with m=1 decode steps. To batch N sequences:

1. Convert the KV cache to `(B, layers, max_cache_len, kv_dim)`.
2. Change `run_decode_step` to accept `&[u32]` (one token per sequence) and return `Array2<f32>` (B × hidden).
3. Mask at `stacked_n = B + shield_k` instead of `1 + shield_k` — amortises the per-call CPU overhead across B sequences.
4. Auto resolves to HD₃ much more readily once `B ≥ 7` (stacked_n = B+8 ≥ 15, pad to 16, ratio ≤ 1.07).
5. Sampling loop diverges per sequence — early-stop on EOS per sequence; the others continue. Standard "right-padding with EOS" trick.

**Where it'd apply:** the extract path runs one chunk at a time (serial; can't easily batch across chunks because the prompts differ). But within a single chunk, B=1 by nature. Batching is most useful at **inference serving time** when multiple users hit the runner concurrently. For a single-tenant extraction bench, batching helps only if we can batch across chunks (process all 7 chunks' prefills in parallel, then their decodes in parallel up to their respective EOS) — that's a non-trivial scheduler.

**Likely highest return:** rerank workloads. `CausalDiscriminatorRerankService` already calls `forward::run` per (query, candidate) pair — N candidates = N forwards. Batching N candidates into one forward at m=N would amortise everything across the batch.

## Reproducing the bench

```bash
cd /home/timo/repos/private-rag
BENCH_MAX_CHUNKS=1 \
RUST_LOG=info \
cargo run -p gelo-snp-runner --release --example extract_and_query_bench
```

The example downloads `Qwen/Qwen3-4B` (~7.5 GB bf16) and `Qwen/Qwen3-Embedding-0.6B` from the HF cache on first run.

`BENCH_MAX_CHUNKS` is the diagnostic short-circuit added this session. Drop it to run the full 7-chunk doc (~40 min wall at current rate).

For per-stage breakdown the orchestrator now calls `gelo_protocol::profile::snapshot().dump(...)` after each chunk — appears in stderr automatically. For Auto-mask resolution traces, add `RUST_LOG=info,gelo_protocol=debug`. **Beware:** the per-offload `auto-mask resolved` log floods at ~20+ events/sec; the bench monitor in the previous session had to switch from a permissive filter to `grep -v 'auto-mask'` to stay readable.

## Open follow-ups (lower priority)

- `~120 s unaccounted` in the v7 profile (33 % of wall). Likely `compute_logits` per-decode-step row-dot over 151 936 vocab; one `profile::time("tee:compute_logits", …)` around the loop in `gelo-embedder/src/decoder/generation.rs:104-127` would confirm.
- `resolve_mask_kind_for_shape` is called per-offload, not just per `begin_forward_pass`. Each call is an integer compare — imperceptible — but architecturally redundant. Cache the resolution on `SessionMask` once at sample time.
- `with_shared_weights` test-only constructor — `gelo-rag/tests/gelo_embedder_accuracy.rs` + `gelo-gpu-wgpu/tests/qwen3_overhead_{bench,breakdown}.rs` use it. Fine pattern but worth documenting that production code must use `new` (the take-on-provision constructor).
- bf16 token_embedding clones at `Clone for DecoderWeights` will copy the full bf16 buffer — fine for synth tests, expensive for real Qwen3-4B (~770 MB clone). Production never clones, but if a benchmark ever does, surprise.

## Suggested skills for the next session

- **`diagnose`** — for the `tee:compute_logits` confirmation and any of the perm-attention enablement work.
- **`grill-with-docs`** — before flipping perm-attention defaults, the attack-suite gate (AloePri `c5_perm_attn`?) needs design discussion. The docs to grill: `docs/prototype/gelo-llm.html` (§06 sampling, §07 obfuscation), `docs/research/private-llm-inference-round-2.md`, `feedback_aloepri_capture_resource_reuse.md`.
- **`verify`** — after any of the three perf changes, confirm the bench still extracts 10 entities + 4 relations on the v7 fixture chunk (`Acme Corp / Alice / Bob / Helvetia Foundation / OpenSouce / Paris / National Cryptography Lab / European AI Trust Foundation` plus 2). Output should be deterministic given greedy sampling + fixed `tee_user_x_sk = [0xb2; 32]`.

## Tasks state at end of session

All inline session tasks closed. The TaskCreate/Update tracker had tasks #6-#12 covering instrumentation, bench example, GPU+Arc refactor, bf16 loader, and bench execution; all completed.

## Follow-up landed 2026-05-21 (same day, second session) — §A shipped

Section A (`gelo:shield_stack` optimisation) is **done**. Section B
(perm-attention) and Section C (batched decode) remain open.

**What changed (one logical commit):**

- New module `crates/gelo-protocol/src/gaussian.rs` — public
  `fill_gaussian(dest: &mut [f32], sigma: f32, rng: &mut impl
  RngCore)`. Bulk-RNG fill via `RngCore::fill_bytes` + Box-Muller
  in `wide::f32x8` lanes (8-lane AVX2 on stable Rust, scalar
  fallback). 5 unit tests gate mean/variance/scaling/zero-σ/tail/
  deterministic-seed.
- Wired into `shield::stack_shield` (per-offload legacy path) and
  `sim::fill_shield_rows_inline` (per-forward-pass scratch-reuse
  hot path).
- New criterion bench `crates/gelo-protocol/benches/shield_gaussian.rs`
  (decode k=15 / prefill k=8, widths d=2560 and d=1024) compares the
  legacy `rand_distr::StandardNormal::sample` loop against
  `fill_gaussian` directly. Run:
  `cargo bench -p gelo-protocol --bench shield_gaussian`.
- New convenience `InProcessTrustedExecutor::with_haar_mask()`
  setter (mirrors the existing `with_hd3/dct4/auto_mask` API).
- Two stale-test fixes drive-by:
  - `sim::tests::auto_dispatch_resolves_by_pad_ratio` updated for
    the 7/5 Auto threshold (commit `b49ba7a` changed it from 4/3
    but didn't update the test).
  - `snapshot_capture.rs::synth_executor` pinned to Haar via
    `with_haar_mask()` so its hard-coded `[n+k, d]` shape
    assertions stay valid after the executor default flipped to
    Auto (commit `2838ae5`).

**Microbench (stable RUSTFLAGS, no target-cpu=native):**

| shape | legacy_scalar | fill_gaussian | speedup |
|---|---:|---:|---:|
| d=2560 / k=15 (decode) | 551.74 µs | **345.00 µs** | **1.60×** |
| d=2560 / k=8  (prefill) | 306.40 µs | **196.09 µs** | **1.56×** |
| d=1024 / k=15 | 221.35 µs | 155.83 µs | 1.42× |
| d=1024 / k=8  | 120.64 µs | 77.41 µs | 1.56× |

With `target-cpu=native`: 1.84× at decode (not enabled by default;
worth pursuing as a separate workspace-wide experiment).

**Clean E2E (extract_and_query_bench, BENCH_MAX_CHUNKS=1):**

| stage | v7 µs/call | clean run µs/call | Δ/call | Δ total |
|---|---:|---:|---:|---:|
| `gelo:shield_stack` | **486** | **307** | **−179 (−37 %)** | **−13.2 s** |
| `gelo:mask_apply:hd3` | 192 | 189 | −3 | −0.2 s |
| `gelo:mask_unapply:hd3` | 191 | 186 | −5 | −0.7 s |
| `tee:attn_cached` | 4 197 | 3 787 | −410 | −7.6 s |

Generate wall **361.2 → 341.56 s = −5.4 %**. Profile total
242 → 220 s. Output byte-identical to v7: 10 entities, 4 raw
relations (1 malformed dropped → 3 final), greedy decode
deterministic. Per-call shield_stack carries through from
microbench (345 µs) → E2E (307 µs); E2E faster than microbench
because the scratch-reuse path amortises the `Vec` allocation.

**Bottleneck ranking after this change (clean run shares):**

1. `tee:attn_cached` — 31.7 % (next lever; §B in this doc)
2. `engine:matmul` — 21.5 % (GPU)
3. `engine:matmul_many` — 18.6 % (GPU)
4. `gelo:mask_unapply:hd3` — 10.9 %
5. `gelo:shield_stack` — **10.3 %** (down from #4 at 14.8 %)
6. `gelo:mask_apply:hd3` — 6.3 %

shield_stack drops from bucket #4 to #5. Mask apply/unapply are
unchanged (already on the radix-8 HD₃ SIMD path); their share
ticks up only because the total shrank.

**Open follow-ups specific to the shield path (lower priority):**

- `target-cpu=native` workspace-wide — would extend the win to
  ~1.84×. Saves ~10 s more wall but changes attribution for every
  other crate's perf measurement. Schedule as a deliberate
  experiment so the delta is clean.
- AVX-512 hand-rolled inner loop — `wide` only emits AVX2 (8 lanes).
  Zen 5 has full AVX-512F (16 lanes). Estimated ~1.6× extra on the
  shield kernel ≈ 3 s wall claw-back. Diminishing returns vs the
  remaining shield_stack share.
- Xoshiro256++ separate shield-RNG — bulk-fill is ~3× faster than
  ChaCha20. Security-safe (shield ≠ key material) but adds a second
  RNG state and re-runs of the `c2_default` AloePri attack-suite gate.
  ~1 day of work, ≤ 3 s wall saved.

Memory entry: `shield_simd_gaussian_landed.md`.

## Follow-up landed 2026-05-21 (same day, third session) — §A fully exhausted

Commit `3eca59e`: polar (Marsaglia) rejection + Xoshiro256++ shield
RNG.  Motivated by: moving attention to GPU (§B / perm-attention)
will roughly double the number of mask/unmask/shield cycles per
forward, so shield-row cost must drop *before* attn offload lands,
not after.

**What changed:**

- `gaussian::fill_gaussian` SIMD body rewritten from Box-Muller to
  polar method.  Drops `sin_cos` (the dominant transcendental,
  ~35 % of the prior kernel).  Branchless 8-lane SIMD with
  `move_mask` + bit-walk compaction; rejected lanes' factor is
  computed and discarded (-inf/NaN sink).  Pool sized at 1.4×
  target_pairs — covers the 21.5 % rejection rate with multi-σ
  safety margin.
- New `shield_rng: Xoshiro256PlusPlus` field on
  `InProcessTrustedExecutor`, seeded deterministically from
  `MaskSeed` via a fixed ChaCha20 stream split
  (`SHIELD_RNG_STREAM = 0xCAFE_F00D_5EED_E11D`) so the main RNG's
  stream-0 position is untouched.  Plumbed through both shield call
  sites; mask sampling and all other crypto-relevant draws still go
  through `self.rng: ChaCha20Rng`.

**Microbench** (`cargo bench -p gelo-protocol --bench
shield_gaussian`, d=2560 / k=15 decode):

| variant | µs/call |
|---|---:|
| `legacy_scalar` (Ziggurat + ChaCha20) | 214 |
| `fill_gaussian` Box-Muller + ChaCha20 | 148 |
| `fill_gaussian` polar + ChaCha20 | 154 (≈ tied) |
| `fill_gaussian_xoshiro` polar + Xoshiro | **61** |

**Synergy, not additivity**: polar's win is only realised when
paired with Xoshiro.  Polar adds 40 % more RNG bytes (1.4× pool)
which on ChaCha20 cancels the sin_cos savings; on Xoshiro the RNG
is cheap enough that the SIMD-body win dominates.  Land both or
neither.

**Clean E2E** (`extract_and_query_bench`, BENCH_MAX_CHUNKS=1):

| stage | post-144d764 | post-3eca59e | Δ |
|---|---:|---:|---:|
| `gelo:shield_stack` ms | 22 665 | **11 672** | **−10 993 (−48.5 %)** |
| `gelo:shield_stack` µs/call | 307 | **158** | **−149 µs/call** |
| `gelo:shield_stack` share | 10.3 % | **5.1 %** | dropped #4 → **#6** |
| `tee:attn_cached` s | 69.94 | 89.71 | **+19.77** ← noise |
| generate wall s | 341.56 | 342.78 | ≈ flat |

**Wall didn't budge** on this single measurement because
`tee:attn_cached` happened to be at the top of its ±15 % noise
band.  Across our three clean runs that bucket has been at
77.5 / 69.9 / 89.7 s on the same fixture — characteristic of
the shared Strix Halo iGPU/CPU memory subsystem when other tasks
warm/cool the unified memory.  Average-case wall is ~330 s, ~31 s
below v7's 361 s.

**Cumulative since v7** (both commits):

| metric | v7 | post-3eca59e | factor |
|---|---:|---:|---:|
| shield_stack µs/call | 486 | **158** | **3.08×** |
| shield_stack bucket | 35.9 s (14.8 %) | 11.67 s (5.1 %) | **−67 %** |
| bucket rank | #4 | **#6** | — |

**Output**: 10 entities + 4 raw / 3 final relations — byte-identical
to v7 across both stages.  Greedy decode determinism survives the
Xoshiro RNG switch because shield rows are post-stripped and never
propagate to logits; the mask `A` itself still comes from ChaCha20.

**Open security gate** (P3 only): the AloePri `c2_default`
attack-suite must be re-run before the Xoshiro shield-RNG lands
in any externally-attested deployment.  Theoretically the
shield-vs-key distinction is part of the protocol design (key
material is the mask `A`, not the shield) but the empirical no-
leakage claim against the new RNG bit-pattern is not yet
re-validated.  Tracked in memory `aloepri_hd3_gate_phase_a_b.md`.

**Bottleneck ranking after both commits (clean run shares):**

1. `tee:attn_cached` — 39.3 % (the obvious next target — §B)
2. `engine:matmul` — 19.2 % (GPU)
3. `engine:matmul_many` — 18.9 % (GPU)
4. `gelo:mask_unapply:hd3` — 10.7 %
5. `gelo:mask_apply:hd3` — 6.2 %
6. `gelo:shield_stack` — 5.1 %

§A is fully exhausted at the wide::f32x8 SIMD width.  Further
shield work needs either AVX-512 hand-rolling (~2 s wall, ~3 days)
or workspace-wide `target-cpu=native` (broader experiment).
Pivot to §B (perm-attention) next.
