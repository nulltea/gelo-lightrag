---
type: handoff
status: current
created: 2026-05-22
updated: 2026-05-22
tags: [qwen3, mask]
---

# Handoff — 2026-05-22 — Qwen3-4B batched-decode mask sweep at B=8 n=2040 + bottleneck triage

Focus area for the next session: act on the bottlenecks this bench
surfaced, plus complete D2 / D3 from the M1.11 plan. Two benches ran
this session — short-context B=2 sweep over {128, 256, 512} for
sanity, then targeted B=8 n=2040 (pow2-aligned at s=2048) with Auto
vs HD₃-shield-to-pow2. Haar previously characterised as 2-3× slower
under M1.11 and skipped at B=8.

## Status on M1.11 (not duplicated here — see commits + plan)

- D1 (substrate + KV cache + generate_batched + tests) **landed**;
  see commits `9e5a758`, `8859558`, `dea199a`, `87b07d6`,
  `61043e2`, `33f5de6`, `407142b`.
- D1.6 decode-step microbench (synth Q4B-shape, B=8): final
  **5.23× vs serial**.
- D2 (orchestrator rewire) substrate landed (`DecoderRuntime::generate_extraction_batched`
  override + trait method); orchestrator per-chunk loop in
  `lightrag-private` still **needs the rewire**.
- D3 (AloePri `c5_batched_decode_shared_a` spot-check + crossover
  bench + Phase-1b re-eval) **untouched**.
- §08 mask-family narrative rewritten in `gelo-llm.html` (HD₃ +
  shield-to-pow2 as current design; DCT-IV deprecated). The
  synthesised projections there should be **replaced with real
  numbers** from this session — see §"Update doc figures" below.

Plan: `docs/plans/m1-11-batched-decode.md`.

## What ran (and the bench file)

`crates/gelo-gpu-wgpu/tests/qwen3_4b_batched_mask_sweep.rs`
(uncommitted; `#[ignore]` real-weight bench).

- Real Qwen/Qwen3-4B from HF cache (~14 GB f32 working set).
- Sequential per-mask-config executor (one alive at a time —
  three-alive triggered Vulkan command-submission OOM at warmup;
  see §"OOM root cause" below).
- `forward::run_prefill_batched` → K=4 × `run_decode_step_batched`,
  per-cell profile reset + snapshot.

Final config: `BATCH_SIZE = 8`, `DECODE_STEPS = 4`, lengths =
`[2040]`, configs = `[Auto, Hd3ShieldToPow2]` (Haar skipped per
prior characterisation).

## Headline results — B=8, n=2040 (pow2-aligned s=2048)

| Metric | Auto | HD₃ (shield-to-pow2) | Δ |
|---|---:|---:|---:|
| TTFT | **127.7 s** | **126.8 s** | < 1 % |
| TPOT (mean) | 1 613.9 ms | 1 611.5 ms | < 0.2 % |
| Per-seq total wall | 16.8 s | 16.7 s | ~ |

Auto and HD₃-shield-to-pow2 are statistically identical here —
expected: at the pow2-aligned shape Auto resolves to HD₃, so both
run the same kernel. The cell validates that the shield-to-pow2
strategy *under-the-hood* is on par with Auto when the operand
already lands on pow2.

## Comparison vs B=2 (short sweep earlier this session)

| Config | B | TTFT | Per-seq total wall | Δ per-seq wall vs B=2 |
|---|---:|---:|---:|---:|
| Auto n=2048 | 2 | 49.1 s | 26.5 s | (base) |
| **Auto n=2040** | **8** | **127.7 s** | **16.8 s** | **−37 %** |
| HD₃ n=2048 | 2 | 65.1 s | 34.5 s | (base) |
| **HD₃ n=2040** | **8** | **126.8 s** | **16.7 s** | **−52 %** |

TTFT grows ~2.5× absolute (vs 4× more sequences) — sub-linear
scaling via GPU dispatch amortisation across the wider batch.
**Per-sequence wall drops 37-52 %** going B=2 → B=8 at the same
per-block shape. That's the M1.11 batching win, on real weights, at
production-shape long context.

## Per-op breakdown — prefill (TTFT 127.7 s, Auto)

| Bucket | Time | Share | Notes |
|---|---:|---:|---|
| `engine:matmul_many` | 35.3 s | **29.5 %** | QKV-fused + gate/up-fused GPU dispatches |
| `engine:matmul` | 24.5 s | **20.5 %** | O + down GPU dispatches |
| **GPU subtotal** | **59.8 s** | **50.0 %** | |
| `tee:attn_inplace_many` | 21.1 s | **17.6 %** | Per-sequence in-TEE prefill attention; **R1.4 trigger now real** |
| `gelo:mask_unapply` | 19.4 s | 16.2 % | HD₃ Aᵀ apply, rayon over B blocks |
| `gelo:mask_apply` | 11.4 s | 9.5 % | HD₃ A apply, rayon over B blocks |
| `gelo:shield_stack` | 2.5 s | 2.1 % | D1.7 parallel shield fill — well-amortised |
| `tee:residual` | 2.4 s | 2.0 % | |
| `tee:qk_norm` | 1.4 s | 1.2 % | |
| `tee:swiglu_activate` | 0.9 s | 0.7 % | |
| `tee:rope` | 0.4 s | 0.4 % | |
| `tee:rmsnorm` | 0.4 s | 0.3 % | |
| other (embed_lookup, mask_sample) | < 0.1 s | 0 % | |

GPU 50 % / CPU mask 25.7 % / in-TEE attention 17.6 % / other 6.7 %.

## Per-op breakdown — decode (TPOT 1.61 s × 4 steps, Auto)

| Bucket | Time | Share | Notes |
|---|---:|---:|---|
| `tee:attn_cached_inplace_many` | 3.54 s | **55.0 %** | n_kv ≈ 2 044, per-sequence in-TEE causal attention; **the dominant decode bucket** |
| `engine:matmul` | 1.14 s | 17.7 % | |
| `engine:matmul_many` | 0.92 s | 14.2 % | |
| `gelo:mask_unapply` | 0.35 s | 5.5 % | |
| `gelo:shield_stack` | 0.22 s | 3.4 % | |
| `gelo:mask_apply` | 0.22 s | 3.4 % | Per-sequence A_b at s=16 — tiny |
| other | 0.05 s | 0.8 % | |

At decode the GPU bucket collapses to ~32 % (smaller per-call work
on small `(B, 1, d)` Q-shape); CPU attention over the n_kv prefix
takes over.

## Bottlenecks flagged

1. **`tee:attn_inplace_many` is now genuinely the R1.4 trigger**
   (17.6 % at prefill, 55 % at decode at n_kv=2k). The doc-time
   threshold of "ship R1.4 when this bucket > 10 %" is met at
   long-context batched prefill and dominates at long-context
   batched decode. R1.4 = single dispatch through
   `engine.fused_attention_batched` with per-sequence
   right-padding + causal mask folded into one additive
   `(B·num_heads, n_q, n_kv)` tensor. Substrate is already wired
   for it (the `fused_attention_batched` engine method + the
   `permuted_attention_cached` legacy path) — what's missing is
   the reshape + mask-construction + dispatch in
   `decoder_block_batched` (prefill) and
   `decoder_block_cached_batched` (decode). Estimated 2-3 days of
   work. **Single biggest M1.11+ perf lever left.**

2. **GPU duplicates 8 GiB of weights between host RAM and VRAM.**
   Memory residency table in session transcript — `register_weight_bf16`
   uploads weights to VRAM at provision, but the host `Arc<Array2<bf16>>`
   stays alive in `DecoderWeights`. At `verify_probes == 0` and
   all-layers-offloaded (Qwen3-4B default config) the host copy is
   pure waste. **Fix:** make `provision_weight_bf16_shared` consume
   the Arc and drop it once VRAM upload completes. 1-day code
   change; saves ~7 GiB host RAM.

3. **iGPU VRAM budget is artificially capped** even though Strix
   Halo UMA is 64 GiB. The earlier B=2 n=4096 / B=8 n=2048 OOMs
   were the wgpu/Vulkan per-submission command-buffer cap
   (~8 GiB), not real memory pressure. **Action:** investigate
   wgpu allocator config — bump submission cap to use the full
   UMA budget. Unblocks B=16/32 at long context.

4. **KV cache stays host-resident under current threat model.**
   At B=8 max_cache_len=2052 it's ~5 GB host. Moving it to VRAM
   needs an encrypted-KV-on-GPU scheme (SCX-class research, ~12
   month effort). Park.

5. **Mask CPU apply+unapply at 25.7 % combined** is the second
   bucket. Rayon-parallel across B blocks already (R1.5); each
   block bound by FWHT memory bandwidth. Further wins need either
   AVX-512 hand-rolled FWHT (incremental) or bf16 mask GEMM
   (deferred per `bf16_mask_gemm_skipped`).

6. **Masked-operand round-trip** = `2 × B · s_pad · d · 4 B` of
   memcpy per offload between host RAM and VRAM. On UMA there's
   no actual data movement needed — a wgpu-mapped buffer the TEE
   writes the masked operand directly into would eliminate the
   per-offload copy. Substrate + engine API change; sits behind
   #3 (need UMA-aware wgpu config first).

7. **LM-head / `token_embedding` (deferred to M1.13).** The tied
   input/output embedding (152 064 × 2560 bf16 = 778 MB) is
   host-resident today because `embedding_lookup` (prefill input)
   and `compute_logits` (decode-step output projection) both run
   in-TEE. The earlier perf memory estimated `compute_logits` at
   ~120 s/forward in the v7 extraction fixture (151 k-vocab ×
   2560 dot product per decoded token, single-thread bf16
   widening). That's potentially as large as the entire GPU-matmul
   bucket. **Action item before M1.13 commits:** add
   `profile::time("tee:compute_logits", ...)` around the loop in
   `crates/gelo-embedder/src/decoder/generation.rs::compute_logits`
   and re-run a Qwen3-4B extraction generate. If the bucket is
   confirmed at ≥ 10 % of decode wall, M1.13 lifts it to a masked
   LM-head offload through the R3 mapped-buffer API. Threat-model
   note: GPU sees the (1, vocab) logit output even masked — for
   the strongest sampling-privacy story this stays TEE-resident
   and we accept the cost. Decision rides on the measured share.

## UMA vs production-dGPU representativeness (READ BEFORE INVESTING IN R3)

All Q3-4B numbers above were collected on **Strix Halo iGPU with
64 GiB UMA**. UMA means host RAM and "VRAM" are the same physical
DRAM pool; the wgpu "upload" / "download" calls are just memcpys
within shared memory at ~80 GB/s DDR5 bandwidth. **This is not
representative of the production deployment topology.**

Production target (per `docs/dev/logs/path-2-status.md` and related):
SEV-SNP CVM with VFIO-passthrough discrete GPU connected over PCIe.
The masked-operand round-trip is a genuine PCIe DMA transfer
(Gen5 x16 ≈ 64 GB/s nominal, ~30-40 GB/s realised on AMD/NVIDIA
dGPUs). The semantics differ fundamentally:

| | iGPU UMA (this bench) | dGPU PCIe (production) |
|---|---|---|
| Upload path | Host write → mapped buffer (same physical RAM) → GPU read | Host write → DRAM → PCIe DMA over wire → GPU VRAM → GPU read |
| R3 zero-copy applicable? | **Yes** — eliminates pure-memcpy | **No** — DMA transit is unavoidable |
| Round-trip cost dominated by | DDR5 bandwidth + wgpu API overhead | PCIe bandwidth + DMA setup latency |
| Optimisation lever | UMA-mapped buffer (R3 fast path) | Fewer round-trips per layer (matmul_many fusion, async pipeline) |

**What this means for M1.12 R3:**

- R3's UMA fast path is genuinely useful **for development /
  evaluation on this hardware** (Strix Halo class iGPUs).
- R3's copy fallback path is what runs in production on dGPU. It
  doesn't accelerate the round-trip — PCIe DMA is the floor.
- The "~48 GiB memcpy / forward" claim and the "TTFT 127 s → 100 s"
  estimate apply to **iGPU UMA only**. Do not generalise to
  production dGPU.

For production dGPU optimisation, the relevant levers are
different:

- **matmul_many fusion** (M1.11 R1.6 — already shipped). Fewer
  round-trips by batching QKV / SwiGLU gate+up.
- **Async pipelining** — overlap CPU mask-apply for layer L+1 with
  GPU matmul for layer L. Untried; could halve the per-layer wall
  on a sufficiently provisioned dGPU.
- **Persistent VRAM-resident intermediates** — keep the unmasked
  output in VRAM across the unmask → next-layer-mask-apply round-
  trip. Breaks the GELO protocol as currently specified (the
  unmask step is in-TEE, can't happen on the GPU), so this needs
  a protocol re-validation.
- **PCIe Gen5 vs Gen4** — production hardware selection. The TTFT
  floor on Gen4 x16 is roughly 2× Gen5.

**Recommendation:** keep M1.12 R3 (the UMA-mapped path) as planned
for iGPU dev velocity, but mark it explicitly as "iGPU dev path".
A separate M1.14-ish milestone should target production dGPU on
real SEV-SNP hardware (the §M5.9 hardware bring-up in
`future-rnd.md`) and validate that **the GELO substrate's
per-offload round-trip cost is acceptable at PCIe bandwidth**
before assuming any of the M1.12 wins carry over.

## OOM root cause (won't repeat — captured here for next session)

Original bench had three executors alive simultaneously sharing
the GPU via `clone_shared`. Each clone owns its per-call scratch
buffers and pending Vulkan command queue independently. With three
executors:
- Each ~36 layers × 4 dispatches per layer × ~80 MB queued = ~12 GiB
  pending command-buffer memory per executor.
- × 3 executors ≈ 36 GiB pending — exceeds RADV per-submission cap
  (~8 GiB on this driver).

Fix applied: sequential executors (one alive at a time, dropped
between configs). Weight upload happens 3× total but RSS stays
bounded.

Per-cell B/n safe-zone table (`B · s_pad ≤ ~30 000` conservative
cap on this iGPU until the wgpu allocator is reconfigured):

| n_prompt | s_pad | safe B | comfortable B |
|---:|---:|:---:|:---:|
| 512 | 1 024 | ≤ 30 | 16 |
| 1 024 | 2 048 | ≤ 16 | 8 |
| 2 048 | 4 096 | ≤ 8 | 4 |
| 4 096 | 8 192 | ≤ 4 | 2 |

## Earlier B=2 long-context numbers (full {512, 1024, 2048, 4096} cells)

The previous run captured 11 of 12 cells before kill (HD₃ @ n=4096
incomplete). Headline finding: **Haar's `O(s³)` QR sample blows
up at long n** — TTFT 10.5 → 30.6 → 105 → 407 s across {512, 1024,
2048, 4096}. Auto + HD₃ retire the QR cost; both within ~30 % of
each other. Full table is in session transcript; reproducible via
`BATCH_SIZE = 2` + `DEFAULT_PROMPT_LENGTHS = [512, 1024, 2048, 4096]`
on the bench file.

## Action items for the next session

Ordered by ratio of (impact ÷ engineering cost):

1. **Drop host weight duplicates** post-VRAM upload. 1 day; saves
   ~7 GiB host RAM at default config. No perf change but the bench
   harness can then comfortably load Q3-4B at B=16 without RSS
   stress.

2. **wgpu UMA allocator reconfigure.** Investigate
   `WgpuDeviceDescriptor`/`Limits` configuration to expose the full
   64 GiB UMA budget. Unblocks B=16/32 + n=4096 cells that today
   command-submission-OOM. Time-box 1-2 days for the wgpu config
   spike.

3. **R1.4 — batched-attention GPU kernel** through
   `engine.fused_attention_batched` with per-sequence right-padding
   + causal mask. 2-3 days. Hits the largest remaining bucket
   (17.6 % at prefill, 55 % at decode).

4. **D2 — orchestrator rewire** in `lightrag-private`. Per-chunk
   loop in `extract_kg_from_chunks` replaced by one
   `decoder.generate_extraction_batched` call. Substrate already
   landed. ~1 day. Realises the headline 5× wall-time win on the
   v7 extraction fixture.

5. **D3 — AloePri `c5_batched_decode_shared_a` spot-check** at B=8
   gates flipping the shared-A default-on. ~3 days. Currently
   shared-A path is gated behind `BATCHED_DECODE_SHARED_A=1` env;
   default = per-sequence A_b.

6. **Update gelo-llm.html §08 figures** with real B=8 n=2040
   numbers from this bench. The synthesised "HD₃ + shield-to-pow2
   ~32 s @ n=2048" projections I wrote there should be replaced
   with the measured 49 s (Auto B=2) / 65 s (HD₃ B=2) / 127 s
   (Auto B=8) / 127 s (HD₃ B=8). 1-hour edit.

7. **Commit the sweep bench file**
   (`qwen3_4b_batched_mask_sweep.rs`). Currently uncommitted;
   archive the bench so future re-runs are reproducible. 15-min.

## Memory residency reference

Full residency tables (current state, threat-model-permissible,
offload-max recommendations) captured inline in the session
transcript. Key items:

- **VRAM-stationary today**: weights only.
- **VRAM-stationary possible (threat-model permitting)**: masked
  operand + GEMM output via UMA mapping.
- **Must stay host (TEE-resident)**: mask `A`, permutation `π`,
  KV cache, plaintext activations, shield rows pre-fold, tokens.

## Suggested skills for the next session

- **`diagnose`** for the wgpu UMA allocator investigation
  (item #2) — diagnostic-shaped, needs reproduction + bisection
  + instrumentation discipline.
- **`grill-with-docs`** before flipping any AloePri-gated defaults
  (the `c5_batched_decode_shared_a` work in item #5).
- **`verify`** after R1.4 lands — confirm batched-attention-kernel
  output matches the per-sequence in-TEE stopgap at f32 floor on
  the same fixture.

## Reproducing this session's benches

```bash
# Short-context B=2 sweep (small-length sanity, ~2 min)
# (edit BATCH_SIZE = 2, DEFAULT_PROMPT_LENGTHS = [128, 256, 512])
cargo test -p gelo-gpu-wgpu --release \
  --test qwen3_4b_batched_mask_sweep \
  -- --ignored --nocapture

# Long-context B=2 sweep ({512, 1024, 2048, 4096}, ~25 min;
# previous run captured 11 of 12 cells before kill)
# (edit BATCH_SIZE = 2, DEFAULT_PROMPT_LENGTHS = [512, 1024, 2048, 4096])
# Haar @ n=4096 alone takes ~7 min; whole sweep is the long pole

# B=8 sweep at pow2 (~3 min, ran clean)
# (edit BATCH_SIZE = 8, DEFAULT_PROMPT_LENGTHS = [2040], configs = [Auto, Hd3ShieldToPow2])
```

All three configurations sit in the same `qwen3_4b_batched_mask_sweep.rs`
file with the constants at the top — pick the matrix by editing
those constants.
