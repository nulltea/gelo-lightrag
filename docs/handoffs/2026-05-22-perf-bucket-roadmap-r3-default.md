# Handoff — 2026-05-22 — Perf-bucket roadmap after R3 + LM-head default-flip

Focus area for the next session: pick up the M1.12 perf roadmap at
**bucket B (batched GPU attention kernel)**. R1 + Q#1 + R3 landed
this session; LM-head GPU offload becomes the **default and only
path** (the optional `LM_HEAD_GPU_OFFLOAD` env knob is gone). A
**c6 AloePri condition variant** is wired structurally — the
attack-driver run is the security gate that retroactively validates
the default-flip and remains the only blocker between the current
state and "shipped".

## State on disk

Commits this session (master, ahead of origin/master by 26):

| Commit | What |
|---|---|
| `4686b8f` | R1 — `provision_decoder_into` helper across 3 production callers + 3 benches |
| `5f85f14` | Q#1 — `tee:compute_logits` profile bucket instrumentation |
| `b257042` | R3 — LM-head GPU masked offload behind `LM_HEAD_GPU_OFFLOAD=1` |
| `caf1601` | docs(gelo-llm): LM-head GPU offload § + alias scrub + plan / handoff |
| **(this session, post-handoff)** | LM-head offload default-and-only + c6 condition stub |

Plan + prior handoff (don't duplicate context — read these first):

- [`docs/plans/m1-12-tee-gpu-throughput.md`](../plans/m1-12-tee-gpu-throughput.md) — the M1.12 spec (R1 / R3 / R4 / c6 / Q#1 / Q#2)
- [`docs/handoffs/2026-05-22-q3-4b-b8-mask-sweep.md`](2026-05-22-q3-4b-b8-mask-sweep.md) — Qwen3-4B B=8 bottleneck triage that drove M1.12

Microbench reference (real numbers reproducible from these):

- `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`
  - default knobs: variant=Q4B, B=8, n=2048, K=64 (B=8 set this session per request)
  - override via `GELO_BENCH_VARIANT`, `GELO_BENCH_B`, `GELO_BENCH_N`, `GELO_BENCH_MAX_TOKENS`
  - two test fns: `m1_12_r1_q1_microbench` (R1 + Q#1 + R3 verdict at B=1) and `m1_12_per_op_breakdown_prefill_decode` (prefill/decode split, baseline vs R3)

Reproduce the headline B=8 numbers:

```bash
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 GELO_BENCH_MAX_TOKENS=64 \
  cargo test -p gelo-gpu-wgpu --release \
  --test qwen3_m1_12_r1_q1_microbench -- --ignored --nocapture \
  m1_12_per_op_breakdown_prefill_decode
```

## Headline measurement (Qwen3-4B, B=8, n=2048, K=64, this session)

```
Phase    Variant    wall (s)    aggregate tok/s   per-tok-per-seq (ms)
─────────────────────────────────────────────────────────────────────
Prefill  baseline   192.13       85.3              —
Prefill  R3         179.59       91.4              —
Decode   baseline   304.90        1.68            595
Decode   R3         112.99        4.53            221      (= 2.70× speedup vs baseline)

Δ decode wall under R3:  −62.9 %
Δ tee:compute_logits:    −97.6 %     (195 831 ms → 4 726 ms; residual is the profile::time wrapper)
token-prefix match:      64 / 64     on real Qwen3-4B weights
```

R3 multiplier grows with batch (1.82× at B=1 K=32, 2.0× at B=1 K=64,
**2.70× at B=8 K=64**) because the in-TEE LM-head loop scales linearly
in B while the GPU LM-head matmul amortises GPU dispatch across the
`1+k=16` shield-aligned rows for free.

Memory entry with full numbers: `m1_12_r1_q1_microbench_findings.md`
(auto-memory MEMORY.md index).

## Refined perf-bucket roadmap — execution order

The roadmap was reordered this session based on the measured per-bucket
shares at B=8. Skip the ones already done; pick up at item 2.

### 1. (DONE in this session) c6 attack-suite scaffolding + R3 default-on

c6 condition variant is wired in `evals/aloepri-attacks/`. **The
attack-driver run is the remaining gate.** Engineering complete; need
to capture snapshots at the new LM-head shape and run the recovery
attacks against c2 baseline:

```bash
cargo run --release -p aloepri-attack-snapshot-runner -- \
    --condition c6 --max-prompts 64 --max-tokens 1 --engine gpu_fp16 \
    --output snapshots/qwen3-1.7b
# then the Python attack-driver run (anchor_ica / jade / jd / gram_error)
# against the c6 condition output vs the c2 baseline. Same comparison
# methodology as the c5 gate. ~3 days.
```

Acceptance: attack accuracy on c6 within sample-noise of c2. If c6
flags, revert the default and re-design (LM-head shape is 37× wider
than QKV — known to scale recovery surface).

### 2. Batched GPU attention kernel (the R1.4 lever)

Current bottleneck (post-R3, B=8 n_kv ≈ 2 100): `tee:attn_cached_inplace_many`
at **49.7 % of decode wall** (58.3 s of 113 s). At prefill the
`tee:attn_inplace_many` analogue is 11.6 %.

**Engineering**: substrate already exposes `engine.fused_attention_batched`.
Missing: per-sequence right-padding + causal-mask construction +
dispatch wire-up in the two decoder-block call sites
(`decoder_block_batched` for prefill, `decoder_block_cached_batched`
for decode). Estimate **2–3 days** per the prior handoff (R1.4).

**Impact estimate**: if the bucket goes to zero on the GPU, ~50 %
decode wall reduction on top of R3 (i.e. another 1.8–2× on top of
the 2.7× R3 already gives at B=8). Best-case Qwen3-4B B=8 decode
~55 s for 64 × 8 = 512 tokens → ~9 tok/s aggregate.

**Validation**: existing parity tests in `decoder_parity.rs` cover the
mask round-trip; need new tests that the batched GPU attention path
matches the in-TEE reference per-sequence.

### 3. bf16 mask GEMM on GPU — prefill bandwidth-contention lever

Current bottleneck (B=8 prefill, n=2048): `gelo:mask_unapply` 24.5 %
(45 s) + `gelo:mask_apply` 14.9 % (27 s) — **39 % of prefill wall on
CPU DDR5**, contending with GPU matmul on the same UMA bus.

**Engineering**: 1–2 weeks. OpenBLAS `cblas_sbgemm` is the 1-day path
but pulls in the BLAS dep on the offload side; hand-roll over wgpu
compute shader is cleaner. The `bf16_mask_gemm_skipped` memory's
"~10 % TTFT, gain shrinks after HD₃" estimate was at B=1 — at B=8 the
bucket scales linearly so the share **and** the win are larger
(~25–30 %).

**Impact estimate**: ~25–30 % prefill wall reduction on iGPU UMA via
unbussing the CPU-side FWHT bandwidth. On dGPU PCIe the win is
structural in a different way — frees CPU thread occupancy and removes
the FWHT memory-bandwidth ceiling on the host.

**Order interaction with bucket 4 (R4)**: shipping bucket 3 first
collapses R4's payoff (no CPU mask bucket to overlap with). Decide
order based on the dGPU timeline — see bucket 4.

### 4. Async pipelining (M1.12 R4) — DECIDE BEFORE STARTING

Plan-estimated 25–30 % prefill wall via overlapping CPU mask (layer
N+1) with GPU matmul (layer N).

**On iGPU UMA**: best case ~15 %. CPU and GPU share the same DDR5
bus; overlap doesn't reduce total bytes moved.

**On production dGPU PCIe**: genuinely valuable — PCIe DMA + GPU
matmul are physically separate from CPU mask FWHT.

**Open Q#2 (gate before starting)**: does RADV actually overlap wgpu
submissions, or serialise them under the queue? Spike-measure first
(half day) — if it serialises, R4 is dead on this iGPU. Plan calls
this out explicitly.

**Order decision rule (preserve in the plan)**:

- If production dGPU is < 3 months away → **skip R4 on iGPU**; revisit
  on dGPU. Save 5–8 days of substrate refactor that returns 15 %
  marginal at best on iGPU.
- If dGPU is the long tail (≥ 6 months) → **ship R4 on iGPU** for
  dev-velocity. Q#2 spike first; commit to R4 only if RADV overlaps.

### 5. Varlen / chunked / continuous batching

Per-sequence orchestration improvements. **Zero win in the current
bench** (identical-length prompts). In production extraction with
variable chunk lengths: ~10–30 % per-prompt wall.

**Components**:
- **Varlen mask** — skip pad rows in mask FWHT + GPU matmul when
  prompts have different lengths. Substrate API change (`(B, n_b, d)`
  ragged instead of `(B, n_max, d)` stacked). ~1–2 weeks.
- **Chunked prefill** — interleave chunks of one prompt with decode of
  others. Orchestrator-level. Latency-shaping, not throughput.
- **Continuous batching** — replace finished sequences without
  draining. Throughput-oriented; helps end-to-end serving latency.

**Order**: parallel track, ideally picked up by whichever crate-team
owns LightRAG orchestrator (`crates/lightrag-private/`). Doesn't move
the bench but is the right production lever.

### 6. Slalom-additive hybrid for linear projections — R&D milestone

Per `private_llm_inference_round_3` research memo. Splits each masked
offload into a plaintext-public matmul + an additive secret
correction; the public matmul rides any commodity-optimised inference
kernel.

**Potential**: 40–60 % wall reduction at prefill AND decode if it works.

**Engineering**: multi-week protocol-level redesign + AloePri-class
attack-suite re-validation (the additive correction shape isn't
covered by existing GELO §3.2 bounds).

**Order**: separate milestone, attack-validation gated. Highest
ceiling, lowest confidence. Don't put on the M1.12-extension critical
path; pre-spike via Python sim first.

### 7. Production dGPU substrate bring-up (M5.9)

Strix Halo iGPU UMA is the architectural ceiling for buckets 2 / 3 /
4. SEV-SNP + VFIO discrete GPU lifts it — HBM ~3 TB/s vs DDR5 ~80
GB/s, ~40× memory-bandwidth ceiling. New floor: PCIe DMA (~30 GB/s
realised) on the offload round-trip.

**Order**: hand off to M5.9 hardware bring-up. Re-run the microbench
+ the per-op breakdown bench there before re-prioritising — many
"iGPU-only" wins will look different, and several "iGPU-blocked"
levers (textbook batched-prefill scaling, async pipelining) become
genuinely large.

## Suggested skills for the next session

- **`code-review`** before the bucket-2 (batched GPU attention) PR
  lands — substrate parity is load-bearing for the M1.12 win to ride
  to production.
- **`verify`** after each bucket lands — confirm the
  per-op-breakdown microbench shows the expected bucket reduction
  before claiming the win.
- **`grill-with-docs`** before flipping default-on after bucket 2
  (the in-TEE attention is the last "GELO §3.2 sensitive-layer
  exclusion" defence; lifting it to GPU under permutation is a
  threat-model change worth grilling).
- **`diagnose`** if Q#2 (RADV serialisation spike) goes against R4.

## Memory tail (load-bearing for the next session)

Auto-memory entries created or updated this session:

- `m1_12_r1_q1_microbench_findings.md` — R1 needs `malloc_trim(0)` to
  show RSS drop; `tee:compute_logits` is 46–58 % of decode wall on
  real weights; R3 lands −45 % wall at B=1 / −63 % wall at B=8.
- (existing) `hd3_mask_landed.md`, `qwen3_4b_perf_2026_05_20.md`,
  `private_llm_inference_round_3.md` — round-3 perf research.

Pinned conventions worth re-reading:

- `CLAUDE.md` §"HTML docs" — public docs (gelo-llm.html etc.) avoid
  plan-time aliases (M1.11, Option A, v1, etc.). Re-baseline any new
  HTML section through that lens.
- `feedback_html_docs_design_not_code.md` — HTML lead with design +
  measured result, not dev-log.

## What I'd start with on day 1

1. Read `docs/plans/m1-12-tee-gpu-throughput.md` (the M1.12 spec).
2. Run the c6 capture + attack-driver (item 1's remaining gate).
   Either passes and the R3 default is retroactively validated, or it
   flags and the engineering is reverted.
3. Concurrently / next: bucket 2 (batched GPU attention kernel). Look
   at `gelo_protocol::engine::fused_attention_batched` for the
   existing entry point; check what's missing in
   `decoder_block_batched` / `decoder_block_cached_batched`.
