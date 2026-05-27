---
type: handoff
status: current
created: 2026-05-26
updated: 2026-05-26
tags: [mask, perf]
---

# Handoff — 2026-05-26 — Mask instrumentation + Auto threshold tune

Five patches landed in this round, all in
`crates/gelo-protocol/src/{hd3.rs, dct4.rs, mask.rs, sim.rs}`.
Net prefill wall delta at the production Qwen3-4B B=8 n=2048
shape: +4.3 % vs the 4-day-old pre-instrumentation baseline
(179.6 s → 187.3 s), **within day-to-day variance** (no clean
A/B; see the roadmap §3.1 #2 — variance sweep gating).

## Patch list

| # | Patch | What | Why |
|---:|---|---|---|
| 1 | Per-family profile categories on PerSequence path | `sim.rs:1126, 1165` — replace flat `gelo:mask_apply` / `unapply` with `:hd3` / `:dct4` / `:haar` | Without this every other finding in the roadmap §1 is invisible |
| 2 | Threshold const dedup | `dct4.rs:352, 382` — use `crate::hd3::FWHT_RAYON_WORK_THRESHOLD` | One source of truth for tuning |
| 3 | `scale_inplace` fused into D₃ | `hd3.rs`, `dct4.rs` — `apply_diag_scaled_inplace[_slice]` replaces final scale pass | Eliminates one full-tensor pass per apply/unapply |
| 4 | Batched scratch reuse + slice mask kernels | `hd3.rs` + `dct4.rs` (new `apply_in_place_slice` / `unapply_in_place_slice`); `sim.rs` (new `per_seq_apply_scratch` HashMap, `unmask_per_sequence` consumes by-value) | Eliminates `to_owned + assign` per block in `build_per_sequence_masked`; ~400 GB allocator churn / long-n prefill |
| 5 | Auto threshold re-tune 7/5 → 8/5 | `mask.rs:147` — `HD3_AUTO_MAX_PAD_RATIO_NUM = 8` | Sweep confirmed HD₃ wins at pad ratios up to 1.59; old threshold of 1.4 under-picked it |

## Honest read on impact

Patches 1, 2, 5 are correctness / architectural — patch 1
unblocked all roadmap §1 measurement and is the round's real
headline. Patches 3 and 4 are bandwidth-savings cleanups whose
theoretical win (~5 s / ~3 % at long-n prefill) is below the
~7 % single-cell variance floor (roadmap §1.5). Patch 5 is
defensible on first-principles grounds: HD₃ measurably wins at
pad ratios up to 1.59 per the sweep, so the old 1.4 threshold
was sub-optimal.

**None of this round's wall-time deltas exceed the variance
floor on single-sample measurement.** The right framing:
architectural cleanup landed, instrumentation now shows the
real bucket attribution (roadmap §1.2 / §1.3), threshold is
correct for HD₃ at shapes where it wins; quantitative wall-time
wins pending the variance sweep (roadmap §3.1 #2) — which is
why every EV in the roadmap §4 buckets is gated on it.

## Tune-verify sweep

The post-tune verify run
(`bench-results/m1-12-auto-tune-verify-2026-05-26_08-42-00.{log,tsv}`)
covered three cells with the new 1.6 threshold:

| B | n | pad ratio | Auto family | prefill wall (s) | decode wall (s) |
|---:|---:|---:|---|---:|---:|
| 1 | 2561 | 1.59 | HD₃ | 31.92 | 21.43 |
| 8 | 320 | 1.56 | HD₃ | 24.22 | 26.82 |
| 8 | 2048 | 1.99 | DCT-IV | 174.92 | 55.08 |

Auto resolved correctly in all three cells. No regression vs
pre-tune at the DCT-IV shape (Auto kept DCT-IV at pad 1.99).
The 1.7-1.8 crossover region is unprobed; roadmap §3.1 #4
captures the follow-up.

## References

- [`docs/plans/gelo-llm-perf-roadmap.md`](../plans/gelo-llm-perf-roadmap.md) — the roadmap this patch round preceded
- `bench-results/m1-12-hd3-perf-sweep-2026-05-26_07-04-58.{log,tsv}` — the 14-cell sweep that drove the patch decisions
- `bench-results/m1-12-auto-tune-verify-2026-05-26_08-42-00.{log,tsv}` — post-tune verification (3 cells)
- [[m1_12_production_mask_is_dct4]] — 2026-05-26 sweep finding memory
- [[hd3_radix8_and_scratch_reuse]] — prior FWHT scratch-reuse work (related context)
