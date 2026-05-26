#!/usr/bin/env bash
# M1.12+ HD₃ perf-roadmap validation sweep.
#
# Runs the `m1_12_sweep_cell` test across:
#   B=1, n ∈ {1538, 1790, 2561}, mask ∈ {auto, hd3}
#   B=8, n ∈ {192, 223, 320},   mask ∈ {auto, hd3}  (matched-aggregate vs B=1)
#   B=8, n = 2048,              mask ∈ {auto, hd3}  (long-n production-shape probe)
# K = 32 decode tokens for every cell.
# Variant pinned to Qwen3-4B (matches plan §3.3 attribution).
#
# Each cell rebuilds the executor from scratch so per-cell RSS is clean.
# Weights live in HF cache, so reload is fast after the first cell.
#
# Output:
#   bench-results/m1-12-hd3-perf-sweep-<timestamp>.log   (full per-cell dump)
#   bench-results/m1-12-hd3-perf-sweep-<timestamp>.tsv   (SWEEP_RESULT lines)
set -euo pipefail

cd "$(dirname "$0")/.."

ts="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="bench-results/m1-12-hd3-perf-sweep-${ts}.log"
TSV="bench-results/m1-12-hd3-perf-sweep-${ts}.tsv"
mkdir -p bench-results

export GELO_BENCH_VARIANT=4b
export GELO_BENCH_MAX_TOKENS=32

# Cell list: (B, n, mask)
cells=(
    "1 1538 auto"
    "1 1538 hd3"
    "1 1790 auto"
    "1 1790 hd3"
    "1 2561 auto"
    "1 2561 hd3"
    "8 192 auto"
    "8 192 hd3"
    "8 223 auto"
    "8 223 hd3"
    "8 320 auto"
    "8 320 hd3"
    "8 2048 auto"
    "8 2048 hd3"
)

echo "=== M1.12+ HD₃ perf-roadmap sweep — Qwen3-4B, K=32, ${#cells[@]} cells ===" | tee "$LOG"
echo "log: $LOG" | tee -a "$LOG"
echo "tsv: $TSV" | tee -a "$LOG"
date | tee -a "$LOG"
echo | tee -a "$LOG"

# Build once so per-cell wall doesn't include compile time.
echo "+ cargo build --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench" | tee -a "$LOG"
cargo build --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench 2>&1 | tee -a "$LOG"

cell_idx=0
for cell in "${cells[@]}"; do
    cell_idx=$((cell_idx + 1))
    read -r b n mask <<<"$cell"
    echo | tee -a "$LOG"
    echo "─── cell $cell_idx / ${#cells[@]}: B=$b n=$n mask=$mask ───" | tee -a "$LOG"
    date | tee -a "$LOG"

    GELO_BENCH_B=$b \
    GELO_BENCH_N=$n \
    GELO_SWEEP_MASK=$mask \
        cargo test --release -p gelo-gpu-wgpu \
            --test qwen3_m1_12_r1_q1_microbench \
            -- --ignored --nocapture m1_12_sweep_cell 2>&1 | tee -a "$LOG"
done

echo | tee -a "$LOG"
echo "=== summary ===" | tee -a "$LOG"
grep -E '^SWEEP_RESULT' "$LOG" | tee "$TSV"

echo | tee -a "$LOG"
date | tee -a "$LOG"
echo "done. tsv at: $TSV"
