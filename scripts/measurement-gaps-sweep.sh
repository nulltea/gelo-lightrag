#!/usr/bin/env bash
# Measurement gaps sweep — addresses §3.1 of the GELO-LLM perf roadmap.
#
# Cells:
#   B=8 n=3500 auto          long-n HD₃ shape (pad 4096 ratio 1.17) — §3.1 #1
#   B=8 n=2400 auto          crossover probe (pad 4096 ratio 1.70) — §3.1 #4
#   B=8 n=2400 hd3           crossover probe vs HD₃-forced
#   B=1 n=2561 auto          B=1 attention capture re-run — §3.1 #3
# K = 32 decode tokens per cell. Variant Qwen3-4B.
#
# B=1 attention capture relies on the dump_sweep_buckets patch that adds
# singular `tee:attn_inplace` and `tee:attn_cached_inplace` to the bucket
# list (landed in the same commit as this script).
#
# Output:
#   bench-results/measurement-gaps-<timestamp>.log   (full per-cell dump)
#   bench-results/measurement-gaps-<timestamp>.tsv   (SWEEP_RESULT lines)
set -euo pipefail

cd "$(dirname "$0")/.."

ts="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="bench-results/measurement-gaps-${ts}.log"
TSV="bench-results/measurement-gaps-${ts}.tsv"
mkdir -p bench-results

export GELO_BENCH_VARIANT=4b
export GELO_BENCH_MAX_TOKENS=32

cells=(
    "8 3500 auto"
    "8 2400 auto"
    "8 2400 hd3"
    "1 2561 auto"
)

echo "=== Measurement-gaps sweep — Qwen3-4B, K=32, ${#cells[@]} cells ===" | tee "$LOG"
echo "log: $LOG" | tee -a "$LOG"
echo "tsv: $TSV" | tee -a "$LOG"
date | tee -a "$LOG"
echo | tee -a "$LOG"

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
