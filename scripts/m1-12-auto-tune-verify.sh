#!/usr/bin/env bash
# Verification sweep for the 2026-05-26 Auto threshold re-tune
# (HD3_AUTO_MAX_PAD_RATIO 7/5 → 8/5).
#
# Three cells:
#   - B=8 n=320  mask=auto: was DCT-IV pre-tune (cell 11 = 24.97 s);
#                           should now pick HD₃ and match the HD3-forced
#                           cell-12 wall (~24.42 s) — a ~2 % win.
#   - B=1 n=2561 mask=auto: was DCT-IV pre-tune (cell 5 = 32.48 s);
#                           should now pick HD₃ and match cell-6 (32.19 s)
#                           — a ~1 % win.
#   - B=8 n=2048 mask=auto: was DCT-IV pre-tune (cell 13 = 187.31 s).
#                           Pad ratio 1.99 > 1.6, so Auto SHOULD STILL
#                           pick DCT-IV. Used as the "no-regression"
#                           guard.
set -euo pipefail
cd "$(dirname "$0")/.."

ts="$(date +%Y-%m-%d_%H-%M-%S)"
LOG="bench-results/m1-12-auto-tune-verify-${ts}.log"
TSV="bench-results/m1-12-auto-tune-verify-${ts}.tsv"
mkdir -p bench-results

export GELO_BENCH_VARIANT=4b
export GELO_BENCH_MAX_TOKENS=32

cells=(
    "8 320 auto"
    "1 2561 auto"
    "8 2048 auto"
)

echo "=== Auto threshold re-tune verification — Qwen3-4B K=32 ===" | tee "$LOG"
echo "log: $LOG" | tee -a "$LOG"
echo "tsv: $TSV" | tee -a "$LOG"
date | tee -a "$LOG"
echo | tee -a "$LOG"

echo "+ cargo build --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench" | tee -a "$LOG"
cargo build --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench 2>&1 | tee -a "$LOG"

idx=0
for cell in "${cells[@]}"; do
    idx=$((idx + 1))
    read -r b n mask <<<"$cell"
    echo | tee -a "$LOG"
    echo "─── verify cell $idx / ${#cells[@]}: B=$b n=$n mask=$mask ───" | tee -a "$LOG"
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
