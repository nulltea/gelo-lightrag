#!/usr/bin/env bash
# Smoke-comparison runner. Runs llama-cli on two GGUFs with identical
# prompt/seed/max-tokens, diffs the model output stream.
#
# Expected: outputs are bit-identical when the matrix-Γ kernel branch
# is correct and Γ = Diag(γ_q).
#
# Usage:
#   ./diff_smoke_outputs.sh <baseline.gguf> <matrix.gguf> [prompt]

set -euo pipefail

BASELINE="${1:?baseline gguf path}"
MATRIX="${2:?matrix gguf path}"
PROMPT="${3:-The capital of France is}"
N=12

CLI="${LLAMA_CLI:-/home/timo/repos/private-rag-path-2/vendor/llama.cpp/build-cpu/bin/llama-cli}"
if [[ -n "${USE_DOCKER:-}" ]]; then
    CLI_CMD=(docker run --rm --device /dev/dri
             -v /tmp:/host-tmp:ro
             aloepri-llama-server:option-c)
else
    CLI_CMD=("$CLI")
fi

run_one() {
    local gguf="$1"
    local label="$2"
    echo "=== $label  ($gguf) ==="
    if [[ -n "${USE_DOCKER:-}" ]]; then
        local container_path="${gguf/#\/tmp\//\/host-tmp\/}"
        docker run --rm --device /dev/dri \
            -v /tmp:/host-tmp:ro \
            aloepri-llama-server:option-c llama-cli \
            -m "$container_path" \
            -p "$PROMPT" \
            -n "$N" --temp 0 --seed 42 -ngl 999 -no-cnv 2>&1 \
            | tee "/tmp/smoke-${label}.log"
    else
        "$CLI" \
            -m "$gguf" \
            -p "$PROMPT" \
            -n "$N" --temp 0 --seed 42 -no-cnv 2>&1 \
            | tee "/tmp/smoke-${label}.log"
    fi
}

run_one "$BASELINE" "scalar"
run_one "$MATRIX"   "matrix"

echo ""
echo "=== diff (full stdout — focus on text after 'system_info') ==="
extract() {
    # llama-cli prints prompt then generated tokens then perf stats. The
    # interesting region starts where 'main: ' setup ends and the prompt is
    # echoed. We strip stderr-style log lines (timestamp-prefixed) and
    # any 'llama_perf' tail.
    awk '
        /^llama_perf/ {exit}
        /^[A-Za-z_].*: / && /n_ctx|n_batch|seed|info|build/ {next}
        {print}
    ' "/tmp/smoke-${1}.log"
}
if diff -u <(extract scalar) <(extract matrix); then
    echo "MATCH — scalar and matrix outputs are identical"
else
    echo "DIFF FOUND — scalar ≠ matrix"
fi
