#!/usr/bin/env bash
# Run the e2e smoke against a *locally cargo-built* gelo-snp-runner.
#
# Skips the VM entirely — useful for CI and for fast iteration before
# rebuilding the CVM image. The only thing this misses vs. the VM-sim
# path is the OS-boundary / systemd-unit / cloud-init surface, which is
# exercised by deploy/sev-snp/sim/run-cvm-sim.sh + e2e-smoke.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
PORT="${PORT:-7878}"

cleanup() {
    if [[ -n "${RUNNER_PID:-}" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
        kill "$RUNNER_PID" 2>/dev/null || true
        wait "$RUNNER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "==> building gelo-snp-runner"
cargo build -q -p gelo-snp-runner --features snp,mock --manifest-path "$REPO_ROOT/Cargo.toml"

echo "==> launching runner (SNP_MODE=mock, port $PORT)"
SNP_MODE=mock RUST_LOG=warn,gelo_snp_runner=info \
    "$REPO_ROOT/target/debug/gelo-snp-runner" &
RUNNER_PID=$!

# Wait up to 10s for the listener to come up.
for _ in $(seq 1 20); do
    if curl -fsS -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/health" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

"$REPO_ROOT/deploy/sev-snp/sim/e2e-smoke.sh" "http://127.0.0.1:$PORT"
