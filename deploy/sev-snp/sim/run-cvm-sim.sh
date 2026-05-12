#!/usr/bin/env bash
# Launch the CVM image in regular QEMU/KVM (no SEV-SNP host needed).
#
# This is the Tier-2 simulator: the *same* image production runs, booted
# in an ordinary guest VM. The binary inside detects SNP_MODE=mock and
# uses the bundled mock issuer; a shim character device pretends to be
# /dev/sev-guest. Verifies OS boundary, systemd unit lifecycle, model-
# weight loading, mock-`/dev/sev-guest` path, and the full ingest →
# attest → embed → response loop with whatever GPU is exposed (or
# llvmpipe software adapter if none).
#
# Usage:
#   deploy/sev-snp/sim/run-cvm-sim.sh [--snp-mode mock|production] [--port 7878]
#
# The image is expected to live at target/cvm-image/gelo-cvm-image.qcow2;
# build it first with deploy/sev-snp/cvm-image/setup-cvm-image.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
IMAGE="${IMAGE:-$REPO_ROOT/target/cvm-image/gelo-cvm-image.qcow2}"
SNP_MODE="${SNP_MODE:-mock}"
PORT="${PORT:-7878}"
SSH_PORT="${SSH_PORT:-2222}"
MEM="${MEM:-4G}"
CPUS="${CPUS:-4}"
QEMU="${QEMU:-qemu-system-x86_64}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --snp-mode) SNP_MODE="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --image) IMAGE="$2"; shift 2;;
        --mem) MEM="$2"; shift 2;;
        --cpus) CPUS="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

if [[ ! -f "$IMAGE" ]]; then
    echo "image not found: $IMAGE" >&2
    echo "build it first with deploy/sev-snp/cvm-image/setup-cvm-image.sh" >&2
    exit 1
fi

# Inject SNP_MODE into the VM via the cloud-init datasource (cidata ISO).
# In Tier 2 we override /etc/gelo-snp/runner.env from outside so each run
# can pick its mode without baking it in.
CIDATA="$(mktemp -d)"
trap 'rm -rf "$CIDATA"' EXIT
cat > "$CIDATA/meta-data" <<EOF
instance-id: gelo-snp-sim
local-hostname: gelo-snp-sim
EOF
cat > "$CIDATA/user-data" <<EOF
#cloud-config
write_files:
  - path: /etc/gelo-snp/runner.env
    permissions: '0644'
    content: |
      SNP_MODE=$SNP_MODE
runcmd:
  - systemctl restart gelo-snp-runner.service
EOF
genisoimage -output "$CIDATA/cidata.iso" -volid cidata -joliet -rock \
    "$CIDATA/user-data" "$CIDATA/meta-data" >/dev/null

echo "==> launching CVM-sim:"
echo "    image:    $IMAGE"
echo "    SNP_MODE: $SNP_MODE"
echo "    HTTP:     http://127.0.0.1:$PORT"
echo "    SSH:      ssh -p $SSH_PORT ubuntu@127.0.0.1"

exec "$QEMU" \
    -enable-kvm \
    -m "$MEM" -smp "$CPUS" \
    -drive file="$IMAGE",if=virtio,format=qcow2 \
    -drive file="$CIDATA/cidata.iso",if=virtio,format=raw,readonly=on \
    -nic user,model=virtio,hostfwd=tcp:127.0.0.1:$PORT-:7878,hostfwd=tcp:127.0.0.1:$SSH_PORT-:22 \
    -nographic \
    -serial mon:stdio
