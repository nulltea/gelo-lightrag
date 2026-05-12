#!/usr/bin/env bash
# Build a CVM-bootable Ubuntu 24.04 image with `gelo-snp-runner` baked in.
#
# Output: $OUT_DIR/gelo-cvm-image.qcow2 (thin image, ~50 MB; weights are
# fetched on first boot by the gelo-fetch-weights systemd unit).
#
# Intended environment:
#   - any x86_64 Linux host with virt-customize, qemu-img, and the
#     `cargo` toolchain installed.
#   - the runner is built with `--release --features snp,mock` so the
#     same image works in T2 (SNP_MODE=mock) and T3 (SNP_MODE=production).
#   - SEV-SNP host kernel + OVMF must be installed *separately* on the
#     host before launching this image in T3 — see deploy/sev-snp/hetzner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/target/cvm-image}"
BASE_IMAGE_URL="${BASE_IMAGE_URL:-https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img}"
BASE_IMAGE="${OUT_DIR}/noble-base.qcow2"
TARGET_IMAGE="${OUT_DIR}/gelo-cvm-image.qcow2"

echo "==> output dir: $OUT_DIR"
mkdir -p "$OUT_DIR"

echo "==> building gelo-snp-runner (release, both attestation paths)"
(cd "$REPO_ROOT" && cargo build -p gelo-snp-runner --release --features snp,mock)

if [[ ! -f "$BASE_IMAGE" ]]; then
    echo "==> fetching Ubuntu 24.04 base image"
    curl -L --fail "$BASE_IMAGE_URL" -o "$BASE_IMAGE.partial"
    mv "$BASE_IMAGE.partial" "$BASE_IMAGE"
fi

echo "==> copying base image to $TARGET_IMAGE"
qemu-img convert -O qcow2 "$BASE_IMAGE" "$TARGET_IMAGE"
qemu-img resize "$TARGET_IMAGE" 10G

OVERLAY_DIR="$REPO_ROOT/deploy/sev-snp/cvm-image/overlay"

# virt-customize injects the runner binary + overlay + systemd units into
# the image. The `--firstboot-command` hook creates the `gelo` service
# user; the runner systemd unit references it.
echo "==> applying overlay via virt-customize"
sudo virt-customize -a "$TARGET_IMAGE" \
    --upload "$REPO_ROOT/target/release/gelo-snp-runner:/usr/local/bin/gelo-snp-runner" \
    --copy-in "$OVERLAY_DIR/etc:/" \
    --run-command "useradd --system --no-create-home --shell /usr/sbin/nologin gelo || true" \
    --run-command "install -d -o gelo -g gelo -m 0750 /var/lib/gelo-snp" \
    --run-command "systemctl enable gelo-snp-runner.service gelo-fetch-weights.service" \
    --selinux-relabel

echo "==> done: $TARGET_IMAGE"
echo "    size: $(du -h "$TARGET_IMAGE" | cut -f1)"
echo ""
echo "Next: launch in regular QEMU (T2) via deploy/sev-snp/sim/run-cvm-sim.sh"
echo "      launch on real EPYC (T3) via deploy/sev-snp/hetzner/launch-cvm.sh"
