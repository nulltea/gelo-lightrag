#!/usr/bin/env bash
# Run a command inside the gelo-attack container with the repo mounted
# and the GPU passed through. Usage:
#   evals/aloepri-attacks/run-in-container.sh python3 -m pytest tests
#   evals/aloepri-attacks/run-in-container.sh bash        # interactive
#
# The repo mounts at /work; the HF weight cache mounts so in-container
# GELO/transformers reuse the host's download. Cargo target is kept on a
# named volume so host (Vulkan) and container (CUDA) builds don't clobber
# each other's artifacts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${GELO_ATTACK_IMAGE:-gelo-attack}"

exec podman run --rm -it \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    -v "${REPO_ROOT}:/work:z" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface:z" \
    -v gelo-attack-cargo-target:/work/target-container \
    -e CARGO_TARGET_DIR=/work/target-container \
    -w /work \
    "${IMAGE}" "$@"
