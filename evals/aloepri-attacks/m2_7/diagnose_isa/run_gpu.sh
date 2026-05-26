#!/usr/bin/env bash
# Wrap a probe script execution in the aloepri-ima-trainer ROCm container,
# so torch.cuda.is_available() == True and ridge solves auto-route to
# rocSOLVER. Mirrors run_in_gpu_container.sh's mounts/env.
set -euo pipefail

REPO=/home/timo/repos/private-rag-path-2
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video | cut -d: -f3)"

exec docker run --rm \
    --device /dev/dri --device /dev/kfd \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --user "$(id -u):$(id -g)" \
    --shm-size 8G \
    -v "$REPO:$REPO" \
    -v "$HF_CACHE:$HF_CACHE" \
    -v "/tmp:/tmp" \
    -e HF_HOME="$HF_CACHE" \
    -e HOME="$HOME" \
    -e ROCBLAS_USE_HIPBLASLT=1 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -w "$REPO" \
    aloepri-ima-trainer:latest \
    python3 "$@"
