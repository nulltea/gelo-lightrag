#!/usr/bin/env bash
# Run the IMA-EmbedRow attack harness on the AMD Strix Halo iGPU via
# the aloepri-ima-trainer container.
#
# Forwards all CLI args to the IMA driver. The HF cache, repo, and
# /tmp are bind-mounted at the *same* paths inside the container so
# host-side absolute paths in CLI args (e.g.
# `--plain /home/timo/.cache/huggingface/path-2-aloepri/...`) resolve.
#
# Set IMA_DRIVER to switch driver script:
#   IMA_DRIVER=run_ima_embedrow_attacks.py            (default, single-key paper-faithful)
#   IMA_DRIVER=run_ima_embedrow_attacks_multikey.py   (multi-key paper-faithful)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
IMA_CKPT_DIR="${IMA_CKPT_DIR:-$HOME/.cache/aloepri-ima-checkpoints}"
RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video | cut -d: -f3)"

mkdir -p "$IMA_CKPT_DIR"

# If the caller didn't pass --paper-checkpoint-dir explicitly, we
# append it here so it points at the host-side $HOME-derived dir
# we just mkdir'd + mounted. Container has HOME unset under
# `--user 1000:1000`, so a Path.home() default inside Python would
# resolve to "/" — which user 1000 can't write to.
has_ckpt_flag=0
for arg in "$@"; do
    if [[ "$arg" == "--paper-checkpoint-dir" || "$arg" == --paper-checkpoint-dir=* ]]; then
        has_ckpt_flag=1
        break
    fi
done

extra_args=()
if [[ $has_ckpt_flag -eq 0 ]]; then
    extra_args+=(--paper-checkpoint-dir "$IMA_CKPT_DIR")
fi

exec docker run --rm \
    --device /dev/dri --device /dev/kfd \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --user "$(id -u):$(id -g)" \
    --shm-size 16G \
    -v "$REPO_DIR:$REPO_DIR" \
    -v "$HF_CACHE:$HF_CACHE" \
    -v "$IMA_CKPT_DIR:$IMA_CKPT_DIR" \
    -v "/tmp:/tmp" \
    -e HF_HOME="$HF_CACHE" \
    -e HOME="$HOME" \
    -e ROCBLAS_USE_HIPBLASLT=1 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -w "$REPO_DIR" \
    aloepri-ima-trainer:latest \
    python3 "evals/aloepri-attacks/m2_7/${IMA_DRIVER:-run_ima_embedrow_attacks.py}" "${extra_args[@]}" "$@"
