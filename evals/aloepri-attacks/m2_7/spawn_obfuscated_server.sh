#!/usr/bin/env bash
#
# Spawn a GPU-backed patched llama-server against an AloePri GGUF.
# Used by token-stream, quality, and tensor-dump capture harnesses.
#
# Bound only to localhost; named `aloepri-m2_7-server` so it doesn't
# collide with the persistent `llama-swap` container.
#
# Memory note: 9 GB iGPU residency at fp32. Stop with
# `docker stop aloepri-m2_7-server` when done.

set -euo pipefail

OBF_GGUF="${OBF_GGUF:-/home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-alg2-fp32.gguf}"
PORT="${PORT:-8061}"
CONTAINER="${CONTAINER:-aloepri-m2_7-server}"
IMAGE="${IMAGE:-aloepri-llama-server:m2_7-attn-output}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CTX="${CTX:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-1024}"
TENSOR_FILTER="${TENSOR_FILTER:-}"
TENSOR_DUMP_PATH="${TENSOR_DUMP_PATH:-}"

if [[ ! -f "$OBF_GGUF" ]]; then
    echo "obfuscated GGUF not found at $OBF_GGUF — pass OBF_GGUF=... to override" >&2
    exit 1
fi

# Refuse to start if a container of this name already exists.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "container ${CONTAINER} already exists — docker rm -f ${CONTAINER} first" >&2
    exit 2
fi

OBF_DIR=$(dirname "$OBF_GGUF")
OBF_NAME=$(basename "$OBF_GGUF")

# /dev/dri/renderD* is owned by group `render`; /dev/kfd is needed by
# ROCm-backed images. Missing either permission can silently fall back
# to CPU, so the harness always passes both groups and all present GPU
# devices. This keeps capture/quality runs from depending on hand-written
# docker invocations.
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video  | cut -d: -f3)

DOCKER_DEVICES=(--device /dev/dri)
if [[ -e /dev/kfd ]]; then
    DOCKER_DEVICES+=(--device /dev/kfd)
fi

SERVER_ARGS=(
    -m "/models/$OBF_NAME"
    -ngl 999 -np 1 --flash-attn "$FLASH_ATTN"
    -c "$CTX" --ubatch-size "$UBATCH_SIZE"
    --host 0.0.0.0 --port 8080
)
if [[ -n "$TENSOR_FILTER" ]]; then
    SERVER_ARGS+=(--tensor-filter "$TENSOR_FILTER")
fi
if [[ -n "$TENSOR_DUMP_PATH" ]]; then
    SERVER_ARGS+=(--tensor-dump-path "$TENSOR_DUMP_PATH")
fi

VOLUME_ARGS=(-v "$OBF_DIR:/models:ro")
if [[ -n "$TENSOR_DUMP_PATH" ]]; then
    # Assumes dump path like /dump/m2_7_dump.bin. Host side is controlled by
    # DUMP_DIR, defaulting to /tmp for compatibility with capture_hidden_states.py.
    DUMP_DIR="${DUMP_DIR:-/tmp}"
    VOLUME_ARGS+=(-v "$DUMP_DIR:/dump")
fi

echo "[M2.7 spawn] image=$IMAGE port=$PORT model=$OBF_NAME render_gid=$RENDER_GID video_gid=$VIDEO_GID flash_attn=$FLASH_ATTN"
if [[ -n "$TENSOR_FILTER" ]]; then
    echo "[M2.7 spawn] tensor_filter=$TENSOR_FILTER tensor_dump_path=$TENSOR_DUMP_PATH dump_dir=${DUMP_DIR:-}"
fi
exec docker run --rm -d \
    --name "$CONTAINER" \
    --user 1000:1000 \
    --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
    -p "127.0.0.1:$PORT:8080" \
    "${VOLUME_ARGS[@]}" \
    "${DOCKER_DEVICES[@]}" \
    "$IMAGE" \
    "${SERVER_ARGS[@]}"
