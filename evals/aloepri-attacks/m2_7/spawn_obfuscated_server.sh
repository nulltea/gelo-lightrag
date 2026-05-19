#!/usr/bin/env bash
#
# Spawn a llama.cpp:server-vulkan container against the §05 obfuscated
# Qwen3 1.7B GGUF. Used by `capture_token_streams.py` for TFMA / SDA.
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
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-vulkan}"

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

echo "[M2.7 spawn] image=$IMAGE port=$PORT model=$OBF_NAME"
exec docker run --rm -d \
    --name "$CONTAINER" \
    -p "127.0.0.1:$PORT:8080" \
    -v "$OBF_DIR:/models:ro" \
    --device /dev/dri \
    "$IMAGE" \
    -m "/models/$OBF_NAME" \
    -ngl 999 -np 1 --flash-attn on \
    -c 4096 --ubatch-size 1024 \
    --host 0.0.0.0 --port 8080
