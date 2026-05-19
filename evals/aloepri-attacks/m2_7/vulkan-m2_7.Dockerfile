# M2.7 patched Vulkan server build.
#
# Builds llama.cpp with the tensor-dump hook (HiddenState +
# AttnScore eval-callback wired into the server) applied on top
# of upstream master. Tag the produced image as
# `aloepri-llama-server:m2_7` so it doesn't collide with the
# persistent `llama-swap` image.
#
# Prerequisites (one-time):
#   git submodule update --init --recursive vendor/llama.cpp
#   bash evals/aloepri-attacks/m2_7/apply-patches.sh
#
# (The submodule is pinned at the clean upstream commit; the
# tensor-dump patch lives at evals/aloepri-attacks/m2_7/patches/
# and is applied to the working tree only. Re-run --revert before
# updating the submodule pin.)
#
# Build:
#   docker build \
#     -f evals/aloepri-attacks/m2_7/vulkan-m2_7.Dockerfile \
#     -t aloepri-llama-server:m2_7 \
#     vendor/llama.cpp
#
# Run (HiddenState only):
#   docker run --rm --name aloepri-m2_7-server \
#     -p 127.0.0.1:8061:8080 \
#     -v /home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b:/models:ro \
#     --device /dev/dri \
#     aloepri-llama-server:m2_7 \
#     -m /models/keymat-h128-pi-noise-alg2-fp32.gguf \
#     -ngl 999 -np 1 --flash-attn on -c 4096 \
#     --host 0.0.0.0 --port 8080
#
# Run (AttnScore — flash-attn MUST be off so Q·Kᵀ materialises):
#   …same as above but with `--flash-attn off`.
#
# Only changes from upstream `.devops/vulkan.Dockerfile`:
#   1. Build directory is the M2.7-patched tree (caller passes
#      `vendor/llama.cpp` as the build context).
#   2. Final stage labels the image so its lineage is auditable.

ARG UBUNTU_VERSION=26.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt update && apt install -y git build-essential cmake wget xz-utils
RUN apt install -y libssl-dev curl \
    libxcb-xinput0 libxcb-xinerama0 libxcb-cursor-dev libvulkan-dev glslc spirv-headers

WORKDIR /app
COPY . .

# Same build flags as upstream `.devops/vulkan.Dockerfile`, just on
# the patched source.
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_TESTS=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full

FROM ubuntu:$UBUNTU_VERSION AS base

RUN apt-get update \
    && apt-get install -y libgomp1 curl libvulkan1 mesa-vulkan-drivers \
       libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0
COPY --from=build /app/full/llama-server /app
WORKDIR /app

LABEL org.aloepri.flavor="m2_7-tensor-capture"
LABEL org.aloepri.source="vendor/llama.cpp with patch in tools/server/server-tensor-capture.{h,cpp}"
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
