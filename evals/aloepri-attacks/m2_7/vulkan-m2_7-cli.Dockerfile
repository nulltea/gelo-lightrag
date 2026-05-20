# M2.7 patched Vulkan build with both llama-server AND llama-cli.
# Built on top of the existing vulkan-m2_7.Dockerfile — adds llama-cli
# to the final-stage image so we can run greedy-decode comparisons for
# the Option C matrix-Γ smoke test.
#
# Build:
#   docker build \
#     -f evals/aloepri-attacks/m2_7/vulkan-m2_7-cli.Dockerfile \
#     -t aloepri-llama-server:option-c-cli \
#     vendor/llama.cpp
ARG UBUNTU_VERSION=26.04

FROM ubuntu:$UBUNTU_VERSION AS build
RUN apt update && apt install -y git build-essential cmake wget xz-utils
RUN apt install -y libssl-dev curl \
    libxcb-xinput0 libxcb-xinerama0 libxcb-cursor-dev libvulkan-dev glslc spirv-headers

WORKDIR /app
COPY . .
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_TESTS=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON && \
    cmake --build build --config Release -j$(nproc)
RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;
RUN mkdir -p /app/full && cp build/bin/* /app/full

FROM ubuntu:$UBUNTU_VERSION AS base
RUN apt-get update \
    && apt-get install -y libgomp1 curl libvulkan1 mesa-vulkan-drivers \
       libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama-cli /app/
COPY --from=build /app/full/llama-server /app/
COPY --from=build /app/full/llama-completion /app/
WORKDIR /app
ENV LLAMA_ARG_HOST=0.0.0.0
LABEL org.aloepri.flavor="option-c-matrix-gamma"
LABEL org.aloepri.source="vendor/llama.cpp with matrix-Γ Qwen3 q_norm/k_norm patch"
