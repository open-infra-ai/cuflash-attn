# syntax=docker/dockerfile:1
# cuflash Docker image.
#
# Multi-stage build: a devel stage compiles the library, tests, and benchmarks;
# the final stage is a slim CUDA runtime image carrying only the built
# artifacts (no toolchain, source tree, or build intermediates).

ARG CUDA_VERSION=12.4.1

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Ubuntu 22.04 ships CMake 3.22 (>= the 3.20 the presets require) and a
# C++17-capable GCC, so no third-party CMake download is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Build for the full supported architecture range so the image runs from
# Volta (sm_70) through Hopper (sm_90).
RUN cmake --preset release \
      -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
    && cmake --build --preset release --parallel "$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

LABEL maintainer="cuflash Team"
LABEL description="cuflash: from-scratch CUDA FlashAttention — scalar to WMMA tensor-core forward, FlashDecoding/Split-KV, Roofline analysis"

WORKDIR /workspace/cuflash

# Bring over only what is needed to run: the shared library, the test and
# benchmark binaries, and the public headers.
COPY --from=builder /src/build/release/libcuflash_attn.so ./
COPY --from=builder /src/build/release/cuflash_attn_tests ./
COPY --from=builder /src/build/release/cuflash_attn_bench ./
COPY --from=builder /src/build/release/cuflash_attn_api_smoke ./
COPY --from=builder /src/include ./include

ENV LD_LIBRARY_PATH=/workspace/cuflash:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]
