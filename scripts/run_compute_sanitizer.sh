#!/usr/bin/env bash
# Run the GPU test suite under NVIDIA compute-sanitizer.
#
# Requires a CUDA-capable GPU and compute-sanitizer (shipped with the CUDA
# toolkit). This is the primary automated check for out-of-bounds shared/global
# memory access, data races, uninitialized reads, and divergent __syncthreads —
# none of which the host-only CI runners can exercise.
#
# Usage: ./scripts/run_compute_sanitizer.sh [build-dir]
#   build-dir defaults to "build". The test binary must already be built.

set -euo pipefail

BUILD_DIR="${1:-build}"
SANITIZER="${COMPUTE_SANITIZER:-compute-sanitizer}"
TEST_BIN="$BUILD_DIR/cuflash_tests"

if ! command -v "$SANITIZER" >/dev/null 2>&1; then
    echo "error: $SANITIZER not found on PATH (install the CUDA toolkit)" >&2
    exit 1
fi
if [ ! -x "$TEST_BIN" ]; then
    echo "error: test binary not found at $TEST_BIN; build the tests first" >&2
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "error: no CUDA-capable GPU detected" >&2
    exit 1
fi

status=0
for tool in memcheck racecheck initcheck synccheck; do
    echo "==> compute-sanitizer --tool=$tool"
    # --error-exitcode 1 makes definite errors fail the run; potential hazards
    # (e.g. racecheck "hazard") are reported but do not fail on their own.
    if ! "$SANITIZER" --tool "$tool" --error-exitcode 1 "$TEST_BIN"; then
        echo "compute-sanitizer ($tool) reported errors" >&2
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "compute-sanitizer: all tools clean"
fi
exit "$status"
