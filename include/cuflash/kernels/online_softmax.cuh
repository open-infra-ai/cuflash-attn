#pragma once

#include <cuda_runtime.h>

#include "cuflash/export.h"
#include "cuflash/flash_attention.h"  // For FlashAttentionError

namespace cuflash {
namespace kernels {

// =============================================================================
// Online Softmax Operations
// =============================================================================
// Numerically stable streaming softmax computation.
//
// Online softmax processes data in blocks without materializing the full
// softmax matrix. It maintains two state values:
//   - m: running maximum
//   - l: running sum of exp(x - m)
//
// This enables O(N) memory for attention computation.

// -----------------------------------------------------------------------------
// Low-level Primitives (user controls iteration)
// -----------------------------------------------------------------------------

/// Initialize online softmax state arrays.
/// Sets state_m to -infinity and state_l to 0.
///
/// @param state_m    Device array [rows] for running maximum values
/// @param state_l    Device array [rows] for running sum values
/// @param rows       Number of rows to initialize
/// @param stream     CUDA stream
/// @return           SUCCESS on success, error code on failure
CUFLASH_EXPORT FlashAttentionError online_softmax_init(float* state_m, float* state_l, int rows,
                                                       cudaStream_t stream = nullptr);

/// Update online softmax state with a new block's statistics.
/// For each row: new_m = max(old_m, block_max), then rescale sums.
///
/// @param block_max  Device array [rows] - maximum values in current block
/// @param block_sum  Device array [rows] - sum of exp(x - block_max) in block
/// @param state_m    Device array [rows] - running maximum (inout)
/// @param state_l    Device array [rows] - running sum (inout)
/// @param rows       Number of rows
/// @param stream     CUDA stream
/// @return           SUCCESS on success, error code on failure
CUFLASH_EXPORT FlashAttentionError online_softmax_update(const float* block_max,
                                                         const float* block_sum, float* state_m,
                                                         float* state_l, int rows,
                                                         cudaStream_t stream = nullptr);

/// Finalize online softmax and produce outputs.
/// Computes logsumexp = m + log(l) and normalizer = 1/l.
///
/// @param state_m    Device array [rows] - final maximum values
/// @param state_l    Device array [rows] - final sum values
/// @param logsumexp  Device array [rows] - output: m + log(l)
/// @param normalizer Device array [rows] - output: 1/l
/// @param rows       Number of rows
/// @param stream     CUDA stream
/// @return           SUCCESS on success, error code on failure
CUFLASH_EXPORT FlashAttentionError online_softmax_finalize(const float* state_m,
                                                           const float* state_l, float* logsumexp,
                                                           float* normalizer, int rows,
                                                           cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
// High-level Convenience API
// -----------------------------------------------------------------------------

/// Compute online softmax in one call.
/// Processes input in blocks of size `block_size` and produces softmax output.
///
/// @param input      Device array [rows, cols] - input scores
/// @param output     Device array [rows, cols] - softmax output
/// @param logsumexp  Device array [rows] - logsumexp values
/// @param rows       Number of rows
/// @param cols       Number of columns
/// @param block_size Tile size for internal processing
/// @param stream     CUDA stream
/// @return           SUCCESS on success, error code on failure
CUFLASH_EXPORT FlashAttentionError online_softmax_forward(const float* input, float* output,
                                                          float* logsumexp, int rows, int cols,
                                                          int block_size,
                                                          cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
// Warp-level Primitives (for use in custom kernels)
// -----------------------------------------------------------------------------

/// Warp-level max reduction.
/// All threads in warp call with their value; all return the max.
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

/// Warp-level sum reduction.
/// All threads in warp call with their value; all return the sum.
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

}  // namespace kernels
}  // namespace cuflash
