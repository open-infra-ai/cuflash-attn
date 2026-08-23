#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuflash/export.h"
#include "cuflash/flash_attention.h"  // For FlashAttentionError

namespace cuflash {
namespace kernels {

// =============================================================================
// Tile I/O Operations
// =============================================================================
// Load and store tiles between global memory and shared memory.
// These are fundamental primitives for FlashAttention's tiled computation.
//
// Template parameters:
//   BLOCK_ROWS, BLOCK_COLS - Tile dimensions (compile-time for optimization)
//
// All operations handle boundary conditions when dimensions are not divisible
// by block size. Elements outside bounds are zero-padded on load and ignored
// on store.

// -----------------------------------------------------------------------------
// Load Operations
// -----------------------------------------------------------------------------

/// Load a tile from global memory to a buffer (FP32).
/// The destination buffer should be pre-allocated with size >= BLOCK_ROWS * BLOCK_COLS.
///
/// @param src           Source tensor in global memory [max_rows, max_cols]
/// @param dst           Destination buffer [BLOCK_ROWS, BLOCK_COLS]
/// @param row_start     Starting row in source tensor
/// @param col_start     Starting column in source tensor
/// @param max_rows      Total rows in source tensor
/// @param max_cols      Total columns in source tensor
/// @param src_stride    Row stride of source tensor (typically max_cols)
/// @param stream        CUDA stream
/// @return              SUCCESS on success, error code on failure
template<int BLOCK_ROWS, int BLOCK_COLS>
CUFLASH_EXPORT FlashAttentionError load_tile(const float* src, float* dst, int row_start,
                                             int col_start, int max_rows, int max_cols,
                                             int src_stride, cudaStream_t stream = nullptr);

/// Load a tile from global memory to a buffer (FP16).
/// Converts half to float during load for numerical stability.
template<int BLOCK_ROWS, int BLOCK_COLS>
CUFLASH_EXPORT FlashAttentionError load_tile(const half* src, float* dst, int row_start,
                                             int col_start, int max_rows, int max_cols,
                                             int src_stride, cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
// Store Operations
// -----------------------------------------------------------------------------

/// Store a tile from a buffer to global memory (FP32).
///
/// @param src           Source buffer [BLOCK_ROWS, BLOCK_COLS]
/// @param dst           Destination tensor in global memory [max_rows, max_cols]
/// @param row_start     Starting row in destination tensor
/// @param col_start     Starting column in destination tensor
/// @param max_rows      Total rows in destination tensor
/// @param max_cols      Total columns in destination tensor
/// @param dst_stride    Row stride of destination tensor (typically max_cols)
/// @param stream        CUDA stream
/// @return              SUCCESS on success, error code on failure
template<int BLOCK_ROWS, int BLOCK_COLS>
CUFLASH_EXPORT FlashAttentionError store_tile(const float* src, float* dst, int row_start,
                                              int col_start, int max_rows, int max_cols,
                                              int dst_stride, cudaStream_t stream = nullptr);

/// Store a tile from a buffer to global memory (FP16).
/// Converts float to half during store.
template<int BLOCK_ROWS, int BLOCK_COLS>
CUFLASH_EXPORT FlashAttentionError store_tile(const float* src, half* dst, int row_start,
                                              int col_start, int max_rows, int max_cols,
                                              int dst_stride, cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
// Round-trip Test Helper
// -----------------------------------------------------------------------------

/// Test helper: load a tile, then store it back.
/// Useful for verifying tile I/O correctness.
template<int BLOCK_ROWS, int BLOCK_COLS>
CUFLASH_EXPORT FlashAttentionError load_store_tile_roundtrip(const float* src, float* dst,
                                                             int row_start, int col_start,
                                                             int max_rows, int max_cols, int stride,
                                                             cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cuflash
