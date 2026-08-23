#pragma once

#include <cuda_runtime.h>

#include "cuflash/export.h"
#include "cuflash/flash_attention.h"  // For FlashAttentionError

namespace cuflash {
namespace kernels {

// =============================================================================
// Tile-level Matrix Multiplication Operations
// =============================================================================
// These are fundamental primitives for FlashAttention's computation.
// All matrices are in GPU global memory.
//
// Template parameters:
//   M, N, K - Matrix dimensions (compile-time for kernel optimization)
//
// Performance note: These kernels use shared memory tiling and are optimized
// for the tile sizes used in FlashAttention (32, 64, 128).

// -----------------------------------------------------------------------------
// C = A @ B^T (Attention score computation)
// -----------------------------------------------------------------------------

/// Compute C = A @ B^T where A is MxK and B is NxK.
/// This is the primary operation for computing attention scores: S = Q @ K^T.
///
/// @param A       Input matrix A [M, K] in global memory
/// @param B       Input matrix B [N, K] in global memory
/// @param C       Output matrix C [M, N] in global memory
/// @param scale   Scaling factor applied to result (typically 1/sqrt(head_dim))
/// @param stream  CUDA stream
/// @return        SUCCESS on success, error code on failure
template<int M, int N, int K>
CUFLASH_EXPORT FlashAttentionError matmul_ABt(const float* A, const float* B, float* C, float scale,
                                              cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
// C = A @ B
// -----------------------------------------------------------------------------

/// Compute C = A @ B where A is MxK and B is KxN.
/// Used for computing attention output: O = P @ V.
template<int M, int N, int K>
CUFLASH_EXPORT FlashAttentionError matmul_AB(const float* A, const float* B, float* C, float scale,
                                             cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
// C += A @ B (Accumulate)
// -----------------------------------------------------------------------------

/// Compute C += A @ B where A is MxK and B is KxN.
/// Used for accumulating partial results across KV blocks.
template<int M, int N, int K>
CUFLASH_EXPORT FlashAttentionError matmul_AB_acc(const float* A, const float* B, float* C,
                                                 float scale, cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
// C = A^T @ B
// -----------------------------------------------------------------------------

/// Compute C = A^T @ B where A is KxM and B is KxN.
/// Used for computing gradients in backward pass.
template<int M, int N, int K>
CUFLASH_EXPORT FlashAttentionError matmul_AtB(const float* A, const float* B, float* C, float scale,
                                              cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cuflash
