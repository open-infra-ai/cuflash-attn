// Data Type Support Tests
// Feature: cuflash, Property 6: 数据类型支持

#include <gtest/gtest.h>
#if CUFLASH_ENABLE_RAPIDCHECK
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuflash/flash_attention.h"

namespace cuflash {
namespace test {

// Test FP16 forward pass produces results close to FP32
// (FP16 backward is now supported - see FP16Backward tests)
TEST(DTypeTest, FP16ForwardMatchesFP32) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create FP32 inputs
    std::vector<float> h_Q_f32(qkv_size), h_K_f32(qkv_size), h_V_f32(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_f32[i] = dist(gen);
        h_K_f32[i] = dist(gen);
        h_V_f32[i] = dist(gen);
    }

    // Convert to FP16
    std::vector<half> h_Q_f16(qkv_size), h_K_f16(qkv_size), h_V_f16(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_f16[i] = __float2half(h_Q_f32[i]);
        h_K_f16[i] = __float2half(h_K_f32[i]);
        h_V_f16[i] = __float2half(h_V_f32[i]);
    }

    // FP32 computation
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_f32, *d_L_f32;
    cudaMalloc(&d_Q_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_K_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_V_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_O_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_L_f32, l_size * sizeof(float));

    cudaMemcpy(d_Q_f32, h_Q_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f32, h_K_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f32, h_V_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32, batch_size,
                                       num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_O_f32(qkv_size);
    cudaMemcpy(h_O_f32.data(), d_O_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // FP16 computation
    half *d_Q_f16, *d_K_f16, *d_V_f16, *d_O_f16;
    float* d_L_f16;
    cudaMalloc(&d_Q_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_K_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_V_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_O_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_L_f16, l_size * sizeof(float));

    cudaMemcpy(d_Q_f16, h_Q_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f16, h_K_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f16, h_V_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);

    err = flash_attention_forward(d_Q_f16, d_K_f16, d_V_f16, d_O_f16, d_L_f16, batch_size,
                                  num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<half> h_O_f16(qkv_size);
    cudaMemcpy(h_O_f16.data(), d_O_f16, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);

    // Compare FP16 and FP32 results (allow larger tolerance for FP16)
    float max_diff = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_O_f32[i] - __half2float(h_O_f16[i]));
        max_diff = std::max(max_diff, diff);
    }

    // FP16 has lower precision, allow 1e-2 tolerance
    EXPECT_LT(max_diff, 1e-2f) << "Max diff between FP32 and FP16: " << max_diff;

    cudaFree(d_Q_f32);
    cudaFree(d_K_f32);
    cudaFree(d_V_f32);
    cudaFree(d_O_f32);
    cudaFree(d_L_f32);
    cudaFree(d_Q_f16);
    cudaFree(d_K_f16);
    cudaFree(d_V_f16);
    cudaFree(d_O_f16);
    cudaFree(d_L_f16);
}

// Test FP16 backward pass produces finite gradients
TEST(DTypeTest, FP16BackwardBasic) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create FP16 inputs
    std::vector<half> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size), h_dO(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = __float2half(dist(gen));
        h_K[i] = __float2half(dist(gen));
        h_V[i] = __float2half(dist(gen));
        h_dO[i] = __float2half(dist(gen));
    }

    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    float* d_L;
    half *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(half));
    cudaMalloc(&d_K, qkv_size * sizeof(half));
    cudaMalloc(&d_V, qkv_size * sizeof(half));
    cudaMalloc(&d_O, qkv_size * sizeof(half));
    cudaMalloc(&d_L, l_size * sizeof(float));
    cudaMalloc(&d_dO, qkv_size * sizeof(half));
    cudaMalloc(&d_dQ, qkv_size * sizeof(half));
    cudaMalloc(&d_dK, qkv_size * sizeof(half));
    cudaMalloc(&d_dV, qkv_size * sizeof(half));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);

    // Run FP16 forward
    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    // Run FP16 backward
    err = flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV, batch_size,
                                   num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    // Verify gradients are finite
    std::vector<half> h_dQ(qkv_size), h_dK(qkv_size), h_dV(qkv_size);
    cudaMemcpy(h_dQ.data(), d_dQ, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK.data(), d_dK, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV.data(), d_dV, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < qkv_size; i++) {
        EXPECT_TRUE(std::isfinite(__half2float(h_dQ[i])))
            << "dQ at index " << i << " is not finite";
        EXPECT_TRUE(std::isfinite(__half2float(h_dK[i])))
            << "dK at index " << i << " is not finite";
        EXPECT_TRUE(std::isfinite(__half2float(h_dV[i])))
            << "dV at index " << i << " is not finite";
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    cudaFree(d_dO);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
}

// Test head_dim=128 (the largest supported dimension)
TEST(DTypeTest, HeadDim128) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 128;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    std::vector<float> h_O(qkv_size), h_L(l_size);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Check output is finite
    for (size_t i = 0; i < qkv_size; i++) {
        EXPECT_TRUE(std::isfinite(h_O[i])) << "Output at index " << i << " is not finite";
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

TEST(DTypeTest, FP16BackwardSuccess) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<half> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size), h_dO(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = __float2half(dist(gen));
        h_K[i] = __float2half(dist(gen));
        h_V[i] = __float2half(dist(gen));
        h_dO[i] = __float2half(dist(gen));
    }

    half *d_Q, *d_K, *d_V, *d_O;
    float* d_L;
    half *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(half));
    cudaMalloc(&d_K, qkv_size * sizeof(half));
    cudaMalloc(&d_V, qkv_size * sizeof(half));
    cudaMalloc(&d_O, qkv_size * sizeof(half));
    cudaMalloc(&d_L, l_size * sizeof(float));
    cudaMalloc(&d_dO, qkv_size * sizeof(half));
    cudaMalloc(&d_dQ, qkv_size * sizeof(half));
    cudaMalloc(&d_dK, qkv_size * sizeof(half));
    cudaMalloc(&d_dV, qkv_size * sizeof(half));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    err = flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV, batch_size,
                                   num_heads, seq_len, head_dim, scale, false, 0);
    EXPECT_EQ(err, FlashAttentionError::SUCCESS);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    cudaFree(d_dO);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
}

// Test FP16 backward matches FP32 backward (relaxed tolerance)
TEST(DTypeTest, FP16BackwardMatchesFP32) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create FP32 inputs
    std::vector<float> h_Q_f32(qkv_size), h_K_f32(qkv_size), h_V_f32(qkv_size), h_dO_f32(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_f32[i] = dist(gen);
        h_K_f32[i] = dist(gen);
        h_V_f32[i] = dist(gen);
        h_dO_f32[i] = dist(gen);
    }

    // Convert to FP16
    std::vector<half> h_Q_f16(qkv_size), h_K_f16(qkv_size), h_V_f16(qkv_size), h_dO_f16(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_f16[i] = __float2half(h_Q_f32[i]);
        h_K_f16[i] = __float2half(h_K_f32[i]);
        h_V_f16[i] = __float2half(h_V_f32[i]);
        h_dO_f16[i] = __float2half(h_dO_f32[i]);
    }

    // FP32 forward + backward
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_f32, *d_L_f32;
    float *d_dO_f32, *d_dQ_f32, *d_dK_f32, *d_dV_f32;
    cudaMalloc(&d_Q_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_K_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_V_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_O_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_L_f32, l_size * sizeof(float));
    cudaMalloc(&d_dO_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_dQ_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_dK_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_dV_f32, qkv_size * sizeof(float));

    cudaMemcpy(d_Q_f32, h_Q_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f32, h_K_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f32, h_V_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO_f32, h_dO_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32, batch_size,
                                       num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    err = flash_attention_backward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32, d_dO_f32, d_dQ_f32,
                                   d_dK_f32, d_dV_f32, batch_size, num_heads, seq_len, head_dim,
                                   scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_dQ_f32(qkv_size), h_dK_f32(qkv_size), h_dV_f32(qkv_size);
    cudaMemcpy(h_dQ_f32.data(), d_dQ_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK_f32.data(), d_dK_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV_f32.data(), d_dV_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // FP16 forward + backward
    half *d_Q_f16, *d_K_f16, *d_V_f16, *d_O_f16;
    float* d_L_f16;
    half *d_dO_f16, *d_dQ_f16, *d_dK_f16, *d_dV_f16;
    cudaMalloc(&d_Q_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_K_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_V_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_O_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_L_f16, l_size * sizeof(float));
    cudaMalloc(&d_dO_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_dQ_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_dK_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_dV_f16, qkv_size * sizeof(half));

    cudaMemcpy(d_Q_f16, h_Q_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f16, h_K_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f16, h_V_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO_f16, h_dO_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);

    err = flash_attention_forward(d_Q_f16, d_K_f16, d_V_f16, d_O_f16, d_L_f16, batch_size,
                                  num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    err = flash_attention_backward(d_Q_f16, d_K_f16, d_V_f16, d_O_f16, d_L_f16, d_dO_f16, d_dQ_f16,
                                   d_dK_f16, d_dV_f16, batch_size, num_heads, seq_len, head_dim,
                                   scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<half> h_dQ_f16(qkv_size), h_dK_f16(qkv_size), h_dV_f16(qkv_size);
    cudaMemcpy(h_dQ_f16.data(), d_dQ_f16, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK_f16.data(), d_dK_f16, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV_f16.data(), d_dV_f16, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);

    // Compare FP16 vs FP32 gradients
    float max_diff_dQ = 0.0f, max_diff_dK = 0.0f, max_diff_dV = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_dQ_f32[i] - __half2float(h_dQ_f16[i]));
        max_diff_dQ = std::max(max_diff_dQ, diff);
        diff = std::abs(h_dK_f32[i] - __half2float(h_dK_f16[i]));
        max_diff_dK = std::max(max_diff_dK, diff);
        diff = std::abs(h_dV_f32[i] - __half2float(h_dV_f16[i]));
        max_diff_dV = std::max(max_diff_dV, diff);
    }

    EXPECT_LT(max_diff_dQ, 1e-2f) << "Max dQ diff: " << max_diff_dQ;
    EXPECT_LT(max_diff_dK, 1e-2f) << "Max dK diff: " << max_diff_dK;
    EXPECT_LT(max_diff_dV, 1e-2f) << "Max dV diff: " << max_diff_dV;

    cudaFree(d_Q_f32);
    cudaFree(d_K_f32);
    cudaFree(d_V_f32);
    cudaFree(d_O_f32);
    cudaFree(d_L_f32);
    cudaFree(d_dO_f32);
    cudaFree(d_dQ_f32);
    cudaFree(d_dK_f32);
    cudaFree(d_dV_f32);
    cudaFree(d_Q_f16);
    cudaFree(d_K_f16);
    cudaFree(d_V_f16);
    cudaFree(d_O_f16);
    cudaFree(d_L_f16);
    cudaFree(d_dO_f16);
    cudaFree(d_dQ_f16);
    cudaFree(d_dK_f16);
    cudaFree(d_dV_f16);
}

// Test BF16 forward pass produces results close to FP32.
// BF16 keeps FP32's exponent range but only a 7-bit mantissa (coarser than
// FP16's 10-bit mantissa), so the tolerance is slightly wider than FP16's.
TEST(DTypeTest, BF16ForwardMatchesFP32) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create FP32 inputs
    std::vector<float> h_Q_f32(qkv_size), h_K_f32(qkv_size), h_V_f32(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_f32[i] = dist(gen);
        h_K_f32[i] = dist(gen);
        h_V_f32[i] = dist(gen);
    }

    // Convert to BF16
    std::vector<__nv_bfloat16> h_Q_bf16(qkv_size), h_K_bf16(qkv_size), h_V_bf16(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_bf16[i] = __float2bfloat16(h_Q_f32[i]);
        h_K_bf16[i] = __float2bfloat16(h_K_f32[i]);
        h_V_bf16[i] = __float2bfloat16(h_V_f32[i]);
    }

    // FP32 computation
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_f32, *d_L_f32;
    cudaMalloc(&d_Q_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_K_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_V_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_O_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_L_f32, l_size * sizeof(float));

    cudaMemcpy(d_Q_f32, h_Q_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f32, h_K_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f32, h_V_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32, batch_size,
                                       num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_O_f32(qkv_size);
    cudaMemcpy(h_O_f32.data(), d_O_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // BF16 computation
    __nv_bfloat16 *d_Q_bf16, *d_K_bf16, *d_V_bf16, *d_O_bf16;
    float* d_L_bf16;
    cudaMalloc(&d_Q_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_L_bf16, l_size * sizeof(float));

    cudaMemcpy(d_Q_bf16, h_Q_bf16.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_bf16, h_K_bf16.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_bf16, h_V_bf16.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    err = flash_attention_forward(d_Q_bf16, d_K_bf16, d_V_bf16, d_O_bf16, d_L_bf16, batch_size,
                                  num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_O_bf16(qkv_size);
    cudaMemcpy(h_O_bf16.data(), d_O_bf16, qkv_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Compare BF16 and FP32 results
    float max_diff = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_O_f32[i] - __bfloat162float(h_O_bf16[i]));
        max_diff = std::max(max_diff, diff);
    }

    // BF16 mantissa is coarser than FP16; allow 2e-2 tolerance
    EXPECT_LT(max_diff, 2e-2f) << "Max diff between FP32 and BF16: " << max_diff;

    cudaFree(d_Q_f32);
    cudaFree(d_K_f32);
    cudaFree(d_V_f32);
    cudaFree(d_O_f32);
    cudaFree(d_L_f32);
    cudaFree(d_Q_bf16);
    cudaFree(d_K_bf16);
    cudaFree(d_V_bf16);
    cudaFree(d_O_bf16);
    cudaFree(d_L_bf16);
}

// Test BF16 backward pass produces finite gradients
TEST(DTypeTest, BF16BackwardFinite) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<__nv_bfloat16> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size), h_dO(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = __float2bfloat16(dist(gen));
        h_K[i] = __float2bfloat16(dist(gen));
        h_V[i] = __float2bfloat16(dist(gen));
        h_dO[i] = __float2bfloat16(dist(gen));
    }

    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
    float* d_L;
    __nv_bfloat16 *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_L, l_size * sizeof(float));
    cudaMalloc(&d_dO, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dQ, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dK, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dV, qkv_size * sizeof(__nv_bfloat16));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    err = flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV, batch_size,
                                   num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_dQ(qkv_size), h_dK(qkv_size), h_dV(qkv_size);
    cudaMemcpy(h_dQ.data(), d_dQ, qkv_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK.data(), d_dK, qkv_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV.data(), d_dV, qkv_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < qkv_size; i++) {
        EXPECT_TRUE(std::isfinite(__bfloat162float(h_dQ[i])))
            << "dQ at index " << i << " is not finite";
        EXPECT_TRUE(std::isfinite(__bfloat162float(h_dK[i])))
            << "dK at index " << i << " is not finite";
        EXPECT_TRUE(std::isfinite(__bfloat162float(h_dV[i])))
            << "dV at index " << i << " is not finite";
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
    cudaFree(d_dO);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
}

// Test BF16 backward matches FP32 backward (relaxed tolerance)
TEST(DTypeTest, BF16BackwardMatchesFP32) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create FP32 inputs
    std::vector<float> h_Q_f32(qkv_size), h_K_f32(qkv_size), h_V_f32(qkv_size), h_dO_f32(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_f32[i] = dist(gen);
        h_K_f32[i] = dist(gen);
        h_V_f32[i] = dist(gen);
        h_dO_f32[i] = dist(gen);
    }

    // Convert to BF16
    std::vector<__nv_bfloat16> h_Q_bf16(qkv_size), h_K_bf16(qkv_size), h_V_bf16(qkv_size),
        h_dO_bf16(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q_bf16[i] = __float2bfloat16(h_Q_f32[i]);
        h_K_bf16[i] = __float2bfloat16(h_K_f32[i]);
        h_V_bf16[i] = __float2bfloat16(h_V_f32[i]);
        h_dO_bf16[i] = __float2bfloat16(h_dO_f32[i]);
    }

    // FP32 forward + backward
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_f32, *d_L_f32;
    float *d_dO_f32, *d_dQ_f32, *d_dK_f32, *d_dV_f32;
    cudaMalloc(&d_Q_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_K_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_V_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_O_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_L_f32, l_size * sizeof(float));
    cudaMalloc(&d_dO_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_dQ_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_dK_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_dV_f32, qkv_size * sizeof(float));

    cudaMemcpy(d_Q_f32, h_Q_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f32, h_K_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f32, h_V_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO_f32, h_dO_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32, batch_size,
                                       num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    err = flash_attention_backward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32, d_dO_f32, d_dQ_f32,
                                   d_dK_f32, d_dV_f32, batch_size, num_heads, seq_len, head_dim,
                                   scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_dQ_f32(qkv_size), h_dK_f32(qkv_size), h_dV_f32(qkv_size);
    cudaMemcpy(h_dQ_f32.data(), d_dQ_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK_f32.data(), d_dK_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV_f32.data(), d_dV_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // BF16 forward + backward
    __nv_bfloat16 *d_Q_bf16, *d_K_bf16, *d_V_bf16, *d_O_bf16;
    float* d_L_bf16;
    __nv_bfloat16 *d_dO_bf16, *d_dQ_bf16, *d_dK_bf16, *d_dV_bf16;
    cudaMalloc(&d_Q_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_L_bf16, l_size * sizeof(float));
    cudaMalloc(&d_dO_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dQ_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dK_bf16, qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_dV_bf16, qkv_size * sizeof(__nv_bfloat16));

    cudaMemcpy(d_Q_bf16, h_Q_bf16.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_bf16, h_K_bf16.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_bf16, h_V_bf16.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO_bf16, h_dO_bf16.data(), qkv_size * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);

    err = flash_attention_forward(d_Q_bf16, d_K_bf16, d_V_bf16, d_O_bf16, d_L_bf16, batch_size,
                                  num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    err = flash_attention_backward(d_Q_bf16, d_K_bf16, d_V_bf16, d_O_bf16, d_L_bf16, d_dO_bf16,
                                   d_dQ_bf16, d_dK_bf16, d_dV_bf16, batch_size, num_heads, seq_len,
                                   head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_dQ_bf16(qkv_size), h_dK_bf16(qkv_size), h_dV_bf16(qkv_size);
    cudaMemcpy(h_dQ_bf16.data(), d_dQ_bf16, qkv_size * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK_bf16.data(), d_dK_bf16, qkv_size * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV_bf16.data(), d_dV_bf16, qkv_size * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);

    // Compare BF16 vs FP32 gradients (BF16 mantissa coarser than FP16)
    float max_diff_dQ = 0.0f, max_diff_dK = 0.0f, max_diff_dV = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_dQ_f32[i] - __bfloat162float(h_dQ_bf16[i]));
        max_diff_dQ = std::max(max_diff_dQ, diff);
        diff = std::abs(h_dK_f32[i] - __bfloat162float(h_dK_bf16[i]));
        max_diff_dK = std::max(max_diff_dK, diff);
        diff = std::abs(h_dV_f32[i] - __bfloat162float(h_dV_bf16[i]));
        max_diff_dV = std::max(max_diff_dV, diff);
    }

    EXPECT_LT(max_diff_dQ, 2e-2f) << "Max dQ diff: " << max_diff_dQ;
    EXPECT_LT(max_diff_dK, 2e-2f) << "Max dK diff: " << max_diff_dK;
    EXPECT_LT(max_diff_dV, 2e-2f) << "Max dV diff: " << max_diff_dV;

    cudaFree(d_Q_f32);
    cudaFree(d_K_f32);
    cudaFree(d_V_f32);
    cudaFree(d_O_f32);
    cudaFree(d_L_f32);
    cudaFree(d_dO_f32);
    cudaFree(d_dQ_f32);
    cudaFree(d_dK_f32);
    cudaFree(d_dV_f32);
    cudaFree(d_Q_bf16);
    cudaFree(d_K_bf16);
    cudaFree(d_V_bf16);
    cudaFree(d_O_bf16);
    cudaFree(d_L_bf16);
    cudaFree(d_dO_bf16);
    cudaFree(d_dQ_bf16);
    cudaFree(d_dK_bf16);
    cudaFree(d_dV_bf16);
}

#if CUFLASH_ENABLE_RAPIDCHECK
// Property test: FP16 results should be close to FP32
// Feature: cuflash, Property 6: 数据类型支持
// Validates: Requirements 7.4
RC_GTEST_PROP(DTypeProperty, FP16ClosesToFP32, ()) {
    int batch_size = 1;
    int num_heads = 1;
    int seq_len = *rc::gen::inRange(4, 33);
    int head_dim = *rc::gen::element(32, 64, 128);  // Include 128 for full coverage
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q_f32(qkv_size), h_K_f32(qkv_size), h_V_f32(qkv_size);
    std::vector<half> h_Q_f16(qkv_size), h_K_f16(qkv_size), h_V_f16(qkv_size);

    for (size_t i = 0; i < qkv_size; i++) {
        float val = *rc::gen::inRange(-100, 100) * 0.01f;
        h_Q_f32[i] = val;
        h_Q_f16[i] = __float2half(val);
        val = *rc::gen::inRange(-100, 100) * 0.01f;
        h_K_f32[i] = val;
        h_K_f16[i] = __float2half(val);
        val = *rc::gen::inRange(-100, 100) * 0.01f;
        h_V_f32[i] = val;
        h_V_f16[i] = __float2half(val);
    }

    // Allocate and run FP32
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_f32, *d_L_f32;
    cudaMalloc(&d_Q_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_K_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_V_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_O_f32, qkv_size * sizeof(float));
    cudaMalloc(&d_L_f32, l_size * sizeof(float));

    cudaMemcpy(d_Q_f32, h_Q_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f32, h_K_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f32, h_V_f32.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q_f32, d_K_f32, d_V_f32, d_O_f32, d_L_f32, batch_size,
                                       num_heads, seq_len, head_dim, scale, false, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_O_f32(qkv_size);
    cudaMemcpy(h_O_f32.data(), d_O_f32, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Allocate and run FP16
    half *d_Q_f16, *d_K_f16, *d_V_f16, *d_O_f16;
    float* d_L_f16;
    cudaMalloc(&d_Q_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_K_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_V_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_O_f16, qkv_size * sizeof(half));
    cudaMalloc(&d_L_f16, l_size * sizeof(float));

    cudaMemcpy(d_Q_f16, h_Q_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_f16, h_K_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_f16, h_V_f16.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice);

    err = flash_attention_forward(d_Q_f16, d_K_f16, d_V_f16, d_O_f16, d_L_f16, batch_size,
                                  num_heads, seq_len, head_dim, scale, false, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<half> h_O_f16(qkv_size);
    cudaMemcpy(h_O_f16.data(), d_O_f16, qkv_size * sizeof(half), cudaMemcpyDeviceToHost);

    // Compare
    float max_diff = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        float diff = std::abs(h_O_f32[i] - __half2float(h_O_f16[i]));
        max_diff = std::max(max_diff, diff);
    }
    RC_ASSERT(max_diff < 1e-2f);

    cudaFree(d_Q_f32);
    cudaFree(d_K_f32);
    cudaFree(d_V_f32);
    cudaFree(d_O_f32);
    cudaFree(d_L_f32);
    cudaFree(d_Q_f16);
    cudaFree(d_K_f16);
    cudaFree(d_V_f16);
    cudaFree(d_O_f16);
    cudaFree(d_L_f16);
}

#endif

}  // namespace test
}  // namespace cuflash
