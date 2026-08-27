// Numerical Stability Tests
// Feature: cuflash, Property 4: 数值稳定性

#include <gtest/gtest.h>
#if CUFLASH_ENABLE_RAPIDCHECK
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#endif
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuflash/flash_attention.h"

namespace cuflash {
namespace test {

// Test with large values (should not overflow)
TEST(NumericalStabilityTest, LargeValues) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    // Use large values
    std::vector<float> h_Q(qkv_size, 100.0f);
    std::vector<float> h_K(qkv_size, 100.0f);
    std::vector<float> h_V(qkv_size, 100.0f);
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

    // Check no NaN or Inf
    for (size_t i = 0; i < qkv_size; i++) {
        EXPECT_FALSE(std::isnan(h_O[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(h_O[i])) << "Inf at index " << i;
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

// Test with small values (should not underflow)
TEST(NumericalStabilityTest, SmallValues) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    // Use small values
    std::vector<float> h_Q(qkv_size, 1e-6f);
    std::vector<float> h_K(qkv_size, 1e-6f);
    std::vector<float> h_V(qkv_size, 1e-6f);
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

    for (size_t i = 0; i < qkv_size; i++) {
        EXPECT_FALSE(std::isnan(h_O[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(h_O[i])) << "Inf at index " << i;
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

// Test with mixed extreme values
TEST(NumericalStabilityTest, MixedExtremeValues) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);

    // Mix of large and small values
    for (size_t i = 0; i < qkv_size; i++) {
        if (i % 3 == 0)
            h_Q[i] = 50.0f;
        else if (i % 3 == 1)
            h_Q[i] = -50.0f;
        else
            h_Q[i] = 0.001f;

        h_K[i] = h_Q[i];
        h_V[i] = h_Q[i];
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

    for (size_t i = 0; i < qkv_size; i++) {
        EXPECT_FALSE(std::isnan(h_O[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(h_O[i])) << "Inf at index " << i;
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

#if CUFLASH_ENABLE_RAPIDCHECK
// Property test: No NaN or Inf for any valid input
// Feature: cuflash, Property 4: 数值稳定性
// Validates: Requirements 4.4, 8.3
RC_GTEST_PROP(NumericalStabilityProperty, NoNaNOrInf, ()) {
    int batch_size = 1;
    int num_heads = 1;
    int seq_len = *rc::gen::inRange(1, 33);
    int head_dim = *rc::gen::element(32, 64);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);

    // Generate values with varying magnitudes
    for (size_t i = 0; i < qkv_size; i++) {
        int magnitude = *rc::gen::inRange(-3, 3);  // 10^-3 to 10^3
        float base = *rc::gen::inRange(-100, 100) * 0.01f;
        h_Q[i] = base * std::pow(10.0f, magnitude);

        magnitude = *rc::gen::inRange(-3, 3);
        base = *rc::gen::inRange(-100, 100) * 0.01f;
        h_K[i] = base * std::pow(10.0f, magnitude);

        magnitude = *rc::gen::inRange(-3, 3);
        base = *rc::gen::inRange(-100, 100) * 0.01f;
        h_V[i] = base * std::pow(10.0f, magnitude);
    }

    std::vector<float> h_O(qkv_size);

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
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < qkv_size; i++) {
        RC_ASSERT(!std::isnan(h_O[i]));
        RC_ASSERT(!std::isinf(h_O[i]));
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

#endif

}  // namespace test
}  // namespace cuflash
