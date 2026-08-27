// Causal Mask Tests
// Feature: cuflash, Property 5: 因果掩码正确性

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

// Test that causal mask correctly prevents attending to future positions
// Property 5: For any causal attention, position i's output should only depend on positions 0..i
TEST(CausalMaskTest, FutureIndependence) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    // Compute output with original K, V
    std::vector<float> h_O1(qkv_size), h_L1(l_size);

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
                                       head_dim, scale, true, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O1.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Modify K, V at future positions (last half of sequence)
    std::vector<float> h_K2 = h_K, h_V2 = h_V;
    int modify_start = seq_len / 2;
    for (int i = modify_start; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            h_K2[i * head_dim + d] = dist(gen) * 10.0f;  // Different values
            h_V2[i * head_dim + d] = dist(gen) * 10.0f;
        }
    }

    // Compute output with modified K, V
    std::vector<float> h_O2(qkv_size), h_L2(l_size);

    cudaMemcpy(d_K, h_K2.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V2.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len, head_dim,
                                  scale, true, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O2.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Check that positions before modify_start are unchanged
    for (int i = 0; i < modify_start; i++) {
        for (int d = 0; d < head_dim; d++) {
            float diff = std::abs(h_O1[i * head_dim + d] - h_O2[i * head_dim + d]);
            EXPECT_LT(diff, 1e-5f)
                << "Position " << i << " should not be affected by future changes";
        }
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

#if CUFLASH_ENABLE_RAPIDCHECK
// Property test: Causal mask correctness
// Feature: cuflash, Property 5: 因果掩码正确性
// Validates: Requirements 5.1
RC_GTEST_PROP(CausalMaskProperty, FutureIndependenceProperty, ()) {
    int batch_size = 1;
    int num_heads = 1;
    int seq_len = *rc::gen::inRange(4, 33);
    int head_dim = *rc::gen::element(32, 64);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = *rc::gen::inRange(-100, 100) * 0.01f;
        h_K[i] = *rc::gen::inRange(-100, 100) * 0.01f;
        h_V[i] = *rc::gen::inRange(-100, 100) * 0.01f;
    }

    // Position to test
    int test_pos = *rc::gen::inRange(0, seq_len - 1);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_O1(qkv_size);
    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, true, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O1.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Modify K, V at positions after test_pos
    std::vector<float> h_K2 = h_K, h_V2 = h_V;
    for (int i = test_pos + 1; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            h_K2[i * head_dim + d] = *rc::gen::inRange(-1000, 1000) * 0.01f;
            h_V2[i * head_dim + d] = *rc::gen::inRange(-1000, 1000) * 0.01f;
        }
    }

    cudaMemcpy(d_K, h_K2.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V2.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_O2(qkv_size);
    err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len, head_dim,
                                  scale, true, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O2.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Output at test_pos should be unchanged
    for (int d = 0; d < head_dim; d++) {
        float diff = std::abs(h_O1[test_pos * head_dim + d] - h_O2[test_pos * head_dim + d]);
        RC_ASSERT(diff < 1e-5f);
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
