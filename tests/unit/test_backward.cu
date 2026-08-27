// Backward Pass Tests
// Feature: cuflash, Property 2: 反向传播梯度等价性

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

// CPU reference implementation of attention backward pass
void reference_attention_backward(const std::vector<float>& Q, const std::vector<float>& K,
                                  const std::vector<float>& V, const std::vector<float>& dO,
                                  std::vector<float>& dQ, std::vector<float>& dK,
                                  std::vector<float>& dV, int batch_size, int num_heads,
                                  int seq_len, int head_dim, float scale, bool causal) {
    std::fill(dQ.begin(), dQ.end(), 0.0f);
    std::fill(dK.begin(), dK.end(), 0.0f);
    std::fill(dV.begin(), dV.end(), 0.0f);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int bh_offset = (b * num_heads + h) * seq_len * head_dim;

            // Compute attention matrix P
            std::vector<std::vector<float>> P(seq_len, std::vector<float>(seq_len));

            for (int i = 0; i < seq_len; i++) {
                float max_score = -INFINITY;
                std::vector<float> scores(seq_len);

                for (int j = 0; j < seq_len; j++) {
                    if (causal && j > i) {
                        scores[j] = -INFINITY;
                    } else {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            dot +=
                                Q[bh_offset + i * head_dim + d] * K[bh_offset + j * head_dim + d];
                        }
                        scores[j] = dot * scale;
                    }
                    max_score = std::max(max_score, scores[j]);
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < seq_len; j++) {
                    P[i][j] = scores[j] / sum_exp;
                }
            }

            // Compute dV = P^T @ dO
            for (int j = 0; j < seq_len; j++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int i = 0; i < seq_len; i++) {
                        sum += P[i][j] * dO[bh_offset + i * head_dim + d];
                    }
                    dV[bh_offset + j * head_dim + d] = sum;
                }
            }

            // Compute dP = dO @ V^T
            std::vector<std::vector<float>> dP(seq_len, std::vector<float>(seq_len));
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        sum += dO[bh_offset + i * head_dim + d] * V[bh_offset + j * head_dim + d];
                    }
                    dP[i][j] = sum;
                }
            }

            // Compute dS = P * (dP - D) where D_i = sum_j(P_ij * dP_ij)
            std::vector<std::vector<float>> dS(seq_len, std::vector<float>(seq_len));
            for (int i = 0; i < seq_len; i++) {
                float D_i = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    D_i += P[i][j] * dP[i][j];
                }
                for (int j = 0; j < seq_len; j++) {
                    if (causal && j > i) {
                        dS[i][j] = 0.0f;
                    } else {
                        dS[i][j] = P[i][j] * (dP[i][j] - D_i);
                    }
                }
            }

            // Compute dQ = dS @ K * scale
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        sum += dS[i][j] * K[bh_offset + j * head_dim + d];
                    }
                    dQ[bh_offset + i * head_dim + d] = sum * scale;
                }
            }

            // Compute dK = dS^T @ Q * scale
            for (int j = 0; j < seq_len; j++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int i = 0; i < seq_len; i++) {
                        sum += dS[i][j] * Q[bh_offset + i * head_dim + d];
                    }
                    dK[bh_offset + j * head_dim + d] = sum * scale;
                }
            }
        }
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

// Basic backward test
TEST(BackwardTest, BasicSmall) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size), h_dO(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
        h_dO[i] = dist(gen);
    }

    // Reference gradients
    std::vector<float> ref_dQ(qkv_size), ref_dK(qkv_size), ref_dV(qkv_size);
    reference_attention_backward(h_Q, h_K, h_V, h_dO, ref_dQ, ref_dK, ref_dV, batch_size, num_heads,
                                 seq_len, head_dim, scale, false);

    // GPU computation
    float *d_Q, *d_K, *d_V, *d_O, *d_L, *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));
    cudaMalloc(&d_dO, qkv_size * sizeof(float));
    cudaMalloc(&d_dQ, qkv_size * sizeof(float));
    cudaMalloc(&d_dK, qkv_size * sizeof(float));
    cudaMalloc(&d_dV, qkv_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    // Forward pass first
    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    // Backward pass
    err = flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV, batch_size,
                                   num_heads, seq_len, head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_dQ(qkv_size), h_dK(qkv_size), h_dV(qkv_size);
    cudaMemcpy(h_dQ.data(), d_dQ, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK.data(), d_dK, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV.data(), d_dV, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    float dQ_diff = max_abs_diff(h_dQ, ref_dQ);
    float dK_diff = max_abs_diff(h_dK, ref_dK);
    float dV_diff = max_abs_diff(h_dV, ref_dV);

    EXPECT_LT(dQ_diff, 1e-3f) << "dQ max diff: " << dQ_diff;
    EXPECT_LT(dK_diff, 1e-3f) << "dK max diff: " << dK_diff;
    EXPECT_LT(dV_diff, 1e-3f) << "dV max diff: " << dV_diff;

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

TEST(BackwardTest, CausalMultiHeadHeadDim128) {
    const int batch_size = 1;
    const int num_heads = 2;
    const int seq_len = 16;
    const int head_dim = 128;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size), h_dO(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
        h_dO[i] = dist(gen);
    }

    std::vector<float> ref_dQ(qkv_size), ref_dK(qkv_size), ref_dV(qkv_size);
    reference_attention_backward(h_Q, h_K, h_V, h_dO, ref_dQ, ref_dK, ref_dV, batch_size, num_heads,
                                 seq_len, head_dim, scale, true);

    float *d_Q, *d_K, *d_V, *d_O, *d_L, *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));
    cudaMalloc(&d_dO, qkv_size * sizeof(float));
    cudaMalloc(&d_dQ, qkv_size * sizeof(float));
    cudaMalloc(&d_dK, qkv_size * sizeof(float));
    cudaMalloc(&d_dV, qkv_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, true, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    err = flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV, batch_size,
                                   num_heads, seq_len, head_dim, scale, true, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_dQ(qkv_size), h_dK(qkv_size), h_dV(qkv_size);
    cudaMemcpy(h_dQ.data(), d_dQ, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK.data(), d_dK, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV.data(), d_dV, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_LT(max_abs_diff(h_dQ, ref_dQ), 2e-3f);
    EXPECT_LT(max_abs_diff(h_dK, ref_dK), 2e-3f);
    EXPECT_LT(max_abs_diff(h_dV, ref_dV), 2e-3f);

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

#if CUFLASH_ENABLE_RAPIDCHECK
// Property test: Backward pass gradient equivalence
// Feature: cuflash, Property 2: 反向传播梯度等价性
// Validates: Requirements 2.1, 2.3, 2.4, 8.2
RC_GTEST_PROP(BackwardProperty, GradientEquivalence, ()) {
    int batch_size = 1;
    int num_heads = *rc::gen::inRange(1, 3);
    int seq_len = *rc::gen::inRange(1, 33);
    int head_dim = *rc::gen::element(32, 64);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size), h_dO(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = *rc::gen::inRange(-100, 100) * 0.01f;
        h_K[i] = *rc::gen::inRange(-100, 100) * 0.01f;
        h_V[i] = *rc::gen::inRange(-100, 100) * 0.01f;
        h_dO[i] = *rc::gen::inRange(-100, 100) * 0.01f;
    }

    std::vector<float> ref_dQ(qkv_size), ref_dK(qkv_size), ref_dV(qkv_size);
    reference_attention_backward(h_Q, h_K, h_V, h_dO, ref_dQ, ref_dK, ref_dV, batch_size, num_heads,
                                 seq_len, head_dim, scale, false);

    float *d_Q, *d_K, *d_V, *d_O, *d_L, *d_dO, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));
    cudaMalloc(&d_dO, qkv_size * sizeof(float));
    cudaMalloc(&d_dQ, qkv_size * sizeof(float));
    cudaMalloc(&d_dK, qkv_size * sizeof(float));
    cudaMalloc(&d_dV, qkv_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    err = flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK, d_dV, batch_size,
                                   num_heads, seq_len, head_dim, scale, false, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_dQ(qkv_size), h_dK(qkv_size), h_dV(qkv_size);
    cudaMemcpy(h_dQ.data(), d_dQ, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK.data(), d_dK, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV.data(), d_dV, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    RC_ASSERT(max_abs_diff(h_dQ, ref_dQ) < 1e-3f);
    RC_ASSERT(max_abs_diff(h_dK, ref_dK) < 1e-3f);
    RC_ASSERT(max_abs_diff(h_dV, ref_dV) < 1e-3f);

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

#endif

}  // namespace test
}  // namespace cuflash
