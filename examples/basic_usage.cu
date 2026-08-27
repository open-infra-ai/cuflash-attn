// Basic usage example for cuflash

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "cuflash/flash_attention.h"

int main() {
    // Example dimensions
    const int batch_size = 2;
    const int num_heads = 8;
    const int seq_len = 128;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::cout << "cuflash Example" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "num_heads: " << num_heads << std::endl;
    std::cout << "seq_len: " << seq_len << std::endl;
    std::cout << "head_dim: " << head_dim << std::endl;
    std::cout << "scale: " << scale << std::endl;
    std::cout << std::endl;

    // Calculate sizes
    const size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    const size_t l_size = batch_size * num_heads * seq_len;
    auto check_cuda = [](cudaError_t err, const char* operation) {
        if (err != cudaSuccess) {
            std::cerr << operation << " failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    };

    // Allocate host memory
    std::vector<float> h_Q(qkv_size);
    std::vector<float> h_K(qkv_size);
    std::vector<float> h_V(qkv_size);
    std::vector<float> h_O(qkv_size);
    std::vector<float> h_L(l_size);

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    // Allocate device memory
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr, *d_L = nullptr;
    if (!check_cuda(cudaMalloc(&d_Q, qkv_size * sizeof(float)), "cudaMalloc(d_Q)") ||
        !check_cuda(cudaMalloc(&d_K, qkv_size * sizeof(float)), "cudaMalloc(d_K)") ||
        !check_cuda(cudaMalloc(&d_V, qkv_size * sizeof(float)), "cudaMalloc(d_V)") ||
        !check_cuda(cudaMalloc(&d_O, qkv_size * sizeof(float)), "cudaMalloc(d_O)") ||
        !check_cuda(cudaMalloc(&d_L, l_size * sizeof(float)), "cudaMalloc(d_L)")) {
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_L);
        return 1;
    }

    // Copy to device
    if (!check_cuda(cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy(Q)") ||
        !check_cuda(cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy(K)") ||
        !check_cuda(cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice),
                    "cudaMemcpy(V)")) {
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_L);
        return 1;
    }

    // Run forward pass
    auto error = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                  seq_len, head_dim, scale, false, 0);

    if (error != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "Error: " << cuflash::get_error_string(error) << std::endl;
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_L);
        return 1;
    }

    // Copy result back
    if (!check_cuda(cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(O)")) {
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_L);
        return 1;
    }

    std::cout << "Flash Attention forward pass completed successfully!" << std::endl;
    std::cout << "Output[0]: " << h_O[0] << std::endl;
    std::cout << "Output shape: [" << batch_size << ", " << num_heads << ", " << seq_len << ", "
              << head_dim << "]" << std::endl;

    // Example with causal masking
    std::cout << std::endl << "Running with causal masking..." << std::endl;

    error = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                             seq_len, head_dim, scale, true,  // causal = true
                                             0);

    if (error != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "Error: " << cuflash::get_error_string(error) << std::endl;
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_L);
        return 1;
    }

    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        cudaFree(d_L);
        return 1;
    }
    std::cout << "Causal attention completed successfully!" << std::endl;

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);

    return 0;
}
