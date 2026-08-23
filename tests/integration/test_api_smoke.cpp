#include <iostream>
#include <string>

#include "cuflash/flash_attention.h"

int main() {
    float query = 0.0F;
    float output = 0.0F;
    float logsumexp = 0.0F;

    auto null_err = cuflash::flash_attention_forward(nullptr, &query, &query, &output, &logsumexp,
                                                     1, 1, 1, 32, 1.0F, false, nullptr);
    if (null_err != cuflash::FlashAttentionError::NULL_POINTER) {
        std::cerr << "Expected NULL_POINTER from C++ API, got " << static_cast<int>(null_err)
                  << std::endl;
        return 1;
    }

    auto invalid_dim_err = cuflash::flash_attention_forward(
        &query, &query, &query, &output, &logsumexp, 0, 1, 1, 32, 1.0F, false, nullptr);
    if (invalid_dim_err != cuflash::FlashAttentionError::INVALID_DIMENSION) {
        std::cerr << "Expected INVALID_DIMENSION from C++ API, got "
                  << static_cast<int>(invalid_dim_err) << std::endl;
        return 1;
    }

    int abi_err = cuflash_attention_forward_f32(nullptr, &query, &query, &output, &logsumexp, 1, 1,
                                                1, 32, 1.0F, false, nullptr);
    if (abi_err != static_cast<int>(cuflash::FlashAttentionError::NULL_POINTER)) {
        std::cerr << "Expected NULL_POINTER from C ABI, got " << abi_err << std::endl;
        return 1;
    }

    const char* message =
        cuflash::get_error_string(cuflash::FlashAttentionError::UNSUPPORTED_HEAD_DIM);
    if (message == nullptr || std::string(message).empty()) {
        std::cerr << "Error string lookup returned an empty message" << std::endl;
        return 1;
    }

    std::cout << "API smoke test passed" << std::endl;
    return 0;
}
