#include <cmath>

#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const size_t size = input.size();
    std::vector<float> result(size);

    // GELU approximation:
    // GELU(x) = x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
    const float c = -1.702f;

    const float* in_ptr = input.data();
    float* out_ptr = result.data();

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        float x = in_ptr[i];
        out_ptr[i] = x / (1.0f + std::exp(c * x));
    }

    return result;
}
