#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const auto n = static_cast<std::ptrdiff_t>(input.size());

    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCoefficient = 0.044715f;

    const float* __restrict in = input.data();
    float* __restrict out = output.data();

#pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float x3 = x * x * x;
        const float z = kSqrtTwoOverPi * (x + kCoefficient * x3);

        out[i] = x / (1.0f + std::exp(-2.0f * z));
    }

    return output;
}
