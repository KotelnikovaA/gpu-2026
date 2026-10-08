#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCoefficient = 0.044715f;
    const auto n = static_cast<std::ptrdiff_t>(input.size());

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float z = kSqrtTwoOverPi * (x + kCoefficient * x3);

        output[i] = 0.5f * x * (1.0f + std::tanh(z));
    }

    return output;
}
