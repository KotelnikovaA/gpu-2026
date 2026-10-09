#include "gelu_omp.h"
#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result_vector(input.size());

    const std::size_t kInputVectorSize = input.size();
    const float* kInputData = input.data();
    float* result_data = result_vector.data();

    constexpr float kSqrt2DevPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;

    #pragma omp parallel for simd schedule(static)
    for (std::size_t i = 0; i < kInputVectorSize; ++i) {
        const float kX = kInputData[i];
        const float kFabsX = std::fabs(kX);

        const float kDoubleFabsX = (2.0f * kSqrt2DevPi) * kFabsX * (1.0f + kCoeff * kFabsX * kFabsX);
        const float kExp = std::exp(-kDoubleFabsX);

        result_data[i] = (kX >= 0.0f ? kX : kX * kExp) / (1.0f + kExp);
    }

    return result_vector;
}