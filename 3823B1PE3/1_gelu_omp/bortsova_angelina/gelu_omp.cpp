#include "gelu_omp.h"

#include <cmath>
#include <cstdint>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    const std::int64_t element_count = static_cast<std::int64_t>(input.size());

    #pragma omp parallel for schedule(static) if(element_count >= 4096)
    for (std::int64_t index = 0; index < element_count; ++index) {
        float value = input[index];
        float exponent = 1.5957691216f * value * (1.0f + 0.044715f * value * value);
        result[index] = value / (1.0f + std::exp(-exponent));
    }
    return result;
}
