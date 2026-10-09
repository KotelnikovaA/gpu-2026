#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <vector>
#include <omp.h>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3,unroll-loops,fast-math,tree-vectorize")
#endif

namespace {
constexpr float kC1 = 1.5957691216f;
constexpr float kC2 = 0.0713548163f;
}  // namespace

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    if (n == 0) {
        return output;
    }

    const float* __restrict__ in = input.data();
    float* __restrict__ out = output.data();

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float u = x * (kC1 + kC2 * x * x);
        out[i] = x / (1.0f + std::exp(-u));
    }

    return output;
}
