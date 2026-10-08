#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace {

void Check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

struct Workspace {
    float* data = nullptr;
    std::size_t capacity = 0;

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) return;

        float* replacement = nullptr;
        Check(cudaMalloc(&replacement, bytes));
        cudaFree(data);

        data = replacement;
        capacity = bytes;
    }

    ~Workspace() {
        cudaFree(data);
    }
};

template<bool Bounds>
__global__ void Multiply(const float* __restrict__ a,
                         const float* __restrict__ b,
                         float* __restrict__ c, int n) {
    const int first_row = blockIdx.y * 32 + threadIdx.y;
    const int first_col = blockIdx.x * 64 + threadIdx.x;
    const int second_col = first_col + 32;

    if constexpr (Bounds) {
        if (first_row >= n || first_col >= n) return;
    }

    float first[8] = {};
    float second[8] = {};

    for (int k = 0; k < n; ++k) {
        const std::size_t offset = std::size_t(k) * n;

        const float b0 = b[offset + first_col];
        const float b1 = (!Bounds || second_col < n)
            ? b[offset + second_col] : 0.0f;

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = first_row + r * 4;

            if (!Bounds || row < n) {
                const float av = a[std::size_t(row) * n + k];

                first[r] = fmaf(av, b0, first[r]);
                second[r] = fmaf(av, b1, second[r]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = first_row + r * 4;

        if (!Bounds || row < n) {
            const std::size_t offset = std::size_t(row) * n;

            c[offset + first_col] = first[r];

            if (!Bounds || second_col < n)
                c[offset + second_col] = second[r];
        }
    }
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b, int n) {
    if (n <= 0) return {};

    const std::size_t count = std::size_t(n) * n;
    if (a.size() != count || b.size() != count)
        throw std::invalid_argument("Size");

    const std::size_t bytes = count * sizeof(float);

    static thread_local Workspace workspace;
    workspace.Reserve(3 * bytes);

    float* da = workspace.data;
    float* db = da + count;
    float* dc = db + count;

    Check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    Check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(32, 4);
    const dim3 grid(1 + (n - 1) / 64, 1 + (n - 1) / 32);

    if (n % 64 == 0)
        Multiply<false><<<grid, block>>>(da, db, dc, n);
    else
        Multiply<true><<<grid, block>>>(da, db, dc, n);

    Check(cudaGetLastError());

    std::vector<float> result(count);
    Check(cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost));

    return result;
}
