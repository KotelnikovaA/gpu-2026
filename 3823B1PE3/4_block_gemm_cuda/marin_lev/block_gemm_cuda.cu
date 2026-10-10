#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr int kOutputTile = 64;
constexpr int kInnerTile = 32;

void EnsureCudaSuccess(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

struct GpuStorage {
    float* data = nullptr;
    std::size_t capacity = 0;

    void EnsureCapacity(std::size_t bytes) {
        if (bytes <= capacity) return;

        float* next = nullptr;
        EnsureCudaSuccess(cudaMalloc(&next, bytes));
        cudaFree(data);

        data = next;
        capacity = bytes;
    }

    ~GpuStorage() {
        cudaFree(data);
    }
};

template<bool CheckEdges>
__global__ void AccumulateSharedTiles(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c, int n) {
    __shared__ float shared_a[kOutputTile][kInnerTile];
    __shared__ float shared_b[kInnerTile][kOutputTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 32 + tx;

    const int base_row = blockIdx.y * kOutputTile;
    const int base_col = blockIdx.x * kOutputTile;

    float first[8] = {};
    float second[8] = {};

    for (int start = 0; start < n; start += kInnerTile) {
#pragma unroll
        for (int i = tid; i < kOutputTile * kInnerTile; i += 256) {
            const int r = i / kInnerTile;
            const int k = i % kInnerTile;
            const int row = base_row + r;
            const int col = start + k;

            shared_a[r][k] = (!CheckEdges || (row < n && col < n))
                ? a[std::size_t(row) * n + col] : 0.0f;
        }

#pragma unroll
        for (int i = tid; i < kInnerTile * kOutputTile; i += 256) {
            const int k = i / kOutputTile;
            const int x = i % kOutputTile;
            const int row = start + k;
            const int col = base_col + x;

            shared_b[k][x] = (!CheckEdges || (row < n && col < n))
                ? b[std::size_t(row) * n + col] : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < kInnerTile; ++k) {
            const float b0 = shared_b[k][tx];
            const float b1 = shared_b[k][tx + 32];

#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float av = shared_a[ty + r * 8][k];

                first[r] = fmaf(av, b0, first[r]);
                second[r] = fmaf(av, b1, second[r]);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = base_row + ty + r * 8;
        const int col = base_col + tx;

        if (!CheckEdges || row < n) {
            const std::size_t offset = std::size_t(row) * n;

            if (!CheckEdges || col < n)
                c[offset + col] = first[r];

            if (!CheckEdges || col + 32 < n)
                c[offset + col + 32] = second[r];
        }
    }
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b, int n) {
    if (n <= 0) return {};

    const std::size_t count = std::size_t(n) * n;
    if (a.size() != count || b.size() != count)
        throw std::invalid_argument("Size");

    const std::size_t bytes = count * sizeof(float);

    static thread_local GpuStorage storage;
    storage.EnsureCapacity(3 * bytes);

    float* da = storage.data;
    float* db = da + count;
    float* dc = db + count;

    EnsureCudaSuccess(cudaMemcpy(
        da, a.data(), bytes, cudaMemcpyHostToDevice));
    EnsureCudaSuccess(cudaMemcpy(
        db, b.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(32, 8);
    const unsigned int tiles = 1 + (n - 1) / kOutputTile;
    const dim3 grid(tiles, tiles);

    if (n % kOutputTile == 0)
        AccumulateSharedTiles<false><<<grid, block>>>(da, db, dc, n);
    else
        AccumulateSharedTiles<true><<<grid, block>>>(da, db, dc, n);

    EnsureCudaSuccess(cudaGetLastError());

    std::vector<float> result(count);
    EnsureCudaSuccess(cudaMemcpy(
        result.data(), dc, bytes, cudaMemcpyDeviceToHost));

    return result;
}
