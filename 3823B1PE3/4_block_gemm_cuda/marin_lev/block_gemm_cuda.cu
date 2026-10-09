#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr int kTileSize = 32;

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

__global__ void MultiplyTiles(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c, int n) {
    __shared__ float tile_a[kTileSize][kTileSize];
    __shared__ float tile_b[kTileSize][kTileSize];
    __shared__ float tile_c[kTileSize][kTileSize];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = blockIdx.y * kTileSize + y;
    const int col = blockIdx.x * kTileSize + x;

    tile_c[y][x] = 0.0f;

    for (int start = 0; start < n; start += kTileSize) {
        tile_a[y][x] = (row < n && start + x < n)
            ? a[std::size_t(row) * n + start + x] : 0.0f;

        tile_b[y][x] = (start + y < n && col < n)
            ? b[std::size_t(start + y) * n + col] : 0.0f;

        __syncthreads();

        float sum = tile_c[y][x];

#pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum = fmaf(tile_a[y][k], tile_b[k][x], sum);
        }

        tile_c[y][x] = sum;

        __syncthreads();
    }

    if (row < n && col < n)
        c[std::size_t(row) * n + col] = tile_c[y][x];
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

    const dim3 block(kTileSize, kTileSize);
    const unsigned int tiles = 1 + (n - 1) / kTileSize;
    const dim3 grid(tiles, tiles);

    MultiplyTiles<<<grid, block>>>(da, db, dc, n);
    EnsureCudaSuccess(cudaGetLastError());

    std::vector<float> result(count);
    EnsureCudaSuccess(cudaMemcpy(
        result.data(), dc, bytes, cudaMemcpyDeviceToHost));

    return result;
}
