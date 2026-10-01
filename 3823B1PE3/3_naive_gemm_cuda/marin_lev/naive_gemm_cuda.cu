#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int row = blockIdx.y * blockDim.y + warp;
    const int col = blockIdx.x * blockDim.x + lane;

    if (row >= n || col >= n) {
        return;
    }

    const float* a_row = a + row * n;

    float sum = 0.0f;

#pragma unroll 4
    for (int k = 0; k < n; ++k) {
        const float* b_row = b + k * n;
        sum += a_row[k] * b_row[col];
    }

    c[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t matrix_size = static_cast<size_t>(n) * n;

    if (a.size() != matrix_size || b.size() != matrix_size) {
        throw std::invalid_argument("Size");
    }

    std::vector<float> c(matrix_size);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    const size_t bytes = matrix_size * sizeof(float);

    if (cudaMalloc(&device_a, bytes) != cudaSuccess ||
        cudaMalloc(&device_b, bytes) != cudaSuccess ||
        cudaMalloc(&device_c, bytes) != cudaSuccess) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        throw std::runtime_error("Alloc");
    }

    if (cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        throw std::runtime_error("Copy");
    }

    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = 8;

    const dim3 block(kWarpSize, kWarpsPerBlock);

    const dim3 grid(
        (n + kWarpSize - 1) / kWarpSize,
        (n + kWarpsPerBlock - 1) / kWarpsPerBlock
    );

    NaiveGemmKernel<<<grid, block>>>(device_a, device_b, device_c, n);

    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        throw std::runtime_error("Kernel");
    }

    if (cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        throw std::runtime_error("Copy");
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return c;
}
