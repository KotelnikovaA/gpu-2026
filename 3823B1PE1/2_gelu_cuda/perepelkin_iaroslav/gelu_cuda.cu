#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void geluKernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-1.702f * x));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const size_t size = input.size();
    float* d_in = nullptr;
    float* d_out = nullptr;

    // 1. Device memory allocation
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));

    // 2. Copy data from host to device
    cudaMemcpyAsync(d_in, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    geluKernel<<<numBlocks, blockSize>>>(d_in, d_out, size);

    // 4. Allocate result vector on host
    std::vector<float> result(size);

    // 5. Copy data from device to host and synchronize
    cudaMemcpyAsync(result.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // 6. Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return result;
}
