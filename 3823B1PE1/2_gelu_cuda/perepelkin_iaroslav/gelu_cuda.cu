#include <cuda_runtime.h>
#include <cmath>

#include "gelu_cuda.h"

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

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static size_t current_capacity = 0;

    if (current_capacity < size) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_in, size * sizeof(float));
        cudaMalloc(&d_out, size * sizeof(float));
        current_capacity = size;
    }

    cudaMemcpyAsync(d_in, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    geluKernel<<<numBlocks, blockSize>>>(d_in, d_out, size);

    std::vector<float> result(size);

    cudaMemcpyAsync(result.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return result;
}
