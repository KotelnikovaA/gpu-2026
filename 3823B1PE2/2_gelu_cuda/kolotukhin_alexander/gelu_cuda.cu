#include "gelu_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_ERROR(X)                                          \
    do {                                                        \
        cudaError_t error = (X);                                \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d — %s: %s\n",   \
                    __FILE__, __LINE__,                         \
                    cudaGetErrorName(error),                    \
                    cudaGetErrorString(error));                 \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)

namespace {

constexpr float k1 = 1.5957691216057308f;
constexpr float k2 = 0.044715f;
constexpr int threads_per_block = 256;

__global__ void gelu_kernel(float* __restrict__ data, int input_size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < input_size) {
        const float x = data[i];
        data[i] = x / (1.0f + __expf(- k1 * x * (1 + k2 * x * x)));
    }
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int input_size = static_cast<int>(input.size());
    const std::size_t bytes = static_cast<std::size_t>(input_size) * sizeof(float);

    std::vector<float> output(input_size);
    if (input_size == 0) {
        return output;
    }

    static float* data_buffer = nullptr;
    static int capacity = 0;
    if (input_size > capacity) {
        if (data_buffer) {
            CHECK_ERROR(cudaFree(data_buffer));
        }
        CHECK_ERROR(cudaMalloc(&data_buffer, bytes));
        capacity = input_size;
    }

    CHECK_ERROR(cudaMemcpy(data_buffer, input.data(), bytes, cudaMemcpyHostToDevice));

    const int num_blocks = (input_size + threads_per_block - 1) / threads_per_block;
    gelu_kernel<<<num_blocks, threads_per_block>>>(data_buffer, input_size);

    CHECK_ERROR(cudaMemcpy(output.data(), data_buffer, bytes, cudaMemcpyDeviceToHost));
    return output;
}
