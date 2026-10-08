#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;

__device__ float WarpMax(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

__device__ float WarpSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

__global__ void SoftmaxKernel(const float* input, float* output, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float* max_values = shared;
    float* sums = shared + kWarps;
    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        local_max = fmaxf(local_max, input[row * cols + col]);
    local_max = WarpMax(local_max);
    if ((threadIdx.x & 31) == 0) max_values[threadIdx.x / 32] = local_max;
    __syncthreads();
    if (threadIdx.x < 32) {
        local_max = threadIdx.x < kWarps ? max_values[threadIdx.x] : -CUDART_INF_F;
        local_max = WarpMax(local_max);
        if (threadIdx.x == 0) max_values[0] = local_max;
    }
    __syncthreads();
    const float row_max = max_values[0];
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        local_sum += expf(input[row * cols + col] - row_max);
    local_sum = WarpSum(local_sum);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = local_sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        local_sum = threadIdx.x < kWarps ? sums[threadIdx.x] : 0.0f;
        local_sum = WarpSum(local_sum);
        if (threadIdx.x == 0) sums[0] = local_sum;
    }
    __syncthreads();
    const float denominator = sums[0];
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        output[row * cols + col] = expf(input[row * cols + col] - row_max) / denominator;
}
void CheckCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.size() % static_cast<std::size_t>(row_count) != 0)
        throw std::invalid_argument("row_count must be positive and divide input size");
    const int cols = static_cast<int>(input.size() / row_count);
    std::vector<float> output(input.size());
    if (input.empty()) return output;
    const std::size_t bytes = input.size() * sizeof(float);
    static float* workspace = nullptr;
    static std::size_t capacity = 0;
    if (bytes > capacity) {
        if (workspace) CheckCuda(cudaFree(workspace), "cudaFree(workspace)");
        CheckCuda(cudaMalloc(&workspace, 2 * bytes), "cudaMalloc(workspace)");
        capacity = bytes;
    }
    float* device_input = workspace;
    float* device_output = workspace + capacity / sizeof(float);
    CheckCuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");
    SoftmaxKernel<<<row_count, kThreads, 2 * kWarps * sizeof(float)>>>(device_input, device_output, row_count, cols);
    CheckCuda(cudaGetLastError(), "SoftmaxKernel launch");
    CheckCuda(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost), "copy output");
    return output;
}
