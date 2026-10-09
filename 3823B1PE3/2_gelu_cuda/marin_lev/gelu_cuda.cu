#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace {

void Check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

struct Buffer {
    float* data = nullptr;
    std::size_t capacity = 0;

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) return;

        float* next = nullptr;
        Check(cudaMalloc(&next, bytes));
        cudaFree(data);

        data = next;
        capacity = bytes;
    }

    ~Buffer() {
        cudaFree(data);
    }
};

__device__ __forceinline__ float Gelu(float x) {
    const float z =
        0.7978845608028654f * (x + 0.044715f * x * x * x);
    return x / (1.0f + expf(-2.0f * z));
}

__global__ void Kernel(const float* __restrict__ input,
                       float* __restrict__ output,
                       std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n / 4) {
        float4 v = reinterpret_cast<const float4*>(input)[i];

        v.x = Gelu(v.x);
        v.y = Gelu(v.y);
        v.z = Gelu(v.z);
        v.w = Gelu(v.w);

        reinterpret_cast<float4*>(output)[i] = v;
    }

    if (i < n % 4) {
        const std::size_t tail = n / 4 * 4 + i;
        output[tail] = Gelu(input[tail]);
    }
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) return {};

    const std::size_t n = input.size();
    const std::size_t bytes = n * sizeof(float);

    static thread_local Buffer device_input, device_output;
    device_input.Reserve(bytes);
    device_output.Reserve(bytes);

    Check(cudaMemcpy(device_input.data, input.data(), bytes,
                     cudaMemcpyHostToDevice));

    constexpr unsigned int threads = 256;
    const std::size_t groups = 1 + (n - 1) / 4;
    const unsigned int blocks =
        static_cast<unsigned int>(1 + (groups - 1) / threads);

    Kernel<<<blocks, threads>>>(
        device_input.data, device_output.data, n);
    Check(cudaGetLastError());

    std::vector<float> output(n);
    Check(cudaMemcpy(output.data(), device_output.data, bytes,
                     cudaMemcpyDeviceToHost));

    return output;
}
