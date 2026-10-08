#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

struct DeviceBuffer {
    float* data = nullptr;
    std::size_t capacity = 0;

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        if (data != nullptr) {
            CheckCuda(cudaFree(data), "cudaFree");
            data = nullptr;
            capacity = 0;
        }
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes),
                  "cudaMalloc");
        capacity = bytes;
    }

    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }
};

__global__ void GeluKernel(float* data, std::size_t n) {
    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = start; i < n; i += stride) {
        const float x = data[i];
        const float x3 = x * x * x;
        const float z = 0.7978845608028654f * (x + 0.044715f * x3);
        data[i] = x / (1.0f + expf(-2.0f * z));
    }
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t n = input.size();
    const std::size_t bytes = n * sizeof(float);
    static thread_local DeviceBuffer buffer;
    buffer.Reserve(bytes);

    CheckCuda(cudaMemcpy(buffer.data, input.data(), bytes,
                         cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    constexpr unsigned int kBlockSize = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>(1 + (n - 1) / kBlockSize, 65535));

    GeluKernel<<<blocks, kBlockSize>>>(buffer.data, n);
    CheckCuda(cudaGetLastError(), "GeluKernel launch");

    std::vector<float> output(n);
    CheckCuda(cudaMemcpy(output.data(), buffer.data, bytes,
                         cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    return output;
}
