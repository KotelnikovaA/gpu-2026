#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <thread>

namespace {

constexpr int kBlockSize = 256;

// 0.5 * x * (1 + tanh(z)) == x / (1 + exp(-2z)), z = sqrt(2/pi) * (x + 0.044715 * x^3)
constexpr float kA = -2.0f * 0.7978845608028654f;
constexpr float kB = kA * 0.044715f;

__device__ __forceinline__ float Gelu(float x) {
    return __fdividef(x, 1.0f + __expf(x * (kA + kB * x * x)));
}

// In-place: a single device buffer is enough
__global__ void GeluKernel(float* __restrict__ data, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n4 = n / 4;
    if (i < n4) {
        float4 v = reinterpret_cast<float4*>(data)[i];
        v.x = Gelu(v.x);
        v.y = Gelu(v.y);
        v.z = Gelu(v.z);
        v.w = Gelu(v.w);
        reinterpret_cast<float4*>(data)[i] = v;
    } else if (i - n4 < n % 4) {
        data[n4 * 4 + (i - n4)] = Gelu(data[n4 * 4 + (i - n4)]);
    }
}

// Device memory is allocated once and reused between calls
struct DeviceBuffer {
    float* data = nullptr;
    size_t capacity = 0;

    float* Get(size_t count) {
        if (count > capacity) {
            cudaFree(data);
            cudaMalloc(&data, count * sizeof(float));
            capacity = count;
        }
        return data;
    }
};

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) return {};
    const size_t bytes = n * sizeof(float);

    static DeviceBuffer buffer;
    float* data = buffer.Get(n);

    // Result allocation (zero fill + page faults) runs in parallel with transfer & computations
    std::vector<float> output;
    std::thread allocator([&output, n] { output.resize(n); });

    cudaMemcpy(data, input.data(), bytes, cudaMemcpyHostToDevice);
    const size_t threads = n / 4 + n % 4;
    GeluKernel<<<static_cast<unsigned>((threads + kBlockSize - 1) / kBlockSize), kBlockSize>>>(data, n);

    allocator.join();
    cudaMemcpy(output.data(), data, bytes, cudaMemcpyDeviceToHost);
    return output;
}
