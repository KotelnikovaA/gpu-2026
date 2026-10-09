#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr float kC1 = 1.5957691216f;
constexpr float kC2 = 0.0713548163f;

__device__ __forceinline__ float GeluFast(float x) {
    const float u = x * (kC1 + kC2 * x * x);
    return x / (1.0f + __expf(-u));
}

__global__ void GeluVectorKernel(float4* data, int n4) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 v = data[idx];
        v.x = GeluFast(v.x);
        v.y = GeluFast(v.y);
        v.z = GeluFast(v.z);
        v.w = GeluFast(v.w);
        data[idx] = v;
    }
}

__global__ void GeluTailKernel(float* data, int start, int n) {
    const int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = GeluFast(data[idx]);
    }
}

struct GpuMemoryManager {
    float* dev_ptr = nullptr;
    std::size_t capacity = 0;

    float* Acquire(std::size_t n) {
        if (n > capacity) {
            if (dev_ptr != nullptr) {
                cudaFree(dev_ptr);
            }
            cudaMalloc(&dev_ptr, n * sizeof(float));
            capacity = n;
        }
        return dev_ptr;
    }

    ~GpuMemoryManager() {
        if (dev_ptr != nullptr) {
            cudaFree(dev_ptr);
        }
    }
};

GpuMemoryManager g_gpu_mem;

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }

    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    float* d_data = g_gpu_mem.Acquire(static_cast<std::size_t>(n));

    std::vector<float> output;
    std::thread host_alloc_thread([&output, n] { output.resize(n); });

    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    const int n4 = n / 4;
    constexpr int kBlockSize = 256;

    if (n4 > 0) {
        const int num_blocks = (n4 + kBlockSize - 1) / kBlockSize;
        GeluVectorKernel<<<num_blocks, kBlockSize>>>(reinterpret_cast<float4*>(d_data), n4);
    }

    const int remainder = n % 4;
    if (remainder > 0) {
        const int tail_start = n4 * 4;
        GeluTailKernel<<<1, remainder>>>(d_data, tail_start, n);
    }

    host_alloc_thread.join();

    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    return output;
}
