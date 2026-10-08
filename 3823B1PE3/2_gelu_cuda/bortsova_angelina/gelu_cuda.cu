#include "gelu_cuda.h"
#include <cuda_runtime.h>

namespace {

__global__ void ApplyGelu(float* values, size_t element_count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        float value = values[index];
        float exponent = 1.5957691216f * value * (1.0f + 0.044715f * value * value);
        values[index] = value / (1.0f + __expf(-exponent));
    }
}

struct DeviceBuffer {
    float* values = nullptr;
    size_t capacity = 0;
    cudaStream_t stream;

    DeviceBuffer() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~DeviceBuffer() {
        cudaFree(values);
        cudaStreamDestroy(stream);
    }

    void Reserve(size_t element_count) {
        if (element_count > capacity) {
            cudaFree(values);
            cudaMalloc(&values, element_count * sizeof(float));
            capacity = element_count;
        }
    }
};
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    static DeviceBuffer device_buffer;
    device_buffer.Reserve(input.size());
    size_t byte_count = input.size() * sizeof(float);
    cudaMemcpyAsync(device_buffer.values, input.data(), byte_count,
                    cudaMemcpyHostToDevice, device_buffer.stream);

    constexpr unsigned int THREADS_PER_BLOCK = 256;
    unsigned int block_count = static_cast<unsigned int>(
        (input.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    ApplyGelu<<<block_count, THREADS_PER_BLOCK, 0, device_buffer.stream>>>(
        device_buffer.values, input.size());

    std::vector<float> result(input.size());
    cudaMemcpyAsync(result.data(), device_buffer.values, byte_count,
                    cudaMemcpyDeviceToHost, device_buffer.stream);
    cudaStreamSynchronize(device_buffer.stream);
    return result;
}
