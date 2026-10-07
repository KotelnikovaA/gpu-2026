#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>

namespace {
    constexpr int block_thread_count = 256;

void CheckCUDA(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

struct DeviceBuffer {
    float* device_values = nullptr;
    size_t element_capacity = 0;

    ~DeviceBuffer() {
        cudaFree(device_values);
    }

    void Reserve(size_t element_count) {
        if (element_count > element_capacity) {
            CheckCUDA(cudaFree(device_values));
            device_values = nullptr;
            element_capacity = 0;
            CheckCUDA(cudaMalloc(&device_values, element_count * sizeof(float)));
            element_capacity = element_count;
        }
    }
};

__global__ void RowSoftmax(float* matrix, int row_size) {
    __shared__ float partial_values[block_thread_count];
    size_t row_start = static_cast<size_t>(blockIdx.x) * row_size;
    int thread_index = threadIdx.x;
    float row_maximum = -CUDART_INF_F;

    for (int column = thread_index; column < row_size; column += blockDim.x) {
        row_maximum = fmaxf(row_maximum, matrix[row_start + column]);
    }
    partial_values[thread_index] = row_maximum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            partial_values[thread_index] = fmaxf(partial_values[thread_index],
                                                partial_values[thread_index + stride]);
        }
        __syncthreads();
    }
    row_maximum = partial_values[0];
    __syncthreads();

    float row_sum = 0.0f;
    for (int column = thread_index; column < row_size; column += blockDim.x) {
        float exponential = __expf(matrix[row_start + column] - row_maximum);
        matrix[row_start + column] = exponential;
        row_sum += exponential;
    }
    partial_values[thread_index] = row_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            partial_values[thread_index] += partial_values[thread_index + stride];
        }
        __syncthreads();
    }
    float inverse_sum = 1.0f / partial_values[0];
    for (int column = thread_index; column < row_size; column += blockDim.x) {
        matrix[row_start + column] *= inverse_sum;
    }
}
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty()) {
        return {};
    }

    size_t byte_count = input.size() * sizeof(float);
    int row_size = static_cast<int>(input.size() / row_count);
    static thread_local DeviceBuffer device_buffer;
    device_buffer.Reserve(input.size());
    CheckCUDA(cudaMemcpy(device_buffer.device_values, input.data(),
                         byte_count, cudaMemcpyHostToDevice));

    RowSoftmax<<<row_count, block_thread_count>>>(device_buffer.device_values, row_size);
    CheckCUDA(cudaGetLastError());

    std::vector<float> result(input.size());
    CheckCUDA(cudaMemcpy(result.data(), device_buffer.device_values,
                         byte_count, cudaMemcpyDeviceToHost));
    return result;
}
