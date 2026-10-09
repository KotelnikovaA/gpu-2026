#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

namespace {

constexpr int COLUMNS_PER_THREAD = 8;

__global__ void TransposeMatrix(const float* source, float* destination, int matrix_size) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < matrix_size && column < matrix_size) {
        destination[column * matrix_size + row] = source[row * matrix_size + column];
    }
}

__global__ void MultiplyMatrices(const float* __restrict__ transposed_left, const float* __restrict__ right_matrix, float* __restrict__ transposed_result, int matrix_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int first_column = (blockIdx.y * blockDim.y + threadIdx.y) * COLUMNS_PER_THREAD;
    if (row >= matrix_size || first_column >= matrix_size) {
        return;
    }

    float column_sums[COLUMNS_PER_THREAD] = {};
    #pragma unroll 4
    for (int inner = 0; inner < matrix_size; ++inner) {
        float left_value = transposed_left[inner * matrix_size + row];
        #pragma unroll
        for (int column_offset = 0; column_offset < COLUMNS_PER_THREAD; ++column_offset) {
            int column = first_column + column_offset;
            if (column < matrix_size) {
                column_sums[column_offset] = fmaf(left_value,
                    right_matrix[inner * matrix_size + column], column_sums[column_offset]);
            }
        }
    }

    #pragma unroll
    for (int column_offset = 0; column_offset < COLUMNS_PER_THREAD; ++column_offset) {
        int column = first_column + column_offset;
        if (column < matrix_size) {
            transposed_result[column * matrix_size + row] = column_sums[column_offset];
        }
    }
}

struct DeviceBuffers {
    float* values = nullptr;
    size_t capacity = 0;
    cudaStream_t stream;

    DeviceBuffers() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~DeviceBuffers() {
        cudaFree(values);
        cudaStreamDestroy(stream);
    }

    void Reserve(size_t element_count) {
        if (element_count > capacity) {
            cudaFree(values);
            cudaMalloc(&values, 4 * element_count * sizeof(float));
            capacity = element_count;
        }
    }
};

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& left_matrix, const std::vector<float>& right_matrix, int matrix_size) {
    if (left_matrix.empty()) {
        return {};
    }

    static DeviceBuffers device_buffers;
    size_t element_count = static_cast<size_t>(matrix_size) * matrix_size;
    size_t byte_count = element_count * sizeof(float);
    device_buffers.Reserve(element_count);
    float* device_left = device_buffers.values;
    float* device_right = device_left + element_count;
    float* device_transposed_left = device_right + element_count;
    float* device_transposed_result = device_transposed_left + element_count;

    cudaMemcpyAsync(device_left, left_matrix.data(), byte_count,
                    cudaMemcpyHostToDevice, device_buffers.stream);
    cudaMemcpyAsync(device_right, right_matrix.data(), byte_count,
                    cudaMemcpyHostToDevice, device_buffers.stream);

    dim3 transpose_threads(32, 8);
    dim3 transpose_blocks((matrix_size + 31) / 32, (matrix_size + 7) / 8);
    TransposeMatrix<<<transpose_blocks, transpose_threads, 0, device_buffers.stream>>>(
        device_left, device_transposed_left, matrix_size);

    dim3 threads(32, 4);
    dim3 blocks((matrix_size + threads.x - 1) / threads.x,
                (matrix_size + threads.y * COLUMNS_PER_THREAD - 1) /
                (threads.y * COLUMNS_PER_THREAD));
    MultiplyMatrices<<<blocks, threads, 0, device_buffers.stream>>>(
        device_transposed_left, device_right, device_transposed_result, matrix_size);
    TransposeMatrix<<<transpose_blocks, transpose_threads, 0, device_buffers.stream>>>(
        device_transposed_result, device_left, matrix_size);

    std::vector<float> result(element_count);
    cudaMemcpyAsync(result.data(), device_left, byte_count,
                    cudaMemcpyDeviceToHost, device_buffers.stream);
    cudaStreamSynchronize(device_buffers.stream);
    return result;
}
