#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {

constexpr int tile_size = 128;
constexpr int inner_tile_size = 32;
constexpr int rows_per_thread = 8;
constexpr int columns_per_thread = 8;

void CheckCUDA(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

struct DeviceMatrices {
    float* left_matrix = nullptr;
    float* right_matrix = nullptr;
    float* result_matrix = nullptr;

    ~DeviceMatrices() {
        cudaFree(left_matrix);
        cudaFree(right_matrix);
        cudaFree(result_matrix);
    }
};

__global__ void BlockGemm(const float* __restrict__ left_matrix,
                          const float* __restrict__ right_matrix,
                          float* __restrict__ result_matrix, int matrix_size) {
    __shared__ float left_tile[tile_size][inner_tile_size];
    __shared__ float right_tile[inner_tile_size][tile_size];

    int first_row = blockIdx.y * tile_size + threadIdx.y;
    int first_column = blockIdx.x * tile_size + threadIdx.x;
    int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int thread_count = blockDim.x * blockDim.y;
    float sums[rows_per_thread][columns_per_thread] = {};

    for (int tile_offset = 0; tile_offset < matrix_size; tile_offset += inner_tile_size) {
        for (int element_index = thread_index;
             element_index < tile_size * inner_tile_size; element_index += thread_count) {
            int left_tile_row = element_index / inner_tile_size;
            int left_tile_column = element_index % inner_tile_size;
            int matrix_row = blockIdx.y * tile_size + left_tile_row;
            int left_column = tile_offset + left_tile_column;
            left_tile[left_tile_row][left_tile_column] =
                matrix_row < matrix_size && left_column < matrix_size
                ? left_matrix[matrix_row * matrix_size + left_column] : 0.0f;

            int right_tile_row = element_index / tile_size;
            int right_tile_column = element_index % tile_size;
            int right_row = tile_offset + right_tile_row;
            int matrix_column = blockIdx.x * tile_size + right_tile_column;
            right_tile[right_tile_row][right_tile_column] =
                right_row < matrix_size && matrix_column < matrix_size
                ? right_matrix[right_row * matrix_size + matrix_column] : 0.0f;
        }
        __syncthreads();

        #pragma unroll 4
        for (int inner_index = 0; inner_index < inner_tile_size; ++inner_index) {
            float right_values[columns_per_thread];
            #pragma unroll
            for (int column_offset = 0; column_offset < columns_per_thread; ++column_offset) {
                int tile_column = threadIdx.x + column_offset * blockDim.x;
                right_values[column_offset] = right_tile[inner_index][tile_column];
            }

            #pragma unroll
            for (int row_offset = 0; row_offset < rows_per_thread; ++row_offset) {
                int tile_row = threadIdx.y + row_offset * blockDim.y;
                float left_value = left_tile[tile_row][inner_index];
                #pragma unroll
                for (int column_offset = 0; column_offset < columns_per_thread; ++column_offset) {
                    sums[row_offset][column_offset] = fmaf(left_value, right_values[column_offset],
                                                           sums[row_offset][column_offset]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int row_offset = 0; row_offset < rows_per_thread; ++row_offset) {
        int row = first_row + row_offset * blockDim.y;
        #pragma unroll
        for (int column_offset = 0; column_offset < columns_per_thread; ++column_offset) {
            int column = first_column + column_offset * blockDim.x;
            if (row < matrix_size && column < matrix_size) {
                result_matrix[row * matrix_size + column] = sums[row_offset][column_offset];
            }
        }
    }
}

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& left_matrix,
                               const std::vector<float>& right_matrix,
                               int matrix_size) {
    size_t element_count = static_cast<size_t>(matrix_size) * matrix_size;
    size_t byte_count = element_count * sizeof(float);
    DeviceMatrices device_matrices;
    CheckCUDA(cudaMalloc(&device_matrices.left_matrix, byte_count));
    CheckCUDA(cudaMalloc(&device_matrices.right_matrix, byte_count));
    CheckCUDA(cudaMalloc(&device_matrices.result_matrix, byte_count));
    CheckCUDA(cudaMemcpy(device_matrices.left_matrix, left_matrix.data(),
                         byte_count, cudaMemcpyHostToDevice));
    CheckCUDA(cudaMemcpy(device_matrices.right_matrix, right_matrix.data(),
                         byte_count, cudaMemcpyHostToDevice));

    dim3 block_size(tile_size / columns_per_thread, tile_size / rows_per_thread);
    dim3 grid_size((matrix_size + tile_size - 1) / tile_size,
                   (matrix_size + tile_size - 1) / tile_size);
    BlockGemm<<<grid_size, block_size>>>(device_matrices.left_matrix,
                                        device_matrices.right_matrix,
                                        device_matrices.result_matrix, matrix_size);
    CheckCUDA(cudaGetLastError());

    std::vector<float> result_matrix(element_count);
    CheckCUDA(cudaMemcpy(result_matrix.data(), device_matrices.result_matrix,
                         byte_count, cudaMemcpyDeviceToHost));
    return result_matrix;
}
