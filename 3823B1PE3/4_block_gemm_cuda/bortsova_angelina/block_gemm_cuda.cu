#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

namespace {

constexpr int TILE_ROWS = 64;
constexpr int TILE_COLUMNS = 32;
constexpr int ROWS_PER_THREAD = 4;

__global__ void MultiplyBlocks(const float* __restrict__ left_matrix, const float* __restrict__ right_matrix, float* __restrict__ result_matrix, int matrix_size) {
    __shared__ float left_block[TILE_ROWS][TILE_COLUMNS];
    __shared__ float right_block[TILE_COLUMNS][TILE_COLUMNS];
    int column = blockIdx.x * TILE_COLUMNS + threadIdx.x;
    int first_row = blockIdx.y * TILE_ROWS + threadIdx.y;
    float row_sums[ROWS_PER_THREAD] = {};

    for (int block_start = 0; block_start < matrix_size; block_start += TILE_COLUMNS) {
        int left_column = block_start + threadIdx.x;
        #pragma unroll
        for (int row_offset = 0; row_offset < ROWS_PER_THREAD; ++row_offset) {
            int local_row = threadIdx.y + row_offset * blockDim.y;
            int matrix_row = blockIdx.y * TILE_ROWS + local_row;
            left_block[local_row][threadIdx.x] = matrix_row < matrix_size && left_column < matrix_size ? left_matrix[matrix_row * matrix_size + left_column] : 0.0f;
        }
        for (int local_row = threadIdx.y; local_row < TILE_COLUMNS; local_row += blockDim.y) {
            int matrix_row = block_start + local_row;
            right_block[local_row][threadIdx.x] = matrix_row < matrix_size && column < matrix_size ? right_matrix[matrix_row * matrix_size + column] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < TILE_COLUMNS; ++inner) {
            float right_value = right_block[inner][threadIdx.x];
            #pragma unroll
            for (int row_offset = 0; row_offset < ROWS_PER_THREAD; ++row_offset) {
                int local_row = threadIdx.y + row_offset * blockDim.y;
                row_sums[row_offset] = fmaf(left_block[local_row][inner], right_value, row_sums[row_offset]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int row_offset = 0; row_offset < ROWS_PER_THREAD; ++row_offset) {
        int row = first_row + row_offset * blockDim.y;
        if (row < matrix_size && column < matrix_size) {
            result_matrix[row * matrix_size + column] = row_sums[row_offset];
        }
    }
}

struct GpuWorkspace {
    float* storage = nullptr;
    size_t capacity = 0;
    cudaStream_t stream;

    GpuWorkspace() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~GpuWorkspace() {
        cudaFree(storage);
        cudaStreamDestroy(stream);
    }

    void Resize(size_t element_count) {
        if (element_count > capacity) {
            cudaFree(storage);
            cudaMalloc(&storage, 3 * element_count * sizeof(float));
            capacity = element_count;
        }
    }
};

}

std::vector<float> BlockGemmCUDA(const std::vector<float>& left_matrix, const std::vector<float>& right_matrix, int matrix_size) {
    if (left_matrix.empty()) {
        return {};
    }

    static GpuWorkspace workspace;
    size_t element_count = static_cast<size_t>(matrix_size) * matrix_size;
    size_t byte_count = element_count * sizeof(float);

    workspace.Resize(element_count);

    float* device_left = workspace.storage;
    float* device_right = device_left + element_count;
    float* device_result = device_right + element_count;

    cudaMemcpyAsync(device_left, left_matrix.data(), byte_count, cudaMemcpyHostToDevice, workspace.stream);
    cudaMemcpyAsync(device_right, right_matrix.data(), byte_count, cudaMemcpyHostToDevice, workspace.stream);

    dim3 threads(TILE_COLUMNS, TILE_ROWS / ROWS_PER_THREAD);
    dim3 blocks((matrix_size + TILE_COLUMNS - 1) / TILE_COLUMNS, (matrix_size + TILE_ROWS - 1) / TILE_ROWS);
    MultiplyBlocks<<<blocks, threads, 0, workspace.stream>>>(device_left, device_right, device_result, matrix_size);

    std::vector<float> result(element_count);
    cudaMemcpyAsync(result.data(), device_result, byte_count, cudaMemcpyDeviceToHost, workspace.stream);
    cudaStreamSynchronize(workspace.stream);

    return result;
}
