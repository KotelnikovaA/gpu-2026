#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int ROWS_PER_THREAD = TILE_SIZE / BLOCK_Y;

__global__ void BlockGemmKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int gx = blockIdx.x * TILE_SIZE + tx;
    const int gy = blockIdx.y * TILE_SIZE + ty;

    float sum[ROWS_PER_THREAD] = {0.0f};

    for (int k = 0; k < n; k += TILE_SIZE) {
        // Load tile from global to shared memory
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            sA[ty + i * BLOCK_Y][tx] = A[(gy + i * BLOCK_Y) * n + (k + tx)];
            sB[ty + i * BLOCK_Y][tx] = B[(k + ty + i * BLOCK_Y) * n + gx];
        }
        
        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            const float b_val = sB[i][tx];
            
            #pragma unroll
            for (int j = 0; j < ROWS_PER_THREAD; ++j) {
                sum[j] += sA[ty + j * BLOCK_Y][i] * b_val;
            }
        }
        
        __syncthreads();
    }

    // Write back to global memory
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        C[(gy + i * BLOCK_Y) * n + gx] = sum[i];
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n == 0) return {};

    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static int current_n = 0;

    if (n > current_n) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        current_n = n;
    }

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks(n / TILE_SIZE, n / TILE_SIZE);
    
    BlockGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    std::vector<float> result(elements);

    cudaMemcpyAsync(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return result;
}
