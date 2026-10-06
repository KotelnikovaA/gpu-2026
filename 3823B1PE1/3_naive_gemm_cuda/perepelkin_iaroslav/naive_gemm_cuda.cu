#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void GemmKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_vec = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n && col_vec * 4 < n) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        #pragma unroll 8
        for (int k = 0; k < n; ++k) {
            float a_val = A[row * n + k];
            
            float4 b_vec = reinterpret_cast<const float4*>(&B[k * n + col_vec * 4])[0];

            sum.x += a_val * b_vec.x;
            sum.y += a_val * b_vec.y;
            sum.z += a_val * b_vec.z;
            sum.w += a_val * b_vec.w;
        }

        reinterpret_cast<float4*>(&C[row * n + col_vec * 4])[0] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n == 0) {
        return {};
    }

    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t current_capacity = 0;

    if (elements > current_capacity) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        current_capacity = elements;
    }

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(32, 8); 
    dim3 blocks((n / 4 + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    
    GemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    std::vector<float> result(elements);

    cudaMemcpyAsync(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return result;
}
