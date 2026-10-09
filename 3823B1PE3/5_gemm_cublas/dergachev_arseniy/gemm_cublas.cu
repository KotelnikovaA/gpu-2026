#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace {

struct GemmContext {
    cudaStream_t cuda_stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    float* device_matrix_buffer = nullptr;
    std::size_t matrix_capacity_elements = 0;

    GemmContext() {
        cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, cuda_stream);
    }

    ~GemmContext() {
        if (cublas_handle != nullptr) {
            cublasDestroy(cublas_handle);
        }
        if (device_matrix_buffer != nullptr) {
            cudaFree(device_matrix_buffer);
        }
        if (cuda_stream != nullptr) {
            cudaStreamDestroy(cuda_stream);
        }
    }

    void Reserve(std::size_t size) {
        if (size <= matrix_capacity_elements) {
            return;
        }
        if (device_matrix_buffer != nullptr) {
            cudaFree(device_matrix_buffer);
            device_matrix_buffer = nullptr;
            matrix_capacity_elements = 0;
        }
        cudaMalloc(reinterpret_cast<void**>(&device_matrix_buffer),
                   3 * size * sizeof(float));
        matrix_capacity_elements = size;
    }
};

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const std::size_t dimension = static_cast<std::size_t>(n);
    const std::size_t size = dimension * dimension;
    if (size == 0) {
        return {};
    }

    static thread_local GemmContext gemm_context;
    gemm_context.Reserve(size);

    const std::size_t bytes = size * sizeof(float);
    float* device_a = gemm_context.device_matrix_buffer;
    float* device_b = device_a + gemm_context.matrix_capacity_elements;
    float* device_c = device_b + gemm_context.matrix_capacity_elements;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaMemcpyAsync(device_a, a.data(), bytes,
                    cudaMemcpyHostToDevice, gemm_context.cuda_stream);
    cudaMemcpyAsync(device_b, b.data(), bytes,
                    cudaMemcpyHostToDevice, gemm_context.cuda_stream);
    cublasSgemm(gemm_context.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha, device_b, n, device_a, n,
                &beta, device_c, n);

    std::vector<float> result(size);
    cudaMemcpyAsync(result.data(), device_c, bytes,
                    cudaMemcpyDeviceToHost, gemm_context.cuda_stream);
    cudaStreamSynchronize(gemm_context.cuda_stream);
    return result;
}
