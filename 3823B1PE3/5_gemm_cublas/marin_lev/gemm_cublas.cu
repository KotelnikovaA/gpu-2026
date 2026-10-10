#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <stdexcept>

namespace {

void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void CheckBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error");
    }
}

struct Workspace {
    cublasHandle_t handle = nullptr;
    float* data = nullptr;
    std::size_t capacity = 0;

    Workspace() {
        CheckBlas(cublasCreate(&handle));
    }

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity) return;

        float* next = nullptr;
        CheckCuda(cudaMalloc(&next, bytes));
        cudaFree(data);
        data = next;
        capacity = bytes;
    }

    ~Workspace() {
        cublasDestroy(handle);
        cudaFree(data);
    }
};

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                            const std::vector<float>& b,
                            int n) {
    if (n <= 0) return {};

    const std::size_t count = static_cast<std::size_t>(n) * n;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("Incorrect matrix size");
    }

    const std::size_t bytes = count * sizeof(float);
    static thread_local Workspace workspace;
    workspace.Reserve(3 * bytes);

    float* device_a = workspace.data;
    float* device_b = device_a + count;
    float* device_c = device_b + count;

    CheckCuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CheckBlas(cublasSgemm(workspace.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         n, n, n, &alpha, device_b, n, device_a, n,
                         &beta, device_c, n));

    std::vector<float> c(count);
    CheckCuda(cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost));
    return c;
}
