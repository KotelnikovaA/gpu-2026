#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
void CheckCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void CheckCublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + " failed");
}

class Workspace {
public:
    ~Workspace() {
        if (storage_) cudaFree(storage_);
        if (handle_) cublasDestroy(handle_);
    }

    void Ensure(std::size_t bytes) {
        if (!handle_) CheckCublas(cublasCreate(&handle_), "cublasCreate");
        if (bytes <= capacity_) return;
        if (storage_) CheckCuda(cudaFree(storage_), "cudaFree(workspace)");
        CheckCuda(cudaMalloc(&storage_, 3 * bytes), "cudaMalloc(workspace)");
        capacity_ = bytes;
    }

    cublasHandle_t handle() const { return handle_; }
    float* a() const { return storage_; }
    float* b() const { return storage_ + capacity_ / sizeof(float); }
    float* c() const { return storage_ + 2 * capacity_ / sizeof(float); }

private:
    cublasHandle_t handle_ = nullptr;
    float* storage_ = nullptr;
    std::size_t capacity_ = 0;
};
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0 || a.size() != static_cast<std::size_t>(n) * n || b.size() != static_cast<std::size_t>(n) * n)
        throw std::invalid_argument("matrices must have exactly n*n elements");
    std::vector<float> c(a.size(), 0.0f);
    if (n == 0) return c;
    const std::size_t bytes = a.size() * sizeof(float);
    static Workspace workspace;
    workspace.Ensure(bytes);
    CheckCuda(cudaMemcpy(workspace.a(), a.data(), bytes, cudaMemcpyHostToDevice), "copy A");
    CheckCuda(cudaMemcpy(workspace.b(), b.data(), bytes, cudaMemcpyHostToDevice), "copy B");
    constexpr float alpha = 1.0f, beta = 0.0f;
    // Row-major C=A*B is column-major C^T=B^T*A^T.
    CheckCublas(cublasSgemm(workspace.handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                            &alpha, workspace.b(), n, workspace.a(), n,
                            &beta, workspace.c(), n), "cublasSgemm");
    CheckCuda(cudaMemcpy(c.data(), workspace.c(), bytes, cudaMemcpyDeviceToHost), "copy C");
    return c;
}
