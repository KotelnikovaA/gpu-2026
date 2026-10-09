#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockX = 32;      // threads along columns (one warp -> 128 sequential columns)
constexpr int kBlockY = 4;       // threads along rows
constexpr int kRowsPerThread = 8;
constexpr int kColsPerThread = 4;  // one float4
constexpr int kBlockCols = kBlockX * kColsPerThread;
constexpr int kBlockRows = kBlockY * kRowsPerThread;

__device__ __forceinline__ void Fma4(float4& acc, float a, const float4& b) {
    acc.x = fmaf(a, b.x, acc.x);
    acc.y = fmaf(a, b.y, acc.y);
    acc.z = fmaf(a, b.z, acc.z);
    acc.w = fmaf(a, b.w, acc.w);
}

// Each thread computes a kRowsPerThread x 4 tile of C in registers.
// threadIdx.x -> 4 columns: B and C are accessed with sequential float4 within a warp,
// A is a float4 broadcast along k. Per 4 steps of k: 4 loads of B + kRowsPerThread loads of A
// feed 16 * kRowsPerThread FMAs, so the kernel is no longer bound by global memory accesses.
__global__ void NaiveGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int col0 = (blockIdx.x * kBlockX + threadIdx.x) * kColsPerThread;
    const int row0 = (blockIdx.y * kBlockY + threadIdx.y) * kRowsPerThread;

    float4 acc[kRowsPerThread] = {};
    for (int k = 0; k < n; k += 4) {
        float4 bv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) bv[q] = *reinterpret_cast<const float4*>(b + (k + q) * n + col0);
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const float4 av = *reinterpret_cast<const float4*>(a + (row0 + r) * n + k);
            Fma4(acc[r], av.x, bv[0]);
            Fma4(acc[r], av.y, bv[1]);
            Fma4(acc[r], av.z, bv[2]);
            Fma4(acc[r], av.w, bv[3]);
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        *reinterpret_cast<float4*>(c + (row0 + r) * n + col0) = acc[r];
    }
}

// Fallback for small matrices (n < 128): one element per thread
constexpr int kSmallBlock = 16;

__global__ void SmallGemmKernel(const float* a, const float* b, float* c, int n) {
    const int col = blockIdx.x * kSmallBlock + threadIdx.x;
    const int row = blockIdx.y * kSmallBlock + threadIdx.y;
    if (row >= n || col >= n) return;
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
    c[row * n + col] = sum;
}

struct Buffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t capacity = 0;

    void Reserve(size_t count) {
        if (count <= capacity) return;
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaMalloc(&a, count * sizeof(float));
        cudaMalloc(&b, count * sizeof(float));
        cudaMalloc(&c, count * sizeof(float));
        capacity = count;
    }
};

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) return {};
    const size_t count = size_t(n) * n;
    const size_t bytes = count * sizeof(float);

    static Buffers buf;  // device memory is allocated once and reused between calls
    buf.Reserve(count);

    cudaMemcpy(buf.a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf.b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n % kBlockCols == 0 && n % kBlockRows == 0) {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid(n / kBlockCols, n / kBlockRows);
        NaiveGemmKernel<<<grid, block>>>(buf.a, buf.b, buf.c, n);
    } else {
        const int blocks = (n + kSmallBlock - 1) / kSmallBlock;
        SmallGemmKernel<<<dim3(blocks, blocks), dim3(kSmallBlock, kSmallBlock)>>>(buf.a, buf.b, buf.c, n);
    }

    // Kernel launch is asynchronous, so the result is allocated on host while it runs
    std::vector<float> c(count);
    cudaMemcpy(c.data(), buf.c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
