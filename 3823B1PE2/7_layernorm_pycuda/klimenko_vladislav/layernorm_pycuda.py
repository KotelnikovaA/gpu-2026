import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


CUDA_SRC = r"""
#include <math.h>

#define WARP_SIZE 32

__device__ __forceinline__
float warp_reduce_sum(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__
void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_input = input + (size_t)row * row_size;
    float*       row_output = output + (size_t)row * row_size;

    __shared__ float shared_sum[WARP_SIZE];
    __shared__ float shared_sq[WARP_SIZE];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    float local_sum = 0.0f;
    float local_sq  = 0.0f;

    if (row_size >= 4 && (row_size & 3) == 0)
    {
        const int n4 = row_size >> 2;
        const float4* in4 = reinterpret_cast<const float4*>(row_input);

        for (int i = tid; i < n4; i += blockDim.x)
        {
            const float4 v = in4[i];
            local_sum += v.x + v.y + v.z + v.w;
            local_sq  += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    }
    else
    {
        for (int i = tid; i < row_size; i += blockDim.x)
        {
            const float x = row_input[i];
            local_sum += x;
            local_sq  += x * x;
        }
    }

    const int lane      = tid & (WARP_SIZE - 1);
    const int warp      = tid >> 5;
    const int warp_count = blockDim.x >> 5;

    local_sum = warp_reduce_sum(local_sum);
    local_sq  = warp_reduce_sum(local_sq);

    if (lane == 0)
    {
        shared_sum[warp] = local_sum;
        shared_sq[warp]  = local_sq;
    }
    __syncthreads();

    if (warp == 0)
    {
        float v_sum = (lane < warp_count) ? shared_sum[lane] : 0.0f;
        float v_sq  = (lane < warp_count) ? shared_sq[lane]  : 0.0f;

        v_sum = warp_reduce_sum(v_sum);
        v_sq  = warp_reduce_sum(v_sq);

        if (lane == 0)
        {
            const float mean = v_sum / (float)row_size;
            float variance   = v_sq / (float)row_size - mean * mean;

            // Guard against tiny negative values from float rounding.
            variance = fmaxf(variance, 0.0f);

            s_mean    = mean;
            s_inv_std = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    const float mean    = s_mean;
    const float inv_std = s_inv_std;

    if (row_size >= 4 && (row_size & 3) == 0)
    {
        const int n4 = row_size >> 2;
        const float4* in4  = reinterpret_cast<const float4*>(row_input);
        const float4* g4   = reinterpret_cast<const float4*>(gamma);
        const float4* b4   = reinterpret_cast<const float4*>(beta);
        float4*       out4 = reinterpret_cast<float4*>(row_output);

        for (int i = tid; i < n4; i += blockDim.x)
        {
            const float4 v = in4[i];
            const float4 g = g4[i];
            const float4 b = b4[i];

            float4 o;
            o.x = ((v.x - mean) * inv_std) * g.x + b.x;
            o.y = ((v.y - mean) * inv_std) * g.y + b.y;
            o.z = ((v.z - mean) * inv_std) * g.z + b.z;
            o.w = ((v.w - mean) * inv_std) * g.w + b.w;

            out4[i] = o;
        }
    }
    else
    {
        for (int i = tid; i < row_size; i += blockDim.x)
        {
            const float x = row_input[i];
            const float normalized = (x - mean) * inv_std;
            row_output[i] = normalized * gamma[i] + beta[i];
        }
    }
}
"""

_module = SourceModule(
    CUDA_SRC,
    options=["--use_fast_math"],
    no_extern_c=True,
)

_layernorm_kernel = _module.get_function("layernorm_kernel")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    if row_size <= 0:
        raise ValueError("row_size must be positive")

    x = np.asarray(input, dtype=np.float32)

    if x.size == 0:
        return np.empty(0, dtype=np.float32)

    if x.size % row_size != 0:
        raise ValueError("input size must be divisible by row_size")

    gamma_np = np.asarray(gamma, dtype=np.float32)
    beta_np  = np.asarray(beta,  dtype=np.float32)

    if gamma_np.size != row_size:
        raise ValueError("gamma size must equal row_size")
    if beta_np.size != row_size:
        raise ValueError("beta size must equal row_size")

    x        = np.ascontiguousarray(x)
    gamma_np = np.ascontiguousarray(gamma_np)
    beta_np  = np.ascontiguousarray(beta_np)

    row_count = x.size // row_size

    threads = 1
    while threads * 2 <= row_size and threads < 256:
        threads *= 2
    if threads < 32:
        threads = 32  

    output = np.empty_like(x)

    d_input  = cuda.mem_alloc(x.nbytes)
    d_gamma  = cuda.mem_alloc(gamma_np.nbytes)
    d_beta   = cuda.mem_alloc(beta_np.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    try:
        cuda.memcpy_htod(d_input, x)
        cuda.memcpy_htod(d_gamma, gamma_np)
        cuda.memcpy_htod(d_beta,  beta_np)

        _layernorm_kernel(
            d_input,
            d_gamma,
            d_beta,
            d_output,
            np.int32(row_size),
            np.float32(eps),
            block=(threads, 1, 1),
            grid=(row_count, 1, 1),
        )

        cuda.memcpy_dtoh(output, d_output)

    finally:
        d_input.free()
        d_gamma.free()
        d_beta.free()
        d_output.free()

    return output.reshape(x.shape)