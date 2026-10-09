import math
import numpy as np
from numba import cuda, float32

THREADS_PER_BLOCK = 256

@cuda.jit(device=True)
def _block_sum(value, partial_sums):
    thread_index = cuda.threadIdx.x
    partial_sums[thread_index] = value
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if thread_index < stride:
            partial_sums[thread_index] += partial_sums[thread_index + stride]
        cuda.syncthreads()
        stride //= 2

    total = partial_sums[0]
    cuda.syncthreads()
    return total

@cuda.jit
def _normalize_rows(values, gamma, beta, row_size, eps):
    row_start = cuda.blockIdx.x * row_size
    thread_index = cuda.threadIdx.x
    partial_sums = cuda.shared.array(THREADS_PER_BLOCK, float32)
    row_reference = values[row_start]

    difference_sum = float32(0)
    for column in range(thread_index, row_size, cuda.blockDim.x):
        difference_sum += values[row_start + column] - row_reference
    mean_difference = _block_sum(difference_sum, partial_sums) / float32(row_size)

    squared_sum = float32(0)
    for column in range(thread_index, row_size, cuda.blockDim.x):
        centered_value = values[row_start + column] - row_reference - mean_difference
        squared_sum += centered_value * centered_value
    variance = _block_sum(squared_sum, partial_sums) / float32(row_size)
    inverse_std = float32(1) / math.sqrt(variance + eps)

    for column in range(thread_index, row_size, cuda.blockDim.x):
        centered_value = values[row_start + column] - row_reference - mean_difference
        normalized_value = centered_value * inverse_std
        values[row_start + column] = normalized_value * gamma[column] + beta[column]

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    input_values = np.ascontiguousarray(input, dtype=np.float32)
    scale_values = np.ascontiguousarray(gamma, dtype=np.float32)
    bias_values = np.ascontiguousarray(beta, dtype=np.float32)
    device_values = cuda.to_device(input_values)
    device_scale = cuda.to_device(scale_values)
    device_bias = cuda.to_device(bias_values)
    row_count = input_values.size // row_size

    _normalize_rows[row_count, THREADS_PER_BLOCK](
        device_values, device_scale, device_bias, row_size, np.float32(eps)
    )
    return device_values.copy_to_host()
