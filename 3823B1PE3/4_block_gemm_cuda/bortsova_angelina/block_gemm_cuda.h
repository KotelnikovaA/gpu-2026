#ifndef BLOCK_GEMM_CUDA_H
#define BLOCK_GEMM_CUDA_H
#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& left_matrix, const std::vector<float>& right_matrix, int matrix_size);

#endif
