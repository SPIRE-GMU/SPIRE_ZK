// multiplication-tensor.cuh
#ifndef LIBFF_MULTIPLICATION_TENSOR_CUH
#define LIBFF_MULTIPLICATION_TENSOR_CUH

#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <vector>
#include <cuda.h>

namespace libff
{

    __global__ void create_function_local();
    __global__ void wmma_gemm_a_col_major_b_col_major(
        uint8_t const *A, uint8_t const *B, int32_t *C, uint32_t m, uint32_t n, uint32_t k,
        uint32_t lda, uint32_t ldb, uint32_t ldc);
    __device__ void launch_wmma_mm(uint8_t const *A, uint8_t const *B, int32_t *C, uint32_t m, uint32_t n,
                                   uint32_t k);
    __device__ uint8_t *getMultiplicationResultFromMatrix(int32_t *matrix, int rows, int cols);
    __device__ unsigned char *getMultiplicationResult(unsigned char *num1, unsigned char *num2, int size1, int size2);
    // __device__ int getDataSize(uint8_t *res, int size);
} // namespace libff

#endif // LIBFF_MULTIPLICATION_TENSOR_CUH
