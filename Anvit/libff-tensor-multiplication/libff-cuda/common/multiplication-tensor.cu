// multiplication-tensor.cuh
#ifndef LIBFF_MULTIPLICATION_TENSOR_CU
#define LIBFF_MULTIPLICATION_TENSOR_CU

// multiplication-tensor.cu
#include "multiplication-tensor.cuh"
#include "../mini-mp-cuda/mini-mp-cuda.cuh"

namespace libff
{
    __global__ void create_function_local()
    {
    }
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
    __device__ void checkLast(const char *const file, int const line)
    {
        cudaError_t const err{cudaGetLastError()};
        if (err != cudaSuccess)
        {
            printf("%s : on %s : %d\n", cudaGetErrorString(err), file, line);
        }
    }
    __global__ void wmma_gemm_a_col_major_b_col_major(
        uint8_t const *A, uint8_t const *B, int32_t *C, uint32_t m, uint32_t n, uint32_t k,
        uint32_t lda, uint32_t ldb, uint32_t ldc)
    {
        constexpr int WMMA_M{16};
        constexpr int WMMA_N{16};
        constexpr int WMMA_K{16};

        int warps_per_block_x = 3;
        int warps_per_block_y = 3;

        uint32_t const warpM = blockIdx.x * warps_per_block_x + (threadIdx.x / warpSize);

        uint32_t const warpN = blockIdx.y * warps_per_block_y + threadIdx.y;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, nvcuda::wmma::col_major> a_frag{};
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, nvcuda::wmma::col_major> b_frag{};
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag{};
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag{};

        nvcuda::wmma::fill_fragment(acc_frag, static_cast<int32_t>(0));

        // Loop over K.
        for (uint32_t ki{0}; ki < k; ki += WMMA_K)
        {
            // Determine the first element of the mma matrices on the linear memory.
            // Matrix A mma matrix
            uint32_t const matrix_mma_a_row_idx{warpM * WMMA_M};
            uint32_t const matrix_mma_a_col_idx{ki};
            // Matrix B mma matrix
            uint32_t const matrix_mma_b_row_idx{ki};
            uint32_t const matrix_mma_b_col_idx{warpN * WMMA_N};

            // Bounds checking
            if (matrix_mma_a_row_idx < (m) &&
                matrix_mma_a_col_idx < (k) &&
                matrix_mma_b_row_idx < (k) &&
                matrix_mma_b_col_idx < (n))
            {
                uint8_t const *matrix_mma_a_mptr{A + matrix_mma_a_row_idx +
                                                 matrix_mma_a_col_idx * lda};
                uint8_t const *matrix_mma_b_mptr{B + matrix_mma_b_row_idx +
                                                 matrix_mma_b_col_idx * ldb};
                // Load the mma matrix inputs.
                nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
                nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);
                // Perform the matrix multiplication
                nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                __syncwarp();
                __syncthreads();
            }
        }

        uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
        uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

        if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n)
        {
            int32_t *matrix_mma_c_mptr{C + matrix_mma_c_row_idx +
                                       matrix_mma_c_col_idx * ldc};
            nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc, nvcuda::wmma::mem_col_major);

            for (uint32_t i = 0; i < c_frag.num_elements; i++)
            {
                c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
            }
            nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc,
                                            nvcuda::wmma::mem_col_major);
            __syncwarp();
            __syncthreads();
        }
        CHECK_LAST_CUDA_ERROR();
    }
    __device__ void launch_wmma_mm(uint8_t const *A, uint8_t const *B, int32_t *C, uint32_t m, uint32_t n,
                                   uint32_t k)
    {
        uint32_t const lda{m};
        uint32_t const ldb{n};
        uint32_t const ldc{m};

        constexpr int WMMA_M{16};
        constexpr int WMMA_N{16};
        constexpr int WMMA_K{16};

        constexpr int WARP_SIZE{32};

        dim3 gridDim;
        dim3 blockDim;
        int warps_per_block_x = 3;
        int warps_per_block_y = 3;

        int const total_warps_x = m / 16;
        int const total_warps_y = n / 16;

        blockDim.x = warps_per_block_x * WARP_SIZE;
        blockDim.y = warps_per_block_y;

        gridDim.x = (total_warps_x + warps_per_block_x - 1) / warps_per_block_x;
        gridDim.y = (total_warps_y + warps_per_block_y - 1) / warps_per_block_y;

        // printf("warpX: %u, warpY: %u, m: %u, n: %u, k: %u\n", num_warps_x, num_warps_y, m, n, k);

        // printf("GridX: %u, gridY: %u, blockX: %u, blockY: %u\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

        wmma_gemm_a_col_major_b_col_major<<<gridDim, blockDim>>>(A, B, C, m, n, k, lda, ldb, ldc);

        __syncthreads();
        __syncwarp();
        CHECK_LAST_CUDA_ERROR();
    }
    __device__ uint8_t *getMultiplicationResultFromMatrix(int32_t *matrix, int rows, int cols)
    {
        int size = 4;
        uint64_t mask = 0xF;
        uint8_t *result = new uint8_t[rows + cols + 6];
        int64_t *groupSum = new int64_t[rows + cols + 6];
        memset(groupSum, 0, sizeof(int64_t) * (rows + cols + 6));
        memset(result, 0, sizeof(uint8_t) * (rows + cols + 6));
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                groupSum[i + j] += matrix[i * rows + j];
            }
        }
        for (size_t i = 0; i < (rows + cols); i++)
        {
            result[i] = static_cast<uint8_t>(groupSum[i] & mask);
            groupSum[i + 1] += groupSum[i] >> size;
        }
        
        uint8_t *newresult = new uint8_t[rows + cols + 2];
        memset(newresult, 0, sizeof(uint8_t) * (rows + cols + 2));
        for (int i = 0; i < (rows); i++)
        {
            newresult[i] = result[2 * i] | (result[2 * i + 1] << 4);
        }
        
        // delete result;
        // delete groupSum;
        return newresult;
    }
    __device__ unsigned char *getMultiplicationResult(unsigned char *num1, unsigned char *num2, int size1, int size2)
    {
        // printf("Inside multiplication\n");
        int num1_size = ((size1 + 7) / 8);
        int num2_size = ((size2 + 7) / 8);
        int max_size = num1_size > num2_size ? num1_size : num2_size;
        if (max_size == 0)
            max_size = 16;
        else max_size *=2;
        uint32_t padded_size = ((max_size + 15) / 16) * 16;

        uint8_t *h_A = new uint8_t[padded_size * padded_size];
        uint8_t *h_B = new uint8_t[padded_size * padded_size];
        int32_t *h_C = new int32_t[padded_size * padded_size];
        memset(h_A, 0, padded_size * padded_size * sizeof(uint8_t));
        memset(h_B, 0, padded_size * padded_size * sizeof(uint8_t));
        memset(h_C, 0, padded_size * padded_size * sizeof(int32_t));

        int limb1 = (num1_size + (sizeof(mp_limb_t_) - 1)) / (sizeof(mp_limb_t_));
        int limb2 = (num2_size + (sizeof(mp_limb_t_) - 1)) / (sizeof(mp_limb_t_));

        // for (int i = 0; i < limb1; i++)
        // {
        //     for (int j = sizeof(mp_limb_t_) - 1; j >= 0; j--)
        //     {
        //         // h_A[2 * i * sizeof(mp_limb_t_) + 2 * j] = num1[i * sizeof(mp_limb_t_) + j] & 0x0F;
        //         // h_A[2 * i * sizeof(mp_limb_t_) + 2 * j + 1] = (num1[i * sizeof(mp_limb_t_) + sizeof(mp_limb_t_) - j - 1] & 0xF0) >> 4;
        //         h_A[2*i *sizeof(mp_limb_t_) + 2*j] = (num1[i*sizeof(mp_limb_t_) +j] &0xF0)>>4;
        //         h_A[2*i * sizeof(mp_limb_t_) + 2*j + 1] = num1[i*sizeof(mp_limb_t_) +j] &0xF;
        //     }
        // }
        // for (int i = 0; i < limb2; i++)
        // {
        //     for (int j = sizeof(mp_limb_t_) - 1; j >= 0; j--)
        //     {
        //         // h_B[(2 * i * sizeof(mp_limb_t_) + 2 * j) * padded_size] = num2[i * sizeof(mp_limb_t_) + sizeof(mp_limb_t_) - j - 1] & 0x0F;
        //         // h_B[(2 * i * sizeof(mp_limb_t_) + 2 * j + 1) * padded_size] = (num2[i * sizeof(mp_limb_t_) + sizeof(mp_limb_t_) - j - 1] & 0xF0) >> 4;
        //         h_B[(2*i * sizeof(mp_limb_t_) + 2*j)*padded_size] = (num2[i*sizeof(mp_limb_t_) +j] &0xF0)>>4;
        //         h_B[(2*i * sizeof(mp_limb_t_) + 2*j + 1) * padded_size] = num2[i*sizeof(mp_limb_t_) +j] &0xF;
        //     }
        // }
        for(int i=0;i<num1_size;i++)
        {
            h_A[i*2] = num1[i]&0xF;
            h_A[i*2+1] = (num1[i]&0xF0)>>4;
        }
        for(int i=0;i<num2_size;i++)
        {
            h_B[(i*2)*padded_size] = num2[i]&0xF;
            h_B[(i*2+1)*padded_size] = (num2[i]&0xF0)>>4;
        }

        // printf("Operand1 loaded:\n");
        // for(int i=0;i<padded_size;i++)
        //     printf("%02x ",h_A[i]);
        // printf("\nOperand2 loaded:\n");
        // for(int i=0;i<padded_size;i++)
        //     printf("%02x ",h_B[i*padded_size]);
        
        launch_wmma_mm(h_A, h_B, h_C, padded_size, padded_size, padded_size);

        uint8_t *result = getMultiplicationResultFromMatrix(h_C, padded_size, padded_size);
        __syncthreads();
        __syncwarp();
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return result;
    }
    // __device__ int getDataSize(uint8_t *res, int size)
    // {
    //     int last_index = 0;
    //     for (int i = 0; i < size; i++)
    //     {
    //         if (res[i])
    //         {
    //             last_index = i;
    //         }
    //     }
    //     return last_index + 1;
    // }
} // namespace libff

#endif