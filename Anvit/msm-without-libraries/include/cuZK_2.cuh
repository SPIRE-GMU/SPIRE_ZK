#ifndef __CUZK2_CUH
#define __CUZK2_CUH
#include <cuda_runtime.h>
#include "./vector.cuh"
#include "./FieldG1.cuh"
#include "./Scalar.cuh"
#include "./G1Point.cuh"
#include "./../utils/bls12_381_constants.cu"
#include "./../utils/utils.cu"

class ELL_matrix_opt
{
public:
    Vector<size_t> row_length;
    Vector<size_t> col_idx;
    Vector<G1Point *> data_addr;

    size_t max_row_length;
    size_t row_size;
    size_t col_size;
    size_t total;

    __device__ bool init(size_t max_row_length, size_t row_size, size_t col_size);
    __device__ bool insert(size_t row, size_t col);
    __host__ bool reset(size_t gridSize, size_t blockSize);
};

struct col_idx_data_addr
{
    size_t col_idx;
    size_t data_addr;
};

class CSR_matrix_opt
{
public:
    Vector<size_t> row_ptr;
    Vector<col_idx_data_addr> col_data;

    size_t row_size;
    size_t col_size;
};

class cuZK
{
public:
    Scalar *scalars;
    G1Point *points;
    ELL_matrix_opt *ell_matrix;
    CSR_matrix_opt *csr_matrix;
    Vector<G1Point> *v_buckets;
    Vector<G1Point> *mid_res;
    G1Point *t_instance;
    Scalar *instance;
    G1Point *result;

    size_t total;
    size_t gridSize;
    size_t blockSize;
    size_t num_bits;
    size_t group_grid;
    size_t window_size;
    size_t num_groups;
    size_t sgroup, egroup;
    size_t tnum, max_length;

    cuZK(G1Point *points, Scalar *scalars, size_t total_num);
    ~cuZK();
    cudaEvent_t start, stop;

    void generate_ell_matrix(G1Point *points, Scalar *scalars, size_t total_num);
    void transpose_ell2csr(ELL_matrix_opt **ell_matrix, CSR_matrix_opt *csr_matrix);
    void sparse_matrix_vector_multiplication(int group);
    void sum_buckets();
    void accumulate_result();
    G1Point *run();
    size_t get_runtime();
};

#include "./../src/cuZK_2.cu"

#endif