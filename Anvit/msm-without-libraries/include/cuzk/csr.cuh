#ifndef __CSR_CUH__
#define __CSR_CUH__


#include "vector.cuh"

template<typename T>
class CSR_matrix
{
public:
    vector<size_t> row_ptr;
    vector<size_t> col_idx;
    vector<T> data;

    size_t row_size;
    size_t col_size;
};

template<typename T>
struct CSR_matrix_host
{
    libstl::vector<size_t> row_ptr;
    libstl::vector<size_t> col_idx;
    libstl::vector<T> data;

    size_t row_size;
    size_t col_size;
};

template<typename T, typename H>
void CSR_matrix_device2host(CSR_matrix_host<H>* hcsr, CSR_matrix<T>* dcsr);

template<typename T, typename H>
void CSR_matrix_host2device(CSR_matrix<T>* dcsr, CSR_matrix_host<H>* hcsr);

template<typename T>
void CSR_matrix_device2host(CSR_matrix_host<T>* hcsr, CSR_matrix<T>* dcsr);

template<typename T>
void CSR_matrix_host2device(CSR_matrix<T>* dcsr, CSR_matrix_host<T>* hcsr);



struct col_idx_data_addr
{
    size_t col_idx;
    size_t data_addr;
};


template<typename T>
class CSR_matrix_opt
{
public:
    vector<size_t> row_ptr;
    vector<col_idx_data_addr> col_data;

    size_t row_size;
    size_t col_size;
};


#include "csr.cu"

#endif