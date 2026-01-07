#ifndef CUZK2_CU
#define CUZK2_CU
#include <cuda_runtime.h>
#include "./../include/vector.cuh"
#include "./../include/FieldG1.cuh"
#include "./../include/Scalar.cuh"
#include "./../include/G1Point.cuh"
#include "./../include/cuZK_2.cuh"
#include "./../utils/bls12_381_constants.cu"
#include "./../utils/utils.cu"

/*
ELL Matrix methods
*/
__device__ bool ELL_matrix_opt::init(size_t max_row_length, size_t row_size, size_t col_size)
{
    this->max_row_length = max_row_length;
    this->row_size = row_size;
    this->col_size = col_size;
    size_t zero = 0;
    this->row_length.resize_fill(row_size, zero);
    this->col_idx.resize(row_size * max_row_length);
    return true;
}
__device__ bool ELL_matrix_opt::insert(size_t row, size_t col)
{
    size_t row_ptr = row * this->max_row_length;
    size_t idx = row_ptr + this->row_length[row];
    this->row_length[row] += 1;
    this->col_idx[idx] = col;

    return true;
}
__device__ bool ELL_matrix_opt::reset(size_t gridSize, size_t blockSize)
{
    launch<<<gridSize, blockSize>>>([=] __device__()
                                    {

    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tnum = gridDim.x * blockDim.x;
    while (tid < this->row_length.size())
    {
        this->row_length[tid] = 0;
        tid += tnum;
    }
    __syncthreads(); });
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

/*
    cuzk methods
*/

template <typename T>
__global__ void construct_vector(Vector<T> *vec_ptr, size_t size)
{
    T *s;
    s = new T();
    vec_ptr->resize_fill(size, *s);
}
__global__ void construct_ell_matrix(Scalar *scalars, ELL_matrix_opt **ell_matrix, size_t group, size_t total, size_t window_size)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tnum = gridDim.x * blockDim.x;
    size_t range_s = (total + tnum - 1) / tnum * tid;
    size_t range_e = (total + tnum - 1) / tnum * (tid + 1);
    for (size_t i = range_s; i < range_e && i < total; i++)
    {
        size_t id = 0;
        auto bn_scalar = scalars[i];
        for (size_t j = 0; j < window_size; j++)
        {
            if (bn_scalar.test_bit(group * window_size + j))
            {
                id |= 1 << j;
            }
        }
        (*ell_matrix)->insert(tid, id);
    }

    if (tid == 0)
        (*ell_matrix)->total = total;
}

__global__ void init_ell_matrix(ELL_matrix_opt **ell, size_t total, size_t tnum, size_t max_length, size_t window_size)
{
    (*ell)->init(max_length, tnum, (size_t)1<<window_size);
}

__global__ void test_kernel(Vector<G1Point> *v_buckets, size_t dsgroup, size_t degroup, size_t dwindow_size)
{
    v_buckets = (Vector<G1Point> *)malloc(sizeof(Vector<G1Point>) * (degroup - dsgroup));
    v_buckets[0].resize((size_t)1 << dwindow_size);
}
cuZK::cuZK(G1Point *points, Scalar *scalars, size_t total_num)
{
    printf("Hello world!\n");
    this->points = points;
    this->scalars = scalars;

    this->total = total_num;
    this->gridSize = 512;
    this->blockSize = 32;
    this->num_bits = 255;
    this->window_size = 16;
    this->num_groups = (num_bits + window_size - 1) / window_size;
    this->sgroup = 0;
    this->egroup = num_groups;

    this->group_grid = 1;
    while (group_grid * 2 < gridSize / (egroup - sgroup))
        group_grid *= 2;
    
    CUDA_CHECK(cudaMalloc(&ell_matrix, sizeof(ELL_matrix_opt)));
    this->tnum = gridSize * blockSize;
    this->max_length = (total + tnum - 1) / tnum;
    init_ell_matrix<<<1, 1>>>(&ell_matrix, total, tnum, max_length, window_size);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        printf("Error: %s, File: %s, Line: %d\n", cudaGetErrorName(e), __FILE__, __LINE__);
    CUDA_CHECK(cudaMalloc(&v_buckets, sizeof(Vector<G1Point>) * num_groups));
    CUDA_CHECK(cudaMalloc(&mid_res, sizeof(Vector<G1Point>) * num_groups));

    for (int i = 0; i < num_groups; ++i)
    {
        
        Vector<G1Point> temp_buckets(((size_t)1 << window_size)+10), temp_res(group_grid * blockSize + 10);
        CUDA_CHECK(cudaMemcpy(&v_buckets[i], &temp_buckets, sizeof(Vector<G1Point>), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&mid_res[i], &temp_res, sizeof(Vector<G1Point>), cudaMemcpyHostToDevice));
    }


    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Initialized parameters\n");
}
cuZK::~cuZK()
{
}
__global__ void reset_ell_matrix(ELL_matrix_opt **ell)
{
    (*ell)->reset(512,32);
}
void cuZK::generate_ell_matrix(G1Point *points, Scalar *scalars, size_t total_num)
{
    printf("Generating ell matrix\n");
    CSR_matrix_opt *csr = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&csr, sizeof(CSR_matrix_opt)));
    this->csr_matrix = csr;
    for (int i = sgroup; i < egroup; ++i)
    {
        printf("Generating ell matrix for group %d\n", i);
        
        CUDA_CHECK(cudaStreamSynchronize(0));
        construct_ell_matrix<<<gridSize, blockSize>>>(scalars, &ell_matrix, i, total_num, window_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Constructed ell matrix for group %d\n", i);
        transpose_ell2csr(&ell_matrix, csr_matrix);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Transposed ell matrix to CSR for group %d\n", i);

        sparse_matrix_vector_multiplication(i);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("before reset\n");
        init_ell_matrix<<<1, 1>>>(&ell_matrix, total, tnum, max_length, window_size);
        printf("Sparse matrix vector multiplication for group %d\n", i);
        CUDA_CHECK(cudaDeviceSynchronize());
        
    }
}
__global__ void init_csr(CSR_matrix_opt *pres, ELL_matrix_opt *mtx, size_t *pizero)
{
    printf("mtx->row_size: %lu, mtx->col_size: %lu, mtx->total: %lu\n", mtx->row_size, mtx->col_size, mtx->total);
    pres->col_size = mtx->row_size;
    printf("pres->col_size: %lu\n", pres->col_size);
    pres->row_size = mtx->col_size;
    printf("pres->row_size: %lu\n", pres->row_size);
    *pizero = 0;
    pres->row_ptr.resize_fill(mtx->col_size + 1, *pizero); //(mtx->total + 1, *pizero);//
    printf("pres->row_ptr.size(): %lu\n", pres->row_ptr.size());
    pres->col_data.resize(mtx->total); 
    printf("pres->col_data.size(): %lu\n", pres->col_data.size());
    
}
// count how many values go to each transposed rows
__global__ void scan_ell(ELL_matrix_opt *mtx, CSR_matrix_opt *pres, Vector<size_t> **p_offset)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t t_num = blockDim.x * gridDim.x;
    while (tid < mtx->row_size)
    {
        size_t ptr = tid * mtx->max_row_length;
        for (size_t i = 0; i < mtx->row_length[tid]; i++)
        {
            (**p_offset)[ptr + i] = atomicAdd((unsigned long long *)&pres->row_ptr[mtx->col_idx[ptr + i] + 1], (unsigned long long)1);
        }
        tid += t_num;
    }
}

// convert counts to prefix sums
__global__ void update_csr_row_ptr(CSR_matrix_opt *pres)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t t_num = blockDim.x * gridDim.x;
    size_t range_s = (pres->row_ptr.size() + t_num - 1) / t_num * tid;
    size_t range_e = (pres->row_ptr.size() + t_num - 1) / t_num * (tid + 1);
    for (size_t i = range_s + 1; i < range_e && i < pres->row_ptr.size(); i++)
    {
        pres->row_ptr[i] += pres->row_ptr[i - 1];
    }
}
// optimize prefix sum
__global__ void update_csr_row_ptr2(CSR_matrix_opt **pres, size_t level, size_t stride, size_t limit)
{

    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t idx = tid / (1 << level) * (1 << (level + 1)) + (1 << level) + tid % (1 << level);
    size_t widx = (idx + 1) * stride - 1;
    size_t ridx = (idx / (1 << level) * (1 << level)) * stride - 1;
    
    // if(widx >limit || ridx > limit) printf("Wrong indexing, limit: %lu, ridx: %lu, widx: %lu\n", limit, ridx, widx);

    (*pres)->row_ptr[widx] = (*pres)->row_ptr[widx] + (*pres)->row_ptr[ridx];
}
// optimize prefix sum
__global__ void sum_scanned_tree(CSR_matrix_opt *pres)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t t_num = blockDim.x * gridDim.x;
    size_t range_s = (pres->row_ptr.size() + t_num - 1) / t_num * (tid + 1);
    size_t range_e = (pres->row_ptr.size() + t_num - 1) / t_num * (tid + 2) - 1;
    for (size_t i = range_s; i < range_e && i < pres->row_ptr.size(); i++)
    {
        pres->row_ptr[i] += pres->row_ptr[range_s - 1];
    }
}
// construct csr data
__global__ void update_col_data(ELL_matrix_opt *mtx, CSR_matrix_opt *pres, Vector<size_t> **p_offset, size_t total)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t t_num = blockDim.x * gridDim.x;
    while (tid < mtx->row_size)
    {
        for (size_t i = 0; i < mtx->row_length[tid]; i++)
        {
            size_t ridx = tid * mtx->max_row_length + i;
            size_t widx = (**p_offset)[ridx] + pres->row_ptr[mtx->col_idx[ridx]];
            pres->col_data[widx] = {tid, ridx};
        }
        tid += t_num;
    }
}
void cuZK::transpose_ell2csr(ELL_matrix_opt **mtx, CSR_matrix_opt *pres)
{
    printf("Transpose started\n");

    // pres = create_host<CSR_matrix_opt>();
    // CSR_matrix_opt *csr = nullptr;
    // CUDA_CHECK(cudaMalloc((void **)&csr, sizeof(CSR_matrix_opt)));
    // pres = csr;
    // this->csr_matrix = csr;
    size_t *pizero;
    CUDA_CHECK(cudaMalloc(&pizero, sizeof(size_t)));
    CUDA_CHECK(cudaDeviceSynchronize());
    init_csr<<<1, 1>>>(pres, *mtx, pizero);
    // gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamSynchronize(0));

    printf("CSR initiated\n");

    size_t row_size;
    size_t col_size;
    size_t total_length;
    size_t col_idx_size;

    CUDA_CHECK(cudaMemcpy(&row_size, &(*mtx)->row_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&col_size, &(*mtx)->col_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&total_length, &(*mtx)->total, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&col_idx_size, &(*mtx)->col_idx._size, sizeof(size_t), cudaMemcpyDeviceToHost));


    printf("CSR vectors resized\n");

    Vector<size_t> *p_offset;
    CUDA_CHECK(cudaMalloc(&p_offset, sizeof(Vector<size_t>)));

    construct_vector<<<1,1>>>(p_offset, col_idx_size);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamSynchronize(0));

    printf("Before scanning\n");

    scan_ell<<<gridSize, blockSize>>>(*mtx, pres, &p_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamSynchronize(0));
    printf("Scanning done\n");

    update_csr_row_ptr<<<gridSize, blockSize>>>(pres);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamSynchronize(0));
    printf("Updating csr\n");
    size_t t_num = gridSize * blockSize;
    size_t stride = (col_size + 1 + t_num - 1) / t_num;

    for (size_t i = 0; i < log2(t_num); i++)
    {
        printf("t_num: %lu, i: %lu, log2(t_num): %lu\n", t_num, i, log2(t_num));
        update_csr_row_ptr2<<<gridSize / 2, blockSize>>>(&pres, i, stride, col_size+1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    printf("Update done csr\n");
    sum_scanned_tree<<<gridSize, blockSize>>>(pres);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamSynchronize(0));

    printf("Tree sum counted\n");
    update_col_data<<<gridSize, blockSize>>>(*mtx, pres, &p_offset, total);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamSynchronize(0));
    printf("Transposition complete\n");
}

__global__ void spmv_small_rows(size_t z, CSR_matrix_opt *mtx, G1Point **vec, Vector<G1Point> *res, size_t total)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tnum = blockDim.x * gridDim.x;
    while (idx < mtx->row_size)
    {
        printf("What is going on?\n");
        size_t s = mtx->row_ptr[idx];
        size_t e = mtx->row_ptr[idx + 1];
        if (e - s < z)
        {
            for (size_t i = s; i < e; i++)
            {
                if((*res).size() <= idx) printf("Wrong index inside spmv small. res.size(): %lu, idx: %lu\n", (*res).size(),idx);
                if(i>mtx->col_data.size()) printf("Wrong index inside spmv small. mtx->col_data.size(): %lu, i: %lu\n", mtx->col_data.size(), i);
                if(mtx->col_data[i].data_addr >total) printf("Wrong index inside spmv small. mtx->col_data[%d].data_addr: %lu, total: %lu\n", i, mtx->col_data[i].data_addr, total);
                (*res)[idx] = (*res)[idx].mixed_add((*vec)[mtx->col_data[i].data_addr]);
            }
        }
        idx += tnum;
    }
}
__global__ void spmv_large_rows_partial_sums(size_t s, size_t e, CSR_matrix_opt *mtx, G1Point *vec, Vector<G1Point> *p_max_mid_res)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t idx = s + tid;
    size_t tnum = blockDim.x * gridDim.x;
    while (idx < e)
    {
        (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid].mixed_add(vec[mtx->col_data[idx].data_addr]);
        idx += tnum;
    }
}
__global__ void spmv_large_rows_reduction(size_t count, size_t part, Vector<G1Point> *p_max_mid_res, Vector<G1Point> *res)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid % (count * 2) == 0)
    {
        if (tid + count < gridDim.x * blockDim.x)
        {
            (*p_max_mid_res)[tid] = (*p_max_mid_res)[tid] + (*p_max_mid_res)[tid + count];
        }
    }
    if (tid == 0)
        (*res)[part] = (*p_max_mid_res)[0];
}
__global__ void spmv_very_large_rows(size_t s, size_t e, size_t n_other_total_idx, size_t ptr, size_t row_id, G1Point *instance, G1Point *vec, CSR_matrix_opt *mtx, Vector<G1Point> *c_mid_data, Vector<size_t> *c_mid_idx, unsigned char *smem)
{
    G1Point zero = instance->zero();
    G1Point *s_mid_res = (G1Point *)smem;
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t idx = s + tid;
    size_t bid = threadIdx.x;
    size_t gid = blockIdx.x;
    size_t tnum = blockDim.x * gridDim.x;
    s_mid_res[bid] = zero;

    while (idx < e)
    {
        s_mid_res[bid] = s_mid_res[bid].mixed_add(vec[mtx->col_data[idx].data_addr]);
        idx += tnum;
    }
    __syncthreads();

    // reduce block
    size_t b_count = blockDim.x;
    size_t count = 1;
    while (b_count != 1)
    {
        if (bid % (count * 2) == 0)
        {
            if (bid + count < blockDim.x)
            {
                s_mid_res[bid] = s_mid_res[bid] + s_mid_res[bid + count];
            }
        }
        __syncthreads();
        b_count = (b_count + 1) / 2;
        count *= 2;
    }
    if (bid == 0)
    {
        (*c_mid_data)[ptr + gid] = s_mid_res[0];
    }
    if (tid == 0)
    {
        (*c_mid_idx)[n_other_total_idx] = row_id;
    }
}
__global__ void spmv_reduce_very_large_rows(size_t n_other_total, Vector<size_t> *c_mid_row, Vector<G1Point> *c_mid_data, Vector<size_t> *c_mid_idx, Vector<G1Point> *res)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tnum = blockDim.x * gridDim.x;
    while (idx < n_other_total)
    {
        size_t s = (*c_mid_row)[idx];
        size_t e = (*c_mid_row)[idx + 1];
        size_t addr = (*c_mid_idx)[idx];
        for (size_t i = s; i < e; i++)
        {
            (*res)[addr] = (*res)[addr] + (*c_mid_data)[i];
        }
        idx += tnum;
    }
}
__global__ void print_csr_matrix_debug(CSR_matrix_opt *mtx)
{
    printf("Col_data.size(): %lu, col_size: %lu, row_ptr.size(): %lu, row_size: %lu\n",mtx->col_data.size(),mtx->col_size, mtx->row_ptr.size(),mtx->row_size);
    printf("row_ptr._data: %lu\n", mtx->row_ptr._data);
    // for(int i=0;i<mtx->row_ptr.size(); i++)
    //     printf("%d : %lu ", i, mtx->row_ptr._data[i]);
    // for(int i=0;i<mtx->col_data.size(); i++)
    //     printf("%d : %016x %016x ", i, mtx->col_data[i].col_idx, mtx->col_data[i].data_addr);
}
__global__ void copy_device_to_device(size_t *dst, size_t *src, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while(idx <size)
    {
        dst[idx] = src[idx];
        idx += stride;
    }
}
void cuZK::sparse_matrix_vector_multiplication(int group)
{
    printf("SPMV Start\n");
    CSR_matrix_opt *mtx = this->csr_matrix;
    Vector<G1Point> *res = &this->v_buckets[group];
    print_csr_matrix_debug<<<1,1>>>(mtx);
    printf("initiated pointers\n");
    cudaStream_t streamT;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamT, cudaStreamNonBlocking));

    size_t mtx_row_size;
    CUDA_CHECK(cudaMemcpy(&mtx_row_size, &mtx->row_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    
    printf("Debug 1\n");
    size_t t = gridSize * blockSize;
    size_t B = blockSize;
    size_t G = gridSize;
    size_t z = B * 5 * 2;

    printf("MTX ROW SIZE: %lu\n", mtx_row_size);

    size_t *hrow_ptr;// = new size_t[mtx_row_size + 1];
    CUDA_CHECK(cudaHostAlloc(&hrow_ptr, (mtx_row_size+1)*sizeof(size_t), cudaHostAllocMapped));
    // CUDA_CHECK(cudaMallocHost(&hrow_ptr, (mtx_row_size+1)*sizeof(size_t)));
    assert(hrow_ptr != nullptr);
    void *mtx_row_addr;

    CUDA_CHECK(cudaMemcpy(&mtx_row_addr, &mtx->row_ptr._data, sizeof(void *), cudaMemcpyDeviceToHost));

    
    size_t *d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(size_t) * (mtx_row_size+1))); 

    CUDA_CHECK(cudaDeviceSynchronize());
    copy_device_to_device<<<gridSize, blockSize>>>(d_ptr, (size_t *)mtx_row_addr, mtx_row_size+1);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("debug 2\n");
    CUDA_CHECK(cudaMemcpyAsync(hrow_ptr, d_ptr, (mtx_row_size + 1) * sizeof(size_t), cudaMemcpyDeviceToHost, streamT));
    
    CUDA_CHECK(cudaStreamSynchronize(streamT));
    printf("Setup for spmv complete\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    spmv_small_rows<<<gridSize, blockSize>>>(z, mtx, &points, res, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Small rows summed\n");

    // max
    size_t Gz = G * z;
    for (size_t i = 0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if (e - s < Gz)
            continue;

        Vector<G1Point> *p_max_mid_res;
        CUDA_CHECK(cudaMalloc(&p_max_mid_res, sizeof(Vector<G1Point>)));

        printf("Came here?\n");
        launch<<<1, 1>>>([=] __device__(Vector<G1Point> * p_max_mid_res, size_t t)
                         { p_max_mid_res->resize(t); }, p_max_mid_res, t);
        printf("Staring large rows partial sum\n");
        spmv_large_rows_partial_sums<<<gridSize, blockSize>>>(s, e, mtx, points, p_max_mid_res);
        CUDA_CHECK(cudaDeviceSynchronize());

        // reduction
        size_t t_count = t;
        size_t count = 1;
        while (t_count != 1)
        {
            printf("Reducing partial sum result\n");
            spmv_large_rows_reduction<<<gridSize, blockSize>>>(count, i, p_max_mid_res, res);
            CUDA_CHECK(cudaDeviceSynchronize());
            t_count = (t_count + 1) / 2;
            count *= 2;
        }
    }
    printf("Debug 3\n");
    size_t n_other_total = 0;
    for (size_t i = 0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if ((e - s >= Gz) || (e - s < z))
            continue;
        n_other_total += 1;
    }
    printf("Debug 4, n_other_total = %lu\n", n_other_total);
    if (n_other_total == 0)
    {
        // for (int i = 0; i < gridSize + 1; i++) cudaStreamDestroy(streams[i]);
        CUDA_CHECK(cudaFreeHost(hrow_ptr));
        CUDA_CHECK(cudaStreamDestroy(streamT));
        return;
    }

    printf("Before creating stream\n");
    cudaStream_t streams[G];
    for (int i = 0; i < G; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // libstl::vector<size_t> row_ptr;
    // row_ptr.resize_host(n_other_total + 1);
    size_t *row_ptr = new size_t[n_other_total + 1];
    row_ptr[0] = 0;

    size_t n_other_total_idx = 0;
    for (size_t i = 0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if ((e - s >= Gz) || (e - s < z))
            continue;

        size_t n = (e - s) / z;
        row_ptr[n_other_total_idx + 1] = row_ptr[n_other_total_idx] + n;

        n_other_total_idx += 1;
    }

    Vector<G1Point> *c_mid_data; //= new Vector<G1Point>(row_ptr[n_other_total]);
    Vector<size_t> *c_mid_row;   //= new Vector<size_t>(n_other_total+1);
    Vector<size_t> *c_mid_idx;   //= new Vector<size_t>(n_other_total);

    CUDA_CHECK(cudaMalloc(&c_mid_data, sizeof(Vector<G1Point>)));
    CUDA_CHECK(cudaMalloc(&c_mid_row, sizeof(Vector<size_t>)));
    CUDA_CHECK(cudaMalloc(&c_mid_idx, sizeof(Vector<size_t>)));

    construct_vector<<<1, 1>>>(c_mid_data, row_ptr[n_other_total]);
    construct_vector<<<1, 1>>>(c_mid_row, n_other_total + 1);
    construct_vector<<<1, 1>>>(c_mid_idx, n_other_total);

    void *c_mid_row_addr;
    CUDA_CHECK(cudaMemcpy(&c_mid_row_addr, &c_mid_row->_data, sizeof(void *), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync((void *)c_mid_row_addr, row_ptr, (n_other_total + 1) * sizeof(size_t), cudaMemcpyHostToDevice, streamT));

    // vector_host2device(c_mid_row, &row_ptr, streams[gridSize]);

    size_t stream_id = 0;
    n_other_total_idx = 0;
    for (size_t i = 0; i < mtx_row_size; i++)
    {
        size_t s = hrow_ptr[i];
        size_t e = hrow_ptr[i + 1];
        if ((e - s > G * z) || (e - s < z))
            continue;
        printf("Starting very large rows\n");
        size_t stream_G = (e - s) / z;
        size_t ptr = row_ptr[n_other_total_idx];
        spmv_very_large_rows<<<stream_G, blockSize, blockSize * sizeof(G1Point), streams[stream_id]>>>(s, e, n_other_total_idx, ptr, i, t_instance, points, mtx, c_mid_data, c_mid_idx, (unsigned char *)stream_id);
        CUDA_CHECK(cudaStreamSynchronize(streams[stream_id]));
        CUDA_CHECK(cudaDeviceSynchronize());
        stream_id = (stream_id + 1) % G;
        n_other_total_idx += 1;
    }

    for (size_t i = 0; i < G; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    CUDA_CHECK(cudaStreamSynchronize(streamT));

    spmv_reduce_very_large_rows<<<gridSize, blockSize>>>(n_other_total, c_mid_row, c_mid_data, c_mid_idx, res);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < G; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaStreamDestroy(streamT));

    delete[] hrow_ptr;
    delete[] row_ptr;
}

__global__ void sum_bucket_1(G1Point *vec, const Scalar *scalar, G1Point &t_instance, Vector<G1Point> *v_buckets, Vector<G1Point> *mid_res, size_t group_grid)
{
    size_t gid = blockIdx.x / group_grid;
    size_t gnum = group_grid * blockDim.x;
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gtid = tid % gnum;

    size_t total = v_buckets[gid].size();
    size_t range_s = (total + gnum - 1) / gnum * gtid;
    size_t range_e = (total + gnum - 1) / gnum * (gtid + 1);

    G1Point result = t_instance.zero();
    G1Point running_sum = t_instance.zero();
    Vector<G1Point> *buckets = v_buckets + gid;
    for (size_t i = range_e > total ? total - 1 : range_e - 1; i >= range_s && i > 0; i--)
    {
        running_sum = running_sum + (*buckets)[i];
        result = result + running_sum;
    }

    if (range_s != 0)
        result = result - running_sum;

    result = result + running_sum * range_s;

    mid_res[gid][gtid] = result;
}

__global__ void sum_bucket_2(size_t group_grid, size_t count, Vector<G1Point> *mid_res)
{
    size_t gid = blockIdx.x / group_grid;
    size_t gnum = group_grid * blockDim.x;
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gtid = tid % gnum;

    // reduce local sums to row sum
    if (gtid % (count * 2) == 0)
    {
        if (gtid + count < group_grid * blockDim.x)
        {
            mid_res[gid][gtid] = mid_res[gid][gtid] + mid_res[gid][gtid + count];
        }
    }
}
void cuZK::sum_buckets()
{
    sum_bucket_1<<<group_grid *(egroup - sgroup), blockSize>>>(points, scalars, *t_instance, v_buckets, mid_res, group_grid);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t t_count = group_grid * blockSize;
    size_t count = 1;
    while (t_count != 1)
    {
        sum_bucket_2<<<group_grid *(egroup - sgroup), blockSize>>>(group_grid, count, mid_res);
        CUDA_CHECK(cudaDeviceSynchronize());

        t_count = (t_count + 1) / 2;
        count *= 2;
    }
}

__global__ void accumulate_result_kernel(G1Point *result, Vector<G1Point> *mid_res, size_t window_size, size_t num_groups, size_t sgroup, size_t egroup)
{
    for (size_t k = num_groups - 1; k <= num_groups; k--)
    {
        if (k >= sgroup && k < egroup)
        {
            for (size_t i = 0; i < window_size; i++)
            {
                *result = result->dbl();
            }
            *result = *result + mid_res[k - sgroup][0];
        }
    }
}

void cuZK::accumulate_result()
{
    accumulate_result_kernel<<<1, 1>>>(result, mid_res, window_size, num_groups, sgroup, egroup);
    CUDA_CHECK(cudaDeviceSynchronize());
}
G1Point *cuZK::run()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
    printf("started\n");
    this->generate_ell_matrix(points, scalars, total);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("generated ell matrix\n");
    this->sum_buckets();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("summed buckets\n");
    this->accumulate_result();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("accumulated result\n");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %f ms\n", milliseconds);

    return result;
}
size_t cuZK::get_runtime()
{
}

#endif