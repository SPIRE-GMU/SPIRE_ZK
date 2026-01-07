#ifndef MULTIBUCKETMSM_CU
#define MULTIBUCKETMSM_CU
#include "./../include/MultiBucketMSM.cuh"
#include <set>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <cassert>

void check_cuda_error(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s , File = %s, Line = %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void precomputed_points_kernel(const G1Point *points, G1Point *precomputed_points, size_t total_num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < total_num)
    {
        G1Point p = points[idx];
        precomputed_points[3*idx] = p;
        precomputed_points[3*idx + 1] = p.dbl();
        precomputed_points[3*idx + 2] = precomputed_points[3*idx + 1] + p;
        precomputed_points[3*idx].to_affine();
        precomputed_points[3*idx + 1].to_affine();
        precomputed_points[3*idx + 2].to_affine();
        idx += stride;
    }
}

MultiBucketMSM::MultiBucketMSM(G1Point *_points, Scalar *_scalars, size_t _total_num)
{
    this->points = _points;
    this->scalars = _scalars;
    this->total_num = _total_num;
    this->window_size = 16;//log2(total_num) - (log2(total_num) / 3 - 2);
    printf("Window size: %zu\n", window_size);
    this->num_bits = 255;
    this->num_windows = (num_bits + window_size - 1) / window_size;
    printf("Number of windows: %zu\n", num_windows);
    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    this->sm_count = prop.multiProcessorCount;
    printf("Number of SMs: %zu\n", sm_count);
    this->max_shared_memory_per_block = prop.sharedMemPerBlock;
    cudaMalloc((void **)&precomputed_points, sizeof(G1Point) * total_num * 3);
    cudaMalloc((void **)&this->scalar_parts, sizeof(Scalar_Part) * total_num * num_windows);

    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
}
void MultiBucketMSM::init_buckets()
{
    this->bucket_set_count = 1;

    cudaMalloc((void **)&this->buckets, sizeof(G1Point) * bucket_set_count * num_windows * bucket_size);
    init_G1Point<<<sm_count * 2, 128>>>(this->buckets, bucket_set_count * num_windows * bucket_size);
    cudaMalloc((void **)&this->summed_buckets, sizeof(G1Point) * num_windows * bucket_size);
    init_G1Point<<<sm_count * 2, 128>>>(this->summed_buckets, num_windows * bucket_size);
    cudaMalloc((void **)&this->window_sums, sizeof(G1Point) * num_windows);
    init_G1Point<<<sm_count * 2, 128>>>(this->window_sums, num_windows);
    cudaMalloc((void **)&this->result, sizeof(G1Point));
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
}
MultiBucketMSM::~MultiBucketMSM()
{
    cudaFree(precomputed_points);
    cudaFree(scalar_parts);
    cudaFree(buckets);
    cudaFree(summed_buckets);
    cudaFree(window_sums);
    cudaFree(result);
    delete[] decomposition;
    delete[] bucket_map;
    delete[] bucket_set;
}
size_t omega2(size_t n)
{
    size_t rem = n % 2;
    size_t exponent = 0;
    while (rem == 0)
    {
        exponent++;
        n >>= 1;
        rem = n % 2;
    }
    return exponent;
}

size_t omega3(size_t n)
{
    size_t rem = n % 3;
    size_t exponent = 0;
    while (rem == 0)
    {
        exponent++;
        n = n / 3;
        rem = n % 3;
    }
    return exponent;
}
void construct_bucket_set(long bucket_set[], size_t &bsize, const long q, const long ah, long *bucket_map)
{

    std::set<long> B = {0, 1};

    for (long i = 2; i <= q / 2; ++i)
    {
        if (((omega2(i) + omega3(i)) % 2) == 0)
        {
            B.insert(i);
        }
    }

    for (long i = q / 4; i < q / 2; ++i)
    {
        if ((B.find(i) != B.end()) && (B.find(q - 2 * i) != B.end())) // if i is in B and q-3*i is in B
        {
            B.erase(q - 2 * i);
        }
    }
    for (long i = q / 6; i < q / 4; ++i)
    {
        if ((B.find(i) != B.end()) && (B.find(q - 3 * i) != B.end())) // if i is in B and q-3*i is in B
        {
            B.erase(q - 3 * i);
        }
    }

    for (long i = 1; i <= ah + 1; ++i)
    {
        if (((omega2(i) + omega3(i)) % 2) == 0)
        {
            B.insert(i);
        }
    }

    long index = 0;
    for (auto b : B)
    {
        bucket_set[index] = b;
        bucket_map[b] = index;
        ++index;
    };
    bsize = index;
}
void MultiBucketMSM::computeBucket()
{
    long bucket_set[(size_t)1 << window_size];
    size_t bsize = 0;
    long *bmap = new long[((size_t)1 << window_size) + 1];
    construct_bucket_set(bucket_set, bsize, (size_t)1 << window_size, ((size_t)1 << (window_size - 1)) - 1, bmap);
    this->bucket_size = bsize;
    this->bucket_set = new long[bsize];
    memcpy(this->bucket_set, bucket_set, sizeof(long) * bsize);
    cudaMalloc((void **)&this->bucket_map, sizeof(long) * (((size_t)1 << window_size) + 1));
    cudaMemcpy(this->bucket_map, bmap, sizeof(long) * (((size_t)1 << window_size) + 1), cudaMemcpyHostToDevice);
}
void MultiBucketMSM::generateDecomposition()
{
    size_t q_radix = (size_t)1 << window_size;
    decomposition = new Decomposition[q_radix + 1];
    std::set<int> MULTI_SET = {1, 2, 3};
    for (int m : MULTI_SET)
    {
        for (int i = 0; i < bucket_size; ++i)
        {
            int b = bucket_set[i];
            if (m * b <= q_radix)
                decomposition[q_radix - m * b] = {m, b, 1};
        }
    }
    for (int m : MULTI_SET)
    {
        for (int i = 0; i < bucket_size; ++i)
        {
            int b = bucket_set[i];
            if (m * b <= q_radix)
                decomposition[m * b] = {m, b, 0};
        }
    }
}
void MultiBucketMSM::precompute()
{
    size_t block_size = 256;
    // getOptimalLaunchConfig(precomputed_points_kernel, total_num, &grid_size, &block_size);
    precomputed_points_kernel<<<sm_count * 32, block_size>>>(this->points, this->precomputed_points, this->total_num);
}
__global__ void process_scalar_kernel(const Scalar *scalars, Scalar_Part *sc_parts, const Decomposition *decomposition, const size_t total_num, const int window_size, const int num_windows, const long *bucket_map)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t q = (size_t)1 << window_size;
    while (idx < total_num)
    {
        int carry = 0;
        for (size_t i = 0; i < num_windows; i++)
        {
            size_t sc_part = scalars[idx].get_bits_as_uint32(i * window_size + window_size - 1, i * window_size);
            sc_part += carry;
            if (sc_part > 0)
            {
                size_t m = decomposition[sc_part].m;
                size_t b = decomposition[sc_part].b;
                carry = decomposition[sc_part].flag ? 1 : 0;
                carry ? assert(q - m * b == sc_part) : assert(m * b == sc_part);
                sc_parts[i * total_num + idx].bucket = b;
                sc_parts[i * total_num + idx].flag = carry;
                sc_parts[i * total_num + idx].index = 3 * idx + m - 1;
                sc_parts[i * total_num + idx].bucket_index = bucket_map[b];
            }
            else
            {
                sc_parts[i * total_num + idx].bucket = 0;
                sc_parts[i * total_num + idx].flag = 0;
                sc_parts[i * total_num + idx].index = 0;
                sc_parts[i * total_num + idx].bucket_index = 0;
            }
        }
        idx += stride;
    }
}
void MultiBucketMSM::processScalar()
{
    size_t grid_size, block_size = 128;
    Decomposition *decomposition;
    cudaMalloc((void **)&decomposition, sizeof(Decomposition) * (((size_t)1 << window_size) + 1));
    cudaMemcpy(decomposition, this->decomposition, sizeof(Decomposition) * (((size_t)1 << window_size) + 1), cudaMemcpyHostToDevice);

    // getOptimalLaunchConfig(process_scalar_kernel, total_num, &grid_size, &block_size);
    process_scalar_kernel<<<sm_count * 4, block_size>>>(this->scalars, this->scalar_parts, decomposition, this->total_num, this->window_size, this->num_windows, this->bucket_map);
    cudaDeviceSynchronize();
}

__global__ void compute_bucket_sums_kernel(const Scalar_Part *sc_parts,const G1Point *precomputed_points, G1Point *buckets, const size_t total_num, const size_t bucket_size, const int num_window)
{
    size_t bucket_id = blockIdx.x;
    size_t window = blockIdx.y;
    size_t number_bucket = gridDim.x;
    if (window >= num_window)
        return;
    
    size_t current_bucket = threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t total = (total_num + number_bucket - 1) / number_bucket;
    size_t start = window * total_num + bucket_id * total;
    size_t end = start + total;

    size_t offset = window * number_bucket * bucket_size + bucket_id * bucket_size;

    while(current_bucket < bucket_size)
    {
        for (size_t i = start; i < end && i < (window + 1) * total_num; ++i)
        {
            size_t index = sc_parts[i].index;
            int m = sc_parts[i].bucket;
            int flag = sc_parts[i].flag;
            long bucket = sc_parts[i].bucket;
            long bucket_index = sc_parts[i].bucket_index;
    
            if (bucket > 0 && bucket_index == current_bucket)
            {   
                if (flag == 1)
                {
                    buckets[offset + bucket_index] = buckets[offset + bucket_index].mixed_add(-precomputed_points[index]);
                }
                else
                {
                    buckets[offset + bucket_index] = buckets[offset + bucket_index].mixed_add(precomputed_points[index]);
                }
            }
        }

        current_bucket += stride;
    }

    
}

void MultiBucketMSM::computeBucketSums()
{
    size_t block_size = 256;
    compute_bucket_sums_kernel<<<(bucket_set_count, num_windows), block_size>>>(this->scalar_parts, this->precomputed_points, this->buckets, this->total_num, this->bucket_size, this->num_windows);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
}
__global__ void sum_all_bucket(size_t number_bucket, size_t num_window, const G1Point *buckets, G1Point *summed_buckets, const size_t total_num, const size_t bucket_size)
{
    size_t bucket_set_id = blockIdx.x % number_bucket;
    size_t window_id = blockIdx.x / number_bucket;

    if (window_id >= num_window)
        return;
    if (bucket_set_id >= number_bucket)
        return;

    size_t bucket_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t bucket_stride = blockDim.x * gridDim.x;

    while (bucket_id < bucket_size)
    {
        size_t stride = bucket_size;
        size_t offset = window_id * number_bucket * bucket_size;

        size_t start = offset + bucket_id;
        size_t end = (window_id + 1) * number_bucket * bucket_size;

        for (size_t i = start; i < end; i += stride)
        {
            summed_buckets[window_id * number_bucket + bucket_id] = summed_buckets[window_id * number_bucket + bucket_id].add(buckets[i]);
        }

        bucket_id += bucket_stride;
    }
}

__global__ void sum_windows(size_t number_bucket, size_t num_window, const G1Point *summed_buckets, G1Point *window_sums, const long *bucket_set, const size_t bucket_size)
{
    size_t bucket_set_id = blockIdx.x % number_bucket;
    size_t window_id = blockIdx.x / number_bucket;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (bucket_set_id >= number_bucket)
        return;
    if (window_id >= num_window)
        return;

    extern __shared__ G1Point shared_buckets[];
    size_t total_points_per_thread = (bucket_size + blockDim.x - 1) / blockDim.x;
    size_t start = tid * total_points_per_thread;
    size_t end = start + total_points_per_thread;

    G1Point local_sum = G1Point().zero();
    for (int i = start; i < end && i < (window_id + 1) * bucket_size; ++i)
    {
        local_sum = local_sum.add(summed_buckets[window_id * bucket_size + i] * bucket_set[i % bucket_size]);
    }
    shared_buckets[threadIdx.x] = local_sum;

    __syncthreads();

    if (tid == 0)
    {
        for (int i = 0; i < blockDim.x; ++i)
        {
            window_sums[window_id] = window_sums[window_id].add(shared_buckets[i]);
        }
    }
}
void MultiBucketMSM::computeBuckets()
{
    size_t block_size = 128;
    size_t grid_size = bucket_set_count * num_windows;
    sum_all_bucket<<<grid_size, block_size>>>(bucket_set_count, num_windows, this->buckets, this->summed_buckets, this->total_num, this->bucket_size);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
    block_size = ((max_shared_memory_per_block / (sizeof(G1Point) + 48)) / 32) * 32;
    sum_windows<<<grid_size, block_size, sizeof(G1Point) * block_size>>>(this->bucket_size, this->num_windows, this->summed_buckets, this->window_sums, this->bucket_set, this->bucket_size);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
}
__global__ void reduce_result_kernel(const G1Point *window_sums, G1Point *result, const size_t num_windows, const size_t window_size)
{
    G1Point acc = window_sums[num_windows - 1];
    for (int k = num_windows - 2; k >= 0; --k)
    {
        for (int i = 0; i < window_size; ++i)
            acc = acc.dbl(); // Left-shift accumulator
        acc = acc.add(window_sums[k]);
    }
    *result = acc;
}

void MultiBucketMSM::reduceResult()
{
    reduce_result_kernel<<<1, 1>>>(this->window_sums, this->result, this->num_windows, this->window_size);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
}

G1Point MultiBucketMSM::run()
{
    cudaEvent_t start, stop;


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    precompute();
    printf("Precomputed points\n");
    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

    cudaEvent_t precom;
    cudaEventCreate(&precom);
    cudaEventRecord(precom);
    cudaEventSynchronize(precom);

    float precom_ms = 0;
    cudaEventElapsedTime(&precom_ms, start, precom);
    printf("Precomputation time: %f ms\n", precom_ms);

    computeBucket();
    printf("Bucket computed\n");

    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

    generateDecomposition();
    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

    cudaDeviceSynchronize();

    printf("Decomposition generated\n");
    init_buckets();
    cudaDeviceSynchronize();
    printf("Buckets initialized\n");
    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


    cudaEvent_t pscalar_start, pscalar_stop;
    cudaEventCreate(&pscalar_start);
    cudaEventCreate(&pscalar_stop);
    cudaEventRecord(pscalar_start);
    cudaEventSynchronize(pscalar_start);


    processScalar();
    cudaDeviceSynchronize();

    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

    cudaEventRecord(pscalar_stop);
    cudaEventSynchronize(pscalar_stop);
    float pscalar_ms = 0;
    cudaEventElapsedTime(&pscalar_ms, pscalar_start, pscalar_stop);
    printf("Scalar processing time: %f ms\n", pscalar_ms);

    // printf("Scalar processed\n");
    computeBucketSums();
    cudaDeviceSynchronize();

    cudaEvent_t bucket;
    cudaEventCreate(&bucket);
    cudaEventRecord(bucket);
    cudaEventSynchronize(bucket);
    float bucket_ms = 0;
    cudaEventElapsedTime(&bucket_ms, pscalar_stop, bucket);
    printf("Bucket sum time: %f ms\n", bucket_ms);

    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

    // printf("Bucket sums computed\n");
    computeBuckets();
    cudaDeviceSynchronize();

    cudaEvent_t compute;
    cudaEventCreate(&compute);
    cudaEventRecord(compute);
    cudaEventSynchronize(compute);
    float compute_ms = 0;
    cudaEventElapsedTime(&compute_ms, bucket, compute);
    printf("Compute bucket time: %f ms\n", compute_ms);


    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

    // printf("Buckets computed\n");
    reduceResult();
    cudaDeviceSynchronize();

    check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

    // printf("Result reduced\n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

    return *result;
}

#endif