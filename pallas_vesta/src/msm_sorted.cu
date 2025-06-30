#ifndef __MSM_CUH
#define __MSM_CUH

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#include <cub/cub.cuh>

#include "./../include/Point.cuh"

#define NUM_BITS 256
#define WINDOW_SIZE 16

#define smallBucketSize 256
#define mediumBucketSize 32*256

#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

__global__ void process_scalar_into_bucket(Scalar *scalar,
                                           Point *points,
                                           size_t num_points,
                                           size_t num_window,
                                           uint32_t *scalar_chunks,
                                           uint32_t *indices,
                                           uint32_t *offset,
                                           uint32_t *count)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < num_points)
    {

        for (int current_window = 0; current_window < num_window; current_window++)
        {
            size_t bindex = 0;
            size_t start = current_window * WINDOW_SIZE;
            size_t end = start + WINDOW_SIZE;
            for (size_t i = start, j = 0; i < end; i++, j++)
            {
                if (scalar[idx].test_bit(i))
                {
                    bindex |= (1 << j);
                }
            }
            scalar_chunks[idx + current_window * num_points] = bindex;
            if (bindex != 0)
                atomicAdd(&count[bindex + current_window * ((size_t)1 << WINDOW_SIZE)], 1);
        }

        idx += stride;
    }
}

__global__ void construct_bucket_indices(
    const __restrict__ uint32_t *scalar_chunks,
    uint32_t *indices,
    uint32_t *offset_counter,
    size_t num_points,
    size_t num_bucket)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t curr_window = blockIdx.y;
    while (idx < num_points)
    {
        uint32_t bindex = scalar_chunks[idx + curr_window * num_points];
        if (bindex != 0)
            indices[atomicAdd(&offset_counter[bindex + curr_window * num_bucket], 1)] = idx;
        idx += stride;
    }
}

__global__ void sum_small_bucket(__const__ Point *point, Point *bucket, uint32_t *count, uint32_t *offset, uint32_t *indices, size_t bucket_count)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t curr_window = blockIdx.y;
    Point local_sum, globalPoint;

    while(idx <bucket_count)
    {
        uint32_t cnt = count[idx + bucket_count * curr_window];
        if(cnt > 0 && cnt < smallBucketSize) 
        {
            local_sum = local_sum.zero();
            size_t start = offset[idx + bucket_count * curr_window];
            size_t end = start + cnt;
            for(size_t i = start; i < end; i++)
            {
                globalPoint = point[indices[i]];
                local_sum = local_sum + globalPoint;
            }
            bucket[idx + curr_window * bucket_count] = local_sum;
        }
        idx += stride;
    }

}
__global__ void sum_medium_bucket(__const__ Point *point, Point *sum, uint32_t *offset, uint32_t *count, uint32_t *indices, size_t num_bucket)
{
    size_t bucket = blockIdx.x;
    size_t stride = gridDim.x;
    size_t curr_window = blockIdx.y;
    size_t num_threads = blockDim.x;
    Point globalPoint;

    __shared__ Point shared_sum[num_threads];
    while( bucket < num_bucket)
    {
        size_t curr_count = count[bucket + curr_window * num_bucket];
        if (curr_count < smallBucketSize || curr_count > mediumBucketSize)
        {
            bucket += stride;
            continue;
        }
        size_t per_thread = (curr_count + num_threads - 1) / num_threads;

        size_t idx = threadIdx.x;

        size_t start = offset[bucket + curr_window * num_bucket] + idx * per_thread;
        size_t end = start + per_thread;
        if (end > offset[bucket + curr_window * num_bucket] + curr_count)
            end = offset[bucket + curr_window * num_bucket] + curr_count;
        Point lsum;
        lsum = lsum.zero();
        for (size_t i = start; i < end; i++)
        {
            globalPoint = point[indices[i]];
            lsum = lsum.mixed_add(globalPoint);
        }
        shared_sum[idx] = lsum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                shared_sum[threadIdx.x] = shared_sum[threadIdx.x] + shared_sum[threadIdx.x + stride];
            __syncthreads();
        }
        if (idx == 0)
        {
            sum[bucket + curr_window * num_bucket] = shared_sum[0];
        }
        bucket += stride;
    }
}
__global__ void gather_buckets(__const__ Point *buckets, size_t bucket_count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t bucket_sum_id = blockIdx.x;
    size_t curr_window = blockIdx.y;
    __shared__ Point bucket_result[ blockDim.x ];
    if(idx < bucket_count)
    {
        bucket_result[idx] = buckets[idx];
        __syncthreads();
        
    }
}
void sum_large_bucket()
{
}
void accumulateResult()
{
    
}

Point* cuda_pippenger_msm(Point *points, Scalar *scalars, size_t num_points)
{
    //allocate memory
    Point *buckets, *sorted_points, *window_res, *result;
    uint32_t *scalar_chunks, *offset, *count, *indices, *offset_counter;

    uint32_t window_size = WINDOW_SIZE;
    uint32_t num_windows = (NUM_BITS + window_size - 1)/window_size;
    uint32_t num_bucket = 1<<window_size;

    cudaStream_t small, medium, large;
    cudaStreamCreate(&small);
    cudaStreamCreate(&medium);
    cudaStreamCreate(&large);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    size_t blockSize = 512;
    size_t gridSize = sm_count * 32;

    CUDA_CHECK(cudaMalloc(&buckets, sizeof(Point)*num_bucket*num_windows));
    CUDA_CHECK(cudaMalloc(&sorted_points, sizeof(Point)*num_points));
    CUDA_CHECK(cudaMalloc(&window_res, sizeof(Point)*num_windows));
    CUDA_CHECK(cudaMalloc(&result, sizeof(Point)));

    CUDA_CHECK(cudaMalloc(&scalar_chunks, sizeof(uint32_t)*num_points*num_windows));
    CUDA_CHECK(cudaMalloc(&offset, sizeof(uint32_t)*num_bucket*num_windows));
    CUDA_CHECK(cudaMalloc(&offset_counter, sizeof(uint32_t)*num_bucket*num_windows));
    CUDA_CHECK(cudaMalloc(&count, sizeof(uint32_t)*num_bucket*num_windows));
    CUDA_CHECK(cudaMalloc(&indices, sizeof(uint32_t)*num_points*num_windows));

    // split scalars and construct CSR Matrix
    process_scalar_into_bucket<<<gridSize, blockSize>>>(scalars, points, num_points, num_windows, scalar_chunks, indices, offset, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // perform exclusive scan on the count array to get the offsets
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, count, offset, num_bucket * num_windows);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, count, offset, num_bucket * num_windows);
    CUDA_CHECK(cudaFree(d_temp_storage));

    // Build indices for each bucket
    CUDA_CHECK(cudaMemcpy(offset_counter, offset, sizeof(uint32_t) * num_bucket * num_windows, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    dim3 grid_size(gridSize, num_windows);
    construct_bucket_indices<<<grid_size, blockSize>>>(scalar_chunks, indices, offset_counter, num_points, num_bucket);
    CUDA_CHECK(cudaDeviceSynchronize());

    // use streams to do the tasks parallely
    // 1. sort the points based on the indices
    // 2. 
    uint32_t *h_count;
    CUDA_CHECK(cudaMallocHost(&h_count, sizeof(uint32_t)* num_bucket * num_windows));
    CUDA_CHECK(cudaMemcpy(h_count, count, sizeof(uint32_t)*num_bucket* num_windows, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    sum_small_bucket<<<gridSize, blockSize,0, small>>>();
    sum_medium_bucket<<<gridSize, blockSize, 0, medium>>>();
    std::vector<uint32_t> largeBuckets;
    for(uint32_t i=0; i<num_bucket*num_windows; i++)
    {
        if(h_count[i] > mediumBucketSize) 
        {
            largeBuckets.push_back(i);
        }
    }
    for(int i=0; i<largeBuckets.size(); i++)
    {
        gridSize = (h_count[largeBuckets[i]] + mediumBucketSize - 1)/ mediumBucketSize;
        

    }
        sum_large_bucket(largeBuckets[i], buckets, large);
    CUDA_CHECK(cudaDeviceSynchronize());

    gather_buckets<<<gridSize, blockSize>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    accumulateResult();
    return result;
}

#endif