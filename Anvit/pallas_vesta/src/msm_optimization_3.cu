#ifndef __MSM_CUH
#define __MSM_CUH

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include <cub/cub.cuh>

#include "./../include/Point.cuh"
#include <vector>

#define debug 1

#define WINDOW_SIZE 16
#define NUM_BITS 256
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

__global__ void process_scalar_into_bucket(const Scalar *scalar,
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

__global__ void sum_small_bucket(const Point *point, Point *sum, const uint32_t *offset, const uint32_t *indices,
                                 const uint32_t *count, size_t num_bucket)
{
    extern __shared__ uint32_t tile_idx[];
    size_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    size_t curr_window = blockIdx.y;
    size_t stride = gridDim.x * blockDim.x;
    
    while (bucket < num_bucket) {
        size_t curr_bucket = bucket + curr_window * num_bucket;
        uint32_t n = count[curr_bucket];
        if (n == 0 || n >= 128) return;
    
        // load indices into shared mem tile
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            tile_idx[i] = indices[offset[curr_bucket] + i];
        }
        __syncthreads();
    
        Point acc; acc = acc.zero();
        for (uint32_t i = 0; i < n; ++i) {
            Point p = __ldg(&point[tile_idx[i]]);
            acc = acc.mixed_add(p);
        }
        sum[curr_bucket] = acc;

        bucket += stride;
    } 
    
}
__global__ void sum_medium_bucket(const Point *point, Point *sum, const uint32_t *offset,
                                  const uint32_t *indices, const uint32_t *count, size_t num_bucket)
{
    size_t bucket = blockIdx.x;
    size_t curr_window = blockIdx.y;
    size_t stride = gridDim.x;
    size_t num_threads = blockDim.x;

    extern __shared__ Point shared_sum[];

    while (bucket < num_bucket)
    {
        size_t curr_count = count[bucket + curr_window * num_bucket];
        if (curr_count < 128 || curr_count > 256 * 128)
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
            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
            Point *smem_ptr = shared_sum + idx;
            pipe.producer_acquire();
            cuda::memcpy_async(smem_ptr, &point[indices[i]], sizeof(Point), pipe);
            pipe.producer_commit();

            pipe.consumer_wait();
            Point val = *smem_ptr;
            pipe.consumer_release();
            
            lsum = lsum.mixed_add(val);
        }
        shared_sum[idx] = lsum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
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


__global__ void gather_bucket_parallel(Point *sum, size_t bcount, Point *res)
{
    int idx = threadIdx.x;
    int curr_window = blockIdx.x;
    extern __shared__ Point local_sum[];
    Point running_sum;
    Point result;
    int per_thread = (bcount + blockDim.x - 1) / blockDim.x;
    int start = idx * per_thread + 1;
    int end = start + per_thread;

    if (end > bcount)
        end = bcount;
    local_sum[idx] = local_sum[idx].zero();
    running_sum = running_sum.zero();
    result = result.zero();

    for (int i = end - 1; i >= start; i--)
    {
        running_sum = running_sum + sum[i + curr_window * bcount];
        local_sum[idx] = local_sum[idx] + running_sum;
    }

    __syncthreads();

    local_sum[idx] = local_sum[idx] + running_sum * (per_thread * idx);
    __syncthreads();
    if (idx % 2 == 0)
    {
        local_sum[idx] = local_sum[idx] + local_sum[idx + 1];
    }
    __syncthreads();
    if (idx % 4 == 0)
    {
        local_sum[idx] = local_sum[idx] + local_sum[idx + 2];
    }
    __syncthreads();
    if (idx % 8 == 0)
    {
        local_sum[idx] = local_sum[idx] + local_sum[idx + 4];
    }
    __syncthreads();
    if (idx % 16 == 0)
    {
        local_sum[idx] = local_sum[idx] + local_sum[idx + 8];
    }
    __syncthreads();
    if (idx % 32 == 0)
    {
        local_sum[idx] = local_sum[idx] + local_sum[idx + 16];
    }
    __syncthreads();
    if (idx == 0)
    {
        for (int i = 0; i < blockDim.x; i += 32)
        {
            result = result + local_sum[i];
        }
        res[curr_window] = result;
    }
}
// accumulate all window output
__global__ void accumulate_result(Point *window_res, size_t num_window, Point *res)
{
    Point acc = acc.zero();
    for (int i = num_window - 1; i >= 0; i--)
    {
        for (int j = 0; j < WINDOW_SIZE; j++)
        {
            acc = acc.dbl();
        }
        acc = acc + window_res[i];
    }
    *res = acc;
}

#if debug
__global__ void print_point(Point *p, size_t num = 1)
{
    for (size_t i = 0; i < num; i++)
    {
        if (!p[i].is_zero())
        {
            p[i].to_affine();
            p[i].print();
        }
    }
}

#endif

// driver function to perform multi scalar multiplication
Point* cuda_pippenger_msm(Point *points, Scalar *scalars, size_t num_points)
{
    int num_windows = (NUM_BITS + WINDOW_SIZE - 1) / WINDOW_SIZE;
    size_t num_bucket = ((size_t)1 << WINDOW_SIZE);

    uint32_t *scalar_chunks, *indices; // scalar_chunks put all the scalar chunks in a single array
    // indices array to store the indices of scalars according to bucket
    uint32_t *offset, *offset_counter; // offset for bucket
    uint32_t *count;                   // count for bucket
    uint32_t *h_count;

    cudaStream_t memcpy_stream;


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int sm_count = prop.multiProcessorCount;

    CUDA_CHECK(cudaStreamCreateWithFlags(&memcpy_stream, cudaStreamNonBlocking));
    
    CUDA_CHECK(cudaMalloc(&scalar_chunks, sizeof(uint32_t) * num_points * num_windows));
    CUDA_CHECK(cudaMalloc(&indices, sizeof(uint32_t) * num_points * num_windows));
    CUDA_CHECK(cudaMalloc(&offset, sizeof(uint32_t) * num_bucket * num_windows));
    CUDA_CHECK(cudaMalloc(&offset_counter, sizeof(uint32_t) * num_bucket * num_windows));
    CUDA_CHECK(cudaMalloc(&count, sizeof(uint32_t) * num_bucket * num_windows));
    CUDA_CHECK(cudaMemset(count, 0, sizeof(uint32_t) * num_bucket * num_windows));
    CUDA_CHECK(cudaMemset(offset, 0, sizeof(uint32_t) * num_bucket * num_windows));
    CUDA_CHECK(cudaMemset(offset_counter, 0, sizeof(uint32_t) * num_bucket * num_windows));
    CUDA_CHECK(cudaMemset(scalar_chunks, 0, sizeof(uint32_t) * num_points * num_windows));
    CUDA_CHECK(cudaMemset(indices, 0, sizeof(uint32_t) * num_points * num_windows));

    h_count = new uint32_t[num_bucket * num_windows];

    size_t blockSize = 256;
    size_t gridSize = (num_points + blockSize - 1) / blockSize;

    process_scalar_into_bucket<<<gridSize, blockSize>>>(scalars, points, num_points, num_windows, scalar_chunks, indices, offset, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, count, offset, num_bucket * num_windows);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, count, offset, num_bucket * num_windows);
    CUDA_CHECK(cudaFree(d_temp_storage));

    CUDA_CHECK(cudaMemcpy(offset_counter, offset, sizeof(uint32_t) * num_bucket * num_windows, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    dim3 grid_size(gridSize, num_windows);
    construct_bucket_indices<<<grid_size, blockSize, 0, 0>>>(scalar_chunks, indices, offset_counter, num_points, num_bucket);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    Point *sum;
    CUDA_CHECK(cudaMalloc(&sum, sizeof(Point) * num_bucket * num_windows));
    dim3 block(128);
    dim3 grid(sm_count*2, num_windows); //(num_bucket + blockSize - 1) / blockSize
    sum_small_bucket<<<grid, block, block.x * sizeof(Point)>>>(points, sum, offset, indices, count, num_bucket);
    sum_medium_bucket<<<grid, block, block.x * sizeof(Point)>>>(points, sum, offset, indices, count, num_bucket);
    CUDA_CHECK(cudaDeviceSynchronize());

    Point *window_res;
    CUDA_CHECK(cudaMalloc(&window_res, sizeof(Point) * num_windows));
    dim3 gather_grid(num_windows, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    gather_bucket_parallel<<<gather_grid, 256, 48 * 1024>>>(sum, num_bucket, window_res);
    CUDA_CHECK(cudaDeviceSynchronize());

    Point *res;
    CUDA_CHECK(cudaMalloc(&res, sizeof(Point)));
    accumulate_result<<<1, 1>>>(window_res, num_windows, res);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free memory
    CUDA_CHECK(cudaFree(scalar_chunks));
    CUDA_CHECK(cudaFree(indices));
    CUDA_CHECK(cudaFree(offset));
    CUDA_CHECK(cudaFree(offset_counter));
    CUDA_CHECK(cudaFree(count));
    CUDA_CHECK(cudaFree(sum));
    CUDA_CHECK(cudaFree(window_res));

    return res;
}

#endif