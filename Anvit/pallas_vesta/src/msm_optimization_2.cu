#ifndef __MSM_CUH
#define __MSM_CUH

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "./../include/Point.cuh"

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

__global__ void sort_points_for_bucket(const Point *points,
                                       Point *sorted_points,
                                       const uint32_t *indices,
                                       const uint32_t *offset,
                                       const uint32_t *count,
                                       size_t num_bucket,
                                       size_t curr_window,
                                       size_t num_points)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // size_t curr_window = blockIdx.y;
    while (idx < num_bucket)
    {
        if (count[idx + curr_window * num_bucket] > 0)
        {
            size_t start = offset[idx + curr_window * num_bucket];
            size_t end = start + count[idx + curr_window * num_bucket];
            for (size_t i = start; i < end; i++)
            {
                sorted_points[i%num_points] = points[indices[i]];
            }
        }
        idx += stride;
    }
}

__global__ void sum_small_bucket(const Point *point, Point *sum, const uint32_t *offset, const uint32_t *indices, 
        const uint32_t *count, size_t num_bucket, size_t curr_window, size_t num_points)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    // size_t curr_window = blockIdx.y;
    while (idx < num_bucket)
    {
        if (count[idx + curr_window * num_bucket] > 0 && count[idx + curr_window * num_bucket] < 128)
        {
            Point lsum;
            lsum = lsum.zero();

            size_t start = offset[idx + curr_window * num_bucket];
            size_t end = start + count[idx + curr_window * num_bucket];
            for (size_t i = start; i < end; i++)
            {
                lsum = lsum + point[i%num_points];
            }
            sum[idx + curr_window * num_bucket] = lsum;
        }

        idx += stride;
    }
}
__global__ void sum_medium_bucket(const Point *point, Point *sum, const uint32_t *offset, 
    const uint32_t *indices, const uint32_t *count, size_t num_bucket, size_t curr_window, size_t num_points)
{
    size_t bucket = blockIdx.x;
    // size_t curr_window = blockIdx.y;
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
            lsum = lsum.mixed_add(point[i%num_points]);
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

__global__ void sum_large_bucket()
{

}

// sum all the buckets of a window and store the result to a point
__global__ void gather_buckets(Point *sum, size_t bcount, Point *res)
{
    size_t curr_window = blockIdx.x;
    Point lsum, running_sum;
    lsum = lsum.zero();
    running_sum = running_sum.zero();
    for (size_t i = bcount - 1; i > 0; i--)
    {
        lsum = lsum + sum[i + curr_window * bcount];
        // running_sum = running_sum + lsum;
    }
    res[curr_window] = lsum;
    // *res = running_sum;
}
__global__ void gather_bucket_parallel(const Point *sum, size_t bcount, Point *res)
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
__global__ void accumulate_result(const Point *window_res, size_t num_window, Point *res)
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
__global__ void print_array(uint32_t *arr, size_t num = 1)
{
    for (size_t i = 0; i < num; i++)
    {
        printf("%u ", arr[i]);
    }
    printf("\n");
}
__global__ void print_bucket_count(uint32_t *count, size_t num = 1)
{
    for (size_t i = 0; i < num; i++)
    {
        if(count[i]) printf("bucket %lu : %u, ", i, count[i]);
    }
    printf("\n");
}
#endif

// driver function to perform multi scalar multiplication
void cuda_pippenger_msm(Point *points, Scalar *scalars, size_t num_points)
{
    int num_windows = (NUM_BITS + WINDOW_SIZE - 1) / WINDOW_SIZE;
    size_t num_bucket = ((size_t)1 << WINDOW_SIZE);

    uint32_t *scalar_chunks, *indices; // scalar_chunks put all the scalar chunks in a single array
    // indices array to store the indices of scalars according to bucket
    uint32_t *offset, *offset_counter; // offset for bucket
    uint32_t *count;                   // count for bucket
    uint32_t *h_count;

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    // Scan the scalars and construct bucket element counts
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
    construct_bucket_indices<<<grid_size, blockSize, 0, 0>>>(scalar_chunks, indices, offset_counter, num_points, num_bucket);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_count, count, sizeof(uint32_t) * num_bucket * num_windows, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken to process scalar: %f\n", ms);

    // print_bucket_count<<<1, 1>>>(count, num_bucket * num_windows);
    // print_array<<<1, 1>>>(offset, num_bucket * num_windows);
    // print_array<<<1,1>>>(indices, num_points * num_windows);
    // Sum the buckets
    Point *sum;
    CUDA_CHECK(cudaMalloc(&sum, sizeof(Point) * num_bucket * num_windows));
    dim3 block(64);
    dim3 grid(4*84, 1); //(num_bucket + blockSize - 1) / blockSize
    
    Point *sorted_points;
    CUDA_CHECK(cudaMalloc(&sorted_points, sizeof(Point) * num_points));

    for(int i = 0; i < num_windows; i++)
    {
        sort_points_for_bucket<<<grid, block>>>(points, sorted_points, indices, offset, count, num_bucket, i, num_points);
        CUDA_CHECK(cudaDeviceSynchronize());

        sum_small_bucket<<<grid, block>>>(sorted_points, sum, offset, indices, count, num_bucket, i, num_points);
        sum_medium_bucket<<<grid, block, block.x * sizeof(Point)>>>(sorted_points, sum, offset, indices, count, num_bucket, i, num_points);
        CUDA_CHECK(cudaDeviceSynchronize());
    }


    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t sum_stop;
    cudaEventCreate(&sum_stop);
    cudaEventRecord(sum_stop);
    cudaEventSynchronize(sum_stop);
    float sum_ms;
    cudaEventElapsedTime(&sum_ms, stop, sum_stop);
    printf("Time taken to sum buckets: %f\n", sum_ms);

    Point *window_res;
    CUDA_CHECK(cudaMalloc(&window_res, sizeof(Point) * num_windows));
    dim3 gather_grid(num_windows, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    gather_bucket_parallel<<<gather_grid, 256, 48 * 1024>>>(sum, num_bucket, window_res);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t gather_stop;
    cudaEventCreate(&gather_stop);
    cudaEventRecord(gather_stop);
    cudaEventSynchronize(gather_stop);
    float gather_ms;
    cudaEventElapsedTime(&gather_ms, sum_stop, gather_stop);
    printf("Time taken to gather buckets: %f\n", gather_ms);

    Point *res;
    CUDA_CHECK(cudaMalloc(&res, sizeof(Point)));
    accumulate_result<<<1, 1>>>(window_res, num_windows, res);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t end;
    cudaEventCreate(&end);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float accumulate_ms;
    cudaEventElapsedTime(&accumulate_ms, gather_stop, end);
    printf("Time taken to accumulate result: %f\n", accumulate_ms);

    float total_time;
    cudaEventElapsedTime(&total_time, start, end);
    printf("Total time taken: %f ms\n", total_time);

#if debug
    print_point<<<1, 1>>>(res, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // Free memory
    CUDA_CHECK(cudaFree(scalar_chunks));
    CUDA_CHECK(cudaFree(indices));
    CUDA_CHECK(cudaFree(offset));
    CUDA_CHECK(cudaFree(offset_counter));
    CUDA_CHECK(cudaFree(count));
    CUDA_CHECK(cudaFree(sum));
    CUDA_CHECK(cudaFree(window_res));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(end);
}

#endif