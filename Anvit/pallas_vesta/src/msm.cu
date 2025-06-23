#ifndef __MSM_CUH
#define __MSM_CUH

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

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

// struct to represent a bucket. Each bucket contains count and point indices associated with the bucket
struct bucket
{
    size_t count = 0;
    size_t *points;
    // initialize the point indices array
    __host__ void init(size_t num_point)
    {
        CUDA_CHECK(cudaMalloc(&points, sizeof(size_t) * 10240));
    }
    // method to insert index to the bucket
    __device__ void insert(size_t idx)
    {
        points[atomicAdd((unsigned long long *)&count, 1)] = idx;
    }
};

// host function to initialize 1<<WINDOW_SIZE buckets, each to accomodate num_points (maximum) points in each bucket
void construct_buckets(size_t num_points, bucket *buckets)
{
    for (size_t i = 0; i < (size_t)1 << WINDOW_SIZE; i++)
        buckets[i].init(num_points + 1);
}

// Process the scalars to construct a CSR Matrix like bucket, only processes one window at a time
__global__ void process_scalars(Scalar *sc, Point *points, size_t num_points, bucket *bucket, int current_window)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < num_points)
    {
        size_t bindex = 0;
        size_t start = current_window * WINDOW_SIZE;
        size_t end = start + WINDOW_SIZE;
        for (size_t i = start, j = 0; i < end; i++, j++)
        {
            if (sc[idx].test_bit(i))
            {
                bindex |= (1 << j);
            }
        }
        bucket[bindex].insert(idx);
        idx += stride;
    }
}

// Sum the bucket points. Needs balancing here to improve performance.
__global__ void sum_buckets(Point *point, bucket *bucket, Point *sum, size_t num_bucket)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    Point lsum;
    lsum = lsum.zero();
    while (idx < num_bucket)
    {
        for (size_t i = 0; i < bucket[idx].count; i++)
        {
            lsum = lsum.mixed_add(point[bucket[idx].points[i]]);
        }
        lsum = lsum * idx;

        sum[idx] = lsum;
        idx += stride;
    }
}
// sum all the buckets of a window and store the result to a point
__global__ void gather_buckets(Point *sum, size_t bcount, Point *res)
{
    Point lsum, running_sum;
    lsum = lsum.zero();
    running_sum = running_sum.zero();
    for (size_t i = bcount - 1; i > 0; i--)
    {
        lsum = lsum + sum[i];
        // running_sum = running_sum + lsum;
    }
    *res = lsum;
    // *res = running_sum;
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
__global__ void print_bucket(bucket *buckets)
{
    for (size_t i = 0; i < ((size_t)1 << WINDOW_SIZE); i++)
        if (buckets[i].count)
        {
            printf("\nBucket: %lu, Count: %lu\n", i, buckets[i].count);
            for (int j = 0; j < 10; j++)
            {
                // (buckets[i].points[j])->print();
                printf("%lu ", buckets[i].points[j]);
            }
        }
}

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
void cuda_pippenger_msm(Point *points, Scalar *scalars, size_t num_points)
{
    int num_windows = (NUM_BITS + WINDOW_SIZE - 1) / WINDOW_SIZE;
    bucket *buckets, *d_bucket;
    size_t num_bucket = ((size_t)1 << WINDOW_SIZE);
    buckets = new bucket[num_bucket];
    CUDA_CHECK(cudaMalloc(&d_bucket, sizeof(bucket) * (num_bucket)));
    CUDA_CHECK(cudaDeviceSynchronize());
    construct_buckets(num_points, buckets);

    CUDA_CHECK(cudaMemcpy(d_bucket, buckets, sizeof(bucket) * (num_bucket), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    Point *bucket_sums;
    CUDA_CHECK(cudaMalloc(&bucket_sums, sizeof(Point) * (num_bucket)));

    Point *window_sum;
    CUDA_CHECK(cudaMalloc(&window_sum, sizeof(Point) * num_windows));

    Point *result;
    CUDA_CHECK(cudaMalloc(&result, sizeof(Point)));
    cudaEvent_t start, stop;

    for (int i = 0; i < num_windows; i++)
    {
        
        process_scalars<<<512, 32>>>(scalars, points, num_points, d_bucket, i);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // #if debug
        //         printf("Bucket status %d\n", i);
        //         print_bucket<<<1, 1>>>(d_bucket);
        // #endif
        size_t blockSize = 256;
        size_t gridSize = (num_bucket + blockSize - 1) / blockSize;

        

        sum_buckets<<<gridSize, blockSize>>>(points, d_bucket, bucket_sums, num_bucket);
        CUDA_CHECK(cudaDeviceSynchronize());
        // #if debug
        //         printf("Bucket sums after sum_bucket\n");
        //         print_point<<<1, 1>>>(bucket_sums, num_bucket);
        //         CUDA_CHECK(cudaDeviceSynchronize());
        // #endif


        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaEventSynchronize(start);
        gather_buckets<<<1, 1>>>(bucket_sums, num_bucket, &window_sum[i]);
        CUDA_CHECK(cudaDeviceSynchronize());


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Time taken: %f\n", ms);

        // #if debug
        //         printf("Window %d sum:\n", i);
        //         print_point<<<1, 1>>>(&window_sum[i]);
        //         CUDA_CHECK(cudaDeviceSynchronize());
        // #endif
        // #if debug
        //         print_bucket<<<1, 1>>>(d_bucket);
        //         CUDA_CHECK(cudaDeviceSynchronize());
        // #endif
        // reset bucket
        CUDA_CHECK(cudaMemcpy(d_bucket, buckets, sizeof(bucket) * num_bucket, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        
    }
    accumulate_result<<<1, 1>>>(window_sum, num_windows, result);
    CUDA_CHECK(cudaDeviceSynchronize());

#if debug
    print_point<<<1, 1>>>(result);
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

#endif