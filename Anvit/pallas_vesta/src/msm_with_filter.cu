#include "./../include/Point.cuh"

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
using namespace cooperative_groups;

#define WINDOW_SIZE 16
#define NUM_BITS 256

// 1. process_scalar_into_bucket: warp-aggregated counting
__global__ void process_scalar_into_bucket_optimized(
    const Scalar * __restrict__ scalars,
    size_t num_points,
    size_t num_windows,
    uint32_t * __restrict__ scalar_chunks,
    uint32_t * __restrict__ count)
{
    extern __shared__ uint32_t warp_hist[]; // [warpSize][1<<WINDOW_SIZE/WARPS]
    int warp_id = (threadIdx.x / warpSize);
    int lane = threadIdx.x % warpSize;
    int num_warps = blockDim.x / warpSize;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // zero local warp histogram
    for (int b = lane; b < (1<<WINDOW_SIZE); b += warpSize) {
        warp_hist[warp_id * (1<<WINDOW_SIZE) + b] = 0;
    }
    __syncwarp();

    // build per-warp histograms
    for (; idx < num_points; idx += gridDim.x * blockDim.x) {
        Scalar s = scalars[idx];
        for (int w = 0; w < num_windows; ++w) {
            uint32_t bidx = 0;
            int start = w * WINDOW_SIZE;
            // test WINDOW_SIZE bits
            for (int j = 0; j < WINDOW_SIZE; ++j) {
                bidx |= (s.test_bit(start + j) << j);
            }
            scalar_chunks[idx + w * num_points] = bidx;
            atomicAdd(&warp_hist[warp_id * (1<<WINDOW_SIZE) + bidx], 1);
        }
    }
    __syncwarp();

    // warp leader atomically update global count
    if (lane == 0) {
        for (int b = 0; b < (1<<WINDOW_SIZE); ++b) {
            uint32_t c = warp_hist[warp_id * (1<<WINDOW_SIZE) + b];
            if (c) atomicAdd(&count[b + blockIdx.y * (1<<WINDOW_SIZE)], c);
        }
    }
}

// 2. construct_bucket_indices: use CUB DeviceScan offsets + simple scatter
__global__ void construct_bucket_indices_optimized(
    const uint32_t * __restrict__ scalar_chunks,
    const uint32_t * __restrict__ offset,
    uint32_t * __restrict__ indices,
    size_t num_points,
    size_t num_windows,
    size_t num_buckets)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points * num_windows) return;
    uint32_t bidx = scalar_chunks[idx];
    if (bidx == 0) return;
    // compute global position from offset table (pre-scanned)
    indices[atomicAdd(&offset[idx], 1)] = idx % num_points;
}

// 3. sum_small_bucket: remove pipelines, use __ldg and shared index tiling
__global__ void sum_small_bucket_optimized(
    const Point * __restrict__ points,
    const uint32_t * __restrict__ offset,
    const uint32_t * __restrict__ indices,
    const uint32_t * __restrict__ count,
    Point * __restrict__ sum,
    size_t num_buckets)
{
    extern __shared__ uint32_t tile_idx[];
    size_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= num_buckets) return;
    uint32_t n = count[bucket];
    if (n == 0 || n >= 128) return;

    // load indices into shared mem tile
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        tile_idx[i] = indices[offset[bucket] + i];
    }
    __syncthreads();

    Point acc; acc.set_zero();
    for (uint32_t i = 0; i < n; ++i) {
        Point p = __ldg(&points[tile_idx[i]]);
        acc = acc.mixed_add(p);
    }
    sum[bucket] = acc;
}

// 4. sum_medium_bucket: cooperative_groups reduction
__global__ void sum_medium_bucket_optimized(
    const Point * __restrict__ points,
    const uint32_t * __restrict__ offset,
    const uint32_t * __restrict__ indices,
    const uint32_t * __restrict__ count,
    Point * __restrict__ sum,
    size_t num_buckets)
{
    thread_block tb = this_thread_block();
    int bucket = blockIdx.x;
    uint32_t n = count[bucket];
    if (n < 128 || n > 128*256) return;

    size_t start = offset[bucket];
    Point local = Point::zero();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        Point p = __ldg(&points[indices[start + i]]);
        local = local.mixed_add(p);
    }
    // block-wide reduction
    Point total = reduce(tb, local, plus<Point>());
    if (threadIdx.x == 0) sum[bucket] = total;
}

// 5. gather_bucket_parallel: use CUB BlockScan for segmented sum
__global__ void gather_bucket_parallel_optimized(
    const Point * __restrict__ sum,
    size_t num_buckets,
    Point * __restrict__ window_res)
{
    using BlockScan = cub::BlockScan<Point, 256>;
    __shared__ typename BlockScan::TempStorage temp;

    int w = blockIdx.x;
    int tid = threadIdx.x;
    // load reversed
    Point val;// = Point::zero();
    int idx = num_buckets - 1 - tid;
    if (idx >= 0) val = sum[idx + w*num_buckets];

    Point scanned;
    BlockScan(temp).ExclusiveSum(val, scanned, Point());
    __syncthreads();
    if (tid == 0) window_res[w] = scanned;
}

// 6. accumulate_result: host or simple kernel
__global__ void accumulate_result_optimized(
    const Point * __restrict__ window_res,
    size_t num_windows,
    Point * __restrict__ res)
{
    Point acc; acc = acc.zero();
    for (int i = num_windows - 1; i >= 0; --i) {
        // window shift: acc <<= WINDOW_SIZE
        for (int b = 0; b < WINDOW_SIZE; ++b) acc = acc.dbl();
        acc = acc + window_res[i];
    }
    *res = acc;
}

Point* cuda_pippenger_msm_optimized(
    const Point *d_points,
    const Scalar *d_scalars,
    size_t num_points)
{
    // Compute windows and buckets
    const int num_windows = (NUM_BITS + WINDOW_SIZE - 1) / WINDOW_SIZE;
    const size_t num_buckets = static_cast<size_t>(1) << WINDOW_SIZE;

    // Allocate device buffers
    uint32_t *d_scalar_chunks, *d_count, *d_offset, *d_indices;
    Point    *d_sum, *d_window_res, *d_res;

    size_t chunks_size = num_points * num_windows * sizeof(uint32_t);
    size_t buckets_size = num_buckets * num_windows * sizeof(uint32_t);
    size_t sum_size     = num_buckets * num_windows * sizeof(Point);

    cudaMalloc(&d_scalar_chunks, chunks_size);
    cudaMalloc(&d_count,         buckets_size);
    cudaMalloc(&d_offset,        buckets_size);
    cudaMalloc(&d_indices,       chunks_size);
    cudaMalloc(&d_sum,           sum_size);
    cudaMalloc(&d_window_res,    num_windows * sizeof(Point));
    cudaMalloc(&d_res,           sizeof(Point));

    cudaMemset(d_count, 0,         buckets_size);
    cudaMemset(d_offset,0,         buckets_size);

    // 1) Scalar chunking & bucket counting
    {
        dim3 block(256);
        dim3 grid((num_points + block.x-1)/block.x);
        size_t shared_bytes = (block.x/warpSize) * (1<<WINDOW_SIZE) * sizeof(uint32_t);
        process_scalar_into_bucket_optimized<<<grid,block,shared_bytes>>>(
            d_scalars, num_points, num_windows,
            d_scalar_chunks, d_count);
        cudaDeviceSynchronize();
    }

    // 2) Exclusive scan on count -> offsets
    void   *d_temp_storage = nullptr;
    size_t temp_bytes      = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_bytes,
        d_count, d_offset,
        static_cast<int>(num_buckets * num_windows));
    cudaMalloc(&d_temp_storage, temp_bytes);
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_bytes,
        d_count, d_offset,
        static_cast<int>(num_buckets * num_windows));
    cudaFree(d_temp_storage);

    // 3) Build index lists
    {
        dim3 block(256);
        dim3 grid((num_points*num_windows + block.x-1)/block.x);
        construct_bucket_indices_optimized<<<grid,block>>>(
            d_scalar_chunks, d_offset,
            d_indices, num_points,
            num_windows, num_buckets);
        cudaDeviceSynchronize();
    }

    // 4) Small-bucket summation
    {
        dim3 block(128);
        dim3 grid((num_buckets*num_windows + block.x-1)/block.x);
        size_t shared_idx = block.x * sizeof(uint32_t);
        sum_small_bucket_optimized<<<grid,block,shared_idx>>>(
            d_points, d_offset, d_indices,
            d_count, d_sum,
            num_buckets * num_windows);
        cudaDeviceSynchronize();
    }

    // 5) Medium-bucket summation
    {
        dim3 block(256);
        dim3 grid(num_buckets * num_windows);
        sum_medium_bucket_optimized<<<grid,block>>>(
            d_points, d_offset, d_indices,
            d_count, d_sum,
            num_buckets * num_windows);
        cudaDeviceSynchronize();
    }

    // 6) Gather per-window sums
    {
        dim3 block(256);
        dim3 grid(num_windows);
        size_t shared_bytes = cub::BlockScan<Point,256>::TempStorageSize();
        gather_bucket_parallel_optimized<<<grid,block,shared_bytes>>>(
            d_sum, num_buckets, d_window_res);
        cudaDeviceSynchronize();
    }

    // 7) Final accumulate
    {
        accumulate_result_optimized<<<1,WINDOW_SIZE>>>(
            d_window_res, num_windows,
            d_res);
        cudaDeviceSynchronize();
    }

    // Cleanup intermediate buffers
    cudaFree(d_scalar_chunks);
    cudaFree(d_count);
    cudaFree(d_offset);
    cudaFree(d_indices);
    cudaFree(d_sum);
    cudaFree(d_window_res);

    return d_res;
}