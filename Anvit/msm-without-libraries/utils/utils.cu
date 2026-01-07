#ifndef UTILS_CU
#define UTILS_CU
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <type_traits>
#include "./../include/G1Point.cuh"
#include "./../include/vector.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#endif
__host__ __device__ size_t log2(size_t n)
/* returns ceil(log2(n)), so 1ul<<log2(n) is the smallest power of 2,
   that is not less than n. */
{
    size_t r = ((n & (n-1)) == 0 ? 0 : 1); // add 1 if n is not power of 2

    while (n > 1)
    {
        n >>= 1;
        r++;
    }

    return r;
}
__global__ void init_G1Point(G1Point* points, size_t size)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tnum = gridDim.x * blockDim.x;
    while(tid < size)
    {
        points[tid] = points[tid].zero();
        tid += tnum;
    }
}

__global__ void print_G1Point(G1Point* points, size_t size)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tnum = gridDim.x * blockDim.x;
    while(tid < size)
    {
        points[tid].print();
        tid += tnum;
    }
}

__global__ void from_array_to_vector_point(G1Point* points, vector<G1Point> *vec, size_t size)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tnum = gridDim.x * blockDim.x;
    while(tid < size)
    {
        (*vec)[tid] = points[tid];
        tid += tnum;
    }
}
__global__ void from_array_to_vector_scalar(Scalar* scalars, vector<Scalar> *vec, size_t size)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tnum = gridDim.x * blockDim.x;
    while(tid < size)
    {
        (*vec)[tid] = scalars[tid];
        tid += tnum;
    }
}

// General version for function pointers (host-side kernel launch config)
template <typename Func>
void getOptimalLaunchConfig(Func kernel, int totalElements, size_t* grid_size, size_t* block_size)
{
    // Get the actual pointer type if Func is a __global__ function
    using KernelFuncPtr = typename std::remove_pointer<Func>::type*;

    static_assert(std::is_pointer<Func>::value || std::is_function<Func>::value,
                  "KernelFunc must be a function or function pointer");

    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        reinterpret_cast<int*>(block_size), // block_size
        reinterpret_cast<const void*>(kernel), // kernel as void*
        0,
        0
    );

    *grid_size = (totalElements + *block_size - 1) / *block_size;
}
template <typename T>
struct remove_reference
{
    typedef T type;
};

template <typename T>
struct remove_reference<T &>
{
    typedef T type;
};

template <typename T>
struct remove_reference<T &&>
{
    typedef T type;
};
template <typename T>
__device__ T &&forward(typename remove_reference<T>::type &arg) { return static_cast<T &&>(arg); }

template <typename T>
__device__ T &&forward(typename remove_reference<T>::type &&arg) { return static_cast<T &&>(arg); }


template <typename T>
__device__ T *create(T val)
{
    T *result;
    result = (T*)malloc(sizeof(T));
    *result = val;
    return result;
}

template <typename T>
T *create_host()
{
    T *result;
    cudaMalloc((void **)&result, sizeof(T));
    return result;
}

template <typename Func, typename... Args>
__global__ void launch(const Func func, Args &&...args) { func(forward<Args>(args)...); }

template <typename Func, typename... Args>
__global__ void launch_with_shared(const Func func, Args &&...args)
{
    extern __shared__ unsigned char s[];
    func(forward<Args>(args)..., s);
}

template <typename T, typename... Args>
__host__ void construct_host(T *ptr, Args &&...args)
{
    launch<<<1, 1>>>(
        [=] __device__(Args &&...args)
        {
            new ((void *)ptr) T(forward<Args>(args)...);
        },
        args...);
    cudaStreamSynchronize(0);
}
#endif