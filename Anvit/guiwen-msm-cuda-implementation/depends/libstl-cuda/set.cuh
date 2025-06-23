#ifndef __STL_CUDA_SET_CUH__
#define __STL_CUDA_SET_CUH__

#include "memory.cuh"
#include <vector>
#include <cassert>

namespace libstl
{

    template <typename T>
    class set
    {
    public:
        typedef T *iterator;
        typedef const T *const_iterator;
        typedef set<T> self_type;

    public:
        T *_data;
        size_t _size;
        size_t _capacity;

        AllocatorManager _allocManager;

        __host__ __device__ set();

        __device__ set(size_t max_size);

        __device__ set(size_t max_size, const T& val);

        __host__ __device__ ~set();
        __device__ bool contains(const T value);
        __device__ void insert(const T value);
        __device__ void remove(const T value);

        __device__ set<T> &operator=(const self_type &x);

        __device__ set<T> &operator=(self_type &&x);

        __host__ __device__ T &operator[](size_t n);

        __host__ __device__ const T &operator[](size_t n) const;

        __device__ T &front();

        __device__ const T &front() const;

        __device__ T &back();

        __device__ const T &back() const;

        __device__ iterator begin();

        __device__ const_iterator begin() const;

        __device__ const_iterator cbegin() const;

        __device__ iterator end();

        __device__ const_iterator end() const;

        __device__ const_iterator cend() const;

        __device__ size_t size() const;

        __host__ size_t size_host();

        
    };
}
#endif
