#ifndef __VECTOR_CUH
#define __VECTOR_CUH

#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <cstdio>

template <typename T>
class alignas(16) vector {
public:
    using iterator = T*;
    using const_iterator = const T*;
    using self_type = vector<T>;

    __host__ __device__ vector();
    __device__ vector(size_t size);
    __device__ vector(size_t size, const T &value);
    __host__ __device__ ~vector();

    __device__ self_type& operator=(const self_type &x);
    __device__ self_type& operator=(self_type &&x);

    __device__ T& operator[](size_t index);
    __device__ const T& operator[](size_t index) const;

    __device__ T& front();
    __device__ const T& front() const;
    __device__ T& back();
    __device__ const T& back() const;

    __device__ size_t size() const;
    __host__ size_t size_host() const;
    __device__ void resize(size_t new_size);
    __device__ void resize(size_t n, const T& val);
    __host__ void resize_host(size_t n, const T& val);
    __host__ void resize_host(size_t n);

    __device__ void presize(size_t n, const T& val, size_t gridSize, size_t blockSize);

    __device__ iterator begin();
    __device__ const_iterator begin() const;
    __device__ iterator end();
    __device__ const_iterator end() const;

public:
    T* _data;
    size_t _size;
};
// template<typename T, typename H>
// __host__ void vector_device2host(vector<H>* hv, const vector<T>* dv, cudaStream_t stream = 0);

// template<typename T, typename H>
// __host__ void vector_host2device(vector<T>* dv, const vector<H>* hv, cudaStream_t stream = 0);

// template<typename T>
// __host__ void vector_device2host(vector<T>* hv, const vector<T>* dv, cudaStream_t stream = 0);

// template<typename T>
// __host__ void vector_host2device(vector<T>* dv, const vector<T>* hv, cudaStream_t stream = 0);


template <typename T>
class Vector
{
public:
    __device__ T* _data;
    __device__ size_t _size;
    using iterator = T*;
    using const_iterator = const T*;
    using self_type = Vector<T>;

    __host__ __device__ Vector();
    __host__ __device__ Vector(size_t size);
    __host__ __device__ void resize(size_t size);
    __host__ __device__ bool resize_fill(size_t size, T &val);

    __host__ ~Vector();
    __device__ T& operator[](size_t index);
    __device__ const T& operator[](size_t index) const;


    __device__ self_type& operator=(const self_type &x);
    __device__ self_type& operator=(self_type &&x);

    __host__ void presize(size_t size, size_t grid, size_t block);
    __host__ void presize(size_t size, T &val, size_t grid, size_t block);

    __device__ size_t size();
    __device__ size_t size() const;
    __host__ size_t size_host();
    __host__ size_t size_host() const;

};


#include "./../src/vector.cu"

#endif