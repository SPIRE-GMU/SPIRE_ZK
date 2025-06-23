#ifndef __VECTOR_CU
#define __VECTOR_CU
#include <cuda_runtime.h>
#include "./../include/vector.cuh"
// define the constructors
template <typename T>
__host__ __device__ vector<T>::vector() : _data(nullptr), _size(0) {}

template <typename T>
__device__ vector<T>::vector(size_t size) : _size(size)
{
    _data = size == 0 ? nullptr : (T *)malloc(sizeof(T) * size);
}

template <typename T>
__device__ vector<T>::vector(size_t size, const T &value) : _size(size)
{
    _data = size == 0 ? nullptr : (T *)malloc(sizeof(T) * size);
    for (size_t i = 0; i < size; ++i)
        _data[i] = value;
}

template <typename T>
__host__ __device__ vector<T>::~vector()
{
#ifdef __CUDA_ARCH__
    if (_data)
        free(_data); // device-side
#else
    if (_data)
        free(_data); // host-side
#endif
}

// === Assignment Operators ===

template <typename T>
__device__ typename vector<T>::self_type &vector<T>::operator=(const self_type &x)
{
    if (this == &x)
        return *this;
    if (_data)
        free(_data);
    _size = x._size;
    _data = _size ? (T *)malloc(sizeof(T) * _size) : nullptr;
    for (size_t i = 0; i < _size; ++i)
        _data[i] = x._data[i];
    return *this;
}

template <typename T>
__device__ typename vector<T>::self_type &vector<T>::operator=(self_type &&x)
{
    if (this == &x)
        return *this;
    if (_data)
        free(_data);
    _data = x._data;
    _size = x._size;
    x._data = nullptr;
    x._size = 0;
    return *this;
}

// === Element Access ===

template <typename T>
__device__ T &vector<T>::operator[](size_t index)
{
    return _data[index];
}

template <typename T>
__device__ const T &vector<T>::operator[](size_t index) const
{
    return _data[index];
}

template <typename T>
__device__ T &vector<T>::front()
{
    return _data[0];
}

template <typename T>
__device__ const T &vector<T>::front() const
{
    return _data[0];
}

template <typename T>
__device__ T &vector<T>::back()
{
    return _data[_size - 1];
}

template <typename T>
__device__ const T &vector<T>::back() const
{
    return _data[_size - 1];
}

// === Size and Resize ===

template <typename T>
__device__ size_t vector<T>::size() const
{
    return _size;
}

template <typename T>
__host__ size_t vector<T>::size_host() const
{
    size_t size;
    cudaMemcpy(&size, &_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    return size;
}

template <typename T>
__device__ void vector<T>::resize(size_t new_size)
{
    if (_size == new_size)
        return;

    T *new_data = new_size ? (T *)malloc(sizeof(T) * new_size) : nullptr;
    size_t min_size = new_size < _size ? new_size : _size;

    for (size_t i = 0; i < min_size; ++i)
        new_data[i] = _data[i];

    for (size_t i = _size; i < new_size; ++i)
        new_data[i] = T(); // default init

    if (_data)
        free(_data);
    _data = new_data;
    _size = new_size;
}

template <typename T>
__device__ void vector<T>::resize(size_t n, const T &val)
{
    if (_size == n)
        return;

    T *new_data = n ? (T *)malloc(sizeof(T) * n) : nullptr;
    size_t min_size = n < _size ? n : _size;

    for (size_t i = 0; i < min_size; ++i)
        new_data[i] = _data[i];

    for (size_t i = min_size; i < n; ++i)
        new_data[i] = val;

    if (_data)
        free(_data);
    _data = new_data;
    _size = n;
}
template <typename T>
__host__ void vector<T>::resize_host(size_t n, const T &val)
{
    if (_size == n)
        return;

    T *new_data = nullptr;
    if (n)
        cudaMalloc((void **)&new_data, sizeof(T) * n);
    else
        new_data = nullptr;

    size_t min_size = n < _size ? n : _size;

    cudaMemcpy(new_data, _data, sizeof(T) * min_size, cudaMemcpyDeviceToDevice);

    launch<<<512, 32>>>([=] __device__
                        {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t tnum = gridDim.x * blockDim.x;
        while (idx < n && idx >= min_size){
            new_data[idx] = val;
            idx += tnum;
        } });

    if (_data)
        cudaFree(_data);
    _data = new_data;
    _size = n;
}
template <typename T>
__host__ void vector<T>::resize_host(size_t n)
{
    if (_size == n)
        return;

    T *new_data = nullptr;
    if (n)
        cudaMalloc((void **)&new_data, sizeof(T) * n);
    else
        new_data = nullptr;

    size_t min_size = n < _size ? n : _size;

    cudaMemcpy(new_data, _data, sizeof(T) * min_size, cudaMemcpyDeviceToDevice);

    launch<<<512, 32>>>([=] __device__
                        {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t tnum = gridDim.x * blockDim.x;
        while (idx < n && idx >= min_size){
            new_data[idx] = T();
            idx += tnum;
        } });

    if (_data)
        cudaFree(_data);
    _data = new_data;
    _size = n;
}

template <typename T>
__global__ void presize_kernel(T *dst, const T *src, size_t old_size, size_t new_size, T val)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < new_size)
    {
        if (idx < old_size)
            dst[idx] = src[idx];
        else
            dst[idx] = val;
    }
}

template <typename T>
__device__ void vector<T>::presize(size_t n, const T &val, size_t gridSize, size_t blockSize)
{
    if (_size == n)
        return;
    T *new_data = n ? (T *)malloc(sizeof(T) * n) : nullptr;

    presize_kernel<<<gridSize, blockSize>>>(new_data, _data, _size, n, val);

    if (_data)
        free(_data);
    _data = new_data;
    _size = n;
}

// === Iterators ===

template <typename T>
__device__ typename vector<T>::iterator vector<T>::begin()
{
    return _data;
}

template <typename T>
__device__ typename vector<T>::const_iterator vector<T>::begin() const
{
    return _data;
}

template <typename T>
__device__ typename vector<T>::iterator vector<T>::end()
{
    return _data + _size;
}

template <typename T>
__device__ typename vector<T>::const_iterator vector<T>::end() const
{
    return _data + _size;
}

// template <typename T, typename H>
// __host__ void vector_device2host(vector<H> *hv, const vector<T> *dv, cudaStream_t stream)
// {
//     size_t vector_size;
//     void *vector_addr;
//     cudaMemcpy(&vector_size, &dv->_size, sizeof(size_t), cudaMemcpyDeviceToHost);
//     cudaMemcpy(&vector_addr, &dv->_data, sizeof(void *), cudaMemcpyDeviceToHost);
//     hv->resize_host(vector_size);

//     if (stream == 0)
//         cudaMemcpy(hv->_data, (void *)vector_addr, vector_size * sizeof(H), cudaMemcpyDeviceToHost);
//     else
//         cudaMemcpyAsync(hv->_data, (void *)vector_addr, vector_size * sizeof(H), cudaMemcpyDeviceToHost, stream);
// }

// template <typename T, typename H>
// __host__ void vector_host2device(vector<T> *dv, const vector<H> *hv, cudaStream_t stream)
// {
//     size_t vector_size = hv->_size;
//     dv->presize_host(vector_size, 512, 32);
//     void *vector_addr;
//     cudaMemcpy(&vector_addr, &dv->_data, sizeof(void *), cudaMemcpyDeviceToHost);
//     if (stream == 0)
//         cudaMemcpy((void *)vector_addr, hv->_data, vector_size * sizeof(T), cudaMemcpyHostToDevice);
//     else
//         cudaMemcpyAsync((void *)vector_addr, hv->_data, vector_size * sizeof(T), cudaMemcpyHostToDevice, stream);
// }

// template <typename T>
// __host__ void vector_device2host(vector<T> *hv, const vector<T> *dv, cudaStream_t stream)
// {
//     vector_device2host<T, T>(hv, dv, stream);
// }

// template <typename T>
// __host__ void vector_host2device(vector<T> *dv, const vector<T> *hv, cudaStream_t stream)
// {
//     vector_host2device<T, T>(dv, hv, stream);
// }

// Vector methods
#ifndef CUDA_CHECK
#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        exit(code);
    }
}

#endif

template <typename T>
__global__ void copy_data(size_t start, size_t size, T *vec, T val)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < start)
        return;
    if (idx >= size)
        return;
    vec[idx] = val;
}
template <typename T>
__host__ __device__ Vector<T>::Vector()
{
    _data = nullptr;
    _size = 0;
}
template <typename T>
__host__ Vector<T>::Vector(size_t size)
{
#ifdef __CUDA_ARCH__
    _size = size;
    _data = (T *)malloc(sizeof(T) * _size);
#else
    _size = size;
    CUDA_CHECK(cudaMalloc(&_data, sizeof(T) * _size));
#endif
}

template <typename T>
__host__ __device__ void Vector<T>::resize(size_t new_size)
{
#ifdef __CUDA_ARCH__
    if(new_size == _size)
        return;
    T *temp = (T *)malloc(sizeof(T) * new_size);
    if(temp == nullptr)
    {
        printf("Allocation error\n");
        exit(1);   
    }
    size_t small = new_size > _size ? _size : new_size;
    for (size_t i = 0; i < small; ++i)
        temp[i] = _data[i];
    free(_data);
    _data = temp;
    _size = new_size;
#else
    if(new_size == _size) return;
    T *temp;
    CUDA_CHECK(cudaMalloc(&temp, sizeof(T) * new_size));
    size_t small = new_size > _size ? _size : new_size;
    CUDA_CHECK(cudaMemcpy(temp, _data, sizeof(T) * small, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(_data));
    _data = temp;
    _size = new_size;
#endif
}

template <typename T>
__host__ __device__ bool Vector<T>::resize_fill(size_t size, T &val)
{

    resize(size);
#ifdef __CUDA_ARCH__
    for (size_t i = 0; i < size; ++i)
        _data[i] = val;
#else
    copy_data<<<512, 64>>>(0, size, _data, val);
#endif
    return true;
}

template <typename T>
__host__ Vector<T>::~Vector()
{
#ifdef __CUDA_ARCH__
    if (_data)
        free(_data);
#else
    if (_data)
        CUDA_CHECK(cudaFree(_data));
#endif
}

template <typename T>
__device__ T &Vector<T>::operator[](size_t index)
{
    if(index < _size)
        return _data[index];
}

template <typename T>
__device__ const T &Vector<T>::operator[](size_t index) const
{
    if(index < _size)
        return _data[index];
}

template <typename T>
__device__ Vector<T> &Vector<T>::operator=(const self_type &x)
{

    if (this != &x)
    {
        _size = x._size;
        _data = x._data;
    }
    return *this;
}

template <typename T>
__device__ Vector<T> &Vector<T>::operator=(self_type &&x)
{
    if (this != &x)
    {
        _size = x._size;
        _data = x._data;
        x._data = nullptr;
        x._size = 0;
    }
    return *this;
}

template <typename T>
__host__ void Vector<T>::presize(size_t size, size_t grid, size_t block)
{
    resize(size);
}

template <typename T>
__host__ void Vector<T>::presize(size_t size, T &val, size_t grid, size_t block)
{
    resize(size, val);
}

template <typename T>
__device__ size_t Vector<T>::size()
{
    return _size;
}

template <typename T>
__device__ size_t Vector<T>::size() const
{
    return _size;
}

template <typename T>
__host__ size_t Vector<T>::size_host()
{
    size_t s;
    cudaMemcpy(&s, &_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    return s;
}

template <typename T>
__host__ size_t Vector<T>::size_host() const
{
    size_t s;
    cudaMemcpy(&s, &_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    return s;
}

#endif