#ifndef __STL_CUDA_SET_CU__
#define __STL_CUDA_SET_CU__

#include "utility.cuh"
#include "set.cuh"

namespace libstl
{

    /**
     * Default constructor, initializes an empty set. Not recommended for performance reasons unless you don't care about performance.
     * Be careful to use this, insertion in this set will require allocation of memory and copy previous elements which results in performance overhead.
     */
    template <typename T>
    __host__ __device__ set<T>::set() : _size(0), _data(nullptr), _capacity(0)
    {
    }
    /**
     * Constructor with maximum size, initializes the set with default constructed T's. Elements after the maximum size will cause performance overhead.
     */
    template <typename T>
    __device__ set<T>::set(size_t n) : _size(0), _capacity(n)
    {
        _data = _capacity == 0 ? nullptr : (T *)_allocManager.allocate(_capacity * sizeof(T));

        uninitialized_fill_n(_data, _capacity, T());
    }
    /**
     * Constructor with maximum size, initializes the set elements with provided value. Elements after the maximum size will cause performance overhead.
     */
    template <typename T>
    __device__ set<T>::set(size_t n, const T &val) : _size(0), _capacity(n)
    {
        _data = _capacity == 0 ? nullptr : (T *)_allocManager.allocate(_capacity * sizeof(T));
        uninitialized_fill_n(_data, _capacity, val);
    }

    template <typename T>
    __host__ __device__ set<T>::~set()
    {
    }

    template <typename T>
    __device__ bool set<T>::contains(const T value)
    {

        size_t index = _size;
        for (size_t i = 0; i < _size; ++i)
        {
            if (_data[i] == value)
            {
                index = i;
                break;
            }
        }
        return index < _size; 
    }
    template <typename T>
    __device__ void set<T>::insert(const T value)
    {
        if (!contains(value))
        {
            if (_size >= _capacity)
            {
                _capacity = _capacity + 100;
                T *newloc = (T *)_allocManager.allocate(_capacity * sizeof(T));
                uninitialized_copy_n(_data, _size, newloc);
                _data = newloc;
            }

            // Insert the value in the next available position
            _data[_size] = value;
            ++_size;
        }
    }
    template <typename T>
    __device__ void set<T>::remove(const T value)
    {
        size_t index = _size;
        for (size_t i = 0; i < _size; ++i)
        {
            if (_data[i] == value)
            {
                index = i;
                break;
            }
        }
        if (index < _size)
        {
            for (size_t i = index; i < _size - 1; ++i)
            {
                _data[i] = _data[i + 1];
            }
            --_size;
        }
    }

    template <typename T>
    __device__ set<T> &set<T>::operator=(const self_type &x)
    {
        if (this == &x)
            return *this;

        if (_size != x._size)
        {
            _size = x._size;
            _capacity = x._capacity;
            _data = _capacity == 0 ? nullptr : (T *)_allocManager.allocate(_capacity * sizeof(T));
        }

        uninitialized_copy_n(x.begin(), _size, _data);

        return *this;
    }

    template <typename T>
    __device__ set<T> &set<T>::operator=(self_type &&x)
    {
        if (this == &x)
            return *this;

        _size = x._size;
        _capacity = x._capacity;
        _data = x._data;

        x._size = 0;
        x._capacity = 0;
        x._data = nullptr;

        return *this;
    }

    template <typename T>
    __host__ __device__ T &set<T>::operator[](size_t n)
    {
        return _data[n];
    }

    template <typename T>
    __host__ __device__ const T &set<T>::operator[](size_t n) const
    {
        return _data[n];
    }

    template <typename T>
    __device__ T &set<T>::front()
    {
        return _data[0];
    }

    template <typename T>
    __device__ const T &set<T>::front() const
    {
        return _data[0];
    }

    template <typename T>
    __device__ T &set<T>::back()
    {
        return _data[_size - 1];
    }

    template <typename T>
    __device__ const T &set<T>::back() const
    {
        return _data[_size - 1];
    }

    template <typename T>
    __device__ set<T>::iterator set<T>::begin()
    {
        return _data;
    }

    template <typename T>
    __device__ set<T>::const_iterator set<T>::begin() const
    {
        return _data;
    }

    template <typename T>
    __device__ set<T>::const_iterator set<T>::cbegin() const
    {
        return _data;
    }

    template <typename T>
    __device__ set<T>::iterator set<T>::end()
    {
        return _data + _size;
    }

    template <typename T>
    __device__ set<T>::const_iterator set<T>::end() const
    {
        return _data + _size;
    }

    template <typename T>
    __device__ set<T>::const_iterator set<T>::cend() const
    {
        return _data + _size;
    }

    template <typename T>
    __device__ size_t set<T>::size() const
    {
        return _size;
    }

    template <typename T>
    __host__ size_t set<T>::size_host()
    {
        size_t size;
        get_host(&size, &this->_size);
        return size;
    }

}
#endif
