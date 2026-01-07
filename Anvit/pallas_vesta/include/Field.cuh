#ifndef __FIELD_CUH
#define __FIELD_CUH

#include <cstdio>
#include <cuda_runtime.h>
#include "Scalar.cuh"

class __align__(16) Field
{
public:
    __device__ u_int64_t data[4];
    //Constructor without any argument 
    __device__ Field();
    __device__ Field(const u_int64_t *uint64_le, size_t len = 4);
    __device__ Field(const Field &other);
    __device__ Field(uint64_t val);


    //Relational Operators
    __device__ bool operator==(const Field &other);
    __device__ bool operator==(const Field &other) const;
    __device__ bool operator!=(const Field &other);
    __device__ bool operator!=(const Field &other) const;

    __device__ bool operator>=(const Field &other);
    __device__ bool operator>=(const Field &other) const;
    __device__ bool operator<=(const Field &other);
    __device__ bool operator<=(const Field &other) const;

    //Assignment operators
    __device__ Field &operator=(const Field &other);
    

    // Arithmatic operators
    __device__ Field &operator+=(const Field &other);
    __device__ Field &operator-=(const Field &other);
    __device__ Field &operator*=(const Field &other);

    __device__ Field operator*(const Scalar &other);

    __device__ Field operator+(const Field &other);
    __device__ Field operator+(const Field &other) const;
    __device__ Field operator-(const Field &other);
    __device__ Field operator-(const Field &other) const;
    __device__ Field operator*(const Field &other);
    __device__ Field operator*(const Field &other) const;

    //Negation operator
    __device__ Field operator-();
    __device__ Field operator-() const;

    
    // double the field element
    __device__ Field dbl();
    // square of the field element
    __device__ Field squared();
    __device__ Field squared() const;
    // clear the set Field values
    __device__ void clear();
    // Check if equal to zero
    __device__ bool is_zero();
    __device__ bool is_zero() const;


    // go to montgomery representation
    __device__ inline void encode_montgomery();
    // get out of montgomery representation
    __device__ inline void decode_montgomery();
    // find the inverse of the field element 
    __device__ Field inverse();


    __device__ Field one();
    __device__ Field zero();

    __device__ Field as_scalar();

    __device__ void print();

};


#include "./../src/Field.cu"

#endif