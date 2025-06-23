#ifndef __FieldG1_CUH
#define __FieldG1_CUH

#include "Scalar.cuh"

/*
    This class represents an element from the G1 Field of the EC. 
    The values are stored in montgomery form and the operations are performed on montgomery representation. 
*/
class alignas(16) FieldG1
{
public:
    __uint64_t data[6];
    // initialize to zero
    __device__ FieldG1();
    // initialize from byte sequence
    // __device__ FieldG1(const __uint8_t *bytes, size_t length = 48); // Field length 381 bits = 48 bytes

    __device__ FieldG1(const __uint64_t *limbs_in_lendian, int limb_count=6);

    // initialize from another field object
    __device__ FieldG1(const FieldG1 &other);
    // initialize from uint64
    __device__ FieldG1(const __uint64_t scalar64);

    // relational operators
    __device__ bool operator==(const FieldG1 &other);
    __device__ bool operator!=(const FieldG1 &other);
    __device__ bool operator>=(const FieldG1 &other);
    __device__ bool operator<=(const FieldG1 &other);

    // assignment from another FieldG1 object
    __device__ FieldG1 &operator=(const FieldG1 &other);
    // assignment from Scalar object
    __device__ FieldG1 &operator=(const Scalar &other);

    // Operator +=
    __device__ FieldG1 &operator+=(const FieldG1 &other);
    // Operator -=
    __device__ FieldG1 &operator-=(const FieldG1 &other);

    __device__ FieldG1 &operator*=(const FieldG1 &other);
    __device__ FieldG1 &operator*=(const Scalar &other);

    __device__ FieldG1 operator+(const FieldG1 &other);

    __device__ FieldG1 operator-();
    __device__ FieldG1 operator-(const FieldG1 &other);

    __device__ FieldG1 operator*(const FieldG1 &other);
    __device__ FieldG1 operator*(const Scalar &other);

    // double the field element
    __device__ FieldG1 dbl();
    // square of the field element
    __device__ FieldG1 squared();
    // clear the set Field values
    __device__ void clear();
    // Check if equal to zero
    __device__ bool is_zero();

    // get a zero initialized object
    __device__ FieldG1 zero();
    // get an one initialized object
    __device__ FieldG1 one();
    // get a random initialized Field object
    __device__ FieldG1 random();
    // go to montgomery representation
    __device__ inline void encode_montgomery();
    // get out of montgomery representation
    __device__ inline void decode_montgomery();
    // find the inverse of the field element 
    __device__ FieldG1 inverse();

    __host__ __device__ void print();
};

#include "./../src/FieldG1.cu"

#endif