#ifndef __FIELDG1_CU
#define __FIELDG1_CU

#include "./../include/FieldG1.cuh"
#include "./../utils/common_limb_operations.cu"
#include "./../utils/bls12_381_constants.cu"
#include <cassert>

#define LIMBS 6
// initialize to zero
__device__ FieldG1::FieldG1()
{
    for (int i = 0; i < LIMBS; ++i)
        data[i] = 0;
}
// initialize from byte sequence
// __device__ FieldG1::FieldG1(const __uint8_t *bytes, size_t length)
// {
//     for (int i = 0; i < LIMBS; ++i)
//     {
//         data[i] = 0;
//         for (int j = 0; j < 8; ++j)
//         {
//             data[i] |= ((__uint64_t)bytes[i * 8 + j]) << (8 * j);
//         }
//     }

//     encode_montgomery();
// }

__device__ FieldG1::FieldG1(const __uint64_t *limbs_in_lendian, int limb_count)
{
    for (int i = 0; i < limb_count; i++)
        data[i] = limbs_in_lendian[i];
    this->encode_montgomery();
}
// initialize from another field object
__device__ FieldG1::FieldG1(const FieldG1 &other)
{
    for (int i = 0; i < LIMBS; i++)
        data[i] = other.data[i];
}
// initialize from uint64
__device__ FieldG1::FieldG1(const __uint64_t scalar64)
{
    for (int i = 0; i < LIMBS; ++i)
        data[i] = 0;
    data[0] = scalar64;

    encode_montgomery();
}

__device__ bool FieldG1::operator==(const FieldG1 &other)
{
    return equal(this->data, other.data, LIMBS);
}
__device__ bool FieldG1::operator!=(const FieldG1 &other)
{
    return !(operator==(other));
}
__device__ bool FieldG1::operator>=(const FieldG1 &other)
{
    for (int i = LIMBS - 1; i >= 0; i--)
    {
        if (data[i] > other.data[i])
            return true;
        if (data[i] < other.data[i])
            return false;
    }

    return true;
}
__device__ bool FieldG1::operator<=(const FieldG1 &other)
{
    for (int i = LIMBS - 1; i >= 0; i--)
    {
        if (data[i] < other.data[i])
            return true;
        if (data[i] > other.data[i])
            return false;
    }

    return true;
}
// assignment from another FieldG1 object
__device__ FieldG1 &FieldG1::operator=(const FieldG1 &other)
{
    for (int i = 0; i < 6; i++)
        data[i] = other.data[i];
    return *this;
}
// assignment from Scalar object
__device__ FieldG1 &FieldG1::operator=(const Scalar &other)
{
    for (int i = 0; i < 4; i++)
        data[i] = other.data[i];
    data[4] = 0;
    data[5] = 0;
    this->encode_montgomery();
    return *this;
}

// Operator +=
__device__ FieldG1 &FieldG1::operator+=(const FieldG1 &other)
{
    add_limbs(this->data, this->data, other.data, LIMBS);
    conditional_subtract(this->data, bls12_381::modulus, LIMBS);
    return *this;
}
// Operator -=
__device__ FieldG1 &FieldG1::operator-=(const FieldG1 &other)
{
    __uint64_t res[LIMBS];
    bool borrow = sub_limbs(res, this->data, other.data, LIMBS);
    if (borrow)
    {
        add_limbs(this->data, bls12_381::modulus, this->data, LIMBS);
        sub_limbs(this->data, this->data, other.data, LIMBS);
    }
    else
    {
        copy_limbs(this->data, res, LIMBS);
    }
    conditional_subtract(this->data, bls12_381::modulus, LIMBS);
    return *this;
}

__device__ FieldG1 &FieldG1::operator*=(const FieldG1 &other)
{
    __uint64_t res[2 * LIMBS];
    mont_mul(res, this->data, other.data, bls12_381::modulus, bls12_381::mont_inv, LIMBS);
    conditional_subtract(res, bls12_381::modulus, LIMBS);
    copy_limbs(data, res, LIMBS);
    return *this;
}
__device__ FieldG1 &FieldG1::operator*=(const Scalar &other)
{
    FieldG1 temp;
    temp = other;
    *this *= temp;
    return *this;
}

__device__ FieldG1 FieldG1::operator+(const FieldG1 &other)
{
    FieldG1 result;
    add_limbs(result.data, this->data, other.data, LIMBS);
    conditional_subtract(result.data, bls12_381::modulus, LIMBS);
    return result;
}

__device__ FieldG1 FieldG1::operator-()
{
    if (this->is_zero())
        return *this;

    FieldG1 result;
    sub_limbs(result.data, bls12_381::modulus, this->data, LIMBS);  
    return result;
}

__device__ FieldG1 FieldG1::operator-(const FieldG1 &other)
{
    FieldG1 result(*this);
    result -= other;
    return result;
}

__device__ FieldG1 FieldG1::operator*(const FieldG1 &other)
{
    FieldG1 result(*this);
    result *= other;
    return result;
}
__device__ FieldG1 FieldG1::operator*(const Scalar &other)
{
    FieldG1 temp;
    temp = other;
    FieldG1 result(*this);
    result *= temp;
    return result;
}

// double the field element
__device__ FieldG1 FieldG1::dbl()
{
    FieldG1 result;
    for (size_t i = LIMBS - 1; i >= 1; i--)
        result.data[i] = (this->data[i] << 1) | (this->data[i - 1] >> (64 - 1));

    result.data[0] = this->data[0] << 1;
    conditional_subtract(result.data, bls12_381::modulus, LIMBS);
    return result;
}
// square of the field element
__device__ FieldG1 FieldG1::squared()
{
    return (*this) * (*this);
}
// clear the set Field values
__device__ void FieldG1::clear()
{
    for (int i = 0; i < LIMBS; i++)
        data[i] = 0;
}
// Check if equal to zero
__device__ bool FieldG1::is_zero()
{
    return is_zero_limbs(data, LIMBS);
}

// get a zero initialized object
__device__ FieldG1 FieldG1::zero()
{
    FieldG1 result;
    return result;
}
// get an one initialized object
__device__ FieldG1 FieldG1::one()
{
    FieldG1 result;
    copy_limbs(result.data, bls12_381::r_mod_p, LIMBS);
    return result;
}
// get a random initialized Field object
__device__ FieldG1 FieldG1::random()
{
    FieldG1 result;
    for (int i = 0; i < LIMBS; i++)
    {
        result.data[i] = 0x1746567387465673;
    }
    result.encode_montgomery();
    return result;
}
// get the number to montgomery representation
__device__ inline void FieldG1::encode_montgomery()
{
    mont_encode(this->data, this->data, bls12_381::r2_mod_p, bls12_381::modulus, bls12_381::mont_inv, LIMBS);
}
// get out of the montgomery representation
__device__ inline void FieldG1::decode_montgomery()
{
    mont_decode(this->data, this->data, bls12_381::modulus, bls12_381::mont_inv, LIMBS);
}
// find the inverse of the field value (in montgomery representation) 
// this uses the fermat's theorem to find the inverse
__device__ FieldG1 FieldG1::inverse()
{
    FieldG1 base(*this);

    FieldG1 result = one(); // Initialize result to 1

    // Exponent: p - 2
    // BLS12-381 modulus - 2
    __uint64_t exponent[6];
    copy_limbs(exponent, bls12_381::modulus, 6);
    exponent[0] -= 2;

    // Perform exponentiation using square-and-multiply
    #pragma unroll 1
    for (int i = 5; i >= 0; --i) {
        #pragma unroll 1
        for (int bit = 63; bit >= 0; --bit) {
            result = result.squared();
            if ((exponent[i] >> bit) & 1) {
                result = result * base;
            }
        }
    }

    return result;
}
// print a field value as it is stored 
__host__ __device__ void FieldG1::print()
{
    for (int i = 5; i >= 0; i--)
        printf("%016lx ", data[i]);
    printf("\n");
}
#endif