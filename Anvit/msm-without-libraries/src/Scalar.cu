#ifndef __SCALAR_CU
#define __SCALAR_CU
#include "./../include/Scalar.cuh"
#include "./../utils/common_limb_operations.cu"
#include "./../utils/bls12_381_constants.cu"
#include <cassert>

#define LIMBS 4

__device__ Scalar::Scalar()
{
    for (int i = 0; i < LIMBS; i++)
        data[i] = 0;
}
__device__ Scalar::Scalar(const unsigned long long x)
{
    for (int i = 0; i < LIMBS; i++)
        data[i] = 0;
    data[0] = x;
}

__device__ Scalar::Scalar(const __uint8_t *bytes, size_t length)
{
    for (int i = 0; i < LIMBS; ++i)
    {
        data[i] = 0;
        for (int j = 7; j >= 0; --j)
        {
            data[i] |= ((__uint64_t)bytes[i * 8 + j]) << (8 * j);
        }
    }
}
__device__ Scalar::Scalar(const Scalar &other)
{
    for (int i = 0; i < LIMBS; i++)
        data[i] = other.data[i];
}

// assignment from another scalar
__device__ Scalar &Scalar::operator=(const Scalar &other)
{
    for (int i = 0; i < LIMBS; i++)
        data[i] = other.data[i];
    return *this;
}

// relational operators
__device__ bool Scalar::operator==(const Scalar &other)
{
    for (size_t i = 0; i < LIMBS; i++)
        if (data[i] != other.data[i])
            return false;
    return true;
}
__device__ bool Scalar::operator!=(const Scalar &other)
{
    return !(operator==(other));
}
__device__ bool Scalar::operator>=(const Scalar &other)
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
__device__ bool Scalar::operator<=(const Scalar &other)
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

// addition and subtraction operators
__device__ Scalar &Scalar::operator+=(const Scalar &other)
{
    add_limbs(this->data, this->data, other.data, LIMBS);
    conditional_subtract(this->data, bls12_381::fr_modulus, LIMBS);
    return *this;
}
__device__ Scalar &Scalar::operator-=(const Scalar &other)
{
    __uint64_t temp[LIMBS];
    bool borrow = sub_limbs(temp, this->data, other.data, LIMBS);
    if (borrow)
    {
        add_limbs(this->data, this->data, bls12_381::fr_modulus, LIMBS);
        sub_limbs(this->data, this->data, other.data, LIMBS);
    }
    else
    {
        copy_limbs(this->data, temp, LIMBS);
    }
    return *this;
}

__device__ Scalar Scalar::operator+(const Scalar &other)
{
    Scalar result(*this);
    result += other;
    return result;
}
__device__ Scalar Scalar::operator-(const Scalar &other)
{
    Scalar result(*this);
    result -= other;
    return result;
}

// check if the scalar is zero
__device__ bool Scalar::is_zero()
{
    return is_zero_limbs(data, LIMBS);
}
// check if the bit_no is set (1) or not (0)
__device__ bool Scalar::test_bit(size_t bit_no) const
{
    if (bit_no >= sizeof(__uint64_t) * 8 * LIMBS)
        return false;
    const std::size_t part = bit_no / (8 * sizeof(__uint64_t));
    const std::size_t bit = bit_no - (8 * sizeof(__uint64_t) * part);
    const __uint64_t one = 1;
    return (this->data[part] & (one << bit)) != 0;
}
// Set the bit_no as 1
__device__ void Scalar::set_bit(size_t bit_no)
{
    if (bit_no >= LIMBS * 8 * sizeof(__uint64_t))
        return;

    const std::size_t part = bit_no / (8 * sizeof(__uint64_t));
    const std::size_t bit = bit_no - (8 * sizeof(__uint64_t) * part);
    const __uint64_t one = 1;
    this->data[part] |= one << bit;
}

// get a unsigned int32 from bit msb_no to lsb_no
__device__ __uint32_t Scalar::get_bits_as_uint32(size_t msb_no, size_t lsb_no) const
{
    if(msb_no > 255) 
        msb_no = 254;
    assert(msb_no - lsb_no < 32);
    size_t part1 = msb_no / (8 * sizeof(__uint64_t));
    size_t part2 = lsb_no / (8 * sizeof(__uint64_t));
    __uint32_t ret = 0;
    if (part1 == part2)
    {
        size_t mask = ((size_t)1 << (msb_no - lsb_no + 1)) - 1;
        size_t shift = lsb_no - (8 * (sizeof(uint64_t)) * part1);
        ret = (data[part1] >> shift) & mask;
    }
    else
    {
        size_t mask1 = ((size_t)1 << (msb_no - part1 * 8 * sizeof(__uint64_t))) - 1;
        ret |= (data[part1] & mask1);
        size_t bitsize2 = 64 - (lsb_no - part2 * 8 * sizeof(__uint64_t));
        size_t mask2 = ((size_t)1 << bitsize2) - 1;
        ret = ret << bitsize2;
        ret |= (data[part2] >> (lsb_no - part2 * 8 * sizeof(__uint64_t))) & mask2;
    }
    return ret;
}
// get a unsigned int16 from bit msb_no to lsb_no
__device__ __uint16_t Scalar::get_bits_as_uint16(size_t msb_no, size_t lsb_no) const
{
    if(msb_no > 255) 
        msb_no = 254;
    assert(msb_no - lsb_no < 16);
    size_t part1 = msb_no / (8 * sizeof(__uint64_t));
    size_t part2 = lsb_no / (8 * sizeof(__uint64_t));
    __uint16_t ret = 0;
    if (part1 == part2)
    {
        size_t mask = ((size_t)1 << (msb_no - lsb_no + 1)) - 1;
        size_t shift = lsb_no - (8 * (sizeof(uint64_t)) * part1);
        ret = (data[part1] >> shift) & mask;
    }
    else
    {
        size_t mask1 = ((size_t)1 << (msb_no - part1 * 8 * sizeof(__uint64_t))) - 1;
        ret |= (data[part1] & mask1);
        size_t bitsize2 = 64 - (lsb_no - part2 * 8 * sizeof(__uint64_t));
        size_t mask2 = ((size_t)1 << bitsize2) - 1;
        ret = ret << bitsize2;
        ret |= (data[part2] >> (lsb_no - part2 * 8 * sizeof(__uint64_t))) & mask2;
    }
    return ret;
}

// the bit number of the first set (1) bit. LSB is bit no 1 and MSB is bit no 255 (scalars can be at most 255 bits long)
__device__ size_t Scalar::most_significant_set_bit_no()
{
    for (int i = 254; i >= 0; i--)
        if (test_bit(i))
            return i + 1;
    return 0;
}

// get a random Scalar
__device__ Scalar Scalar::random()
{
    for (size_t i = 0; i < LIMBS; i++)
    {
        this->data[i] = 0x1746567387465673;
    }
    return *this;
}
__host__ __device__ void Scalar::print()
{
    for (int i = LIMBS - 1; i >= 0; i--)
        printf("%016lx ", data[i]);
    printf("\n");
}
#endif