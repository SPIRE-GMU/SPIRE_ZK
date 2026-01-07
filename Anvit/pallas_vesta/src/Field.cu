#ifndef __FIELD_CU
#define __FIELD_CU

#include "./../include/Field.cuh"

#include "./../utils/field-helper.cuh"

// Constructor without any argument
__device__ Field::Field()
{
    this->data[0] = 0;
    this->data[1] = 0;
    this->data[2] = 0;
    this->data[3] = 0;
}

__device__ Field::Field(const u_int64_t *uint64_le, size_t len)
{
    copy_limbs(data, uint64_le, len);
    encode_montgomery();
}
__device__ Field::Field(const Field &other)
{
    copy_limbs(data, other.data, 4);
}
__device__ Field::Field(uint64_t val) 
{
    this->data[0] = val;
    this->data[1] = 0;
    this->data[2] = 0;
    this->data[3] = 0;
    encode_montgomery();
}

// Relational Operators
__device__ bool Field::operator==(const Field &other)
{
    return equal(data, other.data, 4);
}
__device__ bool Field::operator==(const Field &other) const
{
    return equal(data, other.data, 4);
}
__device__ bool Field::operator!=(const Field &other)
{
    return !(operator==(other));
}
__device__ bool Field::operator!=(const Field &other) const
{
    return !(operator==(other));
}

__device__ __forceinline__ bool is_greater_than_or_equal(const u_int64_t *a, const u_int64_t *b)
{
    for (int i = 4 - 1; i >= 0; i--)
    {
        if (a[i] > b[i])
            return true;
        if (a[i] < b[i])
            return false;
    }
    return true;
}
__device__ bool Field::operator>=(const Field &other)
{
    return is_greater_than_or_equal(data, other.data);
}
__device__ bool Field::operator>=(const Field &other) const
{
    return is_greater_than_or_equal(data, other.data);
}

__device__ __forceinline__ bool is_less_than_or_equal(const u_int64_t *a, const u_int64_t *b)
{
    for (int i = 4 - 1; i >= 0; i--)
    {
        if (a[i] < b[i])
            return true;
        if (a[i] > b[i])
            return false;
    }
    return true;
}

__device__ bool Field::operator<=(const Field &other)
{
    return is_less_than_or_equal(data, other.data);
}
__device__ bool Field::operator<=(const Field &other) const
{
    return is_less_than_or_equal(data, other.data);
}

// Arithmatic operators
__device__ Field &Field::operator+=(const Field &other)
{
    add_limbs(this->data, this->data, other.data, 4);
    conditional_subtract(this->data, pallas::MODULUS, 4);
    return *this;
}
__device__ Field &Field::operator-=(const Field &other)
{
    __uint64_t res[4];
    bool borrow = sub_limbs(res, this->data, other.data, 4);
    if (borrow)
    {
        add_limbs(this->data, pallas::MODULUS, this->data, 4);
        sub_limbs(this->data, this->data, other.data, 4);
    }
    else
    {
        copy_limbs(this->data, res, 4);
    }
    conditional_subtract(this->data, pallas::MODULUS, 4);
    return *this;
}
__device__ Field &Field::operator*=(const Field &other)
{
    __uint64_t res[2 * 4];
    mont_mul(res, this->data, other.data, pallas::MODULUS, pallas::INV, 4);
    conditional_subtract(res, pallas::MODULUS, 4);
    copy_limbs(data, res, 4);
    return *this;
}

__device__ Field &Field::operator=(const Field &other)
{
    copy_limbs(data, other.data, 4);
    return *this;
}
__device__ Field Field::operator+(const Field &other)
{
    Field result(*this);
    result+=other;
    return result;
}
__device__ Field Field::operator+(const Field &other) const
{
    Field result(*this);
    result += other;
    return result;
}
__device__ Field Field::operator-(const Field &other)
{
    Field result(*this);
    result -= other;
    return result;
}
__device__ Field Field::operator-(const Field &other) const
{
    Field result(*this);
    result -= other;
    return result;
}
__device__ Field Field::operator*(const Field &other)
{
    Field result(*this);
    result *= other;
    return result;
}
__device__ Field Field::operator*(const Field &other) const
{
    Field result(*this);
    result *= other;
    return result;
}
__device__ Field Field::operator*(const Scalar &other) 
{
    Field o(other.data);
    Field result(*this);
    result *= o;
    return result;
}
// Negation operator
__device__ Field Field::operator-()
{
    if (this->is_zero())
        return *this;

    Field result;
    sub_limbs(result.data, pallas::MODULUS, this->data, 4);
    return result;
}
__device__ Field Field::operator-() const
{
    if (this->is_zero())
        return *this;

    Field result;
    sub_limbs(result.data, pallas::MODULUS, this->data, 4);
    return result;
}

// double the field element
__device__ Field Field::dbl()
{
    Field result;
    for (size_t i = 3; i >= 1; i--)
        result.data[i] = (this->data[i] << 1) | (this->data[i - 1] >> (64 - 1));

    result.data[0] = this->data[0] << 1;
    conditional_subtract(result.data, pallas::MODULUS, 4);
    return result;
}
// square of the field element
__device__ Field Field::squared()
{
    return (*this) * (*this);
}
__device__ Field Field::squared() const
{
    return (*this) * (*this);
}
// clear the set Field values
__device__ void Field::clear()
{
    for (int i = 0; i < 4; i++)
        data[i] = 0;
}
// Check if equal to zero
__device__ bool Field::is_zero()
{
    return is_zero_limbs(data, 4);
}
__device__ bool Field::is_zero() const
{
    return is_zero_limbs(data, 4);
}

// go to montgomery representation
__device__ inline void Field::encode_montgomery()
{
    mont_encode(this->data, this->data, pallas::R2, pallas::MODULUS, pallas::INV, 4);
}
// get out of montgomery representation
__device__ inline void Field::decode_montgomery()
{
    mont_decode(this->data, this->data, pallas::MODULUS, pallas::INV, 4);
}
// find the inverse of the field element
__device__ Field Field::inverse()
{
    Field base(*this);

    Field result = one(); // Initialize result to 1

    __uint64_t exponent[4];
    copy_limbs(exponent, pallas::MODULUS, 4);
    exponent[0] -= 2;

    // Perform exponentiation using square-and-multiply
    #pragma unroll 1
    for (int i = 3; i >= 0; --i) {
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

__device__ Field Field::one()
{
    Field result;
    copy_limbs(result.data, pallas::R, 4);
    return result;
}


__device__ Field Field::zero()
{
    Field x;
    return x;
}

__device__ void Field::print()
{
    for (int i = 4 - 1; i >= 0; i--)
        printf("%016lx ", data[i]);
    printf("\n");
}


__device__ Field Field::as_scalar() 
{
    this->decode_montgomery();
    return *this;
}
#endif