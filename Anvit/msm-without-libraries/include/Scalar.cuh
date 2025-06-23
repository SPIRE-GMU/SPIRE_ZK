#ifndef __SCALAR_CUH
#define __SCALAR_CUH
class alignas(16) Scalar
{
public:
    __uint64_t data[4]; // scalars are 255 bits = 32 bytes
    __device__ Scalar();
    __device__ Scalar(const unsigned long long x);
    __device__ Scalar(const __uint8_t *bytes, size_t length = 32);
    __device__ Scalar(const Scalar &other);

    // assignment from another scalar
    __device__ Scalar &operator=(const Scalar &other);

    // relational operators
    __device__ bool operator==(const Scalar &other);
    __device__ bool operator!=(const Scalar &other);
    __device__ bool operator>=(const Scalar &other);
    __device__ bool operator<=(const Scalar &other);

    // addition and subtraction operators
    __device__ Scalar &operator+=(const Scalar &other);
    __device__ Scalar &operator-=(const Scalar &other);

    __device__ Scalar operator+(const Scalar &other);
    __device__ Scalar operator-(const Scalar &other);

    // check if the scalar is zero
    __device__ bool is_zero();
    // check if the bit_no is set (1) or not (0)
    __device__ bool test_bit(size_t bit_no) const;
    // Set the bit_no as 1
    __device__ void set_bit(size_t bit_no);

    // get a unsigned int32 from bit msb_no to lsb_no
    __device__ __uint32_t get_bits_as_uint32(size_t msb_no, size_t lsb_no) const;
    // get a unsigned int16 from bit msb_no to lsb_no
    __device__ __uint16_t get_bits_as_uint16(size_t msb_no, size_t lsb_no) const;

    // the bit number of the first set (1) bit. LSB is bit no 1 and MSB is bit no 255 (scalars can be at most 255 bits long)
    __device__ size_t most_significant_set_bit_no();

    // get a random Scalar
    __device__ Scalar random();
    __host__ __device__ void print();
};

#include "./../src/Scalar.cu"

#endif