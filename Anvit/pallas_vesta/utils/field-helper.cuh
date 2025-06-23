#ifndef __FIELD_HELPER_CUH
#define __FIELD_HELPER_CUH

#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define LIMBS 4

__host__ __device__ __forceinline__ bool is_zero_limbs(const uint64_t *limbs, int n)
{
#pragma unroll
    for (int i = 0; i < n; ++i)
    {
        if (limbs[i] != 0)
            return false;
    }
    return true;
}

__host__ __device__ __forceinline__ bool equal(const uint64_t *a, const uint64_t *b, int n)
{
#pragma unroll
    for (int i = 0; i < n; ++i)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

__host__ __device__ inline void copy_limbs(uint64_t *dest, const uint64_t *src, int n)
{
// #pragma unroll
    for (int i = 0; i < n; ++i)
    {
        dest[i] = src[i];
    }
}

__host__ __device__ __forceinline__ void add_limbs(uint64_t *res, const uint64_t *a, const uint64_t *b, int n)
{
    uint64_t carry = 0;
    for (int i = 0; i < n; ++i)
    {
        uint64_t sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && sum == a[i]);
        res[i] = sum;
    }
}

__host__ __device__ inline bool sub_limbs(uint64_t *res, const uint64_t *a, const uint64_t *b, int n)
{
    uint64_t borrow = 0;
    for (int i = 0; i < n; ++i)
    {
        // Full 128-bit subtraction: a[i] - b[i] - borrow
        __uint128_t diff = (__uint128_t)a[i] - b[i] - borrow;
        res[i] = (uint64_t)diff;

        // Borrow occurs if top bit (MSB) is set after subtraction
        borrow = (diff >> 127) & 1;
    }
    return borrow;
}



__host__ __device__ __forceinline__ void conditional_subtract(uint64_t *res, const uint64_t *modulus, int n)
{
    uint64_t tmp[2 * LIMBS];
    bool borrow = sub_limbs(tmp, res, modulus, n);
    if (!borrow)
    {
        copy_limbs(res, tmp, n);
    }
}

__host__ __device__ inline void shr1(uint64_t *res, const uint64_t *a, int n)
{
    uint64_t carry = 0;
    for (int i = n - 1; i >= 0; --i)
    {
        uint64_t new_carry = a[i] << 63;
        res[i] = (a[i] >> 1) | carry;
        carry = new_carry;
    }
}

__device__ __forceinline__ ulong mac_with_carry(ulong a, ulong b, ulong c, ulong *d)
{
    ulong lo, hi;
    asm(
        "mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64   %1, %2, %3,  0;\r\n"
        "add.cc.u64    %0, %0, %5;    \r\n"
        "addc.u64      %1, %1,  0;    \r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));

    *d = hi;
    return lo;
}

// Returns a + b, puts the carry in d
__device__ __forceinline__ ulong add_with_carry(ulong a, ulong *b)
{
    ulong lo, hi;
    asm(
        "add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64   %1,  0,  0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));

    *b = hi;
    return lo;
}
__device__ inline void mont_mul(
    uint64_t* res,
    const uint64_t* a,
    const uint64_t* b,
    const uint64_t* modulus,
    uint64_t inv,
    int n
) {
    // Temporary buffer t[0..n+1]
    uint64_t t[66] = {0};  // Assumes n ≤ 64

    for (int i = 0; i < n; ++i) {
        uint64_t carry = 0;

        // Step 1: Multiply a * b[i] and add to t
        for (int j = 0; j < n; ++j) {
            t[j] = mac_with_carry(a[j], b[i], t[j], &carry);
        }
        t[n] = add_with_carry(t[n], &carry);
        t[n + 1] = carry;

        // Step 2: Montgomery reduction
        uint64_t m = t[0] * inv;
        carry = 0;
        (void)mac_with_carry(m, modulus[0], t[0], &carry);
        for (int j = 1; j < n; ++j) {
            t[j - 1] = mac_with_carry(m, modulus[j], t[j], &carry);
        }

        t[n - 1] = add_with_carry(t[n], &carry);

        // Carefully handle overflow into t[n]
        __uint128_t sum = (__uint128_t)t[n + 1] + carry;
        t[n] = (uint64_t)sum;
        t[n + 1] = (uint64_t)(sum >> 64);  // safe overflow capture
    }

    // Step 3: Write result (t[0..n-1])
    for (int i = 0; i < n; ++i) {
        res[i] = t[i];
    }

    conditional_subtract(res, modulus, n);
}

__device__ inline void mont_encode(uint64_t *result, const uint64_t *x, const uint64_t *r2_mod_p, const uint64_t *modulus, uint64_t inv, int n)
{
    uint64_t temp[2 * LIMBS];
    mont_mul(temp, x, r2_mod_p, modulus, inv, n);
    copy_limbs(result, temp, n);
    conditional_subtract(result, modulus, n);
}

__device__ inline void mont_decode(uint64_t *result, const uint64_t *x, const uint64_t *modulus, uint64_t inv, int n)
{
    uint64_t one[LIMBS] = {0};
    one[0] = 1;

    uint64_t temp[2 * LIMBS];
    mont_mul(temp, x, one, modulus, inv, n);
    copy_limbs(result, temp, n);
    conditional_subtract(result, modulus, n);
}

#endif