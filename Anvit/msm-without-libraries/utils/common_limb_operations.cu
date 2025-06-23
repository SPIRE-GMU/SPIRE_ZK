#ifndef __COMMON_LIMB_OPERATIONS_CU
#define __COMMON_LIMB_OPERATIONS_CU

#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define LIMBS 4    // BigInt: 256-bit = 4 * 64-bit
#define G1_LIMBS 6 // BigIntMont: 381-bit ~= 6 * 64-bit

__host__ __device__ inline bool is_zero_limbs(const uint64_t *limbs, int n)
{
#pragma unroll
    for (int i = 0; i < n; ++i)
    {
        if (limbs[i] != 0)
            return false;
    }
    return true;
}

__host__ __device__ inline bool equal(const uint64_t *a, const uint64_t *b, int n)
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
#pragma unroll
    for (int i = 0; i < n; ++i)
    {
        dest[i] = src[i];
    }
}

__host__ __device__ inline void add_limbs(uint64_t *res, const uint64_t *a, const uint64_t *b, int n)
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



__host__ __device__ inline void conditional_subtract(uint64_t *res, const uint64_t *modulus, int n)
{
    uint64_t tmp[2 * G1_LIMBS];
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

// __host__ __device__ inline void montgomery_reduce(uint64_t *t, const uint64_t *modulus, uint64_t inv, int n)
// {
//     for (int i = 0; i < n; ++i)
//     {
//         uint64_t m = t[i] * inv;
//         uint64_t carry = 0;
//         for (int j = 0; j < n; ++j)
//         {
//             __uint128_t mul = (__uint128_t)m * modulus[j] + t[i + j] + carry;
//             t[i + j] = (uint64_t)mul;
//             carry = mul >> 64;
//         }
//         t[i + n] += carry;
//     }
//     for (int i = 0; i < n; ++i)
//         t[i] = t[i + n];
// }
// mont_mul: res = a * b * R^(–1) mod modulus
// - a, b: input arrays of length n (little-endian limbs)
// - res: length-n output (little-endian limbs)
// - modulus: length-n modulus
// - inv: –modulus⁻¹ mod 2^64
// - n: number of limbs
// __host__ __device__
// inline void mont_mul(uint64_t* res, const uint64_t* a, const uint64_t* b, const uint64_t* modulus, uint64_t inv, int n)
// {
//     uint64_t t[2 * G1_LIMBS] = {0};  // temporary accumulator

//     // Main Montgomery multiplication loop
//     for (int i = 0; i < n; ++i) {
//         // 1) Multiply a * b[i], add into t
//         uint64_t carry1 = 0;
//         for (int j = 0; j < n; ++j) {
//             __uint128_t prod = (__uint128_t)a[j] * b[i] + t[j] + carry1;
//             t[j] = (uint64_t)prod;
//             carry1 = (uint64_t)(prod >> 64);
//         }
//         t[n] += carry1;  // accumulate high limb properly

//         // 2) Montgomery reduction
//         uint64_t m = t[0] * inv;  // m ≡ -t[0] * modulus⁻¹ mod 2^64
//         uint64_t carry2 = 0;
//         for (int j = 0; j < n; ++j) {
//             __uint128_t sum = (__uint128_t)m * modulus[j] + t[j] + carry2;
//             t[j] = (uint64_t)sum;
//             carry2 = (uint64_t)(sum >> 64);
//         }
//         t[n] += carry2;

//         // 3) Shift: drop t[0], move t[1..n] → t[0..n-1]
//         for (int j = 0; j < n; ++j) {
//             t[j] = t[j + 1];
//         }
//         t[n] = 0;  // reset high limb for next iteration
//     }

//     // 4) Output result
//     for (int i = 0; i < n; ++i) {
//         res[i] = t[i];
//     }

//     // 5) Ensure result < modulus
//     uint64_t borrow = 0;
//     for (int i = 0; i < n; ++i) {
//         __uint128_t diff = (__uint128_t)res[i] - modulus[i] - borrow;
//         borrow = (diff >> 127) & 1;  // if diff < 0 then borrow = 1
//     }

//     if (borrow == 0) {
//         uint64_t temp[G1_LIMBS];
//         uint64_t carry = 0;
//         for (int i = 0; i < n; ++i) {
//             __uint128_t diff = (__uint128_t)res[i] - modulus[i] - carry;
//             temp[i] = (uint64_t)diff;
//             carry = (diff >> 127) & 1;
//         }
//         for (int i = 0; i < n; ++i) res[i] = temp[i];
//     }
// }

__device__ inline ulong mac_with_carry(ulong a, ulong b, ulong c, ulong *d)
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
__device__ inline ulong add_with_carry(ulong a, ulong *b)
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

    // Step 4: Final conditional subtraction (canonicalization)
    // uint64_t tmp[64];  // assumes n ≤ 64
    // uint64_t borrow = 0;

    // for (int i = 0; i < n; ++i) {
    //     __uint128_t diff = (__uint128_t)res[i] - modulus[i] - borrow;
    //     tmp[i] = (uint64_t)diff;
    //     borrow = ((diff >> 127) & 1);  // Check for underflow
    // }

    // if (borrow == 0) {
    //     for (int i = 0; i < n; ++i) {
    //         res[i] = tmp[i];  // write canonicalized result
    //     }
    // }
    conditional_subtract(res, modulus, n);
}

__device__ inline void mont_encode(uint64_t *result, const uint64_t *x, const uint64_t *r2_mod_p, const uint64_t *modulus, uint64_t inv, int n)
{
    uint64_t temp[2 * G1_LIMBS];
    mont_mul(temp, x, r2_mod_p, modulus, inv, n);
    copy_limbs(result, temp, n);
    conditional_subtract(result, modulus, n);
}

__device__ inline void mont_decode(uint64_t *result, const uint64_t *x, const uint64_t *modulus, uint64_t inv, int n)
{
    uint64_t one[G1_LIMBS] = {0};
    one[0] = 1;

    uint64_t temp[2 * G1_LIMBS];
    mont_mul(temp, x, one, modulus, inv, n);
    copy_limbs(result, temp, n);
    conditional_subtract(result, modulus, n);
}

__device__ inline void to_bytes(uint8_t *out, const uint64_t *limbs, int n)
{
#pragma unroll
    for (int i = 0; i < n; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            out[i * 8 + j] = (limbs[i] >> (8 * j)) & 0xff;
        }
    }
}

__host__ __device__ inline void from_u64(uint64_t *limbs, uint64_t val, int n)
{
    for (int i = 0; i < n; ++i)
        limbs[i] = 0;
    limbs[0] = val;
}

__host__ __device__ inline void from_bytes(uint64_t *limbs, const uint8_t *bytes, int n)
{
    for (int i = 0; i < n; ++i)
    {
        limbs[i] = 0;
        for (int j = 0; j < 8; ++j)
        {
            limbs[i] |= ((uint64_t)bytes[i * 8 + j]) << (8 * j);
        }
    }
}

#endif