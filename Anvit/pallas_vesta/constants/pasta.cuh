#pragma once

#ifndef PARAMS_CUH
#define PARAMS_CUH

#include <cstdio>


// Curve constants for pallas y^2 = x^3 + 5
namespace pallas
{
    // char *base_modulus = "28948022309329048855892746252171976963363056481941560715954676764349967630337";
    // char *r_modulus = "28948022309329048855892746252171976963363056481941647379679742748393362948097";

    /// Constant representing the modulus
    /// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
    __device__ const u_int64_t MODULUS[] = {
        0x992d30ed00000001,
        0x224698fc094cf91b,
        0x0000000000000000,
        0x4000000000000000,
    };
    /// INV = -(p^{-1} mod 2^64) mod 2^64
    __device__ const u_int64_t INV = 0x992d30ecffffffff;

    /// R = 2^256 mod p
    __device__ const u_int64_t R[] = {
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    };

    /// R^2 = 2^512 mod p
    __device__ const u_int64_t R2[] = {
        0x8c78ecb30000000f,
        0xd7d30dbd8b0de0e7,
        0x7797a99bc3c95d18,
        0x096d41af7b9cb714,
    };

    /// R^3 = 2^768 mod p
    __device__ const u_int64_t R3[] = {
        0xf185a5993a9e10f9,
        0xf6a68f3b6ac5b1d1,
        0xdf8d1014353fd42c,
        0x2ae309222d2d9910,
    };

    /// `GENERATOR = 5 mod p` is a generator of the `p - 1` order multiplicative
    /// subgroup, or in other words a primitive root of the field.
    __device__ const u_int64_t GENERATOR[] = {
        0x0000000000000005,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    };

    __device__ const u_int64_t u32 = 32;

    /// GENERATOR^t where t * 2^s + 1 = p
    /// with t odd. In other words, this
    /// is a 2^s root of unity.
    __device__ const u_int64_t ROOT_OF_UNITY[] = {
        0xbdad6fabd87ea32f,
        0xea322bf2b7bb7584,
        0x362120830561f81a,
        0x2bce74deac30ebda,
    };

    /// GENERATOR^{2^s} where t * 2^s + 1 = p
    /// with t odd. In other words, this
    /// is a t root of unity.
    __device__ const u_int64_t DELTA[] = {
        0x6a6ccd20dd7b9ba2,
        0xf5e4f3f13eee5636,
        0xbd455b7112a5049d,
        0x0a757d0f0006ab6c,
    };

}

// Curve constants for vesta y^2 = x^3 + 5
namespace vesta
{
    // char *r_modulus = "28948022309329048855892746252171976963363056481941560715954676764349967630337";
    // char *base_modulus = "28948022309329048855892746252171976963363056481941647379679742748393362948097";

    /// Constant representing the modulus
    /// q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
    __device__ const u_int64_t MODULUS[] = {
        0x8c46eb2100000001,
        0x224698fc0994a8dd,
        0x0,
        0x4000000000000000,
    };
    /// INV = -(q^{-1} mod 2^64) mod 2^64
    __device__ const u_int64_t INV = 0x8c46eb20ffffffff;

    /// R = 2^256 mod q
    __device__ const u_int64_t R[] = {
        0x5b2b3e9cfffffffd,
        0x992c350be3420567,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    };

    /// R^2 = 2^512 mod q
    __device__ const u_int64_t R2[] = {
        0xfc9678ff0000000f,
        0x67bb433d891a16e3,
        0x7fae231004ccf590,
        0x096d41af7ccfdaa9,
    };

    /// R^3 = 2^768 mod q
    __device__ const u_int64_t R3[] = {
        0x008b421c249dae4c,
        0xe13bda50dba41326,
        0x88fececb8e15cb63,
        0x07dd97a06e6792c8,
    };

    /// `GENERATOR = 5 mod q` is a generator of the `q - 1` order multiplicative
    /// subgroup, or in other words a primitive root of the field.
    __device__ const u_int64_t GENERATOR[] = {
        0x0000000000000005,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    };

    __device__ const u_int64_t S = 32;

    /// GENERATOR^t where t * 2^s + 1 = q
    /// with t odd. In other words, this
    /// is a 2^s root of unity.
    __device__ const u_int64_t ROOT_OF_UNITY[] = {
        0xa70e2c1102b6d05f,
        0x9bb97ea3c106f049,
        0x9e5c4dfd492ae26e,
        0x2de6a9b8746d3f58,
    };

    /// GENERATOR^{2^s} where t * 2^s + 1 = q
    /// with t odd. In other words, this
    /// is a t root of unity.
    __device__ const u_int64_t DELTA[] = {
        0x8494392472d1683c,
        0xe3ac3376541d1140,
        0x06f0a88e7f7949f8,
        0x2237d54423724166,
    };
}
// typedef enum curves {
//     PALLAS,
//     VESTA
// } curve_type;

// template<curve_type T>
// struct Parameters
// {
//     const Scalar modulus;
//     const u_int64_t inv;
//     const Scalar R;
//     const Scalar R2;
//     const Scalar R3;
//     const Scalar Generator;
//     const Scalar Root_of_Unity;
//     const Scalar Delta;
// };

// template <>
// static struct Parameters<PALLAS>
// {
//     const Scalar modulus= Scalar(pallas::MODULUS);
//     const u_int64_t inv = pallas::INV;
//     const Scalar R = Scalar(pallas::R);
//     const Scalar R2 = Scalar(pallas::R2);
//     const Scalar R3 = Scalar(pallas::R3);
//     const Scalar Generator = Scalar(pallas::GENERATOR);
//     const Scalar Root_of_Unity = Scalar(pallas::ROOT_OF_UNITY);
//     const Scalar Delta = Scalar(pallas::DELTA);
// };

// template <>
// struct Parameters<VESTA>
// {
//     const Scalar modulus= Scalar(vesta::MODULUS);
//     const u_int64_t inv = vesta::INV;
//     const Scalar R = Scalar(vesta::R);
//     const Scalar R2 = Scalar(vesta::R2);
//     const Scalar R3 = Scalar(vesta::R3);
//     const Scalar Generator = Scalar(vesta::GENERATOR);
//     const Scalar Root_of_Unity = Scalar(vesta::ROOT_OF_UNITY);
//     const Scalar Delta = Scalar(vesta::DELTA);
// };
#endif