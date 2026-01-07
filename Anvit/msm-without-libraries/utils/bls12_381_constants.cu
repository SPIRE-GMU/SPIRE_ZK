#ifndef BLS12_381_CONSTANTS_CU
#define BLS12_381_CONSTANTS_CU

#include <stdint.h>
#include <cuda_runtime.h>

#define G1_LIMBS 6 // Fp elements are 381 bits → 6×64
#define FR_LIMBS 4 // Fr elements are 255 bits → 4×64

namespace bls12_381
{

    // -----------------------------------------------------------------------------
    // Base field  p = 0x1a0111ea397fe69a4b1ba7b6434bacd7
    //                  64774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
    // limbs in little-endian:
    //   p[0] = 0xb9feffffffffaaab
    //   p[1] = 0x1eabfffeb153ffff
    //   p[2] = 0x6730d2a0f6b0f624
    //   p[3] = 0x64774b84f38512bf
    //   p[4] = 0x4b1ba7b6434bacd7
    //   p[5] = 0x1a0111ea397fe69a                                  
    static __host__ __device__ const uint64_t modulus[G1_LIMBS] = {
        0xb9feffffffffaaabULL,
        0x1eabfffeb153ffffULL,
        0x6730d2a0f6b0f624ULL,
        0x64774b84f38512bfULL,
        0x4b1ba7b6434bacd7ULL,
        0x1a0111ea397fe69aULL
    };

    // mont_inv = –p⁻¹ mod 2^64 = 0x89f3fffcfffcfffd                     
    static __host__ __device__ const uint64_t mont_inv = 0x89f3fffcfffcfffdULL;

    // R   = 2^384 mod p  (Montgomery factor)                            
    static __host__ __device__ const uint64_t r_mod_p[G1_LIMBS] = {
        0x760900000002fffdULL,
        0xebf4000bc40c0002ULL,
        0x5f48985753c758baULL,
        0x77ce585370525745ULL,
        0x5c071a97a256ec6dULL,
        0x15f65ec3fa80e493ULL
    };

    // R²  = 2^(384·2) mod p                                            
    static __host__ __device__ const uint64_t r2_mod_p[G1_LIMBS] = {
        0xf4df1f341c341746ULL,
        0x0a76e6a609d104f1ULL,
        0x8de5476c4c95b6d5ULL,
        0x67eb88a9939d83c0ULL,
        0x9a793e85b519952dULL,
        0x11988fe592cae3aaULL
    };

    // -----------------------------------------------------------------------------
    // Scalar field  r = 0x73eda753299d7d483339d80809a1d805
    //                     53bda402fffe5bfeffffffff00000001
    // limbs in little-endian:
    //   r[0] = 0xffffffff00000001
    //   r[1] = 0x53bda402fffe5bfe
    //   r[2] = 0x3339d80809a1d805
    //   r[3] = 0x73eda753299d7d48                        
    static __host__ __device__ const uint64_t fr_modulus[FR_LIMBS] = {
        0xffffffff00000001ULL,
        0x53bda402fffe5bfeULL,
        0x3339d80809a1d805ULL,
        0x73eda753299d7d48ULL
    };

    // fr_mont_inv = –r⁻¹ mod 2^64 = 0xfffffffeffffffff                    
    static __host__ __device__ const uint64_t fr_mont_inv = 0xfffffffeffffffffULL;

    // R   = 2^256 mod r                                                
    static __host__ __device__ const uint64_t fr_r_mod[FR_LIMBS] = {
        0x00000001fffffffeULL,
        0x5884b7fa00034802ULL,
        0x998c4fefecbc4ff5ULL,
        0x1824b159acc5056fULL
    };

    // R²  = 2^(256·2) mod r                                           
    static __host__ __device__ const uint64_t fr_r2_mod[FR_LIMBS] = {
        0xc999e990f3f29c6dULL,
        0x2b6cedcb87925c23ULL,
        0x05d314967254398fULL,
        0x0748d9d99f59ff11ULL
    };
    static __host__ __device__ const uint64_t g1_x[6] = {
        0x1a0111ea397fe69aULL,
        0x4b1ba7b6434bacd7ULL,
        0x64774b84f38512bfULL,
        0x6730d2a0f6b0f624ULL,
        0x1eabfffeb153ffffULL,
        0xb9feffffffffaaabULL
    };
    static __host__ __device__ const uint64_t g1_y[6] = {
        0x50e8f6a8ac165b41ULL,
        0xa5c3c2ad0f09cf03ULL,
        0x5d75ea0c036c9e0cULL,
        0xf6741e1e4b33f5f4ULL,
        0xa7faeec3b57c6ab0ULL,
        0x0000002f0b89d97eULL
    };
    static __host__ __device__ const uint64_t coefficient_b = 4;
    

} 

#endif 