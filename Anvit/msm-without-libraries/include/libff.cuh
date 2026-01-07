#ifndef __LIBFF_CUH
#define __LIBFF_CUH

#include <cuda_runtime.h>
#include <cassert>
#include <thread>
#include <libff/algebra/curves/bls12_381/bls12_381_pp.hpp>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <vector>

class libff_compute
{
public:
    std::vector<libff::bls12_381_G1> *points;
    std::vector<libff::bls12_381_Fr> *scalars;

    libff::bls12_381_G1 result;

    size_t numbers;


    libff_compute(size_t size);

    void generate_points();
    void generate_scalars();
    void compute_msm();
    void msm();
};

#include "./../src/libff.cu"
#endif