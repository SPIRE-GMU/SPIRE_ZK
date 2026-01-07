#ifndef __LIBFF_CONVERTER_CUH
#define __LIBFF_CONVERTER_CUH

#include <cuda_runtime.h>
#include <cassert>
#include <thread>
#include <libff/algebra/curves/bls12_381/bls12_381_pp.hpp>
#include "./../include/G1Point.cuh"
#include "./../include/FieldG1.cuh"
#include "./../include/Scalar.cuh"

// Create a G1Point from given uint64_t x,y,z values and then compare it with the G1Point
__global__ void checkEqual(G1Point *p, uint64_t *a, uint64_t *b, uint64_t *c, bool *result)
{
    FieldG1 x(a);
    FieldG1 y(b);
    FieldG1 z(c);
    G1Point p2(x, y, z);
    *result = (*p == p2);
}
// Check if the G1Point is equal to the libff point
// libff point's x,y,z values are taken into array and then copied to the G1Point
// Then the G1Point is compared with base G1Point
bool is_equal_to_libff(G1Point *a, const libff::bls12_381_G1 &b)
{
    uint64_t *x, *y, *z;
    
    cudaMalloc((void **)&x, sizeof(uint64_t) * 6);
    cudaMalloc((void **)&y, sizeof(uint64_t) * 6);
    cudaMalloc((void **)&z, sizeof(uint64_t) * 6);

    
    cudaMemcpy(x, b.X.as_bigint().data, sizeof(uint64_t) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(y, b.Y.as_bigint().data, sizeof(uint64_t) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(z, b.Z.as_bigint().data, sizeof(uint64_t) * 6, cudaMemcpyHostToDevice);

    bool *result;
    cudaMalloc(&result, sizeof(bool));
    checkEqual<<<1,1>>>(a, x, y, z, result);
    cudaDeviceSynchronize();
    bool ret;
    cudaMemcpy(&ret, result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    return ret;
}
// Copy the libff point's x,y,z values to the G1Point (Since it is direct memory copy, montgomery values are copied)
void G1Point_from_Libff(G1Point *a, libff::bls12_381_G1 *b)
{
    cudaMemcpy(&a->X.data, b->X.mont_repr.data, sizeof(uint64_t) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(&a->Y.data, b->Y.mont_repr.data, sizeof(uint64_t) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(&a->Z.data, b->Z.mont_repr.data, sizeof(uint64_t) * 6, cudaMemcpyHostToDevice);
}




G1Point* Get_G1Point_from_Libff(size_t n)
{
    std::vector<G1Point> host_points(n);
    std::vector<libff::bls12_381_G1> libff_points(n);
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            libff::bls12_381_G1 q = libff::bls12_381_G1::random_element();
            libff_points[i] = q;
            G1Point_from_Libff(&host_points[i], &q);  
        }
    };

    size_t chunk_size = (n + num_threads - 1) / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        threads[t] = std::thread(worker, start, end);
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }

    G1Point* device_points = nullptr;
    if (cudaMalloc(&device_points, sizeof(G1Point) * n) != cudaSuccess) {
        std::cerr << "Error allocating memory for G1Point\n";
        exit(1);
    }

    cudaMemcpy(device_points, host_points.data(), sizeof(G1Point) * n, cudaMemcpyHostToDevice);
    return device_points;
}

void Scalar_from_Libff(Scalar *a, libff::bls12_381_Fr *b)
{
    cudaMemcpy(a->data, b->as_bigint().data, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
}

Scalar* Get_Scalar_from_Libff(size_t n)
{
    std::vector<Scalar> host_scalars(n);
    std::vector<libff::bls12_381_Fr> libff_scalars(n);
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            libff::bls12_381_Fr q = libff::bls12_381_Fr::random_element();
            libff_scalars[i] = q;
            Scalar_from_Libff(&host_scalars[i], &q);
        }
    };

    size_t chunk_size = (n + num_threads - 1) / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        threads[t] = std::thread(worker, start, end);
    }

    for (auto& thread : threads)
        if (thread.joinable())
            thread.join();

    Scalar* device_scalars = nullptr;
    if (cudaMalloc(&device_scalars, sizeof(Scalar) * n) != cudaSuccess) {
        std::cerr << "Error allocating memory for Scalar\n";
        exit(1);
    }

    cudaMemcpy(device_scalars, host_scalars.data(), sizeof(Scalar) * n, cudaMemcpyHostToDevice);
    return device_scalars;
}
libff::bls12_381_G1 MSM_libff_cpu_reference(
    const std::vector<libff::bls12_381_G1>& points,
    const std::vector<libff::bls12_381_Fr>& scalars)
{
    assert(points.size() == scalars.size());

    libff::bls12_381_G1 acc = libff::bls12_381_G1::zero();
    for (size_t i = 0; i < points.size(); ++i) {
        acc = acc + (scalars[i] * points[i] );
    }
    return acc;
}

#endif