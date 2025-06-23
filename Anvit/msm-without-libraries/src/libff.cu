#ifndef __LIBFF_CU
#define __LIBFF_CU


#include <thread>
#include "./../include/libff.cuh"


libff_compute::libff_compute(size_t size)
{
    this->numbers = size;
    libff::bls12_381_pp::init_public_params();
    this->points = new std::vector<libff::bls12_381_G1>(size);
    this->scalars = new std::vector<libff::bls12_381_Fr>(size);

}

void libff_compute::generate_points()
{
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            libff::bls12_381_G1 q = libff::bls12_381_G1::random_element();
            q.to_affine_coordinates();
            (*this->points)[i] = q;
        }
    };

    size_t chunk_size = (this->numbers + num_threads - 1) / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, numbers);
        threads[t] = std::thread(worker, start, end);
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}
void libff_compute::generate_scalars()
{
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            libff::bls12_381_Fr q = libff::bls12_381_Fr::random_element();
            (*this->scalars)[i] = q;
        }
    };

    size_t chunk_size = (numbers + num_threads - 1) / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, numbers);
        threads[t] = std::thread(worker, start, end);
    }

    for (auto& thread : threads)
        if (thread.joinable())
            thread.join();

}
void libff_compute::compute_msm()
{
    result = libff::multi_exp<libff::bls12_381_G1, libff::bls12_381_Fr, libff::multi_exp_method_BDLO12>(
        points->begin(), points->end(),
        scalars->begin(), scalars->end(),
        1
    );
}
void libff_compute::msm()
{
    this->generate_points();
    this->generate_scalars();
    this->compute_msm();
}

#endif