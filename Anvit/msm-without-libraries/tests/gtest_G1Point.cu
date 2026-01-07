// utils/gen_g1_test_vectors.cpp
#include <libff/algebra/curves/bls12_381/bls12_381_pp.hpp>
#include <fstream>

int main() {
    libff::bls12_381_pp::init_public_params();

    std::ofstream out("g1_vectors.txt");
    for (int i = 0; i < 10; ++i) {
        auto scalar = libff::bls12_381_Fr::random_element();
        auto g1 = scalar * libff::bls12_381_G1::one();

        out << scalar.as_bigint() << std::endl;
        out << g1.X.as_bigint() << std::endl;
        out << g1.Y.as_bigint() << std::endl;
        out << g1.Z.as_bigint() << std::endl;
    }
}
