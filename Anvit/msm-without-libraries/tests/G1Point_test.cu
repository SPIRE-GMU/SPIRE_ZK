#include <cassert>

#include "./../include/FieldG1.cuh"

#include "./../include/Scalar.cuh"
#include "./../include/G1Point.cuh"

__device__ uint64_t genX[6] = {
    0xacc555c722aea803ULL,
    0x71442fbd0ad62391ULL,
    0x19360c2b6d0929bbULL,
    0x728b874b86b63ecfULL,
    0x7570f7dc170e554dULL,
    0x38e4078bb8da22cULL};

__device__ uint64_t genY[6] = {
    0xd7584bb0fa458df4ULL,
    0xe52ea24532fc73e4ULL,
    0xe1b8819f3a605248ULL,
    0x28efc92ed9753fb9ULL,
    0xf0c34146feb9d46bULL,
    0xbec328b7b998bb4ULL};

__device__ uint64_t genZ[6] = {
    0x1ULL,
    0x0ULL,
    0x0ULL,
    0x0ULL,
    0x0ULL,
    0x0ULL};

__device__ uint64_t gen2X[6] = {
    0x477888446f67e2c1ULL,
    0x11243dfe843cb787ULL,
    0x3009a8d507cc04c8ULL,
    0x096e69ab90d945a5ULL,
    0x5a692d364b5d7a35ULL,
    0x11339e2e2f45de4aULL};

__device__ uint64_t gen2Y[6] = {
    0x067372121475ab62ULL,
    0x1b106d37b9da2024ULL,
    0x37dccd0c1be5f0d2ULL,
    0x876bf083ae314687ULL,
    0x927c839c4a76173fULL,
    0xdbaa2e9b3e5ab45ULL};

__global__ void test_G1Point_basics()
{
    G1Point zero = G1Point().zero();
    G1Point one = G1Point().one();

    assert((zero + zero).is_zero() && "zero + zero should be equal to zero");
    assert((one + zero) == one && "one + zero should be one");
    assert((zero + one) == one && "zero + one should be one");
    assert((one + one) != one && "one + one should not equal to one");

    G1Point dbl = one.dbl();
    assert(dbl == one + one && "one + one should equal to double");
    G1Point triple = one + dbl;
    G1Point three = one * 3;
    assert(three == triple && "one+one+one should equal 3*one");
    G1Point five = one * 5;
    assert(three + dbl == five && "one + one + one + one +one should equal to 5*one");

    assert(one - one == zero && "one - one should equal to zero");
    assert(dbl - one == one && "dbl - one should equal to one");
    assert(five - dbl == three && "five - double should equal three");
    assert(five - three == dbl && "Five - three should equal double");

    FieldG1 x(genX), y(genY), z(genZ);
    G1Point generator(x, y, z);

    FieldG1 x2(gen2X), y2(gen2Y), z2(genZ);
    G1Point generator_times2(x2, y2, z2);

    G1Point gdbl = generator.dbl();
    assert(gdbl == generator_times2 && "2*generator should equal to generator_times2");

    gdbl.to_affine();
    assert(gdbl == generator_times2 && "To affine should still equal to generator_times2");

    G1Point minus_gen(x, -y, z);
    assert(-generator == minus_gen && "Negation of a point should be equal to the -1*y coordinate");
}

__global__ void test_G1Point_scalar_mul()
{
    G1Point G = G1Point().one();
    Scalar two(2);
    Scalar three(3);
    Scalar five(5);
    Scalar zero;
    G1Point twoG = G * two;
    G1Point threeG = G * (three);
    G1Point fiveG = G * (five);

    assert(twoG + threeG == fiveG);
    G1Point zeroG = G * (zero);
    assert(zeroG.is_zero());
    Scalar one(1);
    G1Point G1 = G * (one);
    assert(G1 == G);
}

void run_G1Point_tests()
{
    test_G1Point_basics<<<1, 1>>>();
    cudaDeviceSynchronize();
    test_G1Point_scalar_mul<<<1, 1>>>();
    cudaDeviceSynchronize();
}
int main()
{
    run_G1Point_tests();
    return 0;
}
