// test_FieldG1.cu
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

#include "./../include/FieldG1.cuh"
#include "./../include/Scalar.cuh"

#include "./../utils/bls12_381_constants.cu"

__device__ uint64_t x1[6] = {
    0xacc555c722aea803ULL,
    0x71442fbd0ad62391ULL,
    0x19360c2b6d0929bbULL,
    0x728b874b86b63ecfULL,
    0x7570f7dc170e554dULL,
    0x38e4078bb8da22cULL
};
__device__ uint64_t x2[6] = {
    0xacc555c722aea803ULL,
    0x71442fbd0ad62391ULL,
    0x19360c2b6d0929bbULL,
    0x728b874b86b63ecfULL,
    0x7570f7dc170e554dULL,
    0x18e4078bb8da22cULL
};

__global__ void fieldg1_tests() {
    FieldG1 a(x1);
    FieldG1 b(x2);

    FieldG1 c = a + a;
    assert( c == a.dbl() && "Two additions should be equal to double" );
    FieldG1 d = a + b;
    assert( d == b + a && "Addition should be commutative" );
    assert( d == a + b && "Addition should be associative" );
    assert(a>= b && "a should be greater than b");
    assert(a!= b && "a should not be equal to b");
    assert(a == a && "a should be equal to a");
    assert(b <= a && "b should be less than or equal to a");

    assert( c-a == d-b && "a+a-a should be equal to a+b-b");

    Scalar s(2);
    assert( c == a*s && "c should equal to 2*a");
    c +=a;
    Scalar three(3);
    assert( c == a*three && "c should equal to 3*a");

    c = a;
    assert(c == a && "c should be equal to a"); 
    //test initialize from scalar
    FieldG1 scalar(0x123456789abcdefULL); 
    // value is stored in montgomery representation, decode to get the actual value
    scalar.decode_montgomery();
    assert(scalar.data[0] == 0x123456789abcdefULL && "Scalar should be equal to 0x123456789abcdef");

    c.clear();
    assert(c.is_zero() && "c should be equal to zero");
    c = c.one();
    FieldG1 rmodp(bls12_381::r_mod_p);
    rmodp.decode_montgomery();
    assert(c == rmodp && "c should be equal to r mod p");
    c.decode_montgomery();
    assert(c.data[0]==1 && "Decoded c should be equal to 1");
    
}

int main() {
    fieldg1_tests<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
