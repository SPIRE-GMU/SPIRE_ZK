#include "./../include/Scalar.cuh"
#include<cuda_runtime.h>

__global__ void test()
{
    Scalar a;
    assert(a.is_zero() && "a should be zero");
    Scalar b(0x1234567890abcdefULL);
    assert(b.data[0]==0x1234567890abcdefULL && "b should be equal to 0x1234567890abcdefULL");
    Scalar c(b);
    assert(b == c && "b and c should be equal");
    a = b;
    assert(a == c && "a and c should be equal");
    assert(b.most_significant_set_bit_no() == 61 && "set msb should equal 61");
    b.set_bit(250);
    assert(b.most_significant_set_bit_no() == 251 && "set msb should equal 251");
    assert(b.test_bit(250)==1 && "bit index 250 should contain 1");

    Scalar d = a+c;
    d.set_bit(250);
    assert(d== b+a && "d should equal to b+a");

    assert(d-a == b && "d-a should equal to b");
    assert(d >= a && "d should be greater than or equal to a");
    assert(d!=a && "d should not be equal to a");
    assert(a <= b && "a should be less than or equal to b");
    assert(b.get_bits_as_uint16(15,0)== 0xcdef &&"last 16 bits of b should equal to 0xcdef");
    assert(b.get_bits_as_uint32(31,0) == 0x90abcdef && "last 32 bits of b should be equal to 0x90abcdef");
    b.set_bit(64);
    b.set_bit(66);
    assert(b.get_bits_as_uint16(67,52)== 0x5123 && "bit 67 to 52 should equal to 0x5123");
    assert(b.get_bits_as_uint32(67,36)== 0x51234567 && "bit 67 to 52 should equal to 0x51234567");
}
int main()
{
    test<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}