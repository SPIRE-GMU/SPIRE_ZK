#ifndef __FIELD_UNIT_TEST
#define __FIELD_UNIT_TEST

#include <gtest/gtest.h>
#include "test_header.cuh"
#define TESTS 10

class FieldTests : public ::testing::Test
{
public:
    FieldTests()
    {
    }
    ~FieldTests() override
    {
    }
    void SetUp() override
    {
    }
    void TearDown() override
    {
    }
};

__global__ void montgomery_test_kernel(Field *a, uint64_t b, bool *result)
{
    *a = Field(b);
    size_t t = a->data[0];
    a->decode_montgomery();
    *result = (a->data[0] == b);
    a->encode_montgomery();
    *result &= (t == a->data[0]);
}

TEST_F(FieldTests, check_montgomery)
{
    Field *a;
    size_t ref;
    bool *res, *d_res;
    res = new bool;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < TESTS; i++)
    {
        ref = rand() + rand() + 1;
        montgomery_test_kernel<<<1, 1>>>(a, ref, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res);
    }
}

__global__ void equal_test_kernel(Field *a, Field *b, bool *res)
{
    *res = *a == *b;
}
__global__ void init_field(Field *a, uint64_t val)
{
    *a = Field(val);
}
TEST_F(FieldTests, check_equal_operator)
{
    Field *a, *b;
    size_t ref, ref1;
    bool *res, *d_res;
    res = new bool;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < TESTS; i++)
    {
        ref = rand() + rand() + 1;
        init_field<<<1, 1>>>(a, ref);
        ref1 = rand() + rand() + 1;
        init_field<<<1, 1>>>(b, ref1);
        CUDA_CHECK(cudaDeviceSynchronize());
        equal_test_kernel<<<1, 1>>>(a, b, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res == (ref == ref1));
        init_field<<<1, 1>>>(b, ref);
        CUDA_CHECK(cudaDeviceSynchronize());

        equal_test_kernel<<<1, 1>>>(a, b, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res);
    }
}
__global__ void unequal_test_kernel(Field *a, Field *b, bool *res)
{
    *res = *a != *b;
}

TEST_F(FieldTests, check_not_equal_operator)
{
    Field *a, *b;
    size_t ref, ref1;
    bool *res, *d_res;
    res = new bool;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < TESTS; i++)
    {
        ref = rand() + rand() + 1;
        ref1 = rand() + rand() + 1;

        while (ref == ref1)
        {
            ref += rand();
            ref1 += rand();
        }
        init_field<<<1, 1>>>(a, ref);
        init_field<<<1, 1>>>(b, ref1);
        CUDA_CHECK(cudaDeviceSynchronize());
        unequal_test_kernel<<<1, 1>>>(a, b, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res);
    }
}
__global__ void greater_test_kernel(Field *a, Field *b, bool *res)
{
    a->decode_montgomery();
    b->decode_montgomery();
    *res = *a >= *b;
}
TEST_F(FieldTests, check_greater_than_equal)
{
    Field *a, *b;
    size_t ref, ref1;
    bool *res, *d_res;
    res = new bool;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < TESTS; i++)
    {
        ref = rand();
        ref1 = rand();

        if (ref < ref1)
        {
            size_t temp = ref;
            ref = ref1;
            ref1 = temp;
        }
        init_field<<<1, 1>>>(a, ref);
        init_field<<<1, 1>>>(b, ref1);
        CUDA_CHECK(cudaDeviceSynchronize());
        greater_test_kernel<<<1, 1>>>(a, b, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res);
    }
}
__global__ void less_than_equal_test_kernel(Field *a, Field *b, bool *res)
{
    a->decode_montgomery();
    b->decode_montgomery();
    *res = *a <= *b;
}
TEST_F(FieldTests, check_less_than_equal)
{
    Field *a, *b;
    size_t ref, ref1;
    bool *res, *d_res;
    res = new bool;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < TESTS; i++)
    {
        ref = rand();
        ref1 = rand();

        if (ref > ref1)
        {
            size_t temp = ref;
            ref = ref1;
            ref1 = temp;
        }
        init_field<<<1, 1>>>(a, ref);
        init_field<<<1, 1>>>(b, ref1);
        CUDA_CHECK(cudaDeviceSynchronize());
        less_than_equal_test_kernel<<<1, 1>>>(a, b, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res);
    }
}

__global__ void addition_test_kernel(Field *a, Field *b, size_t ref, size_t ref2, bool *res)
{
    *a = Field(ref);
    *b = Field(ref2);
    Field c = *a+*b;
    Field d(ref + ref2);
    *res = c == d;
}

TEST_F(FieldTests, check_addition)
{
    Field *a, *b;
    size_t ref, ref1;
    bool *res, *d_res;
    res = new bool;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Field)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < TESTS; i++)
    {
        ref = rand();
        ref1 = rand();

        addition_test_kernel<<<1, 1>>>(a, b, ref, ref1, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res);
    }
}

TEST_F(FieldTests, check_negation)
{

}
TEST_F(FieldTests, check_double)
{
}
TEST_F(FieldTests, check_square)
{
}
TEST_F(FieldTests, check_is_zero)
{
}
TEST_F(FieldTests, check_clear)
{
}

#endif