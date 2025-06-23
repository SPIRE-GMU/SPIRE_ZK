#include <ostream>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <libff/algebra/curves/bls12_381/bls12_381_pp.hpp>
#include "./../include/G1Point.cuh"
#include "./../utils/common_limb_operations.cu"
#include "./../utils/libff_converter.cuh"

// Adjust this number to test more or fewer points
#define NUM 10

libff::bls12_381_G1 *points = nullptr;
G1Point *g1_points = nullptr;

__global__ void point_add_kernel(G1Point *a, G1Point *b, G1Point *out) {
    *out = *a + *b;
}
__global__ void point_sub_kernel(G1Point *a, G1Point *b, G1Point *out) {
    *out = *a - *b;
}
__global__ void point_double_kernel(G1Point *a, G1Point *out) {
    *out = a->dbl();
}
__global__ void point_eq_kernel(G1Point *a, G1Point *b, bool *result) {
    *result = (*a == *b);
}
__global__ void point_neq_kernel(G1Point *a, G1Point *b, bool *result) {
    *result = (*a != *b);
}
__global__ void point_zero_kernel(G1Point *p, bool *result) {
    *result = p->is_zero();
}
__global__ void point_to_affine_kernel(G1Point *p, G1Point *affine_out) {
    *affine_out = *p;
    affine_out->to_affine();
}
__global__ void is_Z_zero(G1Point *p, bool *result)
{

    // We need to decode the montgomery representation of Z to check if it is 1
    // Then again put it back to montgomery representation
    p->to_affine();
    p->Z.decode_montgomery();
    *result = p->Z.data[0] == 1;
    for(int i = 1; i < 6; i++)
    {
        *result &= (p->Z.data[i] == 0);
    }
    p->Z.encode_montgomery();
}

class G1Point_Test : public ::testing::Test
{
public:
    G1Point_Test()
    {
    }
    ~G1Point_Test() override
    {
    }

    void SetUp() override
    {
        // Code here will be called just before the test executes.

        // Initialize the libff library
        libff::bls12_381_pp::init_public_params();
        // Allocate memory for points
        points = new libff::bls12_381_G1[NUM];
        
        for (int i = 0; i < NUM; i++)
        {
            // Generate random points
            points[i] = libff::bls12_381_G1::random_element();
        }
        
        // Allocate memory for G1Point on the device
        cudaMalloc((void **)&g1_points, sizeof(G1Point) * NUM);
        cudaDeviceSynchronize();

        // Copy the libff points to the device G1Point
        // Since it is direct assignment to memory, we copy the montgomery representation
        for (int i = 0; i < NUM; i++)
        {
            G1Point_from_Libff(&g1_points[i], &points[i]);
        }
    }
    void TearDown() override
    {
        // Code here will be called just after the test executes.
        free(points);
        cudaFree(g1_points);
        points = nullptr;
    }
};

TEST_F(G1Point_Test, Initialization)
{
    for(int i = 0; i < NUM; i++)
    {
        // Check if the G1Point on device matches the libff point
        ASSERT_TRUE(is_equal_to_libff(&g1_points[i], points[i])) << "Initialization mismatch!";
    }
}
TEST_F(G1Point_Test, PointAddition)
{
    G1Point *d_result;
    cudaMalloc(&d_result, sizeof(G1Point));

    // For n points, we will test n-1 additions
    // i.e. p1 + p2, p2 + p3, ..., p(n-1) + pn
    for(int i=0;i<NUM - 1;i++)
    {
        // Perform point addition on the device
        point_add_kernel<<<1, 1>>>(&g1_points[i], &g1_points[i+1], d_result);
        cudaDeviceSynchronize();
        // Check if the result matches the expected value
        // The expected value is the addition of the two libff points
        // Note: The addition in libff is done using the operator +, which is overloaded inside libff to perform point addition
        auto expected = points[i] + points[i+1];
        ASSERT_TRUE(is_equal_to_libff(d_result, expected)) << "Addition mismatch!";    
    }
    cudaFree(d_result);
}

TEST_F(G1Point_Test, PointSubtraction)
{
    G1Point *d_result;
    cudaMalloc(&d_result, sizeof(G1Point));

    // For n points, we will test n-1 subtractions
    // i.e. p1 - p2, p2 - p3, ..., p(n-1) - pn
    // Note: The subtraction in libff is done using the operator -, which is overloaded inside libff to perform point subtraction
    for(int i=0;i<NUM - 1;i++)
    {
        point_sub_kernel<<<1, 1>>>(&g1_points[i], &g1_points[i+1], d_result);
        cudaDeviceSynchronize();
    
        auto expected = points[i] - points[i+1];
        ASSERT_TRUE(is_equal_to_libff(d_result, expected)) << "Subtraction mismatch!";
    }
    cudaFree(d_result);
}

TEST_F(G1Point_Test, PointDouble)
{
    G1Point *d_result;
    cudaMalloc(&d_result, sizeof(G1Point));

    // For n points, we will test n doublings
    // i.e. p1 + p1, p2 + p2, ..., pn + pn
    // The doubling in libff is done using libff::bls12_381_G1::dbl() 
    for(int i=0;i<NUM;i++)
    {
        point_double_kernel<<<1, 1>>>(&g1_points[i], d_result);
        cudaDeviceSynchronize();
    
        auto expected = points[i].dbl();
        ASSERT_TRUE(is_equal_to_libff(d_result, expected)) << "Doubling mismatch!";
    }
    cudaFree(d_result);
}

TEST_F(G1Point_Test, EqualOperator)
{
    bool *d_result;
    cudaMalloc(&d_result, sizeof(bool));

    // For n points, we will test n equality checks
    // i.e. p1 == p1, p2 == p2, ..., pn == pn
    for(int i=0;i<NUM;i++)
    {
        point_eq_kernel<<<1, 1>>>(&g1_points[i], &g1_points[i], d_result);
        cudaDeviceSynchronize();
    
        bool h_result;
        cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
        ASSERT_TRUE(h_result) << "Equality operator failed!";
    }
    cudaFree(d_result);
}

TEST_F(G1Point_Test, NotEqualOperator)
{
    bool *d_result;
    cudaMalloc(&d_result, sizeof(bool));

    // For n points, we will test n-1 inequality checks
    // i.e. p1 != p2, p2 != p3, ..., p(n-1) != pn
    for(int i=0;i<NUM - 1;i++)
    {
        point_neq_kernel<<<1, 1>>>(&g1_points[i], &g1_points[i+1], d_result);
        cudaDeviceSynchronize();
    
        bool h_result;
        cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
        ASSERT_TRUE(h_result) << "Inequality operator failed!";
    }
    
    cudaFree(d_result);
}

TEST_F(G1Point_Test, PointIsZero)
{
    bool *d_result;
    cudaMalloc(&d_result, sizeof(bool));

    // For n points, we will test n zero checks
    // Since all the points are random, we expect all of them to be non-zero
    // i.e. p1.is_zero(), p2.is_zero(), ..., pn.is_zero() all should return false
    for(int i=0;i<NUM;i++)
    {
        point_zero_kernel<<<1, 1>>>(&g1_points[i], d_result);
        cudaDeviceSynchronize();
    
        bool h_result;
        cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
        ASSERT_FALSE(h_result) << "Point incorrectly reported as zero!";
    }

    // Now we will test the zero point
    G1Point *zero_point;
    cudaMalloc(&zero_point, sizeof(G1Point));
    point_zero_kernel<<<1, 1>>>(zero_point, d_result);
    cudaDeviceSynchronize();
    bool h_result;
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(h_result) << "Zero point incorrectly reported as non-zero!";

    cudaFree(d_result);
}

TEST_F(G1Point_Test, PointToAffine)
{
    G1Point *d_result;
    cudaMalloc(&d_result, sizeof(G1Point));
    // For n points, we will test n affine conversions
    // i.e. p1.to_affine(), p2.to_affine(), ..., pn.to_affine()
    for(int i=0; i< NUM; i++)
    {
        point_to_affine_kernel<<<1, 1>>>(g1_points, d_result);
        cudaDeviceSynchronize();
    
        bool *result;
        cudaMalloc(&result, sizeof(bool));
        // Check if the Z coordinate is 1
        // The Z coordinate is 1 if the point is in affine form
        // i.e. p.Z == 1
        is_Z_zero<<<1, 1>>>(d_result, result);
        cudaDeviceSynchronize();
        bool res;
        cudaMemcpy(&res, result, sizeof(bool), cudaMemcpyDeviceToHost);
    
        ASSERT_TRUE(res) << "Affine conversion failed! Z != 1";
    }
    cudaFree(d_result);
}
__global__ void is_well_formed(G1Point *p, bool *result)
{
    // Check if the point is well-formed
    *result = p->is_well_formed();
}
TEST_F(G1Point_Test, PointIsWellFormed)
{
    bool *d_result;
    cudaMalloc(&d_result, sizeof(bool));
    // For n points, we will test n well-formed checks
    // i.e. p1.is_well_formed(), p2.is_well_formed(), ..., pn.is_well_formed()
    // All the points are random, so we expect all of them to be well-formed
    for(int i=0;i<NUM;i++)
    {
        is_well_formed<<<1, 1>>>(&g1_points[i], d_result);
        cudaDeviceSynchronize();
    
        bool h_result;
        cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
        ASSERT_TRUE(h_result) << "Point is not well formed!";
    }
}
