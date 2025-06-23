#ifndef __G1POINT_CUH
#define __G1POINT_CUH

#include "Scalar.cuh"
#include "FieldG1.cuh"
class alignas(16) G1Point
{
public:
    FieldG1 X, Y, Z;
    // constructor without any argument, set point to infinity
    __device__ G1Point();
    // initialize the points with x,y,z field values
    __device__ G1Point(FieldG1 x, FieldG1 y, FieldG1 z);

    // add is equal to addition in jacobian
    __device__ G1Point operator+(const G1Point &other);

    // equivalent to -1*P
    __device__ G1Point operator-();
    __device__ G1Point operator-() const;

    // equivalent to Return = this + (-other)
    __device__ G1Point operator-(const G1Point &other);

    __device__ bool operator==(const G1Point &other);
    __device__ bool operator!=(const G1Point &other);

    // equivalent to Point*32bit scalar
    __device__ G1Point operator*(const unsigned long scalar32);

    __device__ G1Point operator*(const unsigned long scalar32) const;
    __device__ G1Point operator*(const Scalar scalar);


    // check if equal to infinity
    __device__ bool is_zero();
    // Get a instance of infinity
    __device__ G1Point zero();
    // Get a instance of One
    __device__ G1Point one();
    // Get a random point
    __device__ G1Point random();

    // Double of the point
    __device__ G1Point dbl();
    // Add in jacobian
    __device__ G1Point add(const G1Point &other);
    // Mixed add method where the other point is in affine form
    __device__ G1Point mixed_add(const G1Point &other);

    // check if a valid point on BLS12_381
    __device__ bool is_well_formed();
    // change the coordinate value of this point to affine coordinate
    __device__ void to_affine();

    __device__ void print();
};

#include "./../src/G1Point.cu"

#endif