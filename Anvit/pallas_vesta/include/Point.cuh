#ifndef __POINT_CUH
#define __POINT_CUH

#include "Field.cuh"
class __align__(128) Point
{
public:
    __device__ Field X, Y, Z;
    // constructor without any argument, set point to infinity
    __device__ Point();
    // initialize the points with x,y,z field values
    __device__ Point(const Field &x, const Field &y, const Field &z);

    __device__ Point(const Point &other);

    // assignment operator
    __device__ Point& operator=(const Point &other);

    // add is equal to addition in jacobian
    __device__ Point operator+(const Point &other);

    // equivalent to -1*P
    __device__ Point operator-();
    __device__ Point operator-() const;

    // equivalent to Return = this + (-other)
    __device__ Point operator-(const Point &other);

    __device__ bool operator==(const Point &other);
    __device__ bool operator!=(const Point &other);

    // equivalent to Point*32bit scalar
    __device__ Point operator*(const unsigned long scalar32);

    __device__ Point operator*(const unsigned long scalar32) const;
    __device__ Point operator*(const Scalar scalar);


    // check if equal to infinity
    __device__ bool is_zero();
    // Get a instance of infinity
    __device__ Point zero();
    // Get a instance of One
    __device__ Point one();
    // Get a random point
    __device__ Point random();

    // Double of the point
    __device__ Point dbl();
    // Add in jacobian
    __device__ Point add(const Point &other);
    // Mixed add method where the other point is in affine form
    __device__ Point mixed_add(const Point &other);

    // check if a valid point on BLS12_381
    __device__ bool is_well_formed();
    // change the coordinate value of this point to affine coordinate
    __device__ void to_affine();

    __device__ void print();
};

#include "./../src/Point.cu"

#endif