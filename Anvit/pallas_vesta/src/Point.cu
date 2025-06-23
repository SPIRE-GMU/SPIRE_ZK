#ifndef __POINT_CU
#define __POINT_CU

#include "./../include/Point.cuh"
#include "./../include/Scalar.cuh"

__device__ Point::Point()
{
    this->X = this->X.zero();
    this->Y = this->Y.one();
    this->Z = this->Z.zero();
}
// initialize the points with x,y,z field values
__device__ Point::Point(const Field &x, const Field &y, const Field &z)
{
    this->X = x;
    this->Y = y;
    this->Z = z;
}
__device__ Point::Point(const Point &other)
{
    this->X = other.X;
    this->Y = other.Y;
    this->Z = other.Z;
}
__device__ Point &Point::operator=(const Point &other)
{
    this->X = other.X;
    this->Y = other.Y;
    this->Z = other.Z;
    return *this;
}
// add is equal to addition in jacobian
__device__ Point Point::operator+(const Point &other)
{
    if (this->is_zero())
        return other;
    Point copy_other(other);
    if (copy_other.is_zero())
        return *this;

    /* https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
      Z1Z1 = Z1^2
      Z2Z2 = Z2^2
      U1 = X1*Z2Z2
      U2 = X2*Z1Z1
      S1 = Y1*Z2*Z2Z2
      S2 = Y2*Z1*Z1Z1
      H = U2-U1
      I = (2*H)^2
      J = H*I
      r = 2*(S2-S1)
      V = U1*I
      X3 = r^2-J-2*V
      Y3 = r*(V-X3)-2*S1*J
      Z3 = ((Z1+Z2)2-Z1Z1-Z2Z2)*H
    */

    Field Z1Z1 = this->Z.squared();
    Field Z2Z2 = copy_other.Z.squared();

    Field U1 = this->X * Z2Z2;
    Field U2 = copy_other.X * Z1Z1;

    Field S1 = this->Y * other.Z * Z2Z2; // S1 = Y1 * Z2 * Z2Z2
    Field S2 = this->Z * other.Y * Z1Z1; // S2 = Y2 * Z1 * Z1Z1

    if (U1 == U2 && S1 == S2)
        return this->dbl();

    Field H = U2 - U1;           // H = U2 - U1
    Field I = H.dbl().squared(); // I = (2 * H)^2
    Field J = H * I;             // J = H * I
    Field r = (S2 - S1).dbl();   // r = 2 * (S2 - S1)
    Field V = U1 * I;            // V = U1 * I

    Field X3 = r.squared() - J - V.dbl();                         // X3 = r^2 - J - 2 * V
    Field Y3 = r * (V - X3) - (S1 * J).dbl();                     // Y3 = r * (V-X3)-2 * S1 * J
    Field Z3 = ((this->Z + other.Z).squared() - Z1Z1 - Z2Z2) * H; // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2) * H

    return Point(X3, Y3, Z3);
}

// equivalent to -1*P
__device__ Point Point::operator-()
{
    return Point(this->X, -(this->Y), this->Z);
}
__device__ Point Point::operator-() const
{
    Field y(this->Y);
    return Point(this->X, -y, this->Z);
}

// equivalent to Return = this + (-other)
__device__ Point Point::operator-(const Point &other)
{
    return (*this) + (-(other));
}

__device__ bool Point::operator==(const Point &other)
{
    Point copy(other);
    if (this->is_zero())
        return copy.is_zero();

    if (copy.is_zero())
        return false;

    Field Z1_squared = this->Z.squared();
    Field Z2_squared = copy.Z.squared();

    if ((this->X * Z2_squared) != (copy.X * Z1_squared))
        return false;

    Field Z1_cubed = this->Z * Z1_squared;
    Field Z2_cubed = copy.Z * Z2_squared;

    return !((this->Y * Z2_cubed) != (copy.Y * Z1_cubed));
}
__device__ bool Point::operator!=(const Point &other)
{
    return !(operator==(other));
}
// equivalent to Point*32bit scalar
__device__ Point Point::operator*(Scalar sc)
{
    Point result(this->zero());
    Point base(*this);
    Scalar scalar(sc);
    if (this->is_zero() || scalar.is_zero())
        return this->zero();

    size_t max_bit = scalar.most_significant_set_bit_no();

    for (long i = max_bit - 1; i >= 0; --i)
    {
        result = result.dbl();
        if (scalar.test_bit(i))
        {
            result = result + base;
        }
    }
    return result;
}

__device__ Point Point::operator*(const unsigned long scalar32)
{
    Scalar scalar(scalar32);
    Point p(*this);
    return p * scalar;
}

__device__ Point Point::operator*(const unsigned long scalar32) const
{
    Scalar scalar(scalar32);
    Point p(*this);
    return p * scalar;
}

// check if equal to infinity
__device__ bool Point::is_zero()
{
    return this->Z.is_zero();
}
// Get a instance of infinity
__device__ Point Point::zero()
{
    Point ret;
    return ret;
}
// Get a instance of One, for pasta curves the one refers to (1,2)
__device__ Point Point::one()
{
    Field x, y, z;
    x.data[0] = 1;
    y.data[0] = 2;
    z.data[0] = 1;
    x.encode_montgomery();
    y.encode_montgomery();
    z.encode_montgomery();
    return Point(-x, y, z);
}
// Get a random point
__device__ Point Point::random()
{
    Scalar t;
    t = t.random();
    return (this->one() * t);
}

// Double of the point
__device__ Point Point::dbl()
{
    if (this->is_zero())
        return *this;

    /* https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
      A = X1^2
      B = Y1^2
      C = B^2
      D = 2*((X1+B)^2-A-C)
      E = 3*A
      F = E^2
      X3 = F-2*D
      Y3 = E*(D-X3)-8*C
      Z3 = 2*Y1*Z1
    */
    Field A = this->X.squared();                       // A = X1^2
    Field B = this->Y.squared();                       // B = Y1^2
    Field C = B.squared();                             // C = B^2
    Field D = ((this->X + B).squared() - A - C).dbl(); // D = 2 * ((X1 + B)^2 - A - C)
    Field E = (A.dbl() + A);                           // E = 3 * A
    Field F = E.squared();                             // F = E^2
    Field X3 = F - D.dbl();                            // X3 = F - 2 D
    Field Y3 = E * (D - X3) - C.dbl().dbl().dbl();     // Y3 = E * (D - X3) - 8 * C

    Field Z3 = (this->Y * this->Z).dbl(); // Z3 = 2 * Y1 * Z1

    return Point(X3, Y3, Z3);
}
// Add in jacobian
__device__ Point Point::add(const Point &other)
{
    return (*this) + other;
}
// Mixed add method where the other point is in affine form
__device__ Point Point::mixed_add(const Point &other)
{
    Point copy(other);
    if (this->is_zero())
        return other;

    if (copy.is_zero())
        return *this;

    Field Z1Z1 = this->Z.squared();

    Field U2 = copy.X * Z1Z1;

    Field S2 = this->Z * copy.Y * Z1Z1; // S2 = Y2 * Z1 * Z1Z1

    if (this->X == U2 && this->Y == S2)
        return this->dbl();

    Field H = U2 - this->X;         // H = U2-X1
    Field HH = H.squared();         // HH = H^2
    Field I = HH.dbl().dbl();       // I = 4*HH
    Field J = H * I;                // J = H*I
    Field r = (S2 - this->Y).dbl(); // r = 2*(S2-Y1)
    Field V = this->X * I;          // V = X1*I

    Field X3 = r.squared() - J - V.dbl();           // X3 = r^2-J-2*V
    Field Y3 = r * (V - X3) - (this->Y * J).dbl();  // Y3 = r*(V-X3)-2*Y1*J
    Field Z3 = (this->Z + H).squared() - Z1Z1 - HH; // Z3 = (Z1+H)^2-Z1Z1-HH

    return Point(X3, Y3, Z3);
}

// check if a valid point on BLS12_381
__device__ bool Point::is_well_formed()
{
    if (this->is_zero())
        return true;

    Field X2 = this->X.squared();
    Field Y2 = this->Y.squared();
    Field Z2 = this->Z.squared();

    Field X3 = this->X * X2;
    Field Z3 = this->Z * Z2;
    Field Z6 = Z3.squared();

    Scalar coef(5);

    return (Y2 == X3 + Z6 * coef);
}
// change the coordinate value of this point to affine coordinate
__device__ void Point::to_affine()
{
    Field t;
    if (this->is_zero())
    {
        this->X = t.zero();
        this->Y = t.one();
        this->Z = t.zero();
    }
    else
    {
        Field Z_inv = Z.inverse();
        Field Z2_inv = Z_inv.squared();
        Field Z3_inv = Z2_inv * Z_inv;
        this->X = this->X * Z2_inv;
        this->Y = this->Y * Z3_inv;
        this->Z = t.one();
    }
}

__device__ void Point::print()
{
    printf("\nX=\n");
    this->X.decode_montgomery();
    this->X.print();
    this->X.encode_montgomery();
    printf("\nY=\n");
    this->Y.decode_montgomery();
    this->Y.print();
    this->Y.encode_montgomery();
    printf("\nZ=\n");
    this->Z.decode_montgomery();
    this->Z.print();
    this->Z.encode_montgomery();
}

#endif