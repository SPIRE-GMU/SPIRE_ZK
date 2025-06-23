#ifndef __G1POINT_CU
#define __G1POINT_CU
#include "./../include/G1Point.cuh"

// constructor without any argument, set point to infinity
__device__ G1Point::G1Point()
{
    this->X = this->X.zero();
    this->Y = this->Y.one();
    this->Z = this->Z.zero();
}
// initialize the points with x,y,z field values
__device__ G1Point::G1Point(FieldG1 x, FieldG1 y, FieldG1 z)
{
    this->X = x;
    this->Y = y;
    this->Z = z;
}

// add is equal to addition in jacobian
__device__ G1Point G1Point::operator+(const G1Point &other)
{
    if (this->is_zero())
        return other;
    G1Point copy_other(other);
    if (copy_other.is_zero())
        return *this;

    FieldG1 Z1Z1 = this->Z.squared();
    FieldG1 Z2Z2 = copy_other.Z.squared();

    FieldG1 U1 = this->X * Z2Z2;
    FieldG1 U2 = copy_other.X * Z1Z1;

    FieldG1 S1 = this->Y * other.Z * Z2Z2; // S1 = Y1 * Z2 * Z2Z2
    FieldG1 S2 = this->Z * other.Y * Z1Z1; // S2 = Y2 * Z1 * Z1Z1

    if (U1 == U2 && S1 == S2)
        return this->dbl();

    FieldG1 H = U2 - U1;           // H = U2 - U1
    FieldG1 I = H.dbl().squared(); // I = (2 * H)^2
    FieldG1 J = H * I;             // J = H * I
    FieldG1 r = (S2 - S1).dbl();   // r = 2 * (S2 - S1)
    FieldG1 V = U1 * I;            // V = U1 * I

    FieldG1 X3 = r.squared() - J - V.dbl();                         // X3 = r^2 - J - 2 * V
    FieldG1 Y3 = r * (V - X3) - (S1 * J).dbl();                     // Y3 = r * (V-X3)-2 * S1 * J
    FieldG1 Z3 = ((this->Z + other.Z).squared() - Z1Z1 - Z2Z2) * H; // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2) * H

    return G1Point(X3, Y3, Z3);
}
// equivalent to -1*P
__device__ G1Point G1Point::operator-()
{
    return G1Point(this->X, -(this->Y), this->Z);
}
__device__ G1Point G1Point::operator-() const
{
    FieldG1 y(this->Y);
    return G1Point(this->X, -y, this->Z);
}
// equivalent to Return = this + (-other)
__device__ G1Point G1Point::operator-(const G1Point &other)
{
    G1Point copy(other);
    return (*this) + (-(copy));
}

__device__ bool G1Point::operator==(const G1Point &other)
{
    G1Point copy(other);
    if (this->is_zero())
        return copy.is_zero();

    if (copy.is_zero())
        return false;

    FieldG1 Z1_squared = this->Z.squared();
    FieldG1 Z2_squared = copy.Z.squared();

    if ((this->X * Z2_squared) != (copy.X * Z1_squared))
        return false;

    FieldG1 Z1_cubed = this->Z * Z1_squared;
    FieldG1 Z2_cubed = copy.Z * Z2_squared;

    return !((this->Y * Z2_cubed) != (copy.Y * Z1_cubed));
}
__device__ bool G1Point::operator!=(const G1Point &other)
{
    return !(operator==(other));
}

__device__ G1Point G1Point::operator*(const Scalar sc)
{

    G1Point result(this->zero());
    G1Point base(*this);
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

// equivalent to Point*32bit scalar
__device__ G1Point G1Point::operator*(const unsigned long scalar32)
{
    Scalar scalar(scalar32);
    return (*this) * scalar;
}

__device__ G1Point G1Point::operator*(const unsigned long scalar32) const
{
    Scalar scalar(scalar32);
    G1Point ret(*this);
    return ret * scalar;
}
// check if equal to infinity
__device__ bool G1Point::is_zero()
{
    return this->Z.is_zero();
}
// Get a instance of infinity
__device__ G1Point G1Point::zero()
{
    G1Point ret;
    return ret;
}
// Get a instance of One
__device__ G1Point G1Point::one()
{
    FieldG1 x, y, z;
    copy_limbs(x.data, bls12_381::g1_x, 6);
    copy_limbs(y.data, bls12_381::g1_y, 6);
    z = z.one();
    x.encode_montgomery();
    y.encode_montgomery();
    G1Point ret(x, y, z);
    return ret;
}
// Get a random point
__device__ G1Point G1Point::random()
{
    Scalar t;
    t = t.random();
    return (this->one() * t);
}

// Double of the point
__device__ G1Point G1Point::dbl()
{
    if (this->is_zero())
        return *this;
    FieldG1 A = this->X.squared(); // A = X1^2
    FieldG1 B = this->Y.squared(); // B = Y1^2
    FieldG1 C = B.squared();       // C = B^2

    FieldG1 D = ((this->X + B).squared() - A - C).dbl(); // D = 2 * ((X1 + B)^2 - A - C)

    FieldG1 E = A + A.dbl(); // E = 3 * A
    FieldG1 F = E.squared(); // F = E^2

    FieldG1 X3 = F - D.dbl();                        // X3 = F - 2 D
    FieldG1 Y3 = E * (D - X3) - C.dbl().dbl().dbl(); // Y3 = E * (D - X3) - 8 * C
    FieldG1 Z3 = (this->Y * this->Z).dbl();          // Z3 = 2 * Y1 * Z1

    return G1Point(X3, Y3, Z3);
}
// Add in jacobian
__device__ G1Point G1Point::add(const G1Point &other)
{
    return (*this) + other;
}
// Mixed add method where the other point is in affine form
__device__ G1Point G1Point::mixed_add(const G1Point &other)
{
    G1Point copy(other);
    if (this->is_zero())
        return other;

    if (copy.is_zero())
        return *this;

    FieldG1 Z1Z1 = this->Z.squared();

    FieldG1 U2 = copy.X * Z1Z1;

    FieldG1 S2 = this->Z * copy.Y * Z1Z1; // S2 = Y2 * Z1 * Z1Z1

    if (this->X == U2 && this->Y == S2)
        return this->dbl();

    FieldG1 H = U2 - this->X;         // H = U2-X1
    FieldG1 HH = H.squared();         // HH = H^2
    FieldG1 I = HH.dbl().dbl();       // I = 4*HH
    FieldG1 J = H * I;                // J = H*I
    FieldG1 r = (S2 - this->Y).dbl(); // r = 2*(S2-Y1)
    FieldG1 V = this->X * I;          // V = X1*I

    FieldG1 X3 = r.squared() - J - V.dbl();           // X3 = r^2-J-2*V
    FieldG1 Y3 = r * (V - X3) - (this->Y * J).dbl();  // Y3 = r*(V-X3)-2*Y1*J
    FieldG1 Z3 = (this->Z + H).squared() - Z1Z1 - HH; // Z3 = (Z1+H)^2-Z1Z1-HH

    return G1Point(X3, Y3, Z3);
}

// check if a valid point on BLS12_381
__device__ bool G1Point::is_well_formed()
{
    if (this->is_zero())
        return true;

    FieldG1 X2 = this->X.squared();
    FieldG1 Y2 = this->Y.squared();
    FieldG1 Z2 = this->Z.squared();

    FieldG1 X3 = this->X * X2;
    FieldG1 Z3 = this->Z * Z2;
    FieldG1 Z6 = Z3.squared();

    Scalar coef(bls12_381::coefficient_b);

    return (Y2 == X3 + Z6 * coef);
}
// change the coordinate value of this point to affine coordinate
__device__ void G1Point::to_affine()
{
    FieldG1 t;
    if (this->is_zero())
    {
        this->X = t.zero();
        this->Y = t.one();
        this->Z = t.zero();
    }
    else
    {
        FieldG1 Z_inv = Z.inverse();
        FieldG1 Z2_inv = Z_inv.squared();
        FieldG1 Z3_inv = Z2_inv * Z_inv;
        this->X = this->X * Z2_inv;
        this->Y = this->Y * Z3_inv;
        this->Z = t.one();
    }
}
__device__ void G1Point::print()
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