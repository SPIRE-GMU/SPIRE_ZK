#!/usr/bin/env sage -python
from sage.all import *
import argparse

# method to split sage values to uint64 limbs
def to_uint64_limbs_le(x, num_limbs=4):
    """Convert integer x to a little-endian list of num_limbs uint64_t chunks."""
    limbs = []
    x = int(x)
    for i in range(num_limbs):
        limbs.append(int(x % (1 << 64)))
        x = x >> 64
    return limbs

#Pallas modulus
p = 28948022309329048855892746252171976963363056481941560715954676764349967630337
F = GF(p)
E = EllipticCurve(F, [0, 5])
G = E(-1,2)

#number of point
n =  2**10

#scalars and points to write to file
scalars = []
points = []
#scalars and points to use in sage
s_scalar = []
s_point = []
# generate points and scalars
for _ in range(n):
    s = ZZ.random_element(1, E.order())
    P = s * G
    s_scalar.append(s)
    s_point.append(P)
    scalars.append(to_uint64_limbs_le(s))
    points.append((to_uint64_limbs_le(P[0]), to_uint64_limbs_le(P[1])))

#write as c++ values to files
with open("../constants/msm_sage_values.cuh", "w") as f:
    print("#include<stdint.h>", file=f)
    # Print scalars
    print("const uint64_t sage_scalars[{}][4] = {{".format(n), file=f)
    for s in scalars:
        print("    {{{}}},".format(", ".join(f"0x{s_i:016x}" for s_i in s)), file=f)
    print("};\n", file=f)
    one = [1,0,0,0]
    # Print points
    print(f"const uint64_t sage_points[{n}][3][4] = ", file=f)
    print("{", file=f)
    for i, (x, y) in enumerate(points):
        print("{", file=f)
        # print(f"      // Point {i}", file=f) #for debug purpose to track point index
        print("        {{{}}},".format(", ".join(f"0x{x_i:016x}" for x_i in x)), file=f)
        print("        {{{}}},".format(", ".join(f"0x{y_i:016x}" for y_i in y)), file=f)
        print("        {{{}}}".format(", ".join(f"0x{z_i:016x}" for z_i in one)), file=f)
        print("    }," if i < n - 1 else "    }", file=f)
    print("};", file=f)


    # Compute and print MSM result
    zero = E(0,1,0)
    for s, P in zip(s_scalar, s_point):
        zero = zero + s*P
    
    msm_result = zero
    msm_x = to_uint64_limbs_le(msm_result[0])
    msm_y = to_uint64_limbs_le(msm_result[1])

    # Print sage msm result
    print("const uint64_t sage_msm_result[2][4] = {", file=f)
    print("    {{{}}},".format(", ".join(f"0x{x:016x}" for x in msm_x)), file=f)
    print("    {{{}}}".format(", ".join(f"0x{y:016x}" for y in msm_y)), file=f)
    print("};", file=f)
