#!/usr/bin/env sage -python
from sage.all import *
from multiprocessing import Pool, cpu_count
import os

# method to split sage values to uint64 limbs
def to_uint64_limbs_le(x, num_limbs=4):
    limbs = []
    x = int(x)
    for i in range(num_limbs):
        limbs.append(int(x % (1 << 64)))
        x = x >> 64
    return limbs

# Globals for multiprocessing
p = 28948022309329048855892746252171976963363056481941560715954676764349967630337
F = GF(p)
E = EllipticCurve(F, [0, 5])
G = E(-1,2)

def generate_chunk(chunk_size):
    """Generate a chunk of scalar and point data."""
    scalars_chunk = []
    points_chunk = []
    s_scalar_chunk = []
    s_point_chunk = []
    
    for _ in range(chunk_size):
        s = ZZ.random_element(1, E.order())
        P = s * G
        s_scalar_chunk.append(s)
        s_point_chunk.append(P)
        scalars_chunk.append(to_uint64_limbs_le(s))
        points_chunk.append((to_uint64_limbs_le(P[0]), to_uint64_limbs_le(P[1])))

    return scalars_chunk, points_chunk, s_scalar_chunk, s_point_chunk

def parallel_generate(n, num_procs=None):
    """Use multiprocessing to generate scalars and points in parallel."""
    if num_procs is None:
        num_procs = cpu_count()

    chunk_size = n // num_procs
    args = [chunk_size] * num_procs
    # Fix rounding issues
    args[-1] += n % num_procs

    with Pool(processes=num_procs) as pool:
        results = pool.map(generate_chunk, args)

    # Unpack the results
    scalars, points, s_scalars, s_points = [], [], [], []
    for sc, pt, ssc, spt in results:
        scalars.extend(sc)
        points.extend(pt)
        s_scalars.extend(ssc)
        s_points.extend(spt)

    return scalars, points, s_scalars, s_points
def mult_pair(pair):
    s, P = pair
    return s * P

def reduce_points(points):
    # Binary reduction of list of points
    while len(points) > 1:
        next_level = []
        for i in range(0, len(points)-1, 2):
            next_level.append(points[i] + points[i+1])
        if len(points) % 2 == 1:
            next_level.append(points[-1])  # carry forward last point
        points = next_level
    return points[0]

def parallel_msm(s_scalar, s_point, num_procs=None):
    if num_procs is None:
        num_procs = cpu_count()
    with Pool(processes=num_procs) as pool:
        # Step 1: Parallel s * P
        scalar_mults = pool.map(mult_pair, zip(s_scalar, s_point))

    # Step 2: Parallel reduction (optional - here done serially for clarity)
    return reduce_points(scalar_mults)
# number of points
n = 2**20

# generate in parallel
scalars, points, s_scalar, s_point = parallel_generate(n)

msm_result = parallel_msm(s_scalar, s_point)
# write to file
with open("../constants/msm_sage_values_2.cuh", "w") as f:
    print("#include<stdint.h>", file=f)
    
    print("const uint64_t sage_scalars[{}][4] = {{".format(n), file=f)
    for s in scalars:
        print("    {{{}}},".format(", ".join(f"0x{s_i:016x}" for s_i in s)), file=f)
    print("};\n", file=f)

    one = [1, 0, 0, 0]
    print(f"const uint64_t sage_points[{n}][3][4] = ", file=f)
    print("{", file=f)
    for i, (x, y) in enumerate(points):
        print("{", file=f)
        print("        {{{}}},".format(", ".join(f"0x{x_i:016x}" for x_i in x)), file=f)
        print("        {{{}}},".format(", ".join(f"0x{y_i:016x}" for y_i in y)), file=f)
        print("        {{{}}}".format(", ".join(f"0x{z_i:016x}" for z_i in one)), file=f)
        print("    }," if i < n - 1 else "    }", file=f)
    print("};", file=f)

    # # Compute MSM result
    # zero = E(0,1,0)
    # for s, P in zip(s_scalar, s_point):
    #     zero = zero + s*P

    # msm_result = zero
    msm_x = to_uint64_limbs_le(msm_result[0])
    msm_y = to_uint64_limbs_le(msm_result[1])

    print("const uint64_t sage_msm_result[2][4] = {", file=f)
    print("    {{{}}},".format(", ".join(f"0x{x:016x}" for x in msm_x)), file=f)
    print("    {{{}}}".format(", ".join(f"0x{y:016x}" for y in msm_y)), file=f)
    print("};", file=f)
