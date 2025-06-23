#!/usr/bin/env sage -python
from sage.all import *
import argparse

# Curve moduli
PALLAS_MODULUS = 28948022309329048855892746252171976963363056481941560715954676764349967630337
VESTA_MODULUS  = 28948022309329048855892746252171976963363056481941647379679742748393362948097

def get_field(curve):
    if curve == "pallas":
        return FiniteField(PALLAS_MODULUS), PALLAS_MODULUS
    elif curve == "vesta":
        return FiniteField(VESTA_MODULUS), VESTA_MODULUS
    else:
        raise ValueError(f"Unknown curve: {curve}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a specific field parameter for a given curve.")
    parser.add_argument("curve", type=str, help="Curve name (pallas or vesta)")
    parser.add_argument("param", type=str, help="Parameter name (generate, add, multi, double)")
    parser.add_argument("num_point", type=int, help="Number of point to generate. If the `param` is not generate pass any value")
    parser.add_argument("point_a", type=int, help="Given, generator (-1,2). `point_a` * generator will be the argument point. For add, multi this will be argument 1. For double this point will be doubled")
    parser.add_argument("point_b", type=int, help="The second argument for add, multi")
    args = parser.parse_args()

    try:
        F, MODULUS = get_field(args.curve.lower())
        curve = EllipticCurve(F, [0,5])
        one = curve(-1,2)
        if args.param.lower() == "generate":
            m = []
            for i in range(args.num_point):
                m.append(one * (i+1))
            with open("point_output_generate.txt", "w") as f:
                for i in range(args.num_point):
                    f.write(f"{int(m[i][0]):X} {int(m[i][1]):X} {int(m[i][2]):X} ")

        elif args.param.lower() == "add":
            pa = one * args.point_a
            pb = one * args.point_b
            res = pa + pb
            with open("point_output_add.txt", "w") as f:
                f.write(f"{int(res[0]):X} {int(res[1]):X} {int(res[2]):X} ")
        elif args.param.lower() == "multi":
            pa = one * args.point_a
            pb = args.point_b
            res = pa * pb
            with open("point_output_multi.txt", "w") as f:
                f.write(f"{int(res[0]):X} {int(res[1]):X} {int(res[2]):X} ")
        elif args.param.lower() == "double":
            pa = one * args.point_a
            res = 2* pa
            with open("point_output_dbl.txt", "w") as f:
                f.write(f"{int(res[0]):X} {int(res[1]):X} {int(res[2]):X} ")
        else :
            print("Invalid option provided in test file.\n")
    except ValueError as e:
        print(f"Error: {e}")
