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

def get_value(param, F, MODULUS):
    if param == "modulus":
        return MODULUS
    elif param == "inv":
        inv = Integer(MODULUS).inverse_mod(2**64)
        return 2**64 - inv
    elif param == "r":
        return 2**256 % MODULUS
    elif param == "r2":
        r = 2**256 % MODULUS
        return (r * r) % MODULUS
    elif param == "r3":
        r = 2**256 % MODULUS
        return (r * r * r) % MODULUS
    elif param == "generator":
        return int(F.multiplicative_generator())
    elif param == "root_of_unity":
        order = 2**32
        for g in F:
            if g != 0 and g.multiplicative_order() == order:
                return int(g)
        raise ValueError("No primitive root of unity found")
    else:
        raise ValueError(f"Unknown parameter: {param}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a specific field parameter for a given curve.")
    parser.add_argument("param", type=str, help="Parameter name (modulus, inv, r, r2, r3, generator, root_of_unity)")
    parser.add_argument("curve", type=str, help="Curve name (pallas or vesta)")
    args = parser.parse_args()

    try:
        F, MODULUS = get_field(args.curve.lower())
        value = get_value(args.param.lower(), F, MODULUS)
        with open("sage_constant_output.txt", "w") as f:
            f.write(f"{value:X}")
    except ValueError as e:
        print(f"Error: {e}")
