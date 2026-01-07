# README

## Compilation Instructions

To compile the test file, first, setup the docker environment to use the CUDA 11.5 toolkit.Then, use the following command:

```sh
nvcc -arch=sm_86 -rdc=true --expt-extended-lambda ./test-double-addition.cu libgmp.a  ./libff-cuda/curves/bls12_381/bls12_381_pp_host.cu  ./libff-cuda/curves/bls12_381/bls12_381_init_host.cu   ./libff-cuda/curves/bls12_381/bls12_381_g1_host.cu   ./libff-cuda/curves/bls12_381/bls12_381_g2_host.cu     ./libstl-cuda/memory.cu    ./libff-cuda/common/utils.cu    ./libff-cuda/common/multiplication-tensor.cu    ./libff-cuda/mini-mp-cuda/mini-mp-cuda.cu  -o tda
```

## Requirements
- **NVIDIA GPU** with Compute Capability **8.6** or update the compiling command
- **CUDA Toolkit** installed
- **GMP Library** (libgmp.a) for multiple-precision arithmetic

## Explanation of Compilation Flags
- `-arch=sm_86`: Specifies the target GPU architecture (for Ampere GPUs like RTX 30 series).
- `-rdc=true`: Enables separate compilation for device code.
- `--expt-extended-lambda`: Allows using extended lambda expressions in device code.
- `-o tda`: Outputs the compiled binary as `tda`.

## Running the Executable
After successful compilation, run the program using:
```sh
./tda 
```

## Reference File
The `test_file_reference_sage.ipynb` file contains the reference values to verify the result. It is a sage notebook file. To run this file, your system must have sage installed. To use notebook (jupyter needs to be installed in the system) with sage use the following command:
```sh
sage -n jupyter
```


The test file (`test-double-addition.cu`) contains the method `test_multiplication` that can be modified to test other cases. 