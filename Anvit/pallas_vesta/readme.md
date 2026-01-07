
# PALLAS_VESTA

This project implements CUDA-accelerated Multi-Scalar Multiplication, and different cryptographic operations over the Pallas and Vesta (PASTA) curves, with unit testing using Google Test. 

## Building the Project

Clone the repository and build the project using CMake:

```bash
git clone https://github.com/SPIRE-GMU/SPIRE_ZK.git
cd PALLAS_VESTA
mkdir build && cd build
cmake ..
make
```

This will build two executables:

- `main_exec` – the main application entry point (`src/main.cu`)
- `cuda_tests` – unit tests using Google Test (`tests/test_runner.cu`)

## Running Unit Tests

From the `build/` directory, run:

```bash
ctest
```

Or directly:

```bash
./cuda_tests
```
> This runs all test cases defined using Google Test.
> 
If you want to run specific tests: 
```bash
./cuda_tests --gtest_filter=PointTests.*
```
available options are `CurveConstants`, `PointTests`, and `FieldTests` as of now. 




## Python Scripts

The following Python files are also available for generating reference values:

- `constant_sage.py`
- `point_sage.py`

To run these files you need to install `sagemath`. These are automatically copied to the build directory by CMake. For further run instruction about the arguments, use the following commands:
```bash
sage -python constant_sage.py --help
```
```bash
sage -python point_sage.py --help
```


## Cleaning Up

To clean all build files:

```bash
cd build
rm -rf *
```
