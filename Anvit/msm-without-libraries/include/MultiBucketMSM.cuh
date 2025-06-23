#ifndef __MULTIBUCKETMSM_CUH__
#define __MULTIBUCKETMSM_CUH__

#include <cuda_runtime.h>
#include "./../include/vector.cuh"
#include "./../include/FieldG1.cuh"
#include "./../include/Scalar.cuh"
#include "./../include/G1Point.cuh"
#include "./../utils/bls12_381_constants.cu"
#include "./../utils/utils.cu"

struct alignas(16) Decomposition
{
    long m;
    long b;
    bool flag;
};
struct alignas(16) Scalar_Part
{
    uint32_t bucket;
    int8_t flag;
    size_t index;
    long bucket_index;
    __host__ __device__ Scalar_Part() : bucket(0), flag(0), index(0), bucket_index(0) {}
    __host__ __device__ Scalar_Part(long _bucket, int8_t _flag, size_t _index, int _window)
        : bucket(_bucket), flag(_flag), index(_index), bucket_index(_window) {}
};

class MultiBucketMSM
{
    __device__ G1Point *points;
    __device__ G1Point *precomputed_points;
    __device__ Scalar *scalars;
    size_t total_num;
    size_t window_size;
    size_t num_windows;
    size_t num_bits;
    size_t sm_count;
    size_t max_shared_memory_per_block;
    size_t bucket_set_count;
    size_t bucket_size;
    long *bucket_set;
    Decomposition *decomposition;
    __device__ Scalar_Part *scalar_parts;
    __device__ G1Point *buckets;
    __device__ G1Point *summed_buckets;
    __device__ G1Point *window_sums;

    size_t sum_bucket_size;
    __device__ long *bucket_map;

public:
    __device__ G1Point *result;
    MultiBucketMSM(G1Point *_points, Scalar *_scalars, size_t _total_num);
    ~MultiBucketMSM();
    cudaEvent_t start, stop;
    void init_buckets();
    void computeBucket();
    void generateDecomposition();
    void precompute();
    void processScalar();
    void computeBucketSums();
    void computeBuckets();
    void reduceResult();

    G1Point run();
};

#include "./../src/MultiBucketMSM.cu"

#endif