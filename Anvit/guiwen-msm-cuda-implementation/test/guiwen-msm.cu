struct instance_params;
struct h_instance_params;

#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define CUT_THREADEND

// Create thread
CUTThread start_thread(CUT_THREADROUTINE func, void *data)
{
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

// Wait for thread to finish
void end_thread(CUTThread thread)
{
    pthread_join(thread, NULL);
}

// Destroy thread
void destroy_thread(CUTThread thread)
{
    pthread_cancel(thread);
}

// Wait for multiple threads
void wait_for_threads(const CUTThread *threads, int num)
{
    for (int i = 0; i < num; i++)
        end_thread(threads[i]);
}

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../depends/libstl-cuda/memory.cuh"
#include "../depends/libstl-cuda/vector.cuh"
#include "../depends/libstl-cuda/utility.cuh"

#include "../depends/libff-cuda/fields/bigint_host.cuh"
#include "../depends/libff-cuda/fields/fp_host.cuh"
#include "../depends/libff-cuda/fields/fp2_host.cuh"
#include "../depends/libff-cuda/fields/fp6_3over2_host.cuh"
#include "../depends/libff-cuda/fields/fp12_2over3over2_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_init_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_g1_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_g2_host.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_pp_host.cuh"
#include "../depends/libmatrix-cuda/transpose/transpose_ell2csr.cuh"
#include "../depends/libmatrix-cuda/spmv/csr-balanced.cuh"
#include "../depends/libff-cuda/scalar_multiplication/multiexp.cuh"

#include "../depends/libff-cuda/curves/bls12_381/bls12_381_init.cuh"
#include "../depends/libff-cuda/curves/bls12_381/bls12_381_pp.cuh"

#include <time.h>
#include <set>

using namespace libff;

struct instance_params
{
    bls12_381_Fr instance;
    bls12_381_G1 g1_instance;
    bls12_381_G2 g2_instance;
    bls12_381_GT gt_instance;
};

struct h_instance_params
{
    bls12_381_Fr_host h_instance;
    bls12_381_G1_host h_g1_instance;
    bls12_381_G2_host h_g2_instance;
    bls12_381_GT_host h_gt_instance;
};

template <typename ppT>
struct MSM_params
{
    libstl::vector<libff::Fr<ppT>> vf;
    libstl::vector<libff::G1<ppT>> vg;
};

__global__ void init_params()
{
    gmp_init_allocator_();
    bls12_381_pp::init_public_params();
}

__global__ void instance_init(instance_params *ip)
{
    ip->instance = bls12_381_Fr(&bls12_381_fp_params_r);
    ip->g1_instance = bls12_381_G1(&g1_params);
    ip->g2_instance = bls12_381_G2(&g2_params);
    ip->gt_instance = bls12_381_GT(&bls12_381_fp12_params_q);
}

void instance_init_host(h_instance_params *ip)
{
    ip->h_instance = bls12_381_Fr_host(&bls12_381_fp_params_r_host);
    ip->h_g1_instance = bls12_381_G1_host(&g1_params_host);
    ip->h_g2_instance = bls12_381_G2_host(&g2_params_host);
    ip->h_gt_instance = bls12_381_GT_host(&bls12_381_fp12_params_q_host);
}

template <typename ppT>
__global__ void generate_MP(MSM_params<ppT> *mp, instance_params *ip, size_t size)
{
    new ((void *)mp) MSM_params<ppT>();
    mp->vf.presize(size, 512, 32);
    mp->vg.presize(size, 512, 32);

    libstl::launch<<<512, 32>>>(
        [=] __device__()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            libff::Fr<ppT> f = ip->instance.random_element();
            libff::G1<ppT> g = ip->g1_instance.random_element();
            f ^= idx;
            g = g * idx;
            while (idx < size)
            {
                mp->vf[idx] = f;
                mp->vg[idx] = g;
                f = f + f;
                g = g + g;
                idx += tnum;
            }
        });
    cudaDeviceSynchronize();

    ip->g1_instance.p_batch_to_special(mp->vg, 160, 32);
}

struct Mem
{
    size_t device_id;
    void *mem;
};

void *multi_init_params(void *params)
{
    Mem *device_mem = (Mem *)params;
    cudaSetDevice(device_mem->device_id);
    size_t init_size = 1024 * 1024 * 1024;
    init_size *= 15;
    if (cudaMalloc((void **)&device_mem->mem, init_size) != cudaSuccess)
        printf("device malloc error!\n");
    libstl::initAllocator(device_mem->mem, init_size);
    init_params<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

struct Instance
{
    size_t device_id;
    instance_params **ip;
};

void *multi_instance_init(void *instance)
{
    Instance *it = (Instance *)instance;
    cudaSetDevice(it->device_id);
    if (cudaMalloc((void **)it->ip, sizeof(instance_params)) != cudaSuccess)
        printf("ip malloc error!\n");
    instance_init<<<1, 1>>>(*it->ip);
    cudaDeviceSynchronize();
    return 0;
}

template <typename ppT>
struct MSM
{
    size_t device_id;
    MSM_params<ppT> *mp;
    instance_params *ip;
    libff::G1<ppT> *res;
};

template <typename ppT>
void *multi_MSM(void *msm)
{
    MSM<ppT> *it = (MSM<ppT> *)msm;
    cudaSetDevice(it->device_id);

    size_t lockMem;
    libstl::lock_host(lockMem);
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
    cudaDeviceSynchronize();
    libstl::resetlock_host(lockMem);

    cudaEvent_t eventMSMStart, eventMSMEnd;
    cudaEventCreate(&eventMSMStart);
    cudaEventCreate(&eventMSMEnd);
    cudaEventRecord(eventMSMStart, 0);
    cudaEventSynchronize(eventMSMStart);

    for (size_t i = 0; i < 1; i++)
    {
        it->res = libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(eventMSMEnd, 0);
    cudaEventSynchronize(eventMSMEnd);
    float TimeMSM;
    cudaEventElapsedTime(&TimeMSM, eventMSMStart, eventMSMEnd);
    printf("Time thread %lu for MSM:  %3.5f ms\n", it->device_id, TimeMSM);

    return 0;
}

template <typename ppT_host, typename ppT_device>
void D2H(libff::G1<ppT_host> *hg1, libff::G1<ppT_device> *dg1, libff::G1<ppT_host> *g1_instance)
{
    cudaMemcpy(hg1, dg1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    hg1->set_params(g1_instance->params);
}

template <typename ppT>
void Reduce(libff::G1<ppT> *hg1, libff::Fr<ppT> *instance, size_t total)
{
    int device_count;
    cudaGetDeviceCount(&device_count);

    libff::G1<ppT> g1 = hg1[device_count - 1];

    if (device_count != 1)
    {
        for (size_t i = device_count - 2; i <= device_count - 1; i--)
        {
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if (egroup > num_groups)
                egroup = num_groups;
            if (sgroup > num_groups)
                sgroup = num_groups;
            if (egroup == sgroup)
                continue;

            for (size_t j = 0; j < (egroup - sgroup) * c; j++)
            {
                g1 = g1.dbl();
            }
            g1 = g1 + hg1[i];
        }
    }

    g1.to_special();
}

__device__ long omega2(long n)
{
    long rem = n % 2;
    long exponent = 0;
    while (rem == 0)
    {
        exponent++;
        n >>= 1;
        rem = n % 2;
    }
    return exponent;
}

__device__ long omega3(long n)
{
    long rem = n % 3;
    long exponent = 0;
    while (rem == 0)
    {
        exponent++;
        n = n / 3;
        rem = n % 3;
    }
    return exponent;
}

//Bucket construction 
__device__ int8_t *bucket;
__global__ void construct_B0_bucket(long start, long limit)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // get the thread index
    if (idx >= start && idx <= limit)
    {
        if (((omega2(idx) + omega3(idx)) % 2) == 0)
        {
            bucket[idx] = 1;
        }
    }
}
__global__ void subtract_B1_omega2(long q, long start, long end)
{
    for (long nidx = start; nidx < end; nidx++)
    {
        if (bucket[nidx] && bucket[q - 2 * nidx])
        {                             // if both are in the bucket
            bucket[q - 2 * nidx] = 0; // remove q-2*idx from the bucket
        }
    }
}
__global__ void subtract_B1_omega3(long q, long start, long end)
{
    for (long nidx = start; nidx < end; nidx++)
    {
        if (bucket[nidx] && bucket[q - 3 * nidx])
        {                             // if both are in the bucket
            bucket[q - 3 * nidx] = 0; // remove q-3*idx from the bucket
        }
    }
}
__global__ void construct_bucket(const long q, const long ah)
{
    bucket[0] = 1;
    bucket[1] = 1;

    int blocksize = 128;
    int gridsize = ((q + 2) + blocksize - 1) / blocksize;

    construct_B0_bucket<<<gridsize, blocksize>>>(2, (long)(q / 2)); // construct the B0 bucket for 1 to q/2
    subtract_B1_omega2<<<1, 1>>>(q, (long)(q / 4), (long)(q / 2)); // subtract B1 omega2 from q/2 to q/4
    subtract_B1_omega3<<<1, 1>>>(q, (long)(q / 6), (long)(q / 4)); // subtract B1 omega3 from q/4 to q/6

    gridsize = ((ah + 2) + blocksize - 1) / blocksize;
    
    construct_B0_bucket<<<gridsize, blocksize>>>(1, (long)(ah + 1));
}

//decomposition map construction
struct mbflag
{
    int m;
    long b;
    bool flag;
    __host__ __device__ mbflag() : m(0), b(0), flag(false) {}
    __host__ __device__ mbflag(int _m, long _b, bool _flag) : m(_m), b(_b), flag(_flag) {}
};
__device__ libstl::vector<mbflag> *decompose;
__global__ void decompose_builder1(long q)
{
    if (decompose == nullptr)
        return;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // get the thread index

    if (idx <= q && bucket[idx] == 1) // ensure we only process valid indices in the bucket
    {
        int b = idx;
        for (int m = 1; m <= 3; m++)
        {
            if (m * b <= q)
            {
                (*decompose)[q - m * b].m = m;    // set the m value
                (*decompose)[q - m * b].b = b;    // set the b value
                (*decompose)[q - m * b].flag = 1; // initialize the flag to 0 for direct multiples
            }
        }
    }
}
__global__ void decompose_builder2(long q)
{
    if (decompose == nullptr)
        return;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // get the thread index
    if (idx <= q && bucket[idx])
    {
        int b = idx;
        for (int m = 1; m <= 3; m++)
        {
            if (m * b <= q)
            {
                (*decompose)[m * b].m = m;    // set the m value for multiples of b
                (*decompose)[m * b].b = b;    // set the b value for multiples of b
                (*decompose)[m * b].flag = 0; // set the flag to 0 for direct multiples
            }
        }
    }
}
__global__ void construct_decomposition_map(long q)
{
    decompose = new libstl::vector<mbflag>(q + 2); // create a new set on device

    size_t blocksize = 32;                                   // block size for the kernel launch
    size_t gridsize = ((q + 1) + blocksize - 1) / blocksize; // calculate the grid size

    decompose_builder1<<<gridsize, blocksize>>>(q); // build the first pass of decomposition map
    decompose_builder2<<<gridsize, blocksize>>>(q); // build the second pass of decomposition map

    cudaDeviceSynchronize(); // ensure the kernel has finished executing
}

//function to check for CUDA errors
__host__ __device__ void checkError(char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error here: %s, msg: %s\n", cudaGetErrorString(err), msg);
    }
}

//precomputation related functions
__device__ libstl::vector<libff::G1<bls12_381_pp>> *precomputed_points;

__device__ void set_precompute(libff::G1<bls12_381_pp> *point, size_t idx)
{
    (*precomputed_points)[3 * idx] = *point;
    (*precomputed_points)[3 * idx + 1] = point->dbl();
    (*precomputed_points)[3 * idx + 2] = *point + (*precomputed_points)[3 * idx + 1];
}
__global__ void precompute_points(MSM_params<bls12_381_pp> *mp, size_t size, size_t stream_idx)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tnum = blockDim.x * gridDim.x;
    idx = idx + stream_idx * tnum;
    if (idx < size)
    {
        (*precomputed_points)[3 * idx] = mp->vg[idx];
        (*precomputed_points)[3 * idx + 1] = mp->vg[idx].dbl();
        (*precomputed_points)[3 * idx + 2] = mp->vg[idx] + (*precomputed_points)[3 * idx + 1];
    }
}

//test for precomputed points
__global__ void precompute_test(MSM_params<bls12_381_pp> *mp, size_t size, size_t stream_idx)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tnum = blockDim.x * gridDim.x;
    idx = idx + stream_idx * tnum;
    if (idx < size)
    {
        assert(mp->vg[idx] == (*precomputed_points)[3 * idx]);                                          // ensure the first point matches
        assert(mp->vg[idx].dbl() == (*precomputed_points)[3 * idx + 1]);                                // ensure the second point is a double of the first
        assert((mp->vg[idx] + (*precomputed_points)[3 * idx + 1]) == (*precomputed_points)[3 * idx + 2]); // ensure the third point is the sum of the first and second
    }
}

//test for decomposition function
__global__ void test_decomposition(long q, size_t ah)
{
    assert(decompose != nullptr);                 // Ensure the decomposition map is initialized
    assert(decompose->size() == (size_t)(q + 2)); // Ensure the size of the decomposition map is correct
    for (int i = 0; i <= q; i++)
    {
        int m = (*decompose)[i].m;        // get the m value
        long b = (*decompose)[i].b;       // get the b value
        bool flag = (*decompose)[i].flag; // get the flag
        if (flag)
        {
            // if flag is set, it means this is a direct multiple
            assert(q - m * b == i);
        }
        else
        {
            // if flag is not set, it means this is a composite decomposition
            assert(m * b == i);
        }
        assert(bucket[b] == 1);  // ensure that b is in the bucket
        assert(m > 0 && m <= 3); // ensure m is in the range [1,3]
        if (i < ah)
            assert(flag == 0);
    }
}
#include <cstddef>

// Using C++11 alignas to enforce 16-byte alignment for CUDA
struct alignas(16) Point {
    size_t idx;
    size_t sign;
};

struct alignas(16) Bucket {
    Point* points;   // Pointer to an array of points
    size_t size;     // Number of points
};

struct alignas(16) Window {
    Bucket* buckets; // Pointer to an array of buckets
    size_t size;     // Number of buckets
};


int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Please enter the MSM scales (e.g. 20 represents 2^20) \n");
        return 1;
    }
    int log_size = atoi(argv[1]);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    CUTThread thread[deviceCount];

    bls12_381_pp_host::init_public_params();
    cudaSetDevice(0);

    size_t num_v = (size_t)(1 << log_size);

    // params init
    Mem device_mem[deviceCount];
    for (size_t i = 0; i < deviceCount; i++)
    {
        device_mem[i].device_id = i;
        device_mem[i].mem = NULL;
        thread[i] = start_thread(multi_init_params, &device_mem[i]);
    }
    for (size_t i = 0; i < deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    // instance init
    instance_params *ip[deviceCount];
    Instance instance[deviceCount];
    for (size_t i = 0; i < deviceCount; i++)
    {
        instance[i].device_id = i;
        instance[i].ip = &ip[i];
        thread[i] = start_thread(multi_instance_init, &instance[i]);
    }
    for (size_t i = 0; i < deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    h_instance_params hip;
    instance_init_host(&hip);

    // elements generation
    MSM_params<bls12_381_pp> *mp[deviceCount];
    for (size_t i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        if (cudaMalloc((void **)&mp[i], sizeof(MSM_params<bls12_381_pp>)) != cudaSuccess)
            printf("mp malloc error!\n");
        generate_MP<bls12_381_pp><<<1, 1>>>(mp[i], ip[i], num_v);
    }
    for (size_t i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);

    cudaEvent_t evStart, evEnd;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evEnd);
    cudaEventRecord(evStart, 0); 
    

    size_t *num_bits = new size_t(255); //for bls12_381 the number of bit is 255
    size_t total = mp[0]->vf.size_host(); // number of total points
    size_t log2_total = log2(total); 
    size_t s = log2_total - (log2_total / 3 - 2); //calculate the window size 
    size_t q = (1 << s); //the radix value for splitting scalars
    size_t rhBitLength = *num_bits % s;
    size_t limbCount = *num_bits / s;
    size_t *ah = new size_t(65);
    libstl::launch<<<1, 1>>>(
        [=] __device__(size_t *ah)
        {
            bucket = new int8_t[(1 << (s - 1)) + 2];
            memset(bucket, 0, sizeof(int8_t) * ((1 << (s - 1)) + 2));
        },
        ah);
    cudaDeviceSynchronize(); // ensure the kernel has finished executing

    cudaEventRecord(evEnd, 0);   // end the timer
    cudaEventSynchronize(evEnd); // ensure the event has been recorded
    float TimeInit;
    cudaEventElapsedTime(&TimeInit, evStart, evEnd);
    printf("Time taken to initialize the parameters:  %3.5f ms\n", TimeInit);


    cudaEvent_t eventStart, eventEnd, midEvent;
    cudaEventCreate(&eventStart);
    cudaEventCreate(&eventEnd);
    cudaEventCreate(&midEvent);
    cudaEventRecord(eventStart);        
    cudaEventSynchronize(eventStart);   
    construct_bucket<<<1, 1>>>(q, *ah); 

    cudaStreamSynchronize(0);                 
    cudaEventRecord(midEvent);                
    cudaEventSynchronize(midEvent);           
    construct_decomposition_map<<<1, 1>>>(q); 
    cudaStreamSynchronize(0);                 
   
    cudaEventRecord(eventEnd);      
    cudaEventSynchronize(eventEnd); 
    float TimeBucket;
    cudaEventElapsedTime(&TimeBucket, eventStart, midEvent);
    printf("Time taken to construct the bucket:  %3.5f ms\n", TimeBucket);
    float TimeDecompose;
    cudaEventElapsedTime(&TimeDecompose, midEvent, eventEnd);
    printf("Time taken to construct the decomposition map:  %3.5f ms\n", TimeDecompose);

    
    size_t stream_count = 64;
    cudaStream_t streams[stream_count];

    size_t points_per_stream = (num_v + stream_count -1) / stream_count;

    size_t block = 32;
    size_t grid = (points_per_stream + block - 1) / block;
    libstl::launch<<<1, 1>>>(
        [=] __device__()
        {
            precomputed_points = new libstl::vector<libff::G1<bls12_381_pp>>((num_v+3) * 3, (new libff::G1<bls12_381_pp>(&g1_params))->zero()); // initialize the vector on device
            precomputed_points->presize(num_v * 3, grid, block);
        }
    );
    checkError("precomputed_points init");
    cudaStreamSynchronize(0); 
    cudaDeviceSynchronize();  

    cudaEvent_t precomputeStart;
    cudaEventCreate(&precomputeStart);
    cudaEventRecord(precomputeStart);
    cudaEventSynchronize(precomputeStart);
    
    // printf("Precomputing points serially\n");
    // libstl::launch<<<1,1>>>(
    //     [=]
    //     __device__ (MSM_params<bls12_381_pp>* mp)
    //     {
    //         int count=0;
    //         for(int i=0;i< num_v; i++)
    //         {
    //             if(mp->vg[i].is_well_formed()){
    //                 (*precomputed_points)[3*i] = mp->vg[i];

    //                 (*precomputed_points)[3*i+1] = mp->vg[i].dbl();
    //                 (*precomputed_points)[3*i+2] = mp->vg[i] + (*precomputed_points)[3*i+1];
    //                 count++;
    //             }

    //         }
    //         printf("Initialized %d G1 points in vg on device\n", count);
    //         for(int i=0;i<num_v;i++)
    //         {
    //             assert(mp->vg[i] == (*precomputed_points)[3*i]); // ensure the first point matches
    //             assert(mp->vg[i].dbl() == (*precomputed_points)[3*i+1]); // ensure the second point is a double of the first
    //             assert((mp->vg[i] + (*precomputed_points)[3*i+1]) == (*precomputed_points)[3*i+2]); // ensure the third point is the sum of the first and second
    //         }

    //     }, mp[0]
    // );

    
    //precompute the points
    for (int i = 0; i < stream_count; i++)
    {
        cudaStreamCreate(&streams[i]);

    }
    for(int i=0;i<stream_count; i++)
    {
        precompute_points<<<grid, block, 0, streams[i]>>>(
            mp[0],              // input vector of G1 points
            num_v,              // size of the input vector,
            i);
    }
    for (size_t i = 0; i < stream_count; i++)
    {
        cudaStreamSynchronize(streams[i]); // ensure the kernel has finished executing
        cudaStreamDestroy(streams[i]);     // destroy the stream
    }

    cudaDeviceSynchronize();
    checkError("Precomputation of points could not complete");
    
    cudaEvent_t precomputeEnd;
    cudaEventCreate(&precomputeEnd);
    cudaEventRecord(precomputeEnd);
    cudaEventSynchronize(precomputeEnd);
    float precomputeTime;
    cudaEventElapsedTime(&precomputeTime, precomputeStart, precomputeEnd);
    printf("Time taken to construct the precomputation:  %3.5f ms\n", precomputeTime);
    

    // perform pippenger msm

    // printf("Precomputed Size: %lu\n", (*precomputed_points).size_host());

    // msm
    // MSM<bls12_381_pp> msm[deviceCount];
    // for(size_t i=0; i<deviceCount; i++)
    // {
    //     msm[i].device_id = i;
    //     msm[i].mp = mp[i];
    //     msm[i].ip = ip[i];
    //     thread[i] = start_thread( multi_MSM<bls12_381_pp>, &msm[i] );
    // }
    // for(size_t i=0; i<deviceCount; i++)
    // {
    //     end_thread(thread[i]);
    // }

    // libff::G1<bls12_381_pp_host> hg1[deviceCount];
    // for(size_t i=0; i < deviceCount; i++)
    // {
    //     cudaSetDevice(i);
    //     D2H<bls12_381_pp_host, bls12_381_pp>(&hg1[i], msm[i].res, &hip.h_g1_instance);
    // }

    test_decomposition<<<1, 1>>>(q, *ah); // test the decomposition map

    //__syncthreads();
    cudaStreamSynchronize(0); // ensure the kernel has finished executing
    cudaDeviceSynchronize();



    for (int i = 0; i < stream_count; i++)
    {
        cudaStreamCreate(&streams[i]);

    }
    for(int i=0;i<stream_count; i++)
    {
        precompute_test<<<grid, block, 0, streams[i]>>>(
            mp[0],              // input vector of G1 points
            num_v,              // size of the input vector,
            i);
    }
    for (size_t i = 0; i < stream_count; i++)
    {
        cudaStreamSynchronize(streams[i]); // ensure the kernel has finished executing
        cudaStreamDestroy(streams[i]);     // destroy the stream
    }

    cudaDeviceSynchronize();
    

    // Reduce<bls12_381_pp_host>(hg1, &hip.h_instance, num_v);

    checkError("Error before finishing?\n");
    cudaDeviceSynchronize(); 
    cudaDeviceReset();
    return 0;
}
