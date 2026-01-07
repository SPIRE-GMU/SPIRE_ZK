struct instance_params;
struct h_instance_params;

#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define  CUT_THREADEND

//Create thread
CUTThread start_thread(CUT_THREADROUTINE func, void * data){
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

//Wait for thread to finish
void end_thread(CUTThread thread){
    pthread_join(thread, NULL);
}

//Destroy thread
void destroy_thread( CUTThread thread ){
    pthread_cancel(thread);
}

//Wait for multiple threads
void wait_for_threads(const CUTThread * threads, int num){
    for(int i = 0; i < num; i++)
        end_thread( threads[i] );
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

using namespace libff;



#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        
        //std::exit(EXIT_FAILURE);
    }
}

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


template<typename ppT>
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

__global__ void instance_init(instance_params* ip)
{
    ip->instance = bls12_381_Fr(&bls12_381_fp_params_r);
    ip->g1_instance = bls12_381_G1(&g1_params);
    ip->g2_instance = bls12_381_G2(&g2_params);
    ip->gt_instance = bls12_381_GT(&bls12_381_fp12_params_q);
}

void instance_init_host(h_instance_params* ip)
{
    ip->h_instance = bls12_381_Fr_host(&bls12_381_fp_params_r_host);
    ip->h_g1_instance = bls12_381_G1_host(&g1_params_host);
    ip->h_g2_instance = bls12_381_G2_host(&g2_params_host);
    ip->h_gt_instance = bls12_381_GT_host(&bls12_381_fp12_params_q_host);
}


template<typename ppT>
__global__ void generate_MP(MSM_params<ppT>* mp, instance_params* ip, size_t size)
{
    new ((void*)mp) MSM_params<ppT>();
    mp->vf.presize(size, 512, 32);
    mp->vg.presize(size, 512, 32);

    libstl::launch<<<512, 32>>>
    (
        [=]
        __device__ ()
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t tnum = blockDim.x * gridDim.x;
            libff::Fr<ppT> f = ip->instance.random_element();
            libff::G1<ppT> g = ip->g1_instance.random_element();
            f ^= idx;
            g = g * idx;
            while(idx < size)
            {
                mp->vf[idx] = f;
                mp->vg[idx] = g;
                f = f + f;
                g = g + g;
                idx += tnum;
            }
        }
    );
    cudaDeviceSynchronize();

    ip->g1_instance.p_batch_to_special(mp->vg, 160, 32);
}

struct Mem
{
    size_t device_id;
    void* mem;
};

void* multi_init_params(void* params)
{
    Mem* device_mem = (Mem*) params;
    cudaSetDevice(device_mem->device_id);
    size_t init_size = 1024 * 1024 * 1024;
    init_size *= 15;
    CHECK_LAST_CUDA_ERROR();
    if( cudaMalloc( (void**)&device_mem->mem, init_size ) != cudaSuccess) printf("device malloc error!\n");
    libstl::initAllocator(device_mem->mem, init_size);
    init_params<<<1, 1>>>();
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    return 0;
}

struct Instance
{
    size_t device_id;
    instance_params** ip;
};

void* multi_instance_init(void* instance)
{
    Instance* it = (Instance*)instance;
    cudaSetDevice(it->device_id);
    CHECK_LAST_CUDA_ERROR();
    if( cudaMalloc( (void**)it->ip, sizeof(instance_params)) != cudaSuccess) printf("ip malloc error!\n");
    CHECK_LAST_CUDA_ERROR();
    instance_init<<<1, 1>>>(*it->ip);
    cudaDeviceSynchronize();
    return 0;
}

template<typename ppT>
struct MSM
{
    size_t device_id;
    MSM_params<ppT>* mp;
    instance_params* ip;
    libff::G1<ppT>* res;
};

template<typename ppT>
void* multi_MSM(void* msm)
{
    MSM<ppT>* it = (MSM<ppT>*)msm;
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
    cudaEventCreate( &eventMSMStart);
	cudaEventCreate( &eventMSMEnd);
    cudaEventRecord( eventMSMStart, 0); 
    cudaEventSynchronize(eventMSMStart);
    for(size_t i=0; i<1; i++)
    {
        it->res = libff::p_multi_exp_faster_multi_GPU_host<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method_naive_plain>(it->mp->vg, it->mp->vf, it->ip->instance, it->ip->g1_instance, 512, 32);
        cudaDeviceSynchronize();
    }


    cudaEventRecord( eventMSMEnd, 0);
    cudaEventSynchronize(eventMSMEnd);
    float   TimeMSM;
    cudaEventElapsedTime( &TimeMSM, eventMSMStart, eventMSMEnd );
    printf( "Time thread %lu for MSM:  %3.5f ms\n", it->device_id, TimeMSM );

    return 0;
}

template<typename ppT_host, typename ppT_device>
void D2H(libff::G1<ppT_host>* hg1, libff::G1<ppT_device>* dg1, libff::G1<ppT_host>* g1_instance)
{
    cudaMemcpy(hg1, dg1, sizeof(libff::G1<ppT_device>), cudaMemcpyDeviceToHost);
    hg1->set_params(g1_instance->params);
}


template<typename ppT>
void Reduce(libff::G1<ppT>* hg1, libff::Fr<ppT>* instance, size_t total)
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    libff::G1<ppT> g1 = hg1[device_count-1];

    if(device_count != 1)
    {
        for(size_t i=device_count - 2; i <= device_count - 1; i--)
        {
            size_t log2_total = libff::log2(total);
            size_t c = log2_total - (log2_total / 3 - 2);
            size_t num_bits = instance->size_in_bits();
            size_t num_groups = (num_bits + c - 1) / c;
            size_t sgroup = (num_groups + device_count - 1) / device_count * i;
            size_t egroup = (num_groups + device_count - 1) / device_count * (i + 1);
            if(egroup > num_groups) egroup = num_groups;
            if(sgroup > num_groups) sgroup = num_groups;
            if(egroup == sgroup) continue;

            for(size_t j=0; j < (egroup - sgroup) * c; j++)
            {
                g1 = g1.dbl();
            }
            g1 = g1 + hg1[i];
        }
    }

    g1.to_special();

}

__global__ void test_multiplication(instance_params* ip)
{
    bigint<4> a("40717440697516928871729794459926083509479925425253447106356521628001304033345");
    bigint<4> b("40717440697516928871729794459926083509479925425253447106356521628001304033345");
    libff::Fr<bls12_381_pp> f = ip->instance.random_element();
    libff::Fr<bls12_381_pp> g = ip->instance.random_element();
    
    printf("F\n");
    f.as_bigint().print();
    printf("G=\n");
    g.as_bigint().print();
    printf("F*G = \n");
    f = f*g;
    f.as_bigint().print();

    printf("Modulus:\n");
    ip->instance.params->modulus->print();
    
    
    //libff::G1<bls12_381_pp> q = ip->g1_instance.random_element();
    
    libff::bigint<6L> x_val("3455274221085403918568630834754176304780762356539104640283219223328938319396495915575948888146901796993445978399141");
    libff::bigint<6L> y_val("3112993911598655673602636130584638258648881928712239965254047157896709175076195307280186219428719718278126484846253");
    libff::bigint<6L> z_val("1");
    
    libff::bls12_381_Fq m(ip->g1_instance.params->fq_params, x_val);
    libff::bls12_381_Fq n(ip->g1_instance.params->fq_params, y_val);
    libff::bls12_381_Fq o(ip->g1_instance.params->fq_params, z_val);

    libff::G1<bls12_381_pp> generator(ip->g1_instance.params, m,n,o);


    

    // q.to_affine_coordinates();
    printf("X : \n");
    generator.X.as_bigint().print();
    printf("\nY : \n");
    generator.Y.as_bigint().print();
    printf("\nZ : \n");
    generator.Z.as_bigint().print();
    generator = generator.dbl();
    generator.to_affine_coordinates();
    printf("X : \n");
    generator.X.as_bigint().print();
    printf("\nY : \n");
    generator.Y.as_bigint().print();
    printf("\nZ : \n");
    generator.Z.as_bigint().print();
    
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
		printf("Please enter the MSM scales (e.g. 20 represents 2^20) \n");
		return 1;
	}

    int log_size = atoi(argv[1]);

    int deviceCount;
    cudaGetDeviceCount( &deviceCount );
    CUTThread  thread[deviceCount];

    bls12_381_pp_host::init_public_params();
    cudaSetDevice(0);
    cudaFree(0);
    size_t num_v = (size_t) (1 << log_size);

    // params init 
    Mem device_mem[deviceCount];

    printf("%s %d\n",__FILE__, __LINE__);
    CHECK_LAST_CUDA_ERROR();
    for(size_t i=0; i<deviceCount; i++)
    {
        device_mem[i].device_id = i;

        CHECK_LAST_CUDA_ERROR();
        device_mem[i].mem = NULL;

        CHECK_LAST_CUDA_ERROR();
        thread[i] = start_thread( multi_init_params, &device_mem[i] );

        CHECK_LAST_CUDA_ERROR();
    }

    printf("%s %d\n",__FILE__, __LINE__);
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    // printf("Device count: %d\n", deviceCount);
    // instance init
    instance_params* ip[deviceCount];

    CHECK_LAST_CUDA_ERROR();
    Instance instance[deviceCount];

    printf("%s %d\n",__FILE__, __LINE__);
    CHECK_LAST_CUDA_ERROR();
    for(size_t i=0; i<deviceCount; i++)
    {
        instance[i].device_id = i;

        CHECK_LAST_CUDA_ERROR();
        instance[i].ip = &ip[i];

        CHECK_LAST_CUDA_ERROR();
        thread[i] = start_thread( multi_instance_init, &instance[i] );
    }
    printf("%s %d\n",__FILE__, __LINE__);
    CHECK_LAST_CUDA_ERROR();
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    printf("%s %d\n",__FILE__, __LINE__);
    CHECK_LAST_CUDA_ERROR();
    h_instance_params hip;

    CHECK_LAST_CUDA_ERROR();
    instance_init_host(&hip);


    printf("%s %d\n",__FILE__, __LINE__);

    test_multiplication<<<1,1>>>(ip[0]);

    // CHECK_LAST_CUDA_ERROR();
    // // elements generation
    // MSM_params<bls12_381_pp>* mp[deviceCount];

    // printf("%s %d\n",__FILE__, __LINE__);
    // CHECK_LAST_CUDA_ERROR();
    // for(size_t i=0; i<deviceCount; i++)
    // {
    //     cudaSetDevice(i);
    //     CHECK_LAST_CUDA_ERROR();
    //     cudaDeviceSynchronize();
    //     CHECK_LAST_CUDA_ERROR();
    //     if( cudaMalloc( (void**)&mp[i], sizeof(MSM_params<bls12_381_pp>)) != cudaSuccess) printf("mp malloc error!\n");
    //     CHECK_LAST_CUDA_ERROR();
    //     generate_MP<bls12_381_pp><<<1, 1>>>(mp[i], ip[i], num_v);

    //     CHECK_LAST_CUDA_ERROR();
    // }

    // printf("%s %d\n",__FILE__, __LINE__);
    // for(size_t i=0; i<deviceCount; i++)
    // {
    //     cudaSetDevice(i);
    //     cudaDeviceSynchronize();
    // }
    // cudaSetDevice(0);
    
    // // msm
    // MSM<bls12_381_pp> msm[deviceCount];
    // for(size_t i=0; i<deviceCount; i++)
    // {
    //     msm[i].device_id = i;
    //     msm[i].mp = mp[i];
    //     msm[i].ip = ip[i];
    //     thread[i] = start_thread( multi_MSM<bls12_381_pp>, &msm[i] );
    // }

    // printf("%s %d\n",__FILE__, __LINE__);
    // for(size_t i=0; i<deviceCount; i++)
    // {
    //     end_thread(thread[i]);
    // }

    // printf("%s %d\n",__FILE__, __LINE__);
    // libff::G1<bls12_381_pp_host> hg1[deviceCount];
    // for(size_t i=0; i < deviceCount; i++)
    // {
    //     cudaSetDevice(i);
    //     D2H<bls12_381_pp_host, bls12_381_pp>(&hg1[i], msm[i].res, &hip.h_g1_instance);
    // }

    // Reduce<bls12_381_pp_host>(hg1, &hip.h_instance, num_v);

    cudaDeviceReset();
    return 0;
}
