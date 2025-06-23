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

#include "./libstl-cuda/memory.cuh"
#include "./libstl-cuda/vector.cuh"
#include "./libstl-cuda/utility.cuh"

#include "./libff-cuda/fields/bigint_host.cuh"
#include "./libff-cuda/fields/fp_host.cuh"
#include "./libff-cuda/fields/fp2_host.cuh"
#include "./libff-cuda/fields/fp6_3over2_host.cuh"
#include "./libff-cuda/fields/fp12_2over3over2_host.cuh"
#include "./libff-cuda/curves/bls12_381/bls12_381_init_host.cuh"
#include "./libff-cuda/curves/bls12_381/bls12_381_g1_host.cuh"
#include "./libff-cuda/curves/bls12_381/bls12_381_g2_host.cuh"
#include "./libff-cuda/curves/bls12_381/bls12_381_pp_host.cuh"
#include "./libmatrix-cuda/transpose/transpose_ell2csr.cuh"
#include "./libmatrix-cuda/spmv/csr-balanced.cuh"
#include "./libff-cuda/scalar_multiplication/multiexp.cuh"


#include "./libff-cuda/curves/bls12_381/bls12_381_init.cuh"
#include "./libff-cuda/curves/bls12_381/bls12_381_pp.cuh"

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
    //generate two random scalar and test if the multiplication works
    //based on the implementation of cuzk, random element is not actually random, it produces same number no matter how many time you call
    
    //test the scalar multiplication
    libff::Fr<bls12_381_pp> f = ip->instance.random_element();
    libff::Fr<bls12_381_pp> g = ip->instance.random_element();
    printf("Scalar multiplication with scalar test\n");
    printf("F\n");
    f.as_bigint().print();
    printf("G=\n");
    g.as_bigint().print();
    printf("F*G = \n");
    f = f*g;
    f.as_bigint().print();

    printf("Modulus:\n");
    ip->instance.params->modulus->print();

    printf("\n-----------------------------------\n");
    
    //Initialize a point on the G1 field of the bls12_381 curve represented by y^2=x^3+4
    //First we initialize the point with the values of the generator point obtained from sage
    libff::bigint<6L> x_val("547267894408768087084154039555760353521479753946258632875036726158932984746527535614714820052060149146314557270019");
    libff::bigint<6L> y_val("1835063209175869974242139117441761755355391001264886580587881843166918857183334906933623397100805888438647438806516");
    libff::bigint<6L> z_val("1");
    libff::bls12_381_Fq m(ip->g1_instance.params->fq_params, x_val);
    libff::bls12_381_Fq n(ip->g1_instance.params->fq_params, y_val);
    libff::bls12_381_Fq o(ip->g1_instance.params->fq_params, z_val);

    //initialize the generator variable
    libff::G1<bls12_381_pp> generator(ip->g1_instance.params, m,n,o);
    //print the coordinates stored in the system
    printf("Generator point:\n");
    printf("X : \n");
    generator.X.as_bigint().print();
    printf("\nY : \n");
    generator.Y.as_bigint().print();
    printf("\nZ : \n");
    generator.Z.as_bigint().print();
    
    printf("\n-----------------------------------\n");
    //To test double, use the dbl() method to generate the double of the generator point
    libff::G1<bls12_381_pp> dblGenerator(ip->g1_instance.params);

    
    dblGenerator = generator.dbl();

    //convert to affine so it becomes easier to crosscheck with sage
    dblGenerator.to_affine_coordinates();

    printf("Double of Generator point:\n");
    printf("X : \n");
    dblGenerator.X.as_bigint().print();
    printf("\nY : \n");
    dblGenerator.Y.as_bigint().print();
    printf("\nZ : \n");
    dblGenerator.Z.as_bigint().print();

    libff::bigint<6L> x_val1("2647573539365908156187872320875289844232686969072315270667078416032850847337240712712688001976328196186160732037825");
    libff::bigint<6L> y_val1("2113093938667914396000776806186400354893362127190679233834878499249011208112236064419880982960294133785487484496738");
    libff::bigint<6L> z_val1("1");

    assert(x_val1 == dblGenerator.X.as_bigint());
    assert(y_val1 == dblGenerator.Y.as_bigint());
    assert(z_val1 == dblGenerator.Z.as_bigint());




    printf("\n-----------------------------------\n");

    //Test the addition, G+ (2*G) is taken here. 
    //it uses the default + operation defined in the cuzk project. 
    //the operation is addition but performs a sequence of multiplication
    libff::G1<bls12_381_pp> add_g_2g(ip->g1_instance.params);
    add_g_2g = generator+dblGenerator;

    //convert to affine so it becomes easier to crosscheck with sage
    add_g_2g.to_affine_coordinates();
    printf("Generator + 2*Generator:\n");
    printf("X : \n");
    add_g_2g.X.as_bigint().print();
    printf("\nY : \n");
    add_g_2g.Y.as_bigint().print();
    printf("\nZ : \n");
    add_g_2g.Z.as_bigint().print();


    libff::bigint<6L> x_val2("3985804938496873456137352930929051089705849152952571534528489884143131519159321947847380917414645971891223376872198");
    libff::bigint<6L> y_val2("1613579565203645874155960391187678510556750330299737199914964851572472065713038177477967827161821746801223930803737");
    libff::bigint<6L> z_val2("1");

    assert(x_val2 == add_g_2g.X.as_bigint());
    assert(y_val2 == add_g_2g.Y.as_bigint());
    assert(z_val2 == add_g_2g.Z.as_bigint());
    printf("\n-----------------------------------\n");

    //Test a point multiplication with a scalar. 
    //Since it does a lot of multiplication, it takes a lot of time to compute
    libff::G1<bls12_381_pp> scalar_result(ip->g1_instance.params);

    libff::bigint<4L> scalar("7093260539507");
    scalar_result = scalar*add_g_2g;


    scalar_result.to_affine_coordinates();
    printf("Scalar Multiplication result:\n");
    printf("X : \n");
    scalar_result.X.as_bigint().print();
    printf("\nY : \n");
    scalar_result.Y.as_bigint().print();
    printf("\nZ : \n");
    scalar_result.Z.as_bigint().print();

    libff::bigint<6L> x_val3("2990416832278317204423960161190962033182140278537947825332705471122050139499833334224803208095522007081317890697313");
    libff::bigint<6L> y_val3("3423211919889812752414216786057159302617883252024474251795346068064526621995760135560991251922464634905502966017585");
    libff::bigint<6L> z_val3("1");

    assert(x_val3 == scalar_result.X.as_bigint());
    assert(y_val3 == scalar_result.Y.as_bigint());
    assert(z_val3 == scalar_result.Z.as_bigint());
    
    __syncthreads();
    __syncwarp();
}
int main(int argc, char* argv[])
{

    int deviceCount;
    cudaGetDeviceCount( &deviceCount );
    CUTThread  thread[deviceCount];

    bls12_381_pp_host::init_public_params();
    cudaSetDevice(0);
    cudaFree(0);

    // params init 
    Mem device_mem[deviceCount];

    for(size_t i=0; i<deviceCount; i++)
    {
        device_mem[i].device_id = i;

        device_mem[i].mem = NULL;

        thread[i] = start_thread( multi_init_params, &device_mem[i] );

    }

    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    instance_params* ip[deviceCount];

    Instance instance[deviceCount];

    printf("%s %d\n",__FILE__, __LINE__);
    for(size_t i=0; i<deviceCount; i++)
    {
        instance[i].device_id = i;

        instance[i].ip = &ip[i];

        thread[i] = start_thread( multi_instance_init, &instance[i] );
    }
    for(size_t i=0; i<deviceCount; i++)
    {
        end_thread(thread[i]);
    }

    h_instance_params hip;
    instance_init_host(&hip);


    test_multiplication<<<1,1>>>(ip[0]);

    cudaDeviceReset();
    return 0;
}
