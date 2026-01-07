#include <iostream>
// #include "./../include/G1Point.cuh"
// #include "./../include/FieldG1.cuh"
// #include "./../include/Scalar.cuh"
// #include "./../include/vector.cuh"
#include "./../include/cuZK_2.cuh"
// #include "./../include/MultiBucketMSM.cuh"
// #include "./../utils/bls12_381_constants.cu"
// #include "./../utils/common_limb_operations.cu"
#include "./../utils/libff_converter.cuh"
// #include <libff/algebra/curves/bls12_381/bls12_381_pp.hpp>

#include "./../include/libff.cuh"

// __global__ void print(G1Point *g1_point)
// {
//     g1_point->print();
// }
__global__ void compare(G1Point *a, G1Point *b)
{
    if(*a != *b) 
        printf("Not equal");
}
__global__ void compare(Scalar *a, Scalar *b)
{
    if(*a != *b) 
    {
        printf("Not equal\n");
        printf("A = ");
        a->print();
        printf("B = ");
        b->print();
    }
        
}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("Incorrect input. Pass the scale of msm as argument.\n");
        exit(1);
    }
    int scale = atoi(argv[1]);
    size_t total = (size_t)1<<scale;

    libff_compute base(total);
    base.msm();

    printf("Base \nX = \n");
    base.result.X.as_bigint().print();
    printf("Y =\n");
    base.result.Y.as_bigint().print();
    printf("Z=\n");
    base.result.Z.as_bigint().print();

    G1Point *bresult;
    cudaMalloc(&bresult, sizeof(G1Point));
    G1Point_from_Libff(bresult, &base.result);

    // int num_scale;
    // printf("Enter the number of scalars: (20 refers to 2^20)\n");
    // scanf("%d", &num_scale);
    // size_t total_num = (size_t)1 << num_scale;
    // libff::bls12_381_pp::init_public_params();
    G1Point *points; //= Get_G1Point_from_Libff(total_num);
    Scalar *scalars;// = Get_Scalar_from_Libff(total_num);
    cudaMalloc(&points, sizeof(G1Point)*total);
    cudaMemcpy(points, base.points->data(), sizeof(G1Point)*total, cudaMemcpyHostToDevice);
    cudaMalloc(&scalars, sizeof(Scalar)*total);
    for(size_t i=0;i<total;i++)
    {
        Scalar_from_Libff(&scalars[i], &base.scalars->at(i));
    }
    // cudaMemcpy(scalars, base.scalars->data(), sizeof(Scalar)*total, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // check the correctness of point copy
    // for(int i=0;i<1000;i++)
    // {
    //     G1Point *temp;
    //     cudaMalloc(&temp, sizeof(G1Point));
    //     G1Point_from_Libff(temp, &base.points->at(i));

    //     compare<<<1,1>>>(temp, &points[i]);
    //     cudaDeviceSynchronize();
    //     cudaFree(temp);
    // }
    // for(int i=0;i<10;i++)
    // {
    //     Scalar *temp;
    //     cudaMalloc(&temp, sizeof(Scalar));
    //     printf("As bigint = ");
    //     Scalar_from_Libff(temp, &base.scalars->at(i));

    //     compare<<<1,1>>>(temp, &scalars[i]);
    //     cudaDeviceSynchronize();
    //     cudaFree(temp);
    // }



    // G1Point *result;
    // printf("Total number of points: %zu\n", total);
    // printf("Size of each point: %zu\n", sizeof(G1Point));
    // MultiBucketMSM *msm = new MultiBucketMSM(points, scalars, total);
    // msm->run();
    // cudaDeviceSynchronize();


    // launch<<<1,1>>>(
    //     [=] __device__ (G1Point *result, G1Point *bresult){
    //         result->print();
    //         printf("Result: %s\n", (*result == *bresult)? "Matched": "Not matched");
    //     }, msm->result, bresult

    // );


    
    // vector<G1Point> *host_points;
    // vector<Scalar> *host_scalars;

    // launch<<<1,1>>>(
    //     [=] __device__ (vector<G1Point> *points, vector<Scalar> *scalars){
    //         printf("Hello!\n");
    //         points = (vector<G1Point>*)malloc(sizeof(vector<G1Point>));
    //         points->resize(total_num);
    //         scalars = (vector<Scalar>*)malloc(sizeof(vector<Scalar>));
    //         scalars->resize(total_num);
    //         printf("addresses: %016lx %016lx \n", points, scalars);
    //     }, host_points, host_scalars
    // );

    // // cudaMalloc(&host_points, sizeof(vector<G1Point>));
    // // cudaMalloc(&host_scalars, sizeof(vector<Scalar>));
    // // cudaMalloc(&(*host_points)._data, sizeof(G1Point)*total_num);
    // // cudaMalloc(&(*host_scalars)._data, sizeof(Scalar)*total_num);
    // // cudaMemcpy(&(*host_points)._size, &total_num, sizeof(size_t), cudaMemcpyHostToDevice);
    // // cudaMemcpy(&(*host_scalars)._size, &total_num, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    
    // from_array_to_vector_point<<<1024, 256>>>(points, host_points, total_num);
    // from_array_to_vector_scalar<<<1024, 256>>>(scalars, host_scalars, total_num);
    
    // cudaDeviceSynchronize();

    // // launch<<<1,1>>>(
    // //     [=] __device__ (vector<G1Point> *points, vector<Scalar> *scalars){
    // //         (*points)[0].print();
    // //         (*scalars)[0].print();
    // //     }, host_points, host_scalars
    // // );
    // // cudaFree(points);
    // // cudaFree(scalars);
    // cudaDeviceSynchronize();
    // printf("converted to vector\n");

    cuZK *zk = new cuZK(points, scalars, total);
    G1Point *res = zk->run();

    CUDA_CHECK(cudaDeviceSynchronize());
    if (cudaError_t err = cudaGetLastError())
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    // // scanf("%d", &num_scale);
}