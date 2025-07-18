#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// CUDA kernel for tiled matrix multiplication
__global__ void matmul_tiled_kernel(float* A, float* B, float* C, int M, int N, int P) {
    __shared__ float tileA_s[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB_s[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; //RANK i
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; //RANK k

    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) { // Loop over tiles 
        if (row < M && t * TILE_SIZE + threadIdx.x < N) // Load A tile
            tileA_s[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA_s[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < N && col < P) // Load B tile
            tileB_s[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * P + col];
        else
            tileB_s[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) // Iterate over the RANK j
            sum += tileA_s[threadIdx.y][j] * tileB_s[j][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < P)
        C[row * P + col] = sum;
}

// CPU-side matrix multiplication
void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int P) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
}

// Random initialization
void initialize_matrix(std::vector<float>& mat) {
    for (auto& v : mat)
        v = static_cast<float>(rand()) / RAND_MAX;
}

void check_result(const std::vector<float>& A, const std::vector<float>& B, int M, int N, int P) {
    float max_err = 0.0f;
    for (int i = 0; i < A.size(); ++i) {
        float err = std::abs(A[i] - B[i]);
        max_err = std::max(max_err, err);
    }
    std::cout << "Max error between CPU and GPU results: " << max_err << std::endl;
    assert(max_err < 1e-3f);
}

int main() {
    int M = 512, N = 512, P = 512;

    std::vector<float> h_A(M * N), h_B(N * P), h_C_gpu(M * P), h_C_cpu(M * P);
    initialize_matrix(h_A);
    initialize_matrix(h_B);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * P * sizeof(float));
    cudaMalloc(&d_C, M * P * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((P + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, P);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu.data(), d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    matmul_cpu(h_A, h_B, h_C_cpu, M, N, P);

    check_result(h_C_gpu, h_C_cpu, M, N, P);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Matrix multiplication passed correctness check!" << std::endl;
    return 0;
}