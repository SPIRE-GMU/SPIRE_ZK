#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    size_t count = 524296;

    // Allocate host-pinned memory
    void* host_ptr;
    CUDA_CHECK(cudaMallocHost(&host_ptr, count));

    // Allocate device memory
    void* device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, count));

    // Fill device memory with something (e.g. 0s)
    CUDA_CHECK(cudaMemset(device_ptr, 0, count));

    // Create non-blocking stream
    cudaStream_t streamT;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamT, cudaStreamNonBlocking));

    // Perform async copy
    CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, count, cudaMemcpyDeviceToHost, streamT));

    // Wait for stream
    CUDA_CHECK(cudaStreamSynchronize(streamT));

    // Clean up
    CUDA_CHECK(cudaFreeHost(host_ptr));
    CUDA_CHECK(cudaFree(device_ptr));
    CUDA_CHECK(cudaStreamDestroy(streamT));

    std::cout << "Success!" << std::endl;
    return 0;
}
