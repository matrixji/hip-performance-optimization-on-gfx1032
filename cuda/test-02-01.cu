#include <cuda.h>
#include <iostream>
#include "const.h"
#include "utils.h"


__global__ void null_kernel(float * __restrict__ a) {

}

int main(int argc, char **argv) {
    int total_threads = 1024 * 1024 * 1024;
    int block_dim_x = 1024;
    int block_dim_y = 1;
    int block_dim_z = 1;

    if (argc > 1) {
        block_dim_x = atoi(argv[1]);
    }
    if (argc > 2) {
        block_dim_y = atoi(argv[2]);
    }
    if (argc > 3) {
        block_dim_z = atoi(argv[3]);
    }
    auto threads_per_block = block_dim_x * block_dim_y * block_dim_z;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "Device name: " << props.name << std::endl;
    std::cout << "System major: " << props.major << std::endl;
    std::cout << "System minor: " << props.minor << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_time_ms;
    float *a = nullptr;
    
    CUDA_CHECK(cudaMalloc(&a, sizeof(float)));
    null_kernel<<<1, threads_per_block>>>(a);
    CUDA_CHECK(cudaEventRecord(start));
    null_kernel<<<total_threads/threads_per_block, dim3(block_dim_x, block_dim_y, block_dim_z)>>>(a);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
    std::cout << "Elapsed time: " << elapsed_time_ms << " ms" << std::endl;
    std::cout << "Totals: " << total_threads << "threads, ";
    std::cout << "dim3(" << block_dim_x << ", " << block_dim_y << ", " << block_dim_z << ") ";
    std::cout << total_threads / elapsed_time_ms * 1000 / kSclk << " Threads/Cycle" << std::endl;

    CUDA_CHECK(cudaFree(a));
    return 0;
}