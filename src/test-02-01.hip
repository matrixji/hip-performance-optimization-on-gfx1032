#include <iostream>
#include <hip/hip_runtime.h>
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

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Device name: " << props.name << std::endl;
    std::cout << "System major: " << props.major << std::endl;
    std::cout << "System minor: " << props.minor << std::endl;
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    float elapsed_time_ms;
    float *a = nullptr;
    
    HIP_CHECK(hipMalloc(&a, sizeof(float)));
    null_kernel<<<1, threads_per_block>>>(a);
    HIP_CHECK(hipEventRecord(start));
    null_kernel<<<total_threads/threads_per_block, dim3(block_dim_x, block_dim_y, block_dim_z)>>>(a);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_time_ms, start, stop));
    std::cout << "Elapsed time: " << elapsed_time_ms << " ms" << std::endl;
    std::cout << "Totals: " << total_threads << "threads, ";
    std::cout << "dim3(" << block_dim_x << ", " << block_dim_y << ", " << block_dim_z << ") ";
    std::cout << total_threads / elapsed_time_ms * 1000 / kSclk << " Threads/Cycle" << std::endl;

    HIP_CHECK(hipFree(a));
    return 0;
}