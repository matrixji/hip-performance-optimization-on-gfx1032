// check cuda error
#define CUDA_CHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t error = cmd;                                                    \
    if (error != cudaSuccess) {                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error),  \
              error, __FILE__, __LINE__);                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
