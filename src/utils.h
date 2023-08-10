// check hip error
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
              error, __FILE__, __LINE__);                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
