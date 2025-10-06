#ifndef BOOTSTRAP_CUH
#define BOOTSTRAP_CUH

float *bootstrap_wrapper(float *data, int M, int N, int batch_size, int replicas);

#endif // BOOTSTRAP_CUH