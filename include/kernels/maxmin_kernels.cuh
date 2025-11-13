#ifndef MAXMIN_KERNELS_CUH
#define MAXMIN_KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/types.cuh"

__global__ void max_min_kernel(
    const float *A, // [batch, M, K]
    const float *B, // [batch, K, N]
    float *C_min,   // [batch, M, K, N]
    float *C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size);

#endif // MAXMIN_KERNELS_CUH
