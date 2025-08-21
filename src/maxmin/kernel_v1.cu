#include "../../include/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <float.h>
#include <stdio.h>
#include <algorithm>

__global__ void max_min_kernel(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    float *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{

    extern __shared__ float mins[];

    unsigned k = threadIdx.x;
    unsigned int m = blockIdx.y;
    unsigned int n = blockIdx.x;
    unsigned int b = blockIdx.z;

    
    unsigned int a_idx = b * M * K + m * K + k;
    unsigned int b_idx = b * K * N + k * N + n;
    unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + k;

    mins[k] = fminf(A[a_idx], B[b_idx]);
    C_min[c_min_idx] = mins[k];

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        // REDUCE2 
        if (k < s)
        { 
            mins[k] = fmaxf(mins[k], mins[k + s]);
        }
        __syncthreads();
    }
    if (k == 0)
    {
        C_max[b * M * N + m * N + n] = mins[0];
    }
}
