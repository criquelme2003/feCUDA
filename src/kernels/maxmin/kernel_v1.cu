
#include <float.h>

__global__ void max_min_kernel(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    float *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{
    extern __shared__ float mins[];

    unsigned int k = threadIdx.x;
    unsigned int n = blockIdx.x;
    unsigned int m = blockIdx.y;
    unsigned int b = blockIdx.z;

    unsigned int a_idx = b * M * K + m * K + k;
    unsigned int b_idx = b * K * N + k * N + n;
    unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + k;

    if (k < K)
    {
        mins[k] = fminf(A[a_idx], B[b_idx]);
        C_min[c_min_idx] = mins[k];
    }
    else
    {
        mins[k] = FLT_MAX;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (k < s && (k + s) < K)
        {
            float val_a = mins[k];
            float val_b = mins[k + s];
            mins[k] = fmaxf(val_a, val_b);
        }
        __syncthreads();
    }

    if (k == 0)
    {
        C_max[b * M * N + m * N + n] = mins[0];
    }
}
