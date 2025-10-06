
#include <float.h>

__global__ void max_min_lineal_kernel(
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
    unsigned int b_idx = b * K * N + n * K + k;
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

__global__ void max_min_lineal_kernel_v2(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    float *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{
    extern __shared__ float mins[];

    unsigned tid = threadIdx.x;
    unsigned int m = blockIdx.y;
    unsigned int n = blockIdx.x;
    unsigned int b = blockIdx.z;

    unsigned int a_idx = b * M * K + m * K + tid;
    unsigned int b_idx = b * K * N + n * K + tid;
    unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + tid;

    for (unsigned int k = tid; k < K; k += blockDim.x)
    {

        float min_val = fminf(A[a_idx], B[b_idx]);
        C_min[c_min_idx] = min_val;
    }
    __syncthreads();

    float thread_max = float(-FLT_MAX);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    mins[tid] = thread_max;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        // REDUCE2
        if (tid < s)
        {
            mins[tid] = fmaxf(mins[tid], mins[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        C_max[b * M * N + m * N + n] = mins[0];
    }
}


