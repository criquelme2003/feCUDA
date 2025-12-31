#include <cuda_runtime.h>
#include <float.h>

__global__ void max_min_kernel(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    float *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{

    extern __shared__ float mins[];

    unsigned int local_id = threadIdx.x;
    unsigned int global_id = (blockIdx.x * blockDim.x) + local_id ;
    unsigned int ks_per_block =  blockDim.x / K;

    unsigned int k = global_id % K;
    unsigned int out_id = global_id / K;
    

    unsigned int b = out_id / (M * N);
    unsigned int rem = out_id % (M * N);
    unsigned int m = rem / N;
    unsigned int n = rem - m*N;

    unsigned int a_idx = b * M * K + m * K + k;
    unsigned int b_idx = b * K * N + k * N + n;
    unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + k;

    if (local_id  < (ks_per_block * K) )
    {
        mins[local_id] = fminf(A[a_idx], B[b_idx]);
        C_min[c_min_idx] = mins[local_id];
    }
    else
    {
        mins[local_id] = FLT_MAX;
    }
    __syncthreads();
    unsigned base = (local_id / K) * K;
    for (unsigned int s = K >> 1; s > 0; s >>= 1)
    {
        if (k < s && (k + s) < K)
        {
            float a = mins[base + k];
            float b = mins[base + k +s];
            mins[base + k] = fmaxf(a, b); // usa fminf si quisieras mínimo final
        }
        __syncthreads();
    }

    if (k == 0)
    {
        C_max[out_id] = mins[base];
    }
}
