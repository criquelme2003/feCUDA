#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <float.h>

#define MIN_DIFF 0.001
__global__ void maxmin_threshold_kernel(
    __half *__restrict__ X,        // gen_tensor [B,M,K]
    const __half *__restrict__ X0, // original_tensor [B,K,N]
    int4 *__restrict__ paths,      // output paths
    __half *__restrict__ values,   // min values
    int *__restrict__ counter,     // atomic counter
    __half thr,
    int B,
    int M,
    int N,
    int K,
    int batch_id
)
{
    int b = batch_id >= 0 ? batch_id : blockIdx.z;
    int m = blockIdx.y;
    int n = blockIdx.x;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int out_id = b * M * N + m * N + n;
    extern __shared__ __half smem[];

    __half v = __float2half(-65504.0f);

    // Grid-stride: cada thread procesa múltiples K, aplicando max instantaneamene para controlar K.
    for (int k = tid; k < K; k += block_size)
    {
        int a_idx = b * M * K + m * K + k;
        int b_idx = b * K * N + k * N + n;
        v = __hmax(v, __hmin(X[a_idx], X0[b_idx]));
    }

    smem[tid] = v;
    __syncthreads();

    // Reducción max
    for (int s = (block_size / 2) >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            smem[tid] = __hmax(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    __half k_max = smem[0];

    __syncthreads();

    // Encontrar máximos repetidos y seleccionar caminos
    if (__hsub(k_max, X[out_id]) >= thr)
    {
        // printf(
        //     "max - gen (%f - %f) greater than thr (%f)!!\n",
        //     __half2float(k_max),
        //     __half2float(X[out_id]),
        //     __half2float(thr)
        // );
        for (int k = tid; k < K; k += block_size)
        {
            int a_idx = b * M * K + m * K + k;
            int b_idx = b * K * N + k * N + n;

            __half mi = __hmin(X[a_idx], X0[b_idx]);

            if (__hle(__habs(__hsub(mi, k_max)), __half(MIN_DIFF)))
            {
                // printf("%i, %i]: Finded max (%f) !!\n", out_id, tid, __half2float(k_max));
                int idx = atomicAdd(counter, 1);
                paths[idx] = make_int4(b, m, k, n);
                values[idx] = mi;
            }
        }
    }
    __syncthreads();
    if (tid == 0)
        X[out_id] = k_max;
}

// template __global__ void maxmin_threshold_kernel<__half>(
//     const __half *__restrict__ X,  // gen_tensor [B,M,K]
//     const __half *__restrict__ X0, // original_tensor [B,K,N]
//     int4 *__restrict__ paths,      // output paths
//     __half *__restrict__ values,   // min values
//     int *__restrict__ counter,     // atomic counter
//     __half thr,
//     int B, int M, int N, int K);