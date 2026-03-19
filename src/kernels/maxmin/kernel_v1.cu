#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <float.h>

#define MIN_DIFF 0.01
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

    __half v = __float2half(-FLT_MAX);

    // Grid-stride: cada thread procesa múltiples K, aplicando max instantaneamene para controlar K.
    // IMPORTANTE: leer de X0 (snapshot original, const) para evitar race condition.
    // X[b,m,k] es escrito por otros bloques concurrentes (tid==0 escribe X[out_id]=k_max).
    // Si leyéramos de X veríamos valores ya modificados por otros bloques del mismo m.
    for (int k = tid; k < K; k += block_size)
    {
        int a_idx = b * M * K + m * K + k;
        int b_idx = b * K * N + k * N + n;
        v = __hmax(v, __hmin(X0[a_idx], X0[b_idx]));
    }

    smem[tid] = v;
    __syncthreads();

    // Reducción max
    for (int s = (block_size / 2) ; s > 0; s >>= 1)
    {
        if (tid < s)
            smem[tid] = __hmax(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    __half k_max = smem[0];

    __syncthreads();

    // Encontrar máximos repetidos y seleccionar caminos
    // Comparar contra X0[out_id] (valor original), no X[out_id] (puede estar modificado)
    if (__hsub(k_max, X0[out_id]) >= thr)
    {
        for (int k = tid; k < K; k += block_size)
        {
            int a_idx = b * M * K + m * K + k;
            int b_idx = b * K * N + k * N + n;

            __half mi = __hmin(X0[a_idx], X0[b_idx]);

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

// ─────────────────────────────────────────────────────────────────────────────
// Kernel auxiliar para órdenes > 1:
//   C_next[b,m,n] = max_k  min(C_prev[b,m,k], B_mat[b,k,n])
//   argmax[b,m,n] = k que alcanzó ese máximo
//
// smem layout: [ bsz × __half | bsz × int ]
//   tamaño = blockDim.x * (sizeof(__half) + sizeof(int))
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_step_kernel(
    const __half* __restrict__ C_prev,
    const __half* __restrict__ B_mat,
    __half* __restrict__ C_next,
    int*   __restrict__ argmax,
    int B, int M, int N, int K, int batch_id
)
{
    int b   = (batch_id >= 0) ? batch_id : (int)blockIdx.z;
    int m   = blockIdx.y;
    int n   = blockIdx.x;
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    extern __shared__ char smem_step[];
    __half* s_val = reinterpret_cast<__half*>(smem_step);
    int*    s_k   = reinterpret_cast<int*>(smem_step + bsz * sizeof(__half));

    __half best_val = __float2half(-FLT_MAX);
    int    best_k   = 0;

    for (int k = tid; k < K; k += bsz)
    {
        __half a_val = C_prev[b * M * K + m * K + k];
        __half b_val = B_mat [b * K * N + k * N + n];
        __half mi    = __hmin(a_val, b_val);
        if (__hgt(mi, best_val)) { best_val = mi; best_k = k; }
    }

    s_val[tid] = best_val;
    s_k[tid]   = best_k;
    __syncthreads();

    // Reducción max con tracking de argmax
    for (int s = bsz / 2; s > 0; s >>= 1)
    {
        if (tid < s && __hgt(s_val[tid + s], s_val[tid]))
        {
            s_val[tid] = s_val[tid + s];
            s_k[tid]   = s_k[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        int idx      = b * M * N + m * N + n;
        C_next[idx]  = s_val[0];
        argmax[idx]  = s_k[0];
    }
}