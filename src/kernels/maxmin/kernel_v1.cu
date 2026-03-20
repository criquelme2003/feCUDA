#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>

#define MIN_DIFF 0.01f

// ─────────────────────────────────────────────────────────────────────────────
// maxmin_threshold_kernel — producto max-min con threshold diferencial
//
// Calcula C_out[b,m,n] = max_k  min(A_mat[b,m,k], B_mat[b,k,n])
// y emite las aristas (b,m,k,n) donde la mejora sobre A_mat[b,m,n] supera thr.
//
// Threshold:  k_max - A_mat[b,m,n] >= thr
//   • A_mat es C_prev en cada step iterativo.
//   • step 0: A_mat = A_orig → suprime pares con arista directa (semántica paper).
//   • step s>0: A_mat = C_s → suprime pares ya encontrados (unicidad de nivel).
//
// Parámetros nullable (pueden ser nullptr):
//   paths, values, counter → si counter==nullptr no se emiten aristas
//   argmax                 → si nullptr no se guarda el k ganador
//
// shmem = blockDim.x * (sizeof(__half) + sizeof(int))
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_threshold_kernel(
    const __half* __restrict__ A_mat,  // [B,M,K] factor izq. = C_prev
    const __half* __restrict__ B_mat,  // [B,K,N] factor der. = B_orig
    __half*       __restrict__ C_out,  // [B,M,N] siempre se escribe
    int*          __restrict__ paths,  // nullable — flat int[count*4]
    __half*       __restrict__ values, // nullable
    int*          __restrict__ counter,// nullable — atomic counter
    int*          __restrict__ argmax, // nullable — [B,M,N] k ganador
    __half thr,
    int B, int M, int N, int K,
    int batch_id
)
{
    int b   = (batch_id >= 0) ? batch_id : (int)blockIdx.z;
    int m   = (int)blockIdx.y;
    int n   = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;
    int out_id = b * M * N + m * N + n;

    // smem layout: [bsz × __half | bsz × int]
    extern __shared__ char smem_buf[];
    __half* s_val = reinterpret_cast<__half*>(smem_buf);
    int*    s_k   = reinterpret_cast<int*>(smem_buf + bsz * sizeof(__half));

    // ── Reducción local por thread ──────────────────────────────────────────
    __half best_val = __float2half(-FLT_MAX);
    int    best_k   = 0;

    for (int k = tid; k < K; k += bsz)
    {
        int a_idx = b * M * K + m * K + k;
        int b_idx = b * K * N + k * N + n;
        __half mi = __hmin(A_mat[a_idx], B_mat[b_idx]);
        if (__hgt(mi, best_val)) { best_val = mi; best_k = k; }
    }

    s_val[tid] = best_val;
    s_k[tid]   = best_k;
    __syncthreads();

    // ── Reducción en shared memory con tracking de argmax ───────────────────
    for (int s = bsz / 2; s > 0; s >>= 1)
    {
        if (tid < s && __hgt(s_val[tid + s], s_val[tid]))
        {
            s_val[tid] = s_val[tid + s];
            s_k[tid]   = s_k[tid + s];
        }
        __syncthreads();
    }

    __half k_max = s_val[0];

    // ── Escritura densa (siempre) ────────────────────────────────────────────
    if (tid == 0)
    {
        C_out[out_id] = k_max;
        if (argmax) argmax[out_id] = s_k[0];
    }
    __syncthreads();

    // ── Emisión de aristas (solo si counter != nullptr) ─────────────────────
    // Threshold diferencial: k_max - A_mat[m,n] >= thr
    if (counter && __hsub(k_max, A_mat[out_id]) >= thr)
    {
        for (int k = tid; k < K; k += bsz)
        {
            int a_idx = b * M * K + m * K + k;
            int b_idx = b * K * N + k * N + n;
            __half mi = __hmin(A_mat[a_idx], B_mat[b_idx]);

            if (__hle(__habs(__hsub(mi, k_max)), __float2half(MIN_DIFF)))
            {
                int idx  = atomicAdd(counter, 1);
                int base = idx * 4;
                paths[base + 0] = b;
                paths[base + 1] = m;
                paths[base + 2] = k;
                paths[base + 3] = n;
                values[idx] = mi;
            }
        }
    }
}
