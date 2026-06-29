#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>

// ─────────────────────────────────────────────────────────────────────────────
// maxmin_threshold_kernel — producto max-min tileado por columna, con threshold
//
// Calcula C_new[b,m,n] = max_k min(C_old[b,m,k], B_col[b,k])
// donde B_col es la columna n de B, pre-extraída y contigua en memoria.
//
// C_old permanece completa en GPU. Por cada columna n se hace un lanzamiento.
// B_col debe tener layout [B, K] (columna n de B transpuesta a [B, N, K]).
//
// Threshold diferencial: C_new[b,m,n] - C_old[b,m,n] >= thr
//   • step 0: C_old = A_orig
//   • step s: C_old = C_{s-1}
//
// Lanzamiento: dim3 grid(M, B);  dim3 block(128);
//              shmem = 128 * sizeof(__half)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_threshold_kernel(
    const __half *__restrict__ C_old,   // [B, M, K] completa, fija en GPU
    const __half *__restrict__ B_col,   // [B, K]    columna n de B
    __half *__restrict__ C_new_col,     // [B, M]    columna n de salida
    unsigned long long *__restrict__ counter,  // nullable — cuenta efectos >= thr
    __half thr,
    int B, int M, int K,
    int n                               // índice global de la columna actual
) {
    int b   = (int)blockIdx.y;
    int m   = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;

    extern __shared__ char smem_buf[];
    __half *s_val = reinterpret_cast<__half *>(smem_buf);

    __half best_val = __float2half(-FLT_MAX);

    size_t row_off = (size_t)b * M * K + (size_t)m * K;

    for (int k = tid; k < K; k += bsz) {
        __half mi = __hmin(C_old[row_off + k], B_col[(size_t)b * K + k]);
        if (__hgt(mi, best_val))
            best_val = mi;
    }

    s_val[tid] = best_val;
    __syncthreads();

    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s && __hgt(s_val[tid + s], s_val[tid]))
            s_val[tid] = s_val[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        __half val = s_val[0];
        C_new_col[(size_t)b * M + m] = val;
        if (counter && __hge(__hsub(val, C_old[row_off + n]), thr))
            atomicAdd(counter, 1ULL);
    }
}
