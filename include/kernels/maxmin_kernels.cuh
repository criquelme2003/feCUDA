#ifndef MAXMIN_KERNELS_CUH
#define MAXMIN_KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ─────────────────────────────────────────────────────────────────────────────
// maxmin_threshold_kernel — producto max-min tileado por columna
//
// C_new[b,m,n] = max_k min(C_old[b,m,k], B_col[b,k])
//
// B_col es la columna n de B, pre-extraída con layout [B, K].
// Para obtenerla contigua, B debe estar transpuesta a [B, N, K] en host:
//   cudaMemcpy(d_B_col, &h_B_T[n * K], B * K * sizeof(__half), H2D);
//
// Threshold diferencial:  C_new[b,m,n] - C_old[b,m,n] >= thr
//
// Lanzamiento por columna n:
//   dim3 grid(M, B);  dim3 block(128);
//   size_t shmem = 128 * sizeof(__half);
//   maxmin_threshold_kernel<<<grid, block, shmem>>>(
//       d_C_old, d_B_col, d_C_new_col, d_counter, thr, B, M, K, n);
//   // copiar columna al host: cudaMemcpy(&h_C_new[n*M], d_C_new_col, B*M*sizeof(__half), D2H)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_threshold_kernel(
    const __half *__restrict__ C_old,   // [B, M, K] completa en GPU
    const __half *__restrict__ B_col,   // [B, K]    columna n de B
    __half *__restrict__ C_new_col,     // [B, M]    columna n de salida
    unsigned long long *__restrict__ counter,          // nullable
    __half thr,
    int B, int M, int K,
    int n
);

#endif
