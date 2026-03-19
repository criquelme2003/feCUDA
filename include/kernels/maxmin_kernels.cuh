#ifndef MAXMIN_KERNELS_CUH
#define MAXMIN_KERNELS_CUH

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * KERNEL MAXMIN OPTIMIZADO CON MEMORIA COMPARTIDA
 *
 * Este kernel calcula simultáneamente los valores mínimos y máximos
 * de la operación maxmin usando memoria compartida para la reducción.
 *
 * Configuración de lanzamiento recomendada:
 * - Bloques 3D: dim3(N, M, batch_size)
 * - Threads 1D: dim3(K)
 * - Memoria compartida: K * sizeof(float)
 */

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
);

/**
 * KERNEL DE UN PASO DEL PRODUCTO MAX-MIN ITERATIVO (orden > 1)
 *
 * Calcula C_next[b,m,n] = max_k  min(C_prev[b,m,k], B_mat[b,k,n])
 * y guarda en argmax[b,m,n] el índice k que alcanzó ese máximo.
 *
 * Usado para construir caminos de orden p de forma iterativa:
 *   C_1 = A ⊗ B,  C_2 = C_1 ⊗ B,  ...,  C_p = C_{p-1} ⊗ B
 *
 * Lanzamiento: dim3 grid(N, M, B), dim3 block(128)
 * smem = 128 * (sizeof(__half) + sizeof(int))
 */
__global__ void maxmin_step_kernel(
    const __half* __restrict__ C_prev,  // [B, M, K]
    const __half* __restrict__ B_mat,   // [B, K, N]
    __half* __restrict__ C_next,        // [B, M, N]  salida: nuevo producto
    int*   __restrict__ argmax,         // [B, M, N]  salida: k ganador
    int B,
    int M,
    int N,
    int K,
    int batch_id
);

// extern template __global__ void maxmin_threshold_kernel<__half>(
//     const __half *__restrict__ X,  // gen_tensor [B,M,K]
//     const __half *__restrict__ X0, // original_tensor [B,K,N]
//     int4 *__restrict__ paths,      // output paths
//     __half *__restrict__ values,   // min values
//     int *__restrict__ counter,     // atomic counter
//     __half thr,
//     int B, int M, int N, int K);

#endif
