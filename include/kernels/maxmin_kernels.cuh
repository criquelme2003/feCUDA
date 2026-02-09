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

// extern template __global__ void maxmin_threshold_kernel<__half>(
//     const __half *__restrict__ X,  // gen_tensor [B,M,K]
//     const __half *__restrict__ X0, // original_tensor [B,K,N]
//     int4 *__restrict__ paths,      // output paths
//     __half *__restrict__ values,   // min values
//     int *__restrict__ counter,     // atomic counter
//     __half thr,
//     int B, int M, int N, int K);

#endif
