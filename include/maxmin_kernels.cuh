#ifndef MAXMIN_KERNELS_CUH
#define MAXMIN_KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "types.cuh"

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

__global__ void max_min_kernel(
    const float * A, // [batch, M, K]
    const float * B, // [batch, K, N]
    float * C_min,   // [batch, M, K, N]
    float * C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size);

__global__ void max_min_kernel_v2(
    const float * A, // [batch, M, K]
    const float * B, // [batch, K, N]
    float * C_min,   // [batch, M, K, N]
    float * C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size);

__global__ void max_min_lineal_kernel(
    const float * A, // [batch, M, K]
    const float * B, // [batch, K, N]
    float * C_min,   // [batch, M, K, N]
    float * C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size);

#endif // MAXMIN_KERNELS_CUH
