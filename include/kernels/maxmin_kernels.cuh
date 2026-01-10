#ifndef MAXMIN_KERNELS_CUH
#define MAXMIN_KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>

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
    const float *A, // [batch, M, K]
    const float *B, // [batch, K, N]
    float *C_min,   // [batch, M, K, N]
    float *C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size);

template <typename T, int WARPS_PER_BLOCK>
__global__ void cub_max_min_kernel(
    const T *__restrict__ A, // [batch, M, K]
    const T *__restrict__ B, // [batch, K, N]
    T *__restrict__ C_min,   // [batch, M, K, N]
    T *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size);

extern template __global__ void cub_max_min_kernel<__half,4>(
    const __half*, // [batch, M, K]
    const __half*, // [batch, M, K]
    __half*,       // [batch, M, K]
    __half*,       // [batch, M, K]
    const int, const int, const int, const int);

extern template __global__ void cub_max_min_kernel<float,4>(
    const float*, // [batch, M, K]
    const float*, // [batch, M, K]
    float*,       // [batch, M, K]
    float*,       // [batch, M, K]
    const int, const int, const int, const int);

#endif 
