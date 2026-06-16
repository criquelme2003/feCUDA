#ifndef MAXMIN_KERNELS_CUH
#define MAXMIN_KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * KERNEL MAXMIN CON THRESHOLD DIFERENCIAL
 *
 * Calcula C_out[b,m,n] = max_k min(A_mat[b,m,k], B_mat[b,k,n])
 * y emite aristas donde k_max - A_mat[b,m,n] >= thr.
 *
 * Threshold diferencial:
 *   • Step 0: A_mat = A_orig → suprime pares con arista directa.
 *   • Step s: A_mat = C_s   → suprime pares ya encontrados (unicidad de nivel).
 *
 * Parámetros nullable: paths, values, counter, argmax.
 *   Si counter == nullptr → no se emiten aristas.
 *   Si argmax  == nullptr → no se guarda el k ganador.
 *
 * Lanzamiento recomendado:
 *   dim3 grid(N, M, B);  dim3 block(128);
 *   size_t shmem = 128 * (sizeof(__half) + sizeof(int));
 */
__global__ void maxmin_threshold_kernel(
    const __half *__restrict__ A_mat, // [B,M,K] factor izq. y referencia threshold
    const __half *__restrict__ B_mat, // [B,K,N] factor derecho
    __half *__restrict__ C_out,       // [B,M,N] resultado (siempre se escribe)
    int *__restrict__ argmax,         // nullable — [B,M,N] k ganador por celda
    int *__restrict__ counter,        // nullable — cuenta celdas con efecto >= thr
    __half thr,
    int B,
    int M,
    int N,
    int K,
    int batch_id
);

#endif
