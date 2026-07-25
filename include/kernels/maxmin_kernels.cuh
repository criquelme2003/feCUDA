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
 *   size_t shmem = K * sizeof(__half) + 128 * (sizeof(__half) + sizeof(int));
 *   (K * sizeof(__half) cachea la columna B_mat[b,:,n], compartida por los M
 *   bloques de esa columna, evitando relecturas repetidas desde L2.)
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

// Versión tiled 32×32 (estilo GEMM). Requiere buffers padeados a múltiplo de 32
// en M, N y K, con el padding relleno de un valor negativo. M/N/K son extents
// lógicos; Kpad/Npad son los strides físicos de fila.
__global__ void maxmin_threshold_kernelv2(
    const __half *__restrict__ A, // [B,Mpad,Kpad] factor izq. = C_prev
    const __half *__restrict__ B, // [B,Kpad,Npad] factor der. = B_orig
    __half *__restrict__ C,       // [B,Mpad,Npad] resultado
    int *__restrict__ argmax,     // nullable — [B,Mpad,Npad] k ganador
    int *__restrict__ counter,    // nullable — cuenta celdas con efecto >= thr
    __half thr,
    int numBatches,
    int M,
    int N,
    int K,
    int Kpad,
    int Npad,
    int batch_id
);

#endif
