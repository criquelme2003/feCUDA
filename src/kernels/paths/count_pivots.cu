#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>
#include <headers.cuh>

__device__ bool half_equal_with_margin(__half a, __half b, __half epsilon) {
    // |a - b| <= epsilon
    return __habs(a - b) <= epsilon;
}

// ─────────────────────────────────────────────────────────────────────────────
// maxmin_threshold_kernelv2 — producto max-min con threshold diferencial.
//
// Versión tiled ×BLOCKSIZE estilo GEMM (siboehm). ASUME que A, B, C y argmax están
// padeados a múltiplo de BLOCKSIZE en M, N y K, con la región de padding rellena de un
// valor negativo (neutro para max, dado que las entradas son ≥ 0). Bajo esa
// premisa el hot loop no necesita guardas de OOB: los strides físicos (Kpad para
// A; Npad para B/C/argmax) cubren siempre buffer válido.
//
// Dimensiones:
//   M, N, K         → extents LÓGICOS (para escritura densa y counter).
//   Kpad, Npad     → strides FÍSICOS de fila (múltiplos de BLOCKSIZE).
//   El factor A es [B, Mpad, Kpad]; B/C/argmax son [B, Mpad, Npad].
//
// Parámetros nullable: argmax, counter.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_all_pivots(
    const __half *__restrict__ A, // [B,Mpad,Kpad] factor izq. = C_prev
    const __half *__restrict__ B, // [B,Kpad,Npad] factor der. = B_orig [FIJA]
    __half *__restrict__ C, // [B,Kpad,Npad] factor der. = B_orig [FIJA]
    int *__restrict__ pivots,     // cuenta los pivots por cada C (<=Tmax)
    unsigned long long *__restrict__ counter,    // nullable — cuenta celdas con efecto >= thr
    __half thr,
    int numBatches,
    int M,
    int N,
    int K,
    int Kpad,
    int Npad,
    int batch_id
) {

    // Tile de C que calcula este bloque.
    const uint cRow = blockIdx.x; // sobre filas (M)
    const uint cCol = blockIdx.y; // sobre columnas (N)

    // Memoria compartida para el tile BLOCKSIZE×BLOCKSIZE.
    __shared__ __half As[BLOCKSIZE * BLOCKSIZE];
    __shared__ __half Bs[BLOCKSIZE * BLOCKSIZE];

    // Fila y columna del hilo dentro del tile.
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    const __half *A_base = A; // esquina global de A (para el threshold)

    // Desplazamiento a la esquina de este tile usando strides FÍSICOS.
    A += cRow * BLOCKSIZE * Kpad; // BLOCKSIZE filas de A (stride Kpad)
    B += cCol * BLOCKSIZE;        // BLOCKSIZE columnas de B
    C += cRow * BLOCKSIZE * Npad + cCol * BLOCKSIZE; // tile de C

    __half max_val = __float2half(-FLT_MAX);
    int k_max = 0;

    const int m = cRow * BLOCKSIZE + threadRow;
    const int n = cCol * BLOCKSIZE + threadCol;

    // Kpad es múltiplo de BLOCKSIZE ⇒ el loop divide exacto; las filas/columnas de
    // padding valen negativo y nunca ganan el max.

    int local_max_count = 0;
    for (int bkIdx = 0; bkIdx < Kpad; bkIdx += BLOCKSIZE) {

        // FASE 1 — carga cooperativa: cada thread trae UN half de A y UNO de B.
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * Kpad + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * Npad + threadCol];
        __syncthreads(); // barrera #1

        A += BLOCKSIZE;        // As se desliza a la derecha
        B += BLOCKSIZE * Npad; // Bs se desliza hacia abajo

        // FASE 2 — cómputo del maxmin parcial, 100% desde SMEM.
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            __half mi = __hmin(
                As[threadRow * BLOCKSIZE + dotIdx],
                Bs[dotIdx * BLOCKSIZE + threadCol]
            );
            if (__hgt(mi, max_val)) {
                max_val = mi;
                k_max = bkIdx + dotIdx;
                local_max_count = 1;
            } else if (__heq(mi, max_val)) {
                if(m < M && n < N && bkIdx + dotIdx < K){
                  int actual_k = bkIdx + dotIdx;
                  if (local_max_count < Tmax) {
                      local_max_count += 1;
                  }
                }
                
            }
        }
        __syncthreads(); // barrera #2
    }

    // Escritura densa sólo para celdas dentro de la matriz LÓGICA.
    if (m < M && n < N) {
        C[threadRow * Npad + threadCol] = max_val;
        // write paths
        if (counter && __hge(__hsub(max_val, A_base[m * Kpad + n]), thr)){

          pivots[threadRow * Npad + threadCol] = local_max_count;
          atomicAdd(counter,(unsigned long long) local_max_count);
        }
    }
}
