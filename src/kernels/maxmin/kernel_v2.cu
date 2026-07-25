#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>
#include <headers.cuh>

// ─────────────────────────────────────────────────────────────────────────────
// maxmin_threshold_kernelv2 — producto max-min con threshold diferencial.
//
// Versión tiled 32×32 estilo GEMM (siboehm). ASUME que A, B, C y argmax están
// padeados a múltiplo de 32 en M, N y K, con la región de padding rellena de un
// valor negativo (neutro para max, dado que las entradas son ≥ 0). Bajo esa
// premisa el hot loop no necesita guardas de OOB: los strides físicos (Kpad para
// A; Npad para B/C/argmax) cubren siempre buffer válido.
//
// Dimensiones:
//   M, N, K        → extents LÓGICOS (para escritura densa y counter).
//   Kpad, Npad     → strides FÍSICOS de fila (múltiplos de 32).
//   El factor A es [B, Mpad, Kpad]; B/C/argmax son [B, Mpad, Npad].
//
// Parámetros nullable: argmax, counter.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_threshold_kernelv2(
    const __half *__restrict__ A, // [B,Mpad,Kpad] factor izq. = C_prev
    const __half *__restrict__ B, // [B,Kpad,Npad] factor der. = B_orig [FIJA]
    __half *__restrict__ C,       // [B,Mpad,Npad] siempre se escribe
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
) {

    // Tile de C que calcula este bloque.
    const uint cRow = blockIdx.x;   // sobre filas (M)
    const uint cCol = blockIdx.y;   // sobre columnas (N)

    // Memoria compartida para el tile 32×32.
    __shared__ __half As[32 * 32];
    __shared__ __half Bs[32 * 32];

    // Fila y columna del hilo dentro del tile.
    const uint threadRow = threadIdx.x / 32;
    const uint threadCol = threadIdx.x % 32;

    const __half *A_base = A; // esquina global de A (para el threshold)

    // Desplazamiento a la esquina de este tile usando strides FÍSICOS.
    A += cRow * 32 * Kpad;                 // 32 filas de A (stride Kpad)
    B += cCol * 32;                        // 32 columnas de B
    C += cRow * 32 * Npad + cCol * 32;     // tile de C
    argmax += cRow * 32 * Npad + cCol * 32;

    __half max_val = __float2half(-FLT_MAX);
    int k_max = 0;

    const int m = cRow * 32 + threadRow;
    const int n = cCol * 32 + threadCol;

    // Kpad es múltiplo de 32 ⇒ el loop divide exacto; las filas/columnas de
    // padding valen negativo y nunca ganan el max.
    for (int bkIdx = 0; bkIdx < Kpad; bkIdx += 32) {
      // FASE 1 — carga cooperativa: cada thread trae UN half de A y UNO de B.
      As[threadRow * 32 + threadCol] = A[threadRow * Kpad + threadCol];
      Bs[threadRow * 32 + threadCol] = B[threadRow * Npad + threadCol];
      __syncthreads();                                     // barrera #1

      A += 32;              // As se desliza a la derecha
      B += 32 * Npad;       // Bs se desliza hacia abajo

      // FASE 2 — cómputo del maxmin parcial, 100% desde SMEM.
      for (int dotIdx = 0; dotIdx < 32; ++dotIdx) {
        __half mi = __hmin(As[threadRow * 32 + dotIdx], Bs[dotIdx * 32 + threadCol]);
        if (__hgt(mi, max_val)) {
          max_val = mi;
          k_max = bkIdx + dotIdx;
        }
      }
      __syncthreads();                                     // barrera #2
    }

    // Escritura densa sólo para celdas dentro de la matriz LÓGICA.
    if (m < M && n < N) {
      C[threadRow * Npad + threadCol] = max_val;
      if (argmax)
        argmax[threadRow * Npad + threadCol] = k_max;
      if (counter && __hge(__hsub(max_val, A_base[m * Kpad + n]), thr))
        atomicAdd(counter, 1);
    }
  }
