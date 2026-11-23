#include <cassert>
#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>
#include <headers.cuh>

#define TM 8
#define BK 8
#define BM 64
#define BN 64


// ─────────────────────────────────────────────────────────────────────────────
// maxmin_threshold_kernelv3 — producto max-min con threshold diferencial.
//
// Versión tiled BM×BN estilo GEMM (siboehm). 
// Se cachea en shared memory BM x BK elementos de A, y BN x Bk elementos de B.
// TM = elementos de c a escribir por hilo. 
// BK = steps entre cada iteracion del tiling; tamaño compartido entre As y Bs.
// 
// Dimensiones:
//   M, N, K        → extents LÓGICOS (para escritura densa y counter).
//   Kpad, Npad     → strides FÍSICOS de fila (múltiplos de 32).
//   El factor A es [B, Mpad, Kpad]; B/C/argmax son [B, Mpad, Npad].
//
// Parámetros nullable: argmax, counter.
// ─────────────────────────────────────────────────────────────────────────────


__global__ void maxmin_threshold_kernelv3(
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
    const uint cRow = blockIdx.y;   // sobre filas (M)  
    const uint cCol = blockIdx.x;   // sobre columnas (N)

    // Memoria compartida para tile de A y B
    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BN * BK];

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;
  

    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;
  


    const __half *A_base = A; // esquina global de A (para el threshold)

    // Desplazamiento a la esquina de este tile usando strides FÍSICOS.
    A += cRow * Kpad *BM;                 // BM filas de A (stride Kpad)
    B += cCol * BN;                        // BN columnas de B
    C += cRow * BM * Npad + cCol * BN;     // tile de C
    argmax += cRow * BM * Npad + cCol * BN;


    const int m0 = cRow * BM + threadRow * TM;
    const int n = cCol * BN + threadCol;

    // Kpad es múltiplo de 32 ⇒ el loop divide exacto; las filas/columnas de
    // padding valen negativo y nunca ganan el max.
    
    
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    __half max_vals[TM] = {0.0};
    int max_ks[TM] = {0};
    for (int bkIdx = 0; bkIdx < Kpad; bkIdx += BK) {
      // FASE 1 — carga cooperativa: cada thread trae UN half de A y UNO de B.
      As[innerRowA * BK + innerColA] = A[innerRowA * Kpad + innerColA];
      Bs[innerRowB * BN + innerColB] = B[innerRowB * Npad + innerColB];
      __syncthreads();                                     // barrera #1

      A += BK;              // As se desliza a la derecha
      B += BK*Npad;       // Bs se desliza hacia abajo


      // FASE 2 — cómputo del maxmin parcial, comparando 1 elemento de B con TM de A.
      for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
        
        __half Btmp = Bs[dotIdx * BN + threadCol]; // se guarda 1 elemento de Bs en registros   
        for (int  resIdx = 0; resIdx < TM; ++resIdx){
          __half mi = __hmin(As[(threadRow * TM + resIdx)*BK + dotIdx], Btmp); 
          if (__hgt(mi, max_vals[resIdx])) {
            max_vals[resIdx] = mi;  
            max_ks[resIdx] = bkIdx + dotIdx;
          }
        }
      }
      __syncthreads();                                     // barrera #2
    }

    // Escritura densa sólo para celdas dentro de la matriz LÓGICA.
    for(int resIdx = 0; resIdx < TM ; ++resIdx){
      int m = m0 + resIdx;
      if (m < M && n < N) { 
        C[(threadRow * TM + resIdx) * Npad + threadCol] = max_vals[resIdx];
        if (argmax)
          argmax[(threadRow * TM + resIdx) * Npad + threadCol] = max_ks[resIdx];
        if (counter && __hge(__hsub(max_vals[resIdx], A_base[m * Kpad + n]), thr))
          atomicAdd(counter, 1);
      }
    }
  }
