#include <cassert>
#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>
#include <headers.cuh>

#define TM 8
#define TN 8
#define BK 8
#define BM 64
#define BN 64

// TODO: Investigar transferencia de 2half para optimizar.

// ─────────────────────────────────────────────────────────────────────────────
// maxmin_threshold_kernelv4 — producto max-min con threshold diferencial.
//
// Versión tiled BM×BN estilo GEMM (siboehm). 
// Se cachea en shared memory BM x BK elementos de A, y BN x Bk elementos de B.
// TM  x TN= elementos de c a escribir por hilo.
// BK = steps entre cada iteracion del tiling; tamaño compartido entre As y Bs.
// 
// Dimensiones:
//   M, N, K        → extents LÓGICOS (para escritura densa y counter).
//   Kpad, Npad     → strides FÍSICOS de fila (múltiplos de 32).
//   El factor A es [B, Mpad, Kpad]; B/C/argmax son [B, Mpad, Npad].
//
// Parámetros nullable: argmax, counter.
// ─────────────────────────────────────────────────────────────────────────────


__global__ void maxmin_threshold_kernelv4(
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

    const uint totalResultsBlocktile = BM * BN; // Total de resultados por bloque 
    const uint totalResultsThreadtile = TM * TN; // Total de resultados por hilo  

    const uint numThreadsBlocktile = totalResultsBlocktile / totalResultsThreadtile;  //total de hilos necesarios por bloque
  
    assert(numThreadsBlocktile == blockDim.x);
  

    const int threadCol = threadIdx.x % (BN / TN); // se necesitan BN/TN hilos para cubrir una columna
    const int threadRow = threadIdx.x / (BN / TN);
  
    // Memoria compartida para tile de A y B
    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BN * BK];

    const __half *A_base = A; // esquina global de A (para el threshold)
    
    A += cRow *BM * Kpad ;                 // ubicar el puntero en la fila de A correspondiente (recordar que cRow es el indice del tile, BM el tamaño vertical y Kpad el numero de elementos por fila)
    B += cCol * BN;                        // Lo mismo que en A, solo que no se necesitan saltar elementos ya que las columnas son contiguas
    C += cRow * BM * Npad + cCol * BN;     // ubicar el puntero en el tile de C correspondiente
    argmax += cRow * BM * Npad + cCol * BN;// lo mismo pero para argmax
    
    // indices internos para acceso a SMEM. 
    const uint innerColA = threadIdx.x % (BK / 8); 
    const uint innerRowA = threadIdx.x / (BK / 8);

    const uint innerColB = threadIdx.x % (BN / 8); 
    const uint innerRowB = threadIdx.x / (BN / 8);
    
    // cRow x BM = inicio de la fila de C para el bloque
    const int m0 = cRow * BM + threadRow * TM;

    // cCol x BN = inicio de la columna de C para el bloque
    const int n0 = cCol * BN + threadCol * TN;

    __half max_vals[TM*TN] = {0.0};
    int max_ks[TM*TN] = {0};
    
    // para calculos a nivel thread
    __half regM[TM] = {0.0}; 
    __half regN[TN] = {0.0};  

    for (int bkIdx = 0; bkIdx < Kpad; bkIdx += BK) {
      // FASE 1 — carga cooperativa: cada thread trae elementos de A y de B hasta cubrir BM Y BN.
      float4 tmp =
      reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
      // transpose A during the GMEM to SMEM transfer
        
      
      As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
      
      reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
      __syncthreads();

      __syncthreads();                                     // barrera #1

      A += BK;              // As se desliza a la derecha 
      B += BK*Npad;       // Bs se desliza hacia abajo


      // FASE 2 — cómputo del maxmin parcial, comparando TN elemntos de B con TM  de A
      for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // traer elementos desde SMEM a registros

        for (uint i = 0; i < TM; ++i) {
          regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
        }
        for (uint i = 0; i < TN; ++i) {
          regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }

        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int resIdx = resIdxM * TN + resIdxN;
            __half mi = __hmin(regM[resIdxM], regN[resIdxN]); 
            if (__hgt(mi, max_vals[resIdx])) {
              max_vals[resIdx] = mi;  
              max_ks[resIdx] = bkIdx + dotIdx;
            }
          }
        }
      }
      __syncthreads();                                     // barrera #2
    }

    // Escritura densa sólo para celdas dentro de la matriz LÓGICA.
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        int m = m0 + resIdxM;
        int n = n0 + resIdxN;
        int wi = (threadRow * TM + resIdxM) * Npad + threadCol * TN + resIdxN;
        int resIdx = resIdxM * TN + resIdxN;
        if (m < M && n < N) { 
          C[wi] = max_vals[resIdx];
          if (argmax)
            argmax[wi] = max_ks[resIdx];
          if (counter && __hge(__hsub(max_vals[resIdx], A_base[m * Kpad + n]), thr))
            atomicAdd(counter, 1);
        }
      }  
    }
  }
