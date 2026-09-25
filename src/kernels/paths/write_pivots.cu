#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>
#include <headers.cuh>
  
// ─────────────────────────────────────────────────────────────────────────────
// maxmin_write_all_pivots — segunda pasada: escribe el CSR de testigos.
//
// No recalcula el máximo: C[m,n] ya trae el max-min fresco de esta iteración
// (count_pivots.cu lo escribe incondicionalmente para toda celda lógica, así
// que nunca hay valores reciclados de un swap anterior). Este kernel solo
// busca, para cada celda activa, las posiciones k donde
// min(A[m,k], B[k,n]) == C[m,n] — sin volver a hallar el máximo ni reevaluar
// thr (eso ya lo decidió count_pivots.cu vía pivots_detailed_count).
//
// "Activa" = pivots_detailed_count[m,n] > 0 (esa celda pasó thr en la pasada
// de conteo). Las celdas inactivas no tocan `pivots` — su acc_offset puede
// coincidir con el de la celda vecina activa, y escribir ahí corrompería su
// rango del CSR.
//
// Versión tiled ×BLOCKSIZE estilo GEMM (siboehm). ASUME que A, B, C están
// padeados a múltiplo de BLOCKSIZE en M, N y K, con la región de padding
// rellena de un valor negativo (neutro para max). El hot loop no necesita
// guardas de OOB adicionales: los strides físicos (Kpad para A; Npad para
// B/C) cubren siempre buffer válido.
//
// Dimensiones:
//   M, N, K     → extents LÓGICOS.
//   Kpad, Npad  → strides FÍSICOS de fila (múltiplos de BLOCKSIZE).
//   A es [B, Mpad, Kpad]; B/C son [B, Mpad, Npad].
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_write_all_pivots(
    const __half *__restrict__ A, // [B,Mpad,Kpad] factor izq. = C_prev
    const __half *__restrict__ B, // [B,Kpad,Npad] factor der. = B_orig [FIJA]
    const __half *__restrict__ C, // [B,Mpad,Npad] max-min ya calculado (esta iteración)
    const int *__restrict__ pivots_detailed_count,    // conteo por celda (<=Tmax), de count_pivots.cu
    const int *__restrict__ acc_pivots_detailed_count, // offset exclusivo (scan) por celda — CSR row_offsets
    int *__restrict__ pivots, // OUT — CSR plano de testigos k, tamaño = total de count_pivots
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

    // Desplazamiento a la esquina de este tile usando strides FÍSICOS.
    A += cRow * BLOCKSIZE * Kpad; // BLOCKSIZE filas de A (stride Kpad)
    B += cCol * BLOCKSIZE;        // BLOCKSIZE columnas de B

    const int m = cRow * BLOCKSIZE + threadRow;
    const int n = cCol * BLOCKSIZE + threadCol;
    const int cell = threadRow * Npad + threadCol; // índice físico de esta celda en C/pivots_detailed_count

    // target_max y active se resuelven UNA vez, antes del hot loop: ya se
    // conocen de la pasada de conteo, no hay nada que recalcular aquí.
    __half target_max = __float2half(-FLT_MAX);
    bool active = false;
    int base_offset = 0;
    if (m < M && n < N) {
        int cnt = pivots_detailed_count[cell];
        active = cnt > 0;
        if (active) {
            target_max  = C[cell];
            base_offset = acc_pivots_detailed_count[cell];
        }
    }
    int slot = 0; // siguiente posición libre en pivots[base_offset + slot], crece 0..cnt-1

    // Kpad es múltiplo de BLOCKSIZE ⇒ el loop divide exacto; las filas/columnas de
    // padding valen negativo y nunca igualan target_max.
    for (int bkIdx = 0; bkIdx < Kpad; bkIdx += BLOCKSIZE) {

        // FASE 1 — carga cooperativa: TODO el bloque participa, activo o no
        // (la carga es compartida vía __syncthreads(), no se puede saltar).
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * Kpad + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * Npad + threadCol];
        __syncthreads(); // barrera #1

        A += BLOCKSIZE;        // As se desliza a la derecha
        B += BLOCKSIZE * Npad; // Bs se desliza hacia abajo

        // FASE 2 — sólo celdas activas buscan coincidencias con target_max.
        if (active) {
            for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
                __half mi = __hmin(
                    As[threadRow * BLOCKSIZE + dotIdx],
                    Bs[dotIdx * BLOCKSIZE + threadCol]
                );
                int actual_k = bkIdx + dotIdx;
                if (actual_k < K && slot < Tmax && __heq(mi, target_max)) {
                    pivots[base_offset + slot] = actual_k;
                    slot++;
                }
            }
        }
        __syncthreads(); // barrera #2
    }
}
