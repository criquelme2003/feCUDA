
#pragma once
#include <cuda_fp16.h>
#include <float.h>
#include <cuda_runtime.h>

/* ─── Constantes de tile (ajustar según GPU) ─────────────────────────────── */
#define BM_TILE      8      // filas de C cubiertas por bloque
#define BN_TILE      8      // columnas de C cubiertas por bloque
#define BK_TILE      64     // profundidad del tile en K
#define WARPS_PER_BLOCK 4   // → 128 threads por bloque (4 warps)
#define WARP_SIZE    32
#define BLOCK_THREADS (WARP_SIZE * WARPS_PER_BLOCK)

/* ─── Helpers ────────────────────────────────────────────────────────────── */
#ifndef MIN_DIFF
#define MIN_DIFF 1e-3f
#endif

__device__ __forceinline__
__half hmax_(__half a, __half b) { return __hgt(a, b) ? a : b; }


/* ─── Kernel principal ───────────────────────────────────────────────────── */
/*
 * smem layout (en orden):
 *   [BM_TILE * BK_TILE × half]   ← tile de A  (filas contiguas)
 *   [BN_TILE * BK_TILE × half]   ← tile de B  (cargado transpuesto: dim [K][N] → smem [BN][BK])
 *   [BLOCK_THREADS × half]       ← valores para reducción inter-warp
 *   [BLOCK_THREADS × int]        ← índices  para reducción inter-warp
 */
__global__ void __launch_bounds__(BLOCK_THREADS, 2)
maxmin_tile_kernel(
    const __half* __restrict__ A_mat,   // [B, M, K]
    const __half* __restrict__ B_mat,   // [B, K, N]
    __half*       __restrict__ C_out,   // [B, M, N]
    int*          __restrict__ paths,
    __half*       __restrict__ values,
    int*          __restrict__ counter,
    int*          __restrict__ argmax,
    __half thr,
    int B, int M, int N, int K,
    int batch_id,
    int max_paths
)
{
    /* ── Coordenadas del tile ─────────────────────────────────────────────── */
    int b  = (batch_id >= 0) ? batch_id : (int)blockIdx.z;
    int m0 = (int)blockIdx.y * BM_TILE;   // primera fila del tile
    int n0 = (int)blockIdx.x * BN_TILE;   // primera columna del tile

    int tid   = (int)threadIdx.x;
    int warp  = tid / WARP_SIZE;
    int lane  = tid & (WARP_SIZE - 1);

    /* ── smem ────────────────────────────────────────────────────────────── */
    extern __shared__ char smem_buf[];
    __half* s_A   = reinterpret_cast<__half*>(smem_buf);                         // [BM * BK]
    __half* s_B   = s_A   + BM_TILE * BK_TILE;                                   // [BN * BK]
    __half* s_val = s_B   + BN_TILE * BK_TILE;                                   // [BLOCK_THREADS]
    int*    s_k   = reinterpret_cast<int*>(s_val + BLOCK_THREADS);               // [BLOCK_THREADS]

    /* ── Registros de acumulación por celda ──────────────────────────────── */
    /* Cada thread es responsable de una celda (m_local, n_local) del tile.
     * Con 128 threads y tile 8×8=64 celdas, usamos los primeros 64 threads
     * como "owner" de cada celda; los restantes ayudan sólo en la carga. */
    const int CELLS = BM_TILE * BN_TILE;   // 64
    int cell      = tid;                    // thread i → celda i (si tid < CELLS)
    int m_local   = (tid < CELLS) ? (cell / BN_TILE) : 0;
    int n_local   = (tid < CELLS) ? (cell % BN_TILE) : 0;
    int m_global  = m0 + m_local;
    int n_global  = n0 + n_local;
    bool valid    = (tid < CELLS) && (m_global < M) && (n_global < N);

    __half best_val[CELLS / BLOCK_THREADS + 1];   // 0 o 1 por thread si CELLS<=BLOCK_THREADS
    int    best_k  [CELLS / BLOCK_THREADS + 1];
    /* Para simplificar: cada thread owner guarda exactamente 1 celda */
    __half my_val = __float2half(-FLT_MAX);
    int    my_k   = 0;

    int bA = b * M * K;
    int bB = b * K * N;

    /* ── Loop principal sobre K en tiles de BK ───────────────────────────── */
    for (int k_tile = 0; k_tile < K; k_tile += BK_TILE)
    {
        int k_end = min(k_tile + BK_TILE, K);
        int k_len = k_end - k_tile;

        /* ── Carga de tile A [BM, BK] — coalescente ──────────────────────
         * A[b, m, k]: cada fila m tiene K elementos contiguos.
         * Cargamos BM filas × k_len columnas.
         * Thread tid carga elemento (tid/k_len, tid%k_len). */
        for (int i = tid; i < BM_TILE * BK_TILE; i += BLOCK_THREADS) {
            int mi = i / BK_TILE;
            int ki = i % BK_TILE;
            int mg = m0 + mi;
            int kg = k_tile + ki;
            s_A[i] = (mg < M && kg < K)
                     ? A_mat[bA + mg * K + kg]
                     : __float2half(-FLT_MAX);
        }

        /* ── Carga de tile B [BN, BK] — coalescente en memoria global ────
         * B[b, k, n]: fila k tiene N elementos contiguos.
         * Queremos s_B[ni][ki] = B_mat[bB + (k_tile+ki)*N + (n0+ni)]
         * Cargamos recorriendo (ki, ni) en orden mayor de ki → menor stride.
         * Dentro de un warp, threads consecutivos acceden a n0..n0+BN-1
         * → stride 1 → coalescente. */
        for (int i = tid; i < BK_TILE * BN_TILE; i += BLOCK_THREADS) {
            int ki = i / BN_TILE;
            int ni = i % BN_TILE;
            int kg = k_tile + ki;
            int ng = n0 + ni;
            /* s_B[ni * BK + ki] → acceso en smem por columna de B */
            s_B[ni * BK_TILE + ki] = (kg < K && ng < N)
                                     ? B_mat[bB + kg * N + ng]
                                     : __float2half(-FLT_MAX);
        }

        __syncthreads();

        /* ── Cómputo — cada thread owner recorre el tile K ───────────────── */
        if (valid) {
            const __half* a_row = s_A + m_local * BK_TILE;   // A[m_local, *]
            const __half* b_col = s_B + n_local * BK_TILE;   // B[*, n_local] en smem

            #pragma unroll 8 // 
            for (int ki = 0; ki < k_len; ki++) {
                __half mi = __hmin(a_row[ki], b_col[ki]);
                if (__hgt(mi, my_val)) { my_val = mi; my_k = k_tile + ki; }
            }
        }

        __syncthreads();
    }

    /* ── Escritura densa ─────────────────────────────────────────────────── */
    if (valid) {
        int out_id = b * M * N + m_global * N + n_global;
        C_out[out_id] = my_val;
        if (argmax) argmax[out_id] = my_k;

        /* ── Emisión de aristas (segunda pasada, sólo si se pide) ──────── */
        if (counter) {
            /* Umbral diferencial: k_max - A[m,n] (elemento diagonal) >= thr */
            __half diag = (m_global < K && n_global < N)
                          ? A_mat[bA + m_global * K + n_global]  /* A cuadrada */
                          : __float2half(0.f);
            if (__hsub(my_val, diag) >= thr) {
                /* Segunda pasada sobre K para emitir todas las aristas óptimas */
                for (int k = 0; k < K; k++) {
                    __half a_v = A_mat[bA + m_global * K + k];
                    __half b_v = B_mat[bB + k * N + n_global];
                    __half mi  = __hmin(a_v, b_v);
                    if (__hle(__habs(__hsub(mi, my_val)), __float2half(MIN_DIFF))) {
                        int idx = atomicAdd(counter, 1);
                        if (paths && (max_paths < 0 || idx < max_paths)) {
                            int base = idx * 4;
                            paths[base + 0] = b;
                            paths[base + 1] = m_global;
                            paths[base + 2] = k;
                            paths[base + 3] = n_global;
                            values[idx] = mi;
                        }
                    }
                }
            }
        }
    }
}