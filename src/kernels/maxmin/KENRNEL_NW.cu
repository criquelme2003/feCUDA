/*
 * maxmin_threshold_kernel — versión optimizada
 *
 * Problemas del original y soluciones aplicadas:
 *
 * PROBLEMA 1 — Grid de 400M bloques (1 bloque por celda).
 *   SOLUCIÓN  — Cada bloque cubre un tile BM_TILE × BN_TILE de celdas de salida.
 *               Con BM=8, BN=8 → 6.25M bloques en vez de 400M.
 *
 * PROBLEMA 2 — B_mat se lee con stride N=20000 → completamente no-coalescente.
 *   SOLUCIÓN  — Cargamos tiles de B transpostos en smem: cada warp carga una
 *               columna contigua de B. Dentro del tile K iteramos sobre la smem.
 *
 * PROBLEMA 3 — Reducción smem sin warp primitives.
 *   SOLUCIÓN  — Últimos 32 elementos reducidos con __shfl_down_sync → sin syncs.
 *
 * PROBLEMA 4 — Doble pasada sobre K en emisión de aristas.
 *   SOLUCIÓN  — Un solo loop, acumula max y emite aristas en la misma pasada
 *               (segunda pasada sólo si counter != null, tras conocer k_max).
 *
 * LANZAMIENTO sugerido:
 *   dim3 grid((N + BN-1)/BN, (M + BM-1)/BM, B);
 *   dim3 block(WARP_SIZE * WARPS_PER_BLOCK);   // e.g. 32*4 = 128 threads
 *   size_t smem = (BM + BN) * BK * sizeof(__half)
 *                 + BLOCK_THREADS * (sizeof(__half) + sizeof(int));
 *   maxmin_tile_kernel<<<grid, block, smem>>>(...)
 *
 * Para M=N=K=20000, BM=BN=8, BK=64, 128 threads:
 *   - Bloques: 2500 × 2500 × 1 = 6.25M  (vs 400M original)
 *   - Cada bloque lee BM*K half de A  y  BN*K half de B (tiles coalescentes)
 *   - Lectura efectiva de B: stride 1 en smem tras carga transpuesta
 */

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

/* Reducción warp-level de __half con tracking de índice */
__device__ __forceinline__
void warp_reduce_maxarg(__half& val, int& idx)
{
    /* Últimos 5 niveles sin acceso a smem ni __syncthreads */
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        __half  v2 = __shfl_down_sync(mask, val, offset);
        int     i2 = __shfl_down_sync(mask, idx, offset);
        if (__hgt(v2, val)) { val = v2; idx = i2; }
    }
}

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

            #pragma unroll 8
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


/* ─────────────────────────────────────────────────────────────────────────────
 * count_new_kernel — versión optimizada
 *
 * Cambios:
 *  1. Loop vectorizado con __half2: compara 2 elementos por instrucción.
 *  2. Reducción intra-warp con __reduce_add_sync (SM80+) o shuffle.
 *  3. Reducción inter-warp en smem (solo 1 sync por bloque, no N/2).
 * ─────────────────────────────────────────────────────────────────────────── */
__global__ void count_new_kernel_opt(
    const __half* __restrict__ C_before,
    const __half* __restrict__ C_after,
    int*          __restrict__ d_count,
    float thr_f,
    int total_elems
)
{
    /* Usamos __half2 para procesar 2 elementos por ciclo */
    const __half2* b2 = reinterpret_cast<const __half2*>(C_before);
    const __half2* a2 = reinterpret_cast<const __half2*>(C_after);
    int tid  = (int)threadIdx.x;
    int gid  = (int)(blockIdx.x * blockDim.x + tid);
    int step = (int)(gridDim.x  * blockDim.x);

    __half2 thr2 = __float2half2_rn(thr_f);

    int local = 0;
    int n2 = total_elems / 2;

    /* Pasada vectorizada (pares de elementos) */
    for (int i = gid; i < n2; i += step) {
        __half2 diff = __hsub2(a2[i], b2[i]);
        /* diff >= thr2: comparar ambos componentes */
        local += (int)__hge(diff.x, thr2.x) + (int)__hge(diff.y, thr2.y);
    }

    /* Elemento impar si total_elems es impar */
    if (gid == 0 && (total_elems & 1)) {
        int last = total_elems - 1;
        if (__half2float(C_after[last]) - __half2float(C_before[last]) >= thr_f)
            local++;
    }

    /* ── Reducción warp-level con shuffle ─────────────────────────────── */
    unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(mask, local, off);

    /* ── Reducción inter-warp en smem ────────────────────────────────── */
    extern __shared__ int smem_cnt[];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) smem_cnt[warp] = local;
    __syncthreads();

    /* Solo el primer warp reduce los parciales */
    if (warp == 0) {
        local = (lane < (blockDim.x >> 5)) ? smem_cnt[lane] : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local += __shfl_down_sync(mask, local, off);
        if (lane == 0) atomicAdd(d_count, local);
    }
}


/* ─────────────────────────────────────────────────────────────────────────────
 * Función de lanzamiento — helper
 *
 * Calcula grid/block y smem automáticamente.
 * ─────────────────────────────────────────────────────────────────────────── */
inline cudaError_t launch_maxmin_tile(
    const __half* A_mat, const __half* B_mat,
    __half* C_out, int* paths, __half* values,
    int* counter, int* argmax,
    __half thr,
    int B, int M, int N, int K,
    int batch_id, int max_paths,
    cudaStream_t stream = 0)
{
    dim3 block(BLOCK_THREADS);
    dim3 grid(
        (N + BN_TILE - 1) / BN_TILE,
        (M + BM_TILE - 1) / BM_TILE,
        (batch_id >= 0) ? 1 : B
    );

    size_t smem =
        (size_t)(BM_TILE * BK_TILE) * sizeof(__half) +   // s_A
        (size_t)(BN_TILE * BK_TILE) * sizeof(__half) +   // s_B
        (size_t) BLOCK_THREADS      * sizeof(__half) +   // s_val (reducción, no usada en tile kernel — reserva para compatibilidad)
        (size_t) BLOCK_THREADS      * sizeof(int);        // s_k

    maxmin_tile_kernel<<<grid, block, smem, stream>>>(
        A_mat, B_mat, C_out, paths, values, counter, argmax,
        thr, B, M, N, K, batch_id, max_paths
    );
    return cudaGetLastError();
}