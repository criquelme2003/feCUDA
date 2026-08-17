#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>

#define MIN_DIFF 0.01f

// ─────────────────────────────────────────────────────────────────────────────
// maxmin_threshold_kernel — producto max-min con threshold diferencial
//
// Calcula C_out[b,m,n] = max_k  min(A_mat[b,m,k], B_mat[b,k,n])
// y emite las aristas (b,m,k,n) donde la mejora sobre A_mat[b,m,n] supera thr.
//
// Threshold:  k_max - A_mat[b,m,n] >= thr
//   • A_mat es C_prev en cada step iterativo.
//   • step 0: A_mat = A_orig → suprime pares con arista directa (semántica paper).
//   • step s>0: A_mat = C_s → suprime pares ya encontrados (unicidad de nivel).
//
// Parámetros nullable (pueden ser nullptr):
//   paths, values, counter → si counter==nullptr no se emiten aristas
//   argmax                 → si nullptr no se guarda el k ganador
//
// shmem = K * sizeof(__half) + blockDim.x * (sizeof(__half) + sizeof(int))
//   - K * sizeof(__half): columna B_mat[b,:,n] cacheada una vez por bloque
//   - blockDim.x * (sizeof(__half) + sizeof(int)): buffers de reducción
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_threshold_kernel(
    const __half *__restrict__ A_mat, // [B,M,K] factor izq. = C_prev
    const __half *__restrict__ B_mat, // [B,K,N] factor der. = B_orig
    __half *__restrict__ C_out,       // [B,M,N] siempre se escribe
    int *__restrict__ argmax,         // nullable — [B,M,N] k ganador
    int *__restrict__ counter,        // nullable — cuenta celdas con efecto >= thr
    __half thr,
    int B,
    int M,
    int N,
    int K,
    int batch_id
) {

    // Get necesary idx's
    int b = (batch_id >= 0) ? batch_id : (int)blockIdx.z;
    int m = (int)blockIdx.y;
    int n = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;          // How many threads, (pow of 2)
    int out_id = b * M * N + m * N + n; // Write idx

    // smem layout: [K × __half (columna de B cacheada) | bsz × __half | bsz × int]
    extern __shared__ char smem_buf[];
    __half *s_bcol = reinterpret_cast<__half *>(smem_buf); // B_mat[b,:,n] cacheada
    __half *s_val = reinterpret_cast<__half *>(smem_buf + K * sizeof(__half)); // Save value for reduction
    int *s_k = reinterpret_cast<int *>(
        smem_buf + K * sizeof(__half) + bsz * sizeof(__half)); // save k idx's for reduction

    // ── Carga cooperativa de la columna B_mat[b,:,n] a shared memory ────────
    // Todos los bloques con distinto m comparten esta misma columna n; cachearla
    // evita que cada bloque vuelva a pedirla repetidamente a L2.
    for (int k = tid; k < K; k += bsz) {
        s_bcol[k] = B_mat[b * K * N + k * N + n];
    }
    __syncthreads();

    // ── Reducción local por thread ──────────────────────────────────────────
    __half best_val = __float2half(-FLT_MAX);
    int best_k = 0;

    //  !Important
    // Each thread works on K/bsz elements, K = 10000 -> do 78 comparisons (excepting the last
    // thread)
    for (int k = tid; k < K; k += bsz) {
        int a_idx = b * M * K + m * K + k;
        __half mi = __hmin(A_mat[a_idx], s_bcol[k]);
        if (__hgt(mi, best_val)) {
            best_val = mi;
            best_k = k;
        }
    }

    s_val[tid] = best_val;
    s_k[tid] = best_k;
    __syncthreads();

    // ── Reducción en shared memory con tracking de argmax ───────────────────
    // ! Improvable with warp reduction, the bsz has to be a pow of 2
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s && __hgt(s_val[tid + s], s_val[tid])) {
            s_val[tid] = s_val[tid + s];
            s_k[tid] = s_k[tid + s];
        }
        __syncthreads();
    }


    // ── Escritura densa (siempre) ────────────────────────────────────────────
    if (tid == 0) {
        __half k_max = s_val[0];
        C_out[out_id] = k_max;
        if (argmax)
            argmax[out_id] = s_k[0];
        if (counter && __hge(__hsub(k_max, A_mat[out_id]), thr))
            atomicAdd(counter, 1);
    }
    __syncthreads();

  }
