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
// shmem = blockDim.x * (sizeof(__half) + sizeof(int))
// ─────────────────────────────────────────────────────────────────────────────
__global__ void maxmin_threshold_kernel(
    const __half* __restrict__ A_mat,  // [B,M,K] factor izq. = C_prev
    const __half* __restrict__ B_mat,  // [B,K,N] factor der. = B_orig
    __half*       __restrict__ C_out,  // [B,M,N] siempre se escribe
    int*          __restrict__ paths,  // nullable — flat int[count*4]
    __half*       __restrict__ values, // nullable
    int*          __restrict__ counter,// nullable — atomic counter
    int*          __restrict__ argmax, // nullable — [B,M,N] k ganador
    __half thr,
    int B, int M, int N, int K,
    int batch_id,
    int max_paths  // -1 = sin límite; >= 0 = cap de escritura (sin OOB)
)
{
    int b   = (batch_id >= 0) ? batch_id : (int)blockIdx.z;
    int m   = (int)blockIdx.y;
    int n   = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsz = (int)blockDim.x;
    int out_id = b * M * N + m * N + n;

    // smem layout: [bsz × __half | bsz × int]
    extern __shared__ char smem_buf[];
    __half* s_val = reinterpret_cast<__half*>(smem_buf);
    int*    s_k   = reinterpret_cast<int*>(smem_buf + bsz * sizeof(__half));

    // ── Reducción local por thread ──────────────────────────────────────────
    __half best_val = __float2half(-FLT_MAX);
    int    best_k   = 0;

    for (int k = tid; k < K; k += bsz)
    {
        int a_idx = b * M * K + m * K + k;
        int b_idx = b * K * N + k * N + n;
        __half mi = __hmin(A_mat[a_idx], B_mat[b_idx]);
        if (__hgt(mi, best_val)) { best_val = mi; best_k = k; }
    }

    s_val[tid] = best_val;
    s_k[tid]   = best_k;
    __syncthreads();

    // ── Reducción en shared memory con tracking de argmax ───────────────────
    for (int s = bsz / 2; s > 0; s >>= 1)
    {
        if (tid < s && __hgt(s_val[tid + s], s_val[tid]))
        {
            s_val[tid] = s_val[tid + s];
            s_k[tid]   = s_k[tid + s];
        }
        __syncthreads();
    }

    __half k_max = s_val[0];

    // ── Escritura densa (siempre) ────────────────────────────────────────────
    if (tid == 0)
    {
        C_out[out_id] = k_max;
        if (argmax) argmax[out_id] = s_k[0];
    }
    __syncthreads();

    // ── Emisión de aristas (solo si counter != nullptr) ─────────────────────
    // Threshold diferencial: k_max - A_mat[m,n] >= thr
    if (counter && __hsub(k_max, A_mat[out_id]) >= thr)
    {
        for (int k = tid; k < K; k += bsz)
        {
            int a_idx = b * M * K + m * K + k;
            int b_idx = b * K * N + k * N + n;
            __half mi = __hmin(A_mat[a_idx], B_mat[b_idx]);

            if (__hle(__habs(__hsub(mi, k_max)), __float2half(MIN_DIFF)))
            {
                int idx = atomicAdd(counter, 1);
                if (paths && (max_paths < 0 || idx < max_paths))
                {
                    int base = idx * 4;
                    paths[base + 0] = b;
                    paths[base + 1] = m;
                    paths[base + 2] = k;
                    paths[base + 3] = n;
                    values[idx] = mi;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// count_new_kernel — convergencia en GPU
//
// Cuenta celdas donde C_after[i] - C_before[i] >= thr_f.
// Reemplaza la copia de B×M×N × 2 bytes a CPU en cada step.
// Lanzar: block(256), grid(min((total+255)/256, 1024))
// shmem = 256 * sizeof(int)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void count_new_kernel(
    const __half* __restrict__ C_before,
    const __half* __restrict__ C_after,
    int*          __restrict__ d_count,
    float thr_f,
    int total_elems
)
{
    extern __shared__ int smem_cnt[];
    int tid = (int)threadIdx.x;
    int i   = (int)(blockIdx.x * blockDim.x + tid);

    int local = 0;
    for (; i < total_elems; i += (int)(gridDim.x * blockDim.x))
        if (__half2float(C_after[i]) - __half2float(C_before[i]) >= thr_f)
            local++;

    smem_cnt[tid] = local;
    __syncthreads();

    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem_cnt[tid] += smem_cnt[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(d_count, smem_cnt[0]);
}
