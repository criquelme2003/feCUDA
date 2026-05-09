#include <cuda_fp16.h>
#include <float.h>

#include "indexing.cuh"


// Reducción local: cada thread procesa K/bsz elementos
__device__ __forceinline__ void thread_local_maxmin(
    const __half* __restrict__ A, const __half* __restrict__ B,
    int b, int m, int n, int M, int N, int K,
    int tid, int bsz,
    __half& best_val, int& best_k)
{
    best_val = __float2half(-FLT_MAX);
    best_k = 0;
    for (int k = tid; k < K; k += bsz) {
        __half mi = __hmin(A[idx_A(b,m,k,M,K)], B[idx_B(b,k,n,K,N)]);
        if (__hgt(mi, best_val)) { best_val = mi; best_k = k; }
    }
}

// Reducción en shared memory con tracking de argmax
__device__ __forceinline__ void block_reduce_maxmin(__half* s_val, int* s_k, int tid, int bsz)
{
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s && __hgt(s_val[tid + s], s_val[tid])) {
            s_val[tid] = s_val[tid + s];
            s_k[tid]   = s_k[tid + s];
        }
        __syncthreads();
    }
}