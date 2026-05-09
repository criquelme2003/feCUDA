#include <cuda_fp16.h>

#include "indexing.cuh"

#define MIN_DIFF 0.01f

// path_emission.cuh
__device__ __forceinline__ void emit_tied_paths(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half k_max, int b, int m, int n, int M, int N, int K,
    int tid, int bsz,
    int* paths, __half* values, int* counter, int max_paths)
{
    const __half tol = __float2half(MIN_DIFF);
    for (int k = tid; k < K; k += bsz) {
        __half mi = __hmin(A[idx_A(b,m,k,M,K)], B[idx_B(b,k,n,K,N)]);
        if (__hle(__habs(__hsub(mi, k_max)), tol)) {
            int idx = atomicAdd(counter, 1);
            if (paths && (max_paths < 0 || idx < max_paths)) {
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