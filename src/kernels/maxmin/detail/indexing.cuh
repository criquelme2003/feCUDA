#pragma once
// indexing.cuh
__device__ __forceinline__ int idx_3d(int b, int i, int j, int I, int J) {
    return b * I * J + i * J + j;
}

__device__ __forceinline__ int idx_A(int b, int m, int k, int M, int K) {
    return b * M * K + m * K + k;
}

__device__ __forceinline__ int idx_B(int b, int k, int n, int K, int N) {
    return b * K * N + k * N + n;
}
