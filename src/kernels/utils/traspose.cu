#include "headers.cuh"
#include <cublas_v2.h>

// void transpose_with_cublas_V1(
//     cublasHandle_t handle,
//     const float *input, // [batch, K, N]
//     float *output,      // [batch, N, K]
//     int batch_size, int K, int N)
// {
//     const float alpha = 1.0f, beta = 0.0f;

//     for (int b = 0; b < batch_size; b++)
//     {
//         // Transpose: input[K,N] → output[N,K]
//         cublasSgeam(handle,
//                     CUBLAS_OP_T, CUBLAS_OP_N, // Transpose input, don't transpose "zero matrix"
//                     N, K,                     // Output dimensions
//                     &alpha,
//                     input + b * K * N, K, // Input matrix, leading dimension
//                     &beta,
//                     nullptr, N,             // No second matrix (beta=0)
//                     output + b * N * K, N); // Output matrix, leading dimension
//     }
// }

__global__ void transpose_kernel_optimized(
    const float* __restrict__ input,   // [batch, K, N]
    float* __restrict__ output,        // [batch, N, K]
    int K, int N, int batch_size)
{
    // Tile size para coalesced access
    constexpr int TILE_SIZE = 32;
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 para evitar bank conflicts
    
    int batch = blockIdx.z;
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Offset para este batch
    const float* batch_input = input + batch * K * N;
    float* batch_output = output + batch * N * K;
    
    // Cargar tile a shared memory (coalesced)
    if (y < K && x < N) {
        tile[threadIdx.y][threadIdx.x] = batch_input[y * N + x];
    }
    
    __syncthreads();
    
    // Escribir transpuesto (coalesced)
    int tx = blockIdx.y * TILE_SIZE + threadIdx.x;  // Posición transpuesta
    int ty = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (tx < K && ty < N) {
        batch_output[ty * K + tx] = tile[threadIdx.x][threadIdx.y];
    }
}
