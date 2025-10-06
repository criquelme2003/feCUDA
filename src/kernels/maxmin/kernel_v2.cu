#include <float.h>

// Versión alternativa con mejor acceso a B
__global__ void max_min_kernel_v2(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    float *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{
    extern __shared__ float shared_mem[];
    float* shared_B = shared_mem;
    float* mins = shared_B + blockDim.x;
    
    const int tid = threadIdx.x;
    const int m = blockIdx.y;
    const int n = blockIdx.x;
    const int b = blockIdx.z;
    
    float local_max = -FLT_MAX;
    
    // Procesar en chunks, cargando B a shared memory
    for(int k_start = 0; k_start < K; k_start += blockDim.x) {
        int k = k_start + tid;
        
        // Cargar B a shared memory coalescido
        if(k < K) {
            shared_B[tid] = B[b * K * N + k * N + n];
        } else {
            shared_B[tid] = FLT_MAX; // Valor neutral
        }
        __syncthreads();
        
        // Procesar chunk
        if(k < K) {
            int a_idx = b * M * K + m * K + k;
            int c_min_idx = b * M * N * K + m * N * K + n * K + k;
            
            float min_val = fminf(A[a_idx], shared_B[tid]);
            C_min[c_min_idx] = min_val;
            local_max = fmaxf(local_max, min_val);
        }
        __syncthreads();
    }
    
    // Reducción final
    mins[tid] = local_max;
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            mins[tid] = fmaxf(mins[tid], mins[tid + s]);
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        C_max[b * M * N + m * N + n] = mins[0];
    }
}