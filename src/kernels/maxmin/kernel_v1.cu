#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <float.h>

template <typename T>
__global__ void maxmin_threshold_kernel(
    T *__restrict__ X,         // gen_tensor [B,M,K]
    const T *__restrict__ X0,  // original_tensor [B,K,N]
    int4 *__restrict__ paths,  // output paths
    T *__restrict__ values,    // min values
    int *__restrict__ counter, // atomic counter
    T thr,
    int B, int M, int N, int K, int batch_id)
{
  int b = batch_id >= 0 ? batch_id : blockIdx.z;
  int m = blockIdx.y;
  int n = blockIdx.x;

  int tid = threadIdx.x;
  int block_size = blockDim.x;

  extern __shared__ __half smem[];

  T v = -INFINITY;

  // Grid-stride: cada thread procesa múltiples K, aplicando max instantaneamene para controlar K.
  for (int k = tid; k < K; k += block_size)
  {
    int a_idx = b * M * K + m * K + k;
    int b_idx = b * K * N + k * N + n;
    v = max(v, min(static_cast<float>(X[a_idx]), static_cast<float>(X0[b_idx])));
  }

  smem[tid] = v;
  __syncthreads();

  // Reducción max
  for (int s = block_size >> 1; s > 0; s >>= 1)
  {
    if (tid < s)
      smem[tid] = max(static_cast<float>(smem[tid]), static_cast<float>(smem[tid + s]));
    __syncthreads();
  }

  T mm = smem[0];
  __syncthreads();

  // Encontrar máximos repetidos y seleccionar caminos
  for (int k = tid; k < K; k += block_size)
  {
    int a_idx = b * M * K + m * K + k;
    int b_idx = b * K * N + k * N + n;

    float mi = min(static_cast<float>(X[a_idx]), static_cast<float>(X0[b_idx]));

    if (static_cast<float>(mm) - static_cast<float>(X[a_idx]) >= static_cast<float>(thr) && mi == static_cast<float>(mm))
    {
      int idx = atomicAdd(counter, 1);
      paths[idx] = make_int4(b, m, k, N);
      values[idx] = v;
    }
  }
  __syncthreads();
  X[b * M * N + m * N + n] = mm;
}

template __global__ void maxmin_threshold_kernel<__half>(
    __half *__restrict__ X,        // gen_tensor [B,M,K]
    const __half *__restrict__ X0, // original_tensor [B,K,N]
    int4 *__restrict__ paths,      // output paths
    __half *__restrict__ values,   // min values
    int *__restrict__ counter,     // atomic counter
    __half thr,
    int B, int M, int N, int K,int batch_id);

template __global__ void maxmin_threshold_kernel<float>(
    float *__restrict__ X,        // gen_tensor [B,M,K]
    const float *__restrict__ X0, // original_tensor [B,K,N]
    int4 *__restrict__ paths,     // output paths
    float *__restrict__ values,   // min values
    int *__restrict__ counter,    // atomic counter
    float thr,
    int B, int M, int N, int K,int batch_id);

// template __global__ void maxmin_threshold_kernel<__half>(
//     const __half *__restrict__ X,  // gen_tensor [B,M,K]
//     const __half *__restrict__ X0, // original_tensor [B,K,N]
//     int4 *__restrict__ paths,      // output paths
//     __half *__restrict__ values,   // min values
//     int *__restrict__ counter,     // atomic counter
//     __half thr,
//     int B, int M, int N, int K);