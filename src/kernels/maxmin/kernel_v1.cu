#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>

__global__ void max_min_kernel(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    float *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{

    extern __shared__ float mins[];

    unsigned int local_id = threadIdx.x;
    unsigned int global_id = (blockIdx.x * blockDim.x) + local_id;
    unsigned int ks_per_block = blockDim.x / K;

    unsigned int k = global_id % K;
    unsigned int out_id = global_id / K;

    unsigned int b = out_id / (M * N);
    unsigned int rem = out_id % (M * N);
    unsigned int m = rem / N;
    unsigned int n = rem - m * N;

    unsigned int a_idx = b * M * K + m * K + k;
    unsigned int b_idx = b * K * N + k * N + n;
    unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + k;

    if (local_id < (ks_per_block * K))
    {
        mins[local_id] = fminf(A[a_idx], B[b_idx]);
        C_min[c_min_idx] = mins[local_id];
    }
    else
    {
        mins[local_id] = -INFINITY;
    }
    __syncthreads();
    unsigned base = (local_id / K) * K;
    for (unsigned int s = K >> 1; s > 0; s >>= 1)
    {
        if (k < s && (k + s) < K)
        {
            float a = mins[base + k];
            float b = mins[base + k + s];
            mins[base + k] = fmaxf(a, b); // usa fminf si quisieras mínimo final
        }
        __syncthreads();
    }

    if (k == 0)
    {
        C_max[out_id] = mins[base];
    }
}

template <typename T, int WARPS_PER_BLOCK>
__global__ void cub_max_min_kernel(
    const T *__restrict__ A, // [batch, M, K]
    const T *__restrict__ B, // [batch, K, N]
    T *__restrict__ C_min,   // [batch, M, K, N]
    T *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{
    if (K <= 32)
    {
        int warp = threadIdx.x >> 5; // /32
        int lane = threadIdx.x & 31; // %32

        unsigned int local_id = threadIdx.x;
        unsigned int global_id = (blockIdx.x * blockDim.x) + local_id;
        unsigned int ks_per_block = blockDim.x / 32;

        unsigned int out_id = global_id / 32;

        unsigned int b = out_id / (M * N);
        unsigned int rem = out_id % (M * N);
        unsigned int m = rem / N;
        unsigned int n = rem - m * N;

        unsigned int a_idx = b * M * K + m * K + lane;
        unsigned int b_idx = b * K * N + lane * N + n;
        unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + lane;

        __half warp_value = -INFINITY;
        if (lane < K)
        {
            warp_value = fminf(A[a_idx], B[b_idx]);
            C_min[c_min_idx] = warp_value;
        }

        // Aplicar reduccion con cub a nivel de warp
        using warpReduce = cub::WarpReduce<T>;
        __shared__ typename warpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];

        T out = warpReduce(temp_storage[warp]).Max(warp_value);

        if (lane == 0)
            C_max[out_id] = out;
    }
    else
    {

        unsigned int local_id = threadIdx.x;
        unsigned int global_id = (blockIdx.x * blockDim.x) + local_id;

        unsigned int required_warps_per_K = (K + 31) / 32;
        unsigned int k_launch_size = required_warps_per_K * 32;
        unsigned int out_id = global_id / k_launch_size;

        unsigned int local_K = local_id / k_launch_size;

        int k = local_id % k_launch_size;
        int warp = threadIdx.x >> 5; // /32

        unsigned int b = out_id / (M * N);
        unsigned int rem = out_id % (M * N);
        unsigned int m = rem / N;
        unsigned int n = rem - m * N;

        unsigned int a_idx = b * M * K + m * K + k;
        unsigned int b_idx = b * K * N + k * N + n;
        unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + k;

        T warp_value = -INFINITY;
        if (local_id < (K * (local_K + 1)))
        {
            warp_value = fminf(A[a_idx], B[b_idx]);
            C_min[c_min_idx] = warp_value;
        }

        // Aplicar reduccion con cub a nivel de warp
        using warpReduce = cub::WarpReduce<T>;
        __shared__ typename warpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];
        __shared__ T reduction_results[WARPS_PER_BLOCK];

        T out = warpReduce(temp_storage[warp]).Max(warp_value);
        reduction_results[warp] = out;

        __syncthreads();
        if (k == 0 && out_id >= batch_size * M * N)
        {
            T k_max = -INFINITY;
            int base_k = local_K * required_warps_per_K;
            for (int w = 0; w < required_warps_per_K; w++)
            {
                k_max = fmaxf(k_max, reduction_results[base_k + w]);
            }
            C_max[out_id] = k_max;
        }
    }
}

template __global__ void cub_max_min_kernel<__half,4>(
    const __half*, // [batch, M, K]
    const __half*, // [batch, M, K]
    __half*,       // [batch, M, K]
    __half*,       // [batch, M, K]
    const int, const int, const int, const int);

template __global__ void cub_max_min_kernel<float,4>(
    const float*, // [batch, M, K]
    const float*, // [batch, M, K]
    float*,       // [batch, M, K]
    float*,       // [batch, M, K]
    const int, const int, const int, const int);