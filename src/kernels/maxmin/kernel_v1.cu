
#include <float.h>
#include <cuda_runtime.h>
#include "../../../include/utils/cuda_utils.cuh"
#include "../../../include/core/types.cuh"

constexpr int WARP_SIZE = 32;

__global__ void max_min_kernel(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    float *__restrict__ C_max,   // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{
    extern __shared__ float warp_max_buffer[];

    const unsigned int k = threadIdx.x;
    const unsigned int lane = k & (warpSize - 1);
    const unsigned int warp_id = k >> 5;
    const unsigned int warp_count = (blockDim.x + warpSize - 1) >> 5;

    const unsigned int n = blockIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int b = blockIdx.z;

    const unsigned int a_idx = b * M * K + m * K + k;
    const unsigned int b_idx = b * K * N + k * N + n;
    const unsigned int c_min_idx = b * M * N * K + m * N * K + n * K + k;
    const unsigned int c_max_idx = b * M * N + m * N + n;

    float lane_max = -FLT_MAX;

    if (k < K)
    {
        const float lane_min = fminf(A[a_idx], B[b_idx]);
        C_min[c_min_idx] = lane_min;
        lane_max = lane_min;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        const float other = __shfl_down_sync(0xffffffff, lane_max, offset);
        lane_max = fmaxf(lane_max, other);
    }

    if (lane == 0)
    {
        warp_max_buffer[warp_id] = lane_max;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float block_max = (lane < warp_count) ? warp_max_buffer[lane] : -FLT_MAX;
        unsigned int active_mask = __ballot_sync(0xffffffff, lane < warp_count);

        if (lane < warp_count)
        {
            for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
            {
                const float other = __shfl_down_sync(active_mask, block_max, offset);
                block_max = fmaxf(block_max, other);
            }

            if (lane == 0)
            {
                C_max[c_max_idx] = block_max;
            }
        }
    }
}

