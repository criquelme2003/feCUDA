#ifndef FROM_COLMAJOR_CUH
#define FROM_COLMAJOR_CUH

#include "core/types.cuh"
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Kernel: out-of-place transpose column-major -> row-major
//
//   src: column-major MxN, ld = M   =>  src[r,c] = src[c*M + r]
//   dst: row-major    MxN, ld = N   =>  dst[r,c] = dst[r*N + c]
//
// Shared-memory tile with +1 padding to avoid 32-way bank conflicts.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void colToRowMajorKernel(const T *__restrict__ src, T *__restrict__ dst, int M, int N)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    // -------- Load: coalesced along column-major fast axis (rows) --------
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int r = row + j;
        if (r < M && col < N)
        {
            tile[threadIdx.y + j][threadIdx.x] = src[static_cast<size_t>(col) * M + r];
        }
    }

    __syncthreads();

    // -------- Store: coalesced along row-major fast axis (cols) ----------
    row = blockIdx.y * TILE_DIM + threadIdx.x;
    col = blockIdx.x * TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int c = col + j;
        if (row < M && c < N)
        {
            dst[static_cast<size_t>(row) * N + c] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

template <typename T>
inline void launch_col_to_row_major(const T *d_src, T *d_dst, int M, int N, cudaStream_t stream = 0)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    colToRowMajorKernel<T><<<grid, block, 0, stream>>>(d_src, d_dst, M, N);
    CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Main entry: copy a host column-major matrix to a device row-major
// TensorResult. Output shape is (b=1, m=M, n=N, k=1).
//
//   host_src : pointer to M*N contiguous elements in column-major order
//              (element (r,c) is at host_src[c*M + r])
//   M, N     : matrix dimensions
//
// Pipeline (Option 2):
//   1. pinned staging buffer     (fast PCIe)
//   2. async H2D copy            (raw bytes, no layout change)
//   3. GPU transpose kernel      (col-major -> row-major)
//   4. free temporaries
// ---------------------------------------------------------------------------
template <typename T> TensorResult<T> tensor_from_colmajor_host(const T *host_src, int M, int N)
{
    if (host_src == nullptr)
    {
        throw std::runtime_error("tensor_from_colmajor_host: null source pointer");
    }
    if (M <= 0 || N <= 0)
    {
        throw std::runtime_error("tensor_from_colmajor_host: M and N must be positive");
    }

    const size_t n_elems = static_cast<size_t>(M) * static_cast<size_t>(N);
    const size_t n_bytes = n_elems * sizeof(T);

    // --- Pinned staging buffer -------------------------------------------
    T *pinned = nullptr;
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void **>(&pinned), n_bytes));
    std::memcpy(pinned, host_src, n_bytes);

    // --- Destination TensorResult (row-major on device) ------------------
    TensorResult<T> out(MemorySpace::Device, /*b=*/1, /*m=*/M, /*n=*/N, /*k=*/1);

    // --- Temporary device buffer (holds raw col-major bytes) -------------
    T *d_tmp = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_tmp), n_bytes));

    // --- Async H2D + transpose on a private stream -----------------------
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaMemcpyAsync(d_tmp, pinned, n_bytes, cudaMemcpyHostToDevice, stream));

    launch_col_to_row_major<T>(d_tmp, out.getData(), M, N, stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // --- Cleanup ---------------------------------------------------------
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaFreeHost(pinned));

    return out;
}

#endif // FROM_COLMAJOR_CUH