#include "../../include/utils/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace CudaUtils
{

    void cuda_cleanup_and_check()
    {
        cudaError_t syncError = cudaDeviceSynchronize();
        if (syncError != cudaSuccess)
        {
            std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(syncError));
        }
    }

} // namespace CudaUtils


