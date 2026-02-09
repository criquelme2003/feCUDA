
#include "../include/core/types.cuh"
#include "../include/headers.cuh"
#include "../include/utils.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#define RAND_SEED 1111ULL

__global__ void FloatToHalfKernel(float *d_in, __half *d_out, int total_elements)
{
    int total_threads = gridDim.x * blockDim.x;
    int globalId = blockIdx.x * gridDim.x + threadIdx.x;
    for (int id = globalId; id < total_elements; id += total_threads)
        if (id < total_elements)
            d_out[id] = __float2half(d_in[id]);
}

__global__ void randomInit(curandState_t *states, unsigned long long seed, int total_elements)
{
    int totalthreads = blockDim.x * gridDim.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalId < total_elements)
    {

        curandState *state = states + globalId;
        curand_init(seed, globalId, 0, state);
    }
}
__global__ void
randomFill(__half *d_out, int total_elements, curandState_t *states, unsigned long long seed)
{
    int totalthreads = blockDim.x * gridDim.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalId < total_elements)
    {
        curandState *state = states + globalId;
        d_out[globalId] = __float2half(curand_uniform(state));
    }
}

void test_dims(std::vector<int> ms, std::vector<int> batch_sizes)
{

    for (int M : ms)
    {
        for (int batch_size : batch_sizes)
        {
            printf("B:%d, M: %d\n", batch_size, M);

            int blockSize = 256;

            int total_elements = M * M;
            curandState_t *curand_state;
            CHECK_CUDA(cudaMalloc(&curand_state, sizeof(curandState_t) * total_elements));

            int gridSize = (total_elements + blockSize - 1) / blockSize;

            TensorResult<__half> t1(MemorySpace::Device, batch_size, M, M);

            randomInit<<<static_cast<int>(gridSize), static_cast<int>(blockSize)>>>(
                curand_state,
                RAND_SEED,
                total_elements
            );
            CHECK_CUDA(cudaDeviceSynchronize());

            randomFill<<<static_cast<int>(gridSize), static_cast<int>(blockSize)>>>(
                t1.getData(),
                total_elements,
                curand_state,
                RAND_SEED
            );
            CHECK_CUDA(cudaDeviceSynchronize());

            TensorResult<__half> minres(MemorySpace::Device, batch_size, M, M, M);
            TensorResult<__half> maxres(MemorySpace::Device, batch_size, M, M);

            TensorResult<__half> t2 = t1.clone();
            auto start_cpu = std::chrono::high_resolution_clock::now();
            maxmin(t1, t2, 0.4, 1);
            auto end_cpu = std::chrono::high_resolution_clock::now();

            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

            std::cout << "Execution time CPU: " << duration.count() << " ms" << std::endl;
        }
    }
}

int main()
{
    std::vector<int> batch_sizes = {10};
    std::vector<int> ms = {16,16,16,16};
    test_dims(ms, batch_sizes);
}
