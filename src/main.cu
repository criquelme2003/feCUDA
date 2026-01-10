#include "../include/core/types.cuh"
#include "../include/headers.cuh"
#include "../include/utils.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#define RAND_SEED 1111ULL
#define RAND_STATES 1000

__global__ void FloatToHalfKernel(float *d_in, __half *d_out,
                                  int total_elements) {
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  for (int id = globalId; id < total_elements; id += total_threads)
    if (id < total_elements)
      d_out[id] = __float2half(d_in[id]);
}

__global__ void randomFill(__half *d_out, int total_elements,
                           curandState_t *states, unsigned long long seed) {
  int totalthreads = blockDim.x * gridDim.x;
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = globalId; i < total_elements; i += totalthreads) {
    curandState *state = states + (i % RAND_STATES);
    if (i < RAND_STATES) {
      curand_init(seed, i, 0, state);
    };
    d_out[i] = __float2half(curand_uniform(state));
  }
}

int main() {

  cudaDeviceProp *props;
  std::vector<int> batch_sizes = {10, 100, 1000, 10000};
  const int M = 30;

  cudaEvent_t start, end;
  curandState_t *curand_state;
  CHECK_CUDA(cudaMalloc(&curand_state, sizeof(curandState_t) * RAND_STATES));

  for (auto batch_size : batch_sizes) {
    printf("B:%d, M: %d\n", batch_size, M);

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    const int total_elements = batch_size * M * M;
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;

    TensorResult<__half> t1(MemorySpace::Device, batch_size, M, M);

    randomFill<<<gridSize, blockSize>>>(t1.getData(), total_elements,
                                        curand_state, RAND_SEED);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_KERNEL();

    TensorResult<__half> minres(MemorySpace::Device, batch_size, M, M, M);
    TensorResult<__half> maxres(MemorySpace::Device, batch_size, M, M);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    maxmin<__half>(t1, t1, maxres, minres, start, end);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_cpu - start_cpu);

    std::cout << "Execution time CPU: " << duration.count() << " ms"
              << std::endl;

    CHECK_CUDA(cudaEventSynchronize(end));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, end));
    std::cout << "Execution time GPU" << ms << " ms" << std::endl;
  }
}
