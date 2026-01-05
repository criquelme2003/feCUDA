#include <headers.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <curand.h>
#include <utils.cuh>
#include <cuda_fp16.h>  
#include <cub/cub.cuh>
#include <curand_kernel.h>

#define RAND_SEED 1111ULL

__global__ void FloatToHalfKernel(float *d_in, __half *d_out,int total_elements)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int id = globalId; id < total_elements; id += total_threads)
        if (id < total_elements)
            d_out[id] = __float2half(d_in[id]);
}


__global__ void randomFill(__half *d_out,int total_elements,curandState_t *states,unsigned long long seed){
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    curandState *state = states + (globalId % 1000);
    curand_init(seed, globalId, 0, state);
    if(globalId < total_elements){
        //curand_init(seed,1,0,&state[globalId]);
        d_out[globalId] = __float2half(curand_uniform(state));
    }
}

int main()
    {

        cudaDeviceProp *props;
        curandState_t *curand_state;

        //curandGenerator_t gen;
        //CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        //CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, RAND_SEED));

        
        std::vector<int> batch_sizes = {10000};
        const int M = 100;
        
        cudaEvent_t start, end;
        
        for (auto batch_size : batch_sizes)
        {
            //float *dev_A;
            //CHECK_CUDA(cudaMalloc(&dev_A, sizeof(float) * total_elements));
            // Generar numeros aleatorios
            // CHECK_CURAND(curandGenerateUniform(gen, dev_A, total_elements));
            // FloatToHalfKernel<<<gridSize, blockSize>>>(dev_A, dev_half_A, total_elements);
            // CHECK_KERNEL();
            // cudaDeviceSynchronize();
            // CHECK_CUDA(cudaFree(dev_A));
            
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&end));
            const int total_elements = batch_size * M * M;
            int blockSize = 256;
            int gridSize  = (total_elements + blockSize - 1) / blockSize;

            __half *dev_half_A;
            CHECK_CUDA(cudaMalloc(&dev_half_A, sizeof(__half) * total_elements));

            curandState_t *curand_state;
            CHECK_CUDA(cudaMalloc(&curand_state, sizeof(curandState_t) * 1000));

            randomFill<<<gridSize, blockSize>>>(dev_half_A, total_elements, curand_state, RAND_SEED);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_KERNEL();

            TensorResult<__half> t1(dev_half_A, true, batch_size, M, M);
            TensorResult<__half> minres, maxres;
            maxmin<__half>(t1, t1, maxres, minres, start, end);
            CHECK_KERNEL();

            CHECK_CUDA(cudaEventSynchronize(end));
            float ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, end));
            std::cout << "Batch_size: " << batch_size << ", ms:" << ms << std::endl;
        }
    }
