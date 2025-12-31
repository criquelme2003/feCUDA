#include <headers.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <curand.h>
#include <utils.cuh>

#define RAND_SEED 1111ULL




void operation(int m,int n){
        std::cout << m + n << std::endl;
    }

size_t get_gpu_req_size(int m,int n)
    {
        return m * n * sizeof(int);
    }

size_t get_cpu_req_size(int m,int n)
    {
        return m * n * sizeof(int);
    }

        
int main()
    {


    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, RAND_SEED));

    std::vector<int> batch_sizes = {10, 100, 1000, 10000};
    const int M = 20;

    cudaEvent_t start,end;

    for(auto batch_size : batch_sizes){
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        float *dev_A;
        int total_elements = batch_size * M * M;
        CHECK_CUDA(cudaMalloc(&dev_A, sizeof(float) * total_elements));
        //Generar numeros aleatorios
        CHECK_CURAND(curandGenerateUniform(gen, dev_A, total_elements));
        TensorResult t1(dev_A, true, batch_size, M, M);
        TensorResult minres,maxres;
        maxmin(t1, t1, maxres, minres, start, end);

        cudaEventSynchronize(end);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, end);
        std::cout << "Batch_size: " << batch_size << ", ms:"<< ms << std::endl;
    };
        
    }
