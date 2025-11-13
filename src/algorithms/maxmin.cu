#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "kernels/maxmin_kernels.cuh"
#include "core/types.cuh"
#include "utils.cuh"

namespace
{
    dim3 compute_block_dims(int K)
    {
        unsigned int threads = nextPow2(static_cast<unsigned int>(K));
        threads = std::min(threads, 1024u);
        return dim3(threads == 0 ? 1u : threads);
    }

    dim3 compute_grid_dims(int batch, int M, int N)
    {
        return dim3(N, M, batch);
    }
}

void maxmin(const TensorResult &tensor1, const TensorResult &tensor2,
            TensorResult &max_result, TensorResult &min_result,
            bool keep_in_device)
{
    max_result = TensorResult();
    min_result = TensorResult();

    if (tensor1.data == nullptr || tensor2.data == nullptr)
    {
        printf("Error: Tensores de entrada nulos para maxmin\n");
        return;
    }

    if (tensor1.batch != tensor2.batch || tensor1.M != tensor2.M || tensor1.N != tensor2.M || tensor2.M != tensor1.N)
    {
        printf("Error: maxmin requiere tensores compatibles y cuadrados\n");
        return;
    }

    if (tensor1.K != 1 || tensor2.K != 1)
    {
        printf("Error: maxmin (Sprint 1) solo soporta K=1\n");
        return;
    }

    const int batch = tensor1.batch;
    const int M = tensor1.M;
    const int K = tensor1.N;
    const int N = tensor2.N;

    const size_t size_A = static_cast<size_t>(batch) * M * K * sizeof(float);
    const size_t size_B = static_cast<size_t>(batch) * K * N * sizeof(float);
    const size_t size_C_min = static_cast<size_t>(batch) * M * N * K * sizeof(float);
    const size_t size_C_max = static_cast<size_t>(batch) * M * N * sizeof(float);

    float *d_A = tensor1.is_device_ptr ? tensor1.data : nullptr;
    float *d_B = tensor2.is_device_ptr ? tensor2.data : nullptr;
    bool allocated_A = false;
    bool allocated_B = false;

    if (!tensor1.is_device_ptr)
    {
        CHECK_CUDA(cudaMalloc(&d_A, size_A));
        CHECK_CUDA(cudaMemcpy(d_A, tensor1.data, size_A, cudaMemcpyHostToDevice));
        allocated_A = true;
    }

    if (!tensor2.is_device_ptr)
    {
        CHECK_CUDA(cudaMalloc(&d_B, size_B));
        CHECK_CUDA(cudaMemcpy(d_B, tensor2.data, size_B, cudaMemcpyHostToDevice));
        allocated_B = true;
    }

    float *d_C_min = nullptr;
    float *d_C_max = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C_min, size_C_min));
    CHECK_CUDA(cudaMalloc(&d_C_max, size_C_max));

    dim3 block = compute_block_dims(K);
    dim3 grid = compute_grid_dims(batch, M, N);
    size_t shared_mem = static_cast<size_t>(block.x) * sizeof(float);

    max_min_kernel<<<grid, block, shared_mem>>>(
        d_A, d_B, d_C_min, d_C_max, M, K, N, batch);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (keep_in_device)
    {
        max_result = TensorResult(d_C_max, true, batch, M, N, 1, true);
        min_result = TensorResult(d_C_min, true, batch, M, N, K, true);
    }
    else
    {
        float *h_C_max = static_cast<float *>(std::malloc(size_C_max));
        float *h_C_min = static_cast<float *>(std::malloc(size_C_min));

        if (!h_C_max || !h_C_min)
        {
            printf("Error: No se pudo asignar memoria host para resultados de maxmin\n");
            if (h_C_max)
                std::free(h_C_max);
            if (h_C_min)
                std::free(h_C_min);
        }
        else
        {
            CHECK_CUDA(cudaMemcpy(h_C_max, d_C_max, size_C_max, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_C_min, d_C_min, size_C_min, cudaMemcpyDeviceToHost));
            max_result = TensorResult(h_C_max, false, batch, M, N, 1, true);
            min_result = TensorResult(h_C_min, false, batch, M, N, K, true);
        }

        cudaFree(d_C_max);
        cudaFree(d_C_min);
    }

    if (allocated_A)
    {
        cudaFree(d_A);
    }
    if (allocated_B)
    {
        cudaFree(d_B);
    }
}
