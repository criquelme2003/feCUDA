#include "../../include/core/types.cuh"
#include "../../include/kernels/maxmin_kernels.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

#define MAX_GRID_SIZE 10000
#define MAX_PATHS_PER_ITER 100000

std::vector<std::tuple<int4 *, __half *, int>>
maxmin(TensorResult<__half> &tensor1, TensorResult<__half> &tensor2, __half thr, int order)
{

    // Validar que los tensores sean 3D (K=1) como espera el kernel (WRAPPER
    // MAKES)

    if (order > 1)
    {
        printf(" Order > 1 not implemented\n");
        exit(0);
    }

    std::vector<std::tuple<int4 *, __half *, int>> ret;

    if (tensor1.getK() != 1 || tensor2.getK() != 1)
    {
        printf("Error: maxmin solo acepta tensores 3D (K=1)\n");
        exit(0); // tensor nulo
    }

    // Para el kernel, necesitamos que A sea [batch, M, K] y B sea [batch, K, N]
    // Pero como K=1, efectivamente son [batch, M] y [batch, N]
    int B = tensor1.getBatch();
    int M = tensor1.getM();
    int K = tensor1.getN(); // En el contexto del kernel, N del tensor1 es K
    int N = tensor2.getN();

    // Alocar memoria en device
    __half *d_A, *d_B;

    tensor1.move_to_device();
    tensor2.move_to_device();
    d_A = (__half *)tensor1.getData();
    d_B = (__half *)tensor2.getData();

    for (int iteration = 0; iteration < order; iteration++)
    {
        int4 *d_global_paths;
        __half *d_global_values;
        int h_total_count = 0;

        if ((B * M * N * K) < MAX_PATHS_PER_ITER)
        {
            std::cout << "[MAXMIN C++] EXECUTING COMPLETE ALGORITHM" << std::endl;
            int *d_counter;
            dim3 block(128); // >= K
            dim3 grid(N, M, B);
            size_t shmem = 128 * sizeof(__half);
            cudaMalloc(&d_counter, sizeof(int));
            cudaMemset(d_counter, 0, sizeof(int));

            cudaMalloc(&d_global_paths, MAX_PATHS_PER_ITER * sizeof(int4));
            cudaMalloc(&d_global_values, MAX_PATHS_PER_ITER * sizeof(__half));

            maxmin_threshold_kernel<<<grid, block, shmem>>>(
                d_A,
                d_B,
                d_global_paths,
                d_global_values,
                d_counter,
                thr,
                B,
                M,
                N,
                K,
                -1
            );
            CHECK_CUDA(cudaDeviceSynchronize())

            CHECK_CUDA(cudaGetLastError())

            int temp = 0;
            cudaMemcpy(&temp, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
            h_total_count = temp;
            cudaFree(d_counter);
        }
        else // Para quienes puedan superar el maximo de paths por kernel, se ejecuta un lanzamiento
             // por batches
        {
            std::cout << "[MAXMIN C++] EXECUTING BATCHED ALGORITHM" << std::endl;

            std::vector<__half *> d_values_acc;
            std::vector<int4 *> d_paths_acc;
            std::vector<int> h_counter_acc;
            int mppi = M * K * N;
            int sizeA = M * K;
            int sizeB = K * N;

            dim3 block(128); // >= K
            dim3 grid(N, M, 1);
            size_t shmem = 128 * sizeof(__half);

            int *d_counter;
            cudaMalloc(&d_counter, sizeof(int));
            for (int b_ = 0; b_ < B; b_++)
            {
                int4 *d_paths;
                __half *d_values;
                cudaMemset(d_counter, 0, sizeof(int));

                cudaMalloc(&d_paths, mppi * sizeof(int4));
                cudaMalloc(&d_values, mppi * sizeof(__half));

                __half *localA = d_A + (b_ * sizeA);

                __half *localB = d_B + (b_ * sizeB);
                maxmin_threshold_kernel<<<grid, block, shmem>>>(
                    localA,
                    localB,
                    d_paths,
                    d_values,
                    d_counter,
                    thr,
                    B,
                    M,
                    N,
                    K,
                    b_
                );
                CHECK_CUDA(cudaDeviceSynchronize())
                CHECK_CUDA(cudaGetLastError())

                d_values_acc.push_back(d_values);
                d_paths_acc.push_back(d_paths);

                int temp = 0;
                cudaMemcpy(&temp, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

                h_total_count += temp;
                h_counter_acc.push_back(h_total_count);
            }
            cudaFree(d_counter);

            cudaMalloc(&d_global_paths, h_total_count * sizeof(int4));
            cudaMalloc(&d_global_values, h_total_count * sizeof(__half));

            for (int c = 0; c < B; c++)
            {
                int offset = 0;
                if (c > 0)
                {
                    offset = h_counter_acc[c - 1];
                }

                int count_here = h_counter_acc[c] - offset;

                cudaMemcpy(
                    d_global_paths + offset,
                    d_paths_acc[c],
                    count_here * sizeof(int4),
                    cudaMemcpyDeviceToDevice
                );

                cudaMemcpy(
                    d_global_values + offset,
                    d_values_acc[c],
                    count_here * sizeof(__half),
                    cudaMemcpyDeviceToDevice
                );

                cudaFree(d_paths_acc[c]);
                cudaFree(d_values_acc[c]);
            }
        }

        // CHECK_CUDA(cudaEventRecord(end));
        std::cout << "[MAXMIN C++] Paths finded: " << h_total_count << std::endl;
        ret.push_back(std::make_tuple(d_global_paths, d_global_values, h_total_count));
        
    }

    return ret;
}
