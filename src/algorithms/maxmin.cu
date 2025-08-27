#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include "utils.cuh"
#include "maxmin_kernels.cuh"
#include <device_launch_parameters.h>
#include "types.cuh"
#include "headers.cuh"

// Versión mejorada de maxmin que usa TensorResult y retorna tanto max como min
void maxmin(const TensorResult &tensor1, const TensorResult &tensor2,
            TensorResult &max_result, TensorResult &min_result,
            bool keep_in_device)
{

    // Validar que los tensores sean 3D (K=1) como espera el kernel
    if (tensor1.K != 1 || tensor2.K != 1)
    {
        printf("Error: maxmin_kernel_v1 solo acepta tensores 3D (K=1)\n");
        return; // tensor nulo
    }

    // Para el kernel, necesitamos que A sea [batch, M, K] y B sea [batch, K, N]
    // Pero como K=1, efectivamente son [batch, M] y [batch, N]
    int batch = tensor1.batch;
    int M = tensor1.M;
    int K = tensor1.N; // En el contexto del kernel, N del tensor1 es K
    int N = tensor2.N;

    printf("Ejecutando kernel_v1 con dimensiones: batch=%d, M=%d, K=%d, N=%d\n", batch, M, K, N);

    // Tamaños de memoria
    size_t size_A = batch * M * K * sizeof(float);
    size_t size_B = batch * K * N * sizeof(float);
    size_t size_C_min = batch * M * N * K * sizeof(float);
    size_t size_C_max = batch * M * N * sizeof(float);

    // Alocar memoria en device
    float *d_A, *d_B, *d_C_min, *d_C_max;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_min, size_C_min);
    cudaMalloc(&d_C_max, size_C_max);

    // Preparar datos host si es necesario
    float *h_A = tensor1.data;
    float *h_B = tensor2.data;
    bool liberar_A = false, liberar_B = false;

    if (tensor1.is_device_ptr)
    {
        h_A = (float *)malloc(size_A);
        cudaMemcpy(h_A, tensor1.data, size_A, cudaMemcpyDeviceToHost);
        liberar_A = true;
    }

    if (tensor2.is_device_ptr)
    {
        h_B = (float *)malloc(size_B);
        cudaMemcpy(h_B, tensor2.data, size_B, cudaMemcpyDeviceToHost);
        liberar_B = true;
    }

    // Copiar datos al device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Configurar grid y bloques para el kernel
    dim3 blockSize(nextPow2(K)); // la potencia de 2 mas cercana threads por bloque
    dim3 gridSize(N, M, batch);  // Grid de (N, M, batch)
    size_t shared_mem_size = K * sizeof(float);

    // Ejecutar kernel
    max_min_kernel<<<gridSize, blockSize, shared_mem_size>>>(
        d_A, d_B, d_C_min, d_C_max, M, K, N, batch);

    cudaDeviceSynchronize();

    // Verificar errores del kernel
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        printf("Error en kernel_v1: %s\n", cudaGetErrorString(kernel_error));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_min);
        cudaFree(d_C_max);
        if (liberar_A)
            free(h_A);
        if (liberar_B)
            free(h_B);
        return ;
    }

    // Crear tensor resultado para C_max (solo retornamos C_max para validación)
    float *h_C_max = (float *)malloc(size_C_max);
    float *h_C_min = (float *)malloc(size_C_min); // Si se quiere retornar min también
    cudaMemcpy(h_C_max, d_C_max, size_C_max, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_min, d_C_min, size_C_min, cudaMemcpyDeviceToHost);

    max_result = TensorResult(h_C_max, false, batch, M, N, 1, true);
    min_result = TensorResult(h_C_min, false, batch, M, N, K, true);
    // Limpiar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_min);
    cudaFree(d_C_max);
    if (liberar_A)
        free(h_A);
    if (liberar_B)
        free(h_B);

    printf("Kernel_v1 ejecutado exitosamente\n");
    return ;
}
