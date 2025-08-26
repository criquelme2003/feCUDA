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

    // Verificar estado del dispositivo CUDA
    cudaError_t deviceError = cudaDeviceSynchronize();
    if (deviceError != cudaSuccess)
    {
        printf("Error: El dispositivo CUDA no está disponible [maxmin]: %s\n", cudaGetErrorString(deviceError));
        return;
    }

    // Validaciones básicas
    if (tensor1.data == nullptr || tensor2.data == nullptr)
    {
        fprintf(stderr, "Error: Los punteros de tensores no pueden ser nulos\n");
        return;
    }

    // Validar compatibilidad de dimensiones
    if (tensor1.N != tensor2.M || tensor1.batch != tensor2.batch)
    {
        fprintf(stderr, "Error: Dimensiones incompatibles entre tensores: %d y %d\n", tensor1.N, tensor2.M);
        return;
    }

    // Extraer dimensiones del tensor
    const int batch = tensor1.batch;
    const int M = tensor1.M;
    const int K = tensor1.N;
    const int N = tensor2.N;

    // Calcular tamaños de memoria
    size_t size_A = batch * M * K * sizeof(float);
    size_t size_B = batch * K * N * sizeof(float);
    size_t size_C_min = batch * M * N * K * sizeof(float);
    size_t size_C_max = batch * M * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C_max = nullptr, *d_C_min = nullptr, *d_tB = nullptr;
    float *h_C_max = nullptr, *h_C_min = nullptr;

    cudaMalloc(&d_tB, size_B); // memoria temporal para B transpuesta
    // Asignar memoria para los resultados en device
    if (cudaMalloc(&d_C_max, size_C_max) != cudaSuccess)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para el resultado max en el dispositivo\n");
        cudaFree(d_tB);
        return;
    }

    if (cudaMalloc(&d_C_min, size_C_min) != cudaSuccess)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para el resultado min en el dispositivo\n");
        cudaFree(d_tB);
        cudaFree(d_C_max);
        return;
    }

    // Inicializar memoria de resultados a cero
    if (cudaMemset(d_C_max, 0, size_C_max) != cudaSuccess ||
        cudaMemset(d_C_min, 0, size_C_min) != cudaSuccess)
    {
        fprintf(stderr, "Error: No se pudo inicializar memoria de resultados\n");
        cudaFree(d_tB);
        cudaFree(d_C_max);
        cudaFree(d_C_min);
        return;
    }

    // Copiar datos al dispositivo si es necesario
    if (!tensor1.is_device_ptr)
    {
        // Si tensor1 está en host, copiarlo a device
        if (cudaMalloc(&d_A, size_A) != cudaSuccess)
        {
            fprintf(stderr, "Error: No se pudo asignar memoria para tensor1 en el dispositivo\n");
            cudaFree(d_tB);
            cudaFree(d_C_max);
            cudaFree(d_C_min);
            return;
        }

        if (cudaMemcpy(d_A, tensor1.data, size_A, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            fprintf(stderr, "Error: No se pudo copiar tensor1 al dispositivo\n");
            cudaFree(d_tB);
            cudaFree(d_A);
            cudaFree(d_C_max);
            cudaFree(d_C_min);
            return;
        }
    }
    else
    {
        // Si ya está en el dispositivo, usarlo directamente
        d_A = tensor1.data;
    }

    // Hacer lo mismo para tensor2
    if (!tensor2.is_device_ptr)
    {
        // Si tensor2 está en host, copiarlo a device
        if (cudaMalloc(&d_B, size_B) != cudaSuccess)
        {
            fprintf(stderr, "Error: No se pudo asignar memoria para tensor2 en el dispositivo\n");
            cudaFree(d_tB);
            if (!tensor1.is_device_ptr)
                cudaFree(d_A);
            cudaFree(d_C_max);
            cudaFree(d_C_min);
            return;
        }

        if (cudaMemcpy(d_B, tensor2.data, size_B, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            fprintf(stderr, "Error: No se pudo copiar tensor2 al dispositivo\n");
            cudaFree(d_tB);
            if (!tensor1.is_device_ptr)
                cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_max);
            cudaFree(d_C_min);
            return;
        }
    }
    else
    {
        // Si ya está en el dispositivo, usarlo directamente
        d_B = tensor2.data;
    }

    // trasponer B

    dim3 blockSize(K);
    dim3 gridSize(N, M, batch);

    int mins_size = K + K % 2;

    constexpr int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (K + TILE_SIZE - 1) / TILE_SIZE,
              batch);

    // Transponer B
    transpose_kernel_optimized<<<grid, block, 0, 0>>>(
        d_B, d_tB, K, N, batch);

    // imprimir b y tb

    cudaDeviceSynchronize();
    // imprimir traspose
    max_min_lineal_kernel<<<gridSize, blockSize, mins_size * sizeof(float)>>>(
        d_A, d_tB, d_C_min, d_C_max, M, K, N, batch);

    cudaDeviceSynchronize();
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess)
    {
        fprintf(stderr, "Error en la ejecución del max_reduction_kernel: %s\n",
                cudaGetErrorString(kernelError));
        // ...existing error handling...
        return;
    }
    // Liberar memoria temporal
    cudaFree(d_tB); // Liberar memoria de B transpuesta
    if (!tensor1.is_device_ptr)
        cudaFree(d_A);
    if (!tensor2.is_device_ptr)
        cudaFree(d_B);

    // Si queremos mantener el resultado en el dispositivo
    if (keep_in_device)
    {
        max_result = TensorResult(d_C_max, true, batch, M, N);
        min_result = TensorResult(d_C_min, true, batch, M, N, K);

        return;
    }

    // Si queremos el resultado en el host
    h_C_max = (float *)malloc(size_C_max);
    h_C_min = (float *)malloc(size_C_min);

    if (h_C_max == nullptr || h_C_min == nullptr)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para los resultados en host\n");
        if (h_C_max)
            free(h_C_max);
        if (h_C_min)
            free(h_C_min);
        cudaFree(d_C_max);
        cudaFree(d_C_min);
        return;
    }

    // Copiar de device a host
    if (cudaMemcpy(h_C_max, d_C_max, size_C_max, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_C_min, d_C_min, size_C_min, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Error: No se pudo copiar los resultados al host\n");
        free(h_C_max);
        free(h_C_min);
        cudaFree(d_C_max);
        cudaFree(d_C_min);
        return;
    }

    // Liberar memoria de device
    cudaFree(d_C_max);
    cudaFree(d_C_min);

    // Asignar los resultados a las referencias
    max_result.data = h_C_max;
    max_result.is_device_ptr = false;
    max_result.batch = batch;
    max_result.M = M;
    max_result.N = N;
    max_result.K = 1;

    min_result.data = h_C_min;
    min_result.is_device_ptr = false;
    min_result.batch = batch;
    min_result.M = M;
    min_result.N = N;
    min_result.K = K;
}
