#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include "utils.cuh"
#include "types.cuh"

__global__ void maxmin_kernel(float *A, float *B, float *C_max, float *C_min, int M, int K, int N)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float max_val = -FLT_MAX;

        int g_idx = batch * M * N * K + row * N * K + col * K;
        // Almacenar todos los mínimos para esta posición
        for (int k = 0; k < K; ++k)
        {
            float a = A[batch * M * K + row * K + k];
            float b = B[batch * K * N + k * N + col];
            float min_ab = a < b ? a : b;

            // Actualizar máximo de los mínimos (esto va al tensor max)
            if (min_ab > max_val)
                max_val = min_ab;
            int c_min_idx = g_idx + k;
            C_min[c_min_idx] = min_ab;
        }
        // El tensor de máximos contiene el máximo de los mínimos (resultado tradicional de maxmin)
        int idx = batch * M * N + row * N + col;
        C_max[idx] = max_val;
    }
}

// Versión mejorada de maxmin que usa TensorResult y retorna tanto max como min
void maxmin(const TensorResult &tensor1, const TensorResult &tensor2,
            TensorResult &max_result, TensorResult &min_result,
            bool keep_in_device)
{
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
    int batch = tensor1.batch;
    int M = tensor1.M;
    int K = tensor1.N;
    int N = tensor2.N;

    // Calcular tamaños de memoria
    size_t size_A = batch * M * K * sizeof(float);
    size_t size_B = batch * K * N * sizeof(float);
    size_t size_C_min = batch * M * N * K * sizeof(float);
    size_t size_C_max = batch * M * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C_max = nullptr, *d_C_min = nullptr;
    float *h_C_max = nullptr, *h_C_min = nullptr;

    // Asignar memoria para los resultados en device
    if (cudaMalloc(&d_C_max, size_C_max) != cudaSuccess)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para el resultado max en el dispositivo\n");
        return;
    }

    if (cudaMalloc(&d_C_min, size_C_min) != cudaSuccess)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para el resultado min en el dispositivo\n");
        cudaFree(d_C_max);
        return;
    }

    // Inicializar memoria de resultados a cero
    if (cudaMemset(d_C_max, 0, size_C_max) != cudaSuccess ||
        cudaMemset(d_C_min, 0, size_C_min) != cudaSuccess)
    {
        fprintf(stderr, "Error: No se pudo inicializar memoria de resultados\n");
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
            cudaFree(d_C_max);
            cudaFree(d_C_min);
            return;
        }

        if (cudaMemcpy(d_A, tensor1.data, size_A, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            fprintf(stderr, "Error: No se pudo copiar tensor1 al dispositivo\n");
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
            if (!tensor1.is_device_ptr)
                cudaFree(d_A);
            cudaFree(d_C_max);
            cudaFree(d_C_min);
            return;
        }

        if (cudaMemcpy(d_B, tensor2.data, size_B, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            fprintf(stderr, "Error: No se pudo copiar tensor2 al dispositivo\n");
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

    // Configurar dimensiones de grid y bloques
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y,
                 batch);

    // Ejecutar kernel que calcula tanto max como min
    maxmin_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C_max, d_C_min, M, K, N);
    cudaDeviceSynchronize();
    // Verificar errores de lanzamiento
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess)
    {
        fprintf(stderr, "Error en la ejecución del kernel: %s\n",
                cudaGetErrorString(kernelError));
        if (!tensor1.is_device_ptr)
            cudaFree(d_A);
        if (!tensor2.is_device_ptr)
            cudaFree(d_B);
        cudaFree(d_C_max);
        cudaFree(d_C_min);
        return;
    }

    // Liberar memoria temporal
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
