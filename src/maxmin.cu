#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include "utils.cuh"
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include "types.cuh"

__global__ void min_kernel(
    const float *__restrict__ A, // [batch, M, K]
    const float *__restrict__ B, // [batch, K, N]
    float *__restrict__ C_min,   // [batch, M, K, N]
    const int M, const int K, const int N, const int batch_size)
{
    // Configuración 1D simple
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * M * K * N;
    
    if (global_id >= total_elements) return;
    
    // Descomponer el índice global en coordenadas
    int batch = global_id / (M * K * N);
    int tmp = global_id % (M * K * N);
    int row = tmp / (K * N);
    int tmp2 = tmp % (K * N);
    int k = tmp2 / N;
    int col = tmp2 % N;
    
    // Calcular índices para acceso directo
    const int a_idx = batch * M * K + row * K + k;        // A[batch, row, k]
    const int b_idx = batch * K * N + k * N + col;        // B[batch, k, col]
    
    // Una sola operación por thread
    const float a_val = A[a_idx];
    const float b_val = B[b_idx];
    C_min[global_id] = fminf(a_val, b_val);
}

template <int BLOCK_SIZE = 256>
__global__ void max_reduction_kernel(
    const float *__restrict__ C_min, // [batch, M, K, N]
    float *__restrict__ C_max,       // [batch, M, N]
    const int M, const int K, const int N, const int batch_size)
{
    // Configuración 1D - cada bloque procesa una posición (batch, row, col)
    int block_id = blockIdx.x;
    int total_output_elements = batch_size * M * N;
    
    if (block_id >= total_output_elements) return;
    
    // Descomponer block_id en coordenadas de salida
    int batch = block_id / (M * N);
    int tmp = block_id % (M * N);
    int row = tmp / N;
    int col = tmp % N;
    
    int tid = threadIdx.x;
    
    // Setup para reducción CUB
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // Calcular índice base en C_min para esta posición (batch, row, col)
    const int base_idx = batch * M * K * N + row * K * N + col;
    
    float thread_max = -FLT_MAX;
    
    // Cada thread procesa múltiples elementos de K si es necesario
    for (int k = tid; k < K; k += BLOCK_SIZE) {
        const int idx = base_idx + k * N; // C_min[batch, row, k, col]
        thread_max = fmaxf(thread_max, C_min[idx]);
    }
    
    // Reducción paralela usando CUB
    float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
    
    // Solo el primer thread escribe el resultado
    if (tid == 0) {
        const int out_idx = batch * M * N + row * N + col;
        C_max[out_idx] = block_max;
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

    // Configuración 1D para min_kernel
    const int BLOCK_SIZE = 256;
    int total_min_elements = batch * M * K * N;
    int num_blocks_min = (total_min_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Configuración 1D para max_reduction_kernel
    int total_max_blocks = batch * M * N; // Un bloque por cada elemento de salida

    // Ejecutar kernel que calcula mínimos - configuración 1D
    min_kernel<<<num_blocks_min, BLOCK_SIZE>>>(d_A, d_B, d_C_min, M, K, N, batch);
    cudaDeviceSynchronize();

    // Verificar errores de lanzamiento
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess)
    {
        fprintf(stderr, "Error en la ejecución del min_kernel: %s\n",
                cudaGetErrorString(kernelError));
        // ...existing error handling...
        return;
    }

    // Ejecutar kernel de reducción máxima - configuración 1D
    max_reduction_kernel<BLOCK_SIZE><<<total_max_blocks, BLOCK_SIZE>>>(d_C_min, d_C_max, M, K, N, batch);
    cudaDeviceSynchronize();

    kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess)
    {
        fprintf(stderr, "Error en la ejecución del max_reduction_kernel: %s\n",
                cudaGetErrorString(kernelError));
        // ...existing error handling...
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
