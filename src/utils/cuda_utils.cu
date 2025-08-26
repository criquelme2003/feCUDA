#include "../../include/utils.cuh"
#include "../../include/headers.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Función para limpiar y verificar el estado del dispositivo CUDA
void cuda_cleanup_and_check()
{
    // Sincronizar para asegurar que todas las operaciones han terminado
    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess)
    {
        printf("Warning: Error durante sincronización: %s\n", cudaGetErrorString(syncError));
    }

    // Obtener información de memoria
    size_t free_memory, total_memory;
    cudaError_t memError = cudaMemGetInfo(&free_memory, &total_memory);
    if (memError == cudaSuccess)
    {
        printf("Memoria GPU - Libre: %.2f MB, Total: %.2f MB, Usada: %.2f MB\n",
               free_memory / (1024.0 * 1024.0),
               total_memory / (1024.0 * 1024.0),
               (total_memory - free_memory) / (1024.0 * 1024.0));
    }

    // NO resetear el dispositivo para evitar problemas con contextos persistentes
    printf("Dispositivo CUDA sincronizado\n");
}

// Función para limpiar memoria de TensorResult de forma segura
void safe_tensor_cleanup(TensorResult &tensor)
{
    if (tensor.data && tensor.owns_memory)
    {
        if (tensor.is_device_ptr)
        {
            cudaFree(tensor.data);
        }
        else
        {
            free(tensor.data);
        }
    }
    tensor.data = nullptr;
    tensor.owns_memory = false;
    tensor.batch = tensor.M = tensor.N = tensor.K = 0;
}

// Función para crear una copia del tensor en memoria host
TensorResult copy_tensor(const TensorResult &src)
{
    TensorResult dst;
    size_t size = src.batch * src.M * src.N * src.K * sizeof(float);
    dst.data = (float *)malloc(size);
    memcpy(dst.data, src.data, size);
    dst.is_device_ptr = false;
    dst.batch = src.batch;
    dst.M = src.M;
    dst.N = src.N;
    dst.K = src.K;
    dst.owns_memory = true;
    return dst;
}

// Función para calentar el sistema CUDA
void cuda_warmup()
{
    printf("Calentando sistema CUDA...\n");

    // Verificar estado del dispositivo CUDA
    cudaError_t deviceError = cudaDeviceSynchronize();
    if (deviceError != cudaSuccess)
    {
        printf("Error: El dispositivo CUDA no está disponible[warmup]: %s\n", cudaGetErrorString(deviceError));
        return;
    }

    // Crear un tensor pequeño para warm-up
    int warmup_size = 16;
    size_t data_size = warmup_size * warmup_size * sizeof(float);

    float *h_data = (float *)malloc(data_size);
    if (h_data == nullptr)
    {
        printf("Error: No se pudo asignar memoria para warm-up\n");
        return;
    }

    // Llenar con datos dummy
    for (int i = 0; i < warmup_size * warmup_size; i++)
    {
        h_data[i] = 1.0f + (i % 10) * 0.1f;
    }

    TensorResult warm_tensor(h_data, false, 1, warmup_size, warmup_size, 1, true);

    // Ejecutar varias operaciones para inicializar CUDA completamente
    TensorResult max_result, min_result;

    // Primera operación maxmin para inicializar contexto y kernels
    maxmin(warm_tensor, warm_tensor, max_result, min_result, false);

    // Segunda operación para estabilizar el sistema
    TensorResult max_result2, min_result2;
    maxmin(warm_tensor, warm_tensor, max_result2, min_result2, false);

    // Sincronizar para asegurar que todo esté completado
    cudaDeviceSynchronize();

    // Limpiar memoria del warm-up
    safe_tensor_cleanup(max_result);
    safe_tensor_cleanup(min_result);
    safe_tensor_cleanup(max_result2);
    safe_tensor_cleanup(min_result2);
    safe_tensor_cleanup(warm_tensor);

    // Mostrar información de memoria después del warm-up
    size_t free_memory, total_memory;
    cudaError_t memError = cudaMemGetInfo(&free_memory, &total_memory);
    if (memError == cudaSuccess)
    {
        printf("Warm-up completado. Memoria GPU libre: %.1f MB\n",
               free_memory / (1024.0 * 1024.0));
    }
    else
    {
        printf("Warm-up completado.\n");
    }
}
