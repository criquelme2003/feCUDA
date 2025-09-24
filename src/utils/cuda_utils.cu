#include "../../include/utils/cuda_utils.cuh"
#include "../../include/core/types.cuh"
#include "../../include/utils/logging.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace CudaUtils
{

    void cuda_cleanup_and_check()
    {
        // Sincronizar para asegurar que todas las operaciones han terminado
        cudaError_t syncError = cudaDeviceSynchronize();
        if (syncError != cudaSuccess)
        {
            LOG_WARNING("Error durante sincronización: ", cudaGetErrorString(syncError));
        }

        // Obtener información de memoria
        size_t free_memory, total_memory;
        cudaError_t memError = cudaMemGetInfo(&free_memory, &total_memory);
        if (memError == cudaSuccess)
        {
            LOG_INFO("Memoria GPU - Libre: ", free_memory / (1024.0 * 1024.0),
                     " MB, Total: ", total_memory / (1024.0 * 1024.0),
                     " MB, Usada: ", (total_memory - free_memory) / (1024.0 * 1024.0), " MB");
        }

        LOG_INFO("Dispositivo CUDA sincronizado");
    }

    void cuda_warmup()
    {
        LOG_INFO("Calentando sistema CUDA...");

        // Verificar estado del dispositivo CUDA
        cudaError_t deviceError = cudaDeviceSynchronize();
        if (deviceError != cudaSuccess)
        {
            LOG_WARNING("El dispositivo CUDA no está disponible[warmup]: ", cudaGetErrorString(deviceError));
            return;
        }

        // Crear un tensor pequeño para warm-up
        int warmup_size = 16;
        size_t data_size = warmup_size * warmup_size * sizeof(float);

        float *h_data = (float *)malloc(data_size);
        if (h_data == nullptr)
        {
            LOG_WARNING("No se pudo asignar memoria para warm-up");
            return;
        }

        // Llenar con datos dummy
        for (int i = 0; i < warmup_size * warmup_size; i++)
        {
            h_data[i] = 1.0f + (i % 10) * 0.1f;
        }

        free(h_data);

        // Mostrar información de memoria después del warm-up
        size_t free_memory, total_memory;
        cudaError_t memError = cudaMemGetInfo(&free_memory, &total_memory);
        if (memError == cudaSuccess)
        {
            LOG_INFO("Warm-up completado. Memoria GPU libre: ", free_memory / (1024.0 * 1024.0), " MB");
        }
        else
        {
            LOG_INFO("Warm-up completado.");
        }
    }

    bool check_device_capabilities()
    {
        int device_count;
        cudaGetDeviceCount(&device_count);

        if (device_count == 0)
        {
            LOG_WARNING("No se encontraron dispositivos CUDA");
            return false;
        }

        for (int i = 0; i < device_count; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            LOG_INFO("Dispositivo ", i, ": ", prop.name);
            LOG_INFO("  Compute Capability: ", prop.major, ".", prop.minor);
            LOG_INFO("  Memoria Global: ", prop.totalGlobalMem / (1024 * 1024), " MB");
            LOG_INFO("  Max threads por bloque: ", prop.maxThreadsPerBlock);
            LOG_INFO("  Max dimensiones de bloque: ", prop.maxThreadsDim[0],
                     " x ", prop.maxThreadsDim[1], " x ", prop.maxThreadsDim[2]);
            LOG_INFO("  Max dimensiones de grid: ", prop.maxGridSize[0],
                     " x ", prop.maxGridSize[1], " x ", prop.maxGridSize[2]);
        }

        return true;
    }

} // namespace CudaUtils

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
    
    if (src.is_device_ptr)
    {
        // Copiar en GPU (device to device)
        CHECK_CUDA(cudaMalloc(&dst.data, size));
        CHECK_CUDA(cudaMemcpy(dst.data, src.data, size, cudaMemcpyDeviceToDevice));
        dst.is_device_ptr = true;
    }
    else
    {
        // Copiar en CPU (host to host)
        dst.data = (float *)malloc(size);
        memcpy(dst.data, src.data, size);
        dst.is_device_ptr = false;
    }
    
    dst.batch = src.batch;
    dst.M = src.M;
    dst.N = src.N;
    dst.K = src.K;
    dst.owns_memory = true;
    
    return dst;
}



// Función para forzar copia en CPU
TensorResult copy_tensor_to_cpu(const TensorResult &src)
{
    TensorResult dst;
    size_t size = src.batch * src.M * src.N * src.K * sizeof(float);
    
    dst.data = (float *)malloc(size);
    
    if (src.is_device_ptr)
    {
        CHECK_CUDA(cudaMemcpy(dst.data, src.data, size, cudaMemcpyDeviceToHost));
    }
    else
    {
        memcpy(dst.data, src.data, size);
    }
    
    dst.is_device_ptr = false; // Siempre CPU
    dst.batch = src.batch;
    dst.M = src.M;
    dst.N = src.N;
    dst.K = src.K;
    dst.owns_memory = true;
    
    return dst;
}

// Función para forzar copia en GPU
TensorResult copy_tensor_to_gpu(const TensorResult &src)
{
    TensorResult dst;
    size_t size = src.batch * src.M * src.N * src.K * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&dst.data, size));
    
    if (src.is_device_ptr)
    {
        CHECK_CUDA(cudaMemcpy(dst.data, src.data, size, cudaMemcpyDeviceToDevice));
    }
    else
    {
        CHECK_CUDA(cudaMemcpy(dst.data, src.data, size, cudaMemcpyHostToDevice));
    }
    
    dst.is_device_ptr = true; // Siempre GPU
    dst.batch = src.batch;
    dst.M = src.M;
    dst.N = src.N;
    dst.K = src.K;
    dst.owns_memory = true;
    
    return dst;
}