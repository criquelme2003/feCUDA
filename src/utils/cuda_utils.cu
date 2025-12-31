#include "../../include/utils/cuda_utils.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace CudaUtils
{

    void cuda_cleanup_and_check()
    {
        cudaError_t syncError = cudaDeviceSynchronize();
        if (syncError != cudaSuccess)
        {
            std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(syncError));
        }
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
