#ifndef LOGGING_CUH
#define LOGGING_CUH

// Re-export del logger existente para mantener consistencia
#include "../simple_logger.hpp"

// Funciones específicas para logging en contexto CUDA
namespace CudaLogging
{

    // Log de operaciones CUDA
    void log_cuda_operation(const char *operation, cudaError_t result);

    // Log de memoria CUDA
    void log_memory_usage(size_t used, size_t total);

    // Log de timing de kernels
    void log_kernel_timing(const char *kernel_name, float ms);

    // Log de configuración de kernel
    void log_kernel_config(const char *kernel_name, dim3 grid, dim3 block, size_t shared_mem);
}

#endif // LOGGING_CUH
