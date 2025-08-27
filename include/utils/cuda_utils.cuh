#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Excepción personalizada simple para errores CUDA
class CudaException : public std::runtime_error
{
public:
    explicit CudaException(const std::string &message) : std::runtime_error(message) {}
};

// Macro para verificar errores de CUDA con exceptions
#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess)                                                 \
        {                                                                       \
            std::string error_msg = std::string("CUDA error at ") +             \
                                    __FILE__ + ":" + std::to_string(__LINE__) + \
                                    ": " + cudaGetErrorString(err);             \
            std::cerr << error_msg << std::endl;                                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// Macro para verificar errores de CUDA sin salir del programa
#define CHECK_CUDA_SAFE(call)                                                   \
    {                                                                           \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess)                                                 \
        {                                                                       \
            std::string error_msg = std::string("CUDA error at ") +             \
                                    __FILE__ + ":" + std::to_string(__LINE__) + \
                                    ": " + cudaGetErrorString(err);             \
            std::cerr << error_msg << std::endl;                                \
            return false;                                                       \
        }                                                                       \
    }

// Funciones utilitarias CUDA
namespace CudaUtils
{

    // Función para limpiar y verificar el estado del dispositivo CUDA
    void cuda_cleanup_and_check();

    // Función para calentar el sistema CUDA
    void cuda_warmup();

    // Obtener información del dispositivo
    void print_device_info();

    // Verificar capacidades del dispositivo
    bool check_device_capabilities();
}

// Forward declarations
struct TensorResult;

// Función para limpiar memoria de TensorResult de forma segura
void safe_tensor_cleanup(TensorResult &tensor);

// Función para crear una copia del tensor en memoria host
TensorResult copy_tensor(const TensorResult &src);

#endif // CUDA_UTILS_CUH
