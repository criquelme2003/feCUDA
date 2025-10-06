#ifndef TENSOR_CUH
#define TENSOR_CUH

#include "types.cuh"

// Funciones utilitarias para trabajar con tensores
namespace TensorUtils
{

    // Crear tensor en device memory
    TensorResult create_device_tensor(int batch, int M, int N, int K = 1);

    // Crear tensor en host memory
    TensorResult create_host_tensor(int batch, int M, int N, int K = 1);

    // Copiar tensor de host a device
    TensorResult copy_to_device(const TensorResult &host_tensor);

    // Copiar tensor de device a host
    TensorResult copy_to_host(const TensorResult &device_tensor);

    // Verificar si las dimensiones son compatibles para operaciones
    bool are_compatible(const TensorResult &a, const TensorResult &b);

    // Inicializar tensor con valor específico
    void fill_tensor(TensorResult &tensor, float value);

    // Comparar dos tensores elemento por elemento
    bool tensors_equal(const TensorResult &a, const TensorResult &b, float tolerance = 1e-6f);
}

#endif