#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <headers.cuh>
#include <utils.cuh>
#include <utils/cuda_utils.cuh>

void iterative_maxmin_cuadrado(const TensorResult<> &tensor, float thr, int order,
                               std::vector<TensorResult<>> &result_tensor_paths,
                               std::vector<TensorResult<>> &result_values_paths,
                               std::vector<TensorResult<>> &pure_tensor_paths,
                               std::vector<TensorResult<>> &pure_values_paths,
                               bool keep_in_device)
{
    // FUNCIÓN NO COMPATIBLE CON NUEVA VERSIÓN DE TensorResult
    // 
    // Problemas encontrados:
    // 1. La función accede a campos privados: .data, .M, .N, .K, .batch, .is_device_ptr, .owns_memory
    // 2. Depende de funciones auxiliares que también son incompatibles:
    //    - safe_tensor_cleanup()
    //    - copy_tensor()
    //    - copy_tensor_to_cpu()
    //    - copy_tensor_to_gpu()
    //    - calculate_prima()
    //    - indices()
    //    - armar_caminos_batch()
    // 3. Necesita rediseño completo para usar la nueva API de TensorResult
    //
    // La nueva TensorResult requiere:
    // - Usar constructores con MemorySpace (Device/Host)
    // - Usar métodos públicos: getData(), getBatch(), getM(), getN(), getK()
    // - Usar move_to_device()/move_to_host() para transferencias
    // - No permite crear "vistas" de tensores existentes
    //
    // Para volver a habilitar esta función, será necesario:
    // 1. Rediseñar la gestión de memoria
    // 2. Implementar funciones helper compatibles con la nueva API
    // 3. Agregar métodos de acceso a datos si es necesario

    fprintf(stderr, "ERROR: iterative_maxmin_cuadrado() no es compatible con la nueva versión de TensorResult\n");
    fprintf(stderr, "La función requiere rediseño completo para usar la nueva API\n");
    
    // Limpiar vectores de salida para evitar comportamiento indefinido
    result_tensor_paths.clear();
    result_values_paths.clear();
    pure_tensor_paths.clear();
    pure_values_paths.clear();
    
    return;
}
