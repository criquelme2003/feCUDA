#include <core/types.cuh>
#include <cstdio>
#include <utils.cuh>

// Kernel para calcular prima = maxmin_conjugado - gen_tensor
__global__ void calculate_prima_kernel(float *maxmin_conjugado, float *gen_tensor,
                                       float *prima, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        prima[idx] = maxmin_conjugado[idx] - gen_tensor[idx];
    }
}

void calculate_prima(const TensorResult<> &maxmin_conjugado, const TensorResult<> &gen_tensor,
                     TensorResult<> &prima, bool keep_in_device)
{
    // FUNCIÓN NO COMPATIBLE CON NUEVA VERSIÓN DE TensorResult
    //
    // Problemas encontrados:
    // 1. Accede a campos privados: .batch, .M, .N, .K, .data, .is_device_ptr, .owns_memory
    // 2. Intenta asignar directamente estos campos (que ahora son privados)
    // 3. Nueva versión requiere usar:
    //    - getBatch(), getM(), getN(), getK()
    //    - getData() para acceso a datos
    //    - Constructores con MemorySpace y dimensiones
    //
    // Para reimplementar, se necesitaría:
    // 1. Usar método de construcción nuevo de TensorResult
    // 2. Copiar datos manualmente después de crear el objeto
    // 3. Rediseñar la gestión de ownership y ubicación de memoria
    
    fprintf(stderr, "ERROR: calculate_prima() no es compatible con la nueva versión de TensorResult\n");
    fprintf(stderr, "La función requiere rediseño para acceder a nuevos métodos públicos\n");
    
    // Nota: TensorResult tiene operator= eliminado, por lo que no podemos asignar
    // El parámetro de salida prima permanece sin modificar
    return;
}
