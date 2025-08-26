#include "include/utils.cuh"
#include "include/types.cuh"
#include "include/headers.cuh"

// Declarar la función wrapper del kernel_v1
extern TensorResult maxmin_kernel_v1_wrapper(const TensorResult &tensor_a, const TensorResult &tensor_b);

int main()
{
    printf("=== VALIDACIÓN DEL KERNEL_V1 ===\n\n");

    // Inicializar CUDA
    cuda_warmup();

    // Validar el kernel_v1 usando la función de utilidad
    validar_algoritmos_maxmin(maxmin_kernel_v1_wrapper, "Kernel_V1");

    return 0;
}
