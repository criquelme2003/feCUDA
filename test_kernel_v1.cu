#include "include/utils.cuh"
#include "include/types.cuh"
#include "include/headers.cuh"

int main()
{
    printf("=== VALIDACIÓN DE MAXMIN ===\n\n");

    // Inicializar CUDA
    cuda_warmup();

    // Validar maxmin usando la función de utilidad
    validar_algoritmos_maxmin("MaxMin_CUDA");

    return 0;
}
