#ifndef PATHS_CUH
#define PATHS_CUH

#include "core/types.cuh"

// Función armar_caminos - construcción de caminos
void armar_caminos(const TensorResult &previous_paths, const TensorResult &result_tensor,
                   const TensorResult &result_values, TensorResult &paths,
                   TensorResult &matched_values, int order);

// Funciones auxiliares para construcción de caminos
namespace PathUtils
{

    // Verificar compatibilidad de caminos
    bool are_paths_compatible(const TensorResult &paths1, const TensorResult &paths2);

    // Concatenar caminos
    void concatenate_paths(const TensorResult &paths1, const TensorResult &paths2,
                           TensorResult &result);

    // Validar estructura de caminos
    bool validate_path_structure(const TensorResult &paths);

    // Optimizar memoria de caminos
    void optimize_path_memory(TensorResult &paths);
}

#endif // PATHS_CUH
