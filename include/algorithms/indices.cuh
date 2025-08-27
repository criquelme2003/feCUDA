#ifndef INDICES_CUH
#define INDICES_CUH

#include "core/types.cuh"

// Función indices - procesamiento de índices basado en umbrales
void indices(const TensorResult &min_result, const TensorResult &maxmin_prima,
             TensorResult &result_tensor_filtered, TensorResult &result_tensor_values,
             float threshold);

// Funciones auxiliares para procesamiento de índices
namespace IndexUtils
{

    // Filtrar índices por umbral
    void filter_by_threshold(const TensorResult &input, TensorResult &output, float threshold);

    // Encontrar índices máximos en tensores
    void find_max_indices(const TensorResult &input, TensorResult &indices);

    // Aplicar máscara a tensor
    void apply_mask(const TensorResult &input, const TensorResult &mask, TensorResult &output);
}

#endif // INDICES_CUH
