#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "types.cuh"
#include <vector>

// Funciones exportadas desde archivos .cu
void maxmin(const TensorResult &tensor1, const TensorResult &tensor2,
            TensorResult &max_result, TensorResult &min_result,
            bool keep_in_device = false);

// Función indices
void indices(const TensorResult &min_result, const TensorResult &maxmin_prima,
             TensorResult &result_tensor_filtered, TensorResult &result_tensor_values,
             float threshold);

// Función armar_caminos
void armar_caminos(const TensorResult &previous_paths, const TensorResult &result_tensor,
                   const TensorResult &result_values, TensorResult &paths,
                   TensorResult &matched_values, int order);

// Función calculate_prima
void calculate_prima(const TensorResult &maxmin_conjugado, const TensorResult &gen_tensor,
                     TensorResult &prima);

// Función iterative_maxmin_cuadrado
void iterative_maxmin_cuadrado(const TensorResult &tensor, float thr, int order,
                               std::vector<TensorResult> &result_tensor_paths,
                               std::vector<TensorResult> &result_values_paths,
                               std::vector<TensorResult> &pure_tensor_paths,
                               std::vector<TensorResult> &pure_values_paths);
#endif
