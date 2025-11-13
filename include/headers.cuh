#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "core/types.cuh"
#include <vector>

// Funciones expuestas durante el Sprint 1. El resto de módulos (bootstrap,
// servicios, kernels fusionados, etc.) quedan fuera del alcance temporal.

void maxmin(const TensorResult &tensor1, const TensorResult &tensor2,
            TensorResult &max_result, TensorResult &min_result,
            bool keep_in_device = true);

void indices(const TensorResult &min_result, const TensorResult &maxmin_prima,
             TensorResult &result_tensor_filtered, TensorResult &result_tensor_values,
             float threshold = 0.4f, bool keep_in_device = true);

void armar_caminos_batch(const TensorResult &previous_paths, const TensorResult &result_tensor,
                         const TensorResult &result_values, TensorResult &paths,
                         TensorResult &matched_values, int iteration, int batch_size = 1000,
                         bool keep_in_device = true);

void iterative_maxmin_cuadrado(const TensorResult &tensor, float thr, int order,
                               std::vector<TensorResult> &result_tensor_paths,
                               std::vector<TensorResult> &result_values_paths,
                               std::vector<TensorResult> &pure_tensor_paths,
                               std::vector<TensorResult> &pure_values_paths,
                               bool keep_in_device = true);

#endif
