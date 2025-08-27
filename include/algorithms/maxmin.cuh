#ifndef MAXMIN_CUH
#define MAXMIN_CUH

#include "core/types.cuh"
#include "kernels/maxmin_kernels.cuh"
#include <vector>

// Función principal maxmin - wrapper de alto nivel
void maxmin(const TensorResult &tensor1, const TensorResult &tensor2,
            TensorResult &max_result, TensorResult &min_result,
            bool keep_in_device = false);

// Función calculate_prima para cálculo de prima
void calculate_prima(const TensorResult &maxmin_conjugado, const TensorResult &gen_tensor,
                     TensorResult &prima);

// Función iterative_maxmin_cuadrado (crítica para rendimiento)
void iterative_maxmin_cuadrado(const TensorResult &tensor, float thr, int order,
                               std::vector<TensorResult> &result_tensor_paths,
                               std::vector<TensorResult> &result_values_paths,
                               std::vector<TensorResult> &pure_tensor_paths,
                               std::vector<TensorResult> &pure_values_paths);

#endif // MAXMIN_CUH
