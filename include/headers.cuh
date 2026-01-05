#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "core/types.cuh"
#include <vector>
#include <cuda_fp16.h>

// Funciones exportadas desde archivos .cu

//------------------------------------------------
template <typename T>
void maxmin(const TensorResult<T> &tensor1, const TensorResult<T> &tensor2,
            TensorResult<T> &max_result, TensorResult<T> &min_result, cudaEvent_t &start, cudaEvent_t &end,
            bool keep_in_device = true);


extern template void maxmin<float>(const TensorResult<float> &, const TensorResult<float> &, TensorResult<float> &, TensorResult<float> &, cudaEvent_t&, cudaEvent_t&, bool);

extern template void maxmin<__half>(const TensorResult<__half> &, const TensorResult<__half> &, TensorResult<__half> &, TensorResult<__half> &, cudaEvent_t&, cudaEvent_t&, bool);

//------------------------------------------------


// Función indices
void indices(const TensorResult<> &min_result, const TensorResult<> &maxmin_prima,
             TensorResult<> &result_tensor_filtered, TensorResult<> &result_tensor_values,
             float threshold = 0.4, bool keep_in_device = true);

// Función armar_caminos - construcción de caminos
void armar_caminos_original(const TensorResult<> &previous_paths, const TensorResult<> &result_tensor,
                            const TensorResult<> &result_values, TensorResult<> &paths,
                            TensorResult<> &matched_values, int order, bool keep_in_device = true);

void armar_caminos_batch(const TensorResult<> &previous_paths, const TensorResult<> &result_tensor,
                         const TensorResult<> &result_values, TensorResult<> &paths,
                         TensorResult<> &matched_values, int iteration, int batch_size = 1000, bool keep_in_device = true);

// Función iterative_maxmin_cuadrado (crítica para rendimiento)
void iterative_maxmin_cuadrado(const TensorResult<> &tensor, float thr, int order,
                               std::vector<TensorResult<>> &result_tensor_paths,
                               std::vector<TensorResult<>> &result_values_paths,
                               std::vector<TensorResult<>> &pure_tensor_paths,
                               std::vector<TensorResult<>> &pure_values_paths,
                               bool keep_in_device = true);

#endif
