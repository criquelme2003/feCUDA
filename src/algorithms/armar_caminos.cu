#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <utils.cuh>
#include <core/types.cuh>
#include <utils/memory_utils.cuh>
#include <utils/validation_utils.cuh>

__global__ void find_path_matches_kernel(float *previous_paths, float *result_tensor,
                                         float *result_values, float *output_paths,
                                         float *output_values, int *match_count,
                                         int num_prev_paths, int num_current_tensor,
                                         int prev_cols, int current_cols, int iteration)
{
    int prev_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (prev_idx < num_prev_paths && curr_idx < num_current_tensor)
    {
        // Extraer coordenadas del camino previo
        // previous_paths formato: [batch, start_fila, intermedio1, intermedio2, ..., end_columna]
        int p_batch = (int)previous_paths[prev_idx * prev_cols];                      // batch
        int p_fila = (int)previous_paths[prev_idx * prev_cols + 1];                   // fila inicial
        int p_intermedio = (int)previous_paths[prev_idx * prev_cols + iteration + 2]; // intermedio en posición i+1

        // Extraer coordenadas del resultado actual [batch, fila, intermedio, columna]
        int c_batch = (int)result_tensor[curr_idx * current_cols];          // batch
        int c_fila = (int)result_tensor[curr_idx * current_cols + 1];       // fila
        int c_intermedio = (int)result_tensor[curr_idx * current_cols + 2]; // intermedio
        int c_columna = (int)result_tensor[curr_idx * current_cols + 3];    // nueva columna

        // Condición de match: batch, fila e intermedio deben coincidir
        if (p_batch == c_batch && p_fila == c_fila && p_intermedio == c_intermedio)
        {
            // Found a match - usar atomic add para obtener posición de salida
            int output_idx = atomicAdd(match_count, 1);

            // El nuevo camino tendrá prev_cols + 1 columnas
            int new_cols = prev_cols + 1;
            int output_base = output_idx * new_cols;

            // Copiar todas las columnas del camino previo
            for (int col = 0; col < prev_cols; col++)
            {
                output_paths[output_base + col] = previous_paths[prev_idx * prev_cols + col];
            }

            // Agregar la nueva columna (destino del resultado actual)
            output_paths[output_base + prev_cols] = (float)c_columna;

            // Guardar el valor correspondiente
            output_values[output_idx] = result_values[curr_idx];
        }
    }
}

// Usar aliases para mantener compatibilidad y claridad
template <typename T>
using CudaDevicePtr = MemoryUtils::CudaDevicePtr<T>;

template <typename T>
using HostPtr = MemoryUtils::HostPtr<T>;

using CudaMemoryManager = MemoryUtils::CudaMemoryManager;
using InputValidator = ValidationUtils::InputValidator;

void armar_caminos(const TensorResult &previous_paths, const TensorResult &result_tensor,
                   const TensorResult &result_values, TensorResult &paths,
                   TensorResult &matched_values, int iteration)
{
    // Validación de entrada
    if (!InputValidator::validate_paths_input(previous_paths, result_tensor, result_values))
    {
        return;
    }

    // Extraer dimensiones
    const int num_prev_paths = previous_paths.M;
    const int prev_cols = previous_paths.N; // Debe ser 4 + (iteration - 1)
    const int num_current_tensor = result_tensor.M;
    const int current_cols = result_tensor.N; // Debe ser 4
    const int num_values = result_values.N;
    const int new_cols = 4 + iteration; // El nuevo camino tendrá 4 + iteration columnas

    if (!InputValidator::validate_dimensions(num_current_tensor, num_values))
    {
        return;
    }

    // Calcular tamaños
    const int max_output_size = num_prev_paths * num_current_tensor;
    const size_t prev_size = num_prev_paths * prev_cols * sizeof(float);
    const size_t curr_size = num_current_tensor * current_cols * sizeof(float);
    const size_t values_size = num_values * sizeof(float);

    try
    {
        // Alocar memoria en device con RAII
        CudaDevicePtr<float> d_output_paths(max_output_size * new_cols);
        CudaDevicePtr<float> d_output_values(max_output_size);
        CudaDevicePtr<int> d_match_count(1);

        CHECK_CUDA(cudaMemset(d_match_count.get(), 0, sizeof(int)));

        // Preparar punteros para datos de entrada (usar existentes o crear copias)
        CudaDevicePtr<float> d_previous_paths = previous_paths.is_device_ptr ? CudaDevicePtr<float>(previous_paths.data) : CudaDevicePtr<float>(num_prev_paths * prev_cols);

        CudaDevicePtr<float> d_result_tensor = result_tensor.is_device_ptr ? CudaDevicePtr<float>(result_tensor.data) : CudaDevicePtr<float>(num_current_tensor * current_cols);

        CudaDevicePtr<float> d_result_values = result_values.is_device_ptr ? CudaDevicePtr<float>(result_values.data) : CudaDevicePtr<float>(num_values);

        // Copiar datos si es necesario
        if (!previous_paths.is_device_ptr)
        {
            CHECK_CUDA(cudaMemcpy(d_previous_paths.get(), previous_paths.data, prev_size, cudaMemcpyHostToDevice));
        }
        if (!result_tensor.is_device_ptr)
        {
            CHECK_CUDA(cudaMemcpy(d_result_tensor.get(), result_tensor.data, curr_size, cudaMemcpyHostToDevice));
        }
        if (!result_values.is_device_ptr)
        {
            CHECK_CUDA(cudaMemcpy(d_result_values.get(), result_values.data, values_size, cudaMemcpyHostToDevice));
        }

        // Configurar y lanzar kernel
        const dim3 block_size(16, 16);
        const dim3 grid_size((num_prev_paths + block_size.x - 1) / block_size.x,
                             (num_current_tensor + block_size.y - 1) / block_size.y);

        find_path_matches_kernel<<<grid_size, block_size>>>(
            d_previous_paths.get(), d_result_tensor.get(), d_result_values.get(),
            d_output_paths.get(), d_output_values.get(), d_match_count.get(),
            num_prev_paths, num_current_tensor, prev_cols, current_cols, iteration);

        CHECK_CUDA(cudaDeviceSynchronize());

        // Obtener número de matches
        int match_count;
        CHECK_CUDA(cudaMemcpy(&match_count, d_match_count.get(), sizeof(int), cudaMemcpyDeviceToHost));

        if (match_count > 0)
        {
            // Alocar memoria host para resultados usando RAII
            const size_t final_paths_size = match_count * new_cols * sizeof(float);
            const size_t final_values_size = match_count * sizeof(float);

            HostPtr<float> h_output_paths(match_count * new_cols);
            HostPtr<float> h_output_values(match_count);

            // Copiar resultados a host
            CHECK_CUDA(cudaMemcpy(h_output_paths.get(), d_output_paths.get(), final_paths_size, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_output_values.get(), d_output_values.get(), final_values_size, cudaMemcpyDeviceToHost));

            // Configurar TensorResult de salida (transferir ownership)
            paths.data = h_output_paths.release();
            paths.is_device_ptr = false;
            paths.owns_memory = true;
            paths.batch = previous_paths.batch;
            paths.M = match_count;
            paths.N = new_cols;
            paths.K = 1;

            matched_values.data = h_output_values.release();
            matched_values.is_device_ptr = false;
            matched_values.owns_memory = true;
            matched_values.batch = previous_paths.batch;
            matched_values.M = 1;
            matched_values.N = match_count;
            matched_values.K = 1;
        }
        else
        {
            std::cerr << "Error: No se encontraron matches\n";

            // Configurar TensorResult vacíos
            paths = TensorResult();
            matched_values = TensorResult();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error en armar_caminos: " << e.what() << '\n';
        paths = TensorResult();
        matched_values = TensorResult();
    }
}