#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include "utils.cuh"
#include "types.cuh"

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

void armar_caminos(const TensorResult &previous_paths, const TensorResult &result_tensor,
                   const TensorResult &result_values, TensorResult &paths,
                   TensorResult &matched_values, int iteration)
{
    // Validaciones
    if (previous_paths.data == nullptr)
    {
        printf("Error: previous_paths es nulo\n");
        return;
    }
    if (result_tensor.data == nullptr)
    {
        printf("Error: result_tensor es nulo\n");
        return;
    }
    if (result_values.data == nullptr)
    {
        printf("Error: result_values es nulo\n");
        return;
    };

    // Extraer dimensiones
    int num_prev_paths = previous_paths.M;
    int prev_cols = previous_paths.N; // Debe ser 4 + (iteration - 1)
    int num_current_tensor = result_tensor.M;
    int current_cols = result_tensor.N; // Debe ser 4
    int num_values = result_values.N;

    // El nuevo camino tendrá 4 + iteration columnas
    int new_cols = 4 + iteration;

    if (num_current_tensor != num_values)
    {
        printf("Error: Número de elementos en result_tensor (%d) no coincide con result_values (%d)\n",
               num_current_tensor, num_values);
        return;
    }

    // Calcular tamaños máximos de salida
    int max_output_size = num_prev_paths * num_current_tensor;
    size_t prev_size = num_prev_paths * prev_cols * sizeof(float);
    size_t curr_size = num_current_tensor * current_cols * sizeof(float);
    size_t values_size = num_values * sizeof(float);
    size_t output_paths_size = max_output_size * new_cols * sizeof(float);
    size_t output_values_size = max_output_size * sizeof(float);

    // Alocar memoria en device
    float *d_previous_paths, *d_result_tensor, *d_result_values;
    float *d_output_paths, *d_output_values;
    int *d_match_count;

    CHECK_CUDA(cudaMalloc(&d_output_paths, output_paths_size));
    CHECK_CUDA(cudaMalloc(&d_output_values, output_values_size));
    CHECK_CUDA(cudaMalloc(&d_match_count, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_match_count, 0, sizeof(int)));

    // Copiar datos a device o usar punteros existentes
    if (previous_paths.is_device_ptr)
    {
        d_previous_paths = previous_paths.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_previous_paths, prev_size));
        CHECK_CUDA(cudaMemcpy(d_previous_paths, previous_paths.data, prev_size, cudaMemcpyHostToDevice));
    }

    if (result_tensor.is_device_ptr)
    {
        d_result_tensor = result_tensor.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_result_tensor, curr_size));
        CHECK_CUDA(cudaMemcpy(d_result_tensor, result_tensor.data, curr_size, cudaMemcpyHostToDevice));
    }

    if (result_values.is_device_ptr)
    {
        d_result_values = result_values.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_result_values, values_size));
        CHECK_CUDA(cudaMemcpy(d_result_values, result_values.data, values_size, cudaMemcpyHostToDevice));
    }

    // Configurar kernel
    dim3 block_size(16, 16);
    dim3 grid_size((num_prev_paths + block_size.x - 1) / block_size.x,
                   (num_current_tensor + block_size.y - 1) / block_size.y);

    // Lanzar kernel
    find_path_matches_kernel<<<grid_size, block_size>>>(
        d_previous_paths, d_result_tensor, d_result_values,
        d_output_paths, d_output_values, d_match_count,
        num_prev_paths, num_current_tensor, prev_cols, current_cols, iteration);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Obtener número de matches
    int match_count;
    CHECK_CUDA(cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (match_count > 0)
    {
        // Alocar memoria host para resultados
        size_t final_paths_size = match_count * new_cols * sizeof(float);
        size_t final_values_size = match_count * sizeof(float);

        float *h_output_paths = (float *)malloc(final_paths_size);
        float *h_output_values = (float *)malloc(final_values_size);

        // Copiar resultados a host
        CHECK_CUDA(cudaMemcpy(h_output_paths, d_output_paths, final_paths_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_output_values, d_output_values, final_values_size, cudaMemcpyDeviceToHost));

        // Configurar TensorResult de salida
        paths.data = h_output_paths;
        paths.is_device_ptr = false;
        paths.owns_memory = true;
        paths.batch = previous_paths.batch; // Mantener el batch de previous_paths
        paths.M = match_count;              // 4 + iteration columnas
        paths.N = new_cols;
        paths.K = 1;

        matched_values.data = h_output_values;
        matched_values.is_device_ptr = false;
        matched_values.owns_memory = true;
        matched_values.batch = previous_paths.batch; // Mantener el batch de previous_paths
        matched_values.M = 1;
        matched_values.N = match_count;
        matched_values.K = 1;
    }
    else
    {
        printf("Error: No se encontraron matches\n");

        // Configurar TensorResult vacíos
        paths.data = nullptr;
        paths.is_device_ptr = false;
        paths.batch = 0;
        paths.M = 0;
        paths.N = 0;
        paths.K = 0;

        matched_values.data = nullptr;
        matched_values.is_device_ptr = false;
        matched_values.batch = 0;
        matched_values.M = 0;
        matched_values.N = 0;
        matched_values.K = 0;
    }

    // Limpiar memoria device
    if (d_output_paths)
        cudaFree(d_output_paths);
    if (d_output_values)
        cudaFree(d_output_values);
    if (d_match_count)
        cudaFree(d_match_count);

    // Limpiar copias temporales si se crearon
    if (!previous_paths.is_device_ptr && d_previous_paths)
        cudaFree(d_previous_paths);
    if (!result_tensor.is_device_ptr && d_result_tensor)
        cudaFree(d_result_tensor);
    if (!result_values.is_device_ptr && d_result_values)
        cudaFree(d_result_values);
}