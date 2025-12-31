#include <cuda_runtime.h>
#include <float.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <core/types.cuh>
#include <utils.cuh>


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

void armar_caminos_original(const TensorResult &previous_paths, const TensorResult &result_tensor,
                            const TensorResult &result_values, TensorResult &paths,
                            TensorResult &matched_values, int iteration, bool keep_in_device)
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
    }

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

    // Requerimiento total aproximado (paths + values + entradas + contador)
    size_t required_bytes = output_paths_size + output_values_size + prev_size + curr_size + values_size + sizeof(int);
    size_t free_mem = 0, total_mem = 0;
    cudaError_t mem_status = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (mem_status == cudaSuccess && required_bytes > static_cast<size_t>(free_mem * 0.8))
    {
        printf("Error: memoria insuficiente en armar_caminos (necesita ~%.2f MB, libre %.2f MB)\n",
               required_bytes / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
        return;
    }

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
        if (keep_in_device)
        {
            // Mantener resultados en GPU con tamaño optimizado
            float *d_output_paths_final, *d_output_values_final;
            size_t final_paths_size = match_count * new_cols * sizeof(float);
            size_t final_values_size = match_count * sizeof(float);

            CHECK_CUDA(cudaMalloc(&d_output_paths_final, final_paths_size));
            CHECK_CUDA(cudaMalloc(&d_output_values_final, final_values_size));
            CHECK_CUDA(cudaMemcpy(d_output_paths_final, d_output_paths, final_paths_size, cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_output_values_final, d_output_values, final_values_size, cudaMemcpyDeviceToDevice));

            // Configurar TensorResult en GPU
            paths.data = d_output_paths_final;
            paths.is_device_ptr = true;
            paths.owns_memory = true;
            paths.batch = previous_paths.batch;
            paths.M = match_count;
            paths.N = new_cols;
            paths.K = 1;

            matched_values.data = d_output_values_final;
            matched_values.is_device_ptr = true;
            matched_values.owns_memory = true;
            matched_values.batch = previous_paths.batch;
            matched_values.M = 1;
            matched_values.N = match_count;
            matched_values.K = 1;
        }
        else
        {
            // Transferir resultados a CPU
            size_t final_paths_size = match_count * new_cols * sizeof(float);
            size_t final_values_size = match_count * sizeof(float);

            float *h_output_paths = (float *)malloc(final_paths_size);
            float *h_output_values = (float *)malloc(final_values_size);

            if (!h_output_paths || !h_output_values)
            {
                printf("Error: No se pudo alocar memoria host para resultados\n");
                if (h_output_paths)
                    free(h_output_paths);
                if (h_output_values)
                    free(h_output_values);
            }
            else
            {
                CHECK_CUDA(cudaMemcpy(h_output_paths, d_output_paths, final_paths_size, cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_output_values, d_output_values, final_values_size, cudaMemcpyDeviceToHost));

                // Configurar TensorResult en CPU
                paths.data = h_output_paths;
                paths.is_device_ptr = false;
                paths.owns_memory = true;
                paths.batch = previous_paths.batch;
                paths.M = match_count;
                paths.N = new_cols;
                paths.K = 1;

                matched_values.data = h_output_values;
                matched_values.is_device_ptr = false;
                matched_values.owns_memory = true;
                matched_values.batch = previous_paths.batch;
                matched_values.M = 1;
                matched_values.N = match_count;
                matched_values.K = 1;
            }
        }
    }
    else
    {
        printf("Error: No se encontraron matches\n");

        // Configurar TensorResult vacíos
        paths.data = nullptr;
        paths.is_device_ptr = false;
        paths.owns_memory = false;
        paths.batch = 0;
        paths.M = 0;
        paths.N = 0;
        paths.K = 0;

        matched_values.data = nullptr;
        matched_values.is_device_ptr = false;
        matched_values.owns_memory = false;
        matched_values.batch = 0;
        matched_values.M = 0;
        matched_values.N = 0;
        matched_values.K = 0;
    }

    // Limpiar memoria device temporal
    CHECK_CUDA(cudaFree(d_output_paths));
    CHECK_CUDA(cudaFree(d_output_values));
    CHECK_CUDA(cudaFree(d_match_count));

    // Limpiar copias temporales si se crearon
    if (!previous_paths.is_device_ptr)
        CHECK_CUDA(cudaFree(d_previous_paths));
    if (!result_tensor.is_device_ptr)
        CHECK_CUDA(cudaFree(d_result_tensor));
    if (!result_values.is_device_ptr)
        CHECK_CUDA(cudaFree(d_result_values));
}

// SOLUCIÓN 1: Procesamiento por lotes (batches)
void armar_caminos_batch(const TensorResult &previous_paths, const TensorResult &result_tensor,
                         const TensorResult &result_values, TensorResult &paths,
                         TensorResult &matched_values, int iteration, int batch_size, bool keep_in_device)
{
    // Validaciones básicas
    if (previous_paths.data == nullptr || result_tensor.data == nullptr || result_values.data == nullptr)
    {
        printf("Error: Uno de los tensores de entrada es nulo\n");
        return;
    }

    int num_prev_paths = previous_paths.M;
    int prev_cols = previous_paths.N;
    int num_current_tensor = result_tensor.M;
    int current_cols = result_tensor.N;
    int new_cols = 4 + iteration;

    // Variables para acumulación
    int total_matches = 0;
    float *d_accumulated_paths = nullptr;
    float *d_accumulated_values = nullptr;

    // Si vamos a mantener en device, usamos buffers GPU para acumular
    bool accumulate_on_gpu = keep_in_device;
    std::vector<float> cpu_paths, cpu_values; // Fallback para CPU

    // Procesar en lotes más pequeños
    for (int batch_start = 0; batch_start < num_prev_paths; batch_start += batch_size)
    {
        int current_batch_size = std::min(batch_size, num_prev_paths - batch_start);

        // Crear sub-tensor para este lote
        TensorResult batch_paths;
        batch_paths.data = previous_paths.data + (batch_start * prev_cols);
        batch_paths.is_device_ptr = previous_paths.is_device_ptr;
        batch_paths.owns_memory = false;
        batch_paths.M = current_batch_size;
        batch_paths.N = prev_cols;
        batch_paths.batch = previous_paths.batch;
        batch_paths.K = previous_paths.K;

        // Procesar este lote - usar keep_in_device para el lote si acumulamos en GPU
        TensorResult batch_result_paths, batch_result_values;
        armar_caminos_original(batch_paths, result_tensor, result_values,
                               batch_result_paths, batch_result_values, iteration, accumulate_on_gpu);

        // Acumular resultados si hay matches
        if (batch_result_paths.data != nullptr && batch_result_paths.M > 0)
        {
            int batch_matches = batch_result_paths.M;
            size_t paths_size = batch_matches * new_cols;
            size_t values_size = batch_matches;

            if (accumulate_on_gpu && batch_result_paths.is_device_ptr)
            {
                // Acumular en GPU
                size_t new_paths_bytes = (total_matches + batch_matches) * new_cols * sizeof(float);
                size_t new_values_bytes = (total_matches + batch_matches) * sizeof(float);

                // Crear nuevos buffers más grandes
                float *d_new_paths, *d_new_values;
                CHECK_CUDA(cudaMalloc(&d_new_paths, new_paths_bytes));
                CHECK_CUDA(cudaMalloc(&d_new_values, new_values_bytes));

                // Copiar datos previos si existen
                if (total_matches > 0)
                {
                    size_t prev_paths_bytes = total_matches * new_cols * sizeof(float);
                    size_t prev_values_bytes = total_matches * sizeof(float);

                    CHECK_CUDA(cudaMemcpy(d_new_paths, d_accumulated_paths, prev_paths_bytes, cudaMemcpyDeviceToDevice));
                    CHECK_CUDA(cudaMemcpy(d_new_values, d_accumulated_values, prev_values_bytes, cudaMemcpyDeviceToDevice));

                    // Liberar buffers previos
                    CHECK_CUDA(cudaFree(d_accumulated_paths));
                    CHECK_CUDA(cudaFree(d_accumulated_values));
                }

                // Copiar nuevos datos al final
                size_t batch_paths_bytes = batch_matches * new_cols * sizeof(float);
                size_t batch_values_bytes = batch_matches * sizeof(float);

                CHECK_CUDA(cudaMemcpy(d_new_paths + (total_matches * new_cols),
                                      batch_result_paths.data, batch_paths_bytes, cudaMemcpyDeviceToDevice));
                CHECK_CUDA(cudaMemcpy(d_new_values + total_matches,
                                      batch_result_values.data, batch_values_bytes, cudaMemcpyDeviceToDevice));

                // Actualizar punteros acumulados
                d_accumulated_paths = d_new_paths;
                d_accumulated_values = d_new_values;
            }
            else
            {
                // Acumular en CPU (código previo)
                if (batch_result_paths.is_device_ptr)
                {
                    // GPU → CPU para acumular
                    float *h_temp_paths = (float *)malloc(paths_size * sizeof(float));
                    float *h_temp_values = (float *)malloc(values_size * sizeof(float));

                    CHECK_CUDA(cudaMemcpy(h_temp_paths, batch_result_paths.data, paths_size * sizeof(float), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(h_temp_values, batch_result_values.data, values_size * sizeof(float), cudaMemcpyDeviceToHost));

                    cpu_paths.insert(cpu_paths.end(), h_temp_paths, h_temp_paths + paths_size);
                    cpu_values.insert(cpu_values.end(), h_temp_values, h_temp_values + values_size);

                    free(h_temp_paths);
                    free(h_temp_values);
                }
                else
                {
                    // CPU → CPU
                    cpu_paths.insert(cpu_paths.end(),
                                     batch_result_paths.data,
                                     batch_result_paths.data + paths_size);
                    cpu_values.insert(cpu_values.end(),
                                      batch_result_values.data,
                                      batch_result_values.data + values_size);
                }
            }

            total_matches += batch_matches;

            // Transferir ownership para evitar double free
            batch_result_paths.transfer_ownership(false);
            batch_result_values.transfer_ownership(false);
        }
    }

    // Crear tensores de salida finales
    if (total_matches > 0)
    {
        if (accumulate_on_gpu)
        {
            // Ya tenemos todo en GPU
            paths.data = d_accumulated_paths;
            paths.is_device_ptr = true;
            paths.owns_memory = true;

            matched_values.data = d_accumulated_values;
            matched_values.is_device_ptr = true;
            matched_values.owns_memory = true;
        }
        else if (keep_in_device)
        {
            // Transferir acumulación CPU → GPU
            size_t final_paths_size = cpu_paths.size() * sizeof(float);
            size_t final_values_size = cpu_values.size() * sizeof(float);

            CHECK_CUDA(cudaMalloc(&paths.data, final_paths_size));
            CHECK_CUDA(cudaMalloc(&matched_values.data, final_values_size));
            CHECK_CUDA(cudaMemcpy(paths.data, cpu_paths.data(), final_paths_size, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(matched_values.data, cpu_values.data(), final_values_size, cudaMemcpyHostToDevice));

            paths.is_device_ptr = true;
            paths.owns_memory = true;
            matched_values.is_device_ptr = true;
            matched_values.owns_memory = true;
        }
        else
        {
            // Mantener en CPU
            paths.data = (float *)malloc(cpu_paths.size() * sizeof(float));
            matched_values.data = (float *)malloc(cpu_values.size() * sizeof(float));

            std::copy(cpu_paths.begin(), cpu_paths.end(), paths.data);
            std::copy(cpu_values.begin(), cpu_values.end(), matched_values.data);

            paths.is_device_ptr = false;
            paths.owns_memory = true;
            matched_values.is_device_ptr = false;
            matched_values.owns_memory = true;
        }

        // Configurar dimensiones
        paths.M = total_matches;
        paths.N = new_cols;
        paths.batch = previous_paths.batch;
        paths.K = 1;

        matched_values.M = 1;
        matched_values.N = total_matches;
        matched_values.batch = previous_paths.batch;
        matched_values.K = 1;
    }
    else
    {
        // Sin matches
        paths.data = nullptr;
        paths.is_device_ptr = false;
        paths.owns_memory = false;
        paths.M = 0;
        paths.N = 0;
        paths.batch = 0;
        paths.K = 0;

        matched_values.data = nullptr;
        matched_values.is_device_ptr = false;
        matched_values.owns_memory = false;
        matched_values.M = 0;
        matched_values.N = 0;
        matched_values.batch = 0;
        matched_values.K = 0;
    }
}
