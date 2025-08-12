#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <vector>
#include "utils.cuh"
#include "types.cuh"
#include "headers.cuh"

// Kernel para calcular prima = maxmin_conjugado - gen_tensor
__global__ void calculate_prima_kernel(float *maxmin_conjugado, float *gen_tensor,
                                       float *prima, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        prima[idx] = maxmin_conjugado[idx] - gen_tensor[idx];
    }
}

// Función para calcular prima (efectos de n generación)
void calculate_prima(const TensorResult &maxmin_conjugado, const TensorResult &gen_tensor,
                     TensorResult &prima)
{
    // Verificar que las dimensiones coincidan
    if (maxmin_conjugado.batch != gen_tensor.batch ||
        maxmin_conjugado.M != gen_tensor.M ||
        maxmin_conjugado.N != gen_tensor.N)
    {
        printf("Error: Dimensiones no coinciden para calcular prima\n");
        return;
    }

    int total_elements = maxmin_conjugado.batch * maxmin_conjugado.M * maxmin_conjugado.N;
    size_t size = total_elements * sizeof(float);

    // Alocar memoria para prima
    float *h_prima = (float *)malloc(size);
    float *d_maxmin_conjugado, *d_gen_tensor, *d_prima;

    // Alocar memoria device
    CHECK_CUDA(cudaMalloc(&d_prima, size));

    // Copiar datos a device
    if (maxmin_conjugado.is_device_ptr)
    {
        d_maxmin_conjugado = maxmin_conjugado.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_maxmin_conjugado, size));
        CHECK_CUDA(cudaMemcpy(d_maxmin_conjugado, maxmin_conjugado.data, size, cudaMemcpyHostToDevice));
    }

    if (gen_tensor.is_device_ptr)
    {
        d_gen_tensor = gen_tensor.data;
    }
    else
    {
        CHECK_CUDA(cudaMalloc(&d_gen_tensor, size));
        CHECK_CUDA(cudaMemcpy(d_gen_tensor, gen_tensor.data, size, cudaMemcpyHostToDevice));
    }

    // Lanzar kernel
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    calculate_prima_kernel<<<grid_size, block_size>>>(d_maxmin_conjugado, d_gen_tensor,
                                                      d_prima, total_elements);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copiar resultado a host
    CHECK_CUDA(cudaMemcpy(h_prima, d_prima, size, cudaMemcpyDeviceToHost));

    // Configurar TensorResult de salida
    prima.data = h_prima;
    prima.is_device_ptr = false;
    prima.batch = maxmin_conjugado.batch;
    prima.M = maxmin_conjugado.M;
    prima.N = maxmin_conjugado.N;
    prima.K = maxmin_conjugado.K;

    // Limpiar memoria device
    cudaFree(d_prima);
    if (!maxmin_conjugado.is_device_ptr)
        cudaFree(d_maxmin_conjugado);
    if (!gen_tensor.is_device_ptr)
        cudaFree(d_gen_tensor);
}

// Función principal iterative_maxmin_cuadrado
void iterative_maxmin_cuadrado(const TensorResult &tensor, float thr, int order,
                               std::vector<TensorResult> &result_tensor_paths,
                               std::vector<TensorResult> &result_values_paths,
                               std::vector<TensorResult> &pure_tensor_paths,
                               std::vector<TensorResult> &pure_values_paths)
{
    // Validaciones
    if (thr < 0.0f || thr > 1.0f)
    {
        printf("Error: El threshold debe estar en el rango [0,1]\n");
        return;
    }

    if (order <= 1)
    {
        printf("Error: El order debe ser mayor que 1\n");
        return;
    }

    if (tensor.data == nullptr)
    {
        printf("Error: Tensor de entrada es nulo\n");
        return;
    }

    // Copiar tensor original (asignar ownership)
    TensorResult original_tensor;
    size_t tensor_size = tensor.batch * tensor.M * tensor.N * tensor.K * sizeof(float);
    original_tensor.data = (float *)malloc(tensor_size);
    memcpy(original_tensor.data, tensor.data, tensor_size);
    original_tensor.is_device_ptr = false;
    original_tensor.batch = tensor.batch;
    original_tensor.M = tensor.M;
    original_tensor.N = tensor.N;
    original_tensor.K = tensor.K;
    original_tensor.owns_memory = true;

    // Inicializar gen_tensor como copia del tensor original
    TensorResult gen_tensor;
    gen_tensor.data = (float *)malloc(tensor_size);
    memcpy(gen_tensor.data, tensor.data, tensor_size);
    gen_tensor.is_device_ptr = false;
    gen_tensor.batch = tensor.batch;
    gen_tensor.M = tensor.M;
    gen_tensor.N = tensor.N;
    gen_tensor.K = tensor.K;
    gen_tensor.owns_memory = true;

    // Limpiar vectores de salida
    result_tensor_paths.clear();
    result_values_paths.clear();
    pure_tensor_paths.clear();
    pure_values_paths.clear();

    for (int i = 0; i < order - 1; i++)
    {
        // Calcular min_result y maxmin_conjugado
        TensorResult min_result, maxmin_conjugado;
        maxmin(gen_tensor, original_tensor, maxmin_conjugado, min_result, false);

        // Los resultados de maxmin asignan memoria host sin marcar ownership; marcarlo.
        if (maxmin_conjugado.data)
            maxmin_conjugado.owns_memory = true;
        if (min_result.data)
            min_result.owns_memory = true;

        // Calcular prima = maxmin_conjugado - gen_tensor
        TensorResult prima;
        calculate_prima(maxmin_conjugado, gen_tensor, prima);
        if (prima.data)
            prima.owns_memory = true;

        // Calcular indices con prima y threshold
        TensorResult result_tensor, result_values;
        indices(min_result, prima, result_tensor, result_values, thr);
        if (result_tensor.data)
            result_tensor.owns_memory = true;
        if (result_values.data)
            result_values.owns_memory = true;

        // Hacer copias para las listas puras antes de mover los originales
        TensorResult pure_tensor_copy, pure_values_copy;

        // Copiar result_tensor a pure_tensor_copy
        size_t tensor_size = result_tensor.batch * result_tensor.M * result_tensor.N * result_tensor.K * sizeof(float);
        pure_tensor_copy.data = (float *)malloc(tensor_size);
        memcpy(pure_tensor_copy.data, result_tensor.data, tensor_size);
        pure_tensor_copy.is_device_ptr = false;
        pure_tensor_copy.batch = result_tensor.batch;
        pure_tensor_copy.M = result_tensor.M;
        pure_tensor_copy.N = result_tensor.N;
        pure_tensor_copy.K = result_tensor.K;
        pure_tensor_copy.owns_memory = true;

        // Copiar result_values a pure_values_copy
        size_t values_size = result_values.batch * result_values.M * result_values.N * result_values.K * sizeof(float);
        pure_values_copy.data = (float *)malloc(values_size);
        memcpy(pure_values_copy.data, result_values.data, values_size);
        pure_values_copy.is_device_ptr = false;
        pure_values_copy.batch = result_values.batch;
        pure_values_copy.M = result_values.M;
        pure_values_copy.N = result_values.N;
        pure_values_copy.K = result_values.K;
        pure_values_copy.owns_memory = true;

        // Guardar las copias puras
        pure_tensor_paths.push_back(std::move(pure_tensor_copy));
        pure_values_paths.push_back(std::move(pure_values_copy));

        // Mover los originales a las listas de resultados (que serán procesados)
        result_tensor_paths.push_back(std::move(result_tensor));
        result_values_paths.push_back(std::move(result_values));

        // Verificar si se encontraron efectos (usar listas puras para verificación)
        if (pure_values_paths.back().data == nullptr || pure_values_paths.back().batch == 0)
        {
            if (i == 0)
            {
                printf("Error: No se encontraron efectos con threshold %.4f\n", thr);
                // Limpiar memoria y retornar
                safe_tensor_cleanup(original_tensor);
                safe_tensor_cleanup(gen_tensor);
                safe_tensor_cleanup(min_result);
                safe_tensor_cleanup(maxmin_conjugado);
                safe_tensor_cleanup(prima);
                return;
            }
            else
            {
                printf("Los efectos solo fueron encontrados hasta el orden %d\n", i + 1);
                break;
            }
        }

        // Para el primer orden (i == 0), solo agregamos directamente los paths encontrados
        // Para órdenes superiores (i >= 1), construimos caminos usando armar_caminos
        if (i >= 1)
        {
            int current_order = i + 1; // El orden real es i + 1
            printf("Orden %d: Construyendo caminos...\n", current_order);

            // Para construir caminos del orden actual, necesitamos:
            // - previous_paths: último elemento de result_tensor_paths (resultados procesados)
            // - current_paths: último elemento de pure_tensor_paths (resultados puros)
            // - current_values: último elemento de pure_values_paths (valores puros)
            TensorResult paths, values;

            armar_caminos(result_tensor_paths.back(), // previous_paths: último procesado
                          pure_tensor_paths.back(),   // current_paths: último puro
                          pure_values_paths.back(),   // current_values: valores puros
                          paths, values, i);

            if (paths.batch == 0)
            {
                printf("Solo se encontraron efectos hasta el orden %d\n", current_order - 1);
                // Limpiar memoria y retornar
                safe_tensor_cleanup(original_tensor);
                safe_tensor_cleanup(gen_tensor);
                safe_tensor_cleanup(min_result);
                safe_tensor_cleanup(maxmin_conjugado);
                safe_tensor_cleanup(prima);
                return;
            }

            if (paths.data)
                paths.owns_memory = true;
            if (values.data)
                values.owns_memory = true;

            // Reemplazar los resultados actuales con los caminos construidos
            // En lugar de hacer push_back, reemplazamos el elemento actual
            result_tensor_paths.back() = std::move(paths);
            result_values_paths.back() = std::move(values);
        }
        // Reemplazar gen_tensor por maxmin_conjugado (mover ownership)
        safe_tensor_cleanup(gen_tensor);
        gen_tensor = std::move(maxmin_conjugado);
        // Limpiar temporales restantes
        safe_tensor_cleanup(min_result);
        safe_tensor_cleanup(prima);
    }

    // Limpiar memoria
    safe_tensor_cleanup(original_tensor);
    safe_tensor_cleanup(gen_tensor);
    // Nota: los elementos en result_tensor_paths / result_values_paths mantienen ownership.

    // Limpiar y verificar dispositivo CUDA
    cuda_cleanup_and_check();
}
