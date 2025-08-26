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
    prima.owns_memory = true; // Esta función es responsable de liberar la memoria

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
    

    // Verificar estado del dispositivo CUDA
    cudaError_t deviceError = cudaDeviceSynchronize();
    if (deviceError != cudaSuccess)
    {
        printf("Error: El dispositivo CUDA no está disponible [iterative_maxmin_cuadrado]: %s\n", cudaGetErrorString(deviceError));
        return;
    }

    // Validaciones
    if (thr < 0.0f || thr > 1.0f)
    {
        printf("Error:El threshold debe estar en el rango [0,1] (thr = %.2f) \n", thr);
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
    TensorResult original_tensor = copy_tensor(tensor);

    // Inicializar gen_tensor como copia del tensor original
    TensorResult gen_tensor = copy_tensor(original_tensor);

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

        // Calcular indices con prima y threshold
        TensorResult result_tensor, result_values;

        indices(min_result, prima, result_tensor, result_values, thr);
        pure_tensor_paths.push_back(result_tensor);
        pure_values_paths.push_back(result_values);

        // Mover los originales a las listas de resultados (que serán procesados)
        // result_tensor_paths.push_back(std::move(result_tensor));
        // result_values_paths.push_back(std::move(result_values));

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
            printf("Iteracion %d: Construyendo caminos...\n", i);

            TensorResult previous_paths = (i > 1)
                                              ? copy_tensor(result_tensor_paths.back())
                                              : copy_tensor(pure_tensor_paths[0]);

            // Para construir caminos del
            // - previous_paths: último elemento de result_tensor_paths (resultados procesados)
            // - current_paths: último elemento de pure_tensor_paths (resultados puros)
            // - current_values: último elemento de pure_values_paths (valores puros)
            TensorResult paths, values;

            armar_caminos(previous_paths, // previous_paths: último procesado
                          result_tensor,  // current_paths: último puro
                          result_values,  // current_values: valores puros
                          paths, values, i);

            if (paths.batch == 0)
            {
                printf("Solo se encontraron efectos hasta el orden %d\n", i);

                // Limpiar memoria y retornar
                safe_tensor_cleanup(original_tensor);
                safe_tensor_cleanup(gen_tensor);
                safe_tensor_cleanup(min_result);
                safe_tensor_cleanup(maxmin_conjugado);
                safe_tensor_cleanup(prima);
                i = order; // Forzar salida del loop externo
            }

            result_tensor_paths.push_back(std::move(paths));
            result_values_paths.push_back(std::move(values));
        }
        safe_tensor_cleanup(gen_tensor);
        // Reemplazar gen_tensor por maxmin_conjugado (mover ownership)
        gen_tensor = std::move(maxmin_conjugado);
        // Limpiar temporales restantes
        safe_tensor_cleanup(min_result);
        safe_tensor_cleanup(prima);
        safe_tensor_cleanup(result_tensor);
        safe_tensor_cleanup(result_values);
    }

    // Limpiar memoria
    safe_tensor_cleanup(original_tensor);
    safe_tensor_cleanup(gen_tensor);

    result_tensor_paths.insert(result_tensor_paths.begin(),
                               copy_tensor(pure_tensor_paths.front()));

    result_values_paths.insert(result_values_paths.begin(),
                               copy_tensor(pure_values_paths.front()));

    // Limpiar y verificar dispositivo CUDA
    cuda_cleanup_and_check();
}
