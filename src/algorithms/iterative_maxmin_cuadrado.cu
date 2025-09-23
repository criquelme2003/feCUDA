#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <cuda_utils.cuh>
#include <utils.cuh>
#include <core/types.cuh>
#include <algorithms/maxmin.cuh>
#include <algorithms/indices.cuh>
#include <algorithms/paths.cuh>

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
        printf("---Calculando orden %d\n", i);

        // Calcular min_result y maxmin_conjugado
        TensorResult min_result, maxmin_conjugado;
        maxmin(gen_tensor, original_tensor, maxmin_conjugado, min_result, false);

        // Marcar ownership correcto
        if (maxmin_conjugado.data)
            maxmin_conjugado.owns_memory = true;
        if (min_result.data)
            min_result.owns_memory = true;

        // Calcular prima
        TensorResult prima;
        calculate_prima(maxmin_conjugado, gen_tensor, prima);

        // Calcular indices
        TensorResult result_tensor, result_values;
        indices(min_result, prima, result_tensor, result_values, thr);

        // CORRECCIÓN 1: Crear copias para pure_paths ANTES de usar los originales
        TensorResult pure_tensor_copy = copy_tensor(result_tensor);
        TensorResult pure_values_copy = copy_tensor(result_values);

        pure_tensor_paths.push_back(std::move(pure_tensor_copy));
        pure_values_paths.push_back(std::move(pure_values_copy));

        // Ahora result_tensor y result_values siguen siendo válidos

        // Guardar primeros resultados para debug
        if (i == 0)
        {
            save_tensor_4d_as_file(result_tensor.data, result_tensor.batch,
                                   result_tensor.M, result_tensor.N, result_tensor.K, "paths0.txt");
            save_tensor_4d_as_file(result_values.data, result_values.batch,
                                   result_values.M, result_values.N, result_values.K, "values0.txt");
        }

        // Verificar si se encontraron efectos
        if (result_tensor.data == nullptr || result_tensor.M == 0)
        {
            if (i == 0)
            {
                printf("Error: No se encontraron efectos con threshold %.4f\n", thr);
                // Limpiar y retornar
                safe_tensor_cleanup(original_tensor);
                safe_tensor_cleanup(gen_tensor);
                safe_tensor_cleanup(min_result);
                safe_tensor_cleanup(maxmin_conjugado);
                safe_tensor_cleanup(prima);
                safe_tensor_cleanup(result_tensor);
                safe_tensor_cleanup(result_values);
                return;
            }
            else
            {
                printf("Los efectos solo fueron encontrados hasta el orden %d\n", i + 1);
                break;
            }
        }

        // CORRECCIÓN 2: Manejo correcto de caminos para i >= 1
        if (i >= 1)
        {
            printf("Iteracion %d: Construyendo caminos...\n", i);

            // CORRECCIÓN 3: Usar índices correctos para previous_paths
            TensorResult previous_paths;
            if (i == 1)
            {
                // Para i=1, usar pure_tensor_paths[0] (orden 0)
                previous_paths = copy_tensor(pure_tensor_paths[0]);
            }
            else
            {
                // Para i>1, usar result_tensor_paths[i-2] (último resultado procesado)
                previous_paths = copy_tensor(result_tensor_paths[i - 2]);
            }

            TensorResult paths, values;
            armar_caminos_batch(previous_paths, // previous_paths: caminos previos
                          result_tensor,  // current_paths: nuevos caminos encontrados
                          result_values,  // current_values: valores correspondientes
                          paths, values, i);

            // Debug para armar_caminos
            if (paths.data == nullptr || paths.M == 0)
            {
                printf("Solo se encontraron efectos hasta el orden %d\n", i);
                safe_tensor_cleanup(previous_paths);
                break;
            }

            // CORRECCIÓN 4: Agregar a result_paths de forma segura
            result_tensor_paths.push_back(std::move(paths));
            result_values_paths.push_back(std::move(values));

            safe_tensor_cleanup(previous_paths);
        }

        // Limpiar gen_tensor y reemplazar por maxmin_conjugado
        safe_tensor_cleanup(gen_tensor);
        gen_tensor = std::move(maxmin_conjugado);

        // Limpiar temporales
        safe_tensor_cleanup(min_result);
        safe_tensor_cleanup(prima);
        safe_tensor_cleanup(result_tensor);
        safe_tensor_cleanup(result_values);
    }

    // CORRECCIÓN 5: Insertar orden 0 de forma correcta
    // En lugar de copiar de pure_tensor_paths[0] que puede estar corrompido,
    // mantener una copia separada desde el principio
    if (!pure_tensor_paths.empty() && pure_tensor_paths[0].data != nullptr)
    {
        result_tensor_paths.insert(result_tensor_paths.begin(),
                                   copy_tensor(pure_tensor_paths[0]));
        result_values_paths.insert(result_values_paths.begin(),
                                   copy_tensor(pure_values_paths[0]));
    }

    // Limpiar memoria restante
    safe_tensor_cleanup(original_tensor);
    safe_tensor_cleanup(gen_tensor);

    CudaUtils::cuda_cleanup_and_check();
}


