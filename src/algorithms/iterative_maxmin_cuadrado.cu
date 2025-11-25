#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <string.h>
#include <cuda_utils.cuh>
#include <utils.cuh>
#include <core/types.cuh>
#include <headers.cuh>

void iterative_maxmin_cuadrado(const TensorResult &tensor, float thr, int order,
                               std::vector<TensorResult> &result_tensor_paths,
                               std::vector<TensorResult> &result_values_paths,
                               std::vector<TensorResult> &pure_tensor_paths,
                               std::vector<TensorResult> &pure_values_paths,
                               bool keep_in_device)
{
    // Verificar estado del dispositivo CUDA

    // Validaciones
    if (thr < 0.0f || thr > 1.0f)
    {
        printf("Error: El threshold debe estar en el rango [0,1] (thr = %.2f)\n", thr);
        return;
    }

    if (order <= 1)
    {
        printf("Error: El order debe ser mayor que 1\n");
        return;
    }

    // Validar memoria disponible antes de iniciar
    {
        size_t host_required = static_cast<size_t>(tensor.batch) * tensor.M * tensor.M * tensor.M * sizeof(float);
        unsigned long long mem_available_kb = 0;
        FILE *f = fopen("/proc/meminfo", "r");
        if (f)
        {
            char line[256];
            while (fgets(line, sizeof(line), f))
            {
                if (strncmp(line, "MemAvailable:", 13) == 0)
                {
                    unsigned long long kb = 0;
                    if (sscanf(line + 13, "%llu", &kb) == 1)
                    {
                        mem_available_kb = kb;
                    }
                    break;
                }
            }
            fclose(f);
        }
        if (mem_available_kb > 0)
        {
            unsigned long long available_bytes = mem_available_kb * 1024ULL;
            double req_mb = host_required / (1024.0 * 1024.0);
            double avail_mb = available_bytes / (1024.0 * 1024.0);
            printf("iterative_maxmin_cuadrado: MemAvailable=%.2f MB, host_required=%.2f MB\n", avail_mb, req_mb);
            if (host_required > available_bytes * 0.8)
            {
                printf("iterative_maxmin_cuadrado: RAM insuficiente. Necesita ~%.2f MB, disponible ~%.2f MB\n",
                       req_mb, avail_mb);
                return;
            }
        }

        size_t gpu_required = static_cast<size_t>(tensor.M) * tensor.M * sizeof(float);
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess && free_mem > 0)
        {
            printf("iterative_maxmin_cuadrado: GPU libre=%.2f MB, requerido (MxM)=%.2f MB\n",
                   free_mem / (1024.0 * 1024.0), gpu_required / (1024.0 * 1024.0));
            if (gpu_required > static_cast<size_t>(free_mem * 0.8))
            {
                printf("iterative_maxmin_cuadrado: Memoria GPU insuficiente. Requiere ~%.2f MB, libre ~%.2f MB\n",
                       gpu_required / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
                return;
            }
        }
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
        maxmin(gen_tensor, original_tensor, maxmin_conjugado, min_result, keep_in_device);

                // Marcar ownership correcto
        if (maxmin_conjugado.data)
            maxmin_conjugado.owns_memory = true;
        if (min_result.data)
            min_result.owns_memory = true;

        // Calcular prima
        TensorResult prima;
        calculate_prima(maxmin_conjugado, gen_tensor, prima, keep_in_device);
        // Calcular indices
        TensorResult result_tensor, result_values;
        indices(min_result, prima, result_tensor, result_values, thr, keep_in_device);

        // Para vectores de almacenamiento (siempre CPU para evitar problemas con std::vector)
        TensorResult pure_tensor_copy = copy_tensor_to_cpu(result_tensor);
        TensorResult pure_values_copy = copy_tensor_to_cpu(result_values);
        pure_tensor_paths.push_back(std::move(pure_tensor_copy));
        pure_values_paths.push_back(std::move(pure_values_copy));

        // Ahora result_tensor y result_values siguen siendo válidos

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
            // CORRECCIÓN 3: Usar índices correctos para previous_paths
            TensorResult previous_paths;
            if (i == 1)
            {
                if (keep_in_device)
                {
                    previous_paths = copy_tensor_to_gpu(pure_tensor_paths[0]); // CPU → GPU
                }
                else
                {
                    previous_paths = copy_tensor(pure_tensor_paths[0]); // CPU → CPU
                }
            }
            else
            {
                previous_paths = copy_tensor(result_tensor_paths[i - 2]); // Mantiene ubicación
            }

            TensorResult paths, values;
            armar_caminos_batch(previous_paths, // previous_paths: caminos previos
                                result_tensor,  // current_paths: nuevos caminos encontrados
                                result_values,  // current_values: valores correspondientes
                                paths, values, i, 1000, keep_in_device);

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
