#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <tuple>
#include "headers.cuh"
#include "utils.cuh"
#include "types.cuh"
#include <chrono>

// Declaraciones de funciones
void ejecutar_benchmark_original();

// Estructura para almacenar tiempos de cada función
struct FunctionTimes
{
    float maxmin_accomulative = 0;
    float indices_accomulative = 0;
    float armar_caminos_accomulative = 0;
    float total_accomulative = 0;
};

int main()
{
    printf("=== SISTEMA DE PRUEBAS FECUDA ===\n");

    ejecutar_benchmark_original();

    return 0;
}

void ejecutar_benchmark_original()
{
    cuda_warmup();

    printf("\n=== BENCHMARK ORIGINAL (iterative_maxmin_cuadrado) ===\n");

    TensorResult tensor_desde_archivo;
    bool exito = leer_matriz_3d_desde_archivo("../datasets_txt/CC.txt", tensor_desde_archivo, 10, 16, 16, 1);

    if (!exito)
    {
        printf("\nError: No se pudo cargar el tensor desde archivo\n");
    }

    // === Probando función iterative_maxmin_cuadrado ===

    // Usar el tensor cargado desde archivo si está disponible, sino usar datos hardcodeados
    printf("\n=== Probando función iterative_maxmin_cuadrado ===\n");
    TensorResult test_tensor;
    bool usar_archivo = exito;
    test_tensor = tensor_desde_archivo;

    std::vector<TensorResult> result_tensor_paths;
    std::vector<TensorResult> result_values_paths;
    std::vector<TensorResult> pure_tensor_paths;
    std::vector<TensorResult> pure_values_paths;

    float test_threshold = 0.4f;
    int test_order = 4;

    int iterations = 1000;
    FunctionTimes function_times; // Estructura para almacenar todos los tiempos

    printf("Ejecutando %d iteraciones...\n", iterations);

    for (int i = 0; i < iterations; i++)
    {
        // Usar std::chrono para mayor precisión
        auto start_total = std::chrono::high_resolution_clock::now();

        // Variables para medir tiempos de funciones individuales
        double total_maxmin_time = 0.0;
        double total_indices_time = 0.0;
        double total_armar_caminos_time = 0.0;

        // Copiar tensor original (asignar ownership)
        TensorResult original_tensor = copy_tensor(test_tensor);
        TensorResult gen_tensor = copy_tensor(original_tensor);

        // Limpiar vectores de salida
        result_tensor_paths.clear();
        result_values_paths.clear();
        pure_tensor_paths.clear();
        pure_values_paths.clear();

        for (int j = 0; j < test_order - 1; j++)
        {
            // === MEDIR TIEMPO DE MAXMIN ===
            auto start_maxmin = std::chrono::high_resolution_clock::now();

            TensorResult min_result, maxmin_conjugado;
            maxmin(gen_tensor, original_tensor, maxmin_conjugado, min_result, false);

            auto end_maxmin = std::chrono::high_resolution_clock::now();
            auto duration_maxmin = std::chrono::duration_cast<std::chrono::microseconds>(end_maxmin - start_maxmin);
            total_maxmin_time += duration_maxmin.count() / 1000.0;

            // Los resultados de maxmin asignan memoria host sin marcar ownership; marcarlo.
            if (maxmin_conjugado.data)
                maxmin_conjugado.owns_memory = true;
            if (min_result.data)
                min_result.owns_memory = true;

            // Calcular prima = maxmin_conjugado - gen_tensor
            TensorResult prima;
            calculate_prima(maxmin_conjugado, gen_tensor, prima);

            // === MEDIR TIEMPO DE INDICES ===
            auto start_indices = std::chrono::high_resolution_clock::now();

            TensorResult result_tensor, result_values;
            indices(min_result, prima, result_tensor, result_values, test_threshold);

            auto end_indices = std::chrono::high_resolution_clock::now();
            auto duration_indices = std::chrono::duration_cast<std::chrono::microseconds>(end_indices - start_indices);
            total_indices_time += duration_indices.count() / 1000.0;

            pure_tensor_paths.push_back(result_tensor);
            pure_values_paths.push_back(result_values);

            // Verificar si se encontraron efectos
            if (pure_values_paths.back().data == nullptr || pure_values_paths.back().batch == 0)
            {
                if (j == 0)
                {
                    printf("Error: No se encontraron efectos con threshold %.4f\n", test_threshold);
                    // Limpiar memoria y continuar con siguiente iteración
                    safe_tensor_cleanup(original_tensor);
                    safe_tensor_cleanup(gen_tensor);
                    safe_tensor_cleanup(min_result);
                    safe_tensor_cleanup(maxmin_conjugado);
                    safe_tensor_cleanup(prima);
                    break;
                }
                else
                {
                    break;
                }
            }

            // Para órdenes superiores (j >= 1), construimos caminos usando armar_caminos
            if (j >= 1)
            {
                TensorResult previous_paths = (j > 1)
                                                  ? copy_tensor(result_tensor_paths.back())
                                                  : copy_tensor(pure_tensor_paths[0]);

                // === MEDIR TIEMPO DE ARMAR_CAMINOS ===
                auto start_armar = std::chrono::high_resolution_clock::now();

                TensorResult paths, values;
                armar_caminos(previous_paths, result_tensor, result_values, paths, values, j);

                auto end_armar = std::chrono::high_resolution_clock::now();
                auto duration_armar = std::chrono::duration_cast<std::chrono::microseconds>(end_armar - start_armar);
                total_armar_caminos_time += duration_armar.count() / 1000.0;

                if (paths.batch == 0)
                {
                    j = test_order; // Forzar salida del loop
                }

                result_tensor_paths.push_back(std::move(paths));
                result_values_paths.push_back(std::move(values));
            }

            safe_tensor_cleanup(gen_tensor);
            gen_tensor = std::move(maxmin_conjugado);
            safe_tensor_cleanup(min_result);
            safe_tensor_cleanup(prima);
            safe_tensor_cleanup(result_tensor);
            safe_tensor_cleanup(result_values);
        }

        // Limpiar memoria
        safe_tensor_cleanup(original_tensor);
        safe_tensor_cleanup(gen_tensor);

        if (!pure_tensor_paths.empty())
        {
            result_tensor_paths.insert(result_tensor_paths.begin(),
                                       copy_tensor(pure_tensor_paths.front()));
            result_values_paths.insert(result_values_paths.begin(),
                                       copy_tensor(pure_values_paths.front()));
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
        double time_total_ms = duration_total.count() / 1000.0;

        // Almacenar tiempos
        function_times.maxmin_accomulative += total_maxmin_time;
        function_times.indices_accomulative += total_indices_time;
        function_times.armar_caminos_accomulative += total_armar_caminos_time;
        function_times.total_accomulative += time_total_ms;

        printf("Iteración %d - Total: %.3f ms, Maxmin: %.3f ms, Indices: %.3f ms, Armar_caminos: %.3f ms\n",
               i + 1, time_total_ms, total_maxmin_time, total_indices_time, total_armar_caminos_time);

        // Limpiar vectores para siguiente iteración
        for (auto &tensor : result_tensor_paths)
        {
            safe_tensor_cleanup(tensor);
        }
        for (auto &tensor : result_values_paths)
        {
            safe_tensor_cleanup(tensor);
        }
        for (auto &tensor : pure_tensor_paths)
        {
            safe_tensor_cleanup(tensor);
        }
        for (auto &tensor : pure_values_paths)
        {
            safe_tensor_cleanup(tensor);
        }
    }

    function_times.maxmin_accomulative = function_times.maxmin_accomulative / iterations;
    function_times.indices_accomulative = function_times.indices_accomulative / iterations;
    function_times.armar_caminos_accomulative = function_times.armar_caminos_accomulative / iterations;
    function_times.total_accomulative = function_times.total_accomulative / iterations;

    // Imprimir promedios
    printf("Promedio Total: %f ms\n", function_times.total_accomulative);
    printf("Promedio Maxmin: %f ms\n", function_times.maxmin_accomulative);
    printf("Promedio Indices: %f ms\n", function_times.indices_accomulative);
    printf("Promedio Armar_caminos: %f ms\n", function_times.armar_caminos_accomulative);

    // Liberar memoria del tensor de prueba
    if (usar_archivo)
    {
        if (tensor_desde_archivo.data)
            free(tensor_desde_archivo.data);
    }
    else
    {
        if (test_tensor.data)
            free(test_tensor.data);
    }

    printf("\nPruebas completadas.\n");
}
