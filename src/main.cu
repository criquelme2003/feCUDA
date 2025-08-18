#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <tuple>
#include "headers.cuh"
#include "utils.cuh"
#include "types.cuh"
#include <chrono>

// Estructura para almacenar tiempos de cada función
struct FunctionTimes
{
    std::vector<double> maxmin_times;
    std::vector<double> indices_times;
    std::vector<double> armar_caminos_times;
    std::vector<double> total_times;
};

int main()
{

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

    int iterations = 100;
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
        function_times.maxmin_times.push_back(total_maxmin_time);
        function_times.indices_times.push_back(total_indices_time);
        function_times.armar_caminos_times.push_back(total_armar_caminos_time);
        function_times.total_times.push_back(time_total_ms);

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

    // Guardar los tiempos en archivo CSV con detalles por función
    std::ofstream csv_file("execution_times.csv");
    if (csv_file.is_open())
    {
        // Escribir header detallado
        csv_file << "iteracion,tiempo_total_ms,tiempo_maxmin_ms,tiempo_indices_ms,tiempo_armar_caminos_ms\n";

        // Escribir datos
        for (size_t i = 0; i < function_times.total_times.size(); i++)
        {
            csv_file << (i + 1) << ","
                     << function_times.total_times[i] << ","
                     << function_times.maxmin_times[i] << ","
                     << function_times.indices_times[i] << ","
                     << function_times.armar_caminos_times[i] << "\n";
        }

        csv_file.close();
        printf("\nTiempos detallados guardados en 'execution_times.csv'\n");

        // Calcular estadísticas para tiempo total
        double total_total = 0.0, min_total = function_times.total_times.empty() ? 0.0 : function_times.total_times[0];
        double max_total = function_times.total_times.empty() ? 0.0 : function_times.total_times[0];
        for (double time : function_times.total_times)
        {
            total_total += time;
            if (time < min_total)
                min_total = time;
            if (time > max_total)
                max_total = time;
        }
        double avg_total = function_times.total_times.empty() ? 0.0 : total_total / function_times.total_times.size();

        // Calcular estadísticas para maxmin
        double total_maxmin = 0.0, min_maxmin = function_times.maxmin_times.empty() ? 0.0 : function_times.maxmin_times[0];
        double max_maxmin = function_times.maxmin_times.empty() ? 0.0 : function_times.maxmin_times[0];
        for (double time : function_times.maxmin_times)
        {
            total_maxmin += time;
            if (time < min_maxmin)
                min_maxmin = time;
            if (time > max_maxmin)
                max_maxmin = time;
        }
        double avg_maxmin = function_times.maxmin_times.empty() ? 0.0 : total_maxmin / function_times.maxmin_times.size();

        // Calcular estadísticas para indices
        double total_indices = 0.0, min_indices = function_times.indices_times.empty() ? 0.0 : function_times.indices_times[0];
        double max_indices = function_times.indices_times.empty() ? 0.0 : function_times.indices_times[0];
        for (double time : function_times.indices_times)
        {
            total_indices += time;
            if (time < min_indices)
                min_indices = time;
            if (time > max_indices)
                max_indices = time;
        }
        double avg_indices = function_times.indices_times.empty() ? 0.0 : total_indices / function_times.indices_times.size();

        // Calcular estadísticas para armar_caminos
        double total_armar = 0.0, min_armar = function_times.armar_caminos_times.empty() ? 0.0 : function_times.armar_caminos_times[0];
        double max_armar = function_times.armar_caminos_times.empty() ? 0.0 : function_times.armar_caminos_times[0];
        for (double time : function_times.armar_caminos_times)
        {
            total_armar += time;
            if (time < min_armar)
                min_armar = time;
            if (time > max_armar)
                max_armar = time;
        }
        double avg_armar = function_times.armar_caminos_times.empty() ? 0.0 : total_armar / function_times.armar_caminos_times.size();

        printf("\n=== Estadísticas de ejecución detalladas ===\n");
        printf("Iteraciones: %d\n", iterations);
        printf("\n--- TIEMPO TOTAL ---\n");
        printf("Promedio: %.3f ms\n", avg_total);
        printf("Mínimo: %.3f ms\n", min_total);
        printf("Máximo: %.3f ms\n", max_total);
        printf("Total acumulado: %.3f ms\n", total_total);

        printf("\n--- FUNCIÓN MAXMIN ---\n");
        printf("Promedio: %.3f ms (%.1f%% del total)\n", avg_maxmin, avg_total > 0 ? (avg_maxmin / avg_total) * 100 : 0.0);
        printf("Mínimo: %.3f ms\n", min_maxmin);
        printf("Máximo: %.3f ms\n", max_maxmin);
        printf("Total acumulado: %.3f ms\n", total_maxmin);

        printf("\n--- FUNCIÓN INDICES ---\n");
        printf("Promedio: %.3f ms (%.1f%% del total)\n", avg_indices, avg_total > 0 ? (avg_indices / avg_total) * 100 : 0.0);
        printf("Mínimo: %.3f ms\n", min_indices);
        printf("Máximo: %.3f ms\n", max_indices);
        printf("Total acumulado: %.3f ms\n", total_indices);

        printf("\n--- FUNCIÓN ARMAR_CAMINOS ---\n");
        printf("Promedio: %.3f ms (%.1f%% del total)\n", avg_armar, avg_total > 0 ? (avg_armar / avg_total) * 100 : 0.0);
        printf("Mínimo: %.3f ms\n", min_armar);
        printf("Máximo: %.3f ms\n", max_armar);
        printf("Total acumulado: %.3f ms\n", total_armar);

        printf("\n--- RESUMEN DE DISTRIBUCIÓN ---\n");
        printf("Maxmin: %.1f%% del tiempo total\n", avg_total > 0 ? (avg_maxmin / avg_total) * 100 : 0.0);
        printf("Indices: %.1f%% del tiempo total\n", avg_total > 0 ? (avg_indices / avg_total) * 100 : 0.0);
        printf("Armar_caminos: %.1f%% del tiempo total\n", avg_total > 0 ? (avg_armar / avg_total) * 100 : 0.0);
        double other_percent = avg_total > 0 ? 100.0 - ((avg_maxmin + avg_indices + avg_armar) / avg_total) * 100 : 0.0;
        printf("Otras operaciones: %.1f%% del tiempo total\n", other_percent);
    }
    else
    {
        printf("\nError: No se pudo crear el archivo CSV\n");
    }

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
    return 0;
}
