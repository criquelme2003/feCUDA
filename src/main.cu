#include <iostream>
#include <exception>
#include "algorithms/paths.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/logging.cuh"
#include "utils/file_io.cuh"
#include "core/tensor.cuh"
#include "kernels/kernels.cuh"
#include "test/test.cuh"
#include "../../include/utils.cuh"
#include "temp.cuh"
#include <headers.cuh>
#include <random>
#include <time.h>
#include <bootstrap.cuh>
#include <chrono>
#include <temp.cuh>

int main()
{
    TensorResult t1;
    leer_matriz_3d_desde_archivo("./datasets_txt/CC.txt", t1, 10, 16, 16, 1);
    int M = t1.M, N = t1.N, replicas = 1000;
    int iter = 100;

    auto start_total = std::chrono::high_resolution_clock::now();
    std::vector<double> tiempos_iteracion;

    cudaDeviceReset();
    for (int i = 0; i < iter; i++)
    {
       auto start_iter = std::chrono::high_resolution_clock::now();

        float *bootstrap_res, *d_bootstrap;
        bootstrap_res = (float *)malloc(M * N * replicas * sizeof(float));
        d_bootstrap = bootstrap_wrapper(t1.data, t1.M, t1.N, t1.batch, replicas);
        cudaMemcpy(bootstrap_res, d_bootstrap, M * N * replicas * sizeof(float), cudaMemcpyDeviceToHost);
        TensorResult t2 = TensorResult(bootstrap_res, false, replicas, M, N);
        std::vector<TensorResult> paths;
        std::vector<TensorResult> values;
        std::vector<TensorResult> pure_paths;
        std::vector<TensorResult> pure_values;
        iterative_maxmin_cuadrado(t2, 0.2, 3, paths, values, pure_paths, pure_values, true);

        // Fin de medición para esta iteración
        auto end_iter = std::chrono::high_resolution_clock::now();
        auto duration_iter = std::chrono::duration_cast<std::chrono::microseconds>(end_iter - start_iter);
        tiempos_iteracion.push_back(duration_iter.count() / 1000.0); // Convertir a milisegundos

        // Liberar memoria
        cudaFree(d_bootstrap);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);

    // Calcular estadísticas
    double tiempo_total_ms = duration_total.count();
    double suma_iteraciones = 0.0;
    for (double tiempo : tiempos_iteracion)
    {
        suma_iteraciones += tiempo;
    }
    double promedio_por_iteracion = suma_iteraciones / iter;

    // Mostrar resultados
    std::cout << "=== REPORTE DE TIEMPOS DE EJECUCIÓN ===" << std::endl;
    std::cout << "Número de iteraciones: " << iter << std::endl;
    std::cout << "Tiempo total: " << tiempo_total_ms << " ms" << std::endl;
    std::cout << "Tiempo promedio por iteración: " << promedio_por_iteracion << " ms" << std::endl;
    std::cout << "Tiempo mínimo por iteración: " << *std::min_element(tiempos_iteracion.begin(), tiempos_iteracion.end()) << " ms" << std::endl;
    std::cout << "Tiempo máximo por iteración: " << *std::max_element(tiempos_iteracion.begin(), tiempos_iteracion.end()) << " ms" << std::endl;

    return 0;
}
