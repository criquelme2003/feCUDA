#include <fmt/format.h>
#include <iostream>
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
#include <bootstrap.cuh>
#include <chrono>
#include <temp.cuh>
#include <utils.cuh>

void generate_results()
{

    std::cout << "Generando resultados para validacion...." << std::endl;
    TensorResult cc, ee, ce;
    std::vector<int> reps = {10, 100, 1000, 10000}; // replicas bootstrap
    int orden = 5;                                  // orden fijo

    auto convert_to_cpu = [](const std::vector<TensorResult> &tensors)
    {
        std::vector<TensorResult> cpu_tensors;
        cpu_tensors.reserve(tensors.size());
        for (const auto &tensor : tensors)
        {
            if (tensor.data == nullptr)
            {
                cpu_tensors.emplace_back();
                continue;
            }
            cpu_tensors.push_back(copy_tensor_to_cpu(tensor)); // Garantiza puntero en host
        }
        return cpu_tensors;
    };

    leer_matriz_3d_desde_archivo("./datasets_txt/CC.txt", cc, 10, 16, 16, 1);
    leer_matriz_3d_desde_archivo("./datasets_txt/CE.txt", ce, 10, 16, 4, 1);
    leer_matriz_3d_desde_archivo("./datasets_txt/EE.txt", ee, 10, 4, 4, 1);

    // Generar resultados con tensores
    std::vector<TensorResult> tens = {cc, ee, ce}; // replicas bootstrap
    std::vector<std::string> names = {"cc", "ee", "ce"};
    std::vector<float> thrs = {0.1, 0.3, 0.5, 0.7};


    for (int i = 0; i < tens.size(); i++)
    {
        for (auto thr : thrs)
        {
            cudaDeviceReset();
            std::string name_values = fmt::format("./validation/results/values_{}_{}.txt", names[i], thr);

            std::cout << "generando resultados para:" << name_values << std::endl;
            std::vector<TensorResult> paths;
            std::vector<TensorResult> values;
            std::vector<TensorResult> pure_paths;
            std::vector<TensorResult> pure_values;

            iterative_maxmin_cuadrado(tens[i], thr, orden, paths, values, pure_paths, pure_values, true);
            std::string name_paths = fmt::format("./validation/results/paths_{}_{}.txt", names[i], thr);

            auto cpu_paths = convert_to_cpu(paths);
            auto cpu_values = convert_to_cpu(values);

            save_tensor_vector(cpu_paths, name_paths);
            save_tensor_vector(cpu_values, name_values);
        }
    }

    // Generar resultados con bootstrap
    int M = cc.M;
    int N = cc.N;
    int batch = cc.batch;
    float *data = cc.data;
    float thr = 0.2; // threshold fijo
    for (int i = 0; i < reps.size(); i++)
    {
        cudaDeviceReset();
        int replicas = reps[i];
        float *bootstrap_res, *d_bootstrap;

        bootstrap_res = (float *)malloc(M * N * replicas * sizeof(float));
        d_bootstrap = bootstrap_wrapper(data, M, N, batch, replicas);

        cudaMemcpy(bootstrap_res, d_bootstrap, M * N * replicas * sizeof(float), cudaMemcpyDeviceToHost);

        TensorResult t2 = TensorResult(bootstrap_res, false, replicas, M, N);
        std::vector<TensorResult> paths;
        std::vector<TensorResult> values;
        std::vector<TensorResult> pure_paths;
        std::vector<TensorResult> pure_values;

        iterative_maxmin_cuadrado(t2, thr, orden, paths, values, pure_paths, pure_values, true);
        std::string name_paths = fmt::format("./validation/results/paths_bootstrap_{}.txt", replicas);
        std::string name_values = fmt::format("./validation/results/values_bootstrap_{}.txt", replicas);

        auto cpu_paths = convert_to_cpu(paths);
        auto cpu_values = convert_to_cpu(values);

        save_tensor_vector(cpu_paths, name_paths, true);
        save_tensor_vector(cpu_values, name_values, true);

        cudaFree(d_bootstrap);
    }
    std::cout << "Resultados generados correctamente" << std::endl;
}
