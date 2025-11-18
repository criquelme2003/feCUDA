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
#include <headers.cuh>
#include <random>
#include <bootstrap.cuh>
#include <chrono>
#include <filesystem>

namespace
{
    std::filesystem::path repo_relative(const std::filesystem::path &relative)
    {
        std::filesystem::path base(FECUDA_SOURCE_DIR);
        if (!base.is_absolute())
        {
            base = std::filesystem::current_path() / base;
        }

        std::filesystem::path combined = base / relative;
        try
        {
            return std::filesystem::weakly_canonical(combined);
        }
        catch (...)
        {
            return combined.lexically_normal();
        }
    }
}

void generate_results()
{

    std::cout << "Generando resultados para validacion...." << std::endl;
    TensorResult cc, ee;
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

    const auto datasets_dir = repo_relative("datasets_txt");
    leer_matriz_3d_desde_archivo((datasets_dir / "CC.txt").string().c_str(), cc, 10, 16, 16, 1);
    // leer_matriz_3d_desde_archivo("./datasets_txt/CE.txt", ce, 10, 16, 4, 1);
    leer_matriz_3d_desde_archivo((datasets_dir / "EE.txt").string().c_str(), ee, 10, 4, 4, 1);
    imprimir_tensor(ee);

    // Generar resultados con tensores
    std::vector<TensorResult> tens = {cc, ee}; // replicas bootstrap
    std::vector<std::string> names = {"cc", "ee"};
    std::vector<float> thrs = {0.1, 0.3, 0.5, 0.7};

    for (int i = 0; i < tens.size(); i++)
    {
        for (auto thr : thrs)
        {
            cudaDeviceReset();
            const auto validation_results_dir = repo_relative("validation/results");
            const auto name_values_path = validation_results_dir / fmt::format("values_{}_{}.txt", names[i], thr);
            const auto name_paths_path = validation_results_dir / fmt::format("paths_{}_{}.txt", names[i], thr);

            std::cout << "generando resultados para:" << name_values_path << std::endl;
            std::vector<TensorResult> paths;
            std::vector<TensorResult> values;
            std::vector<TensorResult> pure_paths;
            std::vector<TensorResult> pure_values;

            iterative_maxmin_cuadrado(tens[i], thr, orden, paths, values, pure_paths, pure_values, true);

            auto cpu_paths = convert_to_cpu(paths);
            auto cpu_values = convert_to_cpu(values);

            std::string name = "ee";
            if (std::fabs(thr - 0.7) < 0.00001f && names[i] == name)
            {
                for (auto t : values)
                {
                    imprimir_tensor(t);
                }
            }

            const auto name_structured = validation_results_dir / fmt::format("paths_values_{}_{}.json", names[i], thr);
            save_paths_with_values(cpu_paths, cpu_values, name_structured.string());
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
        const auto validation_results_dir = repo_relative("validation/results");
        const auto name_paths = validation_results_dir / fmt::format("paths_bootstrap_{}.txt", replicas);
        const auto name_values = validation_results_dir / fmt::format("values_bootstrap_{}.txt", replicas);

        auto cpu_paths = convert_to_cpu(paths);
        auto cpu_values = convert_to_cpu(values);

        const auto name_structured = validation_results_dir / fmt::format("paths_values_bootstrap_{}.json", replicas);
        save_paths_with_values(cpu_paths, cpu_values, name_structured.string());

        cudaFree(d_bootstrap);
    }
    std::cout << "Resultados generados correctamente" << std::endl;
}
