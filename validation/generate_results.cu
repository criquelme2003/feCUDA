#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "headers.cuh"
#include "utils.cuh"

#ifndef FECUDA_SOURCE_DIR
#define FECUDA_SOURCE_DIR "."
#endif

namespace
{
    struct DatasetConfig
    {
        std::string alias;
        std::string filename;
        int batch;
        int M;
        int N;
    };

    std::string dataset_path(const std::string &filename)
    {
        return std::string(FECUDA_SOURCE_DIR) + "/datasets_txt/" + filename;
    }

    std::string results_dir()
    {
        return std::string(FECUDA_SOURCE_DIR) + "/validation/results";
    }

    std::string build_results_name(const std::string &alias, float thr)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << thr;
        return "paths_values_" + alias + "_" + oss.str() + ".json";
    }

    std::vector<TensorResult> copy_to_cpu_vector(const std::vector<TensorResult> &src)
    {
        std::vector<TensorResult> out;
        out.reserve(src.size());
        for (const auto &tensor : src)
        {
            if (!tensor.data)
            {
                out.emplace_back();
                continue;
            }
            if (tensor.is_device_ptr)
            {
                out.push_back(copy_tensor_to_cpu(tensor));
            }
            else
            {
                out.push_back(copy_tensor(tensor));
            }
        }
        return out;
    }

    void cleanup_vector(std::vector<TensorResult> &tensors)
    {
        for (auto &tensor : tensors)
        {
            safe_tensor_cleanup(tensor);
        }
        tensors.clear();
    }
}

int main()
{
    std::cout << "Generando artefactos de validacion para Sprint 1...\n";
    std::filesystem::create_directories(results_dir());

    const std::vector<DatasetConfig> datasets = {
        {"cc", "CC.txt", 10, 16, 16},
        {"ee", "EE.txt", 10, 4, 4}};
    const std::vector<float> thresholds = {0.1f, 0.3f, 0.5f, 0.7f};
    const int order = 5;

    for (const auto &config : datasets)
    {
        TensorResult tensor;
        if (!leer_matriz_3d_desde_archivo(dataset_path(config.filename).c_str(),
                                          tensor, config.batch, config.M, config.N, 1))
        {
            std::cerr << "No se pudo cargar " << config.filename << ", se omite.\n";
            continue;
        }

        for (float thr : thresholds)
        {
            std::vector<TensorResult> paths;
            std::vector<TensorResult> values;
            std::vector<TensorResult> pure_paths;
            std::vector<TensorResult> pure_values;

            iterative_maxmin_cuadrado(tensor, thr, order, paths, values, pure_paths, pure_values, false);

            if (paths.empty())
            {
                std::cout << "[WARN] No hubo caminos para " << config.alias << " thr=" << thr << "\n";
            }
            else
            {
                auto cpu_paths = copy_to_cpu_vector(paths);
                auto cpu_values = copy_to_cpu_vector(values);

                const std::string output_file = results_dir() + "/" + build_results_name(config.alias, thr);
                save_paths_with_values(cpu_paths, cpu_values, output_file);
                std::cout << "[OK] " << output_file << " (" << cpu_paths.size() << " conjuntos)\n";

                cleanup_vector(cpu_paths);
                cleanup_vector(cpu_values);
            }

            cleanup_vector(paths);
            cleanup_vector(values);
            cleanup_vector(pure_paths);
            cleanup_vector(pure_values);
        }

        safe_tensor_cleanup(tensor);
    }

    std::cout << "Validaciones listas.\n";
    return 0;
}
