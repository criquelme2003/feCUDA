#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include "headers.cuh"
#include "utils.cuh"

#ifndef FECUDA_SOURCE_DIR
#define FECUDA_SOURCE_DIR "."
#endif

namespace
{
    std::string dataset_path(const std::string &filename)
    {
        return std::string(FECUDA_SOURCE_DIR) + "/datasets_txt/" + filename;
    }

    std::string results_path(const std::string &filename)
    {
        return std::string(FECUDA_SOURCE_DIR) + "/results/" + filename;
    }

    std::vector<TensorResult> copy_to_cpu_vector(const std::vector<TensorResult> &src)
    {
        std::vector<TensorResult> out;
        out.reserve(src.size());
        for (const auto &tensor : src)
        {
            if (tensor.data == nullptr)
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
    std::cout << "FeCUDA Sprint 1 - modo batch local\n";

    TensorResult tensor_cc;
    if (!leer_matriz_3d_desde_archivo(dataset_path("CC.txt").c_str(), tensor_cc, 10, 16, 16, 1))
    {
        std::cerr << "No se pudo cargar datasets_txt/CC.txt\n";
        return 1;
    }

    const float threshold = 0.3f;
    const int order = 4;

    std::vector<TensorResult> paths;
    std::vector<TensorResult> values;
    std::vector<TensorResult> pure_paths;
    std::vector<TensorResult> pure_values;

    iterative_maxmin_cuadrado(tensor_cc, threshold, order,
                              paths, values, pure_paths, pure_values,
                              false);

    if (paths.empty())
    {
        std::cout << "No se encontraron caminos para threshold=" << threshold << "\n";
    }
    else
    {
        std::filesystem::create_directories(results_path(""));
        const std::string output_file = results_path("paths_values_cc_local.json");

        auto cpu_paths = copy_to_cpu_vector(paths);
        auto cpu_values = copy_to_cpu_vector(values);
        save_paths_with_values(cpu_paths, cpu_values, output_file);

        std::cout << "Se encontraron " << paths.size()
                  << " colecciones de caminos. Resultados guardados en "
                  << output_file << "\n";

        cleanup_vector(cpu_paths);
        cleanup_vector(cpu_values);
    }

    cleanup_vector(paths);
    cleanup_vector(values);
    cleanup_vector(pure_paths);
    cleanup_vector(pure_values);
    safe_tensor_cleanup(tensor_cc);
    return 0;
}
