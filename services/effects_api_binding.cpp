#include <cmath>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <chrono>

#include "headers.cuh"
#include "utils.cuh"
#include "core/types.cuh"
#include "algorithms/bootstrap.cuh"

namespace
{
std::string g_last_error;
std::mutex g_error_mutex;

void set_last_error(const std::string &message)
{
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = message;
}

TensorResult ensure_cpu_tensor(const TensorResult &tensor)
{
    if (tensor.data == nullptr || tensor.M == 0)
    {
        return TensorResult();
    }

    if (tensor.is_device_ptr)
    {
        return copy_tensor_to_cpu(tensor);
    }
    return copy_tensor(tensor);
}

void cleanup_tensor_vector(std::vector<TensorResult> &tensors)
{
    for (auto &tensor : tensors)
    {
        safe_tensor_cleanup(tensor);
    }
}

} // namespace

extern "C"
{

const char *generate_effects_json(const float *data,
                                  int batch,
                                  int M,
                                  int N,
                                  float threshold,
                                  int order,
                                  int bootstrap_replicas)
{
    if (data == nullptr)
    {
        set_last_error("El puntero al tensor de entrada es nulo.");
        return nullptr;
    }

    if (batch <= 0 || M <= 0 || N <= 0)
    {
        set_last_error("Dimensiones inválidas para el tensor de entrada.");
        return nullptr;
    }

    if (threshold < 0.0f || threshold > 1.0f)
    {
        set_last_error("El threshold debe estar en el rango [0, 1].");
        return nullptr;
    }

    if (order <= 1)
    {
        set_last_error("El orden debe ser mayor que 1.");
        return nullptr;
    }

    if (bootstrap_replicas < 0)
    {
        set_last_error("El número de replicas bootstrap debe ser mayor o igual a 0.");
        return nullptr;
    }

    const size_t total_elements = static_cast<size_t>(batch) * M * N;
    const size_t total_bytes = total_elements * sizeof(float);

    float *host_copy = static_cast<float *>(malloc(total_bytes));
    if (!host_copy)
    {
        set_last_error("No se pudo asignar memoria para el tensor de entrada.");
        return nullptr;
    }
    std::memcpy(host_copy, data, total_bytes);

    TensorResult original_tensor(host_copy, false, batch, M, N, 1, true);
    TensorResult bootstrap_tensor;
    TensorResult *algorithm_input = &original_tensor;

    size_t free_mem_before = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem_before, &total_mem);

    auto start_total = std::chrono::steady_clock::now();
    double bootstrap_ms = 0.0;
    double algorithm_ms = 0.0;

    if (bootstrap_replicas > 0)
    {
        auto start_bootstrap = std::chrono::steady_clock::now();
        float *d_bootstrap = bootstrap_wrapper(host_copy, M, N, batch, bootstrap_replicas);
        if (!d_bootstrap)
        {
            safe_tensor_cleanup(original_tensor);
            set_last_error("bootstrap_wrapper falló al generar las replicas.");
            return nullptr;
        }

        const size_t bootstrap_elements = static_cast<size_t>(bootstrap_replicas) * M * N;
        const size_t bootstrap_bytes = bootstrap_elements * sizeof(float);

        float *bootstrap_host = static_cast<float *>(malloc(bootstrap_bytes));
        if (!bootstrap_host)
        {
            cudaFree(d_bootstrap);
            safe_tensor_cleanup(original_tensor);
            set_last_error("No se pudo asignar memoria host para las replicas bootstrap.");
            return nullptr;
        }

        cudaError_t copy_status = cudaMemcpy(
            bootstrap_host,
            d_bootstrap,
            bootstrap_bytes,
            cudaMemcpyDeviceToHost);
        cudaFree(d_bootstrap);

        if (copy_status != cudaSuccess)
        {
            free(bootstrap_host);
            safe_tensor_cleanup(original_tensor);
            set_last_error(std::string("Fallo al copiar replicas bootstrap desde GPU: ") +
                           cudaGetErrorString(copy_status));
            return nullptr;
        }

        bootstrap_tensor = TensorResult(bootstrap_host, false, bootstrap_replicas, M, N, 1, true);
        algorithm_input = &bootstrap_tensor;
        auto end_bootstrap = std::chrono::steady_clock::now();
        bootstrap_ms = std::chrono::duration<double, std::milli>(end_bootstrap - start_bootstrap).count();
    }

    std::vector<TensorResult> result_paths;
    std::vector<TensorResult> result_values;
    std::vector<TensorResult> pure_paths;
    std::vector<TensorResult> pure_values;

    try
    {
        auto start_algorithm = std::chrono::steady_clock::now();
        iterative_maxmin_cuadrado(*algorithm_input,
                                  threshold,
                                  order,
                                  result_paths,
                                  result_values,
                                  pure_paths,
                                  pure_values,
                                  false);
        auto end_algorithm = std::chrono::steady_clock::now();
        algorithm_ms = std::chrono::duration<double, std::milli>(end_algorithm - start_algorithm).count();

        std::vector<TensorResult> cpu_paths;
        std::vector<TensorResult> cpu_values;

        cpu_paths.reserve(result_paths.size());
        cpu_values.reserve(result_values.size());

        for (auto &tensor : result_paths)
        {
            cpu_paths.push_back(ensure_cpu_tensor(tensor));
        }

        for (auto &tensor : result_values)
        {
            cpu_values.push_back(ensure_cpu_tensor(tensor));
        }

        std::ostringstream effects_stream;
        effects_stream.setf(std::ios::fixed);
        bool first_entry = true;
        effects_stream << "[\n";
        size_t total_effects = 0;

        for (size_t order_idx = 0; order_idx < cpu_paths.size(); ++order_idx)
        {
            const TensorResult &path_tensor = cpu_paths[order_idx];
            const TensorResult &value_tensor = cpu_values[order_idx];

            if (path_tensor.data == nullptr || value_tensor.data == nullptr)
            {
                continue;
            }

            const int num_paths = path_tensor.M;
            const int path_cols = path_tensor.N;
            const int value_count = (value_tensor.M == 1)
                                        ? value_tensor.N
                                        : value_tensor.M * value_tensor.N;
            const int paired = std::min(num_paths, value_count);

            for (int row = 0; row < paired; ++row)
            {
                if (!first_entry)
                {
                    effects_stream << ",\n";
                }
                first_entry = false;
                ++total_effects;

                effects_stream << "  {\"order\":" << order_idx << ",\"row\":" << row << ",\"path\":[";
                for (int col = 0; col < path_cols; ++col)
                {
                    const int idx = row * path_cols + col;
                    const int coord = static_cast<int>(std::llround(path_tensor.data[idx]));
                    effects_stream << coord;
                    if (col + 1 < path_cols)
                    {
                        effects_stream << ",";
                    }
                }
                effects_stream << "],\"value\":";

                float value = 0.0f;
                if (value_tensor.M == 1)
                {
                    value = value_tensor.data[row];
                }
                else if (value_tensor.N == 1)
                {
                    value = value_tensor.data[row];
                }
                else
                {
                    value = value_tensor.data[row * value_tensor.N];
                }

                effects_stream << std::setprecision(8) << value << "}";
            }
        }

        if (!first_entry)
        {
            effects_stream << "\n";
        }
        effects_stream << "]";

        auto end_total = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();

        size_t free_mem_after = 0;
        size_t total_mem_after = 0;
        cudaMemGetInfo(&free_mem_after, &total_mem_after);

        double gpu_free_before_mb = static_cast<double>(free_mem_before) / (1024.0 * 1024.0);
        double gpu_free_after_mb = static_cast<double>(free_mem_after) / (1024.0 * 1024.0);
        double gpu_delta_mb = gpu_free_before_mb - gpu_free_after_mb;

        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << "{\n"
            << "  \"effects\": " << effects_stream.str() << ",\n"
            << "  \"total_entries\": " << total_effects << ",\n"
            << "  \"metrics\": {\n"
            << "    \"total_processing_ms\": " << std::setprecision(4) << total_ms << ",\n"
            << "    \"algorithm_ms\": " << std::setprecision(4) << algorithm_ms << ",\n"
            << "    \"bootstrap_ms\": " << std::setprecision(4) << bootstrap_ms << ",\n"
            << "    \"bootstrap_replicas\": " << bootstrap_replicas << ",\n"
            << "    \"gpu_memory_free_before_mb\": " << std::setprecision(2) << gpu_free_before_mb << ",\n"
            << "    \"gpu_memory_free_after_mb\": " << std::setprecision(2) << gpu_free_after_mb << ",\n"
            << "    \"gpu_memory_delta_mb\": " << std::setprecision(2) << gpu_delta_mb << "\n"
            << "  }\n"
            << "}";

        const std::string json = oss.str();
        char *output = static_cast<char *>(malloc(json.size() + 1));
        if (!output)
        {
            set_last_error("No se pudo asignar memoria para la respuesta JSON.");
            for (auto &tensor : cpu_paths)
            {
                safe_tensor_cleanup(tensor);
            }
            for (auto &tensor : cpu_values)
            {
                safe_tensor_cleanup(tensor);
            }
            cleanup_tensor_vector(result_paths);
            cleanup_tensor_vector(result_values);
            cleanup_tensor_vector(pure_paths);
            cleanup_tensor_vector(pure_values);
            safe_tensor_cleanup(original_tensor);
            safe_tensor_cleanup(bootstrap_tensor);
            return nullptr;
        }

        std::memcpy(output, json.c_str(), json.size() + 1);

        for (auto &tensor : cpu_paths)
        {
            safe_tensor_cleanup(tensor);
        }
        for (auto &tensor : cpu_values)
        {
            safe_tensor_cleanup(tensor);
        }

        cleanup_tensor_vector(result_paths);
        cleanup_tensor_vector(result_values);
        cleanup_tensor_vector(pure_paths);
        cleanup_tensor_vector(pure_values);
        safe_tensor_cleanup(original_tensor);
        safe_tensor_cleanup(bootstrap_tensor);

        set_last_error("");
        return output;
    }
    catch (const std::exception &ex)
    {
        set_last_error(ex.what());
    }
    catch (...)
    {
        set_last_error("Error desconocido durante la generación de efectos.");
    }

    cleanup_tensor_vector(result_paths);
    cleanup_tensor_vector(result_values);
    cleanup_tensor_vector(pure_paths);
    cleanup_tensor_vector(pure_values);
    safe_tensor_cleanup(original_tensor);
    safe_tensor_cleanup(bootstrap_tensor);

    return nullptr;
}

void free_effects_json(const char *ptr)
{
    if (ptr)
    {
        free(const_cast<char *>(ptr));
    }
}

const char *effects_last_error()
{
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return g_last_error.c_str();
}

} // extern "C"
