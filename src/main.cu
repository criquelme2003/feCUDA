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
#include <headers.cuh>
#include <random>
#include <time.h>
#include <bootstrap.cuh>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <map>
#include <limits>
#include <cmath>
#include <string>

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

    struct BenchConfig
    {
        bool enabled = false;
        bool synthetic = false;
        bool skip_bootstrap = true;
        std::string dataset_path;
        std::string label = "case";
        std::string pattern;
        int batch = 0;
        int M = 0;
        int N = 0;
        float threshold = 0.2f;
        int order = 3;
        int replicas = 100;
        int iterations = 1;
        unsigned int seed = 0;
        bool seed_provided = false;
    };

    BenchConfig parse_bench_args(int argc, char **argv)
    {
        BenchConfig config;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg(argv[i]);
            auto require_value = [&](const char *flag) -> const char *
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error(std::string("Falta valor para ") + flag);
                }
                return argv[++i];
            };

            if (arg == "--bench")
            {
                config.enabled = true;
                continue;
            }
            if (arg == "--dataset")
            {
                config.dataset_path = require_value("--dataset");
                continue;
            }
            if (arg == "--synthetic")
            {
                config.synthetic = true;
                config.dataset_path = "synthetic";
                continue;
            }
            if (arg == "--pattern")
            {
                config.pattern = require_value("--pattern");
                continue;
            }
            if (arg == "--skip-bootstrap")
            {
                config.skip_bootstrap = true;
                continue;
            }
            if (arg == "--label")
            {
                config.label = require_value("--label");
                continue;
            }
            if (arg == "--batch")
            {
                config.batch = std::stoi(require_value("--batch"));
                continue;
            }
            if (arg == "--M")
            {
                config.M = std::stoi(require_value("--M"));
                continue;
            }
            if (arg == "--N")
            {
                config.N = std::stoi(require_value("--N"));
                continue;
            }
            if (arg == "--threshold")
            {
                config.threshold = std::stof(require_value("--threshold"));
                continue;
            }
            if (arg == "--order")
            {
                config.order = std::stoi(require_value("--order"));
                continue;
            }
            if (arg == "--replicas")
            {
                config.replicas = std::stoi(require_value("--replicas"));
                continue;
            }
            if (arg == "--iterations")
            {
                config.iterations = std::stoi(require_value("--iterations"));
                continue;
            }
            if (arg == "--seed")
            {
                config.seed = static_cast<unsigned int>(std::stoul(require_value("--seed")));
                config.seed_provided = true;
                continue;
            }
        }
        return config;
    }

    struct BenchMetrics
    {
        std::vector<double> durations_ms;

        double mean() const
        {
            if (durations_ms.empty())
            {
                return 0.0;
            }
            double sum = 0.0;
            for (double v : durations_ms)
            {
                sum += v;
            }
            return sum / static_cast<double>(durations_ms.size());
        }

        double min() const
        {
            if (durations_ms.empty())
            {
                return 0.0;
            }
            return *std::min_element(durations_ms.begin(), durations_ms.end());
        }

        double max() const
        {
            if (durations_ms.empty())
            {
                return 0.0;
            }
            return *std::max_element(durations_ms.begin(), durations_ms.end());
        }

        double stddev() const
        {
            if (durations_ms.size() < 2)
            {
                return 0.0;
            }
            const double avg = mean();
            double accum = 0.0;
            for (double v : durations_ms)
            {
                const double diff = v - avg;
                accum += diff * diff;
            }
            return std::sqrt(accum / static_cast<double>(durations_ms.size()));
        }
    };
    bool ensure_cuda_device_ready()
    {
        int device_count = 0;
        cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count <= 0)
        {
            const char *msg = (device_status == cudaSuccess) ? "No CUDA devices detected" : cudaGetErrorString(device_status);
            std::cerr << "BENCH_ERROR reason=no_cuda_device message=\"" << msg << "\"\n";
            return false;
        }
        cudaError_t set_status = cudaSetDevice(0);
        if (set_status != cudaSuccess)
        {
            std::cerr << "BENCH_ERROR reason=set_device_failed message=\"" << cudaGetErrorString(set_status) << "\"\n";
            return false;
        }
        return true;
    }

    TensorResult create_random_tensor(const BenchConfig &config, int iteration)
    {
        const size_t total_elements = static_cast<size_t>(config.batch) * config.M * config.N;
        float *data = static_cast<float *>(std::malloc(total_elements * sizeof(float)));
        if (!data)
        {
            throw std::runtime_error("No se pudo asignar memoria para tensor sintético");
        }
        unsigned int seed = config.seed_provided ? (config.seed + static_cast<unsigned int>(iteration)) : static_cast<unsigned int>(std::random_device{}());
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (!config.pattern.empty())
        {
            std::fill(data, data + total_elements, 0.05f);
            if (config.pattern == "diagonal_high")
            {
                for (int b = 0; b < config.batch; ++b)
                {
                    const size_t base = static_cast<size_t>(b) * config.M * config.N;
                    const int diag = std::min(config.M, config.N);
                    for (int i = 0; i < diag; ++i)
                    {
                        data[base + static_cast<size_t>(i) * config.N + i] = 0.95f;
                    }
                }
            }
        }
        else
        {
            for (size_t idx = 0; idx < total_elements; ++idx)
            {
                data[idx] = dist(rng);
            }
        }
        return TensorResult(data, false, config.batch, config.M, config.N, 1, true);
    }
}

bool run_bench_case(const BenchConfig &config, BenchMetrics &metrics)
{
    if (!ensure_cuda_device_ready())
    {
        return false;
    }

    TensorResult base_tensor;
    bool base_tensor_loaded = false;
    if (!config.synthetic)
    {
        if (!leer_matriz_3d_desde_archivo(config.dataset_path.c_str(), base_tensor, config.batch, config.M, config.N, 1))
        {
            std::cerr << "BENCH_ERROR reason=failed_to_load_dataset label=" << config.label << std::endl;
            return false;
        }
        base_tensor_loaded = true;
    }

    auto check_cuda = [](cudaError_t status, const char *context)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string("CUDA error (") + context + "): " + cudaGetErrorString(status));
        }
    };

    for (int i = 0; i < config.iterations; ++i)
    {
        check_cuda(cudaDeviceReset(), "cudaDeviceReset");
        float *bootstrap_res = nullptr;
        float *d_bootstrap = nullptr;
        TensorResult synthetic_tensor;
        TensorResult *source_ptr = nullptr;
        TensorResult bootstrap_tensor;
        TensorResult *t2_ptr = nullptr;

        auto iteration_start = std::chrono::high_resolution_clock::now();

        try
        {
            if (config.synthetic)
            {
                synthetic_tensor = create_random_tensor(config, i);
                source_ptr = &synthetic_tensor;
            }
            else
            {
                source_ptr = &base_tensor;
            }

            if (config.skip_bootstrap)
            {
                t2_ptr = source_ptr;
            }
            else
            {
                bootstrap_res = static_cast<float *>(std::malloc(static_cast<size_t>(source_ptr->M) * source_ptr->N * config.replicas * sizeof(float)));
                if (!bootstrap_res)
                {
                    throw std::runtime_error("No se pudo asignar memoria para bootstrap");
                }
                d_bootstrap = bootstrap_wrapper(source_ptr->data, source_ptr->M, source_ptr->N, source_ptr->batch, config.replicas);
                check_cuda(cudaMemcpy(bootstrap_res, d_bootstrap,
                                      static_cast<size_t>(source_ptr->M) * source_ptr->N * config.replicas * sizeof(float),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy bootstrap");
                bootstrap_tensor = TensorResult(bootstrap_res, false, config.replicas, source_ptr->M, source_ptr->N);
                t2_ptr = &bootstrap_tensor;
            }
            printf("dims: %dx%dx%d",t2_ptr->M,t2_ptr->N,t2_ptr->batch);
            std::vector<TensorResult> paths;
            std::vector<TensorResult> values;
            std::vector<TensorResult> pure_paths;
            std::vector<TensorResult> pure_values;
            iterative_maxmin_cuadrado(*t2_ptr, config.threshold, config.order, paths, values, pure_paths, pure_values, true);
            auto cleanup_vector = [](std::vector<TensorResult> &tensors)
            {
                for (auto &tensor : tensors)
                {
                    safe_tensor_cleanup(tensor);
                }
                tensors.clear();
            };
            cleanup_vector(paths);
            cleanup_vector(values);
            cleanup_vector(pure_paths);
            cleanup_vector(pure_values);
            if (!config.skip_bootstrap)
            {
                safe_tensor_cleanup(*t2_ptr);
            }
            if (config.synthetic)
            {
                safe_tensor_cleanup(synthetic_tensor);
            }
        }
        catch (const std::exception &ex)
        {
            std::cerr << "BENCH_ERROR label=" << config.label << " iteration=" << i << " reason=exception message=\"" << ex.what() << "\"" << std::endl;
            if (d_bootstrap)
            {
                cudaFree(d_bootstrap);
            }
            if (bootstrap_res)
            {
                free(bootstrap_res);
            }
            if (config.synthetic)
            {
                safe_tensor_cleanup(synthetic_tensor);
            }
            return false;
        }

        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        auto iteration_end = std::chrono::high_resolution_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(iteration_end - iteration_start).count();
        metrics.durations_ms.push_back(elapsed_ms);
        std::cout << "BENCH_ITER label=" << config.label << " iteration=" << i << " elapsed_ms=" << elapsed_ms << std::endl;

        if (d_bootstrap)
        {
            check_cuda(cudaFree(d_bootstrap), "cudaFree d_bootstrap");
        }
    }

    if (base_tensor_loaded)
    {
        safe_tensor_cleanup(base_tensor);
    }

    return true;
}

int main(int args, char *argv[])
{
    BenchConfig bench_config;
    try
    {
        bench_config = parse_bench_args(args, argv);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error parsing arguments: " << ex.what() << std::endl;
        return 1;
    }

    if (bench_config.enabled)
    {
        if (bench_config.dataset_path.empty() || bench_config.batch <= 0 || bench_config.M <= 0 || bench_config.N <= 0)
        {
            std::cerr << "BENCH_ERROR missing_required_parameters" << std::endl;
            return 1;
        }
        BenchMetrics metrics;
        if (!run_bench_case(bench_config, metrics))
        {
            return 1;
        }
        std::cout << "BENCH_SUMMARY label=" << bench_config.label
                  << " iterations=" << metrics.durations_ms.size()
                  << " mean_ms=" << metrics.mean()
                  << " min_ms=" << metrics.min()
                  << " max_ms=" << metrics.max()
                  << " std_ms=" << metrics.stddev()
                  << std::endl;
        return 0;
    }

    if (args >= 2)
    {
        std::cout << argv[0] << "," << argv[1] << std::endl;
        std::string arg1 = argv[1];
        if (arg1 == "--validate")
        {
            generate_results();
        }
    }
    else
    {
        std::vector<double> reps = {100};
        TensorResult t1;
        const auto cc_path = repo_relative("datasets_txt/CC.txt");
        leer_matriz_3d_desde_archivo(cc_path.string().c_str(), t1, 10, 16, 16, 1);
        int M = t1.M, N = t1.N;
        int replicas;
        int iter = 1;
        auto start_total = std::chrono::high_resolution_clock::now();
        std::vector<double> tiempos_iteracion;

        cudaDeviceReset();
        for (int i = 0; i < reps.size(); i++)
        {
            replicas = reps[i];
            float *bootstrap_res, *d_bootstrap;
            bootstrap_res = (float *)malloc(M * N * replicas * sizeof(float));
            d_bootstrap = bootstrap_wrapper(t1.data, t1.M, t1.N, t1.batch, replicas);
            cudaMemcpy(bootstrap_res, d_bootstrap, M * N * replicas * sizeof(float), cudaMemcpyDeviceToHost);
            TensorResult t2 = TensorResult(bootstrap_res, false, replicas, M, N);
            auto start_iter = std::chrono::high_resolution_clock::now();
            std::vector<TensorResult> paths;
            std::vector<TensorResult> values;
            std::vector<TensorResult> pure_paths;
            std::vector<TensorResult> pure_values;
            iterative_maxmin_cuadrado(t2, 0.2, 3, paths, values, pure_paths, pure_values, true);

            // Fin de medición para esta iteración
            auto end_iter = std::chrono::high_resolution_clock::now();
            auto duration_iter = std::chrono::duration_cast<std::chrono::microseconds>(end_iter - start_iter);
            tiempos_iteracion.push_back(duration_iter.count() / 1000.0); // Convertir a milisegundos
            cudaFree(d_bootstrap);
        }

        // Liberar memoria
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
    }
    return 0;
}
