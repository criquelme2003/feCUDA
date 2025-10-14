#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <core/tensor.cuh>
#include <utils/cuda_utils.cuh>
#include <utils/logging.cuh>
#include <utils/file_io.cuh>
#include <headers.cuh>

// Medición de tiempo para benchmarks
class Timer
{
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

public:
    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop_ms()
    {
        end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

// Generar datos aleatorios para benchmarks
void fill_random(TensorResult &tensor, float min_val = 0.0f, float max_val = 1.0f)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    const size_t elements = tensor.total_elements();
    for (size_t i = 0; i < elements; ++i)
    {
        tensor.data[i] = dis(gen);
    }
}

// Benchmark de maxmin con diferentes tamaños
void benchmark_maxmin_sizes()
{
    LOG_INFO("=== BENCHMARK MAXMIN - DIFERENTES TAMAÑOS ===");

    const std::vector<std::tuple<int, int, int, int>> sizes = {
        {1, 16, 16, 16},
        {1, 32, 32, 32},
        {1, 64, 64, 64},
        {1, 128, 128, 128},
        {4, 64, 64, 64},
        {8, 32, 32, 32}};

    Timer timer;

    for (const auto &[batch, M, K, N] : sizes)
    {
        try
        {
            LOG_INFO("Benchmarking tamaño: [", batch, ",", M, ",", K, ",", N, "]");

            // Crear tensores
            auto host_A = TensorUtils::create_host_tensor(batch, M, K);
            auto host_B = TensorUtils::create_host_tensor(batch, K, N);

            // Llenar con datos aleatorios
            fill_random(host_A);
            fill_random(host_B);

            // Copiar a GPU
            auto gpu_A = TensorUtils::copy_to_device(host_A);
            auto gpu_B = TensorUtils::copy_to_device(host_B);

            // Crear resultados
            auto result_max = TensorUtils::create_device_tensor(batch, M, N);
            auto result_min = TensorUtils::create_device_tensor(batch, M, N, K);

            // Warmup
            maxmin(gpu_A, gpu_B, result_max, result_min, true);
            CHECK_CUDA(cudaDeviceSynchronize());

            // Benchmark con múltiples iteraciones
            const int iterations = 10;
            timer.start();

            for (int i = 0; i < iterations; ++i)
            {
                maxmin(gpu_A, gpu_B, result_max, result_min, true);
            }

            CHECK_CUDA(cudaDeviceSynchronize());
            double total_time = timer.stop_ms();
            double avg_time = total_time / iterations;

            // Calcular elementos por segundo
            const double elements = static_cast<double>(batch) * M * K * N;
            const double elements_per_sec = elements / (avg_time / 1000.0);

            LOG_INFO("  Tiempo promedio: ", avg_time, " ms");
            LOG_INFO("  Elementos/seg: ", elements_per_sec / 1e6, " M elements/s");
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Error en benchmark tamaño [", batch, ",", M, ",", K, ",", N, "]: ", e.what());
        }
    }
}

// Benchmark de memoria - host vs device transfers
void benchmark_memory_transfers()
{
    LOG_INFO("=== BENCHMARK TRANSFERENCIAS DE MEMORIA ===");

    const std::vector<std::tuple<int, int, int, int>> sizes = {
        {1, 256, 256, 1},
        {1, 512, 512, 1},
        {1, 1024, 1024, 1},
        {4, 512, 512, 1}};

    Timer timer;

    for (const auto &[batch, M, N, K] : sizes)
    {
        try
        {
            LOG_INFO("Benchmark memoria tamaño: [", batch, ",", M, ",", N, ",", K, "]");

            auto host_tensor = TensorUtils::create_host_tensor(batch, M, N, K);
            fill_random(host_tensor);

            const size_t bytes = host_tensor.size_bytes();
            const double mb = bytes / (1024.0 * 1024.0);

            // Host to Device
            timer.start();
            auto device_tensor = TensorUtils::copy_to_device(host_tensor);
            double h2d_time = timer.stop_ms();

            // Device to Host
            timer.start();
            auto host_copy = TensorUtils::copy_to_host(device_tensor);
            double d2h_time = timer.stop_ms();

            LOG_INFO("  Tamaño: ", mb, " MB");
            LOG_INFO("  H2D: ", h2d_time, " ms (", mb / h2d_time * 1000, " MB/s)");
            LOG_INFO("  D2H: ", d2h_time, " ms (", mb / d2h_time * 1000, " MB/s)");
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Error en benchmark memoria: ", e.what());
        }
    }
}

// Benchmark de algoritmo completo iterative_maxmin_cuadrado
void benchmark_iterative_maxmin()
{
    LOG_INFO("=== BENCHMARK ITERATIVE MAXMIN CUADRADO ===");

    try
    {
        // Cargar datos reales para benchmark
        TensorResult tensor_CC;
        bool success = FileIO::leer_matriz_3d_desde_archivo(
            "datasets_txt/CC.txt", tensor_CC, 10, 16, 16, 1);

        if (!success)
        {
            LOG_WARNING("No se pudo cargar dataset CC.txt, usando datos sintéticos");
            tensor_CC = TensorUtils::create_host_tensor(10, 16, 16);
            fill_random(tensor_CC, 0.0f, 1.0f);
        }

        std::vector<TensorResult> result_tensor_paths;
        std::vector<TensorResult> result_values_paths;
        std::vector<TensorResult> pure_tensor_paths;
        std::vector<TensorResult> pure_values_paths;

        const float threshold = 0.4f;
        const int order = 4;

        Timer timer;
        timer.start();

        iterative_maxmin_cuadrado(tensor_CC, threshold, order,
                                  result_tensor_paths, result_values_paths,
                                  pure_tensor_paths, pure_values_paths);

        double total_time = timer.stop_ms();

        LOG_INFO("Tiempo total iterative_maxmin_cuadrado: ", total_time, " ms");
        LOG_INFO("Paths encontrados: ", result_tensor_paths.size());
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error en benchmark iterative_maxmin: ", e.what());
    }
}

// Función principal de benchmarks
void run_all_benchmarks()
{
    LOG_INFO("=== EJECUTANDO TODOS LOS BENCHMARKS ===");

    // Calentar GPU
    CudaUtils::cuda_warmup();

    // Benchmark 1: Tamaños diferentes
    benchmark_maxmin_sizes();

    // Benchmark 2: Transferencias de memoria
    benchmark_memory_transfers();

    // Benchmark 3: Algoritmo completo
    benchmark_iterative_maxmin();

    LOG_INFO("=== BENCHMARKS COMPLETADOS ===");
}
