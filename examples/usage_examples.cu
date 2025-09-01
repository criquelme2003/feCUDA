#include <iostream>
#include "algorithms/maxmin.cuh"
#include <core/tensor.cuh>
#include <utils/cuda_utils.cuh>
#include <utils/logging.cuh>
#include <utils/file_io.cuh>

// Ejemplo básico de uso de maxmin
void example_basic_maxmin()
{
    LOG_INFO("=== EJEMPLO BÁSICO MAXMIN ===");

    try
    {
        // Crear tensores pequeños para demostración
        const int batch = 1, M = 3, K = 3, N = 3;

        auto tensor_A = TensorUtils::create_host_tensor(batch, M, K);
        auto tensor_B = TensorUtils::create_host_tensor(batch, K, N);

        // Llenar con datos de ejemplo
        for (int i = 0; i < 9; ++i)
        {
            tensor_A.data[i] = static_cast<float>(i + 1);
            tensor_B.data[i] = static_cast<float>((i % 3) + 1);
        }

        LOG_INFO("Tensor A (3x3):");
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                std::cout << tensor_A.data[i * 3 + j] << " ";
            }
            std::cout << std::endl;
        }

        LOG_INFO("Tensor B (3x3):");
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                std::cout << tensor_B.data[i * 3 + j] << " ";
            }
            std::cout << std::endl;
        }

        // Ejecutar maxmin
        auto gpu_A = TensorUtils::copy_to_device(tensor_A);
        auto gpu_B = TensorUtils::copy_to_device(tensor_B);

        auto result_max = TensorUtils::create_device_tensor(batch, M, N);
        auto result_min = TensorUtils::create_device_tensor(batch, M, N, K);

        maxmin(gpu_A, gpu_B, result_max, result_min, false);

        // Mostrar resultados
        auto host_max = TensorUtils::copy_to_host(result_max);

        LOG_INFO("Resultado MAX (3x3):");
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                std::cout << host_max.data[i * 3 + j] << " ";
            }
            std::cout << std::endl;
        }

        LOG_INFO("Ejemplo básico completado");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error en ejemplo básico: ", e.what());
    }
}

// Ejemplo de procesamiento de dataset real
void example_dataset_processing()
{
    LOG_INFO("=== EJEMPLO PROCESAMIENTO DATASET ===");

    try
    {
        const char *datasets[] = {"reflexive.txt", "CC.txt", "EE.txt"};

        for (const char *dataset : datasets)
        {
            std::string filepath = std::string("datasets_txt/") + dataset;

            LOG_INFO("Procesando dataset: ", dataset);

            TensorResult tensor;
            bool success = false;

            // Intentar diferentes configuraciones de dimensiones
            if (!success)
            {
                success = FileIO::leer_matriz_3d_desde_archivo(
                    filepath.c_str(), tensor, 1, 6, 6);
            }
            if (!success)
            {
                success = FileIO::leer_matriz_3d_desde_archivo(
                    filepath.c_str(), tensor, 10, 16, 16);
            }

            if (success)
            {
                LOG_INFO("Dataset cargado: [", tensor.batch, ",", tensor.M, ",", tensor.N, ",", tensor.K, "]");

                // Procesar con maxmin (solo una muestra pequeña)
                if (tensor.M <= 64 && tensor.N <= 64)
                {
                    auto gpu_tensor = TensorUtils::copy_to_device(tensor);

                    // Crear segundo tensor (identidad o copia)
                    auto tensor2 = tensor.clone();
                    auto gpu_tensor2 = TensorUtils::copy_to_device(tensor2);

                    auto result_max = TensorUtils::create_device_tensor(
                        tensor.batch, tensor.M, tensor.N);
                    auto result_min = TensorUtils::create_device_tensor(
                        tensor.batch, tensor.M, tensor.N, tensor.K);

                    maxmin(gpu_tensor, gpu_tensor2, result_max, result_min, false);

                    // Guardar resultado
                    std::string output_file = std::string("results/") + dataset + "_processed.txt";
                    auto host_result = TensorUtils::copy_to_host(result_max);

                    FileIO::guardar_tensor_como_archivo(host_result, output_file.c_str());

                    LOG_INFO("Resultado guardado en: ", output_file);
                }
                else
                {
                    LOG_INFO("Dataset muy grande para ejemplo, saltando procesamiento");
                }
            }
            else
            {
                LOG_WARNING("No se pudo cargar dataset: ", dataset);
            }
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error en ejemplo dataset: ", e.what());
    }
}

// Ejemplo de uso del algoritmo iterativo
void example_iterative_algorithm()
{
    LOG_INFO("=== EJEMPLO ALGORITMO ITERATIVO ===");

    try
    {
        // Crear tensor de ejemplo pequeño
        auto tensor = TensorUtils::create_host_tensor(2, 4, 4);

        // Llenar con datos de prueba
        for (int i = 0; i < 32; ++i)
        {
            tensor.data[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        std::vector<TensorResult> result_tensor_paths;
        std::vector<TensorResult> result_values_paths;
        std::vector<TensorResult> pure_tensor_paths;
        std::vector<TensorResult> pure_values_paths;

        const float threshold = 0.3f;
        const int order = 3;

        LOG_INFO("Ejecutando algoritmo iterativo con threshold=", threshold, ", order=", order);

        iterative_maxmin_cuadrado(tensor, threshold, order,
                                  result_tensor_paths, result_values_paths,
                                  pure_tensor_paths, pure_values_paths);

        LOG_INFO("Paths tensor encontrados: ", result_tensor_paths.size());
        LOG_INFO("Paths valores encontrados: ", result_values_paths.size());
        LOG_INFO("Pure tensor paths: ", pure_tensor_paths.size());
        LOG_INFO("Pure values paths: ", pure_values_paths.size());
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error en ejemplo iterativo: ", e.what());
    }
}

// Función principal de ejemplos
void run_all_examples()
{
    LOG_INFO("=== EJECUTANDO TODOS LOS EJEMPLOS ===");

    // Inicializar sistema
    CudaUtils::cuda_warmup();

    // Ejemplo 1: Uso básico
    example_basic_maxmin();

    std::cout << "\n"
              << std::string(50, '=') << "\n\n";

    // Ejemplo 2: Procesamiento de datasets
    example_dataset_processing();

    std::cout << "\n"
              << std::string(50, '=') << "\n\n";

    // Ejemplo 3: Algoritmo iterativo
    example_iterative_algorithm();

    LOG_INFO("=== EJEMPLOS COMPLETADOS ===");
}
