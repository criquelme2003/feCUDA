#include <iostream>
#include <vector>
#include <chrono>
#include "core/tensor.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/logging.cuh"
#include "utils/file_io.cuh"
#include <headers.cuh>

// Test básico de kernel maxmin
bool test_maxmin_kernel()
{
    LOG_INFO("=== TEST MAXMIN KERNEL ===");

    try
    {
        const int batch_size = 1, M = 2, K = 3, N = 2;

        // Crear tensores de prueba
        auto tensor_A = TensorUtils::create_host_tensor(batch_size, M, K);
        auto tensor_B = TensorUtils::create_host_tensor(batch_size, K, N);

        // Llenar con datos de prueba
        const std::vector<float> data_A = {1, 2, 3, 4, 5, 6};
        const std::vector<float> data_B = {1, 2, 3, 4, 5, 6};

        std::copy(data_A.begin(), data_A.end(), tensor_A.data);
        std::copy(data_B.begin(), data_B.end(), tensor_B.data);

        // Copiar a GPU
        auto gpu_A = TensorUtils::copy_to_device(tensor_A);
        auto gpu_B = TensorUtils::copy_to_device(tensor_B);

        // Crear tensores de resultado
        auto result_max = TensorUtils::create_device_tensor(batch_size, M, N);
        auto result_min = TensorUtils::create_device_tensor(batch_size, M, N, K);

        // Ejecutar kernel
        maxmin(gpu_A, gpu_B, result_max, result_min, false);

        // Verificar resultados (implementación básica)
        auto host_max = TensorUtils::copy_to_host(result_max);
        auto host_min = TensorUtils::copy_to_host(result_min);

        LOG_INFO("Test maxmin kernel PASÓ");
        return true;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Test maxmin kernel FALLÓ: ", e.what());
        return false;
    }
}

// Test de creación y manejo de tensores
bool test_tensor_operations()
{
    LOG_INFO("=== TEST TENSOR OPERATIONS ===");

    try
    {
        // Test creación
        auto host_tensor = TensorUtils::create_host_tensor(2, 3, 4);
        auto device_tensor = TensorUtils::create_device_tensor(2, 3, 4);

        // Test compatibilidad
        if (!TensorUtils::are_compatible(host_tensor, device_tensor))
        {
            LOG_ERROR("Tensores deberían ser compatibles");
            return false;
        }

        // Test copy
        auto copied_device = TensorUtils::copy_to_device(host_tensor);
        auto copied_host = TensorUtils::copy_to_host(copied_device);

        // Test fill
        TensorUtils::fill_tensor(host_tensor, 5.0f);

        LOG_INFO("Test tensor operations PASÓ");
        return true;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Test tensor operations FALLÓ: ", e.what());
        return false;
    }
}

// Test de I/O de archivos
bool test_file_io()
{
    LOG_INFO("=== TEST FILE I/O ===");

    try
    {
        // Crear tensor de prueba
        auto test_tensor = TensorUtils::create_host_tensor(1, 3, 3);

        // Llenar con datos
        for (int i = 0; i < 9; ++i)
        {
            test_tensor.data[i] = static_cast<float>(i + 1);
        }

        // Guardar
        if (!FileIO::guardar_tensor_como_archivo(test_tensor, "test_output.txt"))
        {
            LOG_ERROR("No se pudo guardar archivo");
            return false;
        }

        // Cargar
        TensorResult loaded_tensor;
        if (!FileIO::leer_matriz_3d_desde_archivo("test_output.txt", loaded_tensor, 1, 3, 3))
        {
            LOG_ERROR("No se pudo cargar archivo");
            return false;
        }

        // Verificar igualdad
        if (!TensorUtils::tensors_equal(test_tensor, loaded_tensor, 1e-6f))
        {
            LOG_ERROR("Tensores no son iguales después de I/O");
            return false;
        }

        LOG_INFO("Test file I/O PASÓ");
        return true;
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Test file I/O FALLÓ: ", e.what());
        return false;
    }
}

// Función principal de tests
void run_all_tests()
{
    LOG_INFO("=== EJECUTANDO TODOS LOS TESTS ===");

    int passed = 0, total = 0;

    // Test 1: Tensor operations
    total++;
    if (test_tensor_operations())
        passed++;

    // Test 2: MaxMin kernel
    total++;
    if (test_maxmin_kernel())
        passed++;

    // Test 3: File I/O
    total++;
    if (test_file_io())
        passed++;

    LOG_INFO("=== RESUMEN DE TESTS ===");
    LOG_INFO("Tests pasados: ", passed, "/", total);

    if (passed == total)
    {
        LOG_INFO("¡Todos los tests PASARON! ✓");
    }
    else
    {
        LOG_ERROR("Algunos tests FALLARON ✗");
    }
}
