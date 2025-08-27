#include <iostream>
#include <exception>
#include "algorithms/maxmin.cuh"
#include "algorithms/indices.cuh"
#include "algorithms/paths.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/logging.cuh"
#include "utils/file_io.cuh"
#include "core/tensor.cuh"

// Sistema de menú simple
struct MenuSystem
{
    static int show_main_menu()
    {
        std::cout << "=== SISTEMA DE PRUEBAS FECUDA ===\n";
        std::cout << "1. Ejecutar max_min con archivos\n";
        std::cout << "2. Validar Kernel V1\n";
        std::cout << "3. Benchmark original\n";
        std::cout << "4. Tests unitarios\n";
        std::cout << "5. Benchmarks de rendimiento\n";
        std::cout << "Selecciona opción: ";

        int opcion;
        std::cin >> opcion;
        return opcion;
    }
};

// Declaraciones forward de funciones de test y benchmark
void run_tests();
void run_benchmarks();
void ejecutar_max_min_con_archivos(const char *archivo_A, const char *archivo_B,
                                   const char *output_min, const char *output_max,
                                   int batch_size, int M, int K, int N);
void validar_kernel_v1();
void ejecutar_benchmark_original();

int main()
{
    try
    {
        // Inicializar sistema
        CudaUtils::cuda_warmup();
        if (!CudaUtils::check_device_capabilities())
        {
            LOG_ERROR("El dispositivo no cumple con los requisitos mínimos");
            return EXIT_FAILURE;
        }

        const int opcion = MenuSystem::show_main_menu();

        switch (opcion)
        {
        case 1:
            // Ejemplo: reflexive vs reflexive
            ejecutar_max_min_con_archivos(
                "datasets_txt/reflexive.txt",
                "datasets_txt/reflexive.txt",
                "results/reflexive_min.txt",
                "results/reflexive_max.txt",
                1, 6, 6, 6);
            break;
        case 2:
            validar_kernel_v1();
            break;
        case 3:
            ejecutar_benchmark_original();
            break;
        case 4:
            run_tests();
            break;
        case 5:
            run_benchmarks();
            break;
        default:
            std::cout << "Opción inválida\n";
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error en main: ", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// Implementaciones mínimas (luego las moveremos a archivos separados)
void run_tests()
{
    LOG_INFO("Ejecutando tests unitarios...");
    // Llamar a funciones de tests/
}

void run_benchmarks()
{
    LOG_INFO("Ejecutando benchmarks...");
    // Llamar a funciones de benchmarks/
}

void ejecutar_max_min_con_archivos(const char *archivo_A, const char *archivo_B,
                                   const char *output_min, const char *output_max,
                                   int batch_size, int M, int K, int N)
{
    LOG_INFO("Procesando archivos: ", archivo_A, " y ", archivo_B);
    // Implementación temporal - luego mover a examples/
}

void validar_kernel_v1()
{
    LOG_INFO("Validando Kernel V1...");
    // Implementación temporal - luego mover a tests/
}

void ejecutar_benchmark_original()
{
    LOG_INFO("Ejecutando benchmark original...");
    // Implementación temporal - luego mover a benchmarks/
}
