#include <iostream>
#include <exception>
#include "algorithms/maxmin.cuh"
#include "algorithms/indices.cuh"
#include "algorithms/paths.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/logging.cuh"
#include "utils/file_io.cuh"
#include "core/tensor.cuh"
#include "kernels/kernels.cuh"
#include "test/test.cuh"
#include "../../include/utils.cuh"
#include "temp.cuh"
#include <random>
#include <time.h>
#include <bootstrap.cuh>
#include <temp.cuh>

// Sistema de menú simple
struct MenuSystem
{
    static int show_main_menu()
    {
        std::cout << "=== SISTEMA DE PRUEBAS FECUDA ===\n";
        std::cout << "1. Ejecutar test\n";

        int opcion;
        std::cin >> opcion;
        return opcion;
    }
};

// Declaraciones forward de funciones de test y benchmark
void run_tests();

int main()
{
    TensorResult t1;
    leer_matriz_3d_desde_archivo("../datasets_txt/CC.txt", t1, 10, 16, 16, 1);

    float *bootstrap_res, *d_bootstrap;

    bootstrap_res = (float *)malloc(t1.total_elements() * sizeof(float));
    d_bootstrap = bootstrap_wrapper(t1.data, t1.M, t1.N, t1.batch, 100);

    cudaMemcpy(bootstrap_res, d_bootstrap, t1.total_elements() * sizeof(float), cudaMemcpyDeviceToHost);

    save_tensor_4d_as_file(bootstrap_res, 100, 16, 16, 1, "bootstrap_cc.txt");
}

// Implementaciones mínimas (luego las moveremos a archivos separados)
void run_tests()
{

    // TensorResult tensor1, result2;

    // bool carga_ok = leer_matriz_3d_desde_archivo("../datasets_txt/CC.txt", tensor1, 10, 16, 16, 1);

    // if(!carga_ok)
    // {
    //     LOG_ERROR("Fallo al cargar tensor de prueba.");
    //     return;
    // }

    // LOG_INFO("Test feEmpirical().");

    // FEempirical(tensor1, result2, 10);
    // imprimir_tensor(result2);

    // // medir el tiempo
    // float global_time = 0.0f;
    // for (int i = 0; i < 100; ++i)
    // {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     FEempirical(tensor1, result2, 1000);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double, std::milli> duration = end - start;
    //     global_time += duration.count();
    // }
    // LOG_INFO("Tiempo promedio : ", global_time / 100.0f, " ms");

    validar_algoritmos_maxmin("Maxmin v1");
}
