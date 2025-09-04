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
    TensorResult data;
    int reps = 1000;
    leer_matriz_3d_desde_archivo("../datasets_txt/CC.txt", data, 10, 16, 16, 1);

    float *out_data = (float *)malloc(data.M * data.N * 1000 * sizeof(float));
    float *perc = (float *)malloc(1000 * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());

    // Distribución uniforme entre 0 y 10 (enteros)
    std::uniform_int_distribution<int> dist(0, 10);

    for (int i = 0; i < reps; ++i)
    {
        // valores entre 0 y 1 de 1 decimal
        perc[i] = (float) dist(gen) / 10.0f;
    }


    bootstrap(10, data.M, data.N, data.batch, reps, data.data, out_data, perc);

    imprimir_tensor(TensorResult(out_data, false, reps, data.M, data.N, 1, false));

    
    return 0;
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
