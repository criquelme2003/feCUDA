#include <iostream>
#include <vector>
#include <cstring>
#include "headers.cuh"
#include "utils.cuh"
#include "types.cuh"

int main()
{

    TensorResult tensor_desde_archivo;
    bool exito = leer_matriz_3d_desde_archivo("../datasets_txt/CC.txt", tensor_desde_archivo, 10, 16, 16, 1);

    if (!exito)
    {
        printf("\nError: No se pudo cargar el tensor desde archivo\n");
    }

    // === Probando función iterative_maxmin_cuadrado ===
    printf("\n=== Probando función iterative_maxmin_cuadrado ===\n");

    // Usar el tensor cargado desde archivo si está disponible, sino usar datos hardcodeados
    TensorResult test_tensor;
    bool usar_archivo = exito;
    test_tensor = tensor_desde_archivo;

    std::vector<TensorResult> result_tensor_paths;
    std::vector<TensorResult> result_values_paths;
    std::vector<TensorResult> pure_tensor_paths;
    std::vector<TensorResult> pure_values_paths;

    float test_threshold = 0.4f;
    int test_order = 4;

    iterative_maxmin_cuadrado(test_tensor, test_threshold, test_order, result_tensor_paths, result_values_paths, pure_tensor_paths, pure_values_paths);



    // Liberar memoria del tensor de prueba
    if (usar_archivo)
    {
        if (tensor_desde_archivo.data)
            free(tensor_desde_archivo.data);
    }
    else
    {
        if (test_tensor.data)
            free(test_tensor.data);
    }

    printf("\nPruebas completadas.\n");
    return 0;
}
