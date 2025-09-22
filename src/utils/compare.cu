#include "../../include/utils.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Función para comparar dos tensores elemento a elemento
bool comparar_tensores(const TensorResult &tensor_a, const TensorResult &tensor_b,
                       float tolerancia, bool verbose)
{

    // Validaciones básicas
    if (tensor_a.data == nullptr)
    {
        printf("Error: El tensor A es nulo\n");
        return false;
    }

    if (tensor_b.data == nullptr)
    {
        printf("Error: El tensor B es nulo\n");
        return false;
    }

    // Comparar dimensiones
    if (tensor_a.batch != tensor_b.batch ||
        tensor_a.M != tensor_b.M ||
        tensor_a.N != tensor_b.N ||
        tensor_a.K != tensor_b.K)
    {

        printf("Error: Las dimensiones de los tensores no coinciden\n");
        printf("  Tensor A: batch=%d, M=%d, N=%d, K=%d\n",
               tensor_a.batch, tensor_a.M, tensor_a.N, tensor_a.K);
        printf("  Tensor B: batch=%d, M=%d, N=%d, K=%d\n",
               tensor_b.batch, tensor_b.M, tensor_b.N, tensor_b.K);
        return false;
    }

    // Preparar datos para comparar (copiar a host si están en device)
    float *datos_a = nullptr;
    float *datos_b = nullptr;
    bool liberar_a = false;
    bool liberar_b = false;

    size_t size_bytes = tensor_a.size_bytes();

    // Manejar tensor A
    if (tensor_a.is_device_ptr)
    {
        datos_a = (float *)malloc(size_bytes);
        if (datos_a == nullptr)
        {
            printf("Error: No se pudo asignar memoria temporal para tensor A\n");
            return false;
        }

        cudaError_t error = cudaMemcpy(datos_a, tensor_a.data, size_bytes, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("Error: No se pudo copiar tensor A desde device: %s\n", cudaGetErrorString(error));
            free(datos_a);
            return false;
        }
        liberar_a = true;
    }
    else
    {
        datos_a = tensor_a.data;
    }

    // Manejar tensor B
    if (tensor_b.is_device_ptr)
    {
        datos_b = (float *)malloc(size_bytes);
        if (datos_b == nullptr)
        {
            printf("Error: No se pudo asignar memoria temporal para tensor B\n");
            if (liberar_a)
                free(datos_a);
            return false;
        }

        cudaError_t error = cudaMemcpy(datos_b, tensor_b.data, size_bytes, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("Error: No se pudo copiar tensor B desde device: %s\n", cudaGetErrorString(error));
            if (liberar_a)
                free(datos_a);
            free(datos_b);
            return false;
        }
        liberar_b = true;
    }
    else
    {
        datos_b = tensor_b.data;
    }

    // Realizar comparación elemento a elemento
    size_t total_elementos = tensor_a.total_elements();
    size_t diferencias_encontradas = 0;
    size_t max_diferencias_mostrar = 10; // Limitar la salida

    if (verbose)
    {
        printf("\n=== COMPARANDO TENSORES ===\n");
        printf("Dimensiones: batch=%d, M=%d, N=%d, K=%d\n",
               tensor_a.batch, tensor_a.M, tensor_a.N, tensor_a.K);
        printf("Total elementos: %zu\n", total_elementos);
        printf("Tolerancia: %.2e\n", tolerancia);
        printf("\n");
    }

    for (int b = 0; b < tensor_a.batch; b++)
    {
        for (int m = 0; m < tensor_a.M; m++)
        {
            for (int n = 0; n < tensor_a.N; n++)
            {
                for (int k = 0; k < tensor_a.K; k++)
                {

                    size_t idx = b * (tensor_a.M * tensor_a.N * tensor_a.K) +
                                 m * (tensor_a.N * tensor_a.K) +
                                 n * tensor_a.K + k;

                    float val_a = datos_a[idx];
                    float val_b = datos_b[idx];
                    float diferencia = fabsf(val_a - val_b);

                    if (diferencia > tolerancia)
                    {
                        diferencias_encontradas++;

                        if (verbose && diferencias_encontradas <= max_diferencias_mostrar)
                        {
                            printf("Diferencia en [batch=%d, M=%d, N=%d, K=%d]:\n", b, m, n, k);
                            printf("  Tensor A: %.6f\n", val_a);
                            printf("  Tensor B: %.6f\n", val_b);
                            printf("  Diferencia: %.6f (tolerancia: %.2e)\n", diferencia, tolerancia);
                            printf("\n");
                        }
                    }
                }
            }
        }
    }

    // Mostrar resumen
    if (verbose)
    {
        printf("=== RESUMEN DE COMPARACIÓN ===\n");
        printf("Elementos comparados: %zu\n", total_elementos);
        printf("Diferencias encontradas: %zu\n", diferencias_encontradas);

        if (diferencias_encontradas > max_diferencias_mostrar)
        {
            printf("(Solo se mostraron las primeras %zu diferencias)\n", max_diferencias_mostrar);
        }

        if (diferencias_encontradas == 0)
        {
            printf("✅ Los tensores son IDÉNTICOS (dentro de la tolerancia)\n");
        }
        else
        {
            printf("❌ Los tensores son DIFERENTES\n");
            printf("Porcentaje de elementos diferentes: %.2f%%\n",
                   (float)diferencias_encontradas / total_elementos * 100.0f);
        }
        printf("\n");
    }

    // Limpiar memoria temporal
    if (liberar_a)
        free(datos_a);
    if (liberar_b)
        free(datos_b);

    // Retornar true si son iguales, false si hay diferencias
    return (diferencias_encontradas == 0);
}
