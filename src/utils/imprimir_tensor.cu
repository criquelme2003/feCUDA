#include "../../include/utils.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Función para imprimir TensorResult
void imprimir_tensor(const TensorResult &tensor, int max_rows, int max_cols,
                     const char *nombre_tensor, bool mostrar_estadisticas)
{
    if (tensor.data == nullptr)
    {
        printf("Error: El tensor '%s' es nulo o inválido\n", nombre_tensor);
        return;
    }

    // Determinar si es tensor 3D o 4D
    bool es_4d = (tensor.K > 1);

    // Calcular número total de elementos
    size_t total_elementos = tensor.batch * tensor.M * tensor.N * tensor.K;
    size_t size_bytes = total_elementos * sizeof(float);

    printf("\n======== %s ========\n", nombre_tensor);
    printf("Dimensiones: batch=%d, M=%d, N=%d, K=%d\n", tensor.batch, tensor.M, tensor.N, tensor.K);
    printf("Total elementos: %zu\n", total_elementos);
    printf("Tamaño en memoria: %zu bytes (%.2f KB)\n", size_bytes, size_bytes / 1024.0f);
    printf("Ubicación: %s\n", tensor.is_device_ptr ? "GPU (device)" : "CPU (host)");

    // Preparar datos para imprimir
    float *datos_host = nullptr;
    bool necesita_liberar = false;

    if (tensor.is_device_ptr)
    {
        // Si está en device, copiarlo a host temporalmente
        datos_host = (float *)malloc(size_bytes);
        if (datos_host == nullptr)
        {
            printf("Error: No se pudo asignar memoria temporal para imprimir tensor\n");
            return;
        }

        cudaError_t error = cudaMemcpy(datos_host, tensor.data, size_bytes, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("Error: No se pudo copiar tensor desde device: %s\n", cudaGetErrorString(error));
            free(datos_host);
            return;
        }
        necesita_liberar = true;
    }
    else
    {
        // Si ya está en host, usar directamente
        datos_host = tensor.data;
    }

    // Límites para imprimir
    int print_batches = (tensor.batch < 3) ? tensor.batch : 3;
    int print_M = (tensor.M < max_rows) ? tensor.M : max_rows;
    int print_N = (tensor.N < max_cols) ? tensor.N : max_cols;
    int print_K = es_4d ? ((tensor.K < 6) ? tensor.K : 6) : 1;

    // Variables para estadísticas
    float min_val = datos_host[0];
    float max_val = datos_host[0];
    float suma = 0.0f;
    int elementos_procesados = 0;

    // Imprimir datos en formato matricial
    printf("\nDatos (formato matricial):\n");

    for (int b = 0; b < print_batches; ++b)
    {
        if (tensor.batch > 1)
        {
            printf("\n");
            printf("╔═══════════════════════════════════════╗\n");
            printf("║            BATCH %d/%d                ║\n", b + 1, tensor.batch);
            printf("╚═══════════════════════════════════════╝\n");
        }

        if (es_4d)
        {
            // Nueva lógica: Para cada M mostrar una matriz N x K (filas=N, columnas=K)
            for (int i = 0; i < print_M; ++i)
            {
                printf("\n=== M[%d/%d] -> Matriz (N x K) ===\n", i + 1, tensor.M);

                // Encabezados de columnas (K)
                printf("      ");
                int print_K_cols = (tensor.K < 10) ? tensor.K : 10; // limitar columnas K impresas
                for (int k = 0; k < print_K_cols; ++k)
                {
                    printf("   K%-3d ", k);
                }
                if (print_K_cols < tensor.K)
                    printf("  ...");
                printf("\n");

                for (int n = 0; n < print_N; ++n)
                {
                    printf("N%2d:  ", n); // índice de fila N
                    for (int k = 0; k < print_K_cols; ++k)
                    {
                        // idx = b*(M*N*K) + i*(N*K) + n*(K) + k
                        int idx = b * (tensor.M * tensor.N * tensor.K) +
                                  i * (tensor.N * tensor.K) +
                                  n * tensor.K + k;
                        float val = datos_host[idx];

                        if(val<0){
                            //printf("\033[1;31m%8.4f \033[0m", val); // Rojo para negativos
                            printf("\033[32;1m%8.4f \033[0m", val); // Rojo para negativos
                        
                        }else{

                            printf("%8.4f ", val);
                        }
                        if (mostrar_estadisticas)
                        {
                            min_val = (val < min_val) ? val : min_val;
                            max_val = (val > max_val) ? val : max_val;
                            suma += val;
                            elementos_procesados++;
                        }
                    }
                    if (print_K_cols < tensor.K)
                        printf("  ...");
                    printf("\n");
                }
                if (print_N < tensor.N)
                    printf("      ... (%d filas N más)\n", tensor.N - print_N);
            }
            if (print_M < tensor.M)
                printf("\n... (%d bloques M más)\n", tensor.M - print_M);
        }
        else
        {
            // Tensor 3D: Mostrar como una sola matriz

            // Imprimir encabezados de columnas
            printf("     ");
            for (int j = 0; j < print_N; ++j)
            {
                printf("%8d ", j);
            }
            if (print_N < tensor.N)
                printf("  ...");
            printf("\n");

            // Imprimir matriz
            for (int i = 0; i < print_M; ++i)
            {
                printf("%2d:  ", i); // Índice de fila

                for (int j = 0; j < print_N; ++j)
                {
                    // Índice: batch * (M * N) + i * N + j
                    int idx = b * (tensor.M * tensor.N) + i * tensor.N + j;

                    float val = datos_host[idx];
                    printf("%8.4f ", val);

                    // Actualizar estadísticas
                    if (mostrar_estadisticas)
                    {
                        min_val = (val < min_val) ? val : min_val;
                        max_val = (val > max_val) ? val : max_val;
                        suma += val;
                        elementos_procesados++;
                    }
                }

                if (print_N < tensor.N)
                    printf("  ...");
                printf("\n");
            }

            if (print_M < tensor.M)
                printf("     ... (%d filas más)\n", tensor.M - print_M);
        }

        // Separador entre batches
        if (b < print_batches - 1 && tensor.batch > 1)
        {
            printf("\n");
            for (int sep = 0; sep < 50; sep++)
                printf("-");
            printf("\n");
        }
    }

    if (print_batches < tensor.batch)
        printf("\n... (%d batches más)\n", tensor.batch - print_batches);

    // Mostrar estadísticas si se solicita
    if (mostrar_estadisticas && elementos_procesados > 0)
    {
        float promedio = suma / elementos_procesados;

        printf("\n--- Estadísticas (de los elementos mostrados) ---\n");
        printf("Elementos analizados: %d de %zu\n", elementos_procesados, total_elementos);
        printf("Mínimo: %f\n", min_val);
        printf("Máximo: %f\n", max_val);
        printf("Promedio: %f\n", promedio);
        printf("Suma: %f\n", suma);
    }

    printf("\n======== Fin %s ========\n\n", nombre_tensor);

    // Liberar memoria temporal si es necesario
    if (necesita_liberar)
    {
        free(datos_host);
    }
}
