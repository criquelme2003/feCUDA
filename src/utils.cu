#include "utils.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <types.cuh>

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
                        printf("%8.4f ", val);
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

// Función para leer matrices 3D desde archivo de texto
bool leer_matriz_3d_desde_archivo(const char *archivo, TensorResult &tensor,
                                  int batch, int M, int N, int K)
{
    FILE *file = fopen(archivo, "r");
    if (file == nullptr)
    {
        printf("Error: No se pudo abrir el archivo '%s'\n", archivo);
        return false;
    }

    // Calcular número total de elementos
    size_t total_elementos = batch * M * N * K;
    size_t size_bytes = total_elementos * sizeof(float);

    // Alocar memoria para los datos
    float *datos = (float *)malloc(size_bytes);
    if (datos == nullptr)
    {
        printf("Error: No se pudo asignar memoria para %zu elementos\n", total_elementos);
        fclose(file);
        return false;
    }

    printf("Leyendo matriz 3D desde '%s'...\n", archivo);
    printf("Dimensiones esperadas: batch=%d, M=%d, N=%d, K=%d\n", batch, M, N, K);
    printf("Total elementos a leer: %zu\n", total_elementos);

    // Leer datos del archivo
    size_t elementos_leidos = 0;
    char linea[1024];
    int linea_numero = 0;

    while (fgets(linea, sizeof(linea), file) != nullptr && elementos_leidos < total_elementos)
    {
        linea_numero++;

        // Saltar líneas que empiecen con # (comentarios)
        if (linea[0] == '#' || linea[0] == '\n')
            continue;

        // Parsear números de la línea
        char *token = strtok(linea, " \t\n,;");
        while (token != nullptr && elementos_leidos < total_elementos)
        {
            float valor;
            if (sscanf(token, "%f", &valor) == 1)
            {
                datos[elementos_leidos] = valor;
                elementos_leidos++;
            }
            else
            {
                printf("Advertencia: No se pudo parsear '%s' en línea %d\n", token, linea_numero);
            }
            token = strtok(nullptr, " \t\n,;");
        }
    }

    fclose(file);

    // Verificar si se leyeron todos los elementos
    if (elementos_leidos < total_elementos)
    {
        printf("Advertencia: Solo se leyeron %zu de %zu elementos esperados\n",
               elementos_leidos, total_elementos);

        // Rellenar con ceros los elementos faltantes
        for (size_t i = elementos_leidos; i < total_elementos; i++)
        {
            datos[i] = 0.0f;
        }
    }
    else if (elementos_leidos > total_elementos)
    {
        printf("Advertencia: Se encontraron más elementos de los esperados (%zu), ignorando extras\n",
               elementos_leidos);
    }

    // Configurar TensorResult
    tensor.data = datos;
    tensor.is_device_ptr = false;
    tensor.batch = batch;
    tensor.M = M;
    tensor.N = N;
    tensor.K = K;

    printf("Matriz 3D cargada exitosamente: %zu elementos\n", total_elementos);

    // Mostrar algunos valores para verificación
    printf("Primeros valores: ");
    int mostrar = (total_elementos < 10) ? total_elementos : 10;
    for (int i = 0; i < mostrar; i++)
    {
        printf("%.3f ", datos[i]);
    }
    if (total_elementos > mostrar)
        printf("...");
    printf("\n");

    return true;
}

// Función para limpiar y verificar el estado del dispositivo CUDA
void cuda_cleanup_and_check()
{
    // Sincronizar para asegurar que todas las operaciones han terminado
    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess)
    {
        printf("Warning: Error durante sincronización: %s\n", cudaGetErrorString(syncError));
    }

    // Obtener información de memoria
    size_t free_memory, total_memory;
    cudaError_t memError = cudaMemGetInfo(&free_memory, &total_memory);
    if (memError == cudaSuccess)
    {
        printf("Memoria GPU - Libre: %.2f MB, Total: %.2f MB, Usada: %.2f MB\n",
               free_memory / (1024.0 * 1024.0),
               total_memory / (1024.0 * 1024.0),
               (total_memory - free_memory) / (1024.0 * 1024.0));
    }

    // NO resetear el dispositivo para evitar problemas con contextos persistentes
    printf("Dispositivo CUDA sincronizado\n");
}

// Función para limpiar memoria de TensorResult de forma segura
void safe_tensor_cleanup(TensorResult &tensor)
{
    if (tensor.data && tensor.owns_memory)
    {
        if (tensor.is_device_ptr)
        {
            cudaFree(tensor.data);
        }
        else
        {
            free(tensor.data);
        }
    }
    tensor.data = nullptr;
    tensor.owns_memory = false;
    tensor.batch = tensor.M = tensor.N = tensor.K = 0;
}

// Función para crear una copia del tensor en memoria host
TensorResult copy_tensor(const TensorResult &src)
{
    TensorResult dst;
    size_t size = src.batch * src.M * src.N * src.K * sizeof(float);
    dst.data = (float *)malloc(size);
    memcpy(dst.data, src.data, size);
    dst.is_device_ptr = false;
    dst.batch = src.batch;
    dst.M = src.M;
    dst.N = src.N;
    dst.K = src.K;
    dst.owns_memory = true;
    return dst;
}
