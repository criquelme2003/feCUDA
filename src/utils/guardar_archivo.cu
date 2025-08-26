#include "../../include/utils.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Función para guardar tensor como archivo en formato compatible con leer_matriz_3d_desde_archivo
bool guardar_tensor_como_archivo(const TensorResult &tensor, const char *nombre_archivo)
{

    if (tensor.data == nullptr)
    {
        printf("Error: Los datos del tensor son nulos\n");
        return false;
    }

    if (nombre_archivo == nullptr)
    {
        printf("Error: El nombre del archivo es nulo\n");
        return false;
    }

    // Validar que el tensor sea 3D (K=1)
    if (tensor.K != 1)
    {
        printf("Error: Esta función solo acepta tensores 3D (K=1). El tensor tiene K=%d\n", tensor.K);
        return false;
    }

    // Preparar datos para escribir
    float *datos_host = nullptr;
    bool necesita_liberar = false;

    if (tensor.is_device_ptr)
    {
        // Si está en device, copiarlo a host temporalmente
        size_t size_bytes = tensor.size_bytes();
        datos_host = (float *)malloc(size_bytes);
        if (datos_host == nullptr)
        {
            printf("Error: No se pudo asignar memoria temporal para guardar tensor\n");
            return false;
        }

        cudaError_t error = cudaMemcpy(datos_host, tensor.data, size_bytes, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("Error: No se pudo copiar tensor desde device: %s\n", cudaGetErrorString(error));
            free(datos_host);
            return false;
        }
        necesita_liberar = true;
    }
    else
    {
        // Si ya está en host, usar directamente
        datos_host = tensor.data;
    }

    FILE *file = fopen(nombre_archivo, "w");
    if (file == nullptr)
    {
        printf("Error: No se pudo crear el archivo '%s'\n", nombre_archivo);
        if (necesita_liberar)
            free(datos_host);
        return false;
    }

    // Escribir encabezado con información del tensor
    fprintf(file, "# Tensor 3D guardado en formato compatible\n");
    fprintf(file, "# Dimensiones: batch=%d, M=%d, N=%d (K=1)\n",
            tensor.batch, tensor.M, tensor.N);
    fprintf(file, "# Total elementos: %zu\n", tensor.total_elements());
    fprintf(file, "# Ubicación original: %s\n", tensor.is_device_ptr ? "GPU (device)" : "CPU (host)");
    fprintf(file, "# Formato: valores separados por espacios, organizados por batch, M, N\n");
    fprintf(file, "#\n");

    // Escribir todos los datos en líneas, agrupando por batches
    for (int b = 0; b < tensor.batch; b++)
    {
        fprintf(file, "# Batch %d\n", b);

        for (int m = 0; m < tensor.M; m++)
        {
            for (int n = 0; n < tensor.N; n++)
            {
                // Calcular índice simplificado para K=1: b*(M*N) + m*(N) + n
                size_t idx = b * (tensor.M * tensor.N) + m * tensor.N + n;

                fprintf(file, "%.6f ", datos_host[idx]);
            }
            fprintf(file, "\n"); // Nueva línea al final de cada fila M
        }

        fprintf(file, "\n"); // Línea extra entre batches
    }

    fclose(file);

    printf("Tensor 3D guardado exitosamente: %s\n", nombre_archivo);
    printf("  Dimensiones: batch=%d, M=%d, N=%d (K=1)\n",
           tensor.batch, tensor.M, tensor.N);
    printf("  Total elementos: %zu\n", tensor.total_elements());
    printf("  Ubicación original: %s\n", tensor.is_device_ptr ? "GPU" : "CPU");

    // Liberar memoria temporal si es necesario
    if (necesita_liberar)
    {
        free(datos_host);
    }

    return true;
}
