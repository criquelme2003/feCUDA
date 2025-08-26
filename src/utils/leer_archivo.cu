#include "../../include/utils.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
