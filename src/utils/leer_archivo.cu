#include "../../include/utils.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdexcept>

// Include the string library
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

// Función para guardar tensor 4D
void save_tensor_4d_as_file(float *tensor, int B, int M, int N, int K,
                            const std::string &name)
{
    std::ofstream file(name);

    if (!file.is_open())
    {
        std::cerr << "Error: No se pudo abrir " << name << std::endl;
        return;
    }

    // Header: dimensiones
    file << B << " " << M << " " << N << " " << K << std::endl;

    // Datos: todos los números uno tras otro
    int total = B * M * N * K;
    for (int i = 0; i < total; i++)
    {
        file << tensor[i];
        if (i < total - 1)
        {
            file << " ";
        }
    }
    file << std::endl;

    file.close();
    std::cout << "Tensor guardado en " << name << std::endl;
    std::cout << "Dimensiones: " << B << "x" << M << "x" << N << "x" << K << std::endl;
}

// Función para guardar vector de tensores
void save_tensor_vector(const std::vector<TensorResult> &tensors, const std::string &filename)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: No se pudo abrir " << filename << std::endl;
        return;
    }

    // Header: número de tensores
    file << tensors.size() << std::endl;

    // Para cada tensor, guardar sus dimensiones y datos
    for (size_t tensor_idx = 0; tensor_idx < tensors.size(); tensor_idx++)
    {
        const TensorResult &tensor = tensors[tensor_idx];

        // Guardar dimensiones del tensor
        file << tensor.batch << " " << tensor.M << " " << tensor.N << " " << tensor.K << std::endl;

        // Guardar todos los datos del tensor
        int total = tensor.total_elements();
        for (int i = 0; i < total; i++)
        {
            file << (float)tensor.data[i];
            if (i < total - 1)
            {
                file << " ";
            }
        }
        file << std::endl;
    }

    file.close();

    std::cout << "Vector de tensores guardado en " << filename << std::endl;
}