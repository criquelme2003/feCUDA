#include "../../include/utils.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
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
    // file << B << " " << M << " " << N << " " << K << std::endl;

    // Datos: todos los números uno tras otro
    int total = B * M * N * K;
    for (int i = 0; i < total; i++)
    {
        file << tensor[i];
        if (i < total - 1)
        {
            file << ",";
        }
    }
    file << std::endl;

    file.close();
    std::cout << "Tensor guardado en " << name << std::endl;
    std::cout << "Dimensiones: " << B << "x" << M << "x" << N << "x" << K << std::endl;
}

// Función para guardar vector de tensores
void save_tensor_vector(const std::vector<TensorResult> &tensors, const std::string &filename, bool save_info)
{
    std::ofstream file(filename, std::ofstream::out | std::ofstream::trunc);

    if (!file.is_open())
    {
        std::cerr << "Error: No se pudo abrir " << filename << std::endl;
        return;
    }

    // Header: número de tensores
    if (save_info)
        file << tensors.size() << std::endl;

    // Para cada tensor, guardar sus dimensiones y datos
    for (size_t tensor_idx = 0; tensor_idx < tensors.size(); tensor_idx++)
    {
        const TensorResult &tensor = tensors[tensor_idx];

        // Guardar dimensiones del tensor
        if (save_info)
            file << tensor.batch << " " << tensor.M << " " << tensor.N << " " << tensor.K << std::endl;

        // Guardar todos los datos del tensor
        int total = tensor.total_elements();
        for (int i = 0; i < total; i++)
        {
            file << (float)tensor.data[i];
            file << ",";
        }
        if (save_info)
            file << std::endl;
    }

    file.close();

    std::cout << "Vector de tensores guardado en " << filename << std::endl;
}

void save_paths_with_values(const std::vector<TensorResult> &paths,
                            const std::vector<TensorResult> &values,
                            const std::string &filename)
{
    std::ofstream file(filename, std::ofstream::out | std::ofstream::trunc);
    if (!file.is_open())
    {
        std::cerr << "Error: No se pudo abrir " << filename << std::endl;
        return;
    }

    if (paths.size() != values.size())
    {
        std::cerr << "Error: paths y values tienen tamaños distintos (" << paths.size()
                  << " vs " << values.size() << ")\n";
        return;
    }

    const auto original_flags = file.flags();
    const auto original_precision = file.precision();

    file << "[\n";
    bool first_entry = true;

    for (size_t step = 0; step < paths.size(); ++step)
    {
        const TensorResult &path_tensor = paths[step];
        const TensorResult &value_tensor = values[step];

        if (!path_tensor.data || !value_tensor.data)
        {
            continue;
        }

        if (path_tensor.is_device_ptr || value_tensor.is_device_ptr)
        {
            std::cerr << "Error: save_paths_with_values espera tensores en CPU.\n";
            return;
        }

        const int num_paths = path_tensor.M;
        const int path_cols = path_tensor.N;
        const int num_values = value_tensor.M > 1 ? value_tensor.M * value_tensor.N : value_tensor.N;

        if (num_paths != num_values)
        {
            std::cerr << "Advertencia: número de caminos (" << num_paths
                      << ") no coincide con número de valores (" << num_values
                      << ") para el paso " << step << ". Se usará el mínimo.\n";
        }

        const int paired = std::min(num_paths, num_values);
        auto read_value = [&](int logical_index) -> float {
            if (value_tensor.M == 1)
            {
                // Vector fila
                return value_tensor.data[logical_index];
            }
            const int cols = value_tensor.N > 0 ? value_tensor.N : 1;
            const int row_idx = logical_index / cols;
            const int col_idx = logical_index % cols;
            const int idx = row_idx * cols + col_idx;
            return value_tensor.data[idx];
        };

        for (int row = 0; row < paired; ++row)
        {
            if (!first_entry)
            {
                file << ",\n";
            }
            first_entry = false;

            file << "  {\"order\":" << step << ",\"row\":" << row << ",\"path\":[";
            for (int col = 0; col < path_cols; ++col)
            {
                const int idx = row * path_cols + col;
                int coord = static_cast<int>(std::llround(path_tensor.data[idx]));
                file << coord;
                if (col + 1 < path_cols)
                {
                    file << ",";
                }
            }
            file << "],\"value\":";
            file.setf(std::ios::fixed);
            file << std::setprecision(8)
                 << static_cast<double>(read_value(row))
                 << "}";
            file.flags(original_flags);
            file.precision(original_precision);
        }
    }

    if (!first_entry)
    {
        file << "\n";
    }
    file << "]\n";
    file.flags(original_flags);
    file.precision(original_precision);

    std::cout << "Caminos y valores guardados en " << filename << std::endl;
}
