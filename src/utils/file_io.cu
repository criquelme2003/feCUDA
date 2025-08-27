#include "utils/file_io.cuh"
#include "utils/logging.cuh"
#include "core/tensor.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>

namespace FileIO
{

    bool leer_matriz_3d_desde_archivo(const char *archivo, TensorResult &tensor,
                                      int batch, int M, int N, int K)
    {
        FILE *file = fopen(archivo, "r");
        if (file == nullptr)
        {
            LOG_ERROR("No se pudo abrir el archivo: ", archivo);
            return false;
        }

        // Calcular número total de elementos
        const size_t total_elementos = static_cast<size_t>(batch) * M * N * K;
        const size_t size_bytes = total_elementos * sizeof(float);

        // Alocar memoria para los datos usando C++
        float *datos = static_cast<float *>(std::malloc(size_bytes));
        if (datos == nullptr)
        {
            LOG_ERROR("No se pudo asignar memoria para ", total_elementos, " elementos");
            fclose(file);
            return false;
        }

        LOG_INFO("Leyendo matriz 3D desde: ", archivo);
        LOG_DEBUG("Dimensiones esperadas: batch=", batch, ", M=", M, ", N=", N, ", K=", K);

        // Leer elementos del archivo
        size_t elementos_leidos = 0;
        float valor;

        while (fscanf(file, "%f", &valor) == 1 && elementos_leidos < total_elementos)
        {
            datos[elementos_leidos] = valor;
            elementos_leidos++;
        }

        fclose(file);

        if (elementos_leidos != total_elementos)
        {
            LOG_WARNING("Se leyeron ", elementos_leidos, " elementos, se esperaban ", total_elementos);
            // No retornar false, ajustar dimensiones si es necesario
        }

        // Crear TensorResult
        tensor = TensorResult(datos, false, batch, M, N, K, true);

        LOG_INFO("Matriz 3D leída exitosamente: ", elementos_leidos, " elementos");
        return true;
    }

    bool guardar_tensor_como_archivo(const TensorResult &tensor, const char *nombre_archivo)
    {
        if (!tensor.data)
        {
            LOG_ERROR("Tensor no tiene datos válidos");
            return false;
        }

        // Si el tensor está en device, copiarlo a host primero
        TensorResult host_tensor = tensor.is_device_ptr ? TensorUtils::copy_to_host(tensor) : tensor;

        std::ofstream archivo(nombre_archivo);
        if (!archivo.is_open())
        {
            LOG_ERROR("No se pudo crear el archivo: ", nombre_archivo);
            return false;
        }

        const size_t total_elementos = host_tensor.total_elements();

        for (size_t i = 0; i < total_elementos; ++i)
        {
            archivo << host_tensor.data[i];
            if (i < total_elementos - 1)
            {
                archivo << " ";
            }

            // Salto de línea cada N elementos para legibilidad
            if ((i + 1) % host_tensor.N == 0)
            {
                archivo << "\n";
            }
        }

        archivo.close();

        LOG_INFO("Tensor guardado en: ", nombre_archivo, " (", total_elementos, " elementos)");
        return true;
    }

    bool load_dataset(const char *dataset_name, TensorResult &tensor)
    {
        std::string base_path = "datasets_txt/";
        std::string full_path = base_path + dataset_name;

        // Intentar diferentes configuraciones comunes
        const std::vector<std::tuple<int, int, int, int>> configs = {
            {1, 6, 6, 1},    // Reflexive típico
            {10, 16, 16, 1}, // CC, EE típico
            {1, 4, 4, 1},    // Pequeño
            {1, 8, 8, 1}     // Mediano
        };

        for (const auto &[batch, M, N, K] : configs)
        {
            if (leer_matriz_3d_desde_archivo(full_path.c_str(), tensor, batch, M, N, K))
            {
                LOG_INFO("Dataset cargado con configuración [", batch, ",", M, ",", N, ",", K, "]");
                return true;
            }
        }

        LOG_ERROR("No se pudo cargar el dataset con ninguna configuración: ", dataset_name);
        return false;
    }

    bool save_results(const TensorResult &tensor, const char *output_file, const char *format)
    {
        std::string format_str(format);

        if (format_str == "txt")
        {
            return guardar_tensor_como_archivo(tensor, output_file);
        }
        else if (format_str == "csv")
        {
            // Implementar formato CSV si es necesario
            LOG_WARNING("Formato CSV no implementado aún, usando TXT");
            return guardar_tensor_como_archivo(tensor, output_file);
        }
        else
        {
            LOG_ERROR("Formato no soportado: ", format);
            return false;
        }
    }

    bool validate_file_format(const char *filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            return false;
        }

        // Verificar que el archivo contiene números válidos
        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);

        float test_value;
        return (iss >> test_value) ? true : false; // Si puede leer al menos un float, es válido
    }

    bool load_config(const char *config_file)
    {
        LOG_INFO("Cargando configuración desde: ", config_file);

        std::ifstream file(config_file);
        if (!file.is_open())
        {
            LOG_WARNING("No se pudo abrir archivo de configuración, usando valores por defecto");
            return false;
        }

        // Implementación básica de configuración
        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#')
                continue; // Comentarios

            std::istringstream iss(line);
            std::string key, value;

            if (std::getline(iss, key, '=') && std::getline(iss, value))
            {
                // Procesar configuraciones específicas
                LOG_DEBUG("Config: ", key, " = ", value);
                // Aquí se pueden procesar diferentes opciones de configuración
            }
        }

        return true;
    }
}
