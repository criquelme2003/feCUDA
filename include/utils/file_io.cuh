#ifndef FILE_IO_CUH
#define FILE_IO_CUH

#include "core/types.cuh"

// Funciones de I/O para tensores
namespace FileIO
{

    // Función para leer matrices 3D desde archivo de texto
    bool leer_matriz_3d_desde_archivo(const char *archivo, TensorResult &tensor,
                                      int batch, int M, int N, int K = 1);

    // Función para guardar tensor como archivo en formato compatible
    bool guardar_tensor_como_archivo(const TensorResult &tensor, const char *nombre_archivo);

    // Cargar datos de datasets específicos
    bool load_dataset(const char *dataset_name, TensorResult &tensor);

    // Guardar resultados en formato específico
    bool save_results(const TensorResult &tensor, const char *output_file,
                      const char *format = "txt");

    // Validar formato de archivo
    bool validate_file_format(const char *filename);

    // Leer configuración desde archivo
    bool load_config(const char *config_file);
}

#endif // FILE_IO_CUH
