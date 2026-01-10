#include <cuda_runtime.h>
#include <float.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <core/types.cuh>
#include <utils.cuh>


__global__ void find_path_matches_kernel(float *previous_paths, float *result_tensor,
                                         float *result_values, float *output_paths,
                                         float *output_values, int *match_count,
                                         int num_prev_paths, int num_current_tensor,
                                         int prev_cols, int current_cols, int iteration)
{
    int prev_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (prev_idx < num_prev_paths && curr_idx < num_current_tensor)
    {
        // Extraer coordenadas del camino previo
        // previous_paths formato: [batch, start_fila, intermedio1, intermedio2, ..., end_columna]
        int p_batch = (int)previous_paths[prev_idx * prev_cols];                      // batch
        int p_fila = (int)previous_paths[prev_idx * prev_cols + 1];                   // fila inicial
        int p_intermedio = (int)previous_paths[prev_idx * prev_cols + iteration + 2]; // intermedio en posición i+1

        // Extraer coordenadas del resultado actual [batch, fila, intermedio, columna]
        int c_batch = (int)result_tensor[curr_idx * current_cols];          // batch
        int c_fila = (int)result_tensor[curr_idx * current_cols + 1];       // fila
        int c_intermedio = (int)result_tensor[curr_idx * current_cols + 2]; // intermedio
        int c_columna = (int)result_tensor[curr_idx * current_cols + 3];    // nueva columna

        // Condición de match: batch, fila e intermedio deben coincidir
        if (p_batch == c_batch && p_fila == c_fila && p_intermedio == c_intermedio)
        {
            // Found a match - usar atomic add para obtener posición de salida
            int output_idx = atomicAdd(match_count, 1);

            // El nuevo camino tendrá prev_cols + 1 columnas
            int new_cols = prev_cols + 1;
            int output_base = output_idx * new_cols;

            // Copiar todas las columnas del camino previo
            for (int col = 0; col < prev_cols; col++)
            {
                output_paths[output_base + col] = previous_paths[prev_idx * prev_cols + col];
            }

            // Agregar la nueva columna (destino del resultado actual)
            output_paths[output_base + prev_cols] = (float)c_columna;

            // Guardar el valor correspondiente
            output_values[output_idx] = result_values[curr_idx];
        }
    }
}

void armar_caminos_original(const TensorResult<> &previous_paths, const TensorResult<> &result_tensor,
                            const TensorResult<> &result_values, TensorResult<> &paths,
                            TensorResult<> &matched_values, int iteration, bool keep_in_device)
{
    // FUNCIÓN NO COMPATIBLE CON NUEVA VERSIÓN DE TensorResult
    //
    // Problemas encontrados:
    // 1. Accede a campos privados: .data, .M, .N, .K, .batch, .is_device_ptr, .owns_memory
    // 2. Intenta asignar directamente estos campos en los tensores de salida
    // 3. Nueva TensorResult no permite esta asignación directa
    // 4. Depende de constructor antiguo de TensorResult
    //
    // Para reimplementar se requeriría:
    // 1. Usar nuevos métodos públicos: getData(), getBatch(), getM(), getN(), getK()
    // 2. Usar constructores con MemorySpace (Device/Host)
    // 3. Copiar datos después de crear el objeto (no asignación directa)
    // 4. Rediseñar completamente la construcción de tensores de salida
    
    fprintf(stderr, "ERROR: armar_caminos_original() no es compatible con la nueva versión de TensorResult\n");
    fprintf(stderr, "La función requiere rediseño para usar los nuevos métodos públicos\n");
    
    // Nota: TensorResult tiene operator= eliminado, por lo que no podemos asignar
    // Los parámetros de salida permanecen sin modificar
    return;
}

// SOLUCIÓN 1: Procesamiento por lotes (batches)
void armar_caminos_batch(const TensorResult<> &previous_paths, const TensorResult<> &result_tensor,
                         const TensorResult<> &result_values, TensorResult<> &paths,
                         TensorResult<> &matched_values, int iteration, int batch_size, bool keep_in_device)
{
    // FUNCIÓN NO COMPATIBLE CON NUEVA VERSIÓN DE TensorResult
    //
    // Problemas encontrados:
    // 1. Intenta crear "sub-tensores" accediendo a datos privados
    // 2. Asigna directamente campos privados: .data, .M, .N, .K, .batch, .is_device_ptr, .owns_memory
    // 3. Nueva TensorResult no permite crear vistas o asignar campos
    // 4. Depende de armar_caminos_original() que también está deshabilitada
    // 5. Llama a transfer_ownership() que no existe
    //
    // Para reimplementar se requeriría:
    // 1. Rediseñar completamente la estrategia de batching
    // 2. Usar constructores de TensorResult con MemorySpace
    // 3. No crear vistas, sino copiar datos cuando sea necesario
    // 4. Implementar nuevos métodos helper de TensorResult
    
    fprintf(stderr, "ERROR: armar_caminos_batch() no es compatible con la nueva versión de TensorResult\n");
    fprintf(stderr, "La función requiere rediseño completo para la gestión de memoria\n");
    
    // Nota: TensorResult tiene operator= eliminado, por lo que no podemos asignar
    // Los parámetros de salida permanecen sin modificar
    return;
}
