#include <cuda_runtime.h>
#include <float.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <core/types.cuh>
#include <utils.cuh>

__global__ void strainer(float *min_res,
                         float *maxmin_prima,
                         float *values,
                         float *indices,
                         float threshold,
                         int batch,
                         int M, int N, int K,
                         int *output_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_3d = batch * M * N; // Total elementos en maxmin_prima (3D)

    if (idx < total_elements_3d)
    {
        // Convertir índice lineal a coordenadas (b, m, n) para maxmin_prima
        int b = idx / (M * N);
        int m = (idx % (M * N)) / N;
        int n = (idx % (M * N)) % N;

        // Obtener valor de maxmin_prima en posición (b, m, n)
        float maxmin_value = maxmin_prima[idx];

        // Verificar si supera el threshold
        if (maxmin_value > threshold)
        {
            // Buscar en min_res en la misma posición (b, m, n) pero en todas las K
            // min_res tiene dimensiones [batch, M, N, K]

            int base_idx = b * (M * N * K) + m * (N * K) + n * K;
            // Buscar maximo en min_res
            float max_val = -FLT_MAX;
            for (int k = 0; k < K; k++)
            {
                float current_val = min_res[base_idx + k];
                if (current_val > max_val)
                {
                    max_val = current_val;
                }
            }

            // Inicializar base_idx para acceder a min_res
            // Contar cuántas veces se repite este máximo y guardar índices
            for (int k = 0; k < K; k++)
            {
                float current_val = min_res[base_idx + k];
                const float EPSILON = 1e-6f;
                if (fabsf(current_val - max_val) < EPSILON)
                {
                    // Obtener posición en el array de salida usando atomic add
                    int output_pos = atomicAdd(output_count, 1);
                    
                    // Guardar índices 4D [b, m, k, n] en el array indices
                    // Cada elemento ocupa 4 posiciones en el array indices
                    indices[output_pos * 4 + 0] = (float)b; // batch
                    indices[output_pos * 4 + 1] = (float)m; // M
                    indices[output_pos * 4 + 2] = (float)k; // K (donde está el máximo)
                    indices[output_pos * 4 + 3] = (float)n; // N

                    // Guardar el valor máximo en values
                    values[output_pos] = maxmin_value;
                }
            }
        }
    }
}

// FUNCIÓN NO COMPATIBLE CON NUEVA VERSIÓN DE TensorResult
// 
// La función indices() original accede a campos privados de TensorResult:
// - .batch, .M, .N, .K (miembros privados en nueva versión)
// - .data (miembro privado en nueva versión)
// - .is_device_ptr (no existe en nueva versión)
//
// Además, intenta construir TensorResult con constructores obsoletos
// Necesita rediseño completo para usar:
// - Métodos getBatch(), getM(), getN(), getK()
// - Método getData() para acceso a datos
// - Constructor nuevo con MemorySpace y dimensiones
//
// Por el momento, esta función está deshabilitada
void indices(const TensorResult<> &min_result, const TensorResult<> &maxmin_prima,
             TensorResult<> &result_tensor_filtered, TensorResult<> &result_tensor_values,
             float threshold, bool keep_in_device){
    
    fprintf(stderr, "ERROR: indices() no es compatible con la nueva versión de TensorResult\n");
    fprintf(stderr, "La función requiere rediseño completo para usar la nueva API\n");
    
    // Nota: TensorResult tiene operator= eliminado, por lo que no podemos asignar
    // Los parámetros de salida permanecen sin modificar
    
    return;
}
