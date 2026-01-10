

#include "../../include/core/types.cuh"
#include <cstdio>

template<typename T>
TensorResult<T>  batched_tensor(TensorResult<T> t, int batch_size, int iteration){
    // INVIABLE: TensorResult no permite acceso a datos internos para crear subvistas
    // La nueva implementación no soporta crear "vistas" de un tensor existente
    // Se necesitaría rediseñar completamente o asignar nueva memoria
    fprintf(stderr, "ERROR: batched_tensor no es compatible con la nueva versión de TensorResult\n");
    fprintf(stderr, "Se requiere rediseño para crear subvistas de tensores\n");
    
    // Retornar tensor vacío como fallback
    return TensorResult<T>();
}

template TensorResult<float> batched_tensor(TensorResult<float> t, int batch_size, int iteration);

template TensorResult<__half> batched_tensor(TensorResult<__half> t, int batch_size, int iteration);

