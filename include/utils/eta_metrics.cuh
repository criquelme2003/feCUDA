#ifndef ETA_METRICS_CUH
#define ETA_METRICS_CUH

#include <vector>
#include <core/types.cuh>

struct EtaStats
{
    double eta0_max = 0.0;
    double eta0_mean = 0.0;
    double eta0_std = 0.0;
    double eta0_p95 = 0.0;
    size_t samples = 0;
};

// Calcula métricas básicas de longitudes de camino a partir de los tensores
// generados por iterative_maxmin_cuadrado/armar_caminos.
// Se asume que cada fila de los tensores de paths almacena:
// [batch_id, nodo_0, nodo_1, ..., nodo_k], por lo que la longitud en aristas
// se aproxima como (num_columnas - 2).
EtaStats compute_eta_stats(const std::vector<TensorResult> &paths);

#endif // ETA_METRICS_CUH
