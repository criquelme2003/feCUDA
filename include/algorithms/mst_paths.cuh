#ifndef MST_PATHS_CUH
#define MST_PATHS_CUH

#include <core/types.cuh>
#include <utils/eta_metrics.cuh>

// Calcula métricas eta0 usando un MST máximo como aproximación de caminos
// max-min. Retorna true si se pudo construir correctamente.
bool compute_eta_stats_mst(const TensorResult &tensor, EtaStats &stats);

#endif // MST_PATHS_CUH
