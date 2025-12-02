#ifndef GRAPH_GENERATOR_CUH
#define GRAPH_GENERATOR_CUH

#include <string>
#include <vector>
#include <core/types.cuh>

// Regímenes de conectividad que queremos simular.
enum class GraphRegime
{
    Sparse,
    Supercritical,
    Dense
};

struct GraphGeneratorConfig
{
    int batch = 1;
    int M = 0;
    int N = 0;
    GraphRegime regime = GraphRegime::Sparse;
    double avg_degree = 0.5;    // En sparse/supercritical: grado medio objetivo (>=0).
    double max_degree = 4.0;    // Límite superior para supercrítico.
    double dense_p = 0.2;       // Probabilidad directa para denso (si avg_degree no aplica).
    double epsilon = 0.6;       // Peso mínimo para aristas presentes, debe ser > 0.5.
    unsigned int seed = 0;      // Si seed es 0, se usará std::random_device.
    bool seed_provided = false; // True si seed fue configurada explícitamente.
};

struct GraphGenerationStats
{
    double target_p = 0.0;
    double empirical_p = 0.0;
    double empirical_avg_degree = 0.0;
};

// Genera un tensor [batch, M, N] cuadrado (M debe ser igual a N) que representa
// la matriz de adyacencia ponderada de un grafo. Los pesos de arista se
// distribuyen uniformemente en [epsilon, 1] y las ausencias son 0.
// Lanza std::runtime_error ante configuraciones inválidas.
GraphGenerationStats generate_graph_tensor(const GraphGeneratorConfig &config, TensorResult &out_tensor);

// Utilidad para parsear un string de régimen.
GraphRegime parse_graph_regime(const std::string &name);

#endif // GRAPH_GENERATOR_CUH
