#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>

#include <core/types.cuh>
#include <utils.cuh>
#include <utils/graph_generator.cuh>

namespace
{
    std::string to_lower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return s;
    }
}

GraphRegime parse_graph_regime(const std::string &name)
{
    const std::string lower = to_lower(name);
    if (lower == "sparse")
        return GraphRegime::Sparse;
    if (lower == "supercritical" || lower == "supercritico" || lower == "supercrítico")
        return GraphRegime::Supercritical;
    if (lower == "dense" || lower == "denso")
        return GraphRegime::Dense;
    throw std::invalid_argument("Régimen de grafo desconocido: " + name);
}

GraphGenerationStats generate_graph_tensor(const GraphGeneratorConfig &config, TensorResult &out_tensor)
{
    if (config.M <= 0 || config.N <= 0 || config.batch <= 0)
    {
        throw std::runtime_error("Dimensiones inválidas en GraphGeneratorConfig");
    }
    if (config.M != config.N)
    {
        throw std::runtime_error("El generador solo admite grafos cuadrados (M debe ser igual a N)");
    }
    if (config.epsilon <= 0.5 || config.epsilon > 1.0)
    {
        throw std::runtime_error("epsilon debe ser > 0.5 y <= 1.0");
    }

    const int N = config.N;
    const int batch = config.batch;

    double p = 0.0;
    switch (config.regime)
    {
    case GraphRegime::Sparse:
        p = (N > 1) ? (config.avg_degree / static_cast<double>(N - 1)) : 0.0;
        p = std::clamp(p, 0.0, 1.0);
        break;
    case GraphRegime::Supercritical:
    {
        double deg = config.avg_degree;
        deg = std::max(1.0, deg);
        deg = std::min(config.max_degree, deg);
        p = (N > 1) ? (deg / static_cast<double>(N - 1)) : 0.0;
        p = std::clamp(p, 0.0, 1.0);
        break;
    }
    case GraphRegime::Dense:
    {
        if (config.avg_degree > 0.0)
        {
            p = (N > 1) ? (config.avg_degree / static_cast<double>(N - 1)) : 0.0;
        }
        else
        {
            p = config.dense_p;
        }
        // En denso, forzamos una densidad mínima suave para evitar un grafo casi vacío.
        p = std::clamp(p, 0.05, 1.0);
        break;
    }
    default:
        throw std::runtime_error("Regimen de grafo no soportado");
    }

    const size_t total_elements = static_cast<size_t>(batch) * N * N;
    float *data = static_cast<float *>(std::malloc(total_elements * sizeof(float)));
    if (!data)
    {
        throw std::runtime_error("No se pudo asignar memoria para el grafo sintético");
    }
    std::fill(data, data + total_elements, 0.0f);

    const unsigned int seed = config.seed_provided ? config.seed : static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    long long edge_count = 0;
    for (int b = 0; b < batch; ++b)
    {
        const size_t base = static_cast<size_t>(b) * N * N;
        for (int i = 0; i < N; ++i)
        {
            data[base + static_cast<size_t>(i) * N + i] = 0.0f; // Sin loops
            for (int j = i + 1; j < N; ++j)
            {
                const float r = uniform01(rng);
                if (r < p)
                {
                    const float w = config.epsilon + (1.0f - static_cast<float>(config.epsilon)) * uniform01(rng);
                    data[base + static_cast<size_t>(i) * N + j] = w;
                    data[base + static_cast<size_t>(j) * N + i] = w; // simétrico
                    ++edge_count;
                }
            }
        }
    }

    const double possible_edges = static_cast<double>(batch) * (static_cast<double>(N) * (N - 1) / 2.0);
    GraphGenerationStats stats;
    stats.target_p = p;
    stats.empirical_p = (possible_edges > 0.0) ? (static_cast<double>(edge_count) / possible_edges) : 0.0;
    stats.empirical_avg_degree = (batch * N > 0) ? (2.0 * static_cast<double>(edge_count) / (static_cast<double>(batch) * N)) : 0.0;

    out_tensor = TensorResult(data, false, batch, N, N, 1, true);
    return stats;
}
