#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <utils.cuh>
#include <utils/eta_metrics.cuh>

EtaStats compute_eta_stats(const std::vector<TensorResult> &paths)
{
    std::vector<double> lengths;
    lengths.reserve(1024);

    for (const auto &tensor : paths)
    {
        if (tensor.data == nullptr || tensor.M <= 0 || tensor.N <= 1)
        {
            continue;
        }

        // Aseguramos datos en CPU para recorrerlos con seguridad
        TensorResult host_tensor = tensor.is_device_ptr ? copy_tensor_to_cpu(tensor) : copy_tensor(tensor);
        const int rows = host_tensor.M;
        const int cols = host_tensor.N;
        const int path_length = std::max(cols - 2, 0); // Aproximación descrita en el header

        for (int r = 0; r < rows; ++r)
        {
            lengths.push_back(static_cast<double>(path_length));
        }

        safe_tensor_cleanup(host_tensor);
    }

    EtaStats stats;
    stats.samples = lengths.size();
    if (lengths.empty())
    {
        return stats;
    }

    const double sum = std::accumulate(lengths.begin(), lengths.end(), 0.0);
    stats.eta0_mean = sum / static_cast<double>(lengths.size());
    stats.eta0_max = *std::max_element(lengths.begin(), lengths.end());

    double var_accum = 0.0;
    for (double v : lengths)
    {
        const double diff = v - stats.eta0_mean;
        var_accum += diff * diff;
    }
    stats.eta0_std = std::sqrt(var_accum / static_cast<double>(lengths.size()));

    // p95
    std::vector<double> tmp = lengths;
    const size_t idx = static_cast<size_t>(std::floor(0.95 * (tmp.size() - 1)));
    std::nth_element(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(idx), tmp.end());
    stats.eta0_p95 = tmp[idx];

    return stats;
}
