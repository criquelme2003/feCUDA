#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

#include <algorithms/mst_paths.cuh>
#include <utils.cuh>

namespace
{
    struct Edge
    {
        int u;
        int v;
        float w;
    };

    struct DSU
    {
        std::vector<int> parent;
        std::vector<int> rank;
        explicit DSU(int n) : parent(n), rank(n, 0)
        {
            for (int i = 0; i < n; ++i)
                parent[i] = i;
        }

        int find(int x)
        {
            if (parent[x] != x)
                parent[x] = find(parent[x]);
            return parent[x];
        }

        bool unite(int a, int b)
        {
            a = find(a);
            b = find(b);
            if (a == b)
                return false;
            if (rank[a] < rank[b])
                std::swap(a, b);
            parent[b] = a;
            if (rank[a] == rank[b])
                rank[a]++;
            return true;
        }
    };

    TensorResult ensure_host(const TensorResult &tensor)
    {
        if (tensor.is_device_ptr)
        {
            return copy_tensor_to_cpu(tensor);
        }
        return copy_tensor(tensor);
    }
}

bool compute_eta_stats_mst(const TensorResult &tensor, EtaStats &stats)
{
    stats = EtaStats();
    if (tensor.batch != 1)
    {
        printf("MST: solo se admite batch=1 (batch=%d)\n", tensor.batch);
        return false;
    }
    if (tensor.M != tensor.N)
    {
        printf("MST: se requiere tensor cuadrado (M=%d, N=%d)\n", tensor.M, tensor.N);
        return false;
    }

    const int N = tensor.M;
    if (N <= 1)
    {
        return true;
    }

    TensorResult host_tensor = ensure_host(tensor);
    const float *data = host_tensor.data;
    if (!data)
    {
        printf("MST: tensor vacío\n");
        safe_tensor_cleanup(host_tensor);
        return false;
    }

    std::vector<Edge> edges;
    edges.reserve(static_cast<size_t>(N) * (N - 1) / 2);
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            const float w1 = data[i * N + j];
            const float w2 = data[j * N + i];
            const float w = std::max(w1, w2);
            if (w > 0.0f)
            {
                edges.push_back({i, j, w});
            }
        }
    }

    if (edges.empty())
    {
        printf("MST: grafo sin aristas válidas\n");
        safe_tensor_cleanup(host_tensor);
        return false;
    }

    std::sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b)
              { return a.w > b.w; });

    DSU dsu(N);
    std::vector<std::vector<int>> adj(N);
    int added = 0;
    for (const auto &e : edges)
    {
        if (dsu.unite(e.u, e.v))
        {
            adj[e.u].push_back(e.v);
            adj[e.v].push_back(e.u);
            ++added;
            if (added == N - 1)
                break;
        }
    }

    if (added == 0)
    {
        printf("MST: no se pudo construir árbol (grafo desconectado)\n");
        safe_tensor_cleanup(host_tensor);
        return false;
    }

    std::vector<double> lengths;
    lengths.reserve(static_cast<size_t>(N) * (N - 1) / 2);
    std::vector<int> dist(N, -1);
    double max_len = 0.0;

    std::queue<int> q;
    for (int start = 0; start < N; ++start)
    {
        std::fill(dist.begin(), dist.end(), -1);
        while (!q.empty())
            q.pop();

        dist[start] = 0;
        q.push(start);
        while (!q.empty())
        {
            const int u = q.front();
            q.pop();
            for (int v : adj[u])
            {
                if (dist[v] == -1)
                {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }

        for (int v = start + 1; v < N; ++v)
        {
            if (dist[v] >= 0)
            {
                lengths.push_back(static_cast<double>(dist[v]));
                if (dist[v] > max_len)
                {
                    max_len = static_cast<double>(dist[v]);
                }
            }
        }
    }

    if (!lengths.empty())
    {
        const double sum = std::accumulate(lengths.begin(), lengths.end(), 0.0);
        stats.eta0_mean = sum / static_cast<double>(lengths.size());
        double var_accum = 0.0;
        for (double v : lengths)
        {
            const double diff = v - stats.eta0_mean;
            var_accum += diff * diff;
        }
        stats.eta0_std = std::sqrt(var_accum / static_cast<double>(lengths.size()));
        stats.eta0_max = max_len;
        std::vector<double> tmp = lengths;
        const size_t idx = static_cast<size_t>(std::floor(0.95 * (tmp.size() - 1)));
        std::nth_element(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(idx), tmp.end());
        stats.eta0_p95 = tmp[idx];
        stats.samples = lengths.size();
    }
    else
    {
        stats.eta0_max = 0.0;
        stats.eta0_mean = 0.0;
        stats.eta0_std = 0.0;
        stats.eta0_p95 = 0.0;
        stats.samples = 0;
    }

    safe_tensor_cleanup(host_tensor);
    return true;
}
