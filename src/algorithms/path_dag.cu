#include "../../include/algorithms/path_dag.cuh"
#include "../../include/utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <queue>
#include <set>

// ─────────────────────────────────────────────────────────────────────────────
// Lectura de manifest.json — el formato es fijo, generado por
// dump_order_level() en maxminv2_count.cu (ver ese archivo para el escritor).
// ─────────────────────────────────────────────────────────────────────────────

PathManifest load_path_manifest(const std::string &dump_dir) {
    PathManifest manifest;
    manifest.dump_dir = dump_dir;

    std::string manifest_path = dump_dir + "/manifest.json";
    std::ifstream f(manifest_path);
    if (!f.is_open()) {
        std::cerr << "ERROR: no se pudo abrir " << manifest_path << std::endl;
        return manifest;
    }

    nlohmann::json j;
    f >> j;

    if (!j.contains("orders")) {
        std::cerr << "ERROR: manifest.json sin campo \"orders\": " << manifest_path << std::endl;
        return manifest;
    }

    for (const auto &jo : j["orders"]) {
        OrderManifest om;
        om.s = jo.at("s").get<int>();
        om.M = jo.at("M").get<int>();
        om.N = jo.at("N").get<int>();
        om.K = jo.at("K").get<int>();
        om.Mpad = jo.at("Mpad").get<int>();
        om.Npad = jo.at("Npad").get<int>();
        om.padded_elems = jo.at("padded_elems").get<size_t>();
        om.h_counter = jo.at("h_counter").get<int>();
        om.thr = jo.at("thr").get<float>();
        om.file_C = jo.at("files").at("C").get<std::string>();
        om.file_row_offsets = jo.at("files").at("row_offsets").get<std::string>();
        om.file_csr_pivots = jo.at("files").at("csr_pivots").get<std::string>();
        manifest.orders.push_back(om);
    }

    return manifest;
}

Level load_level(const std::string &dump_dir, const OrderManifest &om) {
    Level lvl;
    lvl.Npad = om.Npad;
    lvl.padded_elems = om.padded_elems;
    lvl.h_counter = om.h_counter;
    lvl.row_offsets.resize(om.padded_elems);
    lvl.csr_pivots.resize(om.h_counter);
    lvl.C.resize(om.padded_elems);

    auto read_bin = [&](const std::string &filename, void *dst, size_t count, size_t elem_size) {
        std::string path = dump_dir + "/" + filename;
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) {
            std::cerr << "ERROR: no se pudo abrir " << path << std::endl;
            return;
        }
        size_t read = fread(dst, elem_size, count, f);
        if (read != count) {
            std::cerr << "ERROR: lectura incompleta de " << path << " (" << read << "/" << count
                       << " elementos)" << std::endl;
        }
        fclose(f);
    };

    read_bin(om.file_row_offsets, lvl.row_offsets.data(), om.padded_elems, sizeof(int));
    read_bin(om.file_csr_pivots, lvl.csr_pivots.data(), om.h_counter, sizeof(int));
    read_bin(om.file_C, lvl.C.data(), om.padded_elems, sizeof(__half));

    return lvl;
}

WitnessRange witnesses(const Level &lvl, int m, int n) {
    size_t cell = (size_t)m * lvl.Npad + n;
    int start = lvl.row_offsets[cell];
    int end = (cell + 1 < lvl.padded_elems) ? lvl.row_offsets[cell + 1] : lvl.h_counter;
    return { lvl.csr_pivots.data() + start, end - start };
}

float node_weight(const Level &lvl, int m, int n) {
    size_t cell = (size_t)m * lvl.Npad + n;
    return __half2float(lvl.C[cell]);
}

PathDag build_path_dag(const PathManifest &manifest, int m, int n_target) {
    PathDag dag;

    // Índice manifest.orders[i].s -> i: no se asume que el vector sea denso
    // ni que esté ordenado por s (el manifiesto sólo lista los órdenes con
    // efectos, en el orden en que se escribieron).
    std::map<int, int> s_to_index;
    for (size_t i = 0; i < manifest.orders.size(); i++)
        s_to_index[manifest.orders[i].s] = (int)i;

    // Levels cargados bajo demanda y cacheados — el recorrido puede tocar
    // niveles en cualquier orden (varias raíces en distintos s, expansión
    // estrictamente un nivel hacia atrás desde cada una).
    std::map<int, Level> levels;
    auto get_level = [&](int s) -> Level * {
        auto it = levels.find(s);
        if (it != levels.end()) return &it->second;
        auto idx_it = s_to_index.find(s);
        if (idx_it == s_to_index.end()) return nullptr;
        Level lvl = load_level(manifest.dump_dir, manifest.orders[idx_it->second]);
        return &(levels[s] = std::move(lvl));
    };

    // Raíces: TODOS los órdenes s en que (m, n_target) tuvo testigos propios
    // (no sólo el más alto) — así caminos de distinta longitud hacia el
    // mismo n_target conviven en el mismo resultado.
    std::vector<int> roots;
    for (const auto &om : manifest.orders) {
        Level *lvl = get_level(om.s);
        if (!lvl) continue;
        if (witnesses(*lvl, m, n_target).count > 0)
            roots.push_back(om.s);
    }

    std::set<std::pair<int, int>> visited; // (n, s) ya procesado
    std::queue<std::pair<int, int>> frontier;
    for (int s : roots)
        frontier.push({n_target, s});

    while (!frontier.empty()) {
        auto [n, s] = frontier.front();
        frontier.pop();

        if (visited.count({n, s})) continue;
        visited.insert({n, s});

        Level *lvl = get_level(s);
        if (!lvl) continue; // orden no persistido

        float w = node_weight(*lvl, m, n);

        WitnessRange wr = witnesses(*lvl, m, n);
        // Guardia defensiva: un row_offsets/csr_pivots corruptos o
        // desincronizados (p.ej. un dump generado antes de un fix del
        // escritor) puede producir un rango fuera de los límites reales del
        // CSR. En vez de leer memoria arbitraria, se descarta este nodo y se
        // avisa por stderr — no se aborta el proceso completo por un dump
        // corrupto.
        bool wr_ok = wr.count >= 0 && (size_t)(wr.data - lvl->csr_pivots.data()) + wr.count
                                           <= lvl->csr_pivots.size();
        if (!wr_ok) {
            std::cerr << "WARN: rango de testigos inválido para (n=" << n << ", s=" << s
                       << ") — csr_pivots corrupto o desincronizado, nodo omitido"
                       << std::endl;
            continue;
        }
        std::vector<int> ks(wr.data, wr.data + wr.count);
        dag[{n, s}] = DagEntry{w, ks};

        if (s == 0) continue; // caso base: sin nivel anterior que consultar

        // Poda local: cada testigo k se expande SÓLO si tiene soporte
        // propio (witnesses no vacío) exactamente en s-1 — incluido s-1==0:
        // un nodo en el nivel base sin count>0 ahí no es un camino real
        // (C=0 / sin efecto), es ruido, y se descarta igual que en
        // cualquier otro nivel. No se sigue buscando más atrás, y esto no
        // afecta a los demás testigos ni a otras raíces.
        Level *prev_lvl = get_level(s - 1);
        if (!prev_lvl) continue;

        for (int k : ks) {
            if (witnesses(*prev_lvl, m, k).count > 0)
                frontier.push({k, s - 1});
            // si no: k se descarta, no entra al DAG desde esta rama.
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Segunda pasada — poda en cascada: la primera pasada sólo exige que un
    // testigo tenga SOPORTE (witnesses no vacío) en s-1, pero eso no basta
    // para formar un camino real. Un nodo sólo es parte de un camino válido
    // si:
    //   - s == 0 y su weight >= thr de ese orden (arista base real), o
    //   - s  > 0 y al menos uno de sus testigos sobrevive en (k, s-1)
    //     (recursivo).
    // Sin esto, nodos como (n,1) con único testigo podado por no alcanzar
    // thr en s=0 quedan como callejones sin salida que no completan ningún
    // camino real, pero seguían apareciendo en el resultado.
    //
    // Se resuelve de abajo hacia arriba (s=0 primero, ya está disponible
    // porque el DAG se construyó completo): un nodo depende sólo de nodos
    // con s-1, así que un único barrido ascendente basta — no hace falta
    // iterar hasta punto fijo.
    // ─────────────────────────────────────────────────────────────────────
    std::map<int, float> thr_by_s;
    for (const auto &om : manifest.orders)
        thr_by_s[om.s] = om.thr;

    std::set<std::pair<int, int>> valid; // (n,s) que sobrevive la poda
    // Procesar por s creciente: así (k, s-1) ya está resuelto cuando se
    // evalúa (n, s).
    for (const auto &[node, entry] : dag) {
        if (node.s != 0) continue;
        float thr = thr_by_s.count(0) ? thr_by_s[0] : 0.0f;
        if (entry.weight >= thr)
            valid.insert({node.n, node.s});
    }
    // s de 1 en adelante, en orden — dag está ordenado por DagNode (s primero),
    // así que basta un segundo barrido en orden creciente de s.
    std::set<int> present_s;
    for (const auto &[node, entry] : dag) present_s.insert(node.s);
    for (int s : present_s) {
        if (s == 0) continue;
        for (const auto &[node, entry] : dag) {
            if (node.s != s) continue;
            for (int k : entry.witnesses) {
                if (valid.count({k, s - 1})) {
                    valid.insert({node.n, node.s});
                    break;
                }
            }
        }
    }

    // Filtrar dag: quitar nodos no válidos, y dentro de los válidos, quitar
    // de witnesses los testigos que no sobrevivieron.
    PathDag pruned;
    for (const auto &[node, entry] : dag) {
        if (!valid.count({node.n, node.s})) continue;
        std::vector<int> kept;
        for (int k : entry.witnesses)
            if (node.s > 0 && valid.count({k, node.s - 1}))
                kept.push_back(k);
        pruned[node] = DagEntry{entry.weight, kept};
    }

    return pruned;
}
