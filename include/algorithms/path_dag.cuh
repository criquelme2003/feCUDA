#ifndef PATH_DAG_CUH
#define PATH_DAG_CUH

#include <cuda_fp16.h>
#include <map>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// path_dag — reconstrucción on-demand de caminos a partir de los CSR de
// testigos persistidos por maxminv2_count (ver dump_dir en headers.cuh).
//
// Para el par (m, n_target), pueden existir varios caminos de distinta
// longitud (p.ej. una arista directa en s=0, y otro camino de 5 saltos en
// s=5): (m,n_target) puede tener testigos propios en más de un orden s, cada
// uno una raíz independiente. build_path_dag arma el DAG partiendo de TODAS
// esas raíces a la vez.
//
// Cada nodo del DAG es (n, s); sus aristas salientes son los testigos k que
// participan en min(C_{s-1}[m,k], B[k,n]) == C_s[m,n]. Un testigo k se busca
// EXACTAMENTE en el nivel s-1 (no se retrocede más): si (m,k) no tiene
// testigos propios ahí, k se descarta (poda local) — no se considera nodo
// terminal ni se sigue buscando su soporte en niveles anteriores a s-1,
// salvo el caso base s-1==0, donde cualquier valor es válido (arista base).
//
// El peso de un nodo es C_s[m,n] — todas sus aristas entrantes comparten ese
// valor por definición de testigo.
//
// Enumerar caminos completos queda a cargo del renderizador, recorriendo el
// DAG; aquí sólo se materializa la topología + pesos, acotado por el número
// de nodos (n,s) efectivamente alcanzables, no por el producto de ramas.
// ─────────────────────────────────────────────────────────────────────────────

// Una entrada del manifest.json: dimensiones y rutas de archivo de un orden.
struct OrderManifest {
    int s;
    int M, N, K;
    int Mpad, Npad;
    size_t padded_elems;
    int h_counter;
    float thr;
    std::string file_C;
    std::string file_row_offsets;
    std::string file_csr_pivots;
};

// Manifiesto completo: dump_dir + entradas indexadas por s.
struct PathManifest {
    std::string dump_dir;
    std::vector<OrderManifest> orders; // orders[s] corresponde al orden s (denso, 0..max)
};

// Parsea dump_dir/manifest.json. Termina el proceso (vía CHECK-style error) si
// el archivo no existe o no se puede leer; devuelve orders vacío si el JSON
// no tiene el campo esperado.
PathManifest load_path_manifest(const std::string &dump_dir);

// Un nivel cargado a RAM: CSR de testigos + valores, para consultas repetidas
// de witnesses()/node_weight() sin volver a tocar disco.
struct Level {
    int Npad = 0;
    size_t padded_elems = 0;
    int h_counter = 0;
    std::vector<int> row_offsets; // tamaño padded_elems
    std::vector<int> csr_pivots;  // tamaño h_counter
    std::vector<__half> C;        // tamaño padded_elems
};

// Carga a RAM los 3 .bin del orden om.s (row_offsets, csr_pivots, C).
Level load_level(const std::string &dump_dir, const OrderManifest &om);

// Testigos de la celda (m,n) en este nivel: rango [start,start+count) dentro
// de lvl.csr_pivots. count == 0 si la celda no tuvo efecto en ese orden.
struct WitnessRange {
    const int *data;
    int count;
};
WitnessRange witnesses(const Level &lvl, int m, int n);

// Valor max-min de la celda (m,n) en este nivel (el peso del nodo (n, lvl.s)
// en el DAG).
float node_weight(const Level &lvl, int m, int n);

// Un nodo del DAG de reconstrucción: destino n alcanzado en el orden s,
// partiendo siempre del mismo origen m (fijo por consulta).
struct DagNode {
    int n;
    int s;
    bool operator<(const DagNode &o) const { return s != o.s ? s < o.s : n < o.n; }
};

struct DagEntry {
    float weight;             // C_s[m,n]
    std::vector<int> witnesses; // testigos k: aristas hacia (k, s-1)
};

// DAG completo de reconstrucción para un origen m fijo: nodo -> peso + testigos.
using PathDag = std::map<DagNode, DagEntry>;

// Arma el DAG de testigos para (m, n_target), considerando como raíces TODOS
// los órdenes s en que (m, n_target) tuvo testigos (no sólo el orden más
// alto) — así conviven caminos de distinta longitud hacia el mismo destino.
// Cada testigo se expande sólo si tiene soporte exactamente en el nivel
// inmediatamente anterior (o llega al caso base s=0); si no, se descarta esa
// rama (poda local, no afecta a nodos hermanos). BFS con deduplicación por
// (n,s). Carga los niveles del manifest bajo demanda.
PathDag build_path_dag(const PathManifest &manifest, int m, int n_target);

#endif
