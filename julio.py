"""
Experimento: validación empírica de las cotas teóricas del paper.

Contexto del paper
------------------
- Se analiza el grafo aleatorio dirigido G con N nodos.
- Cada arista (i,j) existe con prob p y tiene peso w(i,j) ~ U(0.55, 1.0).
- Usando epsilon = 0.5 (> 1/2), el algoritmo maxmin encuentra caminos cuyo
  valor max-min supera al peso directo en al menos epsilon.
- Como el peso directo de NO-aristas es 0, la condición se simplifica a:
  "existe un camino de longitud k con todos los pesos > 0.5".

Claim 1 (unicidad de nivel):
  Para cada par (i,j) con i≠j, existe exactamente UN orden k en el que ese
  par se "activa" por primera vez.  Con epsilon > 1/2 no hay activaciones
  duplicadas ni pares que desaparezcan entre órdenes.

Claim 2 (cotas del orden máximo según densidad):
  - Sparse        (grado medio < 1):      E[eta_0] = O(1)
  - Supercrítico  (1 < grado < D const):  E[eta_0] = Theta(log N)
  - Denso         (grado medio > cN):     E[eta_0] = O(1)
"""

import math
import sys
import numpy as np

sys.path.insert(0, '/workspace')
import forgethreads as ft


# ─────────────────────────────────────────────────────────────────────────────
# Generación de grafos
# ─────────────────────────────────────────────────────────────────────────────

def make_graph(N: int, avg_degree: float, rng: np.random.Generator) -> np.ndarray:
    """
    Grafo aleatorio dirigido [1, N, N] float16.
    - Arista (i,j) con prob  p = avg_degree / (N-1).
    - Si existe: peso ~ U(0.55, 1.0)  [siempre > 0.5]
    - Si no:     peso = 0.0           [nunca activa caminos con thr=0.5]
    La diagonal es 0 (sin auto-bucles).
    """
    p = min(avg_degree / max(N - 1, 1), 1.0)
    mask = rng.random((N, N)) < p
    np.fill_diagonal(mask, False)
    weights = rng.uniform(0.55, 1.0, (N, N)).astype(np.float16)
    A = np.where(mask, weights, np.float16(0.0)).reshape(1, N, N)
    return A.astype(np.float16)


# ─────────────────────────────────────────────────────────────────────────────
# Claim 1: verificación de unicidad de nivel
# ─────────────────────────────────────────────────────────────────────────────

def verify_single_level(N: int = 50, avg_degree: float = 3.0,
                        thr: float = 0.5, max_order: int = 40,
                        seed: int = 42) -> dict:
    """
    Para cada par (i,j) con i≠j registra el PRIMER orden en que aparece.
    Verifica dos invariantes:
      - Monotonía: ningún par desaparece entre órdenes  (set crece o se mantiene).
      - Unicidad: cada par tiene exactamente un "primer orden".

    Retorna un dict con métricas.
    """
    rng = np.random.default_rng(seed)
    A = make_graph(N, avg_degree, rng)

    first_order_of = {}   # (m,n) -> orden en que apareció por primera vez
    prev_mn = set()
    pairs_new_per_order = {}
    monotonicity_violations = []

    for order in range(1, max_order + 1):
        paths, values, eff = ft.maxmin(A, A, thr, order)
        p = paths.to_numpy()

        if p.shape[0] == 0:
            current_mn = set()
        else:
            # Columnas: (b, m, k1, ..., n); m=col 1, n=col -1
            current_mn = {(int(row[1]), int(row[-1])) for row in p}

        new_pairs = current_mn - prev_mn
        pairs_new_per_order[order] = len(new_pairs)

        for mn in new_pairs:
            first_order_of[mn] = order

        # Verificar monotonía: los pares de antes ¿siguen estando?
        disappeared = prev_mn - current_mn
        if disappeared:
            monotonicity_violations.extend(
                (mn[0], mn[1], first_order_of.get(mn, '?'), order)
                for mn in disappeared
            )

        prev_mn = current_mn

        if order == eff:        # convergencia detectada por la librería
            break

    return {
        "ok": len(monotonicity_violations) == 0,
        "pairs_new_per_order": pairs_new_per_order,
        "total_pairs_activated": len(first_order_of),
        "monotonicity_violations": monotonicity_violations,
        "effective_order": max(pairs_new_per_order.keys()) if pairs_new_per_order else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Claim 2: E[max_order] en función del régimen de densidad
# ─────────────────────────────────────────────────────────────────────────────

def avg_max_order(N: int, avg_degree: float, n_iter: int = 40,
                  thr: float = 0.5, max_order: int = 60,
                  seed: int = 0) -> tuple:
    """
    Estima E[eta_0] ejecutando n_iter grafos aleatorios con 'avg_degree'
    aristas esperadas por nodo para grafos de N nodos.

    Retorna (media, std) del effective_order.
    """
    rng = np.random.default_rng(seed)
    orders = []

    for _ in range(n_iter):
        A = make_graph(N, avg_degree, rng)
        _, _, eff = ft.maxmin(A, A, thr, max_order)
        orders.append(eff)

    return float(np.mean(orders)), float(np.std(orders))


def run_regime_experiment(N_values: list, avg_degree_fn,
                          label: str, n_iter: int = 30, seed: int = 0):
    """
    Ejecuta el experimento para un régimen concreto sobre una lista de N.
    avg_degree_fn(N) -> float  (puede ser constante o función de N).
    """
    print(f"\n{'─'*60}")
    print(f"  Régimen: {label}")
    print(f"{'─'*60}")
    print(f"  {'N':>5}  {'deg':>6}  {'log N':>7}  {'E[eta]':>8}  {'std':>6}")
    results = {}
    for N in N_values:
        deg = avg_degree_fn(N)
        mean, std = avg_max_order(N, deg, n_iter=n_iter, seed=seed)
        results[N] = (mean, std, deg)
        print(f"  {N:>5}  {deg:>6.2f}  {math.log(N):>7.2f}  {mean:>8.3f}  {std:>6.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Programa principal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Claim 1: unicidad de nivel ───────────────────────────────────────────
    print("=" * 60)
    print("CLAIM 1 — Unicidad de nivel con epsilon = 0.5")
    print("=" * 60)

    for deg, seed in [(2.0, 10), (3.0, 20), (5.0, 30)]:
        res = verify_single_level(N=60, avg_degree=deg, thr=0.5, seed=seed)
        status = "✓ PASS" if res["ok"] else "✗ FAIL"
        print(f"  deg={deg:.1f}  max_order={res['effective_order']}"
              f"  pares activados={res['total_pairs_activated']}"
              f"  violaciones monot.={len(res['monotonicity_violations'])}"
              f"  [{status}]")
        print(f"    nuevos por orden: {res['pairs_new_per_order']}")

    # ── Claim 2: escalado E[eta_0] vs N ─────────────────────────────────────
    print()
    print("=" * 60)
    print("CLAIM 2 — E[max_order] vs N por régimen de densidad")
    print("=" * 60)

    N_values = [20, 40, 80, 160, 320]
    n_iter   = 25

    # Sparse: grado promedio < 1 → E[eta_0] = O(1)
    sparse_res = run_regime_experiment(
        N_values, avg_degree_fn=lambda N: 0.7,
        label="SPARSE  (deg=0.7 < 1)", n_iter=n_iter, seed=1)

    # Supercrítico sparse: deg ~ constante > 1 → E[eta_0] = Theta(log N)
    super_res = run_regime_experiment(
        N_values, avg_degree_fn=lambda N: 3.0,
        label="SUPERCRÍTICO (deg=3.0 > 1)", n_iter=n_iter, seed=2)

    # Denso: deg ~ c*N → E[eta_0] = O(1)
    dense_res = run_regime_experiment(
        N_values, avg_degree_fn=lambda N: 0.5 * N,
        label="DENSO  (deg=N/2)", n_iter=n_iter, seed=3)

    # ── Tabla resumen ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("TABLA RESUMEN — E[eta_0] comparado con log(N)")
    print("=" * 60)
    print(f"  {'N':>5}  {'log N':>6}  {'sparse':>8}  {'supercrit':>10}  {'dense':>7}")
    print(f"  {'':─>5}  {'':─>6}  {'':─>8}  {'':─>10}  {'':─>7}")
    for N in N_values:
        s_m  = sparse_res[N][0]
        sc_m = super_res[N][0]
        d_m  = dense_res[N][0]
        print(f"  {N:>5}  {math.log(N):>6.2f}  {s_m:>8.3f}  {sc_m:>10.3f}  {d_m:>7.3f}")

    # ── Ratio supercrítico / log(N): debería ser ~ constante ────────────────
    print()
    print("  Ratio E[eta_0_supercrit] / log(N)  (debe ser ~cte si Theta(log N)):")
    for N in N_values:
        ratio = super_res[N][0] / math.log(N)
        print(f"    N={N:>4}  ratio={ratio:.4f}")
