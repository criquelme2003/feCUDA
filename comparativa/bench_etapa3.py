"""Mediciones pendientes del documento de la Etapa 3.

1. v2 frente a v4 en dimensiones grandes (hasta n = 10.000).
2. Tiempo por orden de v4 en n = 10.000 y estimación de la duración de las
   2.640 corridas de la Etapa 4.

Usa el mismo generador y la misma convención que time.py (dimensión = tamaño
de la matriz, dos bloques de dim/2). Cada medición se escribe al CSV apenas
termina, para poder reanudar si el job se interrumpe.

Uso:
    python bench_etapa3.py                       # configuración por defecto
    python bench_etapa3.py --dims 1200 5000 10000 --kernels 2 4 --reps 5
    python bench_etapa3.py --summary-only        # solo resume el CSV existente
"""

import argparse
import csv
import os
import statistics as st
import time
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from sparse_generator import sparse_supercritical_block_matrix_julius as generator

SEED = 101
THRESHOLD = 0.5
ORDER = 100  # cota superior, igual que en los experimentos de profundidad

# Grilla de los experimentos de la Etapa 4 (19 c x 7 n x 20 repeticiones).
# Ajustar si cambia el diseño experimental.
# Los valores de n son una aproximación: reemplazarlos por los del diseño real.
EXP_N = [100, 500, 1000, 2500, 5000, 7500, 10000]
EXP_REPS = 20
EXP_N_C = 19
EXP_VALID = 132  # 133 configuraciones, 132 válidas -> 2.640 corridas


def run_once(ft, dim, c, seed, kernel):
    n = dim // 2
    m1, _, _ = generator(n, n, c, seed)
    m2 = m1.copy()
    t0 = time.perf_counter()
    _, _, eta0 = ft.maxmin(m1, m2, THRESHOLD, ORDER, kernel=kernel)
    # ft.maxmin es síncrona: devuelve arreglos de host después de que la GPU termina.
    return time.perf_counter() - t0, eta0


def measure(args):
    import forgethreads as ft

    ft.set_verbose(False)
    new_file = not os.path.exists(args.out)
    done = set()
    if not new_file:
        with open(args.out) as f:
            for r in csv.DictReader(f):
                done.add((int(r["dimension"]), float(r["c"]), int(r["kernel"]), int(r["repeticion"])))

    # Calentamiento: inicializa el contexto CUDA y carga los kernels.
    for k in args.kernels:
        run_once(ft, 200, args.c[0], SEED, k)

    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["dimension", "c", "kernel", "repeticion", "tiempo_s", "eta0", "tiempo_por_orden_s"])
        for dim in args.dims:
            for c in args.c:
                for k in args.kernels:
                    for rep in range(args.reps):
                        if (dim, float(c), k, rep) in done:
                            continue
                        t, eta0 = run_once(ft, dim, c, SEED + rep, k)
                        w.writerow([dim, c, k, rep, f"{t:.6f}", eta0, f"{t / max(eta0, 1):.6f}"])
                        f.flush()
                        print(f"dim={dim} c={c} v{k} rep={rep} t={t:.3f}s eta0={eta0} "
                              f"t/orden={1000 * t / max(eta0, 1):.1f}ms", flush=True)


def summarize(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(int(r["dimension"]), float(r["c"]), int(r["kernel"]))].append(
                (float(r["tiempo_s"]), int(r["eta0"]), float(r["tiempo_por_orden_s"])))

    print("\n=== Resumen por configuración ===")
    print(f"{'dim':>6} {'c':>6} {'kernel':>6} {'reps':>4} {'tiempo [ms]':>18} {'eta0':>6} {'ms/orden':>10}")
    for key in sorted(rows):
        v = rows[key]
        t = [x[0] * 1000 for x in v]
        sd = st.stdev(t) if len(t) > 1 else 0.0
        print(f"{key[0]:>6} {key[1]:>6g} {'v' + str(key[2]):>6} {len(v):>4} "
              f"{st.mean(t):>10.2f} ± {sd:<6.2f} {st.mean(x[1] for x in v):>6.1f} "
              f"{1000 * st.mean(x[2] for x in v):>10.2f}")

    # 1) v2 frente a v4 (tiempo por orden, porque eta0 puede variar entre semillas)
    print("\n=== v2 / v4 (tiempo por orden; > 1 significa que v4 es más rápido) ===")
    for dim, c in sorted({(k[0], k[1]) for k in rows}):
        if (dim, c, 2) in rows and (dim, c, 4) in rows:
            t2 = st.mean(x[2] for x in rows[(dim, c, 2)])
            t4 = st.mean(x[2] for x in rows[(dim, c, 4)])
            print(f"dim={dim:>6} c={c:g}: v2={1000 * t2:.2f} ms  v4={1000 * t4:.2f} ms  v2/v4={t2 / t4:.2f}x")

    # 2) Estimación de las 2.640 corridas con v4: interpola el tiempo por orden
    #    medido y usa eta0 medido (o la cota ORDER si no hay dato de eta0).
    per_order = defaultdict(list)
    etas = []
    for (dim, c, k), v in rows.items():
        if k == 4:
            per_order[dim] += [x[2] for x in v]
            etas += [x[1] for x in v]
    if per_order:
        pts = sorted((d, st.mean(v)) for d, v in per_order.items())

        def t_order(n):
            # ley cúbica anclada al punto medido más cercano
            d, t = min(pts, key=lambda p: abs(p[0] - n))
            return t * (n / d) ** 3

        eta_mean = st.mean(etas)
        frac = EXP_VALID / (len(EXP_N) * EXP_N_C)
        total_mean = sum(t_order(n) * eta_mean for n in EXP_N) * EXP_N_C * EXP_REPS * frac
        total_max = sum(t_order(n) * ORDER for n in EXP_N) * EXP_N_C * EXP_REPS * frac
        n_runs = EXP_VALID * EXP_REPS
        print(f"\n=== Estimación de las {n_runs} corridas (v4) ===")
        for d, t in pts:
            print(f"  medido: n={d}  {1000 * t:.2f} ms/orden")
        print(f"  eta0 medio observado: {eta_mean:.1f}")
        print(f"  con eta0 medio:       {total_mean / 3600:.2f} h")
        print(f"  cota (eta0 = {ORDER}):    {total_max / 3600:.2f} h")
        print("  (n distintos a los medidos se extrapolan con O(n^3); medir n=10000 para anclar el peor caso)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", type=int, nargs="+", default=[1200, 2500, 5000, 10000])
    p.add_argument("--kernels", type=int, nargs="+", default=[2, 4], choices=[1, 2, 3, 4])
    p.add_argument("--c", type=float, nargs="+", default=[1.0],
                   help="parámetro c del generador (por defecto 1, como time.py)")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--out", default="bench_etapa3.csv")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    if not args.summary_only:
        measure(args)
    summarize(args.out)


if __name__ == "__main__":
    main()
