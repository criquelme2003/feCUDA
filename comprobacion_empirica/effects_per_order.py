"""Efectos encontrados en cada orden (iteración) de ft.maxmin, por ejecución.

Grilla: n = 2500 (dos bloques de n/2, misma convención que test2.py),
c en {1, 9/8, 5/4, 3/2, 2, 9/4, 5/2, 11/4, 3}, 20 repeticiones -> 180 ejecuciones.

El CSV queda en formato largo: una fila por (c, n, repeticion, orden) con el
número de efectos de ese orden. Solo aparecen los órdenes con efectos (> 0);
orden_efectivo repite en cada fila el último orden alcanzado por la ejecución.
Una ejecución sin ningún efecto deja una única fila con orden = 0, efectos = 0.
Cada ejecución se escribe al terminar, así que el script se puede reanudar.

Uso:
    python effects_per_order.py
    python effects_per_order.py --out otro.csv --kernel 4
"""

import argparse
import csv
import os

import cupy as cp
import numpy as np

import forgethreads as ft
from matrix_construction import sparse_supercritical_block_matrix2

NS = [2500]
CS = [1, 9/8, 5/4, 3/2, 2, 9/4, 5/2, 11/4, 3]
REPEATS = 20
THR = 0.5
ORDER = 200


def run_once(n, c, seed, kernel):
    half = n // 2
    E, _, _ = sparse_supercritical_block_matrix2(half, half, c, seed=seed)
    m1 = E.reshape(1, n, n).astype(np.float16)
    m2 = m1.copy()
    _, _, eff_order, effects = ft.maxmin(m1, m2, THR, ORDER, kernel=kernel, return_effects=True)
    return eff_order, effects


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="effects_per_order_2500n.csv")
    p.add_argument("--kernel", type=int, default=4, choices=[1, 2, 3, 4])
    args = p.parse_args()

    ft.set_verbose(False)

    done = set()
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    if not new_file:
        with open(args.out) as f:
            for r in csv.DictReader(f):
                done.add((float(r["c"]), int(r["n"]), int(r["repeticion"])))

    # warmup: inicializa el contexto CUDA y carga los kernels
    run_once(200, 4, 0, args.kernel)

    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["c", "n", "repeticion", "seed", "orden", "efectos", "orden_efectivo"])

        for c in CS:
            for n in NS:
                for rep in range(REPEATS):
                    if (float(c), n, rep) in done:
                        continue
                    cp.get_default_memory_pool().free_all_blocks()
                    seed = rep * 1000 + n
                    eff_order, effects = run_once(n, c, seed, args.kernel)

                    for s, e in enumerate(effects):
                        writer.writerow([c, n, rep, seed, s + 1, int(e), eff_order])
                    if len(effects) == 0:
                        # sin efectos: fila centinela para no repetir la ejecución al reanudar
                        writer.writerow([c, n, rep, seed, 0, 0, eff_order])
                    f.flush()

                    total = int(np.sum(effects))
                    pico = int(np.argmax(effects)) + 1 if len(effects) else 0
                    print(f"c={c:g} n={n} rep={rep} orden_efectivo={eff_order} "
                          f"efectos_totales={total} orden_pico={pico}", flush=True)


if __name__ == "__main__":
    main()
