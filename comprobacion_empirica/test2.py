
import forgethreads as ft
import numpy as np
from numba import njit
import time
import csv
import matplotlib.pyplot as plt
import cupy as cp
from matrix_construction import sparse_supercritical_block_matrix2
ft.set_verbose(True)

NS      = [100]
CS      = [50]

REPEATS = 20
THR     = 0.5
ORDER   = 100
CSV_OUT = "sweep_100n_50c_order.csv"


def run_sweep():
    # warmup
    m_w, _, _ = sparse_supercritical_block_matrix2(100, 100, 4, seed=0)
    m_w = m_w.reshape(1, 200, 200).astype(np.float16)
    ft.maxmin(m_w.copy(), m_w.copy(), THR, ORDER)

    results = {}  # c -> n -> list of effective_orders

    import os
    write_header = not os.path.exists(CSV_OUT) or os.path.getsize(CSV_OUT) == 0
    with open(CSV_OUT,   "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["c", "n", "repeticion", "orden_efectivo"])

        for c in CS:
            results[c] = {}
            for n in NS:
                _n = int(n/2)
                if(c < _n):
                  orders = []
                  for rep in range(REPEATS):
                      cp.get_default_memory_pool().free_all_blocks()
                      seed = rep * 1000 + n
                      N_tot = n
                      E, _, _ = sparse_supercritical_block_matrix2(_n, _n, c, seed=seed)
                      m1 = E.reshape(1, N_tot, N_tot).astype(np.float16)
                      m2 = m1.copy()
                      _, _, eff_order = ft.maxmin(m1, m2, THR, ORDER)
                      orders.append(eff_order)

                  results[c][n] = orders
                  print(
                      f"c={c}  n={n}  "
                      f"mean={np.mean(orders):.2f}  "
                      f"min={int(np.min(orders))}  "
                      f"max={int(np.max(orders))}"
                  )

                  for rep, o in enumerate(orders):
                      writer.writerow([c, n, rep, o])
                  f.flush()

    return results


def plot_results(results):
    for c in CS:
        _, ax = plt.subplots(figsize=(9, 5))
        ns    = sorted(results[c].keys())
        means = np.array([np.mean(results[c][n]) for n in ns], dtype=float)
        mins  = np.array([np.min(results[c][n])  for n in ns], dtype=float)
        maxs  = np.array([np.max(results[c][n])  for n in ns], dtype=float)

        ax.fill_between(ns, mins, maxs, alpha=0.20, label="rango [min, max]")
        ax.plot(ns, means, marker="o", label="media")
        ax.plot(ns, mins,  marker="v", linestyle="--", linewidth=0.8, label="mínimo")
        ax.plot(ns, maxs,  marker="^", linestyle="--", linewidth=0.8, label="máximo")

        for n, mean, mn, mx in zip(ns, means, mins, maxs):
            ax.annotate(f"{mean:.2f}", (n, mean), textcoords="offset points", xytext=(0,  6), ha="center", fontsize=7, color="C1")
            ax.annotate(f"{mn:.2f}",   (n, mn),   textcoords="offset points", xytext=(0, -10), ha="center", fontsize=7, color="C2")
            ax.annotate(f"{mx:.2f}",   (n, mx),   textcoords="offset points", xytext=(0,  6), ha="center", fontsize=7, color="C3")

        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], rotation=45, ha="right")
        ax.set_xlabel("n")
        ax.set_ylabel("orden alcanzado")
        ax.set_title(f"Orden alcanzado vs n  (c={c}, thr={THR}, {REPEATS} repeticiones)")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fname = f"sweep_n_order_02_06_block{c}.png"
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"Guardado: {fname}")


results = run_sweep()

plot_results(results)
