
import forgethreads as ft
import numpy as np
from numba import njit
import time
import csv
import matplotlib.pyplot as plt
import cupy as cp

ft.set_verbose(True)

@njit(cache=True, fastmath=True)
def _flat_to_rc(sel, n, m, rows, cols):
    nA = n * (n - 1)
    nB = n * m
    n1 = n - 1 if n > 1 else 1
    m1 = m - 1 if m > 1 else 1

    for i in range(len(sel)):
        f = sel[i]
        if f < nA:
            r = f // n1
            c = f % n1
            if c >= r:
                c += 1
            rows[i] = r
            cols[i] = c
        elif f < nA + nB:
            f -= nA
            rows[i] = f // m
            cols[i] = n + f % m
        else:
            f -= nA + nB
            r  = f // m1
            c  = f % m1
            if c >= r:
                c += 1
            rows[i] = n + r
            cols[i] = n + c


def build_matrix2(n, m, d, seed=None):
    rng = np.random.default_rng(seed)
    N   = n + m
    h   = np.float32(0.5)

    M = np.zeros((N, N), dtype=np.float32)
    M[:n, :n] = rng.random((n, n), dtype=np.float32) * h
    M[n:, n:] = rng.random((m, m), dtype=np.float32) * h
    M[:n, n:] = rng.random((n, m), dtype=np.float32) * h

    np.fill_diagonal(M[:n, :n], 1.0)
    np.fill_diagonal(M[n:, n:], 1.0)

    total_possible = n * (n - 1) + n * m + m * (m - 1)
    target_edges   = min(int(round(d * n)), total_possible)

    if target_edges > 0:
        sel  = rng.choice(total_possible, size=target_edges, replace=False)
        rows = np.empty(target_edges, dtype=np.int64)
        cols = np.empty(target_edges, dtype=np.int64)
        _flat_to_rc(sel, n, m, rows, cols)
        M[rows, cols] = rng.random(target_edges, dtype=np.float32) * h + h
    else:
        print("The distribution is incompatible")
    return M.reshape(1, N, N).astype(np.float16)


NS = [5000]
REPEATS = 10


def run_timing(ns=NS, repeats=REPEATS):
    times_ft = {}

    # warmup
    m1 = build_matrix2(100, 100, 100 * np.log(100), seed=444)
    m2 = m1.copy()
    ft.maxmin(m1, m2, 0.3, 4)
    ft.maxmin(m1, m2, 0.3, 4)
    ft.maxmin(m1, m2, 0.3, 4)

    for n in ns:
        ft_runs = []

        for _ in range(repeats):
            cp.get_default_memory_pool().free_all_blocks()
            m1 = build_matrix2(n, n, n * np.log(2*n), seed=444)
            m2 = m1.copy()
            t0 = time.perf_counter()
            ft.maxmin(m1, m2, 0.5, 4)
            ft_runs.append((time.perf_counter() - t0) * 1000)

        times_ft[n] = ft_runs
        print(f"n={n} done")

    with open("timing_results2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "nodos", "funcion", "repeticion", "tiempo_ms"])
        for n, runs in times_ft.items():
            for i, t in enumerate(runs):
                writer.writerow([n, n * 2, "ft.maxmin", i, round(t, 4)])

    return times_ft


def plot_comparison(times_ft):
    keys_ft = sorted(times_ft)
    xs_ft   = [n * 2 for n in keys_ft]
    mean_ft = [np.mean(times_ft[n]) for n in keys_ft]
    std_ft  = [np.std(times_ft[n])  for n in keys_ft]

    # log
    _, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(xs_ft, mean_ft, yerr=std_ft, marker="s", label="ft.maxmin", capsize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs_ft)
    ax.set_xticklabels([str(x) for x in xs_ft], rotation=45, ha="right")
    ax.set_xlabel("número de nodos (n×2)")
    ax.set_ylabel("tiempo (ms)")
    ax.set_title(f"ft.maxmin — escala log ({REPEATS} repeticiones por n)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("timing2_log.png", dpi=150)
    plt.show()

    # lineal
    _, ax2 = plt.subplots(figsize=(9, 5))
    ax2.errorbar(xs_ft, mean_ft, yerr=std_ft, marker="s", label="ft.maxmin", capsize=4)
    ax2.set_xticks(xs_ft)
    ax2.set_xticklabels([str(x) for x in xs_ft], rotation=45, ha="right")
    ax2.set_xlabel("número de nodos (n×2)")
    ax2.set_ylabel("tiempo (ms)")
    ax2.set_title(f"ft.maxmin — escala real ({REPEATS} repeticiones por n)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("timing2_linear.png", dpi=150)
    plt.show()


def sparse_supercritical_matrix(n, c, seed=None):
    rng = np.random.default_rng(seed)
    p = c / (n - 1)
    E = np.zeros((n, n), dtype=float)
    U = rng.binomial(1, p, size=(n, n))
    E[np.triu_indices(n, k=1)] = U[np.triu_indices(n, k=1)]
    np.fill_diagonal(E, 1)
    return E


def sweep_d(n=10000, thr=0.5, order=25, seed=42):
    ds = np.round(np.arange(1, np.log(n), 0.01), 2)
    max_orders = []

    # for d in ds:
    m1 = sparse_supercritical_matrix(n, 8, seed=seed).reshape(1, n, n).astype(np.float16)
    m2 = m1.copy()
    _, _, eff_order = ft.maxmin(m1, m2, thr, order)
    max_orders.append(eff_order)
    print(f"c={d:.2f}  effective_order={eff_order}")

    _, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ds, max_orders, marker="o")
    ax.set_xlabel("grado promedio c")
    ax.set_ylabel("orden máximo alcanzado")
    ax.set_title(f"Orden máximo vs grado promedio  (n={n}, thr={thr})")
    ax.set_xticks(ds[::20])
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("sweep_d_vs_order.png", dpi=150)
    plt.show()

    return ds, max_orders


sweep_d()
# times_ft = run_timing()
# plot_comparison(times_ft)
