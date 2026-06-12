
import forgethreads as ft
import numpy as np
from numba import njit
from datetime import datetime
import time
import csv
import matplotlib.pyplot as plt
from forgeffects_modules.iterative_maxmin_cuadrado import iterative_maxmin_cuadrado
import tensorflow as tf
import cupy as cp
import gc

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

rng = np.random.default_rng()
ft.set_verbose(True)

@njit(cache=True, fastmath=True)
def _flat_to_rc(sel, n, m, rows, cols):
    """
    Mapea índices planos [0, total_possible) → (row, col) en M sin
    construir la tabla completa de índices.

    Partición:
      [0,        n*(n-1))           → bloque A (n×n) fuera de diagonal
      [n*(n-1),  n*(n-1) + n*m)     → bloque B (n×m)
      [n*(n-1) + n*m, total)        → bloque D (m×m) fuera de diagonal
    """
    nA = n * (n - 1)
    nB = n * m
    n1 = n - 1 if n > 1 else 1   # guarda contra /0 cuando n==1 (nA=0)
    m1 = m - 1 if m > 1 else 1

    for i in range(len(sel)):
        f = sel[i]
        if f < nA:                         # ── bloque A
            r = f // n1
            c = f % n1
            if c >= r:
                c += 1
            rows[i] = r
            cols[i] = c
        elif f < nA + nB:                  # ── bloque B
            f -= nA
            rows[i] = f // m
            cols[i] = n + f % m
        else:                              # ── bloque D
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

    # ── 1. Bloques directamente en float32 (evita float64 intermedio) ─
    M = np.zeros((N, N), dtype=np.float32)
    M[:n, :n] = rng.random((n, n), dtype=np.float32) * h   # [0, 0.5)
    M[n:, n:] = rng.random((m, m), dtype=np.float32) * h
    M[:n, n:] = rng.random((n, m), dtype=np.float32) * h



    # ── 2. Diagonales = 1 sobre la vista (sin copias) ─────────────────
    np.fill_diagonal(M[:n, :n], 1.0)
    np.fill_diagonal(M[n:, n:], 1.0)

    # ── 3. Aristas extra ───────────────────────────────────────────────
    # current_edges es siempre 0 → todos los valores generados son < 0.5
    total_possible = n * (n - 1) + n * m + m * (m - 1)
    target_edges   = min(int(round(d * n)), total_possible)

    if target_edges > 0:
        # choice usa Floyd's sampling → O(k) cuando k << total
        sel  = rng.choice(total_possible, size=target_edges, replace=False)
        rows = np.empty(target_edges, dtype=np.int64)
        cols = np.empty(target_edges, dtype=np.int64)
        _flat_to_rc(sel, n, m, rows, cols)                  # JIT compilado
        M[rows, cols] = rng.random(target_edges, dtype=np.float32) * h + h
    else: 
      print("The distribution is incompatible ")
    return M.reshape(1, N, N).astype(np.float16)
    # return M

# ── Warm-up: compilar el JIT antes del bucle principal ───────────────
# build_matrix2(4, 2, 1.0, seed=0)


# m1 = build_matrix2(5000,5000,2,1111)
# m2 = m1.copy()


# print(datetime.now())
# ft.maxmin(
#       m1, 
#       m2, 
#       0.5, 5,False,2)
# print(datetime.now())

ITERATIVE_MAX_N     = 600   # límite GPU: OOM en VRAM (tensor n³ fp16)
ITERATIVE_CPU_MAX_N = 900   # límite CPU: overflow int32 en indices.py para n > 900
NS = [100, 200, 300, 400, 500, 600, 800, 900, 1000, 2000, 5000]
REPEATS = 10


def run_timing(ns=NS, repeats=REPEATS):
    times_iterative     = {}
    times_iterative_cpu = {}
    times_ft            = {}

    #WARMUP
    m = build_matrix2(100, 100, 100 * np.log(100), seed=444)
    iterative_maxmin_cuadrado(m, 0.3, 4)
    iterative_maxmin_cuadrado(m, 0.3, 4)
    iterative_maxmin_cuadrado(m, 0.3, 4)
    with tf.device("/CPU:0"):
        iterative_maxmin_cuadrado(m, 0.3, 4)
        iterative_maxmin_cuadrado(m, 0.3, 4)
        iterative_maxmin_cuadrado(m, 0.3, 4)

    for n in ns:
        iter_runs     = []
        iter_cpu_runs = []
        ft_runs       = []

        for _ in range(repeats):

            if n <= ITERATIVE_MAX_N:
                with tf.device("/GPU:0"):
                    cp.get_default_memory_pool().free_all_blocks()
                    m = build_matrix2(n, n, n * np.log(n), seed=444)
                    t0 = time.perf_counter()
                    iterative_maxmin_cuadrado(m, 0.3, 4)
                    iter_runs.append((time.perf_counter() - t0) * 1000)
                tf.keras.backend.clear_session()
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()

            if n <= ITERATIVE_CPU_MAX_N:
                with tf.device("/CPU:0"):
                    m = build_matrix2(n, n, n * np.log(n), seed=444)
                    try:
                        t0 = time.perf_counter()
                        iterative_maxmin_cuadrado(m, 0.3, 4)
                        iter_cpu_runs.append((time.perf_counter() - t0) * 1000)
                    except Exception as e:
                        print(f"[CPU] n={n} falló: {e}")
                        break

            cp.get_default_memory_pool().free_all_blocks()
            m1 = build_matrix2(n, n, n * np.log(n), seed=444)
            m2 = m1.copy()
            t0 = time.perf_counter()
            ft.maxmin(m1, m2, 0.3, 4, False)
            ft_runs.append((time.perf_counter() - t0) * 1000)

        if iter_runs:
            times_iterative[n] = iter_runs
        times_iterative_cpu[n] = iter_cpu_runs
        times_ft[n] = ft_runs
        print(f"n={n} done")

    with open("timing_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "nodos", "funcion", "repeticion", "tiempo_ms"])
        for n, runs in times_iterative.items():
            for i, t in enumerate(runs):
                writer.writerow([n, n * 2, "iterative_maxmin_cuadrado_gpu", i, round(t, 4)])
        for n, runs in times_iterative_cpu.items():
            for i, t in enumerate(runs):
                writer.writerow([n, n * 2, "iterative_maxmin_cuadrado_cpu", i, round(t, 4)])
        for n, runs in times_ft.items():
            for i, t in enumerate(runs):
                writer.writerow([n, n * 2, "ft.maxmin", i, round(t, 4)])

    return times_iterative, times_iterative_cpu, times_ft


def plot_comparison(times_iterative, times_iterative_cpu, times_ft):
    keys_iter     = sorted(times_iterative)
    keys_iter_cpu = sorted(times_iterative_cpu)
    keys_ft       = sorted(times_ft)

    xs_iter     = [n * 2 for n in keys_iter]
    xs_iter_cpu = [n * 2 for n in keys_iter_cpu]
    xs_ft       = [n * 2 for n in keys_ft]

    mean_iter     = [np.mean(times_iterative[n])     for n in keys_iter]
    std_iter      = [np.std(times_iterative[n])      for n in keys_iter]
    mean_iter_cpu = [np.mean(times_iterative_cpu[n]) for n in keys_iter_cpu]
    std_iter_cpu  = [np.std(times_iterative_cpu[n])  for n in keys_iter_cpu]
    mean_ft       = [np.mean(times_ft[n])            for n in keys_ft]
    std_ft        = [np.std(times_ft[n])             for n in keys_ft]

    _, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(xs_iter, mean_iter, yerr=std_iter,
                marker="o", label="iterative_maxmin_cuadrado (GPU)", capsize=4)
    ax.errorbar(xs_iter_cpu, mean_iter_cpu, yerr=std_iter_cpu,
                marker="^", label="iterative_maxmin_cuadrado (CPU)", capsize=4)
    ax.errorbar(xs_ft, mean_ft, yerr=std_ft,
                marker="s", label="ft.maxmin", capsize=4)

    ax.set_xscale("log")
    ax.set_yscale("log")

    all_xs = sorted(set(xs_iter) | set(xs_iter_cpu) | set(xs_ft))
    ax.set_xticks(all_xs)
    ax.set_xticklabels([str(x) for x in all_xs], rotation=45, ha="right")

    ax.set_xlabel("número de nodos (n×2)")
    ax.set_ylabel("tiempo (ms)")
    ax.set_title(f"Comparación de tiempos ({REPEATS} repeticiones por n)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("timing_comparison_log.png", dpi=150)
    plt.show()

    # --- escala real ---
    _, ax2 = plt.subplots(figsize=(9, 5))

    ax2.errorbar(xs_iter, mean_iter, yerr=std_iter,
                 marker="o", label="iterative_maxmin_cuadrado (GPU)", capsize=4)
    ax2.errorbar(xs_iter_cpu, mean_iter_cpu, yerr=std_iter_cpu,
                 marker="^", label="iterative_maxmin_cuadrado (CPU)", capsize=4)
    ax2.errorbar(xs_ft, mean_ft, yerr=std_ft,
                 marker="s", label="ft.maxmin", capsize=4)

    ax2.set_xticks(all_xs)
    ax2.set_xticklabels([str(x) for x in all_xs], rotation=45, ha="right")

    ax2.set_xlabel("número de nodos (n×2)")
    ax2.set_ylabel("tiempo (ms)")
    ax2.set_title(f"Comparación de tiempos — escala real ({REPEATS} repeticiones por n)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("timing_comparison_linear.png", dpi=150)
    plt.show()


times_iterative, times_iterative_cpu, times_ft = run_timing()
plot_comparison(times_iterative, times_iterative_cpu, times_ft)

