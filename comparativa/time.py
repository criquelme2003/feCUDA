# Comparativa de tiempos entre maxmin de forgethreads y forgeffects

import time
import csv

import forgethreads as ft

from sparse_generator import sparse_supercritical_block_matrix_julius as generator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from forgeffects_modules.iterative_maxmin_cuadrado import iterative_maxmin_cuadrado

C = 1
SEED = 101
ORDER = 3
THRESHOLD = 0.5
REPEATS = 10

DIMENSIONS = [150,300,600,700,900,1200]


def run_ft(dim, seed):
    n = dim // 2
    m1, _, _ = generator(n, n, C, seed)
    m2 = m1.copy()
    t0 = time.time()
    ft.maxmin(m1, m2, THRESHOLD, ORDER)
    return time.time() - t0


def run_fe(dim, seed):
    n = dim // 2
    m, _, _ = generator(n, n, C, seed)
    t0 = time.time()
    iterative_maxmin_cuadrado(m, THRESHOLD, ORDER)
    return time.time() - t0


def main():
    with open("time_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "funcion", "repeticion", "tiempo_s"])
        t_ft = run_ft(100, 101)
        t_fe = run_fe(100, 101)
        for dim in DIMENSIONS:
            for rep in range(REPEATS):
                t_ft = run_ft(dim, SEED + rep)
                writer.writerow([dim, "ft.maxmin", rep, t_ft])
                f.flush()
                print(f"dim={dim} ft rep={rep} tiempo={t_ft:.6f}s")

            for rep in range(REPEATS):
                t_fe = run_fe(dim, SEED + rep)
                writer.writerow([dim, "iterative_maxmin_cuadrado", rep, t_fe])
                f.flush()
                print(f"dim={dim} fe rep={rep} tiempo={t_fe:.6f}s")


if __name__ == "__main__":
    main()
