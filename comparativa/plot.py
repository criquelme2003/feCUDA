# Grafico de lineas comparando tiempos de ft.maxmin vs iterative_maxmin_cuadrado

import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "time_results_v2.csv"

LABELS = {
    "ft.maxmin": "ft.maxmin",
    "iterative_maxmin_cuadrado": "iterative_maxmin_cuadrado (fe)",
}
MARKERS = {
    "ft.maxmin": "s",
    "iterative_maxmin_cuadrado": "o",
}


def load_times(path=CSV_PATH):
    times = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim = int(row["dimension"])
            funcion = row["funcion"]
            tiempo = float(row["tiempo_s"])
            times[funcion][dim].append(tiempo)
    return times


def plot_comparison(times, out_path="time_comparisonv_median.png"):
    _, ax = plt.subplots(figsize=(9, 5))

    for funcion, per_dim in times.items():
        dims = sorted(per_dim)
        means = [np.median(per_dim[d]) for d in dims]
        stds = [np.std(per_dim[d]) for d in dims]
        ax.errorbar(
            dims, means, yerr=stds,
            marker=MARKERS.get(funcion, "o"),
            label=LABELS.get(funcion, funcion),
            capsize=4,
        )

    all_dims = sorted({d for per_dim in times.values() for d in per_dim})
    ax.set_xticks(all_dims)
    ax.set_xticklabels([str(d) for d in all_dims])

    ax.set_xlabel("dimension")
    ax.set_ylabel("tiempo (s)")
    ax.set_yscale("log")
    ax.set_title("Comparacion de tiempos: ft.maxmin vs iterative_maxmin_cuadrado")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    times = load_times()
    plot_comparison(times)
