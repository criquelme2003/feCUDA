# Grafico de lineas comparando tiempos de ft.maxmin entre dos CSVs distintos

import csv
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

FUNCION = "ft.maxmin"

MARKERS = ["s", "o"]


def load_times(path, funcion=FUNCION):
    times = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["funcion"] != funcion:
                continue
            dim = int(row["dimension"])
            tiempo = float(row["tiempo_s"])
            times[dim].append(tiempo)
    return times


def plot_comparison(csv_path_a, csv_path_b, label_a=None, label_b=None, out_path="time_comparison_maxmin.png"):
    label_a = label_a or csv_path_a
    label_b = label_b or csv_path_b

    times_a = load_times(csv_path_a)
    times_b = load_times(csv_path_b)

    _, ax = plt.subplots(figsize=(9, 5))

    for (times, label, marker) in (
        (times_a, label_a, MARKERS[0]),
        (times_b, label_b, MARKERS[1]),
    ):
        dims = sorted(times)
        means = [np.mean(times[d]) for d in dims]
        stds = [np.std(times[d]) for d in dims]
        ax.errorbar(
            dims, means, yerr=stds,
            marker=marker,
            label=label,
            capsize=4,
        )

    all_dims = sorted(set(times_a) | set(times_b))
    ax.set_xticks(all_dims)
    ax.set_xticklabels([str(d) for d in all_dims])

    ax.set_xlabel("dimension")
    ax.set_ylabel("tiempo (s)")
    ax.set_title(f"Comparacion de tiempos: {FUNCION}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Uso: python {sys.argv[0]} <csv_a> <csv_b> [label_a] [label_b]")
        sys.exit(1)

    csv_a = sys.argv[1]
    csv_b = sys.argv[2]
    label_a = sys.argv[3] if len(sys.argv) > 3 else csv_a
    label_b = sys.argv[4] if len(sys.argv) > 4 else csv_b

    plot_comparison(csv_a, csv_b, label_a, label_b)
