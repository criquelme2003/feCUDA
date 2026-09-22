#!/usr/bin/env python3
"""
Replica los gráficos de mean L(n, c) vs c, en facetas por n, a partir de
output/sweep_n_block_merged_order.csv.

Uso:
    python3 plot_sweep_n_block.py [ruta_csv] [ruta_salida_png]

Por defecto lee ../output/sweep_n_block_merged_order.csv (relativo a este
script) y escribe sweep_n_block_order.png en esta misma carpeta.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "output" / "sweep_n_block_merged_order.csv"
DEFAULT_OUT = HERE / "sweep_n_block_order.png"

# orden de las facetas (mismo orden que el gráfico original)
N_ORDER = [100, 250, 500, 1000, 2500, 5000, 10000]

# líneas verticales de referencia: bordes de los regímenes
VLINES = [1, 5 / 4, 2, 3]

# regímenes empíricos y su marcador asociado
# Régimen                  Intervalo         Marcador
# Baja conectividad        c < 1             o (círculo)
# Ventana de bloque M      1 <= c <= 5/4     ^ (triángulo)
# Interumbral              5/4 < c < 2       s (cuadrado)
# Ventana de bloque N      2 <= c <= 3       D (rombo)
# Dominado por atajos      c > 3             X (equis)
REGIMES = [
    ("Baja conectividad (c < 1)", lambda c: c < 1, "o"),
    ("Ventana de bloque M (1 ≤ c ≤ 5/4)", lambda c: 1 <= c <= 5 / 4, "^"),
    ("Interumbral (5/4 < c < 2)", lambda c: 5 / 4 < c < 2, "s"),
    ("Ventana de bloque N (2 ≤ c ≤ 3)", lambda c: 2 <= c <= 3, "D"),
    ("Dominado por atajos (c > 3)", lambda c: c > 3, "X"),
]


def marker_for_c(c):
    for _, pred, marker in REGIMES:
        if pred(c):
            return marker
    raise ValueError(f"c={c} no cae en ningún régimen definido")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    df = pd.read_csv(csv_path)
    required_cols = {"c", "n", "repeticion", "orden_efectivo"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # estadisticos por (n, c): media y error estandar de la media
    grouped = (
        df.groupby(["n", "c"])["orden_efectivo"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["sem"] = grouped["sem"].fillna(0)

    n_values = [n for n in N_ORDER if n in grouped["n"].unique()]
    n_missing = [n for n in N_ORDER if n not in grouped["n"].unique()]
    if n_missing:
        print(f"Aviso: no hay datos para n = {n_missing}, se omiten esas facetas.")

    ncols = 4
    nrows = int(np.ceil(len(n_values) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, n in enumerate(n_values):
        ax = axes_flat[idx]
        sub = grouped[grouped["n"] == n].sort_values("c")

        # linea de union
        ax.plot(sub["c"], sub["mean"], color="black", linewidth=0.6, zorder=1)

        # barras de error
        ax.errorbar(
            sub["c"],
            sub["mean"],
            yerr=sub["sem"],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
            zorder=2,
        )

        # puntos, con marcador segun rango de c
        for _, row in sub.iterrows():
            ax.scatter(
                row["c"],
                row["mean"],
                marker=marker_for_c(row["c"]),
                s=35,
                facecolor="black",
                edgecolor="black",
                zorder=3,
            )

        # lineas verticales de referencia
        for v in VLINES:
            ax.axvline(v, color="black", linestyle="--", linewidth=0.6, zorder=0)

        ax.set_xscale("log")
        ax.set_title(str(n), backgroundcolor="0.85", fontsize=10)

        # ticks en el eje x con las fracciones/enteros del barrido
        c_ticks = sorted(grouped["c"].unique())
        ax.set_xticks(c_ticks)
        ax.set_xticklabels(
            [format_c(c) for c in c_ticks], rotation=90, fontsize=6
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())

        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, which="major", axis="both", color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)

    # ocultar ejes sobrantes
    for idx in range(len(n_values), len(axes_flat)):
        axes_flat[idx].axis("off")

    # leyenda global de regímenes
    legend_handles = [
        Line2D(
            [], [],
            marker=marker,
            linestyle="none",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            label=label,
        )
        for label, _, marker in REGIMES
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.supxlabel("c", fontsize=11, y=0.06)
    fig.supylabel("mean L(n, c)", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.10, 1, 1))

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out_path}")


def format_c(c):
    """Formatea c como fraccion simple (1/8, 1/4, ...) o entero, igual al original."""
    from fractions import Fraction

    frac = Fraction(c).limit_denominator(8)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


if __name__ == "__main__":
    main()
