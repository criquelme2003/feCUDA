#!/usr/bin/env python3
"""
Lanza barridos de tamaños de grafo para los tres regímenes y genera gráficos
de eta_0 vs N a partir de results/graph_sim.csv.

Uso típico:
    python scripts/graph_sweep.py --fecuda-bin ./build/fecuda_main
        --min-n 64 --max-n 4096 --count 20

Solo graficar (usando resultados existentes):
    python scripts/graph_sweep.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "graph_sim.csv"
DEFAULT_PNG = REPO_ROOT / "results" / "graph_sweep.png"


@dataclass
class RegimeConfig:
    regime: str
    avg_degree: float = 0.0
    max_degree: float = 4.0
    dense_p: float = 0.2
    epsilon: float = 0.6


def geom_sizes(min_n: int, max_n: int, count: int) -> List[int]:
    if count <= 1 or min_n <= 0 or max_n <= 0:
        return [min_n]
    sizes: List[int] = []
    for i in range(count):
        t = i / (count - 1)
        val = int(round(math.exp(math.log(min_n) + (math.log(max_n) - math.log(min_n)) * t)))
        sizes.append(val)
    # dedup + sort
    return sorted(list(dict.fromkeys(sizes)))


def run_regime(
    fecuda_bin: Path,
    sizes: List[int],
    label_prefix: str,
    cfg: RegimeConfig,
    reps_per_n: int,
) -> None:
    """
    Ejecuta el binario principal en modo --simulate-graphs para un conjunto de N,
    usando MST y grafos sintéticos. reps_per_n controla cuántas veces se repite
    el barrido completo para promediar luego.
    """
    scale_str = ",".join(str(x) for x in sizes)
    for rep in range(reps_per_n):
        label = f"{label_prefix}_r{rep}"
        cmd = [
            str(fecuda_bin),
            "--simulate-graphs",
            "--graph",
            "--use-mst",
            "--regime",
            cfg.regime,
            "--epsilon",
            str(cfg.epsilon),
            "--batch",
            "1",
            "--scale-N",
            scale_str,
            "--order",
            "3",
            "--threshold",
            "0.2",
            "--iterations",
            "1",
            "--replicas",
            "1",
            "--label",
            label,
        ]
        if cfg.regime == "dense":
            cmd += ["--dense-p", str(cfg.dense_p)]
        else:
            cmd += ["--avg-degree", str(cfg.avg_degree), "--max-degree", str(cfg.max_degree)]
        print(f"[INFO] Ejecutando {cfg.regime} rep={rep} N={sizes}")
        subprocess.run(cmd, check=True)


def load_results(csv_path: Path, labels: List[str], allow_prefix: bool = True) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}")
    rows: List[Dict[str, str]] = []
    label_set = set(labels)
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lbl = row.get("label", "")
            if lbl in label_set or (allow_prefix and any(lbl.startswith(prefix) for prefix in label_set)):
                rows.append(row)
    return rows


def plot_results(rows: List[Dict[str, str]], png_path: Path, min_samples: int = 1) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib no está disponible; no se generan gráficos.")
        return

    if not rows:
        print("[WARN] No hay filas para graficar.")
        return

    def to_num(row: Dict[str, str], key: str, default: float = math.nan) -> float:
        try:
            return float(row.get(key, default))
        except Exception:
            return default

    # Filtrar filas con muestras suficientes
    filtered = [r for r in rows if to_num(r, "eta_samples", 0) >= min_samples]
    if not filtered:
        print("[WARN] No hay filas con muestras suficientes para graficar.")
        return

    # Agrupar por régimen y N para promediar sobre réplicas/iteraciones
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in filtered:
        regime = row["regime"]
        n_val = int(row["N"])
        grouped[regime][n_val].append(to_num(row, "eta0_mean"))

    regimes = sorted(grouped.keys())
    fig, axes = plt.subplots(2, len(regimes), figsize=(5 * len(regimes), 7), sharex=False, sharey=False)
    if len(regimes) == 1:
        axes = [axes[0], axes[1]]

    for col, regime in enumerate(regimes):
        ns = sorted(grouped[regime].keys())
        xs = ns
        logx = [math.log(x) for x in xs]
        ys = []
        yerr = []
        for n_val in ns:
            vals = grouped[regime][n_val]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ys.append(mean_val)
            yerr.append(std_val)

        ax1 = axes[0] if len(regimes) == 1 else axes[0][col]
        ax1.errorbar(xs, ys, yerr=yerr, fmt="-o", capsize=3, label=f"{regime} (eta0_mean ± std)")
        ax1.set_title(regime)
        ax1.set_xlabel("N")
        ax1.set_xscale("log")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        ax2 = axes[1] if len(regimes) == 1 else axes[1][col]
        ax2.errorbar(logx, ys, yerr=yerr, fmt="-o", capsize=3, label=f"{regime} vs logN")
        ax2.set_xlabel("log N")
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)

        if len(logx) >= 2:
            coeffs = np.polyfit(logx, ys, 1)
            a, b = coeffs
            fit_y = [a * lx + b for lx in logx]
            ax2.plot(logx, fit_y, "--", label=f"fit: a={a:.3f}")
            ax2.legend()

    if len(regimes) == 1:
        axes[0].set_ylabel("eta0_mean vs N")
        axes[1].set_ylabel("eta0_mean vs log N")
    else:
        axes[0][0].set_ylabel("eta0_mean vs N")
        axes[1][0].set_ylabel("eta0_mean vs log N")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    print(f"[INFO] Gráfico guardado en {png_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Barrido y gráficos de grafos sintéticos.")
    parser.add_argument("--fecuda-bin", type=Path, default=REPO_ROOT / "build" / "fecuda_main")
    parser.add_argument("--min-n", type=int, default=20, help="N mínimo del barrido")
    parser.add_argument("--max-n", type=int, default=2000, help="N máximo del barrido")
    parser.add_argument("--count", type=int, default=30, help="Número de tamaños de N a probar")
    parser.add_argument("--reps-per-n", type=int, default=3, help="Número de réplicas por N")
    parser.add_argument("--plot-only", action="store_true", help="Solo lee CSV y grafica")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument(
        "--regimes",
        type=str,
        default="sparse,supercritical,dense",
        help="Tipos de grafos a simular: sparse,supercritical,dense (separados por coma)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Prefijos de labels para filtrar el CSV (por defecto se usan los basados en regimes)",
    )
    args = parser.parse_args()

    sizes = geom_sizes(args.min_n, args.max_n, args.count)
    regimes_requested = [r.strip() for r in args.regimes.split(",") if r.strip()]
    if not regimes_requested:
        print("[ERROR] Debes especificar al menos un régimen en --regimes")
        return 1

    if args.labels:
        labels = [lbl.strip() for lbl in args.labels.split(",") if lbl.strip()]
    else:
        labels = [f"{r}_mst" for r in regimes_requested]

    if not args.plot_only:
        regimes_cfg: List[Tuple[str, RegimeConfig]] = []
        for r in regimes_requested:
            if r == "sparse":
                regimes_cfg.append((f"{r}_mst", RegimeConfig(regime="sparse", avg_degree=0.5, epsilon=0.6)))
            elif r == "supercritical":
                regimes_cfg.append((f"{r}_mst", RegimeConfig(regime="supercritical", avg_degree=3.0, max_degree=5.0, epsilon=0.6)))
            elif r == "dense":
                regimes_cfg.append((f"{r}_mst", RegimeConfig(regime="dense", dense_p=0.2, epsilon=0.6)))
            else:
                print(f"[WARN] Régimen desconocido ignorado: {r}")

        for label_prefix, cfg in regimes_cfg:
            run_regime(
                args.fecuda_bin,
                sizes,
                label_prefix,
                cfg,
                reps_per_n=args.reps_per_n,
            )

    rows = load_results(args.csv, labels, allow_prefix=True)
    plot_results(rows, args.png, min_samples=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
