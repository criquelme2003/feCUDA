#!/usr/bin/env python3
"""
Genera gráficas comparativas a partir del metrics.csv producido por fecuda_bench.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt


PLOTS = [
    ("elapsed_ms", "Tiempo total (ms)", "tiempo_total.png"),
    ("cpu_total_percent", "CPU proceso total (%)", "cpu_total.png"),
    ("cpu_per_core_percent", "CPU promedio por núcleo (%)", "cpu_por_nucleo.png"),
    ("ram_delta_mb", "RAM usada por corrida (MB)", "ram_delta.png"),
    ("ram_after_mb", "RAM total proceso (MB)", "ram_total.png"),
    ("gpu_util_percent", "GPU util promedio (%)", "gpu_util.png"),
    ("gpu_mem_percent", "GPU memoria promedio (%)", "gpu_mem_prom.png"),
    ("gpu_mem_peak_mb", "GPU memoria pico (MB)", "gpu_mem_pico.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera gráficas a partir de un metrics.csv de fecuda_bench"
    )
    parser.add_argument(
        "report_dir",
        type=Path,
        help="Carpeta generada por fecuda_bench (contiene summary.txt y metrics.csv)",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("metrics.csv"),
        help="Ruta al CSV (por defecto report_dir/metrics.csv)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Resolución para las imágenes exportadas",
    )
    return parser.parse_args()


def _parse_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def load_metrics(csv_path: Path) -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = {"label": row["label"]}
            for key in reader.fieldnames or []:
                if key == "label":
                    continue
                parsed[key] = _parse_float(row.get(key, ""))
            rows.append(parsed)
    return rows


def plot_metric(
    rows: List[Dict[str, Optional[float]]],
    metric: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    labels: List[str] = []
    values: List[float] = []
    for row in rows:
        val = row.get(metric)
        if val is None:
            continue
        labels.append(row["label"])
        values.append(val)

    if not labels:
        print(f"[INFO] Métrica '{metric}' sin datos. Se omite la gráfica.")
        return

    width = max(6.0, len(labels) * 0.7)
    plt.figure(figsize=(width, 4.5))
    plt.bar(range(len(labels)), values, color="#2563eb")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"[OK] {metric} -> {output_path}")


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.expanduser().resolve()

    metrics_path = args.metrics_file
    if not metrics_path.is_absolute():
        metrics_path = report_dir / metrics_path
    metrics_path = metrics_path.resolve()

    if not metrics_path.exists():
        raise SystemExit(f"No se encontró el CSV en {metrics_path}")

    rows = load_metrics(metrics_path)
    if not rows:
        raise SystemExit(f"El CSV {metrics_path} está vacío.")

    for metric, ylabel, filename in PLOTS:
        output_path = report_dir / filename
        plot_metric(rows, metric, ylabel, output_path, args.dpi)


if __name__ == "__main__":
    main()
