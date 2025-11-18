#!/usr/bin/env python3
"""Simple benchmarking harness for FeCUDA."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - opcional
    psutil = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
RESULTS_ROOT = REPO_ROOT / "bench" / "results"
STATIC_DEFINITION_PATH = REPO_ROOT / "bench" / "definitions" / "static_dimensions.json"


def load_definition(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"La definición {path} no contiene 'cases' válidos")
    return cases


def ensure_results_dir(tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = RESULTS_ROOT / f"{timestamp}_{tag}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def run_cmd(cmd: List[str], *, env: Optional[Dict[str, str]] = None, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=text, env=env)


def collect_gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"collected_at": datetime.now().isoformat()}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = run_cmd([
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"
            ])
            name, driver, mem_total = [token.strip() for token in result.stdout.strip().split(',')]
            info.update({
                "gpu_name": name,
                "driver_version": driver,
                "memory_total": mem_total,
            })
        except Exception as exc:  # pragma: no cover - dependiente del entorno
            info["gpu_info_error"] = str(exc)
    else:
        info["gpu_info_error"] = "nvidia-smi no disponible"

    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = run_cmd([nvcc, "--version"])
            info["nvcc_version"] = result.stdout.strip().splitlines()[-1]
        except Exception as exc:  # pragma: no cover
            info["nvcc_version_error"] = str(exc)
    return info


class ResourceMonitor:
    def __init__(self, pid: int, interval: float = 0.01) -> None:
        self.pid = pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_peak: Optional[float] = None
        self.rss_peak: Optional[int] = None
        self.gpu_utils: List[float] = []
        self.gpu_mem_used: List[float] = []
        self._cpu_thread: Optional[threading.Thread] = None
        self._gpu_thread: Optional[threading.Thread] = None
        self.nvidia_smi = shutil.which("nvidia-smi")
        self._use_psutil = psutil is not None
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.clock_ticks = os.sysconf("SC_CLK_TCK")
        self.num_cpus = os.cpu_count() or 1

    def start(self) -> None:
        if self._use_psutil:
            self._cpu_thread = threading.Thread(target=self._cpu_loop_psutil, daemon=True)
            self._cpu_thread.start()
        else:
            self._cpu_thread = threading.Thread(target=self._cpu_loop_proc, daemon=True)
            self._cpu_thread.start()
        if self.nvidia_smi:
            self._gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
            self._gpu_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._cpu_thread:
            self._cpu_thread.join()
        if self._gpu_thread:
            self._gpu_thread.join()

    def _cpu_loop_psutil(self) -> None:
        try:
            proc = psutil.Process(self.pid)  # type: ignore[arg-type]
        except Exception:
            return
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            return
        while not self.stop_event.wait(self.interval):
            try:
                cpu = proc.cpu_percent(interval=None)
                mem = proc.memory_info().rss
            except Exception:
                break
            self.cpu_peak = max(self.cpu_peak or 0.0, cpu)
            self.rss_peak = max(self.rss_peak or 0, mem)

    def _cpu_loop_proc(self) -> None:
        last_proc_time: Optional[float] = None
        last_wall: Optional[float] = None
        while not self.stop_event.wait(self.interval):
            try:
                with open(f"/proc/{self.pid}/stat", "r") as fh:
                    fields = fh.read().split()
                utime = float(fields[13])
                stime = float(fields[14])
                rss_pages = int(fields[23])
            except Exception:
                break
            proc_time = (utime + stime) / self.clock_ticks
            now = time.perf_counter()
            if last_proc_time is not None and last_wall is not None:
                delta_proc = proc_time - last_proc_time
                delta_time = now - last_wall
                if delta_time > 0:
                    cpu = (delta_proc / delta_time) * 100.0 / self.num_cpus
                    self.cpu_peak = max(self.cpu_peak or 0.0, cpu)
            last_proc_time = proc_time
            last_wall = now
            rss = rss_pages * self.page_size
            self.rss_peak = max(self.rss_peak or 0, rss)

    def _gpu_loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                result = run_cmd([
                    self.nvidia_smi,
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits"
                ])
                line = result.stdout.strip().splitlines()[0]
                util_str, mem_str = [token.strip() for token in line.split(',')]
                self.gpu_utils.append(float(util_str))
                self.gpu_mem_used.append(float(mem_str))
            except Exception:
                break

    def snapshot(self) -> Dict[str, Optional[float]]:
        rss_mb = (self.rss_peak or 0) / (1024.0 * 1024.0) if self.rss_peak else None
        gpu_util_avg = statistics.mean(self.gpu_utils) if self.gpu_utils else None
        gpu_util_peak = max(self.gpu_utils) if self.gpu_utils else None
        gpu_mem_peak = max(self.gpu_mem_used) if self.gpu_mem_used else None
        return {
            "cpu_peak_percent": self.cpu_peak,
            "rss_peak_mb": rss_mb,
            "gpu_util_avg": gpu_util_avg,
            "gpu_util_peak": gpu_util_peak,
            "gpu_mem_peak_mb": gpu_mem_peak,
        }


def parse_bench_output(text: str) -> Dict[str, Any]:
    iterations: List[float] = []
    summary: Dict[str, Any] = {}

    def parse_fields(segment: str) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for token in segment.strip().split():
            if '=' in token:
                key, value = token.split('=', 1)
                fields[key.strip()] = value.strip()
        return fields

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("BENCH_ITER"):
            fields = parse_fields(line[len("BENCH_ITER"):])
            elapsed = fields.get("elapsed_ms")
            if elapsed:
                iterations.append(float(elapsed))
        elif line.startswith("BENCH_SUMMARY"):
            summary = parse_fields(line[len("BENCH_SUMMARY"):])
    return {"iterations": iterations, "summary": summary}


def parse_bench_errors(text: str) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("BENCH_ERROR"):
            data: Dict[str, str] = {}
            payload = line[len("BENCH_ERROR"):].strip()
            for token in shlex.split(payload):
                if "=" in token:
                    key, value = token.split("=", 1)
                    data[key.strip()] = value.strip().strip('"')
            errors.append(data)
    return errors


def run_case(
    executable: Path,
    case: Dict[str, Any],
    results_dir: Path,
    *,
    env: Dict[str, str],
    enable_ncu: bool,
    monitor_interval: float,
) -> Dict[str, Any]:
    synthetic = case.get("synthetic", False)
    dataset_value = case["dataset"]
    dataset_arg = dataset_value
    dataset_display = dataset_value
    if not synthetic:
        dataset_path = Path(dataset_value)
        if not dataset_path.is_absolute():
            dataset_path = (REPO_ROOT / dataset_value).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
        dataset_arg = str(dataset_path)
        dataset_display = str(dataset_path)

    label = case["label"]
    cmd = [
        str(executable),
        "--bench",
        "--dataset", dataset_arg,
        "--batch", str(case["batch"]),
        "--M", str(case["M"]),
        "--N", str(case["N"]),
        "--threshold", str(case.get("threshold", 0.2)),
        "--order", str(case.get("order", 3)),
        "--replicas", str(case.get("replicas", 100)),
        "--iterations", str(case.get("iterations", 1)),
        "--label", label,
    ]
    if synthetic:
        cmd.append("--synthetic")
        seed = case.get("seed")
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
    pattern = case.get("pattern")
    if pattern:
        cmd.extend(["--pattern", pattern])
    if case.get("skip_bootstrap"):
        cmd.append("--skip-bootstrap")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    monitor = ResourceMonitor(proc.pid, interval=monitor_interval)
    monitor.start()
    stdout, stderr = proc.communicate()
    monitor.stop()

    (results_dir / f"{label}_stdout.log").write_text(stdout)
    if stderr.strip():
        (results_dir / f"{label}_stderr.log").write_text(stderr)

    if proc.returncode != 0:
        errors = parse_bench_errors(stdout) + parse_bench_errors(stderr)
        if errors:
            error_messages = "; ".join(f"{err.get('reason', 'unknown')}: {err.get('message', '')}" for err in errors)
            raise RuntimeError(f"Caso {label} falló (bench_error): {error_messages}")
        raise RuntimeError(f"Caso {label} falló con código {proc.returncode}")

    parsed = parse_bench_output(stdout)
    iterations = parsed["iterations"]
    if not iterations:
        raise RuntimeError(f"Caso {label} no retornó métricas BENCH_ITER")
    summary = parsed["summary"]

    stats = {
        "mean_ms": float(summary.get("mean_ms", statistics.mean(iterations))),
        "min_ms": float(summary.get("min_ms", min(iterations))),
        "max_ms": float(summary.get("max_ms", max(iterations))),
        "std_ms": float(summary.get("std_ms", statistics.pstdev(iterations) if len(iterations) > 1 else 0.0)),
    }

    resource_snapshot = monitor.snapshot()

    if enable_ncu:
        run_ncu_capture(cmd, results_dir, label, env)

    total_ms = stats["mean_ms"] * len(iterations)

    return {
        **stats,
        "total_ms": total_ms,
        **resource_snapshot,
        "label": label,
        "iterations": len(iterations),
        "dataset": dataset_display,
        "batch": case["batch"],
        "M": case["M"],
        "N": case["N"],
        "threshold": case.get("threshold", 0.2),
        "order": case.get("order", 3),
        "replicas": case.get("replicas", 100),
        "timestamp": datetime.now().isoformat(),
    }


def run_ncu_capture(cmd: List[str], results_dir: Path, label: str, env: Dict[str, str]) -> None:
    ncu = shutil.which("ncu")
    if not ncu:
        print("[WARN] ncu no disponible en el PATH, se omite captura", file=sys.stderr)
        return
    export_base = results_dir / f"{label}_ncu"
    log_file = results_dir / f"{label}_ncu.log"
    ncu_cmd = [
        ncu,
        "-f",
        "--target-processes",
        "all",
        "--set",
        "full",
        "--export",
        str(export_base),
        "--log-file",
        str(log_file),
    ] + cmd
    print(f"[INFO] Ejecutando NCU para {label}...", file=sys.stderr)
    try:
        subprocess.run(ncu_cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] NCU falló para {label}: {exc}", file=sys.stderr)


def write_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "label",
        "dataset",
        "batch",
        "M",
        "N",
        "threshold",
        "order",
        "replicas",
        "iterations",
        "mean_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "total_ms",
        "cpu_peak_percent",
        "rss_peak_mb",
        "gpu_util_avg",
        "gpu_util_peak",
        "gpu_mem_peak_mb",
        "timestamp",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dimension_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    summary: List[Dict[str, Any]] = []
    for dimension in ("batch", "M", "N"):
        groups: Dict[int, List[float]] = defaultdict(list)
        for row in rows:
            groups[int(row[dimension])].append(float(row["mean_ms"]))
        for value, measures in groups.items():
            summary.append({
                "dimension": dimension,
                "value": value,
                "mean_ms": statistics.mean(measures),
            })
    if not summary:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dimension", "value", "mean_ms"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_benchmark_markdown(path: Path, tag: str, cases: List[Dict[str, Any]],     gpu_info: Dict[str, Any]) -> None:
    lines = [
        f"# Benchmark Report - {tag}",
        "",
        f"- Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- GPU: {gpu_info.get('gpu_name', 'desconocida')} (driver {gpu_info.get('driver_version', 'n/a')})",
        f"- nvcc: {gpu_info.get('nvcc_version', 'n/a')}",
        "",
        "## Casos",
        "",
        "| Label | Dataset | Batch | M | N | Threshold | Orden | Réplicas | Iteraciones |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        lines.append(
            f"| {case['label']} | {case['dataset']} | {case['batch']} | {case['M']} | {case['N']} | {case.get('threshold', 0.2)} | {case.get('order', 3)} | {case.get('replicas', 100)} | {case.get('iterations', 1)} |"
        )
    lines.extend(
        [
            "",
            "## Objetivo",
            "Describe aquí el objetivo del benchmark...",
            "",
            "## Observaciones",
            "- TODO",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner de benchmarks simples para FeCUDA")
    parser.add_argument("--tag", default="bench", help="Etiqueta del benchmark")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="Directorio de build (para encontrar fecuda_main)")
    parser.add_argument("--executable", type=Path, default=None, help="Ruta explícita de fecuda_main")
    parser.add_argument("--ncu", action="store_true", help="Capturar reporte NCU por caso")
    parser.add_argument("--monitor-interval", type=float, default=0.01, help="Intervalo de muestreo para CPU/GPU (s)")
    args = parser.parse_args()

    definition_path = STATIC_DEFINITION_PATH
    if not definition_path.exists():
        raise FileNotFoundError(f"Definición estática no encontrada: {definition_path}")
    cases = load_definition(definition_path)
    tag = args.tag or definition_path.stem

    executable = args.executable or (args.build_dir / "fecuda_main")
    if not executable.exists():
        raise FileNotFoundError(f"No se encontró el ejecutable: {executable}")

    results_dir = ensure_results_dir(tag)
    gpu_info = collect_gpu_info()
    (results_dir / "gpu_info.json").write_text(json.dumps(gpu_info, indent=2))
    write_benchmark_markdown(results_dir / "benchmark.md", tag, cases, gpu_info)

    env = os.environ.copy()
    env.setdefault("FECUDA_SOURCE_DIR", str(REPO_ROOT))

    if psutil is None:
        print("[WARN] psutil no está instalado; se usará monitor básico /proc", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for case in cases:
        print(f"[INFO] Ejecutando caso {case['label']}...")
        row = run_case(
            executable.resolve(),
            case,
            results_dir,
            env=env,
            enable_ncu=args.ncu,
            monitor_interval=args.monitor_interval,
        )
        rows.append(row)
        print(
            f"[OK] {case['label']} -> mean {row['mean_ms']:.3f} ms (cpu {row.get('cpu_peak_percent') or 0:.1f}%)"
        )

    write_metrics_csv(rows, results_dir / "metrics.csv")
    write_dimension_summary(rows, results_dir / "dimension_summary.csv")
    print(f"Resultados guardados en {results_dir}")


if __name__ == "__main__":
    main()
