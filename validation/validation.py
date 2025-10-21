import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

RESULTS_DIR = Path("/home/carlos/feCUDA/validation/results")
REFERENCE_DIR = Path("/home/carlos/feCUDA/validation/data")

TOLERANCE_DECIMALS = 6  # ~1e-6 precision for value comparisons
VALUE_RANGE = (0.0, 1.0)  # expected range for energies/threshold outputs


def load_structured_entries(path: Path) -> List[Dict[str, Any]]:
    """Load a structured JSON file containing paths and values."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{path} debe contener una lista de entradas.")

    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(data):
        try:
            order = int(entry["order"])
            row = int(entry.get("row", idx))
            path_coords = [int(coord) for coord in entry["path"]]
            value = float(entry["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Entrada inválida en {path}, índice {idx}: {entry}") from exc

        normalized.append(
            {
                "order": order,
                "row": row,
                "path": path_coords,
                "value": value,
            }
        )

    return normalized


def quantize_value(value: float, decimals: int = TOLERANCE_DECIMALS) -> float:
    return round(value, decimals)


def entries_to_counter(entries: Iterable[Dict[str, Any]], decimals: int) -> Counter:
    key_iter = (
        (
            entry["order"],
            tuple(entry["path"]),
            quantize_value(entry["value"], decimals),
        )
        for entry in entries
    )
    return Counter(key_iter)


def compare_structured_files(result_path: Path, reference_path: Path) -> Tuple[bool, Dict[str, Counter]]:
    res_entries = load_structured_entries(result_path)
    ref_entries = load_structured_entries(reference_path)

    res_counter = entries_to_counter(res_entries, TOLERANCE_DECIMALS)
    ref_counter = entries_to_counter(ref_entries, TOLERANCE_DECIMALS)

    only_in_results = res_counter - ref_counter
    only_in_reference = ref_counter - res_counter

    ok = not only_in_results and not only_in_reference
    details = {
        "only_in_results": only_in_results,
        "only_in_reference": only_in_reference,
        "total_results": Counter({None: sum(res_counter.values())}),
        "total_reference": Counter({None: sum(ref_counter.values())}),
    }
    return ok, details


def summarize_counter(counter: Counter, limit: int = 5) -> List[str]:
    items = counter.most_common(limit)
    summary = []
    for (order, path, value), count in items:
        summary.append(
            f"order={order}, path={list(path)}, value≈{value}, count={count}"
        )
    if counter and len(counter) > limit:
        summary.append(f"... ({len(counter) - limit} coincidencias adicionales)")
    return summary


def sanity_check_bootstrap(paths_file: Path) -> List[str]:
    """Run simple consistency checks for bootstrap outputs."""
    issues: List[str] = []
    entries = load_structured_entries(paths_file)

    for idx, entry in enumerate(entries):
        order = entry["order"]
        path = entry["path"]
        value = entry["value"]

        if len(path) != order + 4:
            issues.append(
                f"{paths_file.name}: entrada {idx} tiene longitud de camino {len(path)} "
                f"pero orden {order} (esperado {order + 4})."
            )

        if any(coord < 0 for coord in path):
            issues.append(
                f"{paths_file.name}: entrada {idx} contiene coordenadas negativas {path}."
            )

        if not math.isfinite(value):
            issues.append(
                f"{paths_file.name}: entrada {idx} tiene valor no finito ({value})."
            )

        min_val, max_val = VALUE_RANGE
        if value < min_val - 1e-6 or value > max_val + 1e-6:
            issues.append(
                f"{paths_file.name}: entrada {idx} valor fuera de rango [{min_val}, {max_val}] -> {value}."
            )

    return issues


def validate_non_bootstrap():
    """Compare structured outputs between GPU results and reference data."""
    result_files = sorted(
        f for f in RESULTS_DIR.glob("paths_values_*.json") if "bootstrap" not in f.name
    )

    summary = {"passed": [], "failed": [], "missing_reference": []}

    for result_file in result_files:
        reference_file = REFERENCE_DIR / result_file.name

        if not reference_file.exists():
            summary["missing_reference"].append(result_file.name)
            print(f"[WARN] Referencia faltante para {result_file.name}")
            continue

        ok, details = compare_structured_files(result_file, reference_file)

        if ok:
            summary["passed"].append(result_file.name)
            print(f"[OK] {result_file.name} validado correctamente.")
        else:
            summary["failed"].append(result_file.name)
            print(f"[FAIL] {result_file.name} difiere de la referencia.")

            only_in_results = details["only_in_results"]
            if only_in_results:
                print("  > Presente solo en resultados:")
                for line in summarize_counter(only_in_results):
                    print(f"    - {line}")

            only_in_reference = details["only_in_reference"]
            if only_in_reference:
                print("  > Presente solo en referencia:")
                for line in summarize_counter(only_in_reference):
                    print(f"    - {line}")

    return summary


def validate_bootstrap():
    """Run sanity checks for bootstrap result files."""
    bootstrap_files = sorted(RESULTS_DIR.glob("paths_values_bootstrap_*.json"))
    issues_total: List[str] = []

    for bootstrap_file in bootstrap_files:
        issues = sanity_check_bootstrap(bootstrap_file)
        if issues:
            issues_total.extend(issues)
            print(f"[WARN] Se encontraron inconsistencias en {bootstrap_file.name}:")
            for issue in issues[:5]:
                print(f"    - {issue}")
            if len(issues) > 5:
                print(f"    ... ({len(issues) - 5} adicionales)")
        else:
            print(f"[OK] {bootstrap_file.name} pasó las validaciones básicas.")

    return issues_total


def main():
    print("Validando resultados determinísticos...")
    summary = validate_non_bootstrap()
    print()

    if summary["passed"]:
        print(f"✔ Archivos validados: {len(summary['passed'])}")
    if summary["failed"]:
        print(f"✖ Archivos con diferencias: {len(summary['failed'])}")
    if summary["missing_reference"]:
        print(f"⚠ Referencias faltantes: {len(summary['missing_reference'])}")

    print("\nVerificando salidas de bootstrap...")
    bootstrap_issues = validate_bootstrap()
    if bootstrap_issues:
        print(f"\nTotal de advertencias bootstrap: {len(bootstrap_issues)}")
    else:
        print("\nBootstrap sin inconsistencias detectadas.")


if __name__ == "__main__":
    main()
