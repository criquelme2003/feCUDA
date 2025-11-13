import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
REFERENCE_DIR = BASE_DIR / "data"

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

if __name__ == "__main__":
    main()
