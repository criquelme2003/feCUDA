#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("bench/results")

def main():
    if not ROOT.exists():
        print(f"[WARN] Carpeta {ROOT} no existe. No se genera index.json.")
        return

    index = {}

    # Buscar TODOS los metrics.csv dentro de bench/results/**
    for metrics in ROOT.rglob("metrics.csv"):
        rel_dir = metrics.parent.relative_to(ROOT)
        bench_id = str(rel_dir).replace("\\", "/")

        rel_path = metrics.relative_to(Path("."))
        rel_path_str = str(rel_path).replace("\\", "/")

        index[bench_id] = rel_path_str

    out_path = ROOT / "index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"[INFO] Generado index.json con {len(index)} benchmarks")

if __name__ == "__main__":
    main()
