"""
Reproduce las tablas del capítulo de resultados de la tesis a partir de los CSV
del experimento de profundidad de detención:

  1. Resumen por régimen empírico (media, Q.95, máximo).
  2. Contrastes ventana–valle Δ_M(n), Δ_N(n) con intervalos bootstrap al 95 %.
  3. Diagnósticos de descubrimiento por orden en n = 2500 (D, r̄_q, L).

Datos de entrada (por defecto, en este directorio o en la raíz del repo):
  - sweep_n_block_merged_order.csv     dataset del barrido (tope ORDER = 100)
  - sweep_10000n_9894c_order.csv       re-corrida de n = 10000, c ∈ {9/8, 9/4}
                                       con ORDER = 200; reemplaza las 5
                                       observaciones que tocaron el tope 100
  - effects_per_order_2500n.csv        conteos q_r por orden, n = 2500

Convenciones:
  - η0 = L = orden_efectivo + 1. El binding cuenta los pasos de propagación desde 1
    (primer producto E∘E), mientras que la tesis numera los órdenes de efecto desde 2.
  - Lo mismo para q_r: r = orden + 1.

Bootstrap (contrastes):
  - Para cada n, se remuestrean con reemplazo las 20 réplicas dentro de cada celda (n, c),
    se recalculan las medias por celda, la media de cada régimen (promedio de las medias
    de sus celdas) y los contrastes. Se usan B = 10.000 remuestreos, intervalos percentil
    2,5–97,5 y semilla fija SEED.

Uso:
    python3 tablas_profundidad.py
    python3 tablas_profundidad.py --guardar-merged sweep_n_block_merged_order_v2.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

B = 10_000
SEED = 20261123
TOPE_ANTERIOR = 100


# ---------------------------------------------------------------- utilidades
def buscar(nombre):
    for d in (HERE, ROOT):
        p = d / nombre
        if p.exists():
            return p
    raise FileNotFoundError(f"No se encontró {nombre} en {HERE} ni en {ROOT}")


def regimen(c):
    if c < 1:
        return "Baja conect."
    if c <= 5 / 4:
        return "Ventana M"
    if c < 2:
        return "Interumbral"
    if c <= 3:
        return "Ventana N"
    return "Atajos"


def coma(x, dec=1):
    """Número con coma decimal y punto de miles (formato de la tesis)."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def fraccion(c):
    tabla = {1: "1", 9 / 8: "9/8", 5 / 4: "5/4", 3 / 2: "3/2", 2: "2",
             9 / 4: "9/4", 5 / 2: "5/2", 11 / 4: "11/4", 3: "3"}
    for k, v in tabla.items():
        if np.isclose(c, k):
            return v
    return f"{c:g}"


# ---------------------------------------------------------------- datos
def cargar_barrido(merged_csv, recorrida_csv):
    ref = pd.read_csv(merged_csv)
    new = pd.read_csv(recorrida_csv)[["c", "n", "repeticion", "orden_efectivo"]]

    m = ref.merge(new, on=["c", "n", "repeticion"], how="left", suffixes=("", "_nuevo"))
    comparables = m[m["orden_efectivo_nuevo"].notna() & (m["orden_efectivo"] < TOPE_ANTERIOR)]
    distintas = (comparables["orden_efectivo"] != comparables["orden_efectivo_nuevo"]).sum()
    if distintas:
        raise ValueError(f"La re-corrida no reproduce {distintas} observaciones no censuradas")

    reemplazos = m["orden_efectivo_nuevo"].notna() & (m["orden_efectivo"] != m["orden_efectivo_nuevo"])
    m.loc[m["orden_efectivo_nuevo"].notna(), "orden_efectivo"] = m["orden_efectivo_nuevo"]
    m = m.drop(columns="orden_efectivo_nuevo")
    m["orden_efectivo"] = m["orden_efectivo"].astype(int)

    restantes = (m["orden_efectivo"] >= TOPE_ANTERIOR) & ~m.index.isin(m[reemplazos].index)
    print(f"Barrido: {len(m)} observaciones, {m.groupby(['n', 'c']).ngroups} celdas; "
          f"{len(comparables)} re-corridas idénticas, {reemplazos.sum()} reemplazadas.")
    if (m["orden_efectivo"] == TOPE_ANTERIOR).any():
        print("  Nota: hay observaciones con orden_efectivo = 100; las de la re-corrida "
              "(ORDER = 200) son valores reales, no censurados.")
    return m


# ---------------------------------------------------------------- tablas
def tabla_regimenes(d):
    d = d.assign(L=d["orden_efectivo"] + 1, r=d["c"].map(regimen))
    print("\n% Tabla: resumen por régimen")
    for r in ["Baja conect.", "Ventana M", "Interumbral", "Ventana N", "Atajos"]:
        s = d[d["r"] == r]
        celdas = s.groupby(["n", "c"]).ngroups
        print(f"{r:<13} & {celdas} & {coma(s['L'].mean(), 2)} "
              f"({coma(s['L'].quantile(.95), 1)}) & {s['L'].max()} \\\\")


def tabla_contrastes(d):
    d = d.assign(L=d["orden_efectivo"] + 1, r=d["c"].map(regimen))
    rng = np.random.default_rng(SEED)
    claves = {"M": "Ventana M", "I": "Interumbral", "N": "Ventana N"}

    def contrastes(medias):
        reg = {k: np.mean([v for c, v in medias.items() if regimen(c) == nombre])
               for k, nombre in claves.items()}
        return reg["M"] - reg["I"], reg["N"] - reg["I"]

    print(f"\n% Tabla: contrastes ventana–valle (bootstrap percentil 95 %, B = {B}, semilla {SEED})")
    for n, dn in d.groupby("n"):
        celdas = {c: g["L"].to_numpy() for c, g in dn.groupby("c")
                  if regimen(c) in claves.values()}
        punto = contrastes({c: v.mean() for c, v in celdas.items()})
        boot = np.array([
            contrastes({c: rng.choice(v, len(v)).mean() for c, v in celdas.items()})
            for _ in range(B)
        ])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        celda = lambda i: (f"${coma(punto[i]).replace(',', '{,}')}$ "
                           f"[${coma(lo[i]).replace(',', '{,}')}$; "
                           f"${coma(hi[i]).replace(',', '{,}')}$]")
        print(f"{coma(n, 0):<6} & {celda(0)} & {celda(1)} \\\\")


def tabla_descubrimientos(efectos_csv, barrido):
    e = pd.read_csv(efectos_csv)
    e = e[e["efectos"] > 0].assign(r=lambda x: x["orden"] + 1)

    # chequeos de consistencia
    por_rep = e.groupby(["c", "n", "repeticion"])
    assert (por_rep["orden"].max() == por_rep["orden_efectivo"].first()).all(), \
        "El último orden con efectos no coincide con orden_efectivo"
    chk = por_rep["orden_efectivo"].first().reset_index().merge(
        barrido, on=["n", "repeticion"], suffixes=("", "_ref"))
    chk = chk[np.isclose(chk["c"], chk["c_ref"])]
    dif = (chk["orden_efectivo"] != chk["orden_efectivo_ref"]).sum()
    print(f"\nDescubrimientos: {por_rep.ngroups} ejecuciones; "
          f"{len(chk)} comparadas con el barrido, {dif} diferencias.")

    print("\n% Tabla: diagnósticos de descubrimiento (n = 2500)")
    for c, g in e.groupby("c"):
        rep = g.groupby("repeticion").apply(
            lambda x: pd.Series({
                "D": x["efectos"].sum(),
                "rq": (x["r"] * x["efectos"]).sum() / x["efectos"].sum(),
                "L": x["orden_efectivo"].iloc[0] + 1,
            }), include_groups=False)
        print(f"${fraccion(c)}$ & {coma(rep['D'].mean() / 1e3, 1)} & "
              f"{coma(rep['rq'].mean(), 2)} & {coma(rep['L'].mean(), 2)} \\\\")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged", default="sweep_n_block_merged_order.csv")
    p.add_argument("--recorrida", default="sweep_10000n_9894c_order.csv")
    p.add_argument("--efectos", default="effects_per_order_2500n.csv")
    p.add_argument("--guardar-merged", metavar="CSV",
                   help="guarda el barrido corregido (mismo formato que el merged original)")
    a = p.parse_args()

    barrido = cargar_barrido(buscar(a.merged), buscar(a.recorrida))
    if a.guardar_merged:
        barrido.to_csv(HERE / a.guardar_merged, index=False)
        print(f"Guardado: {HERE / a.guardar_merged}")

    tabla_regimenes(barrido)
    tabla_contrastes(barrido)
    tabla_descubrimientos(buscar(a.efectos), barrido)


if __name__ == "__main__":
    main()
