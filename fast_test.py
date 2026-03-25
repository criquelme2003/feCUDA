import forgethreads as ft
import numpy as np
from numba import njit


rng = np.random.default_rng()
ft.set_verbose(False)


@njit(cache=True, fastmath=True)
def _flat_to_rc(sel, n, m, rows, cols):
    """
    Mapea índices planos [0, total_possible) → (row, col) en M sin
    construir la tabla completa de índices.

    Partición:
      [0,        n*(n-1))           → bloque A (n×n) fuera de diagonal
      [n*(n-1),  n*(n-1) + n*m)     → bloque B (n×m)
      [n*(n-1) + n*m, total)        → bloque D (m×m) fuera de diagonal
    """
    nA = n * (n - 1)
    nB = n * m
    n1 = n - 1 if n > 1 else 1   # guarda contra /0 cuando n==1 (nA=0)
    m1 = m - 1 if m > 1 else 1

    for i in range(len(sel)):
        f = sel[i]
        if f < nA:                         # ── bloque A
            r = f // n1
            c = f % n1
            if c >= r:
                c += 1
            rows[i] = r
            cols[i] = c
        elif f < nA + nB:                  # ── bloque B
            f -= nA
            rows[i] = f // m
            cols[i] = n + f % m
        else:                              # ── bloque D
            f -= nA + nB
            r  = f // m1
            c  = f % m1
            if c >= r:
                c += 1
            rows[i] = n + r
            cols[i] = n + c


def build_matrix(n, m, d, seed=None):
    rng = np.random.default_rng(seed)
    N   = n + m
    h   = np.float32(0.5)

    # ── 1. Bloques directamente en float32 (evita float64 intermedio) ─
    M = np.zeros((N, N), dtype=np.float32)
    M[:n, :n] = rng.random((n, n), dtype=np.float32) * h   # [0, 0.5)
    M[n:, n:] = rng.random((m, m), dtype=np.float32) * h
    M[:n, n:] = rng.random((n, m), dtype=np.float32) * h

    # ── 2. Diagonales = 1 sobre la vista (sin copias) ─────────────────
    np.fill_diagonal(M[:n, :n], 1.0)
    np.fill_diagonal(M[n:, n:], 1.0)

    # ── 3. Aristas extra ───────────────────────────────────────────────
    # current_edges es siempre 0 → todos los valores generados son < 0.5
    total_possible = n * (n - 1) + n * m + m * (m - 1)
    target_edges   = min(int(round(d * n)), total_possible)

    if target_edges > 0:
        # choice usa Floyd's sampling → O(k) cuando k << total
        sel  = rng.choice(total_possible, size=target_edges, replace=False)
        rows = np.empty(target_edges, dtype=np.int64)
        cols = np.empty(target_edges, dtype=np.int64)
        _flat_to_rc(sel, n, m, rows, cols)                  # JIT compilado
        M[rows, cols] = rng.random(target_edges, dtype=np.float32) * h + h

    return M.reshape(1, N, N).astype(np.float16)


# ── Warm-up: compilar el JIT antes del bucle principal ────────────────
def warmup():
    build_matrix(4, 2, 1.0, seed=0)
    
    
warmup()

mtx = build_matrix(10000, 10000, np.log(10000), seed=0)

print(mtx.shape)