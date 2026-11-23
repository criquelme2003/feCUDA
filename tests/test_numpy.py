"""
Tests de la librería forgethreads usando tensores NumPy.
Valida correctitud, determinismo y explora límites de dimensiones.

Requisitos:
  - numpy >= 1.22 (soporte nativo __dlpack__)
  - módulo forgethreads compilado en /workspace
"""

import numpy as np
import sys
import time
import gc
import ctypes

ctypes.CDLL("libcudart.so")

sys.path.insert(0, '/workspace')
import forgethreads as ft

print(f"numpy {np.__version__}")
print(f"forgethreads cargado OK\n")

# ─── Referencia CPU exacta ────────────────────────────────────────────────────

MIN_DIFF_F32 = 0.01

def cpu_maxmin_count(A_f16, thr_f32):
    """Referencia CPU del algoritmo MaxMin.
    A_f16: ndarray float16 [B, M, N] con M == N.
    Retorna el número de paths que supera el threshold."""
    A = A_f16.astype(np.float32)
    B, M, N = A.shape
    assert M == N, "Se requieren matrices cuadradas M=N"
    count = 0
    for b in range(B):
        for m in range(M):
            for n in range(N):
                mins = np.minimum(A[b, m, :], A[b, :, n])  # [K]
                k_max = float(np.max(mins))
                orig  = float(A[b, m, n])
                if (k_max - orig) >= thr_f32:
                    for k in range(M):
                        mi = min(float(A[b, m, k]), float(A[b, k, n]))
                        if abs(mi - k_max) <= MIN_DIFF_F32:
                            count += 1
    return count


# ─── Wrapper GPU ──────────────────────────────────────────────────────────────

class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]
class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]
class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice),
                ("ndim", ctypes.c_int), ("dtype", DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]
class DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", DLTensor),
                ("manager_ctx", ctypes.c_void_p),
                ("deleter", ctypes.c_void_p)]

ctypes.pythonapi.PyCapsule_GetPointer.restype  = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _capsule_count(cap):
    """Lee el primer elemento del shape del DLManagedTensor de la cápsula."""
    ptr     = ctypes.pythonapi.PyCapsule_GetPointer(cap, b"dltensor")
    managed = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensor))
    return int(managed.contents.dl_tensor.shape[0])


def gpu_maxmin_count(A_f16, thr_f32):
    """Ejecuta forgethreads.maxmin con ndarray NumPy float16 [B,M,N].
    El módulo copia los datos CPU→GPU internamente.
    Retorna el número de paths encontrados."""
    A_c = np.ascontiguousarray(A_f16)
    # numpy 2.0+ tiene __dlpack__ nativo; forgethreads lo llama internamente
    paths_cap, values_cap = ft.maxmin(A_c, A_c, float(thr_f32), 1)
    count = _capsule_count(paths_cap)
    # Las capsulas liberan la memoria GPU cuando el GC las destruye
    del paths_cap, values_cap
    gc.collect()
    return count


def make_tensor(B, M, seed=42):
    """Genera tensor reproducible [B, M, M] float16 con valores en (0,1)."""
    rng = np.random.RandomState(seed)
    return rng.rand(B, M, M).astype(np.float16)


PASS = "\033[32mOK\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results_log = []

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}" + (f"  →  {detail}" if detail else ""))
    results_log.append((label, ok))


# ════════════════════════════════════════════════════════════════════════════
# TEST 1: Correctitud GPU vs CPU reference
# ════════════════════════════════════════════════════════════════════════════
print("="*60)
print("TEST 1: Correctitud GPU vs CPU reference")
print("="*60)

correctness_cases = [
    (1,  4, 0.30),
    (1,  8, 0.30),
    (1, 16, 0.30),
    (1, 32, 0.30),
    (1, 16, 0.00),
    (1, 16, 0.50),
    (1, 16, 0.90),
    (4, 16, 0.30),
    (8, 16, 0.30),
]

for B, M, thr in correctness_cases:
    A = make_tensor(B, M)
    cpu_c = cpu_maxmin_count(A, thr)
    gpu_c = gpu_maxmin_count(A, thr)
    check(f"B={B:2d} M={M:3d} thr={thr:.2f}",
          gpu_c == cpu_c,
          f"CPU={cpu_c}  GPU={gpu_c}")


# ════════════════════════════════════════════════════════════════════════════
# TEST 2: Determinismo entre runs
# ════════════════════════════════════════════════════════════════════════════
print()
print("="*60)
print("TEST 2: Determinismo (5 runs con misma entrada)")
print("="*60)

determ_cases = [(1, 16, 0.3), (1, 64, 0.3), (4, 32, 0.3), (8, 32, 0.3)]

for B, M, thr in determ_cases:
    A = make_tensor(B, M)
    counts = [gpu_maxmin_count(A, thr) for _ in range(5)]
    ok = len(set(counts)) == 1
    check(f"B={B:2d} M={M:3d} thr={thr:.1f}",
          ok,
          f"runs={counts}")


# ════════════════════════════════════════════════════════════════════════════
# TEST 3: Exploración de límites de dimensiones de tensores
# ════════════════════════════════════════════════════════════════════════════
print()
print("="*60)
print("TEST 3: Límites de dimensiones del tensor de entrada")
print("="*60)

def probe(B, M, thr=0.3):
    A = make_tensor(B, M)
    t0 = time.perf_counter()
    try:
        c = gpu_maxmin_count(A, thr)
        t_ms = (time.perf_counter() - t0) * 1000
        return True, f"count={c:6d}  t={t_ms:.1f}ms"
    except Exception as e:
        return False, str(e)[:60]

print("\n  — Variando M (B=1) —")
for M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    bmk = 1 * M * M * M
    path = "BATCHED" if bmk >= 100_000 else "COMPLETE"
    ok, info = probe(1, M)
    check(f"B=1   M={M:4d}  [{path:8s}]", ok, info)

print("\n  — Variando B (M=16) —")
for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    bmk = B * 16 * 16 * 16
    path = "BATCHED" if bmk >= 100_000 else "COMPLETE"
    ok, info = probe(B, 16)
    check(f"B={B:4d}  M=16  [{path:8s}]", ok, info)

print("\n  — Casos BATCHED grandes —")
for B, M in [(1, 64), (1, 128), (1, 256), (4, 64), (8, 64), (16, 64)]:
    bmk = B * M * M * M
    ok, info = probe(B, M)
    check(f"B={B:2d}  M={M:3d}  [B×M³={bmk:>9d}]", ok, info)


# ════════════════════════════════════════════════════════════════════════════
# TEST 4: Tiempos GPU vs CPU
# ════════════════════════════════════════════════════════════════════════════
print()
print("="*60)
print("TEST 4: Benchmark GPU vs CPU")
print("="*60)

bench_cases = [(1, 16), (1, 32), (1, 64), (1, 128), (4, 32), (8, 32)]
print(f"\n  {'Config':15s}  {'CPU (ms)':>10s}  {'GPU (ms)':>10s}  {'Speedup':>8s}  Paths")

for B, M in bench_cases:
    A = make_tensor(B, M)
    thr = 0.3

    t0 = time.perf_counter()
    cpu_c = cpu_maxmin_count(A, thr)
    t_cpu = (time.perf_counter() - t0) * 1000

    times_gpu = []
    for _ in range(3):
        t0 = time.perf_counter()
        gpu_c = gpu_maxmin_count(A, thr)
        times_gpu.append((time.perf_counter() - t0) * 1000)
    t_gpu = min(times_gpu)

    speedup = t_cpu / t_gpu if t_gpu > 0 else float('inf')
    print(f"  B={B:<2d} M={M:<4d}         {t_cpu:>10.2f}  {t_gpu:>10.2f}  {speedup:>7.1f}x  {gpu_c}")


# ════════════════════════════════════════════════════════════════════════════
# Resumen
# ════════════════════════════════════════════════════════════════════════════
print()
print("="*60)
passed = sum(1 for _, ok in results_log if ok)
failed = sum(1 for _, ok in results_log if not ok)
print(f"RESUMEN: {passed} OK, {failed} FAIL  ({len(results_log)} checks)")
print("="*60)

if failed > 0:
    print("\nChecks fallados:")
    for name, ok in results_log:
        if not ok:
            print(f"  ✗ {name}")
    sys.exit(1)
