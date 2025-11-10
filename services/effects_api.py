"""API REST para generar efectos iterativos usando la implementación CUDA.

Ejecuta el servicio con:
    uvicorn services.effects_api:app --host 0.0.0.0 --port 8000

Configura la variable de entorno EFFECTS_API_LIB si la biblioteca compartida
`libeffects_api.so` no se encuentra en `build/`.
"""

import ctypes
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator


# -----------------------------------------------------------------------------
# Carga dinámica de la biblioteca compartida
# -----------------------------------------------------------------------------


def _load_library() -> ctypes.CDLL:
    default_path = Path(__file__).resolve().parent.parent / "build" / "libeffects_api.so"
    lib_path = Path(os.environ.get("EFFECTS_API_LIB", default_path))
    if not lib_path.exists():
        raise RuntimeError(
            f"No se encontró la biblioteca compartida en {lib_path}. "
            "Compila el proyecto (cmake --build build) o define EFFECTS_API_LIB."
        )
    return ctypes.CDLL(str(lib_path))


_LIB = _load_library()

_LIB.generate_effects_json.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
]
_LIB.generate_effects_json.restype = ctypes.c_void_p

_LIB.free_effects_json.argtypes = [ctypes.c_void_p]
_LIB.free_effects_json.restype = None

_LIB.effects_last_error.restype = ctypes.c_char_p


def _call_cuda_backend(
    tensor: np.ndarray,
    threshold: float,
    order: int,
    bootstrap_replicas: int,
) -> dict:
    pointer = tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = _LIB.generate_effects_json(
        pointer,
        ctypes.c_int(tensor.shape[0]),
        ctypes.c_int(tensor.shape[1]),
        ctypes.c_int(tensor.shape[2]),
        ctypes.c_float(threshold),
        ctypes.c_int(order),
        ctypes.c_int(bootstrap_replicas),
    )

    if not result_ptr:
        error_ptr = _LIB.effects_last_error()
        message = error_ptr.decode("utf-8") if error_ptr else "Error desconocido en el backend CUDA."
        raise RuntimeError(message)

    try:
        json_bytes = ctypes.string_at(result_ptr)
        data = json.loads(json_bytes.decode("utf-8"))
    finally:
        _LIB.free_effects_json(result_ptr)

    if not isinstance(data, dict) or "effects" not in data:
        raise RuntimeError("Respuesta inválida desde el backend CUDA.")

    return data


# -----------------------------------------------------------------------------
# Validación de entrada / modelos Pydantic
# -----------------------------------------------------------------------------


class EffectsRequest(BaseModel):
    tensor: List[List[List[float]]] = Field(
        ..., description="Tensor 3D con forma [batch, M, N]"
    )
    threshold: float = Field(..., ge=0.0, le=1.0)
    order: int = Field(..., ge=2, description="Orden máximo de efectos a buscar")
    bootstrap_replicas: int = Field(
        0,
        ge=0,
        description="Cantidad de replicas bootstrap a generar (0 para desactivar).",
    )

    @validator("tensor")
    def _validate_tensor(cls, value: List[List[List[float]]]) -> List[List[List[float]]]:
        if not value:
            raise ValueError("El tensor no puede estar vacío.")
        batch = len(value)
        m = len(value[0])
        n = len(value[0][0])
        for b in value:
            if len(b) != m:
                raise ValueError("Todas las muestras deben tener la misma cantidad de filas.")
            for row in b:
                if len(row) != n:
                    raise ValueError("Todas las filas deben tener la misma cantidad de columnas.")
        return value


class EffectsMetrics(BaseModel):
    total_processing_ms: float
    algorithm_ms: float
    bootstrap_ms: float
    bootstrap_replicas: int
    gpu_memory_free_before_mb: float
    gpu_memory_free_after_mb: float
    gpu_memory_delta_mb: float


class EffectsResponse(BaseModel):
    effects: List[dict]
    total_entries: int
    metrics: EffectsMetrics


# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------


app = FastAPI(title="Effects Generation API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",      # el host donde sirves el frontend
        "http://127.0.0.1:8000",
        "https://tu-app.vercel.app"   # añade los dominios que necesites
    ],
    allow_methods=["*"],           # o ["*"] si prefieres
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).resolve().parent / "web"
if _STATIC_DIR.exists():
    app.mount("/app", StaticFiles(directory=_STATIC_DIR), name="app")


@app.get("/", response_class=HTMLResponse, tags=["ui"])
def ui_root() -> HTMLResponse:
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="La interfaz web no está disponible.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health", tags=["health"])
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/effects", response_model=EffectsResponse, tags=["effects"])
def generate_effects(payload: EffectsRequest) -> EffectsResponse:
    tensor_np = np.asarray(payload.tensor, dtype=np.float32)
    if tensor_np.ndim != 3:
        raise HTTPException(status_code=422, detail="El tensor debe tener tres dimensiones [batch, M, N].")

    try:
        result = _call_cuda_backend(
            tensor_np.copy(order="C"),
            float(payload.threshold),
            payload.order,
            payload.bootstrap_replicas,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EffectsResponse(**result)


__all__ = ["app", "generate_effects"]
