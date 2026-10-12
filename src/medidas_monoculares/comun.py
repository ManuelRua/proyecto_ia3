"""Contrato común de los tres métodos y geometría compartida."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DIR_PRETRAINED = ROOT / "pretrained"
DIR_VENDOR = ROOT / "vendor"

# Definición de cada medida. Todas son distancias entre CENTROS articulares o
# sobre la malla, no recorridos de cinta: no son comparables 1:1 con BodyM.
MEDIDAS = {
    "estatura": "suelo a vértex (incluye pelo y calzado visibles)",
    "largo_brazo": "hombro → codo → muñeca",
    "largo_antebrazo": "codo → muñeca",
    "largo_pierna": "cadera → rodilla → tobillo",
    "ancho_hombros": "entre centros articulares de los hombros",
    "largo_torso": "centro de hombros → centro de caderas",
    "contorno_pecho": "solo malla: perímetro del torso bajo la axila",
    "contorno_cintura": "solo malla: perímetro mínimo entre pecho y cadera",
    "contorno_cadera": "solo malla: perímetro máximo a la altura de la pelvis",
}

# Fracción de la estatura entre el trago de la oreja y el vértex (≈13 cm en un
# adulto de 170 cm). Solo la usa el método de landmarks, que no ve la coronilla.
FRACCION_OREJA_VERTEX = 0.076

ARTICULACIONES = (
    "hombro_izq", "hombro_der", "codo_izq", "codo_der", "muneca_izq", "muneca_der",
    "cadera_izq", "cadera_der", "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der",
)


@dataclass
class ResultadoMetodo:
    metodo: str
    modelo: str
    medidas_cm: dict[str, float]
    avisos: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    # Puntos en píxeles de la imagen original, solo para dibujar.
    puntos_2d: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["medidas_cm"] = {k: round(float(v), 1) for k, v in self.medidas_cm.items()}
        data["puntos_2d"] = {
            k: [round(float(x), 1), round(float(y), 1)] for k, (x, y) in self.puntos_2d.items()
        }
        return data


def cargar_imagen(path: Path, lado_max: int = 1280) -> Image.Image:
    """Abre en RGB respetando la orientación EXIF y limita el lado mayor."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(image.size) > lado_max:
        image.thumbnail((lado_max, lado_max), Image.LANCZOS)
    return image


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _media_lados(valores: list[float | None]) -> float | None:
    validos = [v for v in valores if v is not None and np.isfinite(v)]
    return float(np.mean(validos)) if validos else None


def largos_desde_articulaciones(j: dict[str, np.ndarray]) -> dict[str, float]:
    """Largos en cm desde articulaciones 3D en metros.

    `j` puede omitir articulaciones (baja visibilidad): cada largo promedia los
    lados disponibles y se omite si no hay ninguno completo.
    """

    def cadena(*nombres: str) -> float | None:
        if any(n not in j for n in nombres):
            return None
        return sum(_dist(j[a], j[b]) for a, b in zip(nombres, nombres[1:]))

    medidas: dict[str, float | None] = {
        "largo_brazo": _media_lados(
            [cadena(f"hombro_{s}", f"codo_{s}", f"muneca_{s}") for s in ("izq", "der")]
        ),
        "largo_antebrazo": _media_lados(
            [cadena(f"codo_{s}", f"muneca_{s}") for s in ("izq", "der")]
        ),
        "largo_pierna": _media_lados(
            [cadena(f"cadera_{s}", f"rodilla_{s}", f"tobillo_{s}") for s in ("izq", "der")]
        ),
        "ancho_hombros": cadena("hombro_izq", "hombro_der"),
    }
    if all(n in j for n in ("hombro_izq", "hombro_der", "cadera_izq", "cadera_der")):
        centro_hombros = (np.asarray(j["hombro_izq"]) + np.asarray(j["hombro_der"])) / 2
        centro_caderas = (np.asarray(j["cadera_izq"]) + np.asarray(j["cadera_der"])) / 2
        medidas["largo_torso"] = _dist(centro_hombros, centro_caderas)
    return {k: v * 100.0 for k, v in medidas.items() if v is not None}


def elegir_dispositivo(preferido: str = "auto") -> str:
    import torch

    if preferido == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferido == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió cuda pero torch no ve ninguna GPU")
    return preferido


def liberar_memoria() -> None:
    """Entre métodos: con 4 GB de VRAM no caben dos modelos a la vez."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
