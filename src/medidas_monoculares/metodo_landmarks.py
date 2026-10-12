"""Método 1: landmarks 3D métricos de MediaPipe (BlazePose GHUM).

La red regresa 33 articulaciones en metros con origen en la pelvis. La escala
no se mide: la red la infiere de la apariencia, así que tiende a la media de
la población con la que se entrenó. Es la línea base barata (CPU, 30 MB).
"""

from __future__ import annotations

import numpy as np

from src.medidas_monoculares.comun import (
    ARTICULACIONES,
    FRACCION_OREJA_VERTEX,
    ResultadoMetodo,
    largos_desde_articulaciones,
)
from src.medidas_monoculares.pose import Pose


def estatura_por_cadena(mundo: dict[str, np.ndarray]) -> float | None:
    """Suma de segmentos talón→tobillo→rodilla→cadera→hombros→orejas, en metros.

    Sumar segmentos en vez de restar alturas hace la medida independiente de
    que la persona esté algo inclinada respecto de la cámara.
    """
    lados = []
    for s in ("izq", "der"):
        nombres = [f"talon_{s}", f"tobillo_{s}", f"rodilla_{s}", f"cadera_{s}"]
        if all(n in mundo for n in nombres):
            lados.append(
                sum(np.linalg.norm(mundo[a] - mundo[b]) for a, b in zip(nombres, nombres[1:]))
            )
    tronco = ("hombro_izq", "hombro_der", "cadera_izq", "cadera_der", "oreja_izq", "oreja_der")
    if not lados or any(n not in mundo for n in tronco):
        return None
    hombros = (mundo["hombro_izq"] + mundo["hombro_der"]) / 2
    caderas = (mundo["cadera_izq"] + mundo["cadera_der"]) / 2
    orejas = (mundo["oreja_izq"] + mundo["oreja_der"]) / 2
    hasta_orejas = (
        float(np.mean(lados))
        + float(np.linalg.norm(hombros - caderas))
        + float(np.linalg.norm(orejas - hombros))
    )
    return hasta_orejas / (1.0 - FRACCION_OREJA_VERTEX)


def medir(pose: Pose) -> ResultadoMetodo:
    mundo = pose.visibles(pose.puntos_mundo)
    medidas = largos_desde_articulaciones({k: v for k, v in mundo.items() if k in ARTICULACIONES})
    avisos = list(pose.avisos)
    estatura = estatura_por_cadena(mundo)
    if estatura is None:
        avisos.append("estatura no estimada: faltan talones, caderas, hombros u orejas visibles")
    else:
        medidas["estatura"] = estatura * 100.0
    return ResultadoMetodo(
        metodo="landmarks",
        modelo="MediaPipe Pose Landmarker Heavy (pose_world_landmarks)",
        medidas_cm=medidas,
        avisos=avisos,
        puntos_2d={k: tuple(v) for k, v in pose.visibles(pose.puntos_px).items()},
    )
