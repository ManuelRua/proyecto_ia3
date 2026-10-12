"""Medidas sobre una malla corporal en pose de reposo (metros, eje Z vertical)."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from src.medidas_monoculares.comun import largos_desde_articulaciones

# Bandas de búsqueda de cada contorno, como fracción del tramo caderas→hombros
# (0 = centros de cadera, 1 = centros de hombro). El pecho se corta por debajo
# de la axila para que la sección no incluya los brazos.
BANDAS = {
    "contorno_cadera": (-0.20, 0.10, max),
    "contorno_cintura": (0.25, 0.50, min),
    "contorno_pecho": (0.60, 0.76, max),
}
CORTES_POR_BANDA = 7

# Huesos del esqueleto de Anny (MakeHuman) cuyo origen es cada articulación.
HUESO_DE_ARTICULACION = {
    "hombro": "upperarm01", "codo": "lowerarm01", "muneca": "wrist",
    "cadera": "upperleg01", "rodilla": "lowerleg01", "tobillo": "foot",
}


def articulaciones_anny(cuerpo, salida: dict) -> dict[str, np.ndarray]:
    """Articulaciones en reposo a partir de la salida del modelo corporal Anny."""
    origenes = salida["rest_bone_poses"][0, :, :3, 3].cpu().numpy().astype(np.float64)
    return {
        f"{articulacion}_{lado}": origenes[cuerpo.bone_labels.index(f"{hueso}.{sufijo}")]
        for articulacion, hueso in HUESO_DE_ARTICULACION.items()
        for lado, sufijo in (("izq", "L"), ("der", "R"))
    }


def perimetro_torso(mesh: trimesh.Trimesh, z: float, centro_xy: np.ndarray) -> float | None:
    """Perímetro de la envolvente convexa del lazo más cercano al eje del cuerpo.

    La envolvente convexa imita una cinta métrica, que no entra en las
    concavidades. Los brazos producen lazos aparte y quedan descartados.
    """
    seccion = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
    if seccion is None:
        return None
    lazos = [np.asarray(lazo)[:, :2] for lazo in seccion.discrete if len(lazo) >= 8]
    if not lazos:
        return None
    torso = min(lazos, key=lambda lazo: np.linalg.norm(lazo.mean(axis=0) - centro_xy))
    return float(ConvexHull(torso).area)  # en 2D, `area` es el perímetro


def medir_malla(
    vertices: np.ndarray, caras: np.ndarray, articulaciones: dict[str, np.ndarray]
) -> dict[str, float]:
    """Estatura, largos y contornos en cm."""
    medidas = largos_desde_articulaciones(articulaciones)
    # La malla puede traer vértices auxiliares sin caras (pelo, ropa de MakeHuman).
    medidas["estatura"] = float(np.ptp(vertices[np.unique(caras), 2])) * 100.0

    mesh = trimesh.Trimesh(vertices=vertices, faces=caras, process=False)
    caderas = (articulaciones["cadera_izq"] + articulaciones["cadera_der"]) / 2
    hombros = (articulaciones["hombro_izq"] + articulaciones["hombro_der"]) / 2
    tramo = hombros[2] - caderas[2]
    for nombre, (desde, hasta, elegir) in BANDAS.items():
        perimetros = [
            perimetro_torso(mesh, caderas[2] + f * tramo, caderas[:2])
            for f in np.linspace(desde, hasta, CORTES_POR_BANDA)
        ]
        perimetros = [p for p in perimetros if p is not None]
        if perimetros:
            medidas[nombre] = elegir(perimetros) * 100.0
    return medidas
