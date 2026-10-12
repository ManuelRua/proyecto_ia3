"""Método 2: geometría métrica monocular (MoGe-2) + landmarks 2D.

MoGe-2 predice, desde una sola foto y sin calibración, un mapa de puntos 3D en
metros y el campo de visión de la cámara. Aquí se leen esos puntos sobre la
persona: los largos son distancias entre los puntos 3D de cada articulación y
la estatura es la extensión del cuerpo a lo largo de su propio eje.

Los puntos son de la SUPERFICIE visible (ropa incluida), no del centro de la
articulación; el sesgo es de pocos milímetros en largos porque ambos extremos
se desplazan en la misma dirección.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.medidas_monoculares.comun import (
    ARTICULACIONES,
    DIR_PRETRAINED,
    ResultadoMetodo,
    largos_desde_articulaciones,
)
from src.medidas_monoculares.pose import Pose

MODELO_MOGE = "MoGe-2 ViT-S (Ruicheng/moge-2-vits-normal)"
PESOS_MOGE = DIR_PRETRAINED / "moge" / "moge-2-vits-normal.pt"
# El borde de la silueta mezcla profundidad de la persona y del fondo.
EROSION_PX = 2
# La máscara de MediaPipe se desborda sobre el fondo: solo cuenta como persona
# lo que está a esta distancia (m) de su profundidad mediana.
MARGEN_PROFUNDIDAD_M = 0.6


class EstimadorGeometria:
    def __init__(self, dispositivo: str, pesos: Path = PESOS_MOGE):
        import torch
        from moge.model.v2 import MoGeModel

        if not Path(pesos).is_file():
            raise FileNotFoundError(
                f"Falta {pesos}. Ejecuta scripts/medidas_monoculares/descargar_modelos.py"
            )
        self.torch = torch
        self.dispositivo = dispositivo
        self.model = MoGeModel.from_pretrained(str(pesos)).to(dispositivo).eval()

    def inferir(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, float]:
        """Devuelve puntos (H, W, 3) en metros, máscara válida y FOV horizontal en grados."""
        torch = self.torch
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        with torch.inference_mode():
            out = self.model.infer(
                tensor.to(self.dispositivo), use_fp16=self.dispositivo == "cuda"
            )
        puntos = out["points"].float().cpu().numpy()
        valido = out["mask"].cpu().numpy().astype(bool) & np.isfinite(puntos).all(axis=-1)
        fx = float(out["intrinsics"][0, 0])  # normalizada por el ancho
        return puntos, valido, math.degrees(2 * math.atan(0.5 / fx))


def punto_3d(puntos: np.ndarray, mascara: np.ndarray, xy: np.ndarray, radio: int) -> np.ndarray | None:
    """Mediana de los puntos 3D de la persona en una ventana alrededor de un píxel."""
    alto, ancho = mascara.shape
    x, y = int(round(xy[0])), int(round(xy[1]))
    if not (0 <= x < ancho and 0 <= y < alto):
        return None
    y0, y1 = max(0, y - radio), min(alto, y + radio + 1)
    x0, x1 = max(0, x - radio), min(ancho, x + radio + 1)
    ventana = puntos[y0:y1, x0:x1][mascara[y0:y1, x0:x1]]
    if len(ventana) < 3:
        return None
    return np.median(ventana, axis=0)


def extension_sobre_eje(nube: np.ndarray, eje: np.ndarray) -> float:
    """Extensión robusta de una nube de puntos a lo largo de un eje unitario."""
    proyeccion = nube @ eje
    bajo, alto = np.percentile(proyeccion, [0.1, 99.9])
    return float(alto - bajo)


def medir_desde_puntos(puntos: np.ndarray, valido: np.ndarray, pose: Pose) -> ResultadoMetodo:
    persona = pose.mascara & valido
    kernel = np.ones((2 * EROSION_PX + 1, 2 * EROSION_PX + 1), np.uint8)
    nucleo = cv2.erode(persona.astype(np.uint8), kernel).astype(bool)
    avisos = list(pose.avisos)
    if nucleo.any():
        z = puntos[..., 2]
        nucleo &= np.abs(z - np.median(z[nucleo])) <= MARGEN_PROFUNDIDAD_M
    if nucleo.sum() < 500:
        return ResultadoMetodo("profundidad", MODELO_MOGE, {}, avisos + ["silueta demasiado pequeña"])

    filas = np.where(nucleo.any(axis=1))[0]
    alto_px = int(filas[-1] - filas[0] + 1)
    radio = max(3, int(0.012 * alto_px))

    visibles = pose.visibles(pose.puntos_px)
    articulaciones = {}
    for nombre, xy in visibles.items():
        p = punto_3d(puntos, nucleo, xy, radio)
        if p is not None:
            articulaciones[nombre] = p
    medidas = largos_desde_articulaciones(
        {k: v for k, v in articulaciones.items() if k in ARTICULACIONES}
    )

    arriba = [articulaciones[k] for k in ("hombro_izq", "hombro_der") if k in articulaciones]
    abajo = [articulaciones[k] for k in ("tobillo_izq", "tobillo_der") if k in articulaciones]
    if arriba and abajo:
        eje = np.mean(arriba, axis=0) - np.mean(abajo, axis=0)
        eje /= np.linalg.norm(eje)
        extension = extension_sobre_eje(puntos[nucleo], eje)
        # La erosión recorta EROSION_PX arriba y abajo; se devuelve esa franja.
        medidas["estatura"] = extension * (1.0 + 2 * EROSION_PX / alto_px) * 100.0
    else:
        avisos.append("estatura no estimada: hombros o tobillos sin profundidad válida")

    return ResultadoMetodo(
        metodo="profundidad",
        modelo=MODELO_MOGE,
        medidas_cm=medidas,
        avisos=avisos,
        extras={"distancia_camara_m": round(float(np.median(puntos[nucleo][:, 2])), 2)},
        puntos_2d={k: tuple(v) for k, v in visibles.items() if k in articulaciones},
    )
