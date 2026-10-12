"""MediaPipe Pose: landmarks 2D, landmarks 3D métricos y máscara de persona."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.medidas_monoculares.comun import DIR_PRETRAINED
from src.perception.mediapipe_pose import diagnostico_pose

MODELO_POSE = DIR_PRETRAINED / "mediapipe" / "pose_landmarker_heavy.task"

# Índices BlazePose de las articulaciones que usan los métodos.
INDICES = {
    "nariz": 0, "oreja_izq": 7, "oreja_der": 8,
    "hombro_izq": 11, "hombro_der": 12, "codo_izq": 13, "codo_der": 14,
    "muneca_izq": 15, "muneca_der": 16, "cadera_izq": 23, "cadera_der": 24,
    "rodilla_izq": 25, "rodilla_der": 26, "tobillo_izq": 27, "tobillo_der": 28,
    "talon_izq": 29, "talon_der": 30, "punta_izq": 31, "punta_der": 32,
}
VISIBILIDAD_MINIMA = 0.5


@dataclass
class Pose:
    puntos_px: dict[str, np.ndarray]  # (x, y) en píxeles
    puntos_mundo: dict[str, np.ndarray]  # (x, y, z) en metros, origen en la pelvis
    visibilidad: dict[str, float]
    mascara: np.ndarray  # (H, W) bool
    avisos: list[str]

    def visibles(self, puntos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: v for k, v in puntos.items() if self.visibilidad[k] >= VISIBILIDAD_MINIMA}


class DetectorPose:
    def __init__(self, model_path: Path = MODELO_POSE):
        import mediapipe as mp

        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Falta {model_path}. Ejecuta scripts/medidas_monoculares/descargar_modelos.py"
            )
        self.mp = mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def detectar(self, image: Image.Image) -> Pose:
        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
        result = self.landmarker.detect(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.pose_landmarks:
            raise ValueError("MediaPipe no detectó una persona completa")
        landmarks = result.pose_landmarks[0]
        mundo = result.pose_world_landmarks[0]
        ancho, alto = image.size
        mascara = np.squeeze(result.segmentation_masks[0].numpy_view()) > 0.5
        return Pose(
            puntos_px={
                k: np.array([landmarks[i].x * ancho, landmarks[i].y * alto]) for k, i in INDICES.items()
            },
            puntos_mundo={
                k: np.array([mundo[i].x, mundo[i].y, mundo[i].z]) for k, i in INDICES.items()
            },
            visibilidad={k: float(landmarks[i].visibility) for k, i in INDICES.items()},
            mascara=np.ascontiguousarray(mascara),
            avisos=list(diagnostico_pose(landmarks)["warnings"]),
        )
