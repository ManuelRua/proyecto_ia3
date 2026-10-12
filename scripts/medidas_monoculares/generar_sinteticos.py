"""Genera fotos sintéticas de cuerpos Anny con medidas reales conocidas.

No hace falta ningún dataset: cada cuerpo se muestrea del modelo paramétrico
Anny, se dibuja con una cámara conocida y sus medidas se toman de la malla con
las mismas definiciones que usa el método `malla`. Junto a cada PNG queda un
JSON con las medidas reales, que demo.py muestra en la columna "real".

    python scripts/medidas_monoculares/generar_sinteticos.py --cantidad 4

Son maniquíes sin ropa ni textura sobre un fondo dibujado: sirven para
comprobar que todo corre y que la geometría es coherente, no para medir la
precisión que tendrán los métodos con fotos reales.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.medidas_monoculares.medidas_malla import articulaciones_anny, medir_malla

ANCHO, ALTO, FOCAL = 720, 1280, 1100.0
ALTURA_CAMARA_M = 1.0
FONDO_PARED_M = 1.6  # distancia de la persona a la pared del fondo
PIEL = np.array([150, 170, 205], dtype=np.float64)  # BGR
LUZ = np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8])


def a_camara(puntos: np.ndarray, distancia: float) -> np.ndarray:
    """Mundo (x, y fondo, z arriba; suelo en z=0) → cámara (x derecha, y abajo, z al frente)."""
    return np.stack(
        [puntos[..., 0], ALTURA_CAMARA_M - puntos[..., 2], puntos[..., 1] + distancia], axis=-1
    )


def proyectar(camara: np.ndarray) -> np.ndarray:
    return np.stack(
        [FOCAL * camara[..., 0] / camara[..., 2] + ANCHO / 2, FOCAL * camara[..., 1] / camara[..., 2] + ALTO / 2],
        axis=-1,
    )


def dibujar_fondo(distancia: float) -> np.ndarray:
    """Pared, suelo con baldosas de 50 cm y una puerta de 2,03 m: referencias de escala."""
    imagen = np.full((ALTO, ANCHO, 3), (214, 221, 226), np.uint8)

    def poligono(puntos_mundo, color):
        px = proyectar(a_camara(np.asarray(puntos_mundo, dtype=np.float64), distancia))
        cv2.fillPoly(imagen, [np.round(px).astype(np.int32)], color)

    y_pared = FONDO_PARED_M
    poligono([(-6, -distancia + 0.3, 0), (6, -distancia + 0.3, 0), (6, y_pared, 0), (-6, y_pared, 0)], (150, 158, 164))
    for i, x in enumerate(np.arange(-6, 6, 0.5)):
        for k, y in enumerate(np.arange(-distancia + 0.5, y_pared, 0.5)):
            if (i + k) % 2 == 0:
                poligono([(x, y, 0), (x + 0.5, y, 0), (x + 0.5, min(y + 0.5, y_pared), 0), (x, min(y + 0.5, y_pared), 0)], (128, 137, 144))
    poligono([(0.9, y_pared, 0), (1.72, y_pared, 0), (1.72, y_pared, 2.03), (0.9, y_pared, 2.03)], (96, 124, 150))
    return imagen


def dibujar_cuerpo(imagen: np.ndarray, vertices: np.ndarray, caras: np.ndarray, distancia: float) -> None:
    """Algoritmo del pintor con sombreado de Lambert: basta para una prueba de humo."""
    camara = a_camara(vertices, distancia)
    px = proyectar(camara)
    triangulos = camara[caras]
    normales = np.cross(triangulos[:, 1] - triangulos[:, 0], triangulos[:, 2] - triangulos[:, 0])
    normales /= np.linalg.norm(normales, axis=1, keepdims=True) + 1e-12
    brillo = 0.35 + 0.65 * np.abs(normales @ LUZ)
    for i in np.argsort(-triangulos[:, :, 2].mean(axis=1)):
        cv2.fillConvexPoly(
            imagen, np.round(px[caras[i]]).astype(np.int32), (PIEL * brillo[i]).tolist(), lineType=cv2.LINE_AA
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cantidad", type=int, default=4)
    parser.add_argument("--semilla", type=int, default=7)
    parser.add_argument("--salida", default="outputs/medidas_monoculares/sinteticos")
    args = parser.parse_args()

    import anny
    import torch

    with contextlib.redirect_stdout(io.StringIO()):  # Anny lista cientos de archivos
        cuerpo = anny.create_fullbody_model(all_phenotypes=True).to(dtype=torch.float32)
    caras = cuerpo.get_triangular_faces().numpy()
    salida = ROOT / args.salida
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    for i in range(args.cantidad):
        fenotipo = {
            "gender": rng.uniform(0, 1), "age": rng.uniform(0.55, 0.8), "muscle": rng.uniform(0.3, 0.7),
            "weight": rng.uniform(0.25, 0.75), "height": rng.uniform(0.1, 0.9), "proportions": rng.uniform(0.3, 0.7),
        }
        with torch.inference_mode():
            reposo = cuerpo(phenotype_kwargs={k: torch.tensor([v], dtype=torch.float32) for k, v in fenotipo.items()})
        vertices = reposo["rest_vertices"][0].numpy().astype(np.float64)
        medidas = medir_malla(vertices, caras, articulaciones_anny(cuerpo, reposo))

        # Anny mira hacia -y: se apoya en el suelo y la cámara se coloca delante.
        en_suelo = vertices - [0.0, 0.0, vertices[np.unique(caras), 2].min()]
        distancia = float(rng.uniform(2.3, 3.0))
        imagen = dibujar_fondo(distancia)
        dibujar_cuerpo(imagen, en_suelo, caras, distancia)

        nombre = f"sintetico_{i:02d}"
        cv2.imwrite(str(salida / f"{nombre}.png"), imagen)
        (salida / f"{nombre}.json").write_text(
            json.dumps(
                {
                    "medidas_cm": {k: round(v, 1) for k, v in medidas.items()},
                    "fenotipo_anny_0a1": {k: round(float(v), 3) for k, v in fenotipo.items()},
                    "camara": {"distancia_m": round(distancia, 2), "altura_m": ALTURA_CAMARA_M, "focal_px": FOCAL},
                },
                indent=2,
            )
        )
        print(f"{nombre}: estatura {medidas['estatura']:.1f} cm · brazo {medidas['largo_brazo']:.1f} cm · a {distancia:.2f} m")
    print(f"Sintéticos en {salida}")


if __name__ == "__main__":
    main()
