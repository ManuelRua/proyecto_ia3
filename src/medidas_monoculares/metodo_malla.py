"""Método 3: malla corporal paramétrica (Multi-HMR + modelo Anny).

Multi-HMR regresa, desde una foto, la pose y el fenotipo (estatura, peso,
sexo, edad, proporciones...) del modelo corporal Anny. Las medidas se toman
sobre la malla en pose de reposo, así que no dependen de la postura de la foto
ni de la cámara, y la estatura es la del cuerpo (sin pelo ni calzado).

Anny es Apache 2.0; el código y el checkpoint de Multi-HMR tienen licencia NO
COMERCIAL de NAVER, por eso viven en vendor/ y pretrained/ y no se versionan.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.medidas_monoculares.comun import DIR_PRETRAINED, DIR_VENDOR, ResultadoMetodo
from src.medidas_monoculares.medidas_malla import articulaciones_anny, medir_malla

CHECKPOINT = DIR_PRETRAINED / "multihmr" / "multiHMR_672_L_anny.pt"
VENDOR_MULTI_HMR = DIR_VENDOR / "multi-hmr"
NOMBRE_MODELO = "Multi-HMR 672 ViT-L + Anny"
IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FENOTIPOS_PREDICHOS = ("age", "gender", "weight", "height", "muscle", "proportions")

def _registrar_vendor() -> None:
    """Hace importable `multi_hmr_anny` sin arrastrar el renderizador del demo.

    El paquete `utils` de Multi-HMR importa pyrender, smplx y pyvista al
    cargarse. El modelo solo necesita cuatro funciones, así que se registra un
    `utils` mínimo con ellas.
    """
    if "multi_hmr_anny" in sys.modules:
        return
    if not (VENDOR_MULTI_HMR / "multi_hmr_anny").is_dir():
        raise FileNotFoundError(
            f"Falta {VENDOR_MULTI_HMR}. Ejecuta scripts/medidas_monoculares/descargar_modelos.py"
        )
    import torch

    def cargar(nombre: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(
            f"_multihmr_{nombre}", VENDOR_MULTI_HMR / "utils" / f"{nombre}.py"
        )
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        return modulo

    def rotation_to_homogeneous(x):
        homogenea = torch.eye(4, dtype=x.dtype, device=x.device).expand(x.shape[:-2] + (4, 4)).clone()
        homogenea[..., :3, :3] = x
        return homogenea

    camara, imagen = cargar("camera"), cargar("image")
    utils = types.ModuleType("utils")
    utils.unpatch = imagen.unpatch
    utils.inverse_perspective_projection = camara.inverse_perspective_projection
    utils.perspective_projection = camara.perspective_projection
    utils.rotation_to_homogeneous = rotation_to_homogeneous
    sys.modules["utils"] = utils
    sys.path.insert(0, str(VENDOR_MULTI_HMR))


class EstimadorMalla:
    def __init__(self, dispositivo: str, checkpoint: Path = CHECKPOINT):
        import torch

        if not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"Falta {checkpoint}. Ejecuta scripts/medidas_monoculares/descargar_modelos.py"
            )
        _registrar_vendor()
        from multi_hmr_anny.multi_hmr import Multi_HMR

        self.torch = torch
        self.dispositivo = dispositivo
        # El checkpoint guarda un argparse.Namespace: exige weights_only=False.
        # Solo se carga el archivo oficial de NAVER. mmap evita duplicar 1,5 GB en RAM.
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        # pretrained_backbone=True descargaría 1,1 GB de DINOv2 que el checkpoint pisa.
        argumentos = {**vars(ckpt["args"]), "pretrained_backbone": False}
        with contextlib.redirect_stdout(io.StringIO()):  # Anny lista cientos de archivos
            self.model = Multi_HMR(**argumentos)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        del ckpt
        self.model = self.model.to(dispositivo).eval()
        cuerpo = self.model.body_model
        self.caras = cuerpo.get_triangular_faces().cpu().numpy()
        self.vertices_con_cara = np.unique(self.caras)

    def _preparar(self, image: Image.Image):
        """Redimensiona con relleno al cuadrado de entrada; devuelve escala y desplazamiento."""
        lado = self.model.img_size
        escala = lado / max(image.size)
        contenida = ImageOps.contain(image, (lado, lado))
        desplazamiento = ((lado - contenida.width) / 2, (lado - contenida.height) / 2)
        cuadrada = ImageOps.pad(contenida, (lado, lado))
        x = (np.asarray(cuadrada, dtype=np.float32) / 255.0 - IMG_NORM_MEAN) / IMG_NORM_STD
        tensor = self.torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.dispositivo)
        return tensor, escala, desplazamiento

    def medir(self, image: Image.Image) -> tuple[ResultadoMetodo, np.ndarray | None]:
        """Devuelve el resultado y los vértices proyectados (N, 2) en píxeles de la imagen."""
        torch = self.torch
        x, escala, desplazamiento = self._preparar(image)
        with torch.inference_mode(), torch.autocast(self.dispositivo, enabled=self.dispositivo == "cuda"):
            personas = self.model(x, is_training=False, nms_kernel_size=3, det_thresh=0.3)
        if not personas:
            return ResultadoMetodo("malla", NOMBRE_MODELO, {}, ["no se detectó ninguna persona"]), None
        persona = personas[0]  # la más cercana a la cámara
        avisos = [f"{len(personas)} personas detectadas; se mide la más cercana"] if len(personas) > 1 else []

        cuerpo = self.model.body_model
        forma = persona["shape"].float()
        fenotipo = {
            k: forma[None, i] for i, k in enumerate(cuerpo.phenotype_labels) if k in FENOTIPOS_PREDICHOS
        }
        with torch.inference_mode():
            reposo = cuerpo(phenotype_kwargs=fenotipo)
        vertices = reposo["rest_vertices"][0].cpu().numpy().astype(np.float64)
        medidas = medir_malla(vertices, self.caras, articulaciones_anny(cuerpo, reposo))

        K = persona["K"].float()
        v3d = persona["v3d"].float()[self.vertices_con_cara]
        v2d = (v3d @ K.T)[:, :2] / v3d[:, 2:3].clamp(min=1e-3)
        v2d = (v2d.cpu().numpy() - np.asarray(desplazamiento)) / escala
        resultado = ResultadoMetodo(
            metodo="malla",
            modelo=NOMBRE_MODELO,
            medidas_cm=medidas,
            avisos=avisos,
            extras={
                "fenotipo_anny_0a1": {k: round(float(v), 3) for k, v in fenotipo.items()},
                "distancia_camara_m": round(float(persona["transl_pelvis"].flatten()[2]), 2),
                "fov_grados": round(float(np.degrees(persona["fov"].flatten()[0].cpu())), 1),
            },
        )
        return resultado, v2d
