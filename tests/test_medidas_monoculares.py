"""Geometría de src/medidas_monoculares con cuerpos sintéticos de medidas conocidas.

No carga ningún modelo: comprueba que, dadas articulaciones, puntos 3D o una
malla correctos, las medidas que se derivan son las esperadas.
"""

import numpy as np
import pytest

from src.medidas_monoculares.comun import FRACCION_OREJA_VERTEX, largos_desde_articulaciones
from src.medidas_monoculares.metodo_landmarks import estatura_por_cadena
from src.medidas_monoculares.pose import INDICES, Pose


def esqueleto(estatura: float = 1.70) -> dict[str, np.ndarray]:
    """Persona de pie, brazos colgando; coordenadas (x, y arriba, z) en metros."""
    h = estatura
    puntos = {"nariz": (0.0, 0.93 * h, 0.0)}
    for signo, s in ((1, "izq"), (-1, "der")):
        puntos |= {
            f"oreja_{s}": (signo * 0.07, (1 - FRACCION_OREJA_VERTEX) * h, 0.0),
            f"hombro_{s}": (signo * 0.20, 0.82 * h, 0.0),
            f"codo_{s}": (signo * 0.20, 0.82 * h - 0.30, 0.0),
            f"muneca_{s}": (signo * 0.20, 0.82 * h - 0.55, 0.0),
            f"cadera_{s}": (signo * 0.10, 0.52 * h, 0.0),
            f"rodilla_{s}": (signo * 0.10, 0.28 * h, 0.0),
            f"tobillo_{s}": (signo * 0.10, 0.04 * h, 0.0),
            f"talon_{s}": (signo * 0.10, 0.0, 0.0),
            f"punta_{s}": (signo * 0.10, 0.0, 0.0),
        }
    return {k: np.array(v, dtype=np.float64) for k, v in puntos.items()}


def test_largos_desde_articulaciones():
    medidas = largos_desde_articulaciones(esqueleto())
    assert medidas["largo_brazo"] == pytest.approx(55.0)
    assert medidas["largo_antebrazo"] == pytest.approx(25.0)
    assert medidas["ancho_hombros"] == pytest.approx(40.0)
    assert medidas["largo_pierna"] == pytest.approx(0.48 * 170)
    assert medidas["largo_torso"] == pytest.approx(0.30 * 170)


def test_largos_usan_el_lado_disponible_y_omiten_lo_que_falta():
    j = esqueleto()
    del j["codo_izq"], j["rodilla_izq"], j["rodilla_der"]
    medidas = largos_desde_articulaciones(j)
    assert medidas["largo_brazo"] == pytest.approx(55.0)
    assert "largo_pierna" not in medidas


def test_estatura_por_cadena_no_depende_de_la_inclinacion():
    j = esqueleto(1.62)
    assert estatura_por_cadena(j) == pytest.approx(1.62, abs=0.01)
    angulo = np.radians(20)
    rotacion = np.array(
        [[1, 0, 0], [0, np.cos(angulo), -np.sin(angulo)], [0, np.sin(angulo), np.cos(angulo)]]
    )
    inclinado = {k: rotacion @ v for k, v in j.items()}
    assert estatura_por_cadena(inclinado) == pytest.approx(1.62, abs=0.01)


def test_estatura_por_cadena_sin_talones():
    j = esqueleto()
    del j["talon_izq"], j["talon_der"]
    assert estatura_por_cadena(j) is None


def test_profundidad_recupera_estatura_y_brazo_con_camara_inclinada():
    pytest.importorskip("cv2")
    from src.medidas_monoculares.metodo_profundidad import medir_desde_puntos

    estatura, distancia, focal, ancho, alto = 1.70, 2.5, 900.0, 720, 1280
    j = esqueleto(estatura)
    # Cámara a 1 m del suelo, inclinada 8° hacia abajo. Ejes de cámara: x derecha,
    # y abajo, z al frente.
    inclinacion = np.radians(8)
    c, s = np.cos(inclinacion), np.sin(inclinacion)
    rotacion = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def a_camara(p):
        return rotacion @ np.array([p[0], 1.0 - p[1], distancia + p[2]])

    def proyectar(p):
        q = a_camara(p)
        return np.array([focal * q[0] / q[2] + ancho / 2, focal * q[1] / q[2] + alto / 2])

    # La "persona" es un tablero plano de 0,5 m × estatura frente a la cámara.
    origen = a_camara((0.0, 0.0, 0.0))
    normal = rotacion @ np.array([0.0, 0.0, 1.0])
    v, u = np.mgrid[0:alto, 0:ancho].astype(np.float64)
    rayos = np.stack([(u - ancho / 2) / focal, (v - alto / 2) / focal, np.ones_like(u)], axis=-1)
    puntos = rayos * ((origen @ normal) / (rayos @ normal))[..., None]
    mundo = (puntos - origen) @ rotacion  # coordenadas del tablero: x, y abajo desde el suelo
    mascara = (np.abs(mundo[..., 0]) <= 0.25) & (mundo[..., 1] <= 0) & (mundo[..., 1] >= -estatura)

    pose = Pose(
        puntos_px={k: proyectar(p) for k, p in j.items()},
        puntos_mundo=j,
        visibilidad={k: 1.0 for k in INDICES},
        mascara=mascara,
        avisos=[],
    )
    resultado = medir_desde_puntos(puntos, np.ones_like(mascara), pose)
    assert resultado.medidas_cm["estatura"] == pytest.approx(170.0, abs=1.0)
    assert resultado.medidas_cm["largo_brazo"] == pytest.approx(55.0, abs=1.0)
    assert resultado.medidas_cm["ancho_hombros"] == pytest.approx(40.0, abs=1.0)


def test_medir_malla_sobre_un_cilindro():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    from src.medidas_monoculares.medidas_malla import medir_malla

    radio, estatura = 0.15, 1.70
    cilindro = trimesh.creation.cylinder(radius=radio, height=estatura, sections=128)
    cilindro.apply_translation([0, 0, estatura / 2])
    # Reordena a (x, z, y) → (x, y, z arriba) como espera medir_malla.
    j = {k: v[[0, 2, 1]] for k, v in esqueleto(estatura).items()}
    medidas = medir_malla(np.asarray(cilindro.vertices), np.asarray(cilindro.faces), j)
    assert medidas["estatura"] == pytest.approx(170.0, abs=0.01)
    for contorno in ("contorno_pecho", "contorno_cintura", "contorno_cadera"):
        assert medidas[contorno] == pytest.approx(2 * np.pi * radio * 100, rel=0.01)
