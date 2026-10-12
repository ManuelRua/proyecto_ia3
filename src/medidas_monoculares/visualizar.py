"""Panel comparativo: una columna por método y la tabla de medidas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.medidas_monoculares.comun import MEDIDAS, ResultadoMetodo

HUESOS = [
    ("hombro_izq", "hombro_der"), ("cadera_izq", "cadera_der"),
    ("hombro_izq", "cadera_izq"), ("hombro_der", "cadera_der"),
    *[(f"{a}_{s}", f"{b}_{s}") for s in ("izq", "der") for a, b in (
        ("hombro", "codo"), ("codo", "muneca"), ("cadera", "rodilla"),
        ("rodilla", "tobillo"), ("tobillo", "talon"),
    )],
]
TITULOS = {
    "landmarks": "1 · Landmarks 3D (MediaPipe)",
    "profundidad": "2 · Geometría métrica (MoGe-2)",
    "malla": "3 · Malla paramétrica (Multi-HMR + Anny)",
}


def _esqueleto(ax, puntos: dict[str, tuple[float, float]]) -> None:
    for a, b in HUESOS:
        if a in puntos and b in puntos:
            ax.plot(*zip(puntos[a], puntos[b]), color="#00e5ff", linewidth=1.6)
    if puntos:
        xs, ys = zip(*puntos.values())
        ax.scatter(xs, ys, s=10, color="#ffea00", zorder=3)


def panel_comparativo(
    image: Image.Image,
    resultados: list[ResultadoMetodo],
    capas: dict[str, np.ndarray],
    salida: Path,
    referencia_cm: dict[str, float] | None = None,
) -> None:
    """`capas`: mapa de profundidad (H, W) para 'profundidad', vértices (N, 2) para 'malla'."""
    referencia_cm = referencia_cm or {}
    columnas = len(resultados)
    fig = plt.figure(figsize=(3.6 * columnas + 4.6, 3.6 * image.height / image.width + 0.5), dpi=110)
    grid = fig.add_gridspec(1, columnas + 1, width_ratios=[3.6] * columnas + [4.6])

    for i, resultado in enumerate(resultados):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(image)
        capa = capas.get(resultado.metodo)
        if resultado.metodo == "profundidad" and capa is not None:
            ax.imshow(capa, cmap="turbo_r", alpha=0.75)
        if resultado.metodo == "malla" and capa is not None:
            ax.scatter(capa[::2, 0], capa[::2, 1], s=0.3, color="#ff4081", alpha=0.5)
        _esqueleto(ax, resultado.puntos_2d)
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.set_title(TITULOS.get(resultado.metodo, resultado.metodo), fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(grid[0, columnas])
    ax.axis("off")
    filas = [k for k in MEDIDAS if any(k in r.medidas_cm for r in resultados) or k in referencia_cm]
    encabezado = ["medida (cm)", *[r.metodo for r in resultados]]
    if referencia_cm:
        encabezado.append("real")
    celdas = []
    for k in filas:
        fila = [k.replace("_", " ")]
        fila += [f"{r.medidas_cm[k]:.1f}" if k in r.medidas_cm else "—" for r in resultados]
        if referencia_cm:
            fila.append(f"{referencia_cm[k]:.1f}" if k in referencia_cm else "—")
        celdas.append(fila)
    resto = (1.0 - 0.32) / (len(encabezado) - 1)
    tabla = ax.table(
        cellText=celdas,
        colLabels=encabezado,
        colWidths=[0.32] + [resto] * (len(encabezado) - 1),
        loc="upper center",
        cellLoc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1.0, 1.45)
    for (fila, _), celda in tabla.get_celld().items():
        celda.set_edgecolor("#cccccc")
        if fila == 0:
            celda.set_facecolor("#263238")
            celda.set_text_props(color="white", weight="bold")
    avisos = [f"[{r.metodo}] {a}" for r in resultados for a in r.avisos]
    nota = "Prueba de humo: los valores NO están validados.\n" + "\n".join(avisos)
    ax.text(0.0, 0.02, nota, fontsize=7, color="#b71c1c", va="bottom", wrap=True, transform=ax.transAxes)

    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida)
    plt.close(fig)
