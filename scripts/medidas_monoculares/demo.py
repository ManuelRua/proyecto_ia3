"""Estima estatura y largos desde una sola foto con tres métodos y los compara.

    python scripts/medidas_monoculares/demo.py                      # fotos de muestra del repo
    python scripts/medidas_monoculares/demo.py foto.jpg --estatura-real 165
    python scripts/medidas_monoculares/demo.py carpeta/ --metodos landmarks,profundidad

Por cada foto escribe en outputs/medidas_monoculares/ un JSON con las medidas
y un PNG con el panel comparativo. Es una prueba de humo: no evalúa precisión.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.medidas_monoculares import metodo_landmarks
from src.medidas_monoculares.comun import (
    MEDIDAS,
    ResultadoMetodo,
    cargar_imagen,
    elegir_dispositivo,
    liberar_memoria,
)
from src.medidas_monoculares.pose import DetectorPose
from src.medidas_monoculares.visualizar import panel_comparativo

METODOS = ("landmarks", "profundidad", "malla")
MUESTRAS = [
    ROOT / "pattern-engine/assets/tryon-examples/male-navy-suit-target.png",
    ROOT / "pattern-engine/assets/tryon-examples/female-navy-skirt-suit-target.png",
    ROOT / "pattern-engine/assets/garments/male-navy-suit-model-reference-v1.png",
]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}


def reunir_imagenes(entradas: list[str]) -> list[Path]:
    if not entradas:
        return MUESTRAS
    rutas: list[Path] = []
    for entrada in map(Path, entradas):
        if entrada.is_dir():
            rutas += sorted(p for p in entrada.iterdir() if p.suffix.lower() in EXTENSIONES)
        else:
            rutas.append(entrada)
    return rutas


def referencia_de(ruta: Path, estatura_real: float | None) -> dict[str, float]:
    """Medidas reales: `<foto>.json` junto a la foto (lo escribe generar_sinteticos.py)."""
    lateral = ruta.with_suffix(".json")
    referencia = json.loads(lateral.read_text())["medidas_cm"] if lateral.is_file() else {}
    if estatura_real is not None:
        referencia["estatura"] = estatura_real
    return referencia


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("imagenes", nargs="*", help="fotos o carpetas (por defecto, muestras del repo)")
    parser.add_argument("--metodos", default=",".join(METODOS), help=f"lista separada por comas de {METODOS}")
    parser.add_argument("--salida", default="outputs/medidas_monoculares")
    parser.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--estatura-real", type=float, help="cm; solo se muestra junto a las estimaciones")
    args = parser.parse_args()

    metodos = [m.strip() for m in args.metodos.split(",") if m.strip()]
    if desconocidos := set(metodos) - set(METODOS):
        parser.error(f"métodos desconocidos: {sorted(desconocidos)}")
    rutas = reunir_imagenes(args.imagenes)
    salida = (ROOT / args.salida) if not Path(args.salida).is_absolute() else Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    dispositivo = elegir_dispositivo(args.dispositivo)
    print(f"Dispositivo: {dispositivo} · métodos: {metodos} · {len(rutas)} imagen(es)")

    imagenes = {ruta: cargar_imagen(ruta) for ruta in rutas}
    resultados: dict[Path, list[ResultadoMetodo]] = {ruta: [] for ruta in rutas}
    capas: dict[Path, dict[str, np.ndarray]] = {ruta: {} for ruta in rutas}

    def anotar(ruta: Path, resultado: ResultadoMetodo, inicio: float) -> None:
        resultado.extras["segundos"] = round(time.perf_counter() - inicio, 2)
        resultados[ruta].append(resultado)

    # Los métodos se ejecutan de uno en uno sobre todas las fotos para tener un
    # solo modelo en memoria a la vez (4 GB de VRAM, 8 GB de RAM).
    poses = {}
    if {"landmarks", "profundidad"} & set(metodos):
        detector = DetectorPose()
        for ruta in rutas:
            inicio = time.perf_counter()
            try:
                poses[ruta] = detector.detectar(imagenes[ruta])
            except ValueError as error:
                print(f"[pose] {ruta.name}: {error}")
                continue
            if "landmarks" in metodos:
                anotar(ruta, metodo_landmarks.medir(poses[ruta]), inicio)
        detector.close()

    if "profundidad" in metodos:
        from src.medidas_monoculares.metodo_profundidad import EstimadorGeometria, medir_desde_puntos

        estimador = EstimadorGeometria(dispositivo)
        for ruta in rutas:
            if ruta not in poses:
                continue
            inicio = time.perf_counter()
            puntos, valido, fov = estimador.inferir(imagenes[ruta])
            resultado = medir_desde_puntos(puntos, valido, poses[ruta])
            resultado.extras["fov_grados"] = round(fov, 1)
            anotar(ruta, resultado, inicio)
            capas[ruta]["profundidad"] = np.where(valido, puntos[..., 2], np.nan)
        del estimador
        liberar_memoria()

    if "malla" in metodos:
        from src.medidas_monoculares.metodo_malla import EstimadorMalla

        estimador = EstimadorMalla(dispositivo)
        for ruta in rutas:
            inicio = time.perf_counter()
            resultado, vertices_2d = estimador.medir(imagenes[ruta])
            anotar(ruta, resultado, inicio)
            if vertices_2d is not None:
                capas[ruta]["malla"] = vertices_2d
        del estimador
        liberar_memoria()

    for ruta in rutas:
        if not resultados[ruta]:
            continue
        referencia = referencia_de(ruta, args.estatura_real)
        informe = {
            "imagen": str(ruta),
            "aviso": "Prueba de humo sin validar. No usar para cortar tela.",
            "referencia_cm": referencia,
            "definiciones": MEDIDAS,
            "metodos": [r.to_dict() for r in resultados[ruta]],
        }
        (salida / f"{ruta.stem}.json").write_text(
            json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        panel_comparativo(
            imagenes[ruta], resultados[ruta], capas[ruta], salida / f"{ruta.stem}.png", referencia
        )
        print(f"\n{ruta.name}")
        nombres = [r.metodo for r in resultados[ruta]]
        print(f"  {'medida (cm)':<18}" + "".join(f"{n:>13}" for n in nombres) + (f"{'real':>9}" if referencia else ""))
        for medida in MEDIDAS:
            valores = [r.medidas_cm.get(medida) for r in resultados[ruta]]
            if all(v is None for v in valores) and medida not in referencia:
                continue
            linea = f"  {medida:<18}" + "".join(f"{v:>13.1f}" if v is not None else f"{'—':>13}" for v in valores)
            if referencia:
                linea += f"{referencia[medida]:>9.1f}" if medida in referencia else f"{'—':>9}"
            print(linea)
        print("  " + " · ".join(f"{r.metodo} {r.extras['segundos']} s" for r in resultados[ruta]))
    print(f"\nResultados en {salida}")


if __name__ == "__main__":
    main()
