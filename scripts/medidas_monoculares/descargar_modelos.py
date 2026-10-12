"""Descarga los pesos y el código de terceros de src/medidas_monoculares.

Solo usa la biblioteca estándar. Todo queda en pretrained/ y vendor/, que no
se versionan. Reanuda descargas interrumpidas.

    python scripts/medidas_monoculares/descargar_modelos.py            # todo (~1,7 GB)
    python scripts/medidas_monoculares/descargar_modelos.py --sin-malla  # ~170 MB
"""

from __future__ import annotations

import argparse
import http.client
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PESOS = {
    "pose": (
        "pretrained/mediapipe/pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    ),
    "profundidad": (
        "pretrained/moge/moge-2-vits-normal.pt",
        "https://huggingface.co/Ruicheng/moge-2-vits-normal/resolve/main/model.pt",
    ),
    "malla": (
        "pretrained/multihmr/multiHMR_672_L_anny.pt",
        "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_L_anny.pt",
    ),
}
# Código con licencia no comercial de NAVER: se clona, no se copia al repositorio.
MULTI_HMR_REPO = "https://github.com/naver/multi-hmr.git"
MULTI_HMR_COMMIT = "651fb411e1cbcc626aaa5f38805ecab9cc891f7a"
REINTENTOS = 200


def descargar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "sastre-ia/medidas-monoculares"})
    with urllib.request.urlopen(request, timeout=60) as head:
        total = int(head.headers.get("Content-Length", 0))
    if destino.is_file() and total and destino.stat().st_size == total:
        print(f"[ok] {destino.relative_to(ROOT)} ya está completo")
        return
    parcial = destino.with_suffix(destino.suffix + ".part")
    hecho = parcial.stat().st_size if parcial.is_file() else 0
    # Las conexiones lentas se cortan a mitad de archivo: se reanuda con Range.
    for _ in range(REINTENTOS):
        if total and hecho == total:
            break
        request = urllib.request.Request(url, headers={"User-Agent": "sastre-ia/medidas-monoculares"})
        if hecho:
            request.add_header("Range", f"bytes={hecho}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if hecho and response.status != 206:
                    hecho = 0
                with parcial.open("ab" if hecho else "wb") as salida:
                    while chunk := response.read(1 << 18):
                        salida.write(chunk)
                        hecho += len(chunk)
                        print(
                            f"\r[..] {destino.name}: {hecho / 1e6:7.0f} / {total / 1e6:.0f} MB",
                            end="",
                            flush=True,
                        )
        except (OSError, http.client.HTTPException) as error:
            print(f"\n[reintento] {destino.name}: {error}")
            time.sleep(3)
        if not total:
            break
    if total and hecho != total:
        raise RuntimeError(f"Descarga incompleta de {url}: vuelve a ejecutar para reanudar")
    parcial.replace(destino)
    print(f"\n[ok] {destino.relative_to(ROOT)}")


def clonar_multi_hmr() -> None:
    destino = ROOT / "vendor" / "multi-hmr"
    if not (destino / ".git").is_dir():
        destino.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", MULTI_HMR_REPO, str(destino)], check=True)
    subprocess.run(
        ["git", "-C", str(destino), "checkout", "--quiet", MULTI_HMR_COMMIT], check=True
    )
    print(f"[ok] vendor/multi-hmr @ {MULTI_HMR_COMMIT[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sin-malla",
        action="store_true",
        help="omite Multi-HMR + Anny (1,5 GB, licencia no comercial)",
    )
    args = parser.parse_args()
    for nombre, (ruta, url) in PESOS.items():
        if nombre == "malla" and args.sin_malla:
            continue
        descargar(url, ROOT / ruta)
    if not args.sin_malla:
        clonar_multi_hmr()


if __name__ == "__main__":
    sys.exit(main())
