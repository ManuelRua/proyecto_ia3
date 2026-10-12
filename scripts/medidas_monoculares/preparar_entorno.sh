#!/usr/bin/env bash
# Crea .venv-medidas (Linux + uv) y descarga los modelos de src/medidas_monoculares.
# Los argumentos se pasan a descargar_modelos.py, p. ej. --sin-malla.
set -euo pipefail
cd "$(dirname "$0")/../.."

MOGE_COMMIT=74fbce054ebed49800de42d0ad0e83495065719a
UTILS3D_COMMIT=62f09d58509485564e24d5d9f6aac9ee9ebc0c37

[ -d .venv-medidas ] || uv venv --python 3.12 .venv-medidas
uv pip install --python .venv-medidas/bin/python -r envs/requirements-medidas-monoculares.txt
# MoGe declara gradio, flex-gemm y otras dependencias de su demo y de MoGe-3
# que la inferencia de MoGe-2 no usa: se instala sin dependencias.
uv pip install --python .venv-medidas/bin/python --no-deps \
  "moge @ git+https://github.com/microsoft/MoGe.git@${MOGE_COMMIT}" \
  "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@${UTILS3D_COMMIT}"

.venv-medidas/bin/python scripts/medidas_monoculares/descargar_modelos.py "$@"
echo "Listo. Prueba: .venv-medidas/bin/python scripts/medidas_monoculares/demo.py"
