# SATRE-IA

Prototipo local de apoyo al sastre que estima medidas corporales desde dos
fotografías, genera patrones paramétricos y ofrece una visualización virtual
experimental. No usa servicios en la nube.

## Arquitectura oficial

```text
Foto frontal + perfil izquierdo + estatura conocida
                         │
               MediaPipe Pose Lite
             landmarks y control de pose
                         │
                  SAM 2.1 Small
                 dos siluetas binarias
                         │
        exp_008 ExtraTrees sobre perfiles físicos
            peso + 13 medidas + intervalos
                         │
             validación de procedencia
                         │
       FreeSewing Jaeger / Charlie / Penelope
          SVG 1:1 + PDF A4 + proyecto JSON
```

El registro que actúa como fuente única de verdad es
[`configs/modelos_runtime.json`](configs/modelos_runtime.json). Un modelo
descargado no se considera activo si no aparece en una ruta de ejecución.

### Modelos activos

| Etapa | Componente | Estado |
|---|---|---|
| Pose | MediaPipe Pose Landmarker Lite | oficial |
| Segmentación | SAM 2.1 Hiera Small | oficial |
| Regresión | `exp_008_perfiles_arboles` | oficial |
| Patronaje | FreeSewing 4.10.1 | oficial, no es IA |
| Parsing visual | SCHP-ATR ONNX INT8 | diagnóstico 2D bajo demanda |
| Vestidor | FASHN VTON 1.5 | experimental, desactivado por defecto |

SHAPY/SMPL-X y `exp_001`–`exp_007` no forman parte del runtime oficial. Los
experimentos anteriores se conservan como evidencia de comparación y
ablación; no se cargan durante la demo.

## Datos y precisión

El regresor oficial se entrenó y evaluó con BodyM porque contiene pares de
siluetas frontal/lateral, estatura, peso y medidas antropométricas. El protocolo
de captura del prototipo replica esas entradas: teléfono vertical, lente 1×,
altura aproximada de cintura, distancia de 1,7–2,0 m, cuerpo completo y ropa
ceñida.

Resultados independientes guardados en
`experiments/exp_008_perfiles_arboles/results.json`:

| Partición | N | MAE 13 | RMSE 13 | R² 13 | Cobertura 90 % | MAE peso |
|---|---:|---:|---:|---:|---:|---:|
| Test-A | 1684 | 1,92 cm | 2,46 cm | 0,685 | 90,3 % | 3,60 kg |
| Test-B / libre | 1160 | 2,18 cm | 2,92 cm | 0,701 | 86,9 % | 4,86 kg |

El promedio no significa ±2 cm para cada medida. En Test-B, pecho, cintura y
cadera rondan 4–5 cm de MAE. Por eso el sistema muestra intervalos, marca las
medidas críticas y bloquea la declaración de “listo para corte” mientras existan
medidas automáticas o derivadas sin confirmar. Las fotos con ropa holgada están
fuera del protocolo y deben repetirse.

Las métricas ampliadas reproducibles se guardan en
`experiments/exp_008_perfiles_arboles/metrics_extended.json`.

## Ejecutar la demo

Desde la raíz del proyecto, el arranque recomendado es:

```powershell
.\scripts\iniciar_demo.ps1 -ConVestidor
```

Para ejecutar primero todas las pruebas y después iniciar:

```powershell
.\scripts\iniciar_demo.ps1 -ConVestidor -VerificacionCompleta
```

La verificación sin iniciar el servidor se ejecuta con:

```powershell
.\scripts\verificar_demo_defensa.ps1 -Completa
```

El método manual equivalente es:

```powershell
Set-Location pattern-engine
npm start
```

Abrir `http://127.0.0.1:8765/`.

Para habilitar el vestidor generativo local:

```powershell
$env:SATRE_ENABLE_EXPERIMENTAL_TRYON='1'
Set-Location pattern-engine
npm start
```

El perfil de demo usa un candidato determinista, 30 pasos y composición de
identidad. En la RTX 3050 de 6 GB la corrida validada alcanzó 2,72 GB de VRAM y
aproximadamente 90 segundos. La imagen de FASHN visualiza apariencia; no valida
que el patrón ajuste físicamente. La superposición geométrica permanece como
diagnóstico técnico y nunca se presenta como resultado realista.

## Probar y auditar

```powershell
& 'C:\Users\diego\miniconda3\envs\sastre-ia\python.exe' -m pytest -q
Set-Location pattern-engine
npm test
```

Auditoría reproducible de modelos:

```powershell
& 'C:\Users\diego\miniconda3\envs\sastre-ia\python.exe' `
  scripts\auditar_modelos_runtime.py
```

El informe se guarda en
[`docs/AUDITORIA_MODELOS_RUNTIME.md`](docs/AUDITORIA_MODELOS_RUNTIME.md).

## Flujo de demostración recomendado

1. Seleccionar explícitamente “saco + pantalón” o “saco + falda”.
2. Escribir la estatura.
3. Subir frontal y perfil izquierdo con ropa ceñida.
4. Ejecutar la medición y mostrar la procedencia de los tres modelos activos.
5. Revisar las 13 medidas y sus intervalos.
6. Generar el molde como borrador métrico SVG/PDF.
7. Mostrar el vestidor FASHN como módulo experimental separado.

Para una defensa estable conviene tener una sesión masculina y otra femenina ya
procesadas como respaldo. La demostración en vivo debe conservar los avisos de
incertidumbre y no afirmar que una imagen virtual demuestra ajuste de sastrería.

## Documentación principal

- [Selección del modelo de dos vistas](docs/SELECCION_MODELO_DOS_VISTAS.md)
- [Auditoría de modelos activos](docs/AUDITORIA_MODELOS_RUNTIME.md)
- [Configuración de percepción](docs/PERCEPTION_SETUP.md)
- [Decisión del motor de patrones](reports/pattern_engine_decision.md)
- [Estatura y largos desde una sola foto (experimental)](docs/MEDIDAS_MONOCULARES.md)
