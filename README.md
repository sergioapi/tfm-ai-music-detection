# TFM - Detección de música generada mediante IA

Este repositorio contiene el desarrollo de un Trabajo Fin de Máster centrado en la detección de canciones generadas mediante inteligencia artificial.

El proyecto se divide en dos partes. La primera compara dos enfoques de clasificación: un modelo clásico basado en MFCC + StandardScaler + SVM RBF y un modelo profundo basado en MERT-v1-95M congelado + SVM lineal. Tras la comparación, se seleccionó MFCC + SVM como modelo para la aplicación web.

La segunda parte corresponde al desarrollo de VeriSon, una aplicación web que permite subir archivos de audio y obtener una estimación de clasificación entre música de posible origen humano y música generada mediante IA. La aplicación utiliza FastAPI para el backend de inferencia y React + Vite + TypeScript para el frontend.

## Estructura del repositorio

```text
backend/    API FastAPI, inferencia del MVP y tests del backend
frontend/   Interfaz web VeriSon
configs/    Configuracion experimental
data/       Datos, particiones experimentales y artefactos generados
docs/       Evidencia tecnica, decisiones y resultados experimentales
memoria/    Memoria academica del TFM
notebooks/  Exploracion y pruebas iniciales
scripts/    Pipeline experimental y generacion de artefactos
tests/      Tests de experimentacion y pipeline
```

`backend/tests/` cubre el backend y la inferencia del MVP. `tests/` en la raiz
cubre principalmente scripts y flujo experimental.

## Preparacion del entorno Python

Desde la raiz del repositorio, crear y activar el entorno virtual local en
PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para reproducir los experimentos deben utilizarse las particiones definidas en `data/aime_splits.csv`, que corresponden al protocolo experimental empleado en el proyecto.

## Reproduccion experimental

Flujo principal del baseline MFCC + SVM:

```powershell
python scripts/extract_aime_mfcc.py
python scripts/train_mfcc_svm.py
```

Si los audios de AIME ya estan descargados localmente, la extraccion MFCC puede
usar una carpeta ignorada por Git:

```powershell
python scripts/extract_aime_mfcc.py --audio-dir data/audio/aime_raw
```

Flujo principal de MERT congelado + SVM:

```powershell
python scripts/smoke_test_mert.py --device cpu
python scripts/extract_mert_embeddings.py --device cpu
python scripts/train_mert_svm_classifier.py --config configs/mert_svm_classifier.yaml
python scripts/evaluate_mert_svm_test.py --config configs/mert_svm_classifier.yaml
```

Comparacion de modelos y seleccion del modelo de despliegue:

```powershell
python scripts/build_model_comparison.py
```

La evidencia detallada esta en `docs/`, especialmente:

- `docs/aime_audit_summary.md`
- `docs/mfcc_svm_baseline_summary.md`
- `docs/model_comparison_summary.md`
- `docs/decisions/seleccion-modelo-despliegue.md`

## Backend

El backend local se arranca desde la raiz con:

```powershell
.\backend\run-dev.ps1
```

El script usa el Python de `.venv`, configura el entorno necesario para
desarrollo local y ejecuta Uvicorn en `127.0.0.1:8000`.

Para quedar funcional, el backend necesita el artefacto local:

```text
data/models/mfcc_svm_baseline.joblib
```

Ese fichero esta ignorado por Git. Puede generarse con el flujo experimental
MFCC, en particular con `python scripts/train_mfcc_svm.py` una vez disponibles
las caracteristicas necesarias.

## Frontend

Preparar y arrancar la interfaz desde `frontend/`:

```powershell
cd frontend
npm install
npm run dev
```

El frontend requiere configuracion local de entorno para localizar la API.
Tomar `frontend/.env.example` como referencia y crear el archivo local
correspondiente sin versionarlo.

## Tests

Ejecutar toda la suite Python desde la raiz:

```powershell
pytest
```

Ejecutar solo los tests del backend:

```powershell
pytest backend/tests
```

Validar el frontend:

```powershell
cd frontend
npm run check
```

`npm run check` ejecuta lint, build y tests del frontend.
