# VeriSon — detección de música generada mediante IA

Repositorio del Trabajo Fin de Máster que compara MFCC + SVM y MERT congelado + SVM para clasificación humano/IA. También incluye VeriSon, una prueba de concepto web con FastAPI y React/Vite.

**Demo pública:** https://verison-app.vercel.app

La disponibilidad de esta instancia no se garantiza de forma indefinida. El proyecto puede ejecutarse localmente siguiendo las instrucciones de este README.

La documentación completa está en [`memoria/`](memoria/), incluido [`TFM___Memoria.pdf`](memoria/TFM___Memoria.pdf).

## Estructura

```text
backend/       API FastAPI, inferencia y tests del backend
configs/       configuración reproducible del experimento MERT
data/          manifiesto, modelo MFCC y artefactos experimentales
deploy/        Dockerfile del backend
docs/          informes y trazabilidad técnica de los experimentos
frontend/      aplicación React + Vite + TypeScript
memoria/       documentación académica del TFM
scripts/       pipeline experimental
tests/         tests de experimentación y pipeline
```

## Requisitos e instalación

- Python 3.11.9.
- Node.js 20.19.0 y npm 10.8.2.
- Docker, opcionalmente, para construir el contenedor del backend.

Crear el entorno Python desde la raíz:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` corresponde al entorno experimental y de desarrollo. `backend/requirements.txt` define el runtime mínimo que usa el contenedor.

Instalar el frontend:

```powershell
cd frontend
npm ci
cd ..
```

## Reproducción experimental

### AIME y manifiesto

El dataset es [`disco-eth/AIME`](https://huggingface.co/datasets/disco-eth/AIME). Reutiliza siempre el manifiesto fijo `data/aime_splits.csv`; define los clips y las particiones `train`, `val` y `test` del protocolo. Los scripts descargan AIME en streaming cuando no se proporciona `--audio-dir`. Los audios descargados y los artefactos intermedios generados localmente bajo `data/` están ignorados por Git. Los artefactos necesarios para reproducir o ejecutar el proyecto, como `data/aime_splits.csv` y el modelo MFCC utilizado por la aplicación, permanecen versionados.

### MFCC + SVM

```powershell
python scripts/extract_aime_mfcc.py
python scripts/train_mfcc_svm.py
```

Para usar una copia local de los audios:

```powershell
python scripts/extract_aime_mfcc.py --audio-dir data/audio/aime_raw
```

La extracción escribe características y resumen en `data/processed/`. El entrenamiento escribe el modelo, métricas, predicciones y matriz de confusión en `data/models/`, y genera `docs/mfcc_svm_baseline_summary.md`. El modelo usado por la aplicación es `data/models/mfcc_svm_baseline.joblib`.

### MERT congelado + SVM

```powershell
python scripts/smoke_test_mert.py --device cpu
python scripts/extract_mert_embeddings.py --device cpu
python scripts/train_mert_svm_classifier.py --config configs/mert_svm_classifier.yaml
python scripts/evaluate_mert_svm_test.py --config configs/mert_svm_classifier.yaml
```

La configuración fija modelo y revisión en `configs/mert_frozen_embeddings.yaml`. Los embeddings y el resumen estructurado se escriben en `data/processed/`; el smoke test, selección y evaluación generan sus informes en `docs/`.

### Comparación

```powershell
python scripts/build_model_comparison.py
```

El comando usa los artefactos existentes, sin reentrenar ni recalcular embeddings, y genera `docs/model_comparison.json`, `docs/model_comparison_summary.md` y `docs/decisions/seleccion-modelo-despliegue.md`.

## Aplicación web

### Backend

El backend requiere `data/models/mfcc_svm_baseline.joblib`. Desde la raíz:

```powershell
.\backend\run-dev.ps1
```

El script usa `.venv`, configura `MODEL_PATH` y CORS para Vite, y arranca en `http://127.0.0.1:8000`.

Variables disponibles:

| Variable | Valor por defecto |
| --- | --- |
| `MODEL_PATH` | ruta del modelo; el Dockerfile usa `/opt/verison/models/mfcc_svm_baseline.joblib` |
| `CORS_ALLOWED_ORIGINS` | vacío |
| `RESAMPLE_WARMUP_ENABLED` | `false` |
| `MAX_UPLOAD_SIZE_BYTES` | `67108864` |
| `MAX_AUDIO_DURATION_SECONDS` | `300` |
| `TEMP_DIR` | temporal del sistema |

Endpoints principales: `GET /health`, `GET /ready`, `GET /api/v1/model` y `POST /api/v1/analyze`. Se admiten WAV y MP3.

### Frontend y ejecución conjunta local

Crear `frontend/.env.local`:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Con el backend iniciado en otra terminal:

```powershell
cd frontend
npm run dev
```

El cliente se sirve normalmente en `http://localhost:5173`.

## Tests y CI

Desde la raíz:

```powershell
pytest
pytest backend/tests
```

Para el frontend:

```powershell
cd frontend
npm run check
```

GitHub Actions se ejecuta en pushes y pull requests a `main`; valida los tests del backend, `npm run check` y la construcción Docker. No despliega servicios.

## Docker y despliegue

Construir la imagen desde la raíz:

```powershell
docker build -f deploy/backend/Dockerfile .
```

El Dockerfile `deploy/backend/Dockerfile` instala `backend/requirements.txt`, incorpora el backend y el modelo MFCC, y usa el puerto indicado por `PORT` (8000 por defecto).

En Northflank, crear un servicio Docker con contexto en la raíz y ese Dockerfile; definir `CORS_ALLOWED_ORIGINS` con el origen de Vercel y, si se desea el warm-up de despliegue, `RESAMPLE_WARMUP_ENABLED=true`.

En Vercel, desplegar `frontend/` y definir `VITE_API_BASE_URL` con la URL HTTPS del backend. Esta configuración de proveedores es externa al repositorio.
