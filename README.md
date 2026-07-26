# TFM — Detección de música generada mediante IA

Trabajo Fin de Máster orientado a comparar un baseline MFCC + SVM con un modelo de audio preentrenado y desarrollar una prueba de concepto web.

## Tecnologías previstas

- Python y scikit-learn
- JupyterLab
- React y Vite
- FastAPI
- Hugging Face

## Entorno Python

Activar el entorno virtual en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Baseline clasico MFCC + SVM

El manifiesto experimental versionado es `data/aime_splits.csv`. No se regenera para entrenar el baseline.

Extraer caracteristicas MFCC:

```powershell
python scripts/extract_aime_mfcc.py
```

Si los audios ya estan descargados localmente, se puede evitar la descarga desde Hugging Face indicando una carpeta ignorada por Git:

```powershell
python scripts/extract_aime_mfcc.py --audio-dir data/audio/aime_raw
```

Entrenar, seleccionar hiperparametros con validacion y evaluar una unica vez en test:

```powershell
python scripts/train_mfcc_svm.py
```

Artefactos principales:

- `data/processed/aime_mfcc_features.parquet`
- `data/processed/aime_mfcc_failures.csv`
- `data/processed/aime_mfcc_extraction_summary.json`
- `data/models/mfcc_svm_baseline.joblib`
- `data/models/mfcc_svm_metrics.json`
- `data/models/mfcc_svm_predictions.csv`
- `data/models/mfcc_svm_confusion_matrix.png`
- `docs/mfcc_svm_baseline_summary.md`

## Modelo profundo preentrenado

La decision inicial selecciona `m-a-p/MERT-v1-95M` como encoder profundo principal congelado. La trazabilidad esta en `docs/decisions/seleccion-modelo-profundo-mert.md` y la configuracion inicial en `configs/mert_frozen_embeddings.yaml`.

Smoke test rapido de MERT en CPU con una pareja train:

```powershell
python scripts/smoke_test_mert.py --device cpu --max-pairs 1
```

Smoke test completo con las doce muestras train:

```powershell
python scripts/smoke_test_mert.py --device cpu
```

Si hay CUDA disponible:

```powershell
python scripts/smoke_test_mert.py --device cuda
```

Si los audios ya estan descargados localmente:

```powershell
python scripts/smoke_test_mert.py --device cpu --audio-dir data/audio/aime_raw
```

Resumen tecnico del smoke test:

- `docs/mert_smoke_test_summary.md`

Resultado estructurado generado localmente:

- `data/processed/mert_smoke_test_result.json`

Extraer embeddings MERT para los 1000 ejemplos del manifiesto, en CPU y con reanudacion si ya existe un CSV parcial:

```powershell
python scripts/extract_mert_embeddings.py --device cpu
```

Artefactos locales generados por la extraccion:

- `data/processed/aime_mert_embeddings.csv`
- `data/processed/aime_mert_embeddings.parquet`
- `data/processed/aime_mert_embedding_extraction_summary.json`
- `data/processed/aime_mert_embedding_failures.csv`, solo si se registran fallos

Resumen tecnico de la extraccion:

- `docs/mert_embedding_extraction_summary.md`

Ejecutar tests:

```powershell
pytest
```
