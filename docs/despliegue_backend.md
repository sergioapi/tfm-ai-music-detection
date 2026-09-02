# Contrato de despliegue del backend de VeriSon

**Estado:** vigente desde el 27 de agosto de 2026. Este documento resume el contrato reproducible del backend; no sustituye las decisiones de cold start, warm-up o readiness pendientes de T03/T05.

## Servicio público

- **Backend:** Northflank Developer Sandbox, servicio `verison-api`.
- **Región:** Europe - West (London).
- **Plan:** `nf-compute-20`; `0.2 vCPU shared`, `512 MiB` de RAM, una instancia y `1 GiB` de almacenamiento efímero.
- **URL pública:** `https://api--verison-api--xb7vy98gqd48.code.run`.
- **Frontend:** `https://verison-app.vercel.app`, cuya variable de producción `VITE_API_BASE_URL` apunta al backend anterior.
- **Rollback temporal:** Render permanece disponible mientras se completan las tareas posteriores.

La facturación observada durante la validación de Northflank fue “No usage / You have not accrued any costs yet”. Es una observación de esa prueba, no una garantía de coste futuro.

## Construcción y arranque

- **Build type:** Dockerfile con contexto en la raíz del repositorio (`/`).
- **Dockerfile:** `deploy/backend/Dockerfile`.
- **Contexto efectivo:** `deploy/backend/Dockerfile.dockerignore` incluye solo `backend/app`, `backend/requirements.txt` y `data/models/mfcc_svm_baseline.joblib`.
- **Python:** `3.11.9` (`python:3.11.9-slim`).
- **Comando efectivo:** `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`.
- **Puerto interno:** `8000`. `PORT` no necesita declararse: el contenedor usa `8000` por defecto y respeta el valor que aporte la plataforma.

Las dependencias runtime críticas están fijadas en `backend/requirements.txt`:

```text
fastapi==0.141.1
uvicorn==0.52.4
python-multipart==0.0.32
joblib==1.5.3
numpy==2.4.6
librosa==0.11.0
soundfile==0.14.0
scikit-learn==1.8.0
soxr==1.1.0
```

`psutil` sigue temporalmente para profiling y se revisará en T11.

## Variables y defaults

| Variable | Valor efectivo |
| --- | --- |
| `CORS_ALLOWED_ORIGINS` | `https://verison-app.vercel.app` |
| `RESAMPLE_WARMUP_ENABLED` | `false` |
| `MEMORY_PROFILING_ENABLED` | `true` temporalmente durante las mediciones pendientes |
| `MODEL_PATH` | Default del contenedor: `/opt/verison/models/mfcc_svm_baseline.joblib` |
| `PORT` | Default `8000`; no se declara manualmente |
| `MAX_UPLOAD_SIZE_BYTES` | Default `67108864` (64 MiB) |
| `MAX_AUDIO_DURATION_SECONDS` | Default `300` |
| `TEMP_DIR` | Sin declarar; temporal escribible del contenedor |

Los formatos admitidos son WAV y MP3. No se deben cambiar estos valores para las tareas de disponibilidad sin una decisión explícita.

## Artefacto, health y verificación mínima

- **Artefacto canónico:** `data/models/mfcc_svm_baseline.joblib`.
- **SHA-256 esperado:** `ee4359aa9f9942a1179184a28834c5a1b6d901253ac82bce90a32472451a0336`.
- **Probe de plataforma:** HTTP `GET /health` en el puerto `8000`; es una comprobación de carga del modelo, no la futura semántica de readiness de T05.
- **Rutas públicas:** `GET /health`, `GET /api/v1/model`, `POST /api/v1/analyze`.

Después de cada despliegue, comprobar `GET /health` (200), `GET /api/v1/model` (200 y SHA-256 esperado), un WAV válido y un MP3 válido. Verificar CORS desde el origen Vercel configurado; después de los smoke tests cortos, probar un audio largo dentro de 64 MiB y 300 s sin interpretar una etiqueta individual como verdad semántica.

La evidencia de T02 confirmó estos casos, incluido un WAV de `256.130625 s` con 26 fragmentos, sin OOM ni reinicio. La validación manual Vercel → Northflank realizada entonces no es la validación E2E final de T12.
