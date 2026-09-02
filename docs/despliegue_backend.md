# Contrato de despliegue del backend de VeriSon

**Estado:** contrato vigente de la release candidate, actualizado el 2 de septiembre de 2026. T12 permanece pendiente de la evidencia manual E2E final.

## Servicio público

- **Backend:** Northflank Developer Sandbox, servicio `verison-api`.
- **Región:** Europe - West (London).
- **Plan:** `nf-compute-20`; `0.2 vCPU shared`, `512 MiB` de RAM, una instancia y `1 GiB` de almacenamiento efímero.
- **URL pública:** `https://api--verison-api--xb7vy98gqd48.code.run`.
- **Frontend:** `https://verison-app.vercel.app`, cuya variable de producción `VITE_API_BASE_URL` apunta al backend anterior.

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

No se incluye dependencia de profiling; `psutil` y la instrumentación temporal fueron retirados.

## Variables y defaults

| Variable | Valor efectivo |
| --- | --- |
| `CORS_ALLOWED_ORIGINS` | `https://verison-app.vercel.app` |
| `RESAMPLE_WARMUP_ENABLED` | `true` en Northflank |
| `MODEL_PATH` | Default del contenedor: `/opt/verison/models/mfcc_svm_baseline.joblib` |
| `PORT` | Default `8000`; no se declara manualmente |
| `MAX_UPLOAD_SIZE_BYTES` | Default `67108864` (64 MiB) |
| `MAX_AUDIO_DURATION_SECONDS` | Default `300` |
| `TEMP_DIR` | Sin declarar; temporal escribible del contenedor |

Los formatos admitidos son WAV y MP3. No se deben cambiar estos valores para las tareas de disponibilidad sin una decisión explícita.

## Artefacto, health y verificación mínima

- **Artefacto canónico:** `data/models/mfcc_svm_baseline.joblib`.
- **SHA-256 esperado:** `ee4359aa9f9942a1179184a28834c5a1b6d901253ac82bce90a32472451a0336`.
- **Estado operativo/modelo:** `GET /health` en el puerto `8000`.
- **Readiness:** `GET /ready` alcanza `200` con `status=ready` tras completar los warm-ups habilitados.
- **Rutas públicas:** `GET /health`, `GET /ready`, `GET /api/v1/model`, `POST /api/v1/analyze`.

Después de cada despliegue, comprobar `GET /health` (200), `GET /ready` (`200` y `status=ready`), `GET /api/v1/model` (200 y SHA-256 esperado), un WAV válido y un MP3 válido. Verificar CORS desde el origen Vercel configurado; después de los smoke tests cortos, probar un audio largo dentro de 64 MiB y 300 s sin interpretar una etiqueta individual como verdad semántica.

La evidencia operacional confirmó estos casos, incluido un WAV de `256.130625 s` con 26 fragmentos y dos análisis largos simultáneos, sin OOM ni reinicio. La validación manual Vercel → Northflank de T12 debe registrarse por separado antes del cierre final.
