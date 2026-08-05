# MFCC + SVM inference benchmark

## Entorno

- Fecha UTC: `2026-08-03T15:23:26.571342+00:00`.
- Sistema: `Windows-10-10.0.26200-SP0`.
- Python: `3.11.9`.
- NumPy: `2.4.6`.
- librosa: `0.11.0`.
- soundfile/libsndfile: `0.14.0` / `1.2.2`.
- scikit-learn: `1.9.0`.
- joblib: `1.5.3`.
- psutil: `7.2.2`.
- RAM total: `15.77 GB`.

## Artefacto

- Ruta: `data/models/mfcc_svm_baseline.joblib`.
- Tamaño: `136.19 KB`.
- SHA-256: `ee4359aa9f9942a1179184a28834c5a1b6d901253ac82bce90a32472451a0336`.

## Reproducción

```powershell
$env:PYTHONPATH="backend"
$env:MODEL_PATH=(Resolve-Path "data/models/mfcc_svm_baseline.joblib").Path
.\.venv\Scripts\python.exe scripts/benchmark_mfcc_inference.py
```

## Tiempo y memoria de carga

- Carga del servicio: `0.7697 s`.
- RSS inicial: `147.61 MB`.
- RSS después de cargar: `158.40 MB`.
- Incremento RSS: `10.79 MB`.

## Advertencias de carga

- `InconsistentVersionWarning`: Trying to unpickle estimator StandardScaler from version 1.8.0 when using version 1.9.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
- `InconsistentVersionWarning`: Trying to unpickle estimator SVC from version 1.8.0 when using version 1.9.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
- `InconsistentVersionWarning`: Trying to unpickle estimator Pipeline from version 1.8.0 when using version 1.9.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations

## Primera inferencia

- Archivo: `audioldm-2-large_01631.wav`.
- Tiempo total: `3.6454 s`.
- Wall clock: `3.6467 s`.
- Fragmentos: `1`.
- RSS pico: `245.42 MB`.

## Resultados en caliente por audio

| archivo | duración | tamaño | canales | frecuencia | fragmentos | mediana total | máximo total | RTF medio | pico RSS máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `audioldm-2-large_01631.wav` | 10.000 s | 625.06 KB | 1 | 16000 | 1 | 0.0112 s | 0.0120 s | 0.00114 | 247.72 MB |
| `riffusion_02631.wav` | 10.230 s | 881.21 KB | 1 | 44100 | 2 | 0.0276 s | 0.0290 s | 0.00271 | 251.44 MB |
| `udio_04501.wav` | 32.832 s | 6.01 MB | 2 | 48000 | 4 | 0.0856 s | 0.0901 s | 0.00261 | 276.20 MB |
| `suno-v3_05001.wav` | 120.000 s | 21.97 MB | 2 | 48000 | 12 | 0.2721 s | 0.2982 s | 0.00231 | 382.07 MB |
| `mtg-jamendo_06001.wav` | 256.131 s | 46.90 MB | 2 | 48000 | 26 | 0.5754 s | 0.6307 s | 0.00229 | 538.61 MB |

## Tiempo por etapa

| archivo | decode mediana | preprocesado mediana | MFCC mediana | predicción mediana |
| --- | ---: | ---: | ---: | ---: |
| `audioldm-2-large_01631.wav` | 0.0015 s | 0.0001 s | 0.0089 s | 0.0007 s |
| `riffusion_02631.wav` | 0.0026 s | 0.0059 s | 0.0176 s | 0.0007 s |
| `udio_04501.wav` | 0.0150 s | 0.0273 s | 0.0365 s | 0.0007 s |
| `suno-v3_05001.wav` | 0.0503 s | 0.0937 s | 0.1085 s | 0.0010 s |
| `mtg-jamendo_06001.wav` | 0.1072 s | 0.2001 s | 0.2365 s | 0.0014 s |

## Memoria

| archivo | pico RSS mediano | pico RSS máximo | incremento pico máximo |
| --- | ---: | ---: | ---: |
| `audioldm-2-large_01631.wav` | 245.66 MB | 247.72 MB | 2.07 MB |
| `riffusion_02631.wav` | 251.27 MB | 251.44 MB | 5.78 MB |
| `udio_04501.wav` | 273.73 MB | 276.20 MB | 28.78 MB |
| `suno-v3_05001.wav` | 369.43 MB | 382.07 MB | 135.77 MB |
| `mtg-jamendo_06001.wav` | 518.85 MB | 538.61 MB | 294.30 MB |

## Determinismo

| archivo | determinista |
| --- | ---: |
| `audioldm-2-large_01631.wav` | `True` |
| `riffusion_02631.wav` | `True` |
| `udio_04501.wav` | `True` |
| `suno-v3_05001.wav` | `True` |
| `mtg-jamendo_06001.wav` | `True` |

## Observaciones y limitaciones

- La primera inferencia se mide aparte y no se incluye en las estadísticas en caliente.
- Los scores proceden de `decision_function`; no son probabilidades calibradas.
- Este benchmark no evalúa precisión predictiva ni etiquetas reales a nivel de canción.
- El benchmark debe repetirse en Docker o Hugging Face Spaces con versiones fijadas.
- Los límites de tamaño de subida quedan pendientes de pruebas de formatos comprimidos y de la API.
