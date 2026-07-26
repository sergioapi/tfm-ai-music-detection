# Evaluacion final MERT congelado + SVM

## Objetivo

Evaluar una unica vez en test la configuracion cerrada del enfoque profundo con embeddings MERT congelados y clasificador SVM lineal.

## Configuracion cerrada antes de test

- Pipeline: `StandardScaler` + `SVC`.
- Kernel: `linear`.
- `C`: `0.1`.
- `probability`: `false`.
- Score: `decision_function`.
- Umbral de decision: `0.0`.

## Protocolo

- El pipeline evaluado permanecio ajustado unicamente con `train`.
- La seleccion de hiperparametros se realizo unicamente con `val`.
- No se reentreno con `train + val` para mantener el mismo protocolo que el baseline clasico.
- No se reajusto el escalador, no se recalibro el score y no se modifico el umbral.
- No se modifico el modelo despues de observar test.

## Contexto de validacion

- Balanced accuracy val: `0.8667`.
- Precision IA val: `0.8395`.
- Recall IA val: `0.9067`.
- F1 IA val: `0.8718`.
- ROC-AUC val: `0.9218`.
- Matriz de confusion val: `[[62, 13], [7, 68]]`.

## Resultados de test

- Ejemplos de test: `150`.
- Aciertos: `123`.
- Errores: `27`.
- Falsos positivos: `14`.
- Falsos negativos: `13`.
- Balanced accuracy: `0.8200`.
- Precision IA: `0.8158`.
- Recall IA: `0.8267`.
- F1 IA: `0.8212`.
- ROC-AUC: `0.8985`.
- Matriz de confusion test `[label 0, label 1]`: `[[61, 14], [13, 62]]`.

## Desglose por generador IA

| generador | ejemplos IA test | correctos como IA | recall IA | score medio |
| --- | ---: | ---: | ---: | ---: |
| AudioLDM 2 Large | 6 | 6 | 1.0000 | 1.3707 |
| AudioLDM 2 Music | 6 | 5 | 0.8333 | 3.1069 |
| MusicGen Large | 6 | 5 | 0.8333 | 1.9871 |
| MusicGen Medium | 6 | 5 | 0.8333 | 2.1907 |
| MusicGen Small | 7 | 6 | 0.8571 | 1.9761 |
| Mustango | 6 | 5 | 0.8333 | 3.1601 |
| Riffusion | 6 | 5 | 0.8333 | 1.1982 |
| Stable Audio v1 | 7 | 5 | 0.7143 | 2.1055 |
| Stable Audio v2 | 6 | 6 | 1.0000 | 4.6711 |
| Suno v3 | 6 | 5 | 0.8333 | 0.9716 |
| Suno v3.5 | 6 | 5 | 0.8333 | 1.2275 |
| Udio | 7 | 4 | 0.5714 | -0.0791 |

## Metricas operacionales del clasificador

- Carga del artefacto joblib: `0.013222 s`.
- Prediccion total sobre test: `0.016948 s`.
- Latencia media: `0.00011298 s/ejemplo`.
- Tamano del modelo: `1693580` bytes.

Estas metricas cubren solo la carga y ejecucion del clasificador SVM sobre embeddings MERT ya calculados. No representan latencia extremo a extremo de MERT + SVM; la extraccion MERT tiene su propio resumen en `docs/mert_embedding_extraction_summary.md`.

## Artefactos

- Metricas JSON: `data/models/mert_svm_test_metrics.json`.
- Predicciones test: `data/models/mert_svm_test_predictions.csv`.
- Matriz de confusion test: `data/models/mert_svm_test_confusion_matrix.png`.

## Limitaciones

- La evaluacion es a nivel de fragmento en AIME.
- El score SVM no es una probabilidad calibrada.
- Este resultado no demuestra generalizacion fuera de AIME.
- El desglose por generador es diagnostico y no se uso para modificar modelo, umbral ni hiperparametros.

## Veredicto

La evaluacion individual del enfoque profundo queda cerrada en test con la configuracion previamente fijada. El siguiente paso es realizar una comparacion formal con MFCC + SVM en una issue distinta, sin decidir todavia el modelo de despliegue.
