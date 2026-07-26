# Seleccion MERT congelado + SVM

## Objetivo

Entrenar y seleccionar un clasificador supervisado sencillo sobre embeddings MERT ya extraidos, manteniendo el encoder congelado y sin recalcular embeddings.

## Artefacto de entrada

- Parquet: `data/processed/aime_mert_embeddings.parquet`.
- Filas: `1000`.
- Forma de embeddings: `(1000, 768)`.
- Dtype Parquet de embeddings: `float`.
- Valores finitos: `true`.

## Separacion train/val/test

- Train usado para ajuste: `700` ejemplos.
- Validacion usada para seleccion: `150` ejemplos.
- Test cargado solo para validacion estructural: `150` ejemplos.
- No se calcularon metricas predictivas de test.
- No se generaron predicciones ni matriz de confusion de test.

## Candidatos evaluados

Se evaluaron unicamente pipelines `StandardScaler` + `SVC(probability=False)` con kernel lineal o RBF.

| candidato | kernel | C | gamma | balanced accuracy val | precision IA val | recall IA val | F1 IA val | ROC-AUC val | train s | pred val s | latencia val s/ej | tamano bytes |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svm_linear | linear | 0.01 | - | 0.8467 | 0.8421 | 0.8533 | 0.8477 | 0.9083 | 0.0871 | 0.0250 | 0.00016697 | 1993085 |
| svm_linear | linear | 0.1 | - | 0.8667 | 0.8395 | 0.9067 | 0.8718 | 0.9218 | 0.1139 | 0.0185 | 0.00012319 | 1693580 |
| svm_linear | linear | 1 | - | 0.8667 | 0.8395 | 0.9067 | 0.8718 | 0.9212 | 0.1611 | 0.0196 | 0.00013080 | 1660221 |
| svm_linear | linear | 10 | - | 0.8667 | 0.8395 | 0.9067 | 0.8718 | 0.9212 | 0.1232 | 0.0312 | 0.00020795 | 1660221 |
| svm_rbf | rbf | 0.1 | scale | 0.6667 | 0.6344 | 0.7867 | 0.7024 | 0.7225 | 0.1367 | 0.1192 | 0.00079453 | 4310733 |
| svm_rbf | rbf | 0.1 | 0.001 | 0.6400 | 0.6364 | 0.6533 | 0.6447 | 0.7243 | 0.1245 | 0.1318 | 0.00087892 | 4310717 |
| svm_rbf | rbf | 0.1 | 0.01 | 0.5733 | 0.5433 | 0.9200 | 0.6832 | 0.7568 | 0.1493 | 0.1217 | 0.00081154 | 4335373 |
| svm_rbf | rbf | 1 | scale | 0.7267 | 0.7576 | 0.6667 | 0.7092 | 0.8352 | 0.1399 | 0.1011 | 0.00067389 | 3638861 |
| svm_rbf | rbf | 1 | 0.001 | 0.7400 | 0.7727 | 0.6800 | 0.7234 | 0.8384 | 0.1207 | 0.1032 | 0.00068786 | 3503229 |
| svm_rbf | rbf | 1 | 0.01 | 0.6200 | 0.5804 | 0.8667 | 0.6952 | 0.7524 | 0.1414 | 0.1364 | 0.00090951 | 4335373 |
| svm_rbf | rbf | 10 | scale | 0.7867 | 0.8028 | 0.7600 | 0.7808 | 0.8825 | 0.1355 | 0.0940 | 0.00062686 | 3645021 |
| svm_rbf | rbf | 10 | 0.001 | 0.8200 | 0.8158 | 0.8267 | 0.8212 | 0.8919 | 0.1782 | 0.1279 | 0.00085242 | 3386109 |
| svm_rbf | rbf | 10 | 0.01 | 0.6200 | 0.5833 | 0.8400 | 0.6885 | 0.7390 | 0.1343 | 0.1005 | 0.00066982 | 4335373 |

## Regla de seleccion

La configuracion se selecciono maximizando `balanced_accuracy` en validacion; en empate se uso `ROC-AUC`, despues `F1 IA`, despues menor latencia media de prediccion redondeada de forma determinista y, si persistia el empate, preferencia por el kernel lineal.

## Configuracion seleccionada

- Candidato: `svm_linear`.
- Kernel: `linear`.
- `C`: `0.1`.
- `gamma`: `-`.

## Metricas de validacion

- Balanced accuracy: `0.8667`.
- Precision IA: `0.8395`.
- Recall IA: `0.9067`.
- F1 IA: `0.8718`.
- ROC-AUC: `0.9218`.
- Matriz de confusion val `[label 0, label 1]`: `[[62, 13], [7, 68]]`.

## Tiempos y tamano

- Tiempo total de busqueda: `3.1545 s`.
- Tiempo de entrenamiento seleccionado: `0.1139 s`.
- Tiempo total de prediccion val seleccionado: `0.0185 s`.
- Latencia media val seleccionada: `0.00012319 s/ejemplo`.
- Tamano del modelo seleccionado: `1693580` bytes.

## Artefactos

- Modelo: `data/models/mert_svm_selection_model.joblib`.
- Resultados JSON: `data/models/mert_svm_selection_results.json`.
- Predicciones de validacion: `data/models/mert_svm_validation_predictions.csv`.
- Matriz de confusion de validacion: `data/models/mert_svm_validation_confusion_matrix.png`.

## Limitaciones

- La seleccion se basa solo en validacion y no mide todavia rendimiento final.
- No se reentreno con `train + val` en esta fase.
- El score SVM procede de `decision_function` y no es una probabilidad calibrada.

## Siguiente paso

Cerrar la configuracion, decidir si procede reentrenar con `train + val` y evaluar una unica vez sobre test para comparar formalmente con el baseline clasico.
