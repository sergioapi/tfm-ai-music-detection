# Baseline MFCC + SVM

## Que se implemento

Baseline clasico binario a nivel de fragmento con preprocesamiento comun, MFCC y SVM RBF. El pipeline entrenado incluye `StandardScaler` y `SVC(kernel="rbf")`.

## Datos utilizados

- Manifiesto: `data/aime_splits.csv`.
- Dataset: `disco-eth/AIME`.
- Revision AIME: `b84d4be5eda830b6eb714998569dba73530f2601`.
- Clases: `0` musica humana de MTG-Jamendo; `1` musica generada por IA.

## Control de fuga de informacion

Las particiones existentes se respetan sin regenerar el manifiesto. La seleccion de hiperparametros usa solo validacion; test se usa una unica vez para evaluacion final. No se reentrena con `train + val`.

## Configuracion seleccionada

- `C`: `10`.
- `gamma`: `0.01`.

## Metricas

| split | balanced accuracy | precision IA | recall IA | F1 IA | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 0.7800 | 0.7625 | 0.8133 | 0.7871 | 0.8480 |
| test | 0.8333 | 0.8472 | 0.8133 | 0.8299 | 0.9086 |

## Tiempos y artefactos

- Extraccion de caracteristicas: `2461.7319 s`.
- Entrenamiento y busqueda: `0.7314 s`.
- Inferencia validacion: `0.0119 s`.
- Inferencia test: `0.0135 s`.
- Latencia media test: `0.000090 s/fragmento`.
- Tamano del modelo: `139462` bytes.
- Memoria RSS: `No registrada: no se anade dependencia nueva para medir memoria.`

## Incidencias

- Ejemplos procesados: `1000`.
- Ejemplos fallidos registrados: `0`.

## Limitaciones

- AIME se evalua a nivel de fragmento.
- `description` es una agrupacion semantica, no un identificador verificable de cancion.
- La clase humana procede unicamente de MTG-Jamendo.
- La clase IA combina 12 generadores.
- El score SVM no es una probabilidad.
- Un buen resultado dentro de AIME no demuestra generalizacion a otros datasets o generadores.
