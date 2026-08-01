# Comparacion de enfoques y seleccion del modelo de despliegue

## 1. Objetivo y alcance

Este documento compara formalmente los dos enfoques ya implementados para la deteccion binaria de musica humana frente a musica generada por IA en AIME: el baseline MFCC + StandardScaler + SVM RBF y MERT congelado + StandardScaler + SVM lineal sobre embeddings. La comparacion utiliza exclusivamente los artefactos estructurados existentes; no se reentrena, no se recalculan embeddings, no se modifican umbrales y no se ejecutan nuevas predicciones.

## 2. Protocolo comun

- Manifiesto: `data/aime_splits.csv`.
- Ejemplos: `1000`.
- Particiones: train `700`, validacion `150`, test `150`.
- Clase positiva: IA, etiqueta `1`.
- Ambos enfoques usan los mismos `150` IDs de test y las etiquetas reales coinciden con el manifiesto.
- Los scores proceden de `decision_function`; no son probabilidades calibradas.

## 3. Configuracion final de cada enfoque

- MFCC + SVM: `StandardScaler` + `SVC(kernel="rbf", C=10, gamma=0.01, probability=False)`, sobre estadisticos de MFCC.
- MERT + SVM: encoder `m-a-p/MERT-v1-95M` congelado, dos ventanas de `5 s`, pooling temporal del ultimo hidden state, media entre ventanas, embedding de `768` dimensiones y `StandardScaler` + `SVC(kernel="linear", C=0.1, probability=False)`.

## 4. Metricas de validacion

| Modelo | Balanced accuracy | Precision IA | Recall IA | F1 IA | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MFCC + SVM | 0.7800 | 0.7625 | 0.8133 | 0.7871 | 0.8480 |
| MERT + SVM | 0.8667 | 0.8395 | 0.9067 | 0.8718 | 0.9218 |

## 5. Metricas de test

| Modelo | Balanced accuracy | Precision IA | Recall IA | F1 IA | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MFCC + SVM | 0.8333 | 0.8472 | 0.8133 | 0.8299 | 0.9086 |
| MERT + SVM | 0.8200 | 0.8158 | 0.8267 | 0.8212 | 0.8985 |

## 6. Diferencias MERT menos MFCC

| Split | Balanced accuracy | Precision IA | Recall IA | F1 IA | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validacion | 0.0867 | 0.0770 | 0.0933 | 0.0847 | 0.0738 |
| Test | -0.0133 | -0.0314 | 0.0133 | -0.0087 | -0.0101 |

## 7. Matrices de confusion

- MFCC + SVM test: `[[64, 11], [14, 61]]`.
- MERT + SVM validacion: `[[62, 13], [7, 68]]`.
- MERT + SVM test: `[[61, 14], [13, 62]]`.

## 8. Aciertos, falsos positivos y falsos negativos

| Modelo | Split | Aciertos | Falsos positivos | Falsos negativos |
| --- | ---: | ---: | ---: | ---: |
| MFCC + SVM | test | 125 | 11 | 14 |
| MERT + SVM | test | 123 | 14 | 13 |

## 9. Comparacion operacional

- MFCC registra ` 2461.7319 s` de extraccion total de caracteristicas, ` 0.7314 s` de entrenamiento y busqueda, y ` 0.00008974 s/fragmento` para la SVM sobre MFCC ya calculados.
- El pipeline joblib de MFCC ocupa `139462` bytes (0.14 MB). La memoria RSS del baseline no esta registrada.
- MERT registra ` 0.2350 s/clip` de preprocesamiento medio y ` 0.8239 s/clip` de inferencia aproximada del encoder para clips de 10 segundos.
- La carga registrada de MERT es ` 0.6254 s` para el procesador y ` 0.6107 s` para el encoder.
- La SVM de MERT sobre embeddings ya calculados registra ` 0.00011298 s/ejemplo` y su joblib ocupa `1693580` bytes (1.69 MB).
- El snapshot local aproximado del encoder MERT ocupa `377578388` bytes (377.58 MB) y el RSS despues de cargar el encoder fue `499437568` bytes.

Advertencias de comparabilidad:

- Los `1693580` bytes de MERT corresponden solo al `StandardScaler` + SVM, no al pipeline profundo completo.
- El pico RSS aproximado de `10.9 GB` corresponde a la extraccion masiva y no representa la memoria necesaria para una peticion del MVP.
- Los tiempos con streaming remoto no son comparables con inferencia local.
- La latencia de la SVM de MERT no incluye el encoder.
- La latencia de la SVM del baseline no incluye la extraccion de MFCC.
- No existe todavia una medicion extremo a extremo comparable por cancion.

## 10. Interpretacion

MERT fue mejor en validacion: mejora la balanced accuracy, el recall IA, el F1 IA y el ROC-AUC frente al baseline. Sin embargo, esa ventaja no se mantuvo en test. En la particion final, MFCC + SVM obtiene mejor balanced accuracy, precision IA, F1 IA y ROC-AUC. MERT obtiene un recall IA ligeramente superior, pero tambien produce mas falsos positivos.

MFCC + SVM alcanza `125` aciertos sobre `150`, mientras que MERT + SVM alcanza `123`. Las diferencias predictivas son pequenas y no se ha realizado una prueba inferencial, por lo que no se afirma significacion estadistica. La lectura prudente es que MFCC rindio ligeramente mejor dentro del test de AIME, no que generalice mejor fuera de ese contexto.

Operacionalmente, MERT introduce mayor tamano, mas dependencias y mayor complejidad de despliegue por la necesidad de cargar y ejecutar un encoder profundo. Esta complejidad no queda compensada por una mejora predictiva global en test.

## 11. Limitaciones y amenazas a la validez

- Los resultados corresponden a clips AIME de 10 segundos.
- La generalizacion a canciones completas, otros datasets o generadores no esta garantizada.
- Los scores no son probabilidades calibradas.
- No hay medicion comparable extremo a extremo por cancion.
- La agregacion de fragmentos para el MVP permanece abierta.
- Los limites de duracion, tamano y timeout permanecen pendientes de pruebas de integracion.

## 12. Conclusion comparativa

La evidencia disponible permite seleccionar `MFCC + StandardScaler + SVM RBF` como modelo inicial para el MVP web. La seleccion se apoya en el resultado final de test y en la menor complejidad operacional. MERT no se considera un experimento fallido: fue viable en CPU, obtuvo mejores resultados en validacion, alcanzo resultados proximos al baseline en test y mejoro ligeramente el recall IA. Se conserva como parte de la comparacion experimental del TFM, pero no se selecciona para el despliegue inicial.
