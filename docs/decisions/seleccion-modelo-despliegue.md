# Seleccion del modelo de despliegue para el MVP

- Estado: aceptada para el MVP web inicial.
- Issue relacionada: "Comparar enfoques y seleccionar el modelo de despliegue".
- Fase cubierta: comparacion de resultados existentes y decision de integracion.

## Contexto

El TFM dispone de dos enfoques ya implementados y evaluados sobre el mismo manifiesto AIME: `MFCC + StandardScaler + SVM RBF` y `MERT congelado + StandardScaler + SVM lineal` sobre embeddings. La seleccion para el MVP debe basarse exclusivamente en la evidencia existente, sin reentrenar, recalcular embeddings, optimizar umbrales ni consultar nuevas metricas experimentales.

La particion de test contiene 150 clips balanceados y se utiliza como evidencia principal de rendimiento final. La validacion se usa para explicar la seleccion de configuraciones, no para sustituir el resultado de test.

## Alternativas consideradas

1. `MFCC + StandardScaler + SVM RBF`, con `C=10` y `gamma=0.01`.
2. `m-a-p/MERT-v1-95M` congelado + `StandardScaler + SVM lineal`, con `C=0.1`.

## Evidencia predictiva

- MFCC + SVM test: balanced accuracy `0.8333`, precision IA `0.8472`, recall IA `0.8133`, F1 IA `0.8299`, ROC-AUC `0.9086`, matriz `[[64, 11], [14, 61]]`.
- MERT + SVM test: balanced accuracy `0.8200`, precision IA `0.8158`, recall IA `0.8267`, F1 IA `0.8212`, ROC-AUC `0.8985`, matriz `[[61, 14], [13, 62]]`.
- MFCC obtiene `125` aciertos, `11` falsos positivos y `14` falsos negativos.
- MERT obtiene `123` aciertos, `14` falsos positivos y `13` falsos negativos.
- MERT fue mejor en validacion y obtiene un recall IA ligeramente superior en test, pero la ventaja de validacion no se mantiene en el resultado final.

## Evidencia operacional

- El joblib de MFCC ocupa `139462` bytes y evita cargar un encoder profundo.
- La SVM de MERT ocupa `1693580` bytes, pero ese valor corresponde solo al clasificador sobre embeddings y no al pipeline completo.
- El snapshot local aproximado del encoder MERT ocupa `377578388` bytes.
- La inferencia registrada del encoder MERT es aproximadamente `0.8239 s/clip` de 10 segundos en CPU.
- No existe todavia una medicion extremo a extremo comparable por cancion.
- Los tiempos con streaming remoto no son comparables con inferencia local.

## Decision

Se selecciona `MFCC + StandardScaler + SVM RBF` como modelo inicial para el MVP web.

## Justificacion

MFCC + SVM se selecciona porque obtiene mejores resultados en la mayoria de las metricas finales de test, logra dos aciertos mas, produce menos falsos positivos, presenta un artefacto mucho mas pequeno, tiene menos dependencias, evita cargar un encoder profundo y reduce el riesgo de cold start, memoria y despliegue. Hugging Face Spaces fue el destino previsto durante esta decisión inicial; el MVP se desplegó finalmente con FastAPI en Northflank, sin que ello reabra la comparación de modelos.

Esta decision no afirma que MFCC generalice mejor. Los datos solo permiten afirmar que rindio ligeramente mejor dentro del test de AIME utilizado en el protocolo experimental.

## Consecuencias

- El backend reproducira exactamente el preprocesamiento MFCC usado en los experimentos.
- El pipeline joblib se cargara una vez durante el arranque de FastAPI.
- El flujo sincrono sera la opcion inicial por simplicidad, pendiente de confirmacion durante la integracion.
- Redis no se incorpora en esta fase.
- La salida no se presentara como probabilidad calibrada.
- La interfaz debera mostrar una estimacion o puntuacion acompanada de advertencias.
- El sistema no se presentara como detector forense.

## Riesgos

- Los resultados proceden de clips AIME de 10 segundos.
- El comportamiento sobre canciones completas requiere una estrategia de agregacion aun no definida.
- La generalizacion a otros datasets, generadores o condiciones de audio no esta demostrada.
- El preprocesamiento del MVP debe reproducir fielmente el pipeline experimental para evitar desviaciones.

## Limitaciones

- No se ha realizado una prueba inferencial de significacion estadistica.
- Los scores proceden de `decision_function` y no son probabilidades calibradas.
- No hay benchmark extremo a extremo del MVP.
- No se fijan todavia duracion maxima, tamano maximo, timeout, numero de fragmentos ni metodo definitivo de agregacion.

## Cuestiones abiertas

- Limites de tamano y duracion de subida.
- Timeout aceptable para la experiencia web.
- Numero de fragmentos por cancion.
- Estrategia de agregacion por cancion.
- Validación operacional real del backend seleccionado (completada en Northflank; véase `docs/despliegue_backend.md`).

## Condiciones para revisar la decision

La decision debera revisarse si una evaluacion posterior sobre datos externos favorece claramente a MERT, si el MVP exige maximizar recall IA por encima del resto de metricas, si se obtiene una medicion extremo a extremo donde MERT sea operacionalmente viable sin degradar la experiencia web, o si el baseline MFCC muestra fallos sistematicos durante la integracion.

## Estado de MERT

MERT no se considera un experimento fallido. Fue viable en CPU, obtuvo mejores resultados en validacion, alcanzo resultados proximos al baseline en test y mejoro ligeramente el recall IA. Se conserva como parte de la comparacion experimental del TFM y como referencia para trabajos posteriores.
