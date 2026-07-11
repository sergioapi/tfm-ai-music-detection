# Resumen de auditoria de AIME

## Objetivo

Validar si `disco-eth/AIME` puede utilizarse como dataset principal del TFM para una tarea binaria de deteccion de musica generada por IA:

- `label = 0`: audio humano procedente de `MTG-Jamendo`.
- `label = 1`: audio generado por IA, correspondiente al resto de modelos.

La auditoria no entrena modelos finales. Su objetivo es decidir si el dataset permite continuar con el experimento previsto: baseline `MFCC + SVM` frente a un enfoque con encoder profundo preentrenado.

## Resultado general

Veredicto provisional:

`AIME apto con condiciones`

AIME es tecnicamente valido para continuar, siempre que el experimento aplique un preprocesamiento comun y respete particiones por `description`.

## Resultados principales

### Metadatos

- Dataset auditado: `disco-eth/AIME`.
- Revision observada: `b84d4be5eda830b6eb714998569dba73530f2601`.
- Filas: `6500`.
- Columnas: `id`, `model`, `description`, `audio`.
- Ejemplos humanos (`MTG-Jamendo`): `500`.
- Ejemplos IA: `6000`.
- Generadores IA: `12`.
- Cada modelo/fuente contiene `500` ejemplos.
- `description` distintas: `500`.
- Cada `description` contiene exactamente `13` filas:
  - `1` ejemplo `MTG-Jamendo`;
  - `12` ejemplos IA;
  - los `13` valores de `model`.
- No se observaron nulos relevantes.
- No se observaron cadenas vacias relevantes.
- No se observaron `id` duplicados.

### Etiquetas

La regla binaria prevista es coherente con los metadatos:

- `MTG-Jamendo` obtiene siempre `label = 0`.
- Todos los demas modelos obtienen siempre `label = 1`.

Esto permite construir el problema experimental humano/IA de forma directa.

### Agrupacion

`description` es la unidad de agrupacion disponible en AIME. No identifica una cancion original, sino un grupo semantico compartido por ejemplos humanos e IA.

Decision:

- Las particiones futuras deben hacerse por `description`.
- No deben hacerse particiones aleatorias por fila.
- Todos los ejemplos asociados a una misma `description` deben permanecer en la misma particion.

Motivo: evitar fugas entre train, validacion y test.

### Trazabilidad de `id`

Se comprobo la relacion con `disco-eth/AIME-survey`.

Resultado:

- `track-1-id` y `track-2-id` de AIME-survey corresponden a valores de `AIME.id`.
- `id` funciona como identificador interno de clip AIME.
- No se encontro un campo oficial que identifique de forma explicita:
  - cancion original de MTG-Jamendo;
  - artista;
  - album;
  - ruta original;
  - `song_id` original verificable.

Decision:

- No se tratara `id` como identificador de cancion original.
- La memoria debe explicar que AIME se trabaja a nivel de fragmento/clip, no a nivel de cancion completa verificable.

## Audio y preprocesamiento

### Microprueba y aceptacion

Se uso la `description`:

`ambient, blues, piano`

Se descargaron y decodificaron correctamente `13` audios:

- `1` humano (`MTG-Jamendo`);
- `12` IA;
- `13` modelos/fuentes distintos.

Los audios quedaron en una ruta ignorada por Git:

`data/audio/aime_acceptance_raw/`

### Heterogeneidad observada

Los audios brutos no son homogeneos. Se observaron diferencias en:

- duracion;
- sample rate;
- numero de canales;
- subtipo WAV;
- tamano de archivo.

Ejemplos relevantes:

- `MTG-Jamendo`, `id=06001`:
  - WAV `PCM_16`;
  - `48000 Hz`;
  - `2` canales;
  - `256.130625 s`;
  - `49,177,158` bytes.
- `AudioLDM 2 Large`, `id=01631`:
  - WAV `FLOAT`;
  - `16000 Hz`;
  - `1` canal;
  - `10 s`;
  - `640,058` bytes.

Decision:

- No se deben usar los audios brutos directamente como entrada comparable.
- El experimento debe aplicar un preprocesamiento comun antes de extraer MFCC o embeddings.

### Preprocesamiento validado

Se valido un pipeline minimo:

- lectura con `soundfile`;
- conversion a `float32`;
- mezcla a mono mediante media de canales;
- recorte de `10 s`;
- si el audio dura mas de `10 s`, seleccion de ventana por maxima energia media;
- resample a `16000 Hz`;
- salida final de `160000` muestras.

Resultado:

- `13 / 13` audios generaron segmentos validos.
- No se observaron NaN.
- No se observaron infinitos.
- Todos los segmentos finales tuvieron `160000` muestras.

Decision:

- El recorte de `10 s` con normalizacion tecnica es viable.
- El sample rate `16 kHz` queda como decision provisional de auditoria.
- El sample rate definitivo podra depender del encoder profundo elegido.
- El recorte por energia es un pipeline practico del TFM, no una reproduccion garantizada del algoritmo interno de AIME.

## MFCC

Se extrajeron MFCC sobre los `13` fragmentos preprocesados:

- `n_mfcc = 20`;
- media por coeficiente;
- desviacion tipica por coeficiente;
- vector final de `40` valores por audio.

Resultado:

- `X_mfcc.shape == (13, 40)`.
- Todos los valores fueron finitos.

Decision:

- AIME es compatible con el baseline clasico `MFCC + SVM`.
- No se entreno ningun clasificador durante la auditoria.

## Embeddings profundos

Se realizo un smoke test tecnico con:

`MIT/ast-finetuned-audioset-10-10-0.4593`

Este modelo se uso solo para comprobar compatibilidad tecnica con un encoder preentrenado. No queda seleccionado como modelo definitivo del TFM.

Resultado:

- `X_embed.shape == (13, 768)`.
- Todos los valores fueron finitos.
- Ejecucion en CPU.
- No se entreno el modelo.
- No se hizo fine-tuning.
- No se guardaron embeddings en disco.

Decision:

- AIME permite obtener embeddings de un encoder preentrenado sobre fragmentos normalizados.
- La seleccion del encoder definitivo queda pendiente para la fase experimental.

## Viabilidad del subconjunto inicial

La configuracion inicial prevista de `500` humanos y `500` IA es viable:

- existen exactamente `500` ejemplos humanos;
- existen `6000` ejemplos IA;
- existen `500` `description` distintas;
- cada `description` dispone de ejemplos IA;
- es posible seleccionar una IA por cada una de `500` `description` distintas.

Decision:

- El siguiente artefacto debe ser `data/aime_splits.csv`.
- La seleccion debe respetar agrupacion por `description`.
- La seleccion IA debera controlar, en la medida de lo posible, la distribucion entre generadores.

## Decisiones finales

1. Usar AIME como dataset principal del TFM, con veredicto `apto con condiciones`.
2. Formular la tarea como clasificacion binaria humano/IA a nivel de fragmento.
3. Usar `MTG-Jamendo` como clase humana.
4. Usar los 12 generadores restantes como clase IA.
5. Particionar siempre por `description`, nunca por fila.
6. Aplicar preprocesamiento comun antes de cualquier extraccion de caracteristicas.
7. Validar el baseline `MFCC + SVM` sobre fragmentos normalizados.
8. Evaluar un unico encoder profundo preentrenado en la fase experimental, todavia por seleccionar.
9. No presentar el sistema futuro como certificador absoluto de canciones completas, sino como analizador de fragmentos con posible agregacion de scores.

## Riesgos y condiciones a documentar

- AIME no expone un identificador original verificable de cancion MTG-Jamendo.
- `description` agrupa semanticamente, pero no equivale a `song_id`.
- Los audios brutos son tecnicamente heterogeneos.
- La clase humana procede solo de `MTG-Jamendo`.
- La clase IA agrupa modelos con caracteristicas tecnicas y generativas distintas.
- El preprocesamiento comun puede influir en el resultado experimental.
- La futura aplicacion debera comunicar incertidumbre y agregacion por fragmentos.

## Siguiente paso

Crear `data/aime_splits.csv` con una seleccion reproducible y sin fugas:

- `500` ejemplos humanos;
- `500` ejemplos IA;
- particiones por `description`;
- una estrategia documentada para seleccionar una IA por `description`;
- control de distribucion por generador en la seleccion IA.
