# Seleccion del modelo profundo preentrenado MERT

- Estado: aceptada para la configuracion experimental inicial.
- Fecha: 2026-07-23.
- Issue relacionada: "Seleccionar, adaptar y evaluar el modelo profundo preentrenado".
- Fase cubierta: fase 1, preparacion y trazabilidad.

## Contexto

El TFM compara un baseline clasico `MFCC + SVM` con un unico enfoque basado en un modelo profundo preentrenado para una tarea binaria de Synthetic Song Detection. La clase `0` representa musica humana o bona fide y la clase `1` representa musica generada mediante IA. La tarea no se formula como Singing Voice Deepfake Detection, deteccion de voz hablada sintetica, deteccion de audio sintetico general, deteccion de letras generadas ni identificacion del generador concreto.

El dataset principal ya esta fijado: `disco-eth/AIME`, subconjunto de `1000` ejemplos con `500` humanos y `500` IA, `12` generadores de IA, particiones fijas en `data/aime_splits.csv`, distribucion `700/150/150` para train/validacion/test, particionado por `description` y semilla `42`. El manifiesto no debe regenerarse para esta fase.

El baseline `MFCC + SVM RBF` ya esta implementado sobre clips de `10 s` a `16 kHz`. Se resume aqui solo como referencia de comparacion para el modelo profundo:

| split | balanced accuracy | precision IA | recall IA | F1 IA | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| validacion | 0.7800 | 0.7625 | 0.8133 | 0.7871 | 0.8480 |
| test | 0.8333 | 0.8472 | 0.8133 | 0.8299 | 0.9086 |

Estos artefactos y resultados quedan intactos.

## Restricciones

La decision debe ser reproducible sin descargar ni ejecutar todavia el modelo. En fases posteriores se deberan medir las restricciones reales de hardware y calendario, especialmente:

- memoria GPU disponible, incluyendo el escenario objetivo de `6 GB` de VRAM;
- viabilidad de inferencia en CPU;
- tiempo de extraccion completa de embeddings;
- compatibilidad con el despliegue que entonces se preveía en Hugging Face Spaces;
- tiempo restante hasta la convocatoria de septiembre de 2026.

No se considera verificado que MERT quepa en `6 GB` de VRAM ni que su inferencia en CPU sea aceptable.

## Alternativas consideradas

- `m-a-p/MERT-v1-95M`: modelo orientado a comprension musical, con version de aproximadamente `95M` parametros y representaciones de `768` dimensiones.
- `MIT/ast-finetuned-audioset-10-10-0.4593`: alternativa de contingencia basada en AST, ya usada anteriormente solo como smoke test tecnico en la auditoria de AIME.
- Modelos generales de audio: descartados en esta decision inicial por menor ajuste al dominio musical cuando existe una opcion especifica de musica.

## Decision

Se selecciona `m-a-p/MERT-v1-95M` como modelo profundo principal del TFM.

La alternativa de contingencia sera `MIT/ast-finetuned-audioset-10-10-0.4593`, pero solo se activara si MERT resulta tecnicamente inviable o deja de ser adecuado segun los criterios de abandono. AST no se implementa ni se evalua en esta fase.

## Justificacion

La seleccion de MERT se basa en:

- preentrenamiento especifico para comprension musical;
- tamano aproximado de `95M` parametros, mas viable que variantes mayores;
- arquitectura documentada con `12` capas y dimension oculta `768`;
- sample rate de entrenamiento documentado de `24 kHz`;
- disponibilidad publica de pesos, codigo y model card;
- posibilidad de usar el encoder completamente congelado;
- adecuacion esperada a extraccion previa de embeddings;
- mayor cercania al dominio del TFM que modelos entrenados solo para audio general.

Esta decision no implica que el rendimiento final este demostrado. El valor experimental se comprobara primero en validacion y el test permanecera bloqueado hasta cerrar la configuracion.

## Estrategia de adaptacion inicial

La adaptacion inicial usara el encoder congelado como extractor de embeddings. La inferencia debera ejecutarse con `model.eval()` y sin gradientes. No se plantea fine-tuning parcial ni completo en esta fase.

La configuracion inicial queda registrada en `configs/mert_frozen_embeddings.yaml`.

## Entrada, ventanas y pooling

- Entrada mono en `float32`.
- Sample rate de MERT: `24000 Hz`.
- Unidad comun de comparacion: clip de `10 s`.
- Procesamiento interno: `2` ventanas contiguas de `5 s`.
- Pooling por ventana: media temporal del ultimo estado oculto.
- Agregacion entre ventanas: media de los dos embeddings.
- Dimension esperada del embedding final: `768`.

La eleccion de ventanas de `5 s` sigue el contexto de preentrenamiento documentado para MERT-v1-95M. Su efecto real sobre rendimiento, memoria y latencia queda pendiente de medicion.

## Clasificador y seleccion

El clasificador inicial sera un pipeline `StandardScaler` + `SVC(kernel="rbf")`, con la misma rejilla inicial que el baseline:

- `C`: `[0.1, 1, 10, 100]`;
- `gamma`: `["scale", 0.001, 0.01, 0.1]`.

La seleccion de configuracion usara exclusivamente train y validacion. La particion de test no se consultara hasta cerrar la configuracion experimental. No se reentrenara ni se evaluara ningun clasificador en esta fase documental.

## Licencias y trazabilidad

Pesos de `m-a-p/MERT-v1-95M`:

- Licencia verificada en la model card oficial de Hugging Face: `cc-by-nc-4.0`.
- Implicacion para el TFM academico: uso razonable para investigacion y evaluacion academica no comercial, manteniendo atribucion.
- Implicacion para un modelo derivado: cualquier publicacion o distribucion derivada debe revisar y respetar la restriccion no comercial de los pesos.
- Atribucion: necesaria en memoria, repositorio y cualquier artefacto publico asociado.

Repositorio/codigo `yizhilll/MERT`:

- Licencia verificada en GitHub: `Apache-2.0`.
- Implicacion: el codigo tiene una licencia separada de los pesos. La licencia permisiva del repositorio no elimina la restriccion no comercial de los pesos de Hugging Face.

Publicacion:

- La pagina oficial de ICLR 2024 identifica MERT como paper de conferencia ICLR 2024.
- La model card enlaza el identificador `arXiv:2306.00107`.

Revision del modelo:

- Revision inmutable registrada: `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.
- Fuente: pagina oficial de archivos de Hugging Face para `m-a-p/MERT-v1-95M` en esa revision.
- Observacion: antes del primer smoke test se recomienda repetir la comprobacion mediante la API o `git ls-remote` y fijar tambien las versiones de librerias usadas para cargar el modelo.

## Riesgos

- El uso de `trust_remote_code=True` introduce dependencia en codigo remoto que debe revisarse antes de ejecutar.
- La memoria GPU real puede ser insuficiente aunque el encoder este congelado.
- La inferencia CPU podria ser demasiado lenta para el calendario o para el prototipo.
- El coste de extraccion completa de embeddings puede superar el margen temporal disponible.
- El pooling simple puede perder informacion temporal relevante para musica.
- El modelo puede aprender senales de dataset o generador en lugar de propiedades generales de musica IA.
- Las licencias pueden limitar una publicacion posterior del modelo derivado.

## Criterios de abandono

MERT se abandonara y se activara la contingencia AST en fases posteriores si ocurre alguno de estos casos:

1. No puede cargarse de forma reproducible con una revision y versiones fijadas.
2. Produce OOM con batch `1`, encoder congelado y ventanas de `5 s`.
3. Genera embeddings no finitos o errores sistematicos.
4. Su extraccion completa excede los limites temporales que se definan para el TFM.
5. Su latencia o memoria en CPU resulta incompatible con el entorno de despliegue que se seleccione.
6. Existen impedimentos de licencia o publicacion.
7. Su rendimiento en validacion es claramente inferior al baseline tras aplicar la configuracion inicial predefinida.

Estos criterios son reglas para fases posteriores, no resultados observados en esta fase.

## Pendiente de verificar

- Carga reproducible real con revision fijada y versiones concretas de `transformers`, `torch` y dependencias auxiliares.
- Necesidad exacta de `trust_remote_code` en la version de librerias que se use.
- Consumo de VRAM con batch `1`, encoder congelado y ventanas de `5 s`.
- Latencia y memoria en CPU.
- Tiempo total de extraccion para los `1000` ejemplos.
- Finitud y forma de los embeddings sobre muestras reales del manifiesto.
- Rendimiento en validacion frente al baseline.
- Condiciones de publicacion del artefacto derivado si se distribuyen pesos, embeddings o un clasificador entrenado sobre ellos.

## Fuentes oficiales

- Model card: https://huggingface.co/m-a-p/MERT-v1-95M
- Revision fijada: https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5
- Config del modelo: https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/config.json
- Repositorio oficial: https://github.com/yizhilll/MERT
- Publicacion ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html
- OpenReview indicado para trazabilidad: https://openreview.net/forum?id=w3YZ9MSlBu
