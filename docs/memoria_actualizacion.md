# Actualizacion estructural de la memoria

## Alcance vigente

- Clasificacion binaria entre musica humana y musica generada mediante IA.
- Uso de AIME como dataset principal.
- Comparacion entre un baseline MFCC + SVM y un unico modelo profundo preentrenado.
- Evaluacion mediante metricas predictivas y rendimiento operacional.
- Seleccion del modelo con mejor compromiso para una prueba de concepto web.
- Aplicacion planteada como prototipo academico, no como producto SaaS, detector forense ni sistema infalible.
- Convocatoria de septiembre de 2026.

## Decisiones confirmadas

- Clasificacion binaria entre musica humana y musica generada mediante IA.
- AIME como dataset principal.
- Comparacion entre MFCC + SVM y un unico modelo profundo preentrenado.
- Seleccion del modelo segun metricas predictivas y rendimiento operacional.
- Desarrollo de una prueba de concepto web.
- React + Vite como frontend desplegado en Vercel.
- FastAPI y Python como backend desplegado.
- Northflank Developer Sandbox como hosting seleccionado del backend; el contrato vigente está en `docs/despliegue_backend.md`.
- Vercel como hosting del frontend.
- Convocatoria de septiembre de 2026.

## Decisiones provisionales

- Uso de fragmentos de 10 segundos fuera del baseline ya implementado.
- Frecuencia de muestreo comun para ambos modelos.
- Publicacion del modelo en Hugging Face Hub.
- Arquitectura concreta de inferencia de la aplicacion.

## Decisiones pendientes

- Modelo profundo.
- Embeddings congelados, fine-tuning parcial u otra estrategia de adaptacion.
- Estrategia de agregacion por cancion.
- Calibracion o interpretacion del score.
- Flujo sincrono o asincrono.
- Limites de tamano, duracion y formatos.
- Necesidad de Redis, solo si existe evidencia operacional.

## Estado de cada capitulo

- Introduccion: requiere adaptar objetivo, alcance y estructura al planteamiento binario vigente.
- Estudio previo: pendiente de redactar con objetivos, metodologia, planificacion y presupuesto.
- Estado del arte: requiere revision de foco y bibliografia antes de considerarse vigente.
- Descripcion de la propuesta: pendiente de desarrollar a partir del pipeline real y la arquitectura prevista.
- Validacion: pendiente de integrar baseline, modelo profundo y criterios operacionales.
- Conclusiones: pendiente de redactar cuando la validacion este cerrada.
- Resumen y abstract: pendientes de redactar al final.
- Agradecimientos: pendiente de completar.

## Fuentes de verdad del repositorio

- `README.md`.
- `docs/aime_audit_summary.md`.
- `docs/mfcc_svm_baseline_summary.md`.
- `data/aime_splits.csv`.
- `data/models/mfcc_svm_metrics.json`.
- `data/models/mfcc_svm_predictions.csv`.
- `scripts/create_aime_splits.py`.
- `scripts/mfcc_svm_baseline.py`.
- `scripts/extract_aime_mfcc.py`.
- `scripts/train_mfcc_svm.py`.
- `tests/test_aime_splits.py`.
- `tests/test_mfcc_svm_baseline.py`.

## Criterios de estilo y redaccion

- Espanol formal y academico, pero natural.
- Nivel propio de un estudiante de master, no de un investigador experto.
- Evitar lenguaje rimbombante o excesivamente rebuscado.
- Evitar repeticiones de palabras, argumentos y estructuras proximas.
- Variar de forma natural la longitud de frases y parrafos.
- No comenzar varios parrafos consecutivos de la misma manera.
- Cada parrafo debe desarrollar una idea principal.
- Distinguir entre trabajo realizado, diseno previsto, decision provisional y cuestion pendiente.
- No presentar como implementado lo que todavia no se ha desarrollado.
- No inventar referencias, resultados, justificaciones ni decisiones.

## Tareas pendientes de la memoria

- Actualizar la introduccion al alcance vigente.
- Completar el estudio previo con decisiones ya confirmadas y pendientes.
- Auditar y normalizar el estado del arte con referencias verificadas.
- Redactar la descripcion de la propuesta cuando el modelo profundo este decidido.
- Integrar los resultados del baseline y del modelo profundo en validacion.
- Definir como se documentara la prueba de concepto web.
- Completar resumen, abstract, agradecimientos y conclusiones al final del proceso.
