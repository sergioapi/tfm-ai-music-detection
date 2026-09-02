# Roadmap histórico de cierre del despliegue de VeriSon

## 1. Estado de cierre y propósito histórico

Este documento conserva el contexto, las decisiones y el backlog que guiaron el cierre del despliegue del MVP. Se elaboró a partir de la inspección del repositorio del 25 de agosto de 2026 y de la evidencia operacional posterior. Ya no es un roadmap operativo ni la fuente de configuración vigente.

El contrato operativo vigente está en `docs/despliegue_backend.md`. La validación manual E2E T12 está en curso; no debe marcarse como completada hasta que exista su evidencia final.

> Las secciones posteriores describen el estado y las decisiones de su momento. Las referencias a tareas pendientes, profiling, Render o configuraciones anteriores son históricas y no describen el runtime actual.

### Cierre conocido

- T03, T04 y T05: completadas y validadas; el warm-up de resampling y MFCC se coordina con `/ready`, mientras `/health` mantiene su semántica operativa/modelo.
- T08: completada; dos análisis largos simultáneos finalizaron con HTTP 200, sin OOM ni reinicio.
- T11: completada; se retiraron `MemoryProfiler`, `psutil`, RSS, IDs y logs diagnósticos. `InferenceTimings` se conserva como contrato funcional.
- Validación automática de la release candidate: backend 162 tests sin warnings, frontend 29 tests, lint/build correctos y build Docker correcto.
- T12: en curso, pendiente de la evidencia manual E2E final.

Definiciones empleadas:

- **Obligatoria:** sin ella no es correcto cerrar el MVP o su despliegue.
- **Recomendable:** aporta robustez o calidad clara, aunque el MVP podría cerrarse justificadamente sin ella.
- **Opcional:** optimización que aporta valor, pero no resuelve un problema crítico actual.
- **Condicional:** solo se ejecuta si una decisión o evidencia concreta la activa.
- **Crítica:** bloquea la utilización correcta de la aplicación pública.
- **Alta:** afecta significativamente a robustez, rendimiento o cierre técnico.
- **Media:** mejora relevante, pero no urgente.
- **Baja:** optimización marginal o futura.

## 2. Estado actual del sistema

### Producto y alcance

VeriSon es una prueba de concepto web desarrollada para un TFM sobre detección binaria de música humana frente a música generada mediante IA. Su salida es una estimación. No es un detector forense, una certificación de autenticidad, un veredicto jurídico ni un identificador fiable del generador concreto.

### Arquitectura y despliegue

- **Frontend:** React, Vite y TypeScript, desplegado en Vercel en `https://verison-app.vercel.app`.
- **Backend:** FastAPI sobre Python 3.11.9, desplegado en Northflank Developer Sandbox (`Europe - West (London)`) en `https://api--verison-api--xb7vy98gqd48.code.run`.
- **API pública:** `GET /health`, `GET /api/v1/model` y `POST /api/v1/analyze`.
- **Configuración frontend:** `VITE_API_BASE_URL` determina el backend. La capa actual concatena esa base con `/api/v1/analyze`.
- **Configuración backend:** `MODEL_PATH`, `CORS_ALLOWED_ORIGINS`, `MAX_UPLOAD_SIZE_BYTES`, `MAX_AUDIO_DURATION_SECONDS`, `RESAMPLE_WARMUP_ENABLED` y `TEMP_DIR`.
- **Artefacto:** `data/models/mfcc_svm_baseline.joblib` está versionado y el backend valida su estructura y calcula su SHA-256 al cargarlo.
- **Despliegue como código:** `deploy/backend/Dockerfile` y `deploy/backend/Dockerfile.dockerignore` empaquetan el backend de forma agnóstica al proveedor; el contrato operativo vigente se resume en `docs/despliegue_backend.md`. No se usan archivos específicos de proveedor.

### Modelo e inferencia

- Pipeline: MFCC + `StandardScaler` + SVM RBF.
- Parámetros SVM: `C=10`, `gamma=0.01`.
- Fragmentos consecutivos de 10 s; último fragmento rellenado cuando está incompleto.
- Sample rate objetivo: 16 kHz.
- 20 MFCC, agregados mediante media y desviación típica por coeficiente: 40 características.
- Clase humana `0`; clase IA `1`; threshold `0.0`.
- Score: `decision_function`, no probabilidad calibrada.
- Agregación por canción: media del score de decisión ponderada por duración real de cada fragmento.

### Flujo frontend actual

`App.tsx` mantiene estados `idle`, `selected`, `analyzing`, `success` y `error`. Al pulsar “Analizar audio”, llama directamente a `analyzeAudio(file)`. La petición `fetch` al `POST /api/v1/analyze` no tiene timeout ni cancelación, una carencia de robustez general que puede dejar la UI esperando indefinidamente. Tampoco existe aún el preflight de readiness que, tras un deploy/restart, debe esperar al warm-up antes del envío. El frontend no reintenta el POST, muestra errores de red y API mediante mensajes controlados y permite analizar otra canción mediante reset. Los tests cubren el flujo feliz, error de red, formatos y presentación del resultado, pero no las esperas finitas ni la transición separada “Preparando el servicio” → “Analizando”.

### Flujo backend actual

- El modelo se carga una vez durante el lifespan de FastAPI. Si falla, la API arranca degradada y `/health` devuelve `503`.
- El audio subido se copia por bloques a un temporal con límite de 64 MiB y se elimina en éxito o error.
- La duración se valida mediante metadatos de SoundFile contra un límite por defecto de 300 s.
- La inferencia abre el audio una vez y procesa secuencialmente cada fragmento: lectura, conversión a `float32`, mono, selección/relleno, resampling si procede, MFCC y conservación solo de features y metadatos.
- SoundFile lee actualmente cada fragmento como `float64`; después `validate_decoded_audio` lo convierte a `float32`.
- El warm-up experimental de resampling está implementado, desactivado por defecto y se ejecuta en un thread de background si `RESAMPLE_WARMUP_ENABLED=true`.
- `/health` considera listo el servicio en cuanto el artefacto del modelo está cargado. No representa el estado del warm-up; por ello puede devolver `200` durante sus aproximadamente 130 s de ejecución.

### Memoria y rendimiento

El OOM principal se considera mitigado para el MVP gracias al streaming secuencial. La evidencia histórica de Render Free registró picos RSS aproximados de 400–440 MiB sin reproducir el OOM anterior. En el entorno final Northflank (`0.2 vCPU`, `512 MiB`), un WAV de `256.130625 s` y 26 fragmentos completó con `200`, pico RSS `436.2 MiB`, sin OOM ni reinicio.

Las mediciones de Render se conservan como referencia histórica y no se extrapolan. En Northflank, el mismo audio largo caliente tardó aproximadamente `5.75 s`; una petición fría observada tardó aproximadamente `101.27 s` (`90.25 s` de preprocessing y `10.32 s` de MFCC). Esta evidencia abre T03, pero no decide todavía warm-up, MFCC ni readiness.

### Tests y documentación inspeccionados

El backend posee tests unitarios y de integración para configuración, salud, uploads, formatos, límites, temporales, carga de modelo, score, agregación, streaming, equivalencia con la lectura completa, MFCC y profiling. El frontend posee tests de componente y flujo principal. No existe una prueba E2E automatizada Vercel → backend público ni una prueba formal de concurrencia. La prueba manual Vercel → Northflank realizada en T02 no sustituye la validación E2E final de T12.

## 3. Decisiones cerradas

Las siguientes decisiones no deben reabrirse durante esta fase salvo petición explícita del usuario o evidencia técnica nueva y documentada:

1. MFCC + `StandardScaler` + SVM RBF es el modelo del MVP.
2. Configuración: fragmentos de 10 s, 16 kHz, 20 MFCC, mean/std, 40 características, `C=10`, `gamma=0.01`.
3. Clase humana `0`, clase IA `1`, threshold `0.0`.
4. El score es `decision_function`; no es una probabilidad calibrada.
5. La salida debe describirse como estimación y nunca como veredicto forense.
6. El streaming secuencial por fragmentos ya está implementado.
7. El OOM principal está suficientemente mitigado para el MVP; no se rediseña el streaming sin una regresión reproducible.
8. WAV y MP3 son los formatos admitidos y ya funcionan.
9. Límites actuales de referencia: 64 MiB y 300 s.
10. El warm-up de resampling está experimentalmente validado: ejecuta la misma ruta `librosa.resample`/`soxr_hq` y elimina el coste frío del resampling posterior.
11. No cambiar modelo, dataset, splits, threshold, calibración, protocolo experimental, MFCC ni agregación mean/std por defecto.
12. No incorporar Redis, Celery, colas, autenticación, pagos, persistencia de análisis, Kubernetes ni MLOps completo sin un requisito nuevo y explícito.

## 4. Evidencia experimental de despliegue

La evidencia de este apartado combina lo comprobable en el repositorio con mediciones operacionales de Render (históricas) y Northflank (hosting seleccionado). No constituye un benchmark universal ni permite extrapolar automáticamente entre proveedores. Estas pruebas demuestran procesamiento técnico, respuestas HTTP y comportamiento de memoria/latencia; no validan que la etiqueta predicha para una canción individual sea semánticamente verdadera.

### 4.1 Streaming y memoria

La implementación real usa `SoundFile` como iterador de fragmentos y conserva únicamente features de 40 valores y metadatos por fragmento antes de construir la matriz final. Existen tests de equivalencia del decoder secuencial y del servicio streaming frente al flujo anterior de lectura completa.

Resultados observados en Render Free tras el cambio:

- audio de unos 120 s: procesamiento completado;
- audio de unos 225 s: procesamiento completado;
- audio de unos 256,13 s y 26 fragmentos: procesamiento completado con respuesta `200 OK`;
- canción de más de cuatro minutos: procesamiento completado;
- pico RSS observado: aproximadamente 400–440 MiB;
- ejemplo: `audio_duration_seconds≈256.13`, `n_fragments=26`, pico RSS `≈440.8 MiB`, `POST /api/v1/analyze → 200`.

Conclusión: el OOM original no bloquea el MVP. Sigue existiendo poco margen de RAM para dos inferencias concurrentes, pero primero debe medirse en T08.

### 4.2 Resampling frío

Versiones verificadas en Render durante el diagnóstico:

```text
Python      3.11.9
librosa     0.11.0
numpy       2.4.6
scipy       1.17.1
soxr        1.1.0
numba       0.67.0
llvmlite    0.49.0
scikit-learn 1.8.0
```

Ruta efectiva:

```text
librosa.resample
→ librosa.core.audio.resample
→ soxr_hq
→ python-soxr
→ soxr_ext / libsoxr
```

Medición en proceso frío:

```text
resolve ≈ 38–41 s
execute ≈ 89–91 s
total   ≈ 129–130 s
```

Se confirmó lazy loading y coste inicial del backend soxr. Las siguientes ejecuciones son rápidas. No hay evidencia de que Numba JIT sea la causa principal de esta ruta concreta.

### 4.3 Warm-up de resampling

El warm-up usa una señal sintética mono `float32` de un segundo, 48 kHz → 16 kHz, llama exactamente a `librosa.resample` con el backend por defecto `soxr_hq`, no cambia el preprocessing real y no ejecuta MFCC. Se agenda mediante `asyncio.to_thread`, por lo que no bloquea el event loop.

```text
warm-up total ≈ 130 s
```

Después de `resample_warmup status=completed`:

```text
resample_resolve = 0.0000 s
resample_execute ≈ 0.0023 s
preprocess primer fragmento ≈ 0.0985 s
```

Una primera canción real de unos 225 s/23 fragmentos terminó con `200` en aproximadamente 31 s. El experimento demuestra eficacia técnica, no conveniencia operacional: durante el warm-up, `/health` ya puede devolver `200`.

### 4.4 MFCC

El coste frío restante de la primera extracción MFCC es:

```text
primera MFCC ≈ 12–14.5 s
ejemplo mfcc_profile = 12.2967 s
```

Las siguientes extracciones son mucho más rápidas. La causa interna no se ha diagnosticado. Su estudio solo se activa en el hosting elegido si la latencia fría resultante incumple el objetivo operativo acordado; no justifica cambiar `n_mfcc`, el algoritmo ni las 40 features.

### 4.5 Validación inicial en Northflank

El 27 de agosto de 2026 se validó `verison-api` en Northflank Developer Sandbox, región `Europe - West (London)`, plan `nf-compute-20` (`0.2 vCPU shared`, `512 MiB RAM`, una instancia y `1 GiB` efímero). La imagen Docker se construye desde la raíz con `deploy/backend/Dockerfile`; Python es `3.11.9` y las dependencias críticas están fijadas en `backend/requirements.txt`.

- `GET /health` y `GET /api/v1/model` devolvieron `200`;
- el SHA-256 del artefacto fue `ee4359aa9f9942a1179184a28834c5a1b6d901253ac82bce90a32472451a0336`;
- WAV y MP3 cortos devolvieron `200`; CORS permitió `https://verison-app.vercel.app` y rechazó `https://example.com`;
- el flujo manual Vercel → Northflank procesó un MP3 de aproximadamente 120 s y devolvió `200`; no equivale a T12;
- el proceso `pid=3` siguió vivo y caliente tras aproximadamente 7 h 39 min sin inferencias; un MP3 posterior respondió en aproximadamente 0.76 s;
- la facturación mostrada durante esta prueba fue “No usage / You have not accrued any costs yet”; es una observación puntual, no una garantía futura.

Render permanece disponible temporalmente como rollback. La prueba fría y la decisión sobre sus aproximadamente `101.27 s` se tratan exclusivamente en T03.

## 5. Problemas pendientes conocidos

1. El frontend no aplica timeout/cancelación al POST ni recupera la UI ante una espera infinita; además, aún no consulta readiness antes del POST para esperar a que termine el warm-up después de un deploy/restart.
2. La decisión de alcance es consolidar el warm-up; T03 debe validarlo y medir su presupuesto real en Northflank.
3. `/health` mezcla liveness y disponibilidad del modelo, pero no informa de warm-up en curso, completado o fallido.
4. La primera petición fría en Northflank registró aproximadamente 90.25 s de preprocessing y 10.32 s de MFCC; el coste residual de MFCC se acepta para el MVP salvo nueva evidencia bloqueante.
5. No se ha medido el comportamiento con dos análisis simultáneos; con 512 MiB y un pico observado de 436.2 MiB, el riesgo es concreto.
6. La instrumentación temporal de memoria, runtime, preprocessing y MFCC continúa en el código y mantiene `psutil` como dependencia.
7. No existe una validación E2E final del frontend público contra el backend final.

## 6. Inventario de mejoras

| ID | Mejora | Clasificación | Prioridad | Estado | Dependencia principal |
| -- | ------ | ------------- | --------- | ------ | --------------------- |
| T01 | Evaluar y cerrar el hosting del backend | obligatoria | crítica | completada | DG01 resuelta: Northflank seleccionado |
| T02 | Migrar y verificar el backend en el hosting elegido | condicional | alta | completada | Northflank validado; Render retenido como rollback |
| T03 | Cerrar la estrategia operacional de cold start y resampling | obligatoria | alta | completada | Warm-up validado en Northflank |
| T04 | Acotar esperas del frontend y, si aplica, preparar el backend | obligatoria | crítica | completada | Preflight `/ready`, timeout y recuperación validados |
| T05 | Separar liveness y readiness de forma mínima | condicional | alta | completada | `/ready` representa el estado posterior a warm-ups |
| T06 | Diagnosticar y decidir la primera MFCC | condicional | media | descartada | Omitida para el MVP; no hay evidencia bloqueante |
| T07 | Fijar el contrato reproducible de runtime y despliegue | obligatoria | alta | completada | Runtime y contrato Northflank verificados |
| T08 | Medir concurrencia mínima en el entorno final | recomendable | media | completada | Dos análisis largos simultáneos sin OOM ni reinicio |
| T09 | Evaluar lectura directa `float32` | opcional | baja | condicional | DG05; evidencia de memoria/copia |
| T10 | Evaluar un resampler alternativo | condicional | media | descartada | Se mantiene `soxr_hq`; no hay evidencia bloqueante |
| T11 | Retirar instrumentación temporal y cerrar observabilidad mínima | obligatoria | alta | completada | Profiling temporal y dependencia `psutil` retirados |
| T12 | Ejecutar y documentar la validación E2E final | obligatoria | crítica | en curso | Pendiente de evidencia manual E2E |

Distribución revisada: **12 tareas: 6 obligatorias, 1 recomendable, 1 opcional y 4 condicionales**. T04 es obligatoria por su núcleo de robustez general; su subalcance de preflight continúa siendo condicional. “Bloqueada” indica dependencia lógica, no impedimento técnico permanente.

## 7. Mejoras obligatorias

- **T01 — Hosting:** sin un entorno definitivo no puede cerrarse el despliegue ni decidirse qué mitigaciones de cold start tienen sentido.
- **T03 — Cold start/resampling:** consolidar el warm-up sintético de la ruta actual tras deploy/restart y medirlo en Northflank; no se evalúa otro resampler.
- **T04 — Espera finita del frontend:** cualquier petición de análisis debe terminar en éxito técnico o error controlado, sin dejar la UI indefinidamente en “Analizando”. Este núcleo de timeout, cancelación y recuperación es independiente de Render. El preflight para despertar el backend no forma parte obligatoria de todas las ramas.
- **T07 — Reproducibilidad:** el entorno científico y el contrato de despliegue deben ser repetibles para evitar cambios silenciosos e incompatibilidad del joblib.
- **T11 — Limpieza:** los hooks diagnósticos se introdujeron como temporales y no deben quedar indefinidamente en la versión final.
- **T12 — E2E final:** es la evidencia de que el sistema público completo funciona en frío, en caliente, con límites, errores y formatos reales.

T05 y el preflight de T04 están activados porque el warm-up permanecerá en background. No se usan para despertar Northflank: esperan a que el proceso recién creado termine el warm-up antes de enviar el único POST. El timeout/cancelación y la recuperación de la UI siguen siendo obligatorios con cualquier hosting.

## 8. Mejoras recomendables

### T08 — Medir concurrencia mínima

Es recomendable porque el endpoint síncrono puede ejecutar trabajo en threads concurrentes y el margen de RAM observado es estrecho. La primera acción correcta es medir dos peticiones simultáneas controladas, no introducir colas, workers, semáforos o rate limiting. El MVP puede cerrarse sin esta tarea si se documenta explícitamente que la demo está dimensionada para uso individual y T12 no muestra inestabilidad.

## 9. Mejoras opcionales

### T09 — Lectura directa `float32`

El OOM principal ya está resuelto. Leer `float32` directamente podría reducir una copia y el tamaño del fragmento temporal, pero puede alterar waveform, MFCC, score o clasificaciones cercanas a cero. No bloquea el cierre. Solo merece ejecutarse si T08 o nuevas mediciones demuestran que el margen de memoria/copia sigue siendo un problema relevante. Requiere equivalencia explícita y no habilita reentrenamiento por defecto.

## 10. Mejoras condicionales

- **T02 — Migración:** se activa únicamente si DG01 concluye que Render Free no es adecuado y el usuario aprueba un destino concreto.
- **T04 — Subalcance de preflight:** activado. `ensureBackendReady()` hará GET idempotentes y acotados al endpoint de readiness tras deploy/restart; nunca reintenta el POST.
- **T05 — Readiness:** activada porque “HTTP vivo” no equivale a “inferencia preparada” mientras el warm-up se ejecuta en background.
- **T06 — Primera MFCC:** descartada para el MVP salvo nueva evidencia bloqueante.
- **T10 — Resampler alternativo:** descartada para el MVP; se mantiene `soxr_hq` con warm-up.

## 11. Decision gates

### DG01 — Resuelta: Northflank seleccionado

**Resuelta el 27 de agosto de 2026.** Northflank Developer Sandbox es el hosting seleccionado para el backend de la demo académica. La decisión se apoyó en la compatibilidad con el contenedor provider-agnostic, el plan `nf-compute-20` (`0.2 vCPU shared`, `512 MiB RAM`, una instancia y `1 GiB` efímero), la región `Europe - West (London)`, el servicio always-on observado y la validación funcional registrada en la sección 4.5. Render se conserva temporalmente como rollback.

T02 verificó el contrato operativo en el destino. T03 debe volver a medir y decidir el comportamiento frío en Northflank; las mediciones de Render permanecen históricas. DG01 solo condiciona el preflight de T04: el núcleo general que impide esperas infinitas no depende de esta decisión.

### DG02 — Estrategia definitiva de warm-up/resampling

**Decisión de alcance:** consolidar el warm-up sintético existente de la ruta `librosa.resample`/`soxr_hq` como mecanismo de startup. Northflank se mantiene caliente en operación normal; tras deploy/restart evita que el primer usuario pague el coste frío dominante del resampling. El warm-up no ejecuta MFCC: puede permanecer un coste frío residual de la primera MFCC, aceptado para el MVP y sin investigación salvo nueva evidencia bloqueante.

T03 debe validar esta configuración con `RESAMPLE_WARMUP_ENABLED=true` mediante restart controlado, registrar su presupuesto y mantener el mismo preprocessing, resampler, MFCC, modelo y features. No se usa audio real como warm-up. T05 debe consolidar después el contrato de readiness: mientras no exista, `/health` puede devolver `200` durante el warm-up. La configuración definitiva de release será warm-up habilitado más readiness implementada; T03 no requiere mantener un release final público intermedio con warm-up habilitado y sin readiness.

### DG03 — Primera MFCC

**Decisión de alcance:** T06 queda descartada para este MVP. No se investigará, precargará ni modificará la primera MFCC salvo nueva evidencia bloqueante. No cambiar MFCC, `n_mfcc`, mean/std ni las 40 features.

### DG04 — Readiness

**Decisión de alcance:** mantener `/health` como liveness/modelo y añadir en T05 un endpoint mínimo de readiness que represente el estado del warm-up. T04 lo consultará antes del único POST para esperar un deploy/restart, no para despertar una instancia dormida. Si el warm-up falla, el endpoint debe permitir terminar el preflight con un error controlado sin filtrar detalles internos.

### DG05 — Lectura `float32`

Activar T09 únicamente ante evidencia de que una copia/uso de memoria residual afecta al entorno final o a la concurrencia mínima. Si no existe esa evidencia, descartar o posponer. La equivalencia numérica y de clasificación es condición de adopción.

### DG06 — Posible cambio de resampler

**Decisión de alcance:** T10 queda descartada para el MVP. Se conserva `soxr_hq` y el warm-up de su ruta actual; no se evaluará un resampler alternativo salvo nueva evidencia bloqueante.

## 12. Dependencias entre tareas

Flujo principal:

```text
T01 — Decidir hosting
 ↓
T07 — Iniciar: definir/fijar runtime científico crítico
 │
 ├── Mantener Render ───────────────→ T07 — Cerrar sobre Render ─┐
 │                                                              │
 └── Cambiar hosting → T02 — Migrar → T07 — Cerrar en destino ──┤
                                                                ↓
                                  T03 — Cerrar cold start/resampling
                                   │
                                   ├── DG06 activo → T10 → volver a T03
                                   │
                                   ├── DG03 activo → T06
                                   │
                                   ├── warm-up en background → T05 ─┐
                                   │                                │
                                   └── T04 — núcleo general          │
                                       timeout/cancelación           │
                                       (siempre)                     │
                                                                    │
                         spin-down/primer POST problemático ─────────┘
                                   → T04 — preflight GET (condicional)

T03 + T04 completa + T05/T06/T10 activos resueltos
 │
 ├── T08 — Concurrencia mínima (recomendable)
 │      └── evidencia de memoria → DG05 → T09 (opcional)
 │
 └── T11 — Retirar instrumentación temporal
          ↓
       T12 — E2E final
```

Reglas de dependencia:

1. T07 comienza después de T01 con el runtime científico crítico, independientemente de que se mantenga Render o se migre.
2. Si se cambia de hosting, T02 depende de esa fase inicial de T07; después T07 depende de T02 para verificar y cerrar el contrato operativo real. T07 no se marca completada entre ambas etapas.
3. T03 requiere T07 cerrada. Si se migra, no usa mediciones de Render como decisión final y espera a T02 y al cierre posterior de T07.
4. T06 y T10 no se ejecutan antes de T03.
5. T05 debe incorporar todos los warm-ups finalmente elegidos; por eso se ejecuta después de T03/T06.
6. El núcleo general de T04 siempre se ejecuta y usa los presupuestos temporales finales. Su preflight solo consume el contrato de readiness de T05 cuando ambos subalcances están activos; nunca reintenta el POST.
7. T11 espera a que T03, T06, T08, T09 y T10 estén completadas o descartadas, para no retirar métricas aún necesarias.
8. T12 es la última tarea técnica y solo prueba la configuración definitiva.

## 13. Roadmap por fases

### Fase A — Cerrar infraestructura y reproducibilidad

- **Objetivo:** elegir el entorno definitivo y hacer su runtime repetible.
- **Tareas:** T01 → inicio de T07 (runtime científico); si se mantiene Render, cierre de T07 sobre Render; si se migra, T02 → cierre de T07 sobre el destino verificado.
- **Decisión cerrada:** hosting, Python, dependencias directas críticas, artefacto, variables y comandos efectivos.
- **Salida:** backend final desplegable de forma reproducible; no se optimiza aún cold start.

### Fase B — Cerrar disponibilidad en frío

- **Objetivo:** definir el comportamiento desde proceso frío hasta inferencia utilizable.
- **Orden:** T03 → T05 → T04 → T08 → T09 solo si T08 activa DG05.
- **Decisiones de alcance:** warm-up de resampling actual; T05/T04 activos; T06/T10 descartados; presupuestos finitos y capacidad se cerrarán en sus tareas respectivas.
- **Salida:** toda petición abandona la espera en tiempo finito; si hay cold start problemático, el flujo de preparación también queda acotado y nunca duplica el POST.

### Fase C — Robustez proporcional y limpieza

- **Objetivo:** medir el riesgo mínimo de concurrencia y eliminar diagnóstico temporal.
- **Orden:** T08 recomendada → T09 solo si se activa → T11.
- **Decisión cerrada:** capacidad declarada de la demo, adopción o descarte de `float32`, observabilidad final y destino de `psutil`.
- **Salida:** código de producción limpio con solo logs/métricas justificadas.

### Fase D — Validación y cierre

- **Objetivo:** demostrar el funcionamiento público de extremo a extremo.
- **Tarea:** T12.
- **Decisión cerrada:** aceptar o no el despliegue como MVP final.
- **Salida:** matriz E2E fechada, entorno identificado, resultados reales y defectos bloqueantes resueltos o cierre rechazado.

## 14. Fichas de implementación

### T01 — Evaluar y cerrar el hosting del backend

**Estado:** completada\
**Clasificación:** obligatoria\
**Prioridad:** crítica

**Resultado:** DG01 seleccionó Northflank Developer Sandbox para `verison-api`; Render sigue disponible temporalmente como rollback. La evidencia y el contrato operativo se registran en la sección 4.5 y en `docs/despliegue_backend.md`.

**Objetivo:** producir una decisión explícita y trazable: mantener Render Free o seleccionar otro hosting concreto para la demo académica.

**Justificación:** las decisiones de cold start, warm-up, readiness y el preflight de la UX dependen del entorno. La protección general frente a una espera infinita de T04 no depende de esta elección.

**Dependencias:** ninguna.\
**Decision gate previo:** ninguno; esta tarea resuelve DG01.\
**Condición de activación:** siempre.

**Alcance:** inventariar configuración y métricas actuales; comparar solo alternativas plausibles con CPU, RAM, coste/free tier, billing, scale-to-zero, cold start, timeout, warm instance, despliegue, FastAPI, artefacto, complejidad y valor académico; definir criterios; registrar decisión y consecuencias.

**Fuera de alcance:** migrar, cambiar código de inferencia, contratar gasto sin autorización, evaluar una lista extensa de proveedores.

**Áreas probablemente afectadas:** documentación de decisiones en `docs/`, README y memoria solo para registrar la decisión. No requiere cambios funcionales.

**Cambios conceptuales esperados:** ADR/decisión de infraestructura con alternativa elegida, motivos, costes/restricciones y tareas activadas/descartadas.

**Pruebas futuras:** comprobaciones manuales o probes comparables de cold start; no benchmark predictivo.

**Criterios de aceptación:** decisión inequívoca; evidencia fechada; matriz de criterios; URL/plan objetivo; impacto sobre T02–T06/T10; autorización separada si implica coste o tarjeta.

**Riesgos:** comparar planes desactualizados, decidir solo por latencia o asumir que un free tier mantiene instancias calientes.

**Rollback:** revertir únicamente la decisión documental si nueva evidencia la invalida antes de migrar.

**Documentación/memoria:** alta; sustituir “previsto” por la decisión real y distinguir evaluación de resultado.

**Issue independiente:** sí.\
**Commit independiente:** sí, si solo documenta la decisión.\
**Mensaje conceptual de commit:** `docs: decide el hosting definitivo del backend`

### T02 — Migrar y verificar el backend en el hosting elegido

**Estado:** completada\
**Clasificación:** condicional\
**Prioridad:** alta

**Resultado:** el mismo FastAPI, artefacto y contrato API se validaron en Northflank. Health, modelo/SHA-256, WAV, MP3, CORS permitido/rechazado, audio de 256.130625 s, permanencia caliente y flujo manual Vercel → Northflank funcionaron sin OOM ni reinicio. Los valores observados se conservan en la sección 4.5; no sustituyen T03 ni T12.

**Objetivo:** desplegar el mismo FastAPI, modelo y contrato API en el hosting aprobado, con el menor cambio operativo posible.

**Justificación:** las mediciones de Render no son transferibles a otro CPU/runtime.

**Dependencias:** T01 y la fase inicial de T07, que define/fija el runtime científico crítico; autorización del usuario para cualquier cuenta, billing o cambio externo. T02 no requiere que T07 esté cerrada.\
**Decision gate previo:** DG01 = cambiar hosting.\
**Condición de activación:** alternativa concreta aprobada.

**Alcance:** desplegar sobre el runtime crítico definido al iniciar T07; empaquetado mínimo exigido por el proveedor; variables; artefacto; CORS; health/model/analyze; URL; logs de arranque; smoke tests WAV/MP3; devolver a T07 la evidencia efectiva de versiones, build y start command; rollback documentado.

**Fuera de alcance:** cambiar modelo/preprocessing, rediseñar API, añadir base de datos, colas, autenticación o observabilidad empresarial.

**Áreas probablemente afectadas:** `backend/`, archivo mínimo de despliegue si procede, `README.md`, documentación/memoria y configuración Vercel de la URL.

**Cambios conceptuales esperados:** solo adaptador/configuración de hosting y documentación; mismo contrato público.

**Pruebas futuras:** health, model, WAV/MP3 corto, CORS desde Vercel, artefacto SHA-256, límites básicos y reinicio frío.

**Criterios de aceptación:** URL pública estable; respuestas compatibles; artefacto y metadatos de modelo esperados; frontend puede contactar; logs no exponen rutas sensibles; la evidencia del runtime real permite cerrar T07; Render permanece disponible para rollback hasta aceptar el destino.

**Riesgos:** diferencias de timeout/filesystem, costes, variables ausentes, incompatibilidad binaria de dependencias.

**Rollback:** restablecer `VITE_API_BASE_URL` a Render y retirar el despliegue nuevo según el proveedor, sin borrar evidencia.

**Documentación/memoria:** registrar proveedor, plan, región si importa, fecha, limitaciones y coste real.

**Issue independiente:** sí.\
**Commit independiente:** sí.\
**Mensaje conceptual de commit:** `chore: adapta el backend al hosting seleccionado`

### T03 — Cerrar la estrategia operacional de cold start y resampling

**Estado:** pendiente\
**Clasificación:** obligatoria\
**Prioridad:** alta

**Problema:** Northflank evita el spin-down periódico, pero un deploy/restart crea un proceso frío. La primera inferencia fría observada tardó aproximadamente 101 s; el warm-up sintético debe consolidarse para que el primer usuario no pague el coste frío dominante del resampling. Puede persistir un coste frío residual de MFCC, aceptado para el MVP.

**Objetivo:** validar el warm-up sintético de resampling al startup, sin cambiar la ruta de producción, y entregar a T05 la evidencia necesaria para que el release final lo use junto con readiness.

**Justificación:** una solución técnicamente eficaz puede ser inadecuada si el usuario cree que el servicio está listo durante más de dos minutos.

**Dependencias:** T07 cerrada; por tanto, T02 completada y verificada previamente si se activa la migración.\
**Decision gate previo:** DG01 resuelto; DG02 ya fija consolidar el warm-up.\
**Condición de activación:** siempre, en el entorno final.

**Alcance:** validar `RESAMPLE_WARMUP_ENABLED=true` mediante restart controlado en Northflank; conservar la señal sintética mínima 48 kHz → 16 kHz que llama a la misma ruta `librosa.resample`/`soxr_hq`; medir startup, warm-up y primera/segunda inferencia tras restart; conservar la instrumentación temporal necesaria. T05 representa después el estado del warm-up y T04 espera ese estado antes del POST. T03 no declara como release final un despliegue público con warm-up habilitado y sin readiness.

**Fuera de alcance:** volver a demostrar que el warm-up calienta soxr, cambiar resampler directamente, cambiar preprocessing o modelo.

**Áreas probablemente afectadas:** configuración de despliegue, `backend/app/config.py`, `backend/app/main.py` solo si la decisión requiere consolidar/retirar el flag, tests de lifecycle y docs.

**Cambios conceptuales esperados:** consolidar el warm-up actual y registrar su duración; no se activa T10 y no se modifica preprocessing.

**Pruebas futuras:** al menos dos ciclos fríos comparables, warm request de control, audio que requiera 48→16 kHz, captura de estado y tiempos.

**Criterios de aceptación:** warm-up validado con `RESAMPLE_WARMUP_ENABLED=true` en restart controlado de Northflank; duración y comportamiento tras restart registrados; mismo preprocessing y resampler; coste frío residual de MFCC aceptado sin activar T06; T05/T04 preparados para consumir readiness; ninguna afirmación de probabilidad.

**Riesgos:** confundir spin-up de proveedor con lazy loading, medir un proceso ya caliente o optimizar para una única observación.

**Rollback:** volver al flag desactivado o a la configuración anterior; el preprocessing no cambia.

**Documentación/memoria:** registrar cold/warm por separado y explicar limitaciones del plan.

**Issue independiente:** sí.\
**Commit independiente:** sí si cambia configuración; la medición/decisión también puede ser un commit documental propio.\
**Mensaje conceptual de commit:** `perf: cierra la estrategia de warm-up del resampling`

### T04 — Acotar esperas del frontend y, si aplica, preparar el backend

**Estado:** bloqueada\
**Clasificación:** obligatoria; el preflight es un subalcance condicional\
**Prioridad:** crítica

**Problema:** el `POST /api/v1/analyze` no tiene timeout/cancelación y cualquier petición que no termine puede dejar el frontend indefinidamente en “Analizando”. Tras un deploy/restart de Northflank, el warm-up debe terminar antes de enviar el archivo.

**Objetivo:** garantizar que toda petición abandona la espera en tiempo finito, recupera un estado de UI utilizable y nunca duplica el análisis. El preflight consulta readiness mediante GET antes de enviar el archivo una única vez.

**Justificación:** timeout, cancelación, error finito y recuperación son robustez general. El preflight evita que el primer usuario posterior a deploy/restart pague mediante su POST el coste frío dominante del resampling; no elimina el posible coste frío residual de MFCC aceptado para el MVP.

**Dependencias:** T03 para fijar presupuestos temporales compatibles con el entorno final; T05 únicamente si se activa el preflight y necesita readiness.\
**Decision gate previo:** ninguno para el núcleo general; DG01/DG02 y el contrato health/readiness definitivo solo para el preflight.\
**Condición de activación:** el núcleo de timeout/cancelación, error finito y recuperación se implementa siempre. `ensureBackendReady()`, el preflight y los reintentos GET están activados para esperar al warm-up después de deploy/restart.

**Alcance:** como núcleo general, timeout/cancelación controlada del POST basado en el límite real, error accionable, salida de `analyzing`, reset y reintento manual. Si el gate de hosting activa el subalcance: `ensureBackendReady()` con timeout por intento, reintentos solo de GET idempotentes, máximo total, backoff sencillo acotado, estado “Iniciando el servicio…” seguido de “Analizando…” y posible prewarm al cargar sin bloquear la UI.

**Fuera de alcance:** reintentar automáticamente `POST /api/v1/analyze` bajo cualquier circunstancia, subir el archivo durante health, reintentar métodos no idempotentes, service workers, colas, polling complejo u ocultar una espera ilimitada.

**Áreas probablemente afectadas:** `frontend/src/api/`, `App.tsx`, `AudioAnalysisForm.tsx`, estilos mínimos, tests de frontend y configuración de URL.

**Cambios conceptuales esperados:** `AbortController` o equivalente y transición a error recuperable para el POST en todos los hostings; exactamente un POST por acción del usuario. Solo si se activa el preflight: estados separados `starting-backend` y `analyzing`, y GET de disponibilidad repetible.

**Pruebas futuras:** núcleo general: POST que no termina, timeout/cancelación, error finito, salida de `analyzing`, un único POST, fallo sin retry, reset y accesibilidad. Preflight condicional: health tarda y luego responde, 503/errores de red, máximo total, solo GET reintentados y transición `starting-backend` → `analyzing`.

**Criterios de aceptación:** ante cualquier petición bloqueada, el usuario obtiene procesamiento completado o error accionable en tiempo finito y la UI queda recuperable. Nunca se reintenta automáticamente el POST y existe como máximo un POST por acción. Si el hosting requiere despertar, solo se reintentan GET idempotentes y el POST se envía una vez tras la disponibilidad. Los tests verifican conteo de llamadas y transiciones.

**Riesgos:** timeout general menor que un análisis largo válido; confundir su presupuesto con el de cold start; tormenta de probes al cargar muchas pestañas; activar preflight sin necesidad; usar liveness cuando se necesita readiness.

**Rollback:** retirar el preflight si el proveedor no lo necesita conservando el timeout/cancelación general. Solo ante una regresión demostrada se revierte el núcleo, restaurando temporalmente la llamada directa mientras se corrige; no hay cambios en datos.

**Documentación/memoria:** distinguir el manejo general de peticiones bloqueadas del despertar específico del hosting y documentar ambos presupuestos sin prometer latencia universal.

**Issue independiente:** sí.\
**Commit independiente:** sí.\
**Mensaje conceptual de commit:** `fix: acota la espera del análisis y prepara el backend cuando aplica`

### T05 — Separar liveness y readiness de forma mínima

**Estado:** pendiente\
**Clasificación:** condicional\
**Prioridad:** alta

**Problema:** `model_ready=True` se establece antes de agendar el warm-up; `/health` devuelve `200` aunque la primera inferencia todavía afrontaría el coste que se pretende evitar.

**Objetivo:** consolidar el contrato mínimo y comprobable que distingue proceso HTTP vivo de servicio preparado, necesario para que el release final use correctamente el warm-up validado en T03.

**Justificación:** el preflight condicional de T04 no puede decidir cuándo enviar el POST si health declara listo prematuramente. El timeout general de T04 no depende de esta tarea.

**Dependencias:** T03.\
**Decision gate previo:** DG04; al menos un warm-up final en background.\
**Condición de activación:** activada: el warm-up de resampling se mantiene en background y liveness no equivale a inferencia preparada.

**Alcance:** estado explícito `pending/completed/failed` o equivalente; contrato de endpoint mínimo; fallo del warm-up visible sin filtrar detalles; coordinación lifecycle; tests de transición; decidir si `/health` queda como liveness y se añade readiness o si se amplía su semántica compatible.

**Fuera de alcance:** Kubernetes probes, sistemas distribuidos, registro persistente, paneles o scheduler complejo.

**Áreas probablemente afectadas:** `backend/app/main.py`, `backend/app/api/routes.py`, schemas/config, tests health y capa API frontend si consume endpoint nuevo.

**Cambios conceptuales esperados:** readiness depende del modelo y de todos los warm-ups seleccionados; liveness no ejecuta inferencia.

**Pruebas futuras:** warm-up desactivado, en curso, completado, fallido, modelo no disponible y shutdown limpio durante un warm-up en curso: sin espera indefinida, excepción no controlada ni estado de lifecycle incoherente. No se exige detener de forma forzada el trabajo nativo ya iniciado dentro de `asyncio.to_thread`.

**Criterios de aceptación:** estado no ambiguo; el preflight de T04 sabe cuándo enviar el POST; fallo termina en error controlado; no se ejecuta audio real como probe; solución pequeña.

**Riesgos:** romper monitores actuales o mantener readiness eternamente falsa tras fallo no recuperable.

**Rollback:** restaurar el contrato actual y desactivar warm-up si no puede representarse con seguridad.

**Documentación/memoria:** documentar semántica y códigos de estado finales.

**Issue independiente:** sí.\
**Commit independiente:** sí.\
**Mensaje conceptual de commit:** `feat: distingue disponibilidad y preparación de inferencia`

### T06 — Diagnosticar y decidir la primera MFCC

**Estado:** descartada\
**Clasificación:** condicional\
**Prioridad:** media

**Problema:** la primera extracción MFCC tarda 12–14,5 s en el proceso frío observado; no se conoce la distribución interna del coste.

**Decisión:** se omite para el MVP. La primera MFCC no se investigará ni modificará salvo nueva evidencia bloqueante.

**Justificación:** 12–14,5 s puede ser tolerable en una demo tras eliminar los 130 s de resampling; no debe optimizarse antes de medir el hosting final.

**Dependencias:** T03; runtime T07.\
**Decision gate previo:** DG03 = latencia fría inaceptable y MFCC material.\
**Condición de activación:** incumplimiento del criterio acordado.

**Alcance:** instrumentación temporal acotada para separar resolución/lazy import, inicialización y ejecución; repetir proceso frío y caliente; decidir sin cambiar salida; si procede, warm-up sintético de la misma MFCC con forma/dtype de producción.

**Fuera de alcance:** cambiar `n_mfcc=20`, representación, mean/std, vector de 40, librosa por defecto, modelo o reentrenar.

**Áreas probablemente afectadas:** `features.py`, lifecycle si se adopta warm-up, profiling temporal, tests de features/health y docs.

**Cambios conceptuales esperados:** primero diagnóstico; solo después aceptación, precarga o warm-up. Si hay warm-up, T05 debe contemplarlo.

**Pruebas futuras:** cold/warm, misma señal y features/score/etiqueta de salida idénticos, fallo no bloqueante o readiness coherente según decisión. Esta equivalencia comprueba ausencia de cambio técnico, no la verdad semántica de la etiqueta.

**Criterios de aceptación:** causa suficientemente localizada; decisión basada en tiempos; cero cambio de features dentro de tolerancia exacta/apropiada; impacto en startup/readiness registrado.

**Riesgos:** que medir altere lazy loading, duplicar los costes del warm-up o mantener instrumentación.

**Rollback:** retirar warm-up/instrumentación y aceptar el coste documentado.

**Documentación/memoria:** añadir medición y decisión, no hipótesis como hechos.

**Issue independiente:** sí.\
**Commit independiente:** diagnóstico y mitigación deberían ser commits separados si ambos requieren código.\
**Mensaje conceptual de commit:** `perf: diagnostica el coste frío de la primera MFCC`

### T07 — Fijar el contrato reproducible de runtime y despliegue

**Estado:** completada\
**Clasificación:** obligatoria\
**Prioridad:** alta

**Resultado:** Python 3.11.9, las dependencias directas críticas, el Dockerfile, el artefacto, el SHA-256 y el contrato operativo de Northflank quedaron verificados y documentados en `docs/despliegue_backend.md`. `psutil` se mantiene temporalmente por profiling y su retirada corresponde a T11.

**Objetivo:** ejecutar T07 en dos etapas cuando sea necesario: primero definir/fijar el runtime científico crítico y después verificar/documentar el contrato operativo del hosting definitivo, sin fijar indiscriminadamente todas las transitivas.

**Justificación:** actualizaciones silenciosas pueden cambiar compatibilidad binaria, latencia fría, resampling o resultados.

**Dependencias:** T01 para iniciar la tarea. Si DG01 decide migrar, T02 es dependencia del cierre de T07, no de su inicio.\
**Decision gate previo:** DG01 resuelto.\
**Condición de activación:** siempre.

**Alcance:** **fase inicial**, definir/fijar Python y dependencias directas/behavior-critical verificadas (`fastapi`, `uvicorn`, `python-multipart`, `joblib`, `numpy`, `librosa`, `soundfile`, `scikit-learn` y `soxr` si se mantiene la ruta) y decidir un formato requirements/constraints/lock compatible. **Fase de cierre**, sobre Render si se mantiene o después de T02 si se migra: verificar versiones efectivas, build/start, variables sin secretos, CORS, límites, URLs y SHA-256; documentar el contrato real y actualizar referencias obsoletas a hosting.

**Fuera de alcance:** fijar automáticamente `numba`, `llvmlite` y todas las transitivas; actualizar versiones por novedad; regenerar el modelo; unificar el entorno experimental raíz si no afecta al runtime web.

**Áreas probablemente afectadas:** `backend/requirements.txt`, mecanismo de versión Python/config del proveedor, README, docs de despliegue, memoria; `memory.py` solo temporalmente para comprobar versiones.

**Cambios conceptuales esperados:** conjunto pequeño de versiones compatibles e instalación limpia reproducible antes de desplegar; verificación del modelo/entorno y contrato operativo real después. T07 permanece en progreso entre ambas etapas si hay migración.

**Pruebas futuras:** fase inicial: instalación desde cero, suite backend y carga sin `InconsistentVersionWarning`. Fase de cierre: `/api/v1/model` con SHA esperado, smoke WAV/MP3 y log/comando de versiones efectivo en el hosting definitivo.

**Criterios de aceptación:** dos instalaciones limpias resuelven las mismas versiones críticas; scikit-learn coincide con el artefacto; runtime público verificable; docs reflejan Render/Vercel o el hosting realmente elegido; no se exponen secretos. Si hay migración, T07 no se considera completada hasta que T02 aporte la verificación efectiva del nuevo proveedor.

**Riesgos:** versiones verificadas hoy no tener wheel en el proveedor o pins demasiado rígidos de transitivas.

**Rollback:** restaurar requirements anterior y redeploy; conservar captura del entorno que funcionaba.

**Documentación/memoria:** alta; debe existir un runbook pequeño y vigente.

**Issue independiente:** sí.\
**Commit independiente:** sí; si se migra, la fase inicial del runtime y el cierre del contrato pueden terminar en commits T07 separados alrededor de T02.\
**Mensaje conceptual de commit:** `build: fija el runtime reproducible del backend`

### T08 — Medir concurrencia mínima en el entorno final

**Estado:** bloqueada\
**Clasificación:** recomendable\
**Prioridad:** media

**Problema:** no se sabe qué ocurre con dos análisis simultáneos; el pico individual se acerca a la RAM disponible.

**Objetivo:** caracterizar dos peticiones concurrentes controladas antes de decidir cualquier mitigación.

**Justificación:** permite declarar capacidad y detectar OOM/latencia multiplicada con evidencia.

**Dependencias:** T03 completada; T04 completada; todas las tareas condicionales críticas que hayan sido activadas y deban resolverse previamente, completadas.\
**Decision gate previo:** hosting/configuración final estables.\
**Condición de activación:** recomendada; puede omitirse con limitación explícita de demo individual.

**Alcance:** 2 solicitudes, combinaciones corto/corto y largo/largo o largo/corto según coste; RSS, códigos, tiempos, integridad de respuestas, salud posterior; un solo worker/config real.

**Fuera de alcance:** stress test, DDoS, Redis, Celery, colas, semáforos, rate limiting, múltiples workers o solución previa a resultados.

**Áreas probablemente afectadas:** inicialmente ninguna; script/evidencia de prueba solo si el usuario autoriza su implementación; logs temporales; docs.

**Cambios conceptuales esperados:** medir y clasificar capacidad. Una mitigación, si es necesaria, debe convertirse en una tarea nueva aprobada y mínima.

**Pruebas futuras:** ejecuciones repetidas, backend caliente; frío solo si relevante; comprobar que ninguna respuesta mezcla resultados y que el proceso continúa sano.

**Criterios de aceptación:** evidencia con archivos/duración, timestamps, RSS, latencias, estados HTTP y conclusión; no se implementa arquitectura sin gate nuevo.

**Riesgos:** provocar reinicio en producción o confundir límites del cliente con servidor.

**Rollback:** detener la prueba; no hay mutación persistente.

**Documentación/memoria:** registrar capacidad observada y limitación, no promesa de SLA.

**Issue independiente:** sí.\
**Commit independiente:** solo si añade un artefacto de prueba reusable; en caso contrario, evidencia documental.\
**Mensaje conceptual de commit:** `test: documenta la concurrencia mínima del backend`

### T09 — Evaluar lectura directa `float32`

**Estado:** condicional\
**Clasificación:** opcional\
**Prioridad:** baja

**Problema:** SoundFile crea cada chunk en `float64` y luego se convierte a `float32`, con una copia y pico temporal evitables.

**Objetivo:** comprobar si leer directamente `float32` reduce memoria/tiempo sin cambiar resultados relevantes.

**Justificación:** es una microoptimización plausible, no una necesidad actual.

**Dependencias:** T08 o nueva evidencia de memoria; T07.\
**Decision gate previo:** DG05.\
**Condición de activación:** margen insuficiente demostrado y beneficio esperado material.

**Alcance:** rama experimental acotada; comparar waveform por formato, MFCC, scores, labels, RSS y tiempo sobre archivos mono/estéreo, WAV/MP3, sample rates y caso cercano al threshold.

**Fuera de alcance:** reentrenamiento, tolerar cambios de clase, cambiar MFCC/threshold o combinar con otras optimizaciones.

**Áreas probablemente afectadas:** `audio.py`, tests audio/service/model y benchmark/evidencia.

**Cambios conceptuales esperados:** `SoundFile.read(dtype="float32")` solo si la equivalencia y beneficio se aceptan.

**Pruebas futuras:** waveform con tolerancia definida, MFCC/score con tolerancia definida antes de ver resultados, etiqueta de salida idéntica, archivos límite y memoria. La comparación mide equivalencia entre pipelines, no verdad semántica.

**Criterios de aceptación:** ninguna etiqueta de salida cambia en el corpus representativo, especialmente cerca del threshold; diferencias numéricas dentro de tolerancia predefinida; mejora medible; si no, descartar. Esto no valida que las etiquetas sean verdaderas.

**Riesgos:** drift numérico o falsa mejora irrelevante.

**Rollback:** revertir a lectura `float64` + conversión actual.

**Documentación/memoria:** documentar solo si se adopta o si el experimento aporta una conclusión relevante.

**Issue independiente:** sí.\
**Commit independiente:** sí si se adopta.\
**Mensaje conceptual de commit:** `perf: lee fragmentos de audio directamente en float32`

### T10 — Evaluar un resampler alternativo

**Estado:** descartada\
**Clasificación:** condicional\
**Prioridad:** media

**Problema:** `soxr_hq` presenta un coste inicial extraordinario en Render Free; el warm-up puede no ser operacionalmente aceptable.

**Decisión:** se omite para el MVP. Se mantiene `soxr_hq` junto con el warm-up de su ruta actual; no se evaluará una alternativa salvo nueva evidencia bloqueante.

**Justificación:** cambiar resampling puede afectar directamente las features y scores, por lo que no es una optimización por defecto.

**Dependencias:** T03, T07.\
**Decision gate previo:** DG06.\
**Condición de activación:** estrategia actual inaceptable tras decidir hosting.

**Alcance:** seleccionar como máximo alternativas técnicamente compatibles; medir cold/warm, memoria y calidad/equivalencia de salida; comparar corpus representativo; decidir mantener/cambiar.

**Fuera de alcance:** búsqueda abierta de librerías, cambio de sample rate, MFCC, modelo, threshold o reentrenamiento por defecto.

**Áreas probablemente afectadas:** `audio.py`, dependencias solo si se adopta, tests de equivalencia, benchmarks y documentación.

**Cambios conceptuales esperados:** posible parámetro explícito de `res_type` manteniendo 48/44.1/etc. →16 kHz, o confirmación de `soxr_hq`.

**Pruebas futuras:** waveform, MFCC, score y label; cold/warm; WAV/MP3; sample rates habituales; casos cercanos a threshold.

**Criterios de aceptación:** mejora operacional material en el hosting final; tolerancias predefinidas; ninguna regresión de clase no aceptada; dependencia reproducible; si falla, mantener actual.

**Riesgos:** cambiar predicciones o sustituir un cold start por otro.

**Rollback:** restaurar `soxr_hq` y su estrategia T03.

**Documentación/memoria:** registrar motivación, método y equivalencia; no afirmar mejora predictiva.

**Issue independiente:** sí.\
**Commit independiente:** diagnóstico y adopción separados.\
**Mensaje conceptual de commit:** `perf: evalúa una ruta de resampling con menor cold start`

### T11 — Retirar instrumentación temporal y cerrar observabilidad mínima

**Estado:** bloqueada\
**Clasificación:** obligatoria\
**Prioridad:** alta

**Problema:** permanecen `MEMORY_PROFILING_ENABLED`, `psutil`, `MemoryProfiler`, `memory_profile`, `runtime_profile`, `preprocess_profile`, `mfcc_profile` y logs diagnósticos detallados del warm-up.

**Objetivo:** dejar la versión final limpia, conservando únicamente señales operativas pequeñas y justificadas.

**Justificación:** la instrumentación cumplió su función, añade dependencia/ramas y puede generar ruido; retirarla antes de medir perdería evidencia necesaria.

**Dependencias:** T03, T04, T05 y T08 completadas; T06/T10 descartadas y T09 completada, descartada u omitida justificadamente.\
**Decision gate previo:** ninguna investigación pendiente necesita profiling.\
**Condición de activación:** siempre antes de T12.

**Alcance:** inventariar consumidores; retirar `MEMORY_PROFILING_ENABLED`, `MemoryProfiler`, `psutil`, `memory_profile`, `runtime_profile`, `preprocess_profile`, `mfcc_profile`, RSS por fragmento, request IDs y sus ramas/tests/configuración diagnóstica si ya no son necesarios. Conservar mediante `logging` solo carga/fallo de modelo, warm-up `started/completed/failed` con duración, errores reales de análisis y, si se justifica, un log compacto de análisis completado.

**Fuera de alcance:** introducir OpenTelemetry, Prometheus, Sentry, dashboards o rehacer logging.

**Áreas probablemente afectadas:** `memory.py`, `main.py`, `service.py`, `audio.py`, `config.py`, requirements, tests de memoria/config/health y documentación.

**Cambios conceptuales esperados:** borrar diagnóstico temporal; conservar logs mínimos de éxito/fallo sin datos sensibles. `InferenceTimings` puede mantenerse porque ya es salida funcional, salvo decisión explícita separada.

**Pruebas futuras:** suite backend; grep de identificadores retirados; instalación limpia; warm-up si permanece; errores y health; comprobar que no queda dependencia huérfana.

**Criterios de aceptación:** cada instrumento tiene decisión; no quedan flags muertos; `psutil` eliminado si no se usa; logs suficientes para saber startup/warm-up/error; suite pasa.

**Riesgos:** retirar antes de T03/T04/T05/T08 o eliminar observabilidad necesaria para T12.

**Rollback:** restaurar temporalmente profiling mediante commit anterior ante una regresión concreta.

**Documentación/memoria:** describir solo observabilidad final; conservar mediciones históricas como evidencia, no código activo.

**Issue independiente:** sí.\
**Commit independiente:** sí.\
**Mensaje conceptual de commit:** `chore: retira la instrumentación temporal de inferencia`

### T12 — Ejecutar y documentar la validación E2E final

**Estado:** bloqueada\
**Clasificación:** obligatoria\
**Prioridad:** crítica

**Problema:** los tests actuales no demuestran el recorrido público Vercel → backend definitivo ni los casos de cold start, límites y recuperación.

**Objetivo:** generar evidencia final, fechada y reproducible de que el MVP funciona como se presenta.

**Justificación:** es el gate de cierre; no debe usarse para descubrir que ramas condicionales críticas quedaron sin resolver.

**Dependencias:** T01, T03, T04, T07 y T11 completadas; todas las tareas condicionales activadas, completadas; T08 completada si se ejecuta o su omisión justificada y registrada.\
**Decision gate previo:** configuración release candidate congelada.\
**Condición de activación:** siempre al final.

**Alcance:** matriz de sección 18; ejecución contra URLs públicas; identificar versión/commit/config sin secretos; registrar esperado/real, tiempos, HTTP, resultado y evidencia; corregir únicamente defectos bloqueantes mediante tareas separadas.

**Fuera de alcance:** cambiar modelo para mejorar una predicción individual, inventar resultados, ampliar producto o mezclar refactors.

**Áreas probablemente afectadas:** evidencia manual versionable o documento de validación, README/memoria y roadmap/tabla de estado. Código solo en tareas de defecto separadas.

**Cambios conceptuales esperados:** ninguno en inferencia; aceptación o rechazo del release candidate.

**Pruebas futuras:** batería E2E completa descrita en sección 18, más suites backend/frontend en limpio.

**Criterios de aceptación:** todos los casos obligatorios pasan; no hay espera infinita, POST duplicado ni OOM; score se describe como decisión, no probabilidad; cualquier fallo tiene estado/error recuperable; evidencia incluye backend frío y caliente.

**Riesgos:** probar un proceso ya caliente, usar archivos no trazables, confundir label esperado con verdad forense o aceptar errores intermitentes.

**Rollback:** no promover el release; volver a configuración pública estable y abrir tarea específica.

**Documentación/memoria:** incorporar tabla final, arquitectura real, limitaciones, latencias como observaciones y fecha.

**Issue independiente:** sí.\
**Commit independiente:** sí para evidencia y cierre documental.\
**Mensaje conceptual de commit:** `test: documenta la validación E2E final de VeriSon`

## 15. Mejoras que podrían desaparecer si cambia el hosting

No ejecutar estas tareas antes de DG01:

| Trabajo | Por qué podría desaparecer o reducirse |
| ------- | -------------------------------------- |
| Subalcance de preflight de T04 | Un proveedor que acepte el primer POST de forma fiable o no duerma la instancia puede no necesitar `ensureBackendReady()` ni reintentos GET. El núcleo obligatorio de timeout/cancelación y recuperación de la UI no desaparece. |
| T05 readiness de warm-up | Si no hay warm-up en background o el servicio se mantiene caliente, liveness y readiness pueden coincidir para este MVP. |
| T06 warm-up/precarga de MFCC | Descartada para el MVP salvo nueva evidencia bloqueante. |
| T10 resampler alternativo | Descartada para el MVP; se conserva `soxr_hq` con warm-up. |
| Intervalos y presupuesto del preflight GET | Pueden desaparecer si no hay preflight y, si existe, deben derivarse del cold start del destino final, no copiar los ~130 s de Render. El timeout general del POST sigue siendo necesario y se ajusta a la duración válida del análisis. |
| T09 `float32` por presión de RAM | Más RAM puede eliminar el único motivo operacional para asumir riesgo numérico. |

T07 y T12 no desaparecen con un cambio de hosting; se vuelven más importantes. T03 tampoco desaparece: cambia de “mitigar Render” a “demostrar que el nuevo entorno cumple y decidir que no necesita mitigación”. De T04 solo puede desaparecer el preflight específico; impedir que una petición deje la UI indefinidamente en `analyzing` sigue siendo obligatorio con cualquier hosting.

## 16. Instrumentación temporal

### Inventario actual

- Flag `MEMORY_PROFILING_ENABLED`, desactivado por defecto.
- Dependencia `psutil` y clase `MemoryProfiler`.
- `memory_profile` con RSS por fases y fragmentos seleccionados.
- `runtime_profile` para Python/librosa/numpy/scipy/soxr/numba/llvmlite.
- `preprocess_profile` con resolución/ejecución de resampling y otras fases.
- `mfcc_profile` para la primera MFCC.
- logs `resample_warmup` de estado, resolve, execute y total.
- `InferenceTimings` en la respuesta API; estos tiempos forman parte del contrato funcional actual y no equivalen necesariamente a profiling temporal.

### Uso restante

| Instrumento | Tareas que pueden necesitarlo | Retirada/decisión |
| ----------- | ----------------------------- | ----------------- |
| RSS/`psutil` | T03, T08, T09, T10 | retirar en T11 si no queda requisito operativo |
| `runtime_profile` | T07 y diagnóstico reproducible | retirar tras verificar instalación/deploy |
| `preprocess_profile` | T03 y T10 | retirar al cerrar resampling |
| `mfcc_profile` | ya no tiene investigación pendiente | retirar en T11 |
| logs detallados warm-up | T03/T05 | reducir en T11; puede quedar `started/completed/failed` y duración total si warm-up sigue activo |
| `InferenceTimings` | T12 y respuesta actual | mantener salvo decisión específica de contrato; no requiere `psutil` |

T11 debe eliminar código, tests, configuración y dependencia en una misma intervención coherente. La evidencia histórica no se borra. Ningún profiler debe convertir un fallo diagnóstico en fallo de inferencia.

## 17. Estrategia de reproducibilidad

1. **Separar runtime web de entorno experimental.** `backend/requirements.txt` es la fuente del despliegue; el `requirements.txt` raíz contiene herramientas de experimentación y no debe imponerse al backend público.
2. **Fijar Python.** Registrar y, si el hosting lo permite, exigir Python 3.11.9.
3. **Fijar dependencias directas y críticas de comportamiento.** FastAPI, Uvicorn, python-multipart, joblib, NumPy, librosa, SoundFile, scikit-learn 1.8.0 y soxr si continúa siendo el backend de resampling. Las versiones exactas deben partir del entorno verificado y superar instalación/tests.
4. **No fijar transitivas sin razón.** SciPy puede fijarse si la carga/ejecución del pipeline o la resolución reproducible lo exige. Numba/llvmlite deben permanecer transitivas salvo que una investigación demuestre dependencia de comportamiento o incompatibilidad. Documentar la decisión.
5. **Proteger el artefacto.** Conservar el joblib versionado, verificar SHA-256 mediante `/api/v1/model` y asegurar compatibilidad con scikit-learn 1.8.0. No regenerarlo para resolver un problema de instalación.
6. **Instalación limpia.** Probar desde un entorno vacío, ejecutar suite backend y smoke WAV/MP3. No validar únicamente sobre `.venv` existente.
7. **Verificación del deploy.** Capturar de manera reproducible las versiones efectivas y el commit/revisión. El profiling temporal puede ayudar durante T07, pero se retira en T11.
8. **Contrato operativo.** Documentar build command, start command, directorio, variables requeridas, límites, CORS, modelo, health/readiness, URL y plan del proveedor, sin secretos.
9. **Control de cambios.** Actualizar versiones de una en una o en un bloque pequeño, ejecutar equivalencia y E2E, y no mezclar actualización de dependencias con optimizaciones numéricas.
10. **Documentación vigente.** Alinear README, documentación de decisiones y memoria con Render o el nuevo hosting; marcar referencias históricas a Hugging Face Spaces como tales en vez de presentarlas como despliegue actual.

## 18. Estrategia de validación final

T12 debe preparar antes de ejecutar una matriz con: ID de caso, precondición, archivo controlado, resultado esperado, resultado real, tiempo, HTTP, evidencia y estado. No deben incluirse datos personales ni secretos.

### Preparación

- Congelar commit/revisión, URLs y configuración efectiva.
- Ejecutar suites backend y frontend desde instalaciones limpias.
- Verificar SHA-256 y metadatos de `/api/v1/model`.
- Definir archivos de prueba trazables: WAV/MP3 corto, largo cercano a 300 s y/o 64 MiB, inválido y casos de borde.
- Garantizar un proceso realmente frío según el proveedor; registrar cómo se obtuvo.

### Casos obligatorios

| Grupo | Casos y comprobaciones |
| ----- | ---------------------- |
| Disponibilidad | backend frío; backend caliente; URL Vercel → backend público; health/readiness final; backend no disponible; timeout/error finito; recuperación tras error |
| Formato | WAV válido; MP3 válido; extensión/formato no soportado; audio corrupto o no decodificable |
| Tamaño/duración | archivo corto; largo cercano al límite; exactamente en el límite si es viable; por encima de 64 MiB; por encima de 300 s |
| Cold start | primera apertura/primer análisis; no se pierde POST; exactamente un POST; warm-up en curso/completado/fallido si permanece; segunda petición caliente |
| Resultado | esquema válido; coherencia estructural entre label y clase; `decision_threshold=0.0`; `score_type=decision_function`; `score_is_calibrated_probability=false`; warning de estimación; sin interpretar la etiqueta individual como verdad semántica |
| Memoria | canción larga sin OOM; proceso sigue saludable después; temporales eliminados |
| UX | “Iniciando…” si aplica; “Analizando…”; éxito; error comprensible; reset; analizar otra canción; conservar/limpiar archivo según estado; accesibilidad básica de status/alert |
| Seguridad del contrato | no exponer `MODEL_PATH`, rutas internas ni detalles de excepción; CORS solo desde origen configurado |

### Criterio de cierre

- Cero casos críticos fallidos.
- Ninguna espera indefinida ni retry automático del POST.
- WAV y MP3 funcionan desde la URL pública.
- Archivo largo no provoca OOM.
- Los límites devuelven errores controlados y recuperables.
- El resultado se presenta como estimación y el score nunca como porcentaje/probabilidad.
- El éxito de estos casos significa que el pipeline completó técnicamente el procesamiento y devolvió el contrato esperado; no demuestra que la etiqueta de una canción individual sea verdadera.
- Cualquier caso no ejecutable queda justificado; no se marca como “pass” sin evidencia.

## 19. Orden recomendado de ejecución

Este es el backlog operativo. Las ramas “si se activa” no se implementan automáticamente.

1. **T03 — Validar el warm-up de resampling** en Northflank mediante restart controlado; no promoverlo como release final aislado.
2. **T05 — Separar liveness y readiness** y consolidar el contrato necesario para el release final con warm-up.
3. **T04 — Acotar la espera del frontend:** readiness por GET con reintentos acotados, después exactamente un POST con timeout/cancelación y recuperación.
4. **T08 — Medir concurrencia mínima**, o justificar explícitamente la limitación de demo individual.
5. **T09 — Evaluar `float32`** solo si T08 aporta evidencia que active DG05.
6. **T11 — Retirar instrumentación temporal y cerrar observabilidad mínima.**
7. **T12 — Ejecutar y documentar la validación E2E final.**

## 20. Próxima tarea recomendada

La próxima tarea debe ser:

> **T03 — Cerrar la estrategia operacional de cold start y resampling en Northflank**

Debe validar la decisión de warm-up con `RESAMPLE_WARMUP_ENABLED=true` tras restart controlado, sin extrapolar Render ni presentar esa configuración aislada como release final. No debe cambiar modelo, preprocessing, MFCC ni `soxr_hq`; el coste frío residual de MFCC se acepta y T05 consolidará readiness antes del release final.

## 21. Qué NO merece la pena implementar ahora

- Cambiar o reentrenar el modelo, threshold, dataset, splits o protocolo experimental.
- Calibrar probabilidades o convertir `decision_function` en porcentaje.
- Rediseñar el streaming; no existe regresión OOM actual.
- Redis, Celery, colas, múltiples workers, semáforos o rate limiting antes de T08.
- Kubernetes, MLOps completo, observabilidad empresarial o microservicios.
- Autenticación, pagos, historial, base de datos o almacenamiento permanente.
- Reintentos automáticos de `POST /analyze`.
- Warm-up de MFCC antes de diagnóstico y gate del hosting.
- Cambiar resampler antes de DG06.
- Lectura directa `float32` sin evidencia de memoria y equivalencia.
- Pinnear todas las dependencias transitivas de forma indiscriminada.
- Mantener instrumentación diagnóstica “por si acaso” después de T11.
- Añadir formatos distintos de WAV/MP3 o elevar límites sin requisito.
- Construir SLA, autoscaling complejo o arquitectura empresarial para una demo académica.

## 22. Instrucciones para futuros chats de Codex

1. Leer este roadmap antes de editar.
2. Implementar únicamente la tarea solicitada por el usuario.
3. Inspeccionar código antes de modificar.
4. No reabrir decisiones cerradas sin nueva evidencia.
5. Hacer cambios pequeños y verificables.
6. Ejecutar tests relevantes.
7. No inventar resultados.
8. No añadir infraestructura innecesaria.
9. No presentar scores como probabilidades.
10. No modificar modelo, threshold o protocolo experimental salvo petición explícita.
11. Mantener la salida como estimación, no veredicto forense.
12. Indicar cuándo conviene hacer commit y proponer mensaje.
13. Actualizar este roadmap únicamente cuando:
    - una tarea cambie de estado;
    - aparezca nueva evidencia;
    - una decisión modifique el roadmap.
14. No implementar otras tareas “aprovechando” la intervención.
15. Respetar gates y condiciones: una tarea condicional no se activa por iniciativa del chat.
16. Si una tarea produce evidencia que activa otra, detenerse, actualizar estado y pedir al usuario el nuevo ID; no encadenar implementaciones.
17. No hacer commit ni push salvo petición explícita del usuario.
18. Distinguir tests locales, mediciones de hosting y observaciones del usuario; no extrapolar entre entornos.

## 23. Historial de estado

Actualizar esta tabla solo con evidencia real. Una decisión documental puede referenciar su commit; una medición debe indicar entorno y artefacto. No usar “completada” si quedan criterios de aceptación pendientes.

| ID | Estado | Fecha | Evidencia/commit | Notas |
| -- | ------ | ----- | ---------------- | ----- |
| ROADMAP | completada | 2026-08-25 | `docs/roadmap_mejoras_despliegue.md` | Inspección inicial; 12 tareas y seis decision gates definidos. |
| T01 | completada | 2026-08-27 | DG01: Northflank Developer Sandbox seleccionado | Render queda temporalmente como rollback. |
| T02 | completada | 2026-08-27 | Validación Northflank: health/model/SHA, WAV, MP3, CORS, largo y flujo Vercel | Sin OOM ni reinicio; no equivale a T12. |
| T03 | completada | 2026-09-02 | Warm-up de resampling y MFCC validado en Northflank | `RESAMPLE_WARMUP_ENABLED=true`; `/ready` espera los warm-ups. |
| T04 | completada | 2026-09-02 | Preflight `/ready`, timeout/cancelación y recuperación frontend validados | No hay reintento automático del POST. |
| T05 | completada | 2026-09-02 | `/ready` incorporado y validado | `/health` conserva la semántica de estado operativo/modelo. |
| T06 | descartada | 2026-08-31 | Decisión de alcance MVP | No investigar primera MFCC salvo evidencia bloqueante. |
| T07 | completada | 2026-08-27 | `backend/requirements.txt`, `deploy/backend/`, `docs/despliegue_backend.md` y validación Northflank | Contrato reproducible cerrado; `psutil` queda para T11. |
| T08 | completada | 2026-09-02 | Dos análisis largos simultáneos | Ambos HTTP 200; sin OOM ni reinicio. |
| T09 | condicional | 2026-08-31 | — | Solo si T08 activa DG05. |
| T10 | descartada | 2026-08-31 | Decisión de alcance MVP | Mantener `soxr_hq`; no evaluar alternativa salvo evidencia bloqueante. |
| T11 | completada | 2026-09-02 | Limpieza final de profiling | Retirados `MemoryProfiler`, `psutil`, RSS, IDs y logs diagnósticos; `InferenceTimings` se mantiene. |
| T12 | en curso | 2026-09-02 | Protocolo manual E2E iniciado | Pendiente de evidencia manual final. |
