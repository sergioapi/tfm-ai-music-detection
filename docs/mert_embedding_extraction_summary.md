# Extraccion de embeddings MERT

- Estado: `satisfactory`.
- Fecha UTC: `2026-07-25T21:53:24.602665+00:00`.
- Modelo: `m-a-p/MERT-v1-95M`.
- Revision: `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.
- Dispositivo: `cpu`.
- Manifiesto: `data/aime_splits.csv`.
- Revision de AIME en el manifiesto: `b84d4be5eda830b6eb714998569dba73530f2601`.

## Configuracion

- Encoder congelado: `true`.
- Modo evaluacion: `true`.
- Batch size: `1`.
- Precision mixta: `false`.
- Cuantizacion: `false`.
- Entrada: mono `float32`.
- Sample rate: `24000 Hz`.
- Clip: `10 s`, `240000` muestras.
- Ventanas: `2` ventanas contiguas de `5 s`, `120000` muestras por ventana.
- Pooling: media temporal del ultimo estado oculto por ventana y media entre las dos ventanas.
- Dimension esperada: `768`.

## Resultado

- Ejemplos esperados: `1000`.
- Embeddings procesados en esta ejecucion: `1000`.
- Fallos: `0`.
- IDs no encontrados en streaming: `0`.
- Forma final: `(1000, 768)`.
- Valores finitos: `true`.
- IDs exactos frente al manifiesto: `true`.
- Metadatos coincidentes frente al manifiesto: `true`.
- Dtype final de embeddings: `float32`.
- Dtype fisico en Parquet comprobado con `pyarrow.parquet.read_schema`: `float`.
- Distribucion por split: `train=700`, `val=150`, `test=150`.
- Distribucion por etiqueta: `0=500`, `1=500`.

El primer Parquet consolidado quedo con columnas de embedding en `double` por inferencia de tipo de `pandas` al leer el CSV. Se corrigio la consolidacion para castear explicitamente las columnas `mert_*` a `float32` antes de escribir Parquet y se regenero solo el Parquet desde `data/processed/aime_mert_embeddings.csv`. No se volvio a cargar MERT, no se recorrio AIME de nuevo y no se recalcularon embeddings.

## Artefactos locales

Los artefactos se guardaron en rutas ignoradas por Git:

- `data/processed/aime_mert_embeddings.csv`
- `data/processed/aime_mert_embeddings.parquet`
- `data/processed/aime_mert_embedding_extraction_summary.json`

No se guardaron audios ni pesos en el repositorio. El CSV incremental permite reanudar la extraccion si una ejecucion futura se interrumpe.

## Streaming

- Modo: `remote_streaming_once`.
- Filas AIME escaneadas: `6499`.
- Filas del manifiesto encontradas: `1000`.
- Tiempo de streaming y procesamiento asociado: `5995.125535399886 s`.

La ejecucion remota recorrio AIME una sola vez y se detuvo cuando encontro los 1000 IDs pendientes. El tiempo total esta dominado por el streaming remoto y no debe interpretarse como latencia de inferencia del modelo.

## Tiempos

- Carga del procesador: `0.6254458000184968 s`.
- Carga del modelo: `0.6107143999543041 s`.
- Preprocesamiento total: `234.99847060081083 s`.
- Preprocesamiento medio por clip: `0.23499847060081083 s`.
- Inferencia media por ventana: `0.41197334039967975 s`.
- Inferencia media por clip: `0.8239466807993595 s`.
- Tiempo total: `6002.641760000028 s`.

## Memoria

- RSS antes de la ejecucion: `388325376 bytes`.
- RSS antes de cargar el modelo: `394072064 bytes`.
- RSS despues de cargar el modelo: `499437568 bytes`.
- Pico RSS aproximado: `10918952960 bytes`.
- Tamano local aproximado del snapshot cacheado: `377578388 bytes`.
- VRAM: no disponible en esta ejecucion CPU.

Las medidas de RSS son aproximadas y fueron registradas mediante `psutil`. El pico RSS corresponde al proceso completo durante streaming, decodificacion, preprocesamiento, inferencia y escritura; no debe interpretarse como consumo aislado del encoder MERT.

## Versiones

- Python: `3.11.9`.
- torch: `2.13.0+cpu`.
- transformers: `5.13.1`.
- huggingface_hub: `1.21.0`.
- numpy: `2.4.6`.
- pandas: `3.0.3`.
- PyYAML: `6.0.3`.
- psutil: `7.2.2`.

## Veredicto

La extraccion completa de embeddings MERT para los 1000 ejemplos de `data/aime_splits.csv` queda realizada y validada en CPU. La siguiente fase puede entrenar y seleccionar el clasificador sobre estos embeddings usando exclusivamente train y validacion, manteniendo test bloqueado hasta cerrar la configuracion.
