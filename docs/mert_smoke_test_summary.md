# Smoke test tecnico de MERT

- Estado: `satisfactory`.
- Fecha UTC: `2026-07-25T13:30:42.423843+00:00`.
- Modelo: `m-a-p/MERT-v1-95M`.
- Revision: `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.
- Dispositivo evaluado: `cpu`.
- Veredicto final: fase tecnicamente satisfactoria en CPU.

## Configuracion

- Manifiesto: `data/aime_splits.csv`.
- Revision de AIME en el manifiesto: `b84d4be5eda830b6eb714998569dba73530f2601`.
- Split permitido: `train`.
- Muestras procesadas: `12`, correspondientes a `6` parejas humano/IA.
- Fallos: `0`.
- Entrada por clip: `10 s` a `24000 Hz`, `240000` muestras.
- Ventanas por clip: `2` ventanas contiguas de `5 s`, `120000` muestras por ventana.
- Batch size: `1`.
- Precision mixta: `false`.
- Cuantizacion: `false`.
- Encoder congelado: `true`.
- Modo evaluacion: `true`.

## Muestra

| id | description | label | model | split |
| --- | --- | ---: | --- | --- |
| `06390` | pop, soundtrack, space | 0 | MTG-Jamendo | train |
| `04890` | pop, soundtrack, space | 1 | Udio | train |
| `06051` | chillout, jazz, piano | 0 | MTG-Jamendo | train |
| `02901` | chillout, jazz, piano | 1 | Riffusion | train |
| `06020` | ambient, jazz, progressive | 0 | MTG-Jamendo | train |
| `01765` | ambient, jazz, progressive | 1 | AudioLDM 2 Large | train |
| `06419` | rock, experimental, heavymetal | 0 | MTG-Jamendo | train |
| `02397` | rock, experimental, heavymetal | 1 | AudioLDM 2 Music | train |
| `06147` | electronic, ambient, synthesizer | 0 | MTG-Jamendo | train |
| `01345` | electronic, ambient, synthesizer | 1 | MusicGen Large | train |
| `06129` | easylistening, newage, sad | 0 | MTG-Jamendo | train |
| `00622` | easylistening, newage, sad | 1 | MusicGen Medium | train |

## Embeddings

- Dimension esperada: `(768,)`.
- Formas correctas: `true`.
- Embeddings finitos: `true`.
- Todas las ventanas tienen forma `(120000,)`.
- Todos los clips tienen `240000` muestras.
- No se guardan embeddings completos en Git; el resultado estructurado solo conserva forma, estadisticos resumidos y hashes.

## Determinismo

- Muestra usada: `06390`.
- Tolerancia absoluta: `1e-05`.
- Diferencia maxima absoluta: `0.0`.
- Resultado: `true`.
- Tiempo de comprobacion: `1.3999949999852106 s`.

## Tiempos

- Carga del procesador: `0.35439200000837445 s`.
- Carga del modelo: `0.8709335000021383 s`.
- Preprocesamiento total: `2.5144196000183 s`.
- Preprocesamiento medio por clip: `0.20953496666819169 s`.
- Inferencia media por ventana de `5 s`: `0.35903213334677275 s`.
- Inferencia media por clip de `10 s`: `0.7180642666935455 s`.
- Tiempo total del smoke test: `4461.9567391000455 s`.
- Tiempo de streaming remoto de audios AIME: `4434.934058999992 s`.

El tiempo total estuvo dominado por el streaming remoto de AIME hasta localizar los audios seleccionados. No debe interpretarse como latencia de inferencia de MERT. La latencia relevante para el modelo en CPU queda representada por los tiempos medios de inferencia por ventana y por clip.

## Memoria

- RSS antes de la ejecucion: `388968448 bytes`.
- RSS antes de cargar el modelo: `396374016 bytes`.
- RSS despues de cargar el modelo: `498774016 bytes`.
- Pico RSS aproximado: `6394953728 bytes`.
- Tamano local aproximado del snapshot cacheado: `377578388 bytes`.

Las medidas de RSS son aproximadas y fueron registradas mediante `psutil`.

## CUDA

La GPU esta disponible en el sistema como `NVIDIA GeForce RTX 3060 Laptop GPU`, con `6144 MiB` de VRAM, controlador `592.27` y `nvidia-smi` operativo.

CUDA no fue evaluada en este smoke test porque el entorno virtual usado contiene `torch 2.13.0+cpu`, `torch.version.cuda == None` y `torch.cuda.is_available() == False`. La ejecucion:

```powershell
python scripts/smoke_test_mert.py --device cuda --max-pairs 1
```

termino con el error controlado:

```text
MertSmokeTestError: CUDA was requested but is not available
```

Este resultado no es un fallo de MERT ni de la GPU, y no invalida el smoke test satisfactorio en CPU. Instalar una distribucion de PyTorch con CUDA queda como optimizacion opcional posterior.

## Versiones

- Python: `3.11.9`.
- torch: `2.13.0+cpu`.
- transformers: `5.13.1`.
- huggingface_hub: `1.21.0`.
- numpy: `2.4.6`.
- pandas: `3.0.3`.
- PyYAML: `6.0.3`.
- psutil: `7.2.2`.

No se fijan versiones exactas en `requirements.txt` en esta fase para evitar cambios especulativos. La combinacion anterior queda registrada como entorno verificado para el smoke test CPU.

## Revision previa de codigo remoto

Se inspeccionaron solo los archivos de codigo/configuracion de la revision fijada, sin incluir pesos en Git:

| archivo | SHA-256 |
| --- | --- |
| `configuration_MERT.py` | `ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03` |
| `modeling_MERT.py` | `6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c` |
| `config.json` | `ea2627c4c7825cd66f3c944b6b966331604c35928174e0100cd4a82829424e32` |
| `preprocessor_config.json` | `cc5a5e4a5d3b1a758a5ed984b2eaa15bb0522d811d44a9eed82bfca4baa0dc8f` |

Hallazgos:

- Imports observados: `torch`, `torch.nn`, clases HuBERT de `transformers`, `PretrainedConfig`, `BaseModelOutput`, `typing`, `functools`, `operator` y `math`.
- `auto_map`: `AutoConfig -> configuration_MERT.MERTConfig`; `AutoModel -> modeling_MERT.MERTModel`.
- `preprocessor_config.json`: `Wav2Vec2FeatureExtractor`, `sampling_rate=24000`, `do_normalize=true`.
- `config.json`: `hidden_size=768`, `sample_rate=24000`, `feature_extractor_cqt=false`.
- `nnAudio` aparece como dependencia opcional en `modeling_MERT.py`, pero no queda requerida por la configuracion fijada porque `feature_extractor_cqt=false`.
- No se observaron patrones textuales de `subprocess`, `os.system`, `Popen`, `requests`, `urllib`, `socket`, `exec(` ni `eval(` en los archivos inspeccionados.
- No se observaron operaciones explicitas de entrada/salida en esos archivos segun la busqueda textual aplicada.

Esta revision no demuestra seguridad absoluta del codigo remoto; solo documenta los archivos inspeccionados y que no se observaron comportamientos inesperados en esta comprobacion previa.

## Resultado estructurado

El resultado detallado se genero en:

`data/processed/mert_smoke_test_result.json`

Esa ruta esta ignorada por Git y no se incluye en el commit porque contiene resultados temporales de ejecucion. No contiene pesos ni embeddings completos.

## Veredicto

MERT queda validado tecnicamente en CPU para la configuracion inicial de la fase 2:

- carga reproducible con modelo y revision fijados;
- `trust_remote_code=True` revisado previamente;
- entrada real de AIME procedente solo de `train`;
- forma final `(768,)`;
- embeddings finitos;
- determinismo en la misma configuracion;
- modelo congelado y en modo evaluacion;
- batch size `1`;
- sin entrenamiento, sin SVM, sin metricas predictivas y sin consulta de validacion o test.

La siguiente fase tecnica puede preparar la extraccion controlada de embeddings para el subconjunto completo, reutilizando esta configuracion y evitando volver a depender del streaming remoto cuando sea posible.
