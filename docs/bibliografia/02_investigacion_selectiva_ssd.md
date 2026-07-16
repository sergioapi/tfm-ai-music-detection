# Investigación bibliográfica selectiva para el TFM sobre Synthetic Song Detection

## Resumen ejecutivo

Tomando como punto de partida tu auditoría previa y sin repetir los documentos ya analizados, la búsqueda adicional deja bastante claro que las lagunas más importantes sí pueden cerrarse con una bibliografía relativamente corta: la documentación oficial de `disco-eth/AIME`, el paper original asociado a AIME, una única evidencia directamente verificable de uso de AIME para detección, referencias metodológicas sólidas para **MFCC** y **SVM** como baseline clásico, varios trabajos **nuevos de 2026** sobre **Synthetic Song Detection** centrados en generalización/robustez, y documentación oficial suficiente para preparar la **preselección** de encoders de audio preentrenados sin decidir todavía el modelo final. Además, la literatura reciente refuerza dos ideas útiles para tu memoria: que muchos detectores siguen siendo frágiles ante **generadores no vistos, compresión/códecs, duraciones cortas y escenarios operacionales distintos del contexto “streaming limpio”**, y que la **justificación del modelo desplegado** debe descansar sobre todo en benchmarks propios, no en un supuesto estándar bibliográfico inexistente. citeturn61view0turn65view0turn43view3turn42view1turn42view2turn43view1turn42view0turn40view0

Lo más importante sobre **AIME** es que la propia dataset card y el paper asociado lo presentan como un recurso para **evaluación de modelos y métricas de generación musical basada en preferencias humanas**, no como un benchmark canónico de detección. La card documenta **6.000 pistas generadas por 12 modelos**, más **500 pistas de MTG-Jamendo**; además, explicita que `description` contiene **tres etiquetas** usadas como prompt, lo que vuelve razonable tu decisión de vigilar la fuga semántica al diseñar particiones. En las fuentes recuperadas, la **única** evidencia académica verificada de uso de **AIME** específicamente para **AI-generated music detection** es **Fusion Segment Transformer**, que reporta resultados sobre **SONICS y AIME**; no he podido verificar de forma directa más trabajos de detección que usen AIME. citeturn61view0turn65view0turn43view3

Para el **baseline MFCC + SVM**, la literatura adicional no aporta una unión exacta “AIME + MFCC + SVM”, pero sí una justificación metodológica suficiente: en MIR y en detección de audio sintético/tareas cercanas, **MFCC** sigue siendo una representación clásica, barata y reproducible, útil como baseline; al mismo tiempo, varios trabajos muestran que al resumir ventanas cortas en vectores compactos pierde parte de la **estructura temporal**, lo que explica por qué conviene compararlo con un encoder profundo. De manera análoga, **SVM** sigue siendo una opción razonable cuando el objetivo es un **baseline clásico, controlable y de bajo coste**, especialmente si se entrena sobre embeddings o descriptores fijos y se desea maximizar reproducibilidad antes que exprimir el último punto de F1. citeturn59view0turn70academia0turn73search0

En **Synthetic Song Detection** estricto, las referencias nuevas más útiles no son tanto más “papers de benchmark” como trabajos que atacan tus preguntas metodológicas: **MusicDET** plantea explícitamente el problema de los **generadores no vistos** y formula un setting **zero-shot** entrenado solo con música real; **AI-Generated Music Detection in Broadcast Monitoring** demuestra que el rendimiento cae mucho con **duraciones cortas** y **speech masking**; **Beyond Artifacts** propone una vía más **generator-agnostic** basada en rasgos musicales intrínsecos y construye **MUSIC8K** con perturbaciones realistas; **ArtifactNet** enfatiza la robustez a **códecs** y además es útil para tu futura fase operacional porque presenta un detector muy ligero, de **4,0 M** de parámetros totales. citeturn42view1turn42view2turn43view1turn42view0

Las lagunas que siguen abiertas son más específicas y, en tu caso, no deberían bloquear la memoria si las formulas bien. La principal es que **no aparece una tradición consolidada de trabajos con AIME como benchmark de detección**, de modo que su uso debe seguir justificándose principalmente por **documentación oficial + procedencia + auditoría propia**. También sigue abierta una cuestión importante: **no existe un estándar único y estable** para evaluar **latencia, RAM/VRAM, tiempo de carga o inferencia CPU** en Synthetic Song Detection; la bibliografía solo ofrece apoyo parcial, por lo que la selección final del encoder profundo y del modelo desplegado debe justificarse **principalmente con benchmarks propios** bajo tu hardware objetivo y, de forma secundaria, con model cards y documentación oficial. citeturn61view0turn40view0turn27view0turn29view0turn28view4turn50view0

En términos prácticos, la bibliografía actual queda ya bastante cubierta en lo relativo a trabajos nucleares que tú ya has auditado. Lo que **todavía necesitaba trabajo** y ahora queda sustancialmente reforzado es: **AIME**, **fundamentos de MFCC/SVM**, **generalización/robustez**, **sesgos y fugas por particionado/producción**, **segmentación y agregación a nivel canción**, **métricas metodológicas** y **criterios para acotar la futura comparación entre encoders preentrenados**. citeturn61view0turn59view0turn42view2turn42view1turn43view3turn42view0turn62view0turn29view0turn28view4

## Tabla de nuevas referencias recomendadas

### Trabajos y artículos

| Referencia | DOI / arXiv / URL oficial | Tipo | Tarea / dataset / método | Resultados o idea útil | Relación exacta con el TFM | Laguna que cubre | Sección recomendada | Prioridad | Estado |
|---|---|---|---|---|---|---|---|---|---|
| **Davis, S. B.; Mermelstein, P. (1980). _Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences_. IEEE TASSP.** | URL oficial no recuperada en las fuentes consultadas; referencia clásica identificada indirectamente. citeturn72search0 | Artículo fundacional | Reconocimiento de habla; representación cepstral | Es la referencia clásica más citada para introducir MFCC. | Sirve para explicar qué son los MFCC y por qué representan espectro de corto plazo en escala perceptual. | Fundamento MFCC | Metodología / baseline clásico | **Esencial** | **Declarada pero no comprobada completamente** |
| **Cortes, C.; Vapnik, V. (1995). _Support-vector networks_. Machine Learning, 20(3), 273–297.** | URL oficial no recuperada; volumen, número, paginación y fecha verificados indirectamente. citeturn73search2turn73search4 | Artículo fundacional | Clasificación binaria; SVM | Ref. principal para margen máximo y kernels. | Justifica SVM como clasificador binario clásico y estándar. | Fundamento SVM | Metodología / baseline clásico | **Esencial** | **Declarada pero no comprobada completamente** |
| **Novoselov et al. (2015). _STC Anti-spoofing Systems for the ASVspoof 2015 Challenge_.** | arXiv:1507.08074. citeturn70academia0 | Preprint / challenge paper | Detección de speech spoofing; ASVspoof 2015; MFCC y comparación SVM vs DBN | Usa explícitamente **MFCC** y compara **SVM lineal** con **DBN no lineal**. | No es Synthetic Song Detection, pero sí evidencia transferible de que MFCC+SVM es un baseline realista en detección de audio sintético cercana. | MFCC y SVM en tarea próxima | Estado del arte adyacente / justificación del baseline | **Recomendable** | **Verificada** |
| **Nasrullah, Z.; Zhao, Y. (2019). _Music Artist Classification with Convolutional Recurrent Neural Networks_.** | arXiv:1901.04555. citeturn59view0 | Preprint MIR | Clasificación musical; artist20; clip length, split por álbum/canción y agregación canción | Verifica el **producer effect**, explora el efecto de la **longitud de clip** y cita mejora al agregar predicciones de frame a canción. También recuerda que MFCC resume ventanas cortas pero pierde estructura temporal. citeturn59view0 | Útil para justificar riesgos de fuga por particionado, elección de fragmentos y agregación a nivel canción. | Sesgos/fugas, segmentación y agregación | Metodología / amenazas a la validez | **Esencial** | **Verificada** |
| **Grötschla et al. (2025). _Benchmarking Music Generation Models and Metrics via Human Preference Studies_.** | arXiv:2506.19085. citeturn65view0 | Preprint original de AIME | Evaluación de generación musical; 6k canciones, 12 modelos; preferencias humanas | El paper genera **6k canciones** con **12 modelos** y 15k comparaciones pareadas con 2,5k participantes; no define el recurso como benchmark de detección. citeturn65view0 | Es la fuente académica clave para justificar la procedencia y el propósito original de AIME. | AIME y su encuadre correcto | Dataset / justificación del corpus | **Esencial** | **Verificada** |
| **`disco-eth/AIME` dataset card en Hugging Face.** | Dataset card oficial. citeturn61view0 | Documentación oficial | Dataset de evaluación musical; `id`, `model`, `description`, `audio` | La card documenta **6.000 pistas generadas** + **500 de MTG-Jamendo**, `description` como lista de **tres tags** y licencias diferenciadas entre audio generado y pistas Jamendo. citeturn61view0 | Es la mejor fuente para describir AIME sin sobreafirmar que sea benchmark de detección. | Verificación documental de AIME | Dataset / metodología | **Esencial** | **Verificada** |
| **Kim; Park (2026). _Fusion Segment Transformer: Bi-Directional Attention Guided Fusion Network for AI-Generated Music Detection_.** | arXiv:2601.13647. citeturn43view3 | Preprint SSD | AI-generated music detection; **SONICS y AIME**; transformer por segmentos con fusión guiada | El abstract afirma explícitamente que evalúa en **SONICS y AIME** y mejora respecto al modelo previo y baselines recientes. citeturn43view3 | Es, en lo recuperado, la **única** evidencia verificada de uso académico de **AIME** para detección. | AIME usado en detección | Estado del arte / dataset | **Esencial** | **Verificada** |
| **Han; Wang; Gui (2026). _MusicDET: Zero-Shot AI-Generated Music Detection_.** | arXiv:2605.18072. citeturn42view1 | Preprint SSD | Detección zero-shot; FakeMusicCaps y SONICS; normalizing flows guiados por frecuencia | Formula explícitamente el problema de **unseen generators** y entrena solo con música real. citeturn42view1 | Refuerza el argumento de que la generalización a generadores no vistos es un problema central y no resuelto. | Generalización a generadores no vistos | Estado del arte / amenazas a la validez | **Esencial** | **Verificada** |
| **Lopez-Ayala et al. (2026). _AI-Generated Music Detection in Broadcast Monitoring_.** | arXiv:2602.06823. citeturn42view2 | Preprint SSD | Detección de música IA en broadcast; AI-OpenBMAT; SNR y duración | Muestra caídas fuertes cuando la música es corta o está en segundo plano; reporta F1 por debajo de 60% en esos escenarios. citeturn42view2 | Muy útil para no sobreinterpretar resultados en condiciones limpias; apoya la discusión de robustez y duración de fragmentos. | Robustez a duración/SNR; evaluación operacional realista | Estado del arte / amenazas a la validez | **Esencial** | **Verificada** |
| **Han et al. (2026). _Beyond Artifacts: Towards Generalizable Synthetic Song Detection via Music-Intrinsic Features_.** | arXiv:2606.16612. citeturn43view0 | Preprint SSD | SSD; MUSIC8K; mezcla de expertos con rasgos musicales intrínsecos | Presenta **MUSIC8K** con generadores recientes y perturbaciones realistas; reporta mejora de **18,5 puntos F1** sobre el baseline más fuerte en MUSIC8K-O. citeturn43view0turn43view2 | Respaldará muy bien la tesis de que no basta con artefactos de bajo nivel y de que la generalización exige rasgos más “musicales”. | Generalización y robustez | Estado del arte / discusión metodológica | **Esencial** | **Verificada** |
| **Oh (2026). _ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics_.** | arXiv:2604.16254. citeturn42view0 | Preprint SSD | Detección de música IA; ArtifactBench; residuales de códec + CNN ligera | Propone detector de **4,0 M** de parámetros totales; reporta F1 **0,9829** en test no visto y reducción del drift entre códecs del **83%** con entrenamiento codec-aware. citeturn42view0 | Muy útil para dos partes del TFM: robustez a códecs y criterios operacionales/de despliegue por ligereza del modelo. | Robustez a códecs y viabilidad de despliegue | Estado del arte / criterios operacionales | **Esencial** | **Verificada** |
| **Fawcett, T. (2006). _An introduction to ROC analysis_.** | Citación metodológica ampliamente asentada, recuperada indirectamente en fuentes de ROC. citeturn22search0turn22search5 | Artículo metodológico | Evaluación binaria; ROC y AUC | Referencia estándar para justificar **ROC-AUC** y lectura umbral-independiente. | Útil para una memoria que compare modelos con distintos umbrales y quiera reportar ROC-AUC con fundamento metodológico claro. | Métricas predictivas | Metodología / experimentación | **Recomendable** | **Declarada pero no comprobada completamente** |
| **Mowla et al. (2020). _Affective Brain-Computer Interfaces: A Tutorial to Choose Performance Measuring Metric_.** | arXiv:2005.02619. citeturn68academia0 | Tutorial metodológico | Clasificación con desbalance | Recomienda **balanced accuracy** para evitar interpretaciones engañosas con desbalance y menciona su distribución posterior. citeturn68academia0 | No es audio musical, pero sí soporte metodológico claro para elegir balanced accuracy frente a accuracy bruta. | Métricas predictivas | Metodología / experimentación | **Recomendable** | **Verificada** |
| **Giot et al. (2012). _Fast computation of the performance evaluation of biometric systems: application to multibiometric_.** | arXiv:1202.5985. citeturn69academia1 | Artículo metodológico | Biometría / verificación | Describe **EER** como métrica usada para comparar sistemas biométricos. citeturn69academia1 | Te sirve para introducir **EER** solo como métrica transferida desde biometría/audio deepfake, no como métrica nativa de SSD. | Métricas adyacentes | Metodología / comparación con literatura adyacente | **Complementaria** | **Verificada** |

### Documentación oficial y model cards útiles para la futura selección del encoder profundo

| Referencia | DOI / URL oficial | Tipo | Hallazgo útil | Relación exacta con el TFM | Sección recomendada | Prioridad | Estado |
|---|---|---|---|---|---|---|---|
| **Li et al. (2023/2024). _MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training_.** | arXiv:2306.00107. citeturn62view0 | Paper de modelo | Modelo **music-specific**, escala de **95M a 330M**, generaliza a **14 tareas MIR**. citeturn62view0 | Es el candidato más directamente alineado con música dentro de los encoders que estás considerando. | Estado del arte / futura selección de encoder | **Esencial** | **Verificada** |
| **Model card oficial `m-a-p/MERT-v1-95M` y `m-a-p/MERT-v1-330M`.** | Hugging Face. citeturn27view0turn27view1 | Model card oficial | Verifica **licencia cc-by-nc-4.0**, disponibilidad en HF Hub, **24 kHz**, contexto de preentrenamiento **5 s**, tamaños **95M** y **330M**, e indica extracción de embeddings por promediado temporal. citeturn27view0turn27view1 | Fundamenta análisis de dominio, tamaño, sample rate, embeddings congelados y viabilidad aproximada. | Metodología / criterios del modelo profundo | **Esencial** | **Verificada** |
| **Wu et al. (2022). _Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation_.** | arXiv:2211.06687. citeturn51academia3 | Paper de modelo | CLAP se entrena con **LAION-Audio-630K** y persigue representaciones audio-texto generalistas y zero-shot. citeturn51academia3 | Útil para considerar CLAP como encoder congelado, pero su especialización musical es menor que MERT. | Estado del arte / futura selección de encoder | **Recomendable** | **Verificada** |
| **Model card oficial `laion/clap-htsat-unfused` + docs de `transformers` para CLAP.** | Hugging Face. citeturn28view4turn66view2turn66view4 | Model card + docs | Verifica **licencia Apache-2.0**, uso explícito para **feature extraction**, soporte CPU/GPU, **48 kHz** y entrada máxima **10 s** por defecto. citeturn28view4turn66view2 | Muy útil para discutir embeddings congelados y límites prácticos de duración/reamuestreo. | Metodología / criterios del modelo profundo | **Recomendable** | **Verificada** |
| **Gong et al. (2021). _AST: Audio Spectrogram Transformer_.** | arXiv:2104.01778. citeturn25academia2 | Paper de modelo | Encoder general de audio basado en spectrogram transformer; buen rendimiento en AudioSet y ESC-50. citeturn25academia2 | Candidato generalista relativamente compacto y ampliamente disponible. | Estado del arte / futura selección de encoder | **Recomendable** | **Verificada** |
| **Model card oficial `MIT/ast-finetuned-audioset-10-10-0.4593` + docs de AST en Transformers.** | Hugging Face. citeturn29view0turn30view0turn30view3 | Model card + docs | Verifica **licencia BSD-3-Clause**, **86,6M** de parámetros, uso en HF Hub, **16 kHz**, **128 mel bins**, `max_length=1024`, y benchmark interno de inferencia en GPU. citeturn29view0turn30view0turn30view3 | Es el candidato con tamaño verificado más cómodo para una futura prueba bajo restricciones moderadas. | Metodología / criterios del modelo profundo | **Esencial** | **Verificada** |
| **Chen et al. (2022). _BEATs: Audio Pre-Training with Acoustic Tokenizers_.** | arXiv:2212.09058. citeturn50view0 | Paper de modelo | Encoder SSL de audio general; SOTA en AudioSet-2M y ESC-50; código y pesos públicos. citeturn50view0turn50view3 | Relevante como candidato generalista potente, pero menos alineado con música que MERT y con menos verificación recuperada aquí sobre Hub/licencia concreta. | Estado del arte / futura selección de encoder | **Complementaria** | **Verificada** |
| **Hugging Face Hub docs: _Using GPU Spaces_.** | Documentación oficial. citeturn40view0 | Documentación oficial | Detalla hardware actual de Spaces: CPU Basic **16 GB RAM**; T4 small **16 GB VRAM**; L4 **24 GB VRAM**, etc. citeturn40view0 | Es la base documental correcta para discutir viabilidad de despliegue en Spaces y separar esa discusión de tus benchmarks locales con 6 GB VRAM. | Prueba de concepto / despliegue | **Esencial** | **Verificada** |

## Mapa de decisiones y respaldo

| Decisión del TFM | ¿Necesita respaldo bibliográfico? | Referencias útiles | ¿Se justifica sobre todo con bibliografía o con evidencia propia? | Estado actual |
|---|---|---|---|---|
| **Uso de AIME** | Sí, pero con precisión terminológica. | Dataset card de AIME + paper original de preferencias humanas. citeturn61view0turn65view0 | **Mixto**: bibliografía para procedencia y alcance; **auditoría propia** para justificar adecuación al experimento de detección. | **Cerrado** si se redacta explícitamente que AIME no nace como benchmark estándar de detección. |
| **Particionado por `description`** | Sí, aunque no encontrarás una norma “AIME-specific”. | Dataset card de AIME para demostrar que `description` son tres tags de prompt; Nasrullah & Zhao para reforzar la lógica anti-fuga por “producer effect” y split cuidadoso. citeturn61view0turn59view0 | **Principalmente auditoría/diseño propio**, con apoyo bibliográfico indirecto sobre fuga y confounds. | **Parcialmente cubierto**. Conviene presentarlo como decisión metodológica prudente, no como estándar publicado. |
| **MFCC como baseline** | Sí. | Davis & Mermelstein; STC ASVspoof 2015; Nasrullah & Zhao. citeturn72search0turn70academia0turn59view0 | **Bibliografía** para definición/valor histórico; **resultados propios** para defenderlo en tu tarea concreta. | **Cubierto**, salvo DOI clásico por re-verificar si lo quieres impecable en BibTeX. |
| **SVM como baseline** | Sí. | Cortes & Vapnik; STC ASVspoof 2015. citeturn73search0turn70academia0 | **Bibliografía** para el marco clásico; **resultados propios** para defender rendimiento/coste en tu corpus. | **Cubierto**, con la misma cautela de verificación bibliográfica fina que en MFCC. |
| **Fragmentos de audio** | Sí. | Broadcast Monitoring 2026; Nasrullah & Zhao. citeturn42view2turn59view0 | **Mixto**: bibliografía para justificar que la duración influye y que los clips son práctica razonable; **experimentos propios** para fijar la duración final en tu TFM. | **Cubierto**. Conviene no fijar 10 s como decisión universal. |
| **Agregación a nivel de canción** | Sí. | Nasrullah & Zhao documenta mejora al agregar predicciones a canción y discute frame vs song level. citeturn59view0 | **Mixto**: bibliografía para legitimidad metodológica; **evidencia propia** para elegir media, mediana o voto. | **Cubierto**. |
| **Modelo profundo preentrenado** | Sí, pero todavía no para selección definitiva. | MERT, AST, CLAP, BEATs y sus model cards/docs. citeturn62view0turn27view0turn29view0turn51academia3turn28view4turn50view0 | **Primero bibliografía** para preselección y criterios; **después benchmark propio** para decidir. | **Pendiente**, pero ya con shortlist razonable. |
| **Métricas predictivas** | Sí. | SSD recientes reportan sobre todo F1/accuracy; Fawcett para ROC-AUC; Mowla para balanced accuracy; Giot para EER transferido desde biometría. citeturn42view0turn42view2turn43view0turn22search0turn68academia0turn69academia1 | **Bibliografía** para elección y redacción; **resultados propios** para valores concretos. | **Cubierto**. |
| **Métricas operacionales** | Sí, pero débilmente estandarizadas. | Model cards, docs de HF y benchmarks internos de AST. citeturn40view0turn29view0turn30view3turn27view0turn28view4 | **Principalmente benchmarks propios**. | **Abierto pero suficientemente encaminado**. |
| **Selección del modelo desplegado** | Sí. | ArtifactNet para ligereza; model cards/documentación oficial; HF Spaces hardware. citeturn42view0turn40view0turn27view0turn29view0turn28view4 | **Principalmente benchmarks propios** bajo tu entorno de despliegue. | **Pendiente**, como debe estar en esta fase. |

## Bibliografía mínima recomendada

### Referencias esenciales y directas

La selección mínima, si quieres una bibliografía corta pero suficiente para cerrar el estado del arte y la metodología, debería contener al menos la **dataset card oficial de AIME**, el **paper original de AIME**, **Fusion Segment Transformer** como único uso verificado de AIME en detección, y tres trabajos recientes que te fijan el problema metodológico actual: **MusicDET** para **unseen generators**, **Broadcast Monitoring** para **duración/SNR**, y **Beyond Artifacts** o **ArtifactNet** para robustez realista y discusión de señales explotadas por el detector. citeturn61view0turn65view0turn43view3turn42view1turn42view2turn43view1turn42view0

### Referencias metodológicas

Para el baseline clásico, la combinación mínima recomendable es **Davis & Mermelstein** para MFCC, **Cortes & Vapnik** para SVM, y un trabajo adyacente como **STC ASVspoof 2015** para demostrar que MFCC y la comparación SVM/deep han sido razonables en detección de audio sintético cercana. Para particionado, duración y agregación conviene añadir **Nasrullah & Zhao 2019**, porque te da en un solo paper **producer effect**, efecto de **clip length**, y sentido de la agregación **song-level**. Para métricas, con **Fawcett 2006** y **Mowla et al. 2020** basta para sustentar **ROC-AUC** y **balanced accuracy** sin inflar la bibliografía. citeturn70academia0turn59view0turn22search0turn68academia0turn72search0turn73search0

### Referencias adyacentes

Como referencias adyacentes yo mantendría pocas y muy justificadas: **STC ASVspoof 2015** para MFCC/SVM en spoofing próximo y **Giot et al. 2012** solo si necesitas explicar por qué reportas **EER** para dialogar con literatura de deepfake/biometría. No añadiría mucha más voz sintética si no la vas a citar de forma muy precisa, porque corre el riesgo de dispersar el foco del TFM. citeturn70academia0turn69academia1

### Documentación oficial

Para la fase posterior de selección del encoder profundo, la documentación mínima útil es: **MERT paper + model card**, **AST paper + model card**, **CLAP paper + model card/docs**, **BEATs paper**, y **Hugging Face Spaces GPU docs**. Con eso basta para hablar con rigor de **dominio de preentrenamiento, sample rate, tamaño, licencias, embeddings congelados, disponibilidad en Hub y viabilidad aproximada de despliegue** sin sobrecomprometer una decisión todavía abierta. citeturn62view0turn27view0turn29view0turn51academia3turn28view4turn50view0turn40view0

## Entradas BibTeX propuestas

Solo incluyo aquí referencias **verificadas** en las fuentes consultadas. Dejo fuera, de forma deliberada, **Davis & Mermelstein** y **Cortes & Vapnik** hasta re-verificar sus DOI/URL oficiales si quieres un bloque BibTeX absolutamente limpio.

```bibtex
@misc{groetschla2025benchmarking,
  title        = {Benchmarking Music Generation Models and Metrics via Human Preference Studies},
  author       = {Gr{\"o}tschla, Florian and Solak, Ahmet and Lanzend{\"o}rfer, Luca A. and Wattenhofer, Roger},
  year         = {2025},
  eprint       = {2506.19085},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{kim2026fusionsegmenttransformer,
  title        = {Fusion Segment Transformer: Bi-Directional Attention Guided Fusion Network for AI-Generated Music Detection},
  author       = {Kim, Yumin and Park, ???},
  year         = {2026},
  eprint       = {2601.13647},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{han2026musicdet,
  title        = {MusicDET: Zero-Shot AI-Generated Music Detection},
  author       = {Han, Chaolei and Wang, Hongsong and Gui, Jie},
  year         = {2026},
  eprint       = {2605.18072},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{lopezayala2026broadcast,
  title        = {AI-Generated Music Detection in Broadcast Monitoring},
  author       = {Lopez-Ayala, David and Cabello, Asier and Zinemanas, Pablo and Molina, Emilio and Rocamora, Martin},
  year         = {2026},
  eprint       = {2602.06823},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{han2026beyondartifacts,
  title        = {Beyond Artifacts: Towards Generalizable Synthetic Song Detection via Music-Intrinsic Features},
  author       = {Han, Yan and Wen, Zhibin and Wang, Yuan and Shao, Shuangrun and Li, Xiaobing and Xu, Yang and Li, Wei},
  year         = {2026},
  eprint       = {2606.16612},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{oh2026artifactnet,
  title        = {ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics},
  author       = {Oh, Heewon},
  year         = {2026},
  eprint       = {2604.16254},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{nasrullah2019musicartistclassification,
  title        = {Music Artist Classification with Convolutional Recurrent Neural Networks},
  author       = {Nasrullah, Zain and Zhao, Yue},
  year         = {2019},
  eprint       = {1901.04555},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{novoselov2015stcantispoofing,
  title        = {STC Anti-spoofing Systems for the ASVspoof 2015 Challenge},
  author       = {Novoselov, Sergey and Kozlov, Alexandr and Lavrentyeva, Galina and Simonchik, Konstantin and Shchemelinin, Vadim},
  year         = {2015},
  eprint       = {1507.08074},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{li2023mert,
  title        = {MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training},
  author       = {Li, Yizhi and Yuan, Ruibin and Zhang, Ge and Ma, Yinghao and Chen, Xingran and Yin, Hanzhi and Xiao, Chenghao and Lin, Chenghua and Ragni, Anton and Benetos, Emmanouil and Gyenge, Norbert and Dannenberg, Roger and Liu, Ruibo and Chen, Wenhu and Xia, Gus and Shi, Yemin and Huang, Wenhao and Wang, Zili and Guo, Yike and Fu, Jie},
  year         = {2023},
  eprint       = {2306.00107},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{wu2022clap,
  title        = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author       = {Wu, Yusong and Chen, Ke and Zhang, Tianyu and Hui, Yuchen and Nezhurina, Marianna and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  year         = {2022},
  eprint       = {2211.06687},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD}
}
```

```bibtex
@misc{gong2021ast,
  title        = {AST: Audio Spectrogram Transformer},
  author       = {Gong, Yuan and Chung, Yu-An and Glass, James},
  year         = {2021},
  eprint       = {2104.01778},
  archivePrefix= {arXiv},
  primaryClass = {eess.AS}
}
```

```bibtex
@misc{chen2022beats,
  title        = {BEATs: Audio Pre-Training with Acoustic Tokenizers},
  author       = {Chen, Sanyuan and Wu, Yu and Wang, Chengyi and Liu, Shujie and Tompkins, Daniel and Chen, Zhuo and Wei, Furu},
  year         = {2022},
  eprint       = {2212.09058},
  archivePrefix= {arXiv},
  primaryClass = {eess.AS}
}
```

```bibtex
@misc{mowla2020balancedaccuracy,
  title        = {Affective Brain-Computer Interfaces: A Tutorial to Choose Performance Measuring Metric},
  author       = {Mowla, Md Rakibul and Cano, Rachael I. and Dhuyvetter, Katie J. and Thompson, David E.},
  year         = {2020},
  eprint       = {2005.02619},
  archivePrefix= {arXiv},
  primaryClass = {cs.HC}
}
```

```bibtex
@misc{giot2012eer,
  title        = {Fast computation of the performance evaluation of biometric systems: application to multibiometric},
  author       = {Giot, Romain and El-Abed, Mohamad and Rosenberger, Christophe},
  year         = {2012},
  eprint       = {1202.5985},
  archivePrefix= {arXiv}
}
```

## Incorporación en la memoria

| Referencia | Sección concreta de la memoria | Idea o afirmación que debe acompañar | Dónde citarla |
|---|---|---|---|
| Dataset card de AIME | **Dataset** | “AIME reúne 6.000 pistas generadas por 12 modelos, 500 pistas de MTG-Jamendo y conserva el campo `description` con tres tags de prompt; por ello se emplea como corpus documentado y auditado, no como benchmark canónico de detección.” citeturn61view0 | Dataset / Metodología |
| Grötschla et al. 2025 | **Dataset** y **Estado del arte** | “El recurso AIME se originó para benchmarking de generación musical y métricas correlacionadas con preferencia humana, no para detección forense.” citeturn65view0 | Estado del arte / Dataset |
| Fusion Segment Transformer | **Estado del arte** | “Entre las fuentes verificadas, AIME ya ha sido utilizado al menos en un trabajo académico de detección de música generada por IA.” citeturn43view3 | Estado del arte / Dataset |
| Davis & Mermelstein | **Metodología** | “Los MFCC constituyen una representación cepstral clásica de corto plazo y siguen siendo un baseline razonable por simplicidad y coste.” citeturn72search0 | Metodología |
| Cortes & Vapnik | **Metodología** | “El SVM se emplea como clasificador binario clásico de margen máximo sobre descriptores fijos.” citeturn73search0 | Metodología |
| STC ASVspoof 2015 | **Estado del arte adyacente** | “En detección de audio sintético cercana, MFCC y SVM han sido alternativas reales y comparables frente a clasificadores más complejos.” citeturn70academia0 | Estado del arte / Metodología |
| Nasrullah & Zhao 2019 | **Metodología** y **Amenazas a la validez** | “La longitud del fragmento y el esquema de particionado influyen en las métricas; además, la agregación song-level puede mejorar estabilidad.” citeturn59view0 | Metodología / Amenazas a la validez |
| MusicDET | **Estado del arte** | “La generalización a generadores no vistos es un problema de primer orden; por eso interesa comparar modelos con protocolos estrictamente comparables.” citeturn42view1 | Estado del arte / Discusión |
| Broadcast Monitoring 2026 | **Amenazas a la validez** y **Discusión** | “Los buenos resultados en pistas limpias no garantizan robustez cuando la música aparece en fragmentos cortos o bajo speech masking.” citeturn42view2 | Experimentación / Amenazas a la validez |
| Beyond Artifacts | **Estado del arte** | “Los detectores basados solo en artefactos de bajo nivel pueden generalizar peor que enfoques que modelan rasgos musicales más intrínsecos.” citeturn43view1 | Estado del arte / Discusión |
| ArtifactNet | **Estado del arte** y **Prueba de concepto** | “La literatura reciente también explora detectores ligeros y robustos a códec, lo que es relevante para una PoC desplegable.” citeturn42view0 | Estado del arte / Prueba de concepto |
| Fawcett 2006 + Mowla 2020 | **Experimentación** | “Se reportan ROC-AUC y balanced accuracy porque accuracy sola puede ser engañosa, especialmente bajo desbalance; además se incluyen precision/recall/F1 y matriz de confusión.” citeturn22search0turn68academia0 | Experimentación |
| Giot 2012 | **Experimentación** | “EER se usa solo como métrica complementaria para facilitar comparación con literatura adyacente de audio deepfake y biometría.” citeturn69academia1 | Experimentación |
| MERT / AST / CLAP / BEATs | **Metodología** | “La elección del encoder profundo se difiere, pero la preselección se basa en dominio de preentrenamiento, tamaño, sample rate, licencia, disponibilidad de pesos y encaje operacional.” citeturn62view0turn27view0turn29view0turn51academia3turn28view4turn50view0 | Metodología |
| Hugging Face Spaces docs | **Prueba de concepto** | “La viabilidad de despliegue se valorará con benchmarks propios y considerando el hardware disponible en Spaces, que hoy parte de configuraciones muy superiores a 6 GB de VRAM.” citeturn40view0 | Prueba de concepto / Limitaciones |

Un apunte importante para la futura comparación entre encoders: con lo actualmente verificado, **MERT-95M** y **AST** son los candidatos más limpios para una selección posterior bajo restricciones moderadas, pero por razones distintas. **MERT-95M** está preentrenado específicamente para **música**, trabaja a **24 kHz**, usa un contexto de **5 s** y su propia model card enseña una ruta natural de extracción de embeddings; **AST** tiene un tamaño verificado de **86,6M**, licencia BSD y una integración muy madura en `transformers`, lo que lo hace operacionalmente cómodo. **CLAP** es atractivo si te interesa congelar embeddings y aprovechar su cobertura audio-texto, pero su especialización musical es menor; **BEATs** sigue siendo un candidato serio, aunque en las fuentes recuperadas no he podido dejar igual de redondeada la parte de licencia/Hub/tamaño concreto que sí he podido verificar para MERT y AST. La conclusión sobre **6 GB de VRAM** debe presentarse como **inferencial y provisional** hasta medirla: por tamaño verificado y patrón de uso, AST y MERT-95M parecen los más plausibles; MERT-330M es claramente más arriesgado. citeturn27view0turn27view1turn29view0turn30view0turn51academia3turn28view4turn50view0turn40view0

## Lagunas pendientes

La laguna más clara que sigue abierta es la de **AIME como corpus de detección**. Sí tienes la dataset card oficial, sí tienes el paper original, y sí tienes **una** evidencia verificada de uso para AI-generated music detection —**Fusion Segment Transformer**—, pero no he encontrado una pluralidad de trabajos que permita presentar AIME como benchmark asentado dentro de Synthetic Song Detection. La forma correcta de resolver esto en la memoria no es inflar la bibliografía, sino decir explícitamente que **AIME se usa aquí como corpus documentado y auditado**, no como estándar de facto. citeturn61view0turn65view0turn43view3

La segunda laguna es el **particionado por `description`**. La dataset card prueba que ese campo existe y que se deriva de tags de prompt, pero no he localizado una referencia publicada que convierta el split por `description` en protocolo estándar para AIME. Por tanto, la defensa más sólida no será bibliográfica sino metodológica: argumentarás que, dado que `description` refleja el condicionamiento textual, usarlo en el particionado ayuda a limitar fugas semánticas entre train, valid y test. Esa justificación puede apoyarse indirectamente en literatura sobre **confounds por producción/split** en MIR, pero debe presentarse como **decisión prudente de diseño experimental** respaldada por tu auditoría. citeturn61view0turn59view0

La tercera laguna es la ausencia de un estándar fuerte para **métricas operacionales** en SSD. He encontrado soporte útil para inspeccionar **tamaño de modelo, sample rate, licencias, presencia en Hugging Face Hub y disponibilidad de CPU/GPU usage examples**, así como para contextualizar el despliegue en **Hugging Face Spaces**; aun así, nada de eso sustituye un benchmark propio de **tiempo de preprocesamiento, latencia por fragmento y canción, RAM, VRAM, tiempo de carga e inferencia CPU** bajo tu pipeline real. Aquí la evidencia necesaria no es “más bibliografía”, sino una tabla experimental reproducible construida por ti. citeturn40view0turn27view0turn29view0turn28view4

La cuarta laguna, menor pero práctica, es bibliográfica: si quieres un bloque BibTeX impecable para Overleaf, todavía conviene **reverificar directamente** los metadatos finos de **Davis & Mermelstein 1980** y **Cortes & Vapnik 1995** desde su DOI/editorial, porque en esta búsqueda los he recuperado de forma suficiente para usarlos metodológicamente, pero no con el grado de trazabilidad editorial que sí he conseguido para el resto de refs recientes y model cards. Esa verificación final es puramente de higiene bibliográfica, no de fondo científico. citeturn72search0turn73search4