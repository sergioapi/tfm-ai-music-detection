# Revisión puntual de metadatos bibliográficos

## Correcciones cerradas

La única corrección de autoría que he podido cerrar con fuente oficial en esta revisión es **Fusion Segment Transformer**, cuya autoría completa en arXiv figura como **Yumin Kim** y **Seonghyeon Go**. En la misma fuente consta el identificador **arXiv:2601.13647**, con envío del **20 de enero de 2026**, y no aparece una sede editorial revisada por pares en ese registro; por tanto, en este momento debe tratarse como **preprint**. citeturn35view2turn35view0

También he verificado en fuentes editoriales oficiales dos de las referencias metodológicas clásicas que pediste: **Cortes y Vapnik (1995)** en Springer y **Fawcett (2006)** en ScienceDirect. Springer confirma que *Support-vector networks* fue publicado en *Machine Learning*, volumen 20, páginas 273–297, en 1995, con DOI **10.1007/BF00994018**. citeturn4view1 ScienceDirect confirma que *An introduction to ROC analysis* fue publicado en *Pattern Recognition Letters*, volumen 27, número 8, junio de 2006, páginas 861–874, con DOI **10.1016/j.patrec.2005.10.010**. citeturn5view0turn5view1

En cambio, **Davis y Mermelstein (1980)** no queda cerrada como referencia “totalmente verificada” en esta pasada, porque la URL oficial de IEEE Xplore que pude localizar abre una página bloqueada por JavaScript sin exponer metadatos bibliográficos recuperables en este entorno. Por ese motivo, **no la incluyo en el bloque BibTeX depurado** y la dejo expresamente como **pendiente de verificación editorial completa**. citeturn19view0

## Tabla final de control bibliográfico

| Clave BibTeX | Referencia | Naturaleza | Fuente oficial | Estado de verificación | Sección recomendada de la memoria |
|---|---|---|---|---|---|
| `kim2026fusionsegmenttransformer` | Kim, Y.; Go, S. *Fusion Segment Transformer: Bi-Directional Attention Guided Fusion Network for AI-Generated Music Detection*. arXiv:2601.13647, 2026. citeturn35view2 | **Preprint** | arXiv, registro oficial del preprint; DOI DataCite asociado **10.48550/arXiv.2601.13647**. citeturn35view0turn35view2 | **Verificada** | Estado del arte de *Synthetic Song Detection* |
| `cortes1995support` | Cortes, C.; Vapnik, V. *Support-vector networks*. *Machine Learning*, 20, 273–297, 1995. DOI: **10.1007/BF00994018**. citeturn4view1 | **Publicación revisada por pares** | Springer Nature, página oficial del artículo. citeturn4view1 | **Verificada** | Metodología, baseline clásico, justificación de SVM |
| `fawcett2006roc` | Fawcett, T. *An introduction to ROC analysis*. *Pattern Recognition Letters*, 27(8), 861–874, 2006. DOI: **10.1016/j.patrec.2005.10.010**. citeturn5view0turn5view1 | **Publicación revisada por pares** | ScienceDirect / Elsevier, página oficial del artículo. citeturn5view0turn5view1 | **Verificada** | Experimentación, métricas predictivas, ROC-AUC |
| `davis1980comparison` | **No consolidada en esta revisión**: la referencia debe mantenerse fuera del `.bib` depurado hasta cerrar comprobación editorial directa. citeturn19view0 | **Pendiente** | IEEE Xplore, registro oficial localizado pero no legible en este entorno por bloqueo de JavaScript. citeturn19view0 | **Pendiente de verificación editorial completa** | Metodología, fundamentación de MFCC |

## BibTeX depurado

Las siguientes entradas se generan **solo** para las referencias que han quedado totalmente verificadas en fuentes oficiales en esta revisión: arXiv para el preprint de *Fusion Segment Transformer*, Springer para *Support-vector networks* y ScienceDirect para *An introduction to ROC analysis*. citeturn35view2turn4view1turn5view0

```bibtex
@misc{kim2026fusionsegmenttransformer,
  author        = {Yumin Kim and Seonghyeon Go},
  title         = {Fusion Segment Transformer: Bi-Directional Attention Guided Fusion Network for AI-Generated Music Detection},
  year          = {2026},
  eprint        = {2601.13647},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  doi           = {10.48550/arXiv.2601.13647}
}

@article{cortes1995support,
  author  = {Corinna Cortes and Vladimir Vapnik},
  title   = {Support-vector networks},
  journal = {Machine Learning},
  volume  = {20},
  pages   = {273--297},
  year    = {1995},
  doi     = {10.1007/BF00994018}
}

@article{fawcett2006roc,
  author  = {Tom Fawcett},
  title   = {An introduction to ROC analysis},
  journal = {Pattern Recognition Letters},
  volume  = {27},
  number  = {8},
  pages   = {861--874},
  year    = {2006},
  doi     = {10.1016/j.patrec.2005.10.010}
}
```

## Entradas excluidas o pendientes

He aplicado el criterio que pediste de **no conservar entradas con autores incompletos, datos inferidos o signos de interrogación**. En consecuencia, cualquier versión previa de **Fusion Segment Transformer** que no recoja exactamente la autoría **Yumin Kim and Seonghyeon Go** debe sustituirse por la entrada corregida anterior. citeturn35view2

Asimismo, **Davis y Mermelstein (1980)** debe quedar **fuera del bloque BibTeX final** hasta que se complete una comprobación editorial directa en IEEE Xplore o en otra fuente oficial equivalente, porque en esta revisión no he podido cerrar esa verificación con metadatos recuperables desde la página oficial. citeturn19view0