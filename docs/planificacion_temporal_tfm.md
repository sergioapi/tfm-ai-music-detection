# Planificación temporal del TFM

## 1. Datos generales

- **Periodo de planificación:** 22 de junio de 2026 – 15 de septiembre de 2026.
- **Dedicación planificada:** 300 horas.
- **Número de fases:** 12.
- **Enfoque de trabajo:** iterativo e incremental.

## 2. Criterio de planificación

El proyecto sigue un enfoque iterativo e incremental. El trabajo se divide en fases que producen resultados verificables y permiten incorporar progresivamente la investigación, la experimentación, el desarrollo software y la documentación.

La gestión se organiza mediante GitHub Projects y un tablero Kanban. Las tareas se definen mediante issues, se desarrollan en ramas, los cambios se registran mediante commits y se revisan e integran mediante pull requests. La documentación se actualiza progresivamente después de los principales incrementos experimentales y de desarrollo.

## 3. Tabla principal de fases

| Fase | Periodo | Horas | Resultado o entregable |
|---|---|---:|---|
| 1. Alcance, planificación y entorno | 22-30 de junio | 20 h | Alcance y objetivos definidos, viabilidad estudiada y entorno de trabajo configurado |
| 2. Auditoría de AIME y preparación de particiones | 1-11 de julio | 24 h | Conjunto de datos auditado y particiones reproducibles preparadas |
| 3. Revisión bibliográfica y estado del arte | 22 de junio-16 de julio | 26 h | Estado del arte actualizado y bibliografía relevante organizada |
| 4. Baseline MFCC + SVM | 12-14 de julio | 22 h | Baseline implementado y evaluado mediante el protocolo experimental definido |
| 5. Documentación del primer incremento | 15-20 de julio | 18 h | Metodología, dataset, particiones y baseline incorporados a la documentación |
| 6. Selección y adaptación del modelo profundo | 21 de julio-5 de agosto | 52 h | Modelo profundo seleccionado, adaptado y evaluado |
| 7. Comparación experimental y selección del modelo | 6-12 de agosto | 28 h | Comparación predictiva y operacional completada y modelo de despliegue seleccionado |
| 8. Documentación del segundo incremento | 13-16 de agosto | 14 h | Modelo profundo, comparación, resultados y limitaciones documentados |
| 9. Diseño e implementación de la aplicación web | 17 de agosto-2 de septiembre | 48 h | Aplicación web funcional con frontend, backend e integración del modelo seleccionado |
| 10. Pruebas, despliegue y documentación de la aplicación | 3-7 de septiembre | 18 h | Pruebas completadas, despliegue académico realizado y aplicación documentada |
| 11. Revisión final de memoria y artefactos | 8-13 de septiembre | 20 h | Memoria revisada y artefactos técnicos organizados para la entrega |
| 12. Preparación de la presentación | 14-15 de septiembre | 10 h | Presentación y demostración preparadas |
| **Total** | **22 de junio-15 de septiembre** | **300 h** | **TFM preparado para su entrega** |

## 4. Dependencias principales

La auditoría del conjunto de datos precede a los experimentos y proporciona las particiones utilizadas por los modelos. El baseline y el modelo profundo deben compartir el mismo protocolo experimental para que sus resultados sean comparables. La comparación requiere que ambos enfoques hayan sido evaluados y la selección del modelo de despliegue depende conjuntamente de su rendimiento predictivo y operacional.

La aplicación web depende del artefacto, el preprocesamiento y la configuración del modelo seleccionado. El cierre del proyecto depende de la finalización de las pruebas, el despliegue académico y la actualización de la memoria y los artefactos técnicos.