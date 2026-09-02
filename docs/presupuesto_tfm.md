# Presupuesto del TFM

## 1. Criterio general

El presupuesto estima el coste económico asociado al desarrollo del TFM durante una dedicación planificada de 300 horas y el periodo comprendido entre el 22 de junio y el 15 de septiembre de 2026.

Se distinguen los costes imputados del trabajo, la amortización del hardware, la electricidad y la conexión a Internet respecto al coste directo de las herramientas, los servicios y los datos utilizados. Las cantidades relativas al equipo informático, los periféricos, el consumo eléctrico y la conexión se calculan mediante precios estándar y supuestos representativos del mercado español de 2025-2026.

## 2. Recursos humanos

Para estimar el valor económico de la dedicación se toma como referencia el coste laboral medio por hora efectiva publicado por el Instituto Nacional de Estadística para el primer trimestre de 2026, situado en 24,88€.

**Cálculo:** 300 h × 24,88 €/h = 7464.00 €.

| Concepto | Horas | Tarifa horaria | Coste |
|---|---:|---:|---:|
| Trabajo realizado para el desarrollo del TFM | 300 h | 24,88 €/h | 7464.00 € |
| **Total de recursos humanos** | **300 h** |  | **7464.00 €** |

**Fuente:** Instituto Nacional de Estadística, Encuesta Trimestral de Coste Laboral, primer trimestre de 2026.

## 3. Recursos materiales, suministros y comunicaciones

### 3.1. Amortización del hardware

Para el cálculo se adopta como supuesto una vida útil de cuatro años, equivalente a una amortización lineal anual del 25 %. Este porcentaje coincide con el coeficiente lineal máximo previsto por la Agencia Tributaria para equipos destinados a procesos de información.

El periodo imputado es de 86 días. La amortización se calcula mediante:

**Coste amortizado = precio estimado × 86 días / (4 años × 365 días).**

Los precios estimados representan valores habituales del mercado español de 2025-2026:

- Portátil con 16 GB de RAM y GPU dedicada de 6 GB: 900 €.
- Monitor IPS Full HD de 24 pulgadas: 110 €.
- Periféricos básicos, agrupando teclado, ratón y auriculares: 45 €.

| Recurso | Precio estimado | Vida útil | Periodo imputado | Coste amortizado |
|---|---:|---:|---|---:|
| Ordenador portátil | 900,00 € | 4 años | 86 días | 53.01 € |
| Monitor externo | 110,00 € | 4 años | 86 días | 6.48 € |
| Teclado, ratón y auriculares | 45,00 € | 4 años | 86 días | 2.65 € |
| **Total de amortización** |  |  |  | **62.14 €** |

### 3.2. Electricidad

Se utiliza una potencia media estimada de 0,12 kW para el conjunto formado por el portátil, el monitor y los periféricos durante una carga mixta de documentación, programación, procesamiento de audio y ejecución experimental.

Como referencia del precio eléctrico se emplea 0,2669 €/kWh, correspondiente al precio medio doméstico en España durante el segundo semestre de 2025.

**Consumo estimado:** 0,12 kW × 300 h = 36.00 kWh.

**Coste eléctrico:** 36.00 kWh × 0,2669 €/kWh = 9.61 €.
fuente: Eurostat, conjunto de datos nrg_pc_204, segundo semestre de 2025.

| Concepto | Potencia media | Horas | Precio del kWh | Coste |
|---|---:|---:|---:|---:|
| Equipo informático utilizado durante el proyecto | 0,12 kW | 300 h | 0,2669 €/kWh | 9.61 € |
| **Total de electricidad** |  |  |  | **9.61 €** |

### 3.3. Conexión a Internet

Se adopta una cuota estándar de 30 € mensuales para una conexión doméstica de fibra óptica de 300 Mb. El periodo comprendido entre el 22 de junio y el 15 de septiembre equivale aproximadamente a 2.83 meses.

**Cálculo:** 30 €/mes × 2.83 meses = 84.82 €.

| Concepto | Coste mensual | Periodo imputado | Coste |
|---|---:|---:|---:|
| Conexión doméstica de fibra óptica | 30,00 €/mes | 2.83 meses | 84.82 € |
| **Total de comunicaciones** |  |  | **84.82 €** |

## 4. Software, herramientas, servicios y datos

| Recurso | Modalidad | Coste |
|---|---|---:|
| GitHub | Plan gratuito | 0 € |
| Clockify | Plan gratuito | 0 € |
| Visual Studio Code | Gratuito | 0 € |
| Overleaf | Plan gratuito | 0 € |
| Hugging Face Hub | Plan gratuito | 0 € |
| Northflank Developer Sandbox | Prueba validada; 0 € observado durante la prueba, sin garantía futura | 0 € |
| AIME | Acceso gratuito | 0 € |
| **Total de software, servicios y datos** |  | **0 €** |

## 5. Resumen final del presupuesto

| Categoría | Coste |
|---|---:|
| Recursos humanos | 7464.00 € |
| Amortización de hardware | 62.14 € |
| Electricidad | 9.61 € |
| Internet | 84.82 € |
| Software, servicios y datos | 0,00 € |
| **Total estimado** | **7620.57 €** |

## 6. Supuestos utilizados

- Portátil equivalente de gama media con 16 GB de RAM y GPU dedicada de 6 GB: 900 €.
- Monitor IPS Full HD de 24 pulgadas: 110 €.
- Teclado, ratón y auriculares: 45 €.
- Vida útil del hardware: 4 años.
- Periodo imputado: 86 días.
- Potencia media del conjunto informático: 0,12 kW.
- Precio de la electricidad: 0,2669 €/kWh.
- Conexión doméstica de fibra óptica: 30 €/mes.
