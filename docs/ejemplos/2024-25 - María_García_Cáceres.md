|                |     |     | TRABAJO        |     | FIN      | DE MASTER ´ |           |     |       |
| -------------- | --- | --- | -------------- | --- | -------- | ----------- | --------- | --- | ----- |
| Clasificacio´n |     |     |                | de  | G´eneros |             | Musicales |     |       |
| Utilizando     |     |     | Espectrogramas |     |          |             |           | y   | Redes |
Convolucionales
|         |     |             |        |               | Realizado | por         |       |     |            |
| ------- | --- | ----------- | ------ | ------------- | --------- | ----------- | ----- | --- | ---------- |
|         |     |             | Mar´ıa |               | Garc´ıa   | C´aceres    |       |     |            |
|         |     |             | Para   | la obtencio´n |           | del t´ıtulo | de    |     |            |
| M´aster | en  | Ingenier´ıa |        | del           | Software: | Cloud,      | Datos |     | y Gesti´on |
TI
|     |     |              |      |        | Dirigido  | por    |         |     |     |
| --- | --- | ------------ | ---- | ------ | --------- | ------ | ------- | --- | --- |
|     |     |              | Jose | Mar´ıa | Luna      | Romera |         |     |     |
|     |     | Convocatoria |      |        | de Junio, | curso  | 2024/25 |     |     |

Agradecimientos
Quiero agradecer a todas las personas que, de una forma u otra, me han acom-
pan˜ado y apoyado durante el desarrollo de este trabajo y a lo largo del ma´ster.
A Rafa, Aitor, Dani y Alvaro, ´ por ayudarme pr´acticamente desde el primer d´ıa,
hacerme sentir parte del grupo y convertir este camino en algo mucho ma´s agradable.
A mi familia, por su apoyo incondicional incluso en los momentos en los que
solo ten´ıa quejas. A mi padre, especialmente, por animarme a dar el paso de cursar el
ma´ster.
A mi tutor, Jos´e Mar´ıa, por su cercan´ıa y dedicacio´n, siempre dispuesto a resolver
| mis dudas | y a orientarme | con | paciencia. |     |
| --------- | -------------- | --- | ---------- | --- |
A Rub´en, Mar´ıa y Reyes por confiar en m´ı incluso cuando ni yo misma lo hac´ıa.
| Y       | a Fernandito | ¡Bienvenido | a la familia, | enano! |
| ------- | ------------ | ----------- | ------------- | ------ |
| Gracias | a todos.     |             |               |        |
i

Resumen
Este trabajo aborda la clasificacio´n automa´tica de fragmentos musicales en dis-
tintos g´eneros utilizando redes neuronales convolucionales y espectrogramas generados
a partir de diversos archivos de audio. El objetivo principal del proyecto ha sido desa-
rrollar y evaluar un modelo capaz de realizar clasificacio´n multilabel, permitiendo que
un mismo fragmento musical pueda pertenecer a varios g´eneros simulta´neamente, lo
que refleja la complejidad de la mu´sica actual.
En cuanto al an´alisis del estado del arte, se observa una clara evolucio´n desde
m´etodos tradicionales basados en la extraccio´n manual de caracter´ısticas hasta el uso
de arquitecturas profundas y segmentaci´on en fragmentos cortos de audio. Esta u´ltima
estrategia, aplicada con´exito en otros datasets, ha demostrado mejorar la capacidad de
los modelos para capturar patrones representativos y generalizar mejor, especialmente
en tareas de clasificacio´n multilabel.
En el proyecto se ha empleado el dataset MagnaTagATune, conocido por su di-
versidad y riqueza en etiquetas obtenidas a trav´es de encuestas a la poblaci´on, lo que
ha supuesto un reto adicional debido al desbalanceo de clases y a la subjetividad del
etiquetado. Este problema se ha mitigado mediante t´ecnicas de agrupacio´n de g´eneros
y un preprocesamiento espec´ıfico, que incluye la fragmentacio´n de los audios en seg-
mentos de 3 segundos y el uso de ventanas deslizantes, aumentando as´ı la cantidad y
variedad de ejemplos para el entrenamiento del modelo.
Lametodolog´ıacombinalageneraci´ondeespectrogramascomorepresentacio´nvi-
sual del audio y el uso de una arquitectura de red neuronal convolucional optimizada,
que integra t´ecnicas como la normalizacio´n por lotes y el dropout para mejorar la gene-
ralizacio´n y evitar el sobreajuste. El rendimiento del modelo se ha evaluado utilizando
m´etricas espec´ıficas para clasificaci´on multilabel y validacio´n cruzada, obteniendo re-
sultados superiores a los reportados en estudios previos con enfoques similares.
Palabras clave: CNN, Red Convolucional, espectrograma, espectrograma de
Mel, clasificaci´on, clasificacio´n multilabel, MagnaTagATune
ii

Abstract
This work deals with the automatic classification of musical fragments in different
genres using convolutional neural networks and spectrograms generated from different
audio files. The main objective of the project has been to develop and evaluate a model
capable of multilabel classification, allowing the same musical fragment to belong to
several genres simultaneously, which reflects the complexity of today’s music.
Regarding the analysis of the state of the art, there is a clear evolution from
traditionalmethodsbasedonmanualfeatureextractiontotheuseofdeeparchitectures
and segmentation in short audio fragments. The latter strategy, successfully applied in
other datasets, has been shown to improve the models’ ability to capture representative
patterns and to generalise better, especially in multi-label classification tasks.
The project has employed the MagnaTagATune dataset, known for its diversity
and richness in labels obtained through population surveys, which has posed an ad-
ditional challenge due to class imbalance and labelling subjectivity. This problem has
been mitigated by genre clustering techniques and specific preprocessing, including the
fragmentation of the audios into 3-second segments and the use of sliding windows,
thus increasing the number and variety of examples for model training.
Themethodologycombinesthegenerationofspectrogramsasavisualrepresenta-
tionoftheaudioandtheuseofanoptimisedconvolutionalneuralnetworkarchitecture,
which integrates techniques such as batch normalisation and dropout to improve ge-
neralisation and avoid overfitting. The performance of the model has been evaluated
usingspecificmetricsformultilabelclassificationandcross-validation,obtainingresults
superior to those reported in previous studies with similar approaches.
Keywords: CNN, Convolutional Network, spectrogram, Mel-spectrogram, clas-
sification, multilabel classification, MagnaTagATune
iii

´
Indice general
1. Introduccio´n 1
1.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
1.2. Estructura de este documento . . . . . . . . . . . . . . . . . . . . . . . 1
2. Fundamentos teo´ricos 3
2.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2. Inteligencia Artificial . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.3. Aprendizaje Autom´atico . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.3.1. T´ecnicas de Aprendizaje Supervisado . . . . . . . . . . . . . . . 3
2.3.2. T´ecnicas de Aprendizaje No Supervisado . . . . . . . . . . . . . 4
2.4. Clasificacio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.5. Clasificacio´n multilabel . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.6. Redes neuronales convolucionales . . . . . . . . . . . . . . . . . . . . . 6
2.6.1. Capas convolucionales (Conv2D) . . . . . . . . . . . . . . . . . 7
2.6.2. MaxPooling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
2.6.3. Batch Normalization . . . . . . . . . . . . . . . . . . . . . . . . 8
2.6.4. Dropout . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
2.6.5. Capas densas (Dense) . . . . . . . . . . . . . . . . . . . . . . . 9
2.6.6. Funciones de activacio´n . . . . . . . . . . . . . . . . . . . . . . . 9
2.7. M´etricas de evaluaci´on . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
2.7.1. Accuracy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
2.7.2. Curva ROC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
´
2.7.3. AUC (Area Bajo la Curva ROC) . . . . . . . . . . . . . . . . . 11
2.7.4. Precisi´on (Precision) . . . . . . . . . . . . . . . . . . . . . . . . 12
2.7.5. Recall . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
2.7.6. F1 Score . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.8. T´ecnicas de validaci´on y entrenamiento . . . . . . . . . . . . . . . . . . 13
2.8.1. Validacio´n cruzada (K-Fold) . . . . . . . . . . . . . . . . . . . . 13
2.8.2. EarlyStopping . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.8.3. Sobreajuste (Overfitting) . . . . . . . . . . . . . . . . . . . . . . 14
2.9. T´ecnica de optimizacio´n . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2.9.1. M´etodo de aceleracio´n de Nesterov . . . . . . . . . . . . . . . . 14
2.10.Representaciones espectrales del audio . . . . . . . . . . . . . . . . . . 14
2.10.1. Espectrograma . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2.10.2. Escala de Mel . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
2.10.3. Espectrograma de Mel . . . . . . . . . . . . . . . . . . . . . . . 15
3. Estado del Arte 17
3.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
3.2. Enfoques Tradicionales . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
3.3. Segmentaci´on en Fragmentos de 3 Segundos . . . . . . . . . . . . . . . 18
3.4. Redes Neuronales y Representaci´on de Audio . . . . . . . . . . . . . . . 18
iv

3.5. Redes Generativas y T´ecnicas de Deep Learning . . . . . . . . . . . . . 19
3.6. Optimizadores y Arquitecturas . . . . . . . . . . . . . . . . . . . . . . . 19
3.7. Antecedentes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
3.8. Conclusiones . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
4. Estudio previo 23
4.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.2. Objetivos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.3. Metodolog´ıa . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.3.1. Documentaci´on . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
4.3.2. Dataset Utilizado en el Estudio . . . . . . . . . . . . . . . . . . 24
4.3.3. Gesti´on del almacenamiento de los espectrogramas . . . . . . . 25
4.3.4. Implementacio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
4.3.5. Elecci´on de algoritmos . . . . . . . . . . . . . . . . . . . . . . . 26
4.3.6. Entrenamiento y evaluacio´n . . . . . . . . . . . . . . . . . . . . 27
4.4. Planificacio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
4.4.1. Metodolog´ıa de Trabajo . . . . . . . . . . . . . . . . . . . . . . 27
4.4.2. Planificaci´on y fases del proyecto inicial . . . . . . . . . . . . . . 28
4.4.3. Planificaci´on y fases del proyecto real . . . . . . . . . . . . . . . 29
4.5. Presupuesto . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
5. Implementaci´on 33
5.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
5.2. Ana´lisis exploratorio de los datos . . . . . . . . . . . . . . . . . . . . . 33
5.3. Criterios de clasificaci´on de g´eneros musicales . . . . . . . . . . . . . . 34
5.4. Visualizacio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
5.5. Generacio´n de espectrogramas . . . . . . . . . . . . . . . . . . . . . . . 38
5.6. Entrenamiento . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
6. Validacio´n 46
6.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
6.2. Validaci´on del modelo . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
6.3. Comparativa con resultados de estudios previos . . . . . . . . . . . . . 49
7. Conclusiones y trabajo a futuro 50
A. Bibliograf´ıa 52
v

´
Indice de figuras
2.1. T´ecnicas de Aprendizaje Supervisado [41] . . . . . . . . . . . . . . . . . 4
2.2. T´ecnicas de Aprendizaje No Supervisado [41] . . . . . . . . . . . . . . . 5
2.3. Esquema de las CNN por Aphex [2] . . . . . . . . . . . . . . . . . . . . 6
2.4. Explicacio´n curva ROC [40] . . . . . . . . . . . . . . . . . . . . . . . . 11
2.5. Explicacio´n ´area bajo la curva ROC [40] . . . . . . . . . . . . . . . . . 12
2.6. Ejemplo de espectrograma . . . . . . . . . . . . . . . . . . . . . . . . . 15
2.7. Ejemplo de espectrograma de Mel . . . . . . . . . . . . . . . . . . . . . 16
5.1. Distribucio´n canciones por g´enero . . . . . . . . . . . . . . . . . . . . . 36
5.2. Correlacio´n entre g´eneros . . . . . . . . . . . . . . . . . . . . . . . . . . 37
5.3. Nu´mero de g´eneros por cancio´n . . . . . . . . . . . . . . . . . . . . . . 38
5.4. Ejemplos de espectrogramas generados . . . . . . . . . . . . . . . . . . 40
5.5. Redimensionamiento de los espectrogramas . . . . . . . . . . . . . . . . 41
5.6. Distribucio´n de valores de p´ıxeles en un espectrograma . . . . . . . . . 42
5.7. Capas CNN . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
6.1. K-Fold Cross Validation . . . . . . . . . . . . . . . . . . . . . . . . . . 46
6.2. Evolucio´n de las m´etricas de validaci´on por ´epoca . . . . . . . . . . . . 48
vi

´
| Indice | de extractos | de co´digo |
| ------ | ------------ | ---------- |
5.1. Modelo CNN . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
5.2. Entrenamiento con early stopping . . . . . . . . . . . . . . . . . . . . . 44
vii

1. Introducci´on
1.1. Introducci´on
La mu´sica, como forma de expresio´n cultural y art´ıstica, ha evolucionado enor-
memente con el avance de la tecnolog´ıa digital. Hoy en d´ıa, la enorme cantidad de
contenido musical disponible en plataformas digitales plantea retos importantes para
suorganizaci´on,bu´squedayrecomendacio´n.Enestecontexto,laclasificaci´onautoma´ti-
ca de mu´sica por g´eneros o etiquetas se ha convertido en una tarea fundamental para
facilitar la gestio´n y el acceso a grandes colecciones musicales.
El procesamiento de sen˜ales de audio y la extraccio´n de caracter´ısticas relevantes
sonpasosclaveparaabordarestatarea.Entrelasdiversasrepresentacionesposibles,los
espectrogramas destacan por ofrecer una visualizacio´n tiempo-frecuencia que captura
la dina´mica y textura del sonido. Esta representacio´n permite aprovechar t´ecnicas de
visio´n como las redes neuronales convolucionales, que han demostrado gran eficacia en
la detecci´on de patrones complejos en im´agenes y, por extensio´n, en espectrogramas
musicales.
Este trabajo se centra en el disen˜o y evaluaci´on de un modelo basado en CNN
para la clasificaci´on multilabel de fragmentos musicales, utilizando espectrogramas ge-
nerados a partir de archivos de audio. La eleccio´n de un enfoque multilabel responde a
la naturaleza multifac´etica de la mu´sica, donde una pieza puede pertenecer simult´anea-
mente a varios g´eneros o estilos, lo que an˜ade complejidad al problema y requiere una
evaluaci´on cuidadosa mediante m´etricas espec´ıficas.
Para el entrenamiento y validaci´on del modelo se ha utilizado el dataset Magna-
TagATune, reconocido por su diversidad y riqueza en etiquetas musicales. Este dataset
presenta desaf´ıos propios, como el desbalanceo en la distribuci´on de clases. Adema´s,
se ha implementado una estrategia de validacio´n cruzada y mecanismos para evitar el
sobreajuste, garantizando la robustez de los resultados obtenidos.
1.2. Estructura de este documento
A continuacio´n, se describen los contenidos de cada cap´ıtulo:
Introduccio´n: Se presenta el contexto general del trabajo, los objetivos princi-
pales y la motivaci´on que impulsa el estudio. Tambi´en se justifica la relevancia
de aplicar t´ecnicas de deep learning a la clasificacio´n musical.
Fundamentos teo´ricos: Se exponen los conceptos esenciales para comprender
el contenido posterior. Se abordan temas como inteligencia artificial, aprendizaje
automa´tico (supervisado y no supervisado), clasificacio´n (incluyendo clasificacio´n
multilabel) y redes neuronales convolucionales. Adema´s, se detallan las m´etricas
1

de evaluacio´n, t´ecnicas de validacio´n y entrenamiento, y las representaciones es-
| pectrales | del audio utilizadas. |     |     |     |
| --------- | --------------------- | --- | --- | --- |
Estado del arte: Se analiza la literatura relacionada, incluyendo enfoques tra-
dicionales y modernos en la clasificacio´n musical. Se destacan aspectos como la
segmentacio´n de audio, el uso de redes neuronales, arquitecturas relevantes, y se
| contextualiza | el trabajo | respecto | a investigaciones | previas. |
| ------------- | ---------- | -------- | ----------------- | -------- |
Estudio previo: Se describe un estudio preliminar realizado antes del desarro-
llo principal. Se incluyen los objetivos, la metodolog´ıa utilizada, la descripcio´n
del dataset, la gestio´n de espectrogramas, el entorno tecnolo´gico empleado y la
| planificacio´n | temporal | del proyecto. |     |     |
| -------------- | -------- | ------------- | --- | --- |
Implementacio´n: Se detalla el proceso de desarrollo del modelo propuesto. In-
cluye el ana´lisis exploratorio de los datos, los criterios de clasificaci´on utilizados,
la generacio´n de espectrogramas, el proceso de entrenamiento del modelo y otros
| aspectos t´ecnicos | relevantes. |     |     |     |
| ------------------ | ----------- | --- | --- | --- |
Validaci´on: Se presentan los resultados obtenidos durante la validacio´n del mo-
delo. Se realiza una comparativa con estudios previos para evaluar el rendimiento
| y se discuten | los hallazgos | obtenidos. |     |     |
| ------------- | ------------- | ---------- | --- | --- |
Conclusiones y trabajo futuro: Se reflexiona sobre los resultados alcanzados,
las principales dificultades encontradas y las posibles l´ıneas de mejora e investi-
| gacio´n para | trabajos | futuros. |     |     |
| ------------ | -------- | -------- | --- | --- |
2

2. Fundamentos te´oricos
2.1. Introducci´on
En esta seccio´n se explican los conceptos teo´ricos que se han utilizado durante
el desarrollo del modelo, tanto en la parte de entrenamiento como en la evaluaci´on
y validacio´n. Para ello, comenzamos con una introduccio´n a la Inteligencia Artificial,
pasando por los fundamentos del aprendizaje automa´tico y sus principales enfoques.
Finalmente, se detallan conceptos espec´ıficos como la clasificacio´n y la clasificacio´n
multilabel.
2.2. Inteligencia Artificial
La Inteligencia Artificial (tambi´en conocida por sus siglas IA) es un a´rea de
la inform´atica que busca crear programas o sistemas capaces de realizar tareas que
normalmenterequiereninteligenciahumana.Estastareasincluyenentenderellenguaje,
tomar decisiones, aprender de la experiencia, reconocer ima´genes o sonidos [9].
La Unio´n Europea define un sistema de IA como un software que puede reci-
bir informaci´on de su entorno, procesarla, y tomar decisiones o acciones de manera
auto´noma para cumplir ciertos objetivos [9]. Es decir, un sistema de IA no se limita a
seguir instrucciones fijas, sino que puede adaptarse y actuar segu´n la situacio´n, lo cual
lo hace u´til en aplicaciones como asistentes virtuales, coches auto´nomos o sistemas de
recomendacio´n.
2.3. Aprendizaje Autom´atico
El aprendizaje autom´atico (o Machine Learning, ML) es una t´ecnica dentro de
la inteligencia artificial que permite que los sistemas aprendan por s´ı mismos a partir
de los datos. En lugar de programar reglas espec´ıficas, se le dan ejemplos al sistema, y
este aprende a identificar patrones y tomar decisiones por su cuenta [19].
Por ejemplo, si queremos que una ma´quina aprenda a distinguir correos electro´ni-
cos importantes de los que son spam, le damos muchos ejemplos de ambos tipos. Con
el tiempo, el sistema aprende por s´ı mismo a hacer esta distincio´n sin que tengamos
que escribir todas las reglas posibles.
2.3.1. T´ecnicas de Aprendizaje Supervisado
Elaprendizajesupervisadoesuntipodeaprendizajeautoma´ticodondeseentrena
aunmodelocondatosqueyaesta´netiquetados[44].Estosignificaquecadaejemploque
3

seledaalmodeloincluyetantolaentrada(porejemplo,unaimagen)comolarespuesta
correcta(porejemplo,sienlaimagenhayunperrooungato).Deestaforma,elsistema
aprende a relacionar las entradas con las salidas correctas.El aprendizaje supervisado
| se suele | emplear en: |     |     |     |     |     |
| -------- | ----------- | --- | --- | --- | --- | --- |
Clasificacio´n: En este tipo de problemas, el objetivo es predecir una categor´ıa
o etiqueta discreta. Por ejemplo, determinar si un correo electro´nico es spam o
| no, | o identificar | a qu´e g´enero | musical | pertenece | una cancio´n. |     |
| --- | ------------- | -------------- | ------- | --------- | ------------- | --- |
Regresi´on: Aqu´ı, la variable a predecir es un valor num´erico continuo. Un caso
t´ıpico ser´ıa estimar el precio de una vivienda o la temperatura de una ciudad en
| un  | d´ıa determinado. |     |     |     |     |     |
| --- | ----------------- | --- | --- | --- | --- | --- |
Series temporales: Se trabaja con datos recogidos en intervalos regulares de
tiempo para predecir valores futuros bas´andose en patrones y tendencias pasadas.
Por ejemplo, pronosticar las ventas mensuales o la evoluci´on de la temperatura
| a   | lo largo del | an˜o.          |                |     |             |      |
| --- | ------------ | -------------- | -------------- | --- | ----------- | ---- |
|     | Figura       | 2.1: T´ecnicas | de Aprendizaje |     | Supervisado | [41] |
Por ejemplo, siguiendo el esquema de la figura superior 2.1, si queremos que un
sistema reconozca letras escritas a mano (en el gra´fico ser´ıan las etiquetas), le damos
miles de im´agenes de letras y le decimos cu´al letra aparece en cada imagen. Con el
tiempo, el sistema aprende a reconocer letras nuevas por s´ı solo.
| 2.3.2. | T´ecnicas | de Aprendizaje |     | No  | Supervisado |     |
| ------ | --------- | -------------- | --- | --- | ----------- | --- |
IBM [24] define el aprendizaje no supervisado como aquel que se usa cuando no
se tienen etiquetas. En este caso, el sistema intenta encontrar por s´ı mismo patrones
ocultosenlosdatos.Porejemplo,puedeagruparclientesconcomportamientossimilares
4

o detectar situaciones ano´malas, como fallos en sensores, sin que nadie le diga lo que
esta´ “bien” o “mal”.
Este enfoque es muy u´til cuando no es pra´ctico o posible etiquetar grandes canti-
dades de informacio´n. En aplicaciones como el Internet de las Cosas (IoT), donde miles
de dispositivos generan datos continuamente, los m´etodos no supervisados ayudan a
detectar comportamientos anormales sin intervenci´on humana.
Figura 2.2: T´ecnicas de Aprendizaje No Supervisado [41]
Por ejemplo, siguiendo el esquema de la Figura superior 2.2, en una casa inte-
ligente se pueden recopilar datos como el consumo el´ectrico de los electrodom´esticos
(textos, registros, etc.), que se transforman en vectores de caracter´ısticas. Estos vecto-
res se procesan mediante un modelo de aprendizaje no supervisado, el cual identifica
patrones normales de comportamiento sin necesidad de ejemplos etiquetados. Luego,
un modelo predictivo utiliza estos patrones para analizar nuevos datos y asignarles
una probabilidad o un identificador de grupo. De esta forma, si se detecta un consumo
inusual, el sistema puede alertar sobre un posible problema sin haber visto antes ese
caso espec´ıfico.
2.4. Clasificaci´on
Una de las aplicaciones ma´s comunes del aprendizaje automa´tico, tanto super-
visado como no supervisado, es la clasificaci´on. En ella, el modelo aprende a asignar
una entrada a una o varias categor´ıas. Por ejemplo, puede clasificar correos electro´nicos
como “spam” o “no spam”, o identificar emociones en una grabacio´n de voz.
En este proyecto, la tarea principal consiste en clasificar muestras de audio segu´n
diferentes etiquetas, lo cual introduce una variacio´n interesante de la clasificaci´on tra-
dicional: la clasificaci´on multilabel.
5

2.5. Clasificaci´on multilabel
En este proyecto, una misma muestra de audio puede tener varias etiquetas aso-
ciadas,esdecir,puedeperteneceravariosg´eneros.Estosellamaclasificaci´onmultilabel,
y es diferente de la clasificacio´n tradicional, donde cada entrada solo pertenece a una
clase.
En la clasificaci´on multilabel, cada muestra puede estar asociada a mu´ltiples
etiquetasdeformasimult´anea,loqueimplicaqueelmodelodebegenerarunasalidaque
reflejelapresenciaoausenciaindependientedecadaetiqueta.Estosesueleimplementar
aplicando una funcio´n de activacio´n sigmoide (explicado en la seccio´n 2.6.6) en la capa
desalida.Adem´as,laevaluacio´nyelentrenamientodeestosmodelosrequierenm´etricas
y funciones de p´erdida que consideren la naturaleza multilabel, permitiendo optimizar
cada etiqueta por separado y manejando correctamente la coexistencia de mu´ltiples
categor´ıas en una misma instancia.
2.6. Redes neuronales convolucionales
Las redes neuronales convolucionales, conocidas como CNN por sus siglas en
ingl´es, son un tipo de modelo de inteligencia artificial muy bueno para trabajar con
ima´genes. Funcionan especialmente bien cuando los datos tienen una estructura pa-
recida a una rejilla, como ocurre con las ima´genes digitales, que ba´sicamente son una
coleccio´n de p´ıxeles organizados en filas y columnas [5]. Como se puede observar en la
figura 2.3, las CNNs procesan la informaci´on a trav´es de una secuencia de capas espe-
cializadas que transforman progresivamente los datos de entrada en representaciones
ma´s abstractas.
Figura 2.3: Esquema de las CNN por Aphex [2]
6

Estas redes esta´n inspiradas en la forma en que el cerebro humano procesa lo que
vemos, concretamente en c´omo funciona la corteza visual. Gracias a esta inspiraci´on
biolo´gica y su disen˜o t´ecnico, las CNN se han convertido en una herramienta clave en
muchas tareas de visi´on, como reconocer qu´e aparece en una imagen, localizar objetos
dentro de ella o incluso dividirla por partes segu´n su contenido [5].
Gracias a su estructura, las CNN est´an formadas por distintos tipos de capas,
cada una con una funcio´n espec´ıfica dentro del modelo. A continuacio´n, se describen
algunas de las ma´s comunes y fundamentales para el funcionamiento de estas redes:
| 2.6.1. | Capas | convolucionales | (Conv2D) |     |     |
| ------ | ----- | --------------- | -------- | --- | --- |
Las capas convolucionales son el nu´cleo de las redes neuronales convolucionales y
su funcio´n principal es detectar caracter´ısticas espec´ıficas dentro de una imagen, como
bordes, texturas o formas[5]. Para lograrlo, utilizan filtros (tambi´en llamados kernels),
que son pequen˜as matrices de nu´meros que se van deslizando por la imagen, analizando
| fragmentos | pequen˜os | uno a uno. |     |     |     |
| ---------- | --------- | ---------- | --- | --- | --- |
Este proceso se conoce como convoluci´on. Desde un punto de vista matema´tico,
implica combinar los valores del filtro con los de la imagen en cada posicio´n, y el
resultado es una nueva imagen llamada mapa de caracter´ısticas, donde se resaltan los
| patrones | detectados. |     |     |     |     |
| -------- | ----------- | --- | --- | --- | --- |
En una dimensi´on, la operacio´n de convolucio´n entre una entrada x y un filtro w
| se representa | as´ı: |     |     |     |     |
| ------------- | ----- | --- | --- | --- | --- |
(cid:88)
|     |     | s(t) = (x∗w)(t) | =   | ( x(τ)∗w(t−τ) |     |
| --- | --- | --------------- | --- | ------------- | --- |
τ
En el contexto de ima´genes bidimensionales, la convolucio´n se extiende a dos dimen-
siones:
(cid:88)(cid:88)
|     | S(i,j) | = (X ∗W)(i,j) | =   | X(m,n)·W(i−m,j | −n) |
| --- | ------ | ------------- | --- | -------------- | --- |
m n
Donde:
| X   | es la imagen | de entrada.         |     |     |     |
| --- | ------------ | ------------------- | --- | --- | --- |
| W   | es el filtro | o kernel.           |     |     |     |
| S   | es la salida | de la convoluci´on. |     |     |     |
Graciasaestaoperaci´on,laredpuedeaprenderdeformaautom´aticaqu´epatrones
son importantes para la tarea que se le haya asignado, como identificar un objeto o
| clasificar | una imagen | [17]. |     |     |     |
| ---------- | ---------- | ----- | --- | --- | --- |
| 2.6.2.     | MaxPooling |       |     |     |     |
MaxPoolingesunat´ecnicaqueseutilizaparasimplificarlainformaci´onsinperder
lo ma´s importante. Su prop´osito es reducir el taman˜o de las representaciones internas
7

(o mapas de caracter´ısticas) generadas por las capas anteriores, lo que ayuda a que el
| modelo |     | sea ma´s | eficiente | y ra´pido [5]. |     |     |     |
| ------ | --- | -------- | --------- | -------------- | --- | --- | --- |
El m´etodo funciona dividiendo la imagen (o el mapa de caracter´ısticas) en pe-
quen˜as regiones que no se superponen, y dentro de cada una se toma u´nicamente el
valor m´as alto. De esta manera, se conservan las caracter´ısticas ma´s destacadas de cada
zona.
Matema´ticamente,sitomamosunaregio´nRdentrodeunaentradaX,elresultado
| del | MaxPooling |     | se expresa | as´ı: |     |     |     |
| --- | ---------- | --- | ---------- | ----- | --- | --- | --- |
Y = m´ax X(i,j)
(i,j)∈R
Adema´s de reducir el taman˜o de los datos y la carga de ca´lculo, esta operacio´n
hace que la red sea m´as robusta frente a pequen˜os desplazamientos en la imagen, es
decir, sigue reconociendo un objeto aunque est´e levemente cambiado de lugar [5].
| 2.6.3. |     | Batch | Normalization |     |     |     |     |
| ------ | --- | ----- | ------------- | --- | --- | --- | --- |
La normalizacio´n por lotes, conocida como Batch Normalization, es una t´ecnica
muy u´til para mejorar el entrenamiento de redes neuronales profundas [5]. Su objetivo
es hacer que el proceso de aprendizaje sea ma´s ra´pido, m´as estable y ma´s eficiente.
Lo que hace esta t´ecnica es ajustar automa´ticamente los valores que salen de
cada capa para que tengan una media cercana a cero y que su desviaci´on esta´ndar sea
| cercana |      | a uno. |             |                      |     |            |       |
| ------- | ---- | ------ | ----------- | -------------------- | --- | ---------- | ----- |
|         | Para | una    | activacio´n | x, la normalizaci´on |     | se calcula | as´ı: |
x−µ
B
xˆ =
(cid:112)
σ2 +ϵ
B
y = γxˆ+β
Donde:
|     | µ   | y σ2 son | la media | y la varianza | de  | los valores | del mini-lote. |
| --- | --- | -------- | -------- | ------------- | --- | ----------- | -------------- |
|     | B   | B        |          |               |     |             |                |
ϵ es una constante pequen˜a que se an˜ade para evitar errores de divisio´n por cero.
γ y β son valores que la red aprende, y sirven para mantener la flexibilidad del
|     | modelo, | permiti´endole |     | ajustar | la salida | si lo necesita | [5]. |
| --- | ------- | -------------- | --- | ------- | --------- | -------------- | ---- |
En resumen, esta t´ecnica ayuda a que las redes aprendan mejor y m´as ra´pido, sin
que los valores internos se descontrolen durante el entrenamiento.
8

| 2.6.4. | Dropout |     |     |     |     |
| ------ | ------- | --- | --- | --- | --- |
El Dropout es una t´ecnica de regularizacio´n que previene el sobreajuste en una
red neuronal. Durante el entrenamiento, se desactivan aleatoriamente un porcentaje de
neuronas en cada capa, lo que obliga a la red a no depender excesivamente de ninguna
neurona en particular [5]. Matema´ticamente, para una neurona con activacio´n h, el
| Dropout | se aplica | como: |     |     |     |
| ------- | --------- | ----- | --- | --- | --- |
˜
h = h∗r
Donde:
r es una variable aleatoria de Bernoulli con probabilidad p de ser 1 (neurona
|        | activa) y 1−p | de ser | 0 (neurona | desactivada). |     |
| ------ | ------------- | ------ | ---------- | ------------- | --- |
| 2.6.5. | Capas         | densas | (Dense)    |               |     |
Las capas densas, tambi´en conocidas como capas completamente conectadas, co-
nectan cada neurona de la capa anterior con cada neurona de la capa actual. Son
responsables de combinar las caracter´ısticas extra´ıdas por las capas anteriores para
realizar tareas de clasificacio´n o regresio´n [17]. La salida y de una capa densa se calcula
como:
|     |     |     |     | y = f(Wx+b) |     |
| --- | --- | --- | --- | ----------- | --- |
Donde:
|        | x es el vector   | de entrada.    |             |                   |             |
| ------ | ---------------- | -------------- | ----------- | ----------------- | ----------- |
|        | W es la matriz   | de pesos.      |             |                   |             |
|        | b es el vector   | de sesgos.     |             |                   |             |
|        | f es la funcio´n | de activaci´on |             | aplicada elemento | a elemento. |
| 2.6.6. | Funciones        | de             | activacio´n |                   |             |
Las funciones de activacio´n introducen no linealidades en la red, permitiendo que
aprenda representaciones complejas [17]. Las funciones principales son:
ReLU (Rectified Linear Unit): Se define como f(x) = m´ax(0,x). Es amplia-
mente utilizada por su simplicidad y eficacia en la mitigacio´n del problema del
|     | desvanecimiento | del | gradiente | [17]. |     |
| --- | --------------- | --- | --------- | ----- | --- |
Sigmoid: Se define como f(x) = 1 . Es u´til en la capa de salida para tareas
1+e−x
de clasificacio´n multietiqueta, ya que produce valores entre 0 y 1, interpretables
|     | como probabilidades. |     |     |     |     |
| --- | -------------------- | --- | --- | --- | --- |
9

| 2.7. M´etricas |     | de evaluaci´on |     |     |
| -------------- | --- | -------------- | --- | --- |
Las m´etricas de evaluacio´n son fundamentales para medir el rendimiento de un
modelo de inteligencia artificial, ya que permiten cuantificar co´mo de bien se esta´n
| cumpliendo | los objetivos | del sistema | en tareas | espec´ıficas. |
| ---------- | ------------- | ----------- | --------- | ------------- |
En nuestro caso, vamos a centrarnos en aquellas m´etricas que permiten analizar
la capacidad del modelo para clasificar correctamente los datos y manejar posibles
desequilibriosentreclases.Enparticular,seutilizara´nelaccuracy,lacurvaROC,ela´rea
bajo la curva (AUC), la precisio´n (precision), el recall y la m´etrica F1, ya que ofrecen
una visi´on completa del comportamiento del modelo desde distintas perspectivas:
| 2.7.1. | Accuracy |     |     |     |
| ------ | -------- | --- | --- | --- |
Laaccuracymidelaproporci´ondeprediccionescorrectasrealizadasporelmodelo
| sobre el | total de predicciones. |     |     |     |
| -------- | ---------------------- | --- | --- | --- |
Fo´rmula:
|     |     |          |        | TP +TN  |
| --- | --- | -------- | ------ | ------- |
|     |     | Accuracy | =      |         |
|     |     |          | TP +TN | +FP +FN |
Donde:
| TP: | Verdaderos       | Positivos |     |     |
| --- | ---------------- | --------- | --- | --- |
| TN: | Verdaderos       | Negativos |     |     |
| FP: | Falsos Positivos |           |     |     |
| FN: | Falsos Negativos |           |     |     |
Estam´etricaesu´tilenconjuntosdedatosbalanceados,perohayquetenercuidado
en casos en los que las clases est´en desbalanceadas, ya que el modelo puede obtener un
| alto accuracy | simplemente | prediciendo | la clase | mayoritaria. |
| ------------- | ----------- | ----------- | -------- | ------------ |
| 2.7.2.        | Curva ROC   |             |          |              |
La curva ROC es una representacio´n gra´fica que muestra la relaci´on entre la tasa
de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR) a diferentes umbrales
de clasificaci´on.
Fo´rmulas:
| Tasa | de Verdaderos | Positivos | (TPR): |     |
| ---- | ------------- | --------- | ------ | --- |
TP
TPR =
|      |                     |        |     | TP +FN |
| ---- | ------------------- | ------ | --- | ------ |
| Tasa | de Falsos Positivos | (FPR): |     |        |
FP
FPR =
|     |     |     |     | FP +TN |
| --- | --- | --- | --- | ------ |
10

La curva ROC permite visualizar el rendimiento del modelo en distintos puntos
de decisi´on y es especialmente u´til para comparar diferentes modelos.
|     |     | Figura 2.4: | Explicaci´on | curva ROC | [40] |
| --- | --- | ----------- | ------------ | --------- | ---- |
Como se puede ver en la Figura 2.5, se utiliza como referencia la diagonal que
representa un clasificador aleatorio, donde no existe capacidad predictiva real (es decir,
se acierta por puro azar). Los modelos con cierto poder predictivo tienden a ubicarse
por encima de esta diagonal, mientras que un clasificador ideal se acercar´ıa al extremo
superior izquierdo del gra´fico, maximizando la tasa de verdaderos positivos y minimi-
| zando | la de falsos positivos | [40]. |     |     |     |
| ----- | ---------------------- | ----- | --- | --- | --- |
´
| 2.7.3. | AUC (Area | Bajo | la Curva | ROC) |     |
| ------ | --------- | ---- | -------- | ---- | --- |
El AUC o a´rea bajo la curva ROC, cuantifica la capacidad del modelo para
distinguir entre clases. Un AUC de 1.0 indica una clasificaci´on perfecta, mientras que
un AUC de 0.5 sugiere un rendimiento aleatorio [31]. Generalmente, cuanto mayor es
el AUC, mejor es el desempen˜o del modelo en t´erminos de separacio´n entre las clases
| positivas | y negativas. |     |     |     |     |
| --------- | ------------ | --- | --- | --- | --- |
11

|     |     | Figura 2.5: | Explicaci´on | ´area bajo la curva ROC | [40] |
| --- | --- | ----------- | ------------ | ----------------------- | ---- |
En la Figura 2.5 se puede observar co´mo var´ıa el valor del AUC segu´n el poder
predictivo del modelo: un clasificador aleatorio tiene un AUC de 0.5, mientras que
un modelo con cierto poder predictivo puede alcanzar valores alrededor de 0.8. El
clasificador perfecto alcanza el valor ma´ximo de AUC, igual a 1.0. Esta m´etrica resulta
especialmente u´til porque resume en un solo nu´mero el rendimiento general del modelo,
| independientemente |            | del umbral  | de decisi´on | elegido [40]. |     |
| ------------------ | ---------- | ----------- | ------------ | ------------- | --- |
| 2.7.4.             | Precisio´n | (Precision) |              |               |     |
Indica qu´e porcentaje de las etiquetas predichas fueron realmente correctas [17].
Fo´rmula:
TP
|     |     |     | Precision | =   |     |
| --- | --- | --- | --------- | --- | --- |
TP +FP
| 2.7.5. | Recall |     |     |     |     |
| ------ | ------ | --- | --- | --- | --- |
El recall mide capacidad del modelo para identificar todas las instancias positivas
reales.
12

Fo´rmula:
TP
Recall =
|     |     |     |     | TP +FN |
| --- | --- | --- | --- | ------ |
Es crucial en contextos donde es importante minimizar los falsos negativos.
| 2.7.6. | F1 Score |     |     |     |
| ------ | -------- | --- | --- | --- |
El F1 Score es la media armo´nica de la precisio´n y el recall, proporcionando un
| equilibrio | entre ambas | m´etricas. |     |     |
| ---------- | ----------- | ---------- | --- | --- |
Fo´rmula:
Presion∗Recall
|     |     | F1Score | = 2∗ |     |
| --- | --- | ------- | ---- | --- |
Precision+Recall
Es u´til cuando se busca un balance entre precisio´n y recall, especialmente en
| conjuntos | de datos  | desbalanceados. |     |                 |
| --------- | --------- | --------------- | --- | --------------- |
| 2.8.      | T´ecnicas | de validaci´on  |     | y entrenamiento |
Una vez tenemos definido el modelo y preparado el conjunto de datos, el siguiente
paso es entrenarlo y validar su rendimiento. En esta seccio´n se presentan estrategias
clave como la validaci´on cruzada, el uso de mecanismos de parada temprana y m´etodos
para prevenir el sobreajuste e intentar conseguir un modelo robusto.
| 2.8.1. | Validaci´on | cruzada | (K-Fold) |     |
| ------ | ----------- | ------- | -------- | --- |
La validaci´on cruzada K-Fold consiste en dividir el conjunto de datos en k par-
ticiones del mismo taman˜o. En cada iteraci´on, una particio´n se utiliza como conjunto
de validacio´n y las restantes para entrenamiento [34]. Este proceso se repite k veces,
asegurandoquecadaparticio´nactu´ecomoconjuntodevalidacio´nunavez.Estem´etodo
permite evaluar la capacidad de generalizacio´n del modelo y proporciona una estima-
cio´n ma´s robusta del rendimiento, especialmente en conjuntos de datos limitados.
| 2.8.2. | EarlyStopping |     |     |     |
| ------ | ------------- | --- | --- | --- |
El Early Stopping es una t´ecnica de regularizacio´n que detiene el entrenamiento
del modelo cuando el rendimiento en el conjunto de validacio´n deja de mejorar durante
unnu´meropredefinidode´epocas[30].Estaestrategiaprevieneelsobreajuste(explicado
en la secci´on 2.8.3) al evitar que el modelo aprenda patrones espec´ıficos del conjunto
| de entrenamiento | que | no generalizan | bien | a datos no vistos. |
| ---------------- | --- | -------------- | ---- | ------------------ |
13

| 2.8.3. | Sobreajuste | (Overfitting) |     |     |
| ------ | ----------- | ------------- | --- | --- |
El sobreajuste ocurre cuando un modelo aprende demasiado bien los detalles y el
ruido del conjunto de entrenamiento, resultando en un rendimiento deficiente en datos
nuevos [33]. Para mitigar este problema, se emplean t´ecnicas de regularizacio´n como
Dropout (explicado en la secci´on 2.6.4), que desactiva aleatoriamente neuronas.
| 2.9. | T´ecnica | de optimizacio´n |     |     |
| ---- | -------- | ---------------- | --- | --- |
Una vez establecidas las estrategias de validacio´n y entrenamiento, el siguiente
paso es seleccionar una t´ecnica de optimizacio´n que permita ajustar los par´ametros del
modelo de forma eficiente. La elecci´on de un optimizador influye directamente en la
velocidad y estabilidad del aprendizaje. A continuacio´n, se presenta el m´etodo de ace-
leracio´n de Nesterov, una t´ecnica utilizada por su capacidad de mejorar la convergencia
| en redes | neuronales | profundas.      |             |     |
| -------- | ---------- | --------------- | ----------- | --- |
| 2.9.1.   | M´etodo    | de aceleracio´n | de Nesterov |     |
El m´etodo de aceleracio´n de Nesterov, tambi´en conocido como Nesterov Acce-
lerated Gradient (NAG), es una t´ecnica de optimizaci´on que mejora el m´etodo del
gradiente descendente mediante la incorporaci´on del m´etodo del descenso del gradiente
[37]. A diferencia del momentum cla´sico, donde la actualizacio´n se realiza basa´ndose en
la posicio´n actual, NAG realiza un paso hacia adelante en la direcci´on del momentum
antes de calcular el gradiente [13]. Esta anticipaci´on permite al algoritmo corregir la
direccio´n si se detecta que el gradiente cambia significativamente, resultando en una
| convergencia | ma´s r´apida     | y estable. |             |           |
| ------------ | ---------------- | ---------- | ----------- | --------- |
| 2.10.        | Representaciones |            | espectrales | del audio |
Una forma de representar sen˜ales de audio en tareas de procesamiento y an´ali-
sis autom´atico es mediante representaciones espectrales, que muestran co´mo var´ıa el
contenido de frecuencia de una sen˜al a lo largo del tiempo [14]. Estas representaciones
permiten identificar caracter´ısticas espec´ıficas como ritmos, timbres o patrones tempo-
rales, y son fundamentales en aplicaciones como clasificacio´n musical, reconocimiento
| de voz  | y s´ıntesis sonora | [6]. |     |     |
| ------- | ------------------ | ---- | --- | --- |
| 2.10.1. | Espectrograma      |      |     |     |
Como se explica en el trabajo de Paz et al. [38], el espectrograma se genera
mediante la transformada de Fourier de tiempo corto (Short-Time Fourier Transform,
STFT), que descompone la sen˜al en ventanas temporales sucesivas para obtener su
contenido frecuencial. El resultado es una representacio´n bidimensional con el tiempo
14

en el eje horizontal, la frecuencia en el eje vertical y la intensidad codificada por el
color. Dependiendo del objetivo, las frecuencias pueden representarse en escala lineal
o logar´ıtmica.
|     |     | Figura | 2.6: Ejemplo | de espectrograma |     |
| --- | --- | ------ | ------------ | ---------------- | --- |
Entrelasvariantesdelespectrogramaseencuentranelespectrogramadepotencia,
el logar´ıtmico y el de Mel, cuya elecci´on depende del tipo de tarea a realizar, como
| clasificacio´n, | segmentaci´on | o s´ıntesis | [39]. |     |     |
| --------------- | ------------- | ----------- | ----- | --- | --- |
| 2.10.2.         | Escala        | de Mel      |       |     |     |
La escala de Mel es una escala perceptiva de frecuencias basada en c´omo los
seres humanos perciben los cambios de tono, en lugar de usar una escala f´ısica lineal.
Estudios psicoacu´sticos muestran que el o´ıdo humano es ma´s sensible a diferencias
en frecuencias bajas que en altas, lo que motiva una transformacio´n no lineal de las
frecuencias [42]. La relacio´n entre la frecuencia f´ısica y la frecuencia en Mels se expresa
| mediante | la f´ormula: |     |     |     |     |
| -------- | ------------ | --- | --- | --- | --- |
(cid:18) (cid:19)
f
|     |     | Mel(f) | = 2595·log | 1+  | (2.1) |
| --- | --- | ------ | ---------- | --- | ----- |
10
700
Estatransformaci´onsehaconvertidoenunesta´ndarenelprocesamientodeaudio,
ya que ajusta mejor la representaci´on espectral a la percepci´on humana [47].
| 2.10.3. | Espectrograma |     | de Mel |     |     |
| ------- | ------------- | --- | ------ | --- | --- |
El espectrograma de Mel es una variante del espectrograma que utiliza un banco
de filtros triangulares distribuidos segu´n la escala de Mel, donde cada filtro simula
una banda cr´ıtica del o´ıdo humano. El proceso consiste en aplicar la STFT a la sen˜al,
proyectar el resultado sobre la escala de Mel mediante los filtros, y finalmente aplicar
| una compresi´on | logar´ıtmica. |     |     |     |     |
| --------------- | ------------- | --- | --- | --- | --- |
15

Esta representacio´n es especialmente u´til en tareas que involucran percepci´on au-
ditiva, como reconocimiento de voz o clasificaci´on de sonidos ambientales, y ha demos-
trado mejorar el rendimiento de modelos basados en redes neuronales convolucionales
(CNN) [8, 21].
Figura 2.7: Ejemplo de espectrograma de Mel
16

3. Estado del Arte
3.1. Introducci´on
La mu´sica ha experimentado una transformaci´on significativa en la era digital.
El creciente volumen de datos musicales disponibles en formato digital y la diversidad
de g´eneros y subg´eneros han generado la necesidad de desarrollar herramientas au-
toma´ticas que permitan la categorizacio´n de informacio´n de manera eficiente y precisa
[12].
El campo de la clasificaci´on autom´atica de g´eneros musicales se ha beneficiado del
avance en t´ecnicas de procesamiento de sen˜ales, machine learning y deep learning [3].
Las primeras aproximaciones basadas en m´etodos estad´ısticos tradicionales han evolu-
cionado hacia arquitecturas complejas de redes neuronales profundas, siendo capaces
de capturar patrones contenidos dentro de los datos musicales. Estas t´ecnicas han in-
crementado la precisio´n de los sistemas de clasificaci´on, abriendo nuevas posibilidades
para aplicaciones como los sistemas de recomendacio´n o las plataformas de streaming
musical.
Adema´s, para mejorar el rendimiento de los sistemas de clasificacio´n, la segmen-
tacio´n de audio y el preprocesamiento de datos han demostrado ser elementos cruciales
[1]. En particular, el an´alisis de fragmentos cortos de audio ha permitido a los modelos
identificar caracter´ısticas relevantes de los g´eneros musicales, reduciendo la influencia
de elementos irrelevantes o ruidosos presentes en grabaciones m´as largas [32].
Enesteestadodelarte,sepresentanlasprincipalesaproximacionesalproblemade
laclasificacio´nautom´aticadeg´enerosmusicales,abarcandodesdem´etodostradicionales
hasta enfoques modernos basados en redes neuronales y t´ecnicas generativas.
3.2. Enfoques Tradicionales
Los primeros sistemas autom´aticos de clasificaci´on de g´eneros musicales se ba-
saban en modelos de mezcla gaussiana (Gaussian Mixture Model, GMM) y modelos
ocultos de Markov (Hidden Markov Model, HMM) para capturar caracter´ısticas acu´sti-
cas de los audios [36]. Sin embargo, estos m´etodos presentaban limitaciones, como una
menor capacidad para modelar dependencias a largo plazo y relaciones no lineales entre
los datos.
Bala Ganesh et al. [4] indica que el uso de coeficientes cepstrales de frecuencia
Mel (Mel-Frequency Cepstral Coefficients, MFCC) destaca como t´ecnica clave para
la extracci´on de caracter´ısticas. Los Mel-Frequency Cepstral Coefficients capturan las
frecuenciasma´srelevantesparalapercepci´onhumana,loqueloshaceu´tilesentareasde
clasificacio´n. Sin embargo, al depender de estad´ısticas globales, estas t´ecnicas carecen
de la capacidad de analizar patrones complejos que se extienden en el tiempo.
17

Otrostrabajos,comoeldeJainetal.[25],empleanrepresentacioneseneldominio
frecuencialparaextraercaracter´ısticasdelaudioypermitirlaidentificacio´ndepatrones
espec´ıficos. Aunque estas t´ecnicas han sido fundamentales, su aplicabilidad inicial se
limitaba a escenarios con menor complejidad musical y conjuntos de datos reducidos.
3.3. Segmentaci´on en Fragmentos de 3 Segundos
Uno de los descubrimientos m´as significativos en la clasificacio´n autom´atica de
g´enerosmusicaleseslaimportanciadeladuraci´ondelosfragmentosdeaudioutilizados
en el ana´lisis. En el trabajo de Ndou et al. [36], se demuestra que dividir grabaciones
de 30 segundos en fragmentos ma´s cortos, espec´ıficamente de 3 segundos, mejora no-
tablemente el rendimiento de los modelos de clasificaci´on. Adema´s,la investigaci´on de
Dong [12] ya observ´o que la precisi´on de clasificacio´n humana se estabiliza despu´es de
escuchar solo 3 segundos de mu´sica, sin mejorar significativamente con fragmentos ma´s
largos.
Dong [12] indica que el ana´lisis de fragmentos m´as cortos se adopt´o, en parte,
como una estrategia de “divide y vencera´s” porque no es factible alimentar el espectro-
grama completo de una sen˜al musical de 30 segundos a una red neuronal convolucional
(CNN) debido a su alta dimensio´n. Al dividir el audio original de 30 segundos en
mu´ltiples segmentos de 3 segundos, generalmente consecutivos y a menudo con sola-
pamiento, se logra un conjunto de datos considerablemente ma´s amplio y diversificado
para el entrenamiento. Por ejemplo, el estudio de Ndou et al. [36] duplico´ el conjun-
to de datos GTZAN de 1000 pistas de 30 segundos para crear 10,000 fragmentos de
3 segundos. Cada segmento puede tratarse como una instancia independiente para el
entrenamiento del modelo, lo que aumenta la capacidad del modelo para generalizar y
capturar patrones relevantes espec´ıficos del g´enero.
EnelconjuntodedatosGTZAN[36],elusodefragmentosde3segundospermitio´
que un clasificador basado en k-Nearest Neighbors (kNN) alcanzara una precisio´n del
92.69%, superando significativamente los resultados obtenidos con grabaciones com-
pletas de 30 segundos y con otros clasificadores tradicionales. Otro estudio realizado
por Bala Ganesh et al. [4] utilizando una arquitectura GenreNet con optimizaci´on Na-
dam logr´o una precisio´n del 84.50% con fragmentos de 3 segundos de la base de datos
GTZAN, en comparacio´n con solo el 33.00% utilizando las pistas completas de 30 se-
gundos. Esta mejora generalizada se atribuye a que los fragmentos ma´s cortos, y las
caracter´ısticas derivadas de ellos (como los mel-espectrogramas generados a partir de
estos segmentos), capturan mejor las transiciones y variaciones locales en la mu´sica,
esenciales para distinguir g´eneros musicales.
3.4. Redes Neuronales y Representaci´on de Audio
Con la introducci´on de redes neuronales profundas, especialmente redes convolu-
cionales (Convolutional Neural Networks, CNN), la clasificaci´on de g´eneros musicales
experimento´ mejoras significativas en precisio´n y capacidad de generalizar [36, 4]. Las
18

Convolutional Neural Networks pueden procesar espectrogramas como si fueran im´age-
| nes, extrayendo |     | patrones |     | complejos |     | de tiempo | y frecuencia. |     |     |
| --------------- | --- | -------- | --- | --------- | --- | --------- | ------------- | --- | --- |
Para que las redes neuronales puedan procesar datos de audio, estos deben ser
transformados en un formato adecuado. Una representacio´n muy comu´n y efectiva es
el espectrograma, una imagen bidimensional que visualiza la intensidad de las frecuen-
cias de una sen˜al de audio a lo largo del tiempo [14]. Dentro de estas representaciones
visuales, el mel-espectrograma destaca por imitar la percepcio´n auditiva humana, dan-
do mayor importancia a las frecuencias bajas [4]. Al tratar el audio como una imagen
(espectrograma o mel-espectrograma), se pueden aplicar arquitecturas de CNN origi-
| nalmente | disen˜adas |     | para | visio´n | artificial. |     |     |     |     |
| -------- | ---------- | --- | ---- | ------- | ----------- | --- | --- | --- | --- |
En resumen, la aplicacio´n de CNNs a representaciones visuales del audio como los
espectrogramas ha permitido que los modelos aprendan caracter´ısticas discriminativas
de forma automa´tica, superando a menudo las t´ecnicas basadas en caracter´ısticas ma-
nuales y alcanzando niveles de precisio´n significativamente ma´s altos en la clasificacio´n
| de g´eneros | musicales. |             |     |     |     |     |           |         |          |
| ----------- | ---------- | ----------- | --- | --- | --- | --- | --------- | ------- | -------- |
| 3.5.        | Redes      | Generativas |     |     |     | y   | T´ecnicas | de Deep | Learning |
Los modelos generativos, como las redes adversarias generativas (Generative Ad-
versarial Networks, GAN), han demostrado ser herramientas u´tiles para mejorar la
clasificacio´n de g´eneros musicales. Las GAN pueden generar muestras sint´eticas que
imitan audios reales, ampliando as´ı los conjuntos de datos de entrenamiento y redu-
| ciendo | problemas | de  | sobreajuste |     | [14]. |     |     |     |     |
| ------ | --------- | --- | ----------- | --- | ----- | --- | --- | --- | --- |
El enfoque propuesto por Dwivedi and Islam [14] combina GAN con transfor-
madas matema´ticas como la transformada de Fourier y la transformada Wavelet para
extraer caracter´ısticas tanto del dominio de frecuencia como del dominio temporal
[14]. Esto permite capturar patrones espec´ıficos de los g´eneros, como la presencia de
| instrumentos |     | caracter´ısticos |     | o   | estructuras |     | r´ıtmicas. |     |     |
| ------------ | --- | ---------------- | --- | --- | ----------- | --- | ---------- | --- | --- |
Adema´s, la combinacio´n de GAN con autoencoders permite reducir el ruido en
los datos y mejorar la robustez de los modelos. Este enfoque ha demostrado ser parti-
| cularmente | efectivo      |     | en escenarios |     | con | datos         | limitados | [16]. |     |
| ---------- | ------------- | --- | ------------- | --- | --- | ------------- | --------- | ----- | --- |
| 3.6.       | Optimizadores |     |               |     | y   | Arquitecturas |           |       |     |
El uso de optimizadores avanzados, como Adam, RMSProp y NADAM, ha tenido
un impacto significativo en la mejora del rendimiento de los modelos en la clasificaci´on
| de g´eneros | musicales: |     |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
ADAM (Adaptive Moment Estimation): Es un algoritmo de optimizacio´n
basado en el ca´lculo de momentos (promedio y varianza) de los gradientes. Com-
bina las ventajas de algoritmos como AdaGrad (Adaptive Gradient Algorithm)
y RMSProp (Root Mean Square Propagation), adaptando la tasa de aprendizaje
19

para cada para´metro de manera individual. Esto le permite encontrar soluciones
ma´s r´apidamente y de manera ma´s estable, incluso en problemas que involucran
grandes cantidades de datos.[28].
RMSProp (Root Mean Square Propagation): Ajusta co´mo se realizan los
cambios en los par´ametros del modelo, tomando en cuenta los gradientes anterio-
res. Esto ayuda a que el proceso de optimizaci´on sea m´as estable, especialmente
cuando los gradientes tienen fluctuaciones o cambios impredecibles. [22].
NADAM (Nesterov-accelerated Adaptive Moment Estimation): NA-
DAM es una versi´on mejorada de ADAM que incorpora el m´etodo de aceleracio´n
de Nesterov (explicado en el apartado 2.9.1). Mientras que ADAM ajusta la tasa
de aprendizaje de cada para´metro utilizando el promedio y la varianza de los
gradientes, NADAM lo hace de manera ma´s inteligente, anticipando hacia do´nde
se movera´n los gradientes en el siguiente paso. Esta anticipaci´on hace que las
actualizaciones sean m´as precisas, lo que permite que el algoritmo encuentre la
mejor solucio´n en menos tiempo. Al combinar la flexibilidad de Adam con la
prediccio´n de Nesterov, NADAM puede optimizar el modelo m´as ra´pidamente
en ciertos problemas, especialmente cuando los gradientes cambian mucho o son
complicados [13].
Bala Ganesh et al. [4], por ejemplo, emplea NADAM para ajustar los pesos de la
red y logra una precisi´on del 87% al trabajar con fragmentos de 3 segundos. Por otro
lado, tambi´en arquitecturas como ResNet y DenseNet se han utilizado exitosamente
para analizar espectrogramas:
ResNet: ResNet (Red Residual) utiliza conexiones residuales para mitigar el
problemadeldesvanecimientodelgradiente,permitiendoentrenarredesmuypro-
fundas. Estas conexiones facilitan la propagaci´on de la informaci´on a trav´es de la
red, lo que resulta en una mayor capacidad para extraer caracter´ısticas relevantes
de los datos [20].
DenseNet: DenseNet (Red Densamente Conectada) conecta cada capa a todas
las capas posteriores, lo que fomenta la reutilizaci´on de caracter´ısticas y reduce
significativamente el nu´mero de para´metros [23]. Esto facilita una mejor genera-
lizacio´n del modelo al analizar espectrogramas complejos.
Estas arquitecturas permiten extraer caracter´ısticas jera´rquicas, lo que mejora la
capacidad del modelo para generalizar a nuevos datos y contribuye a la robustez en la
clasificacio´n de g´eneros musicales [36, 4].
3.7. Antecedentes
En la secci´on 3.2 he tratado los enfoques tradicionales, pero actualmente se prio-
rizan modelos que aprenden representaciones directamente del audio, reduciendo la
intervenci´on manual. Un ejemplo destacado es el de Nam et al. [35], cuyo modelo en
dos etapas primero proyecta patrones espectrales en un espacio de alta dimensionalidad
y luego utiliza deep learning no supervisado para inicializar una red neuronal que se
20

entrena posteriormente de forma supervisada, logrando buenos resultados en el dataset
MagnaTagATune.
Paralelamente, Dieleman and Schrauwen [11] han realizado aportes fundamen-
tales en el aprendizaje de caracter´ısticas directamente desde el audio. En su trabajo,
investigan la posibilidad de entrenar redes neuronales convolucionales directamente
sobre la sen˜al de audio cruda, sin depender de representaciones intermedias como es-
pectrogramas o MFCCs. Sus resultados muestran que las redes pueden descubrir de
maneraaut´onomadescomposicionesfrecuencialesyrepresentacionesinvariantesdefase
y traslaci´on, abriendo la puerta a sistemas verdaderamente “end-to-end”.
Otro trabajo relevante es el de Van Den Oord et al. [45], donde exploran el apren-
dizajeportransferencia.Supropuestaconsisteenpreentrenarredesprofundasdeforma
supervisada en una tarea relacionada y transferir ese conocimiento a la tarea objetivo
de clasificaci´on musical. Este enfoque demostr´o que el aprendizaje por transferencia
puede mejorar la generalizacio´n y el rendimiento, especialmente cuando los datos de
entrenamiento son limitados.
En el art´ıculo de Dieleman and Schrauwen [10] se desarrollan y comparan tres
estrategias para el aprendizaje de caracter´ısticas multiescala. Demuestran que incor-
porar informaci´on de diferentes escalas temporales mejora el rendimiento en tareas de
autoetiquetado y aprendizaje de similitud, ya que distintos tipos de etiquetas (g´eneros,
instrumentos, estados de a´nimo) pueden depender de patrones presentes en diferentes
escalas temporales.
Hamel et al. [18] profundiza en la importancia del pooling temporal y el aprendi-
zaje multiescala para capturar la estructura jera´rquica de la mu´sica. Sus experimentos
muestran que el uso de arquitecturas capaces de agregar informaci´on a lo largo de
distintas escalas temporales mejora tanto la anotacio´n autom´atica como el ranking de
mu´sica, sentando las bases para el desarrollo de modelos ma´s robustos y vers´atiles.
Los resultados de todos estos trabajos muestran valores de AUC en un rango
muy pro´ximo (entre 0.88 y 0.89), destacando la consistencia y madurez de las t´ecnicas
actuales para el autoetiquetado musical en este conjunto de datos.
3.8. Conclusiones
El an´alisis del estado del arte en la clasificaci´on automa´tica de g´eneros musicales
revela una evolucio´n significativa desde los m´etodos tradicionales basados en la ex-
traccio´n manual de caracter´ısticas hasta el uso de arquitecturas profundas y t´ecnicas
generativas avanzadas. Estas innovaciones han permitido mejorar la precisio´n y robus-
tez de los sistemas, facilitando aplicaciones en recomendacio´n, organizaci´on de grandes
cata´logos musicales y ana´lisis cultural.
Un hallazgo especialmente relevante es la eficacia de la segmentaci´on en frag-
mentos cortos de audio, concretamente de 3 segundos, que ha demostrado mejorar el
rendimiento de los modelos al permitirles capturar patrones locales m´as representativos
y reducir la influencia de variaciones irrelevantes presentes en fragmentos largos. Esta
estrategia, validada en datasets como GTZAN, podr´ıa aplicarse al conjunto de datos
21

MagnaTagATune, donde hasta ahora la mayor´ıa de los enfoques han trabajado a nivel
de clip completo. Implementar la segmentacio´n en MagnaTagATune permitir´ıa aumen-
tar el nu´mero de muestras de entrenamiento, diversificar los patrones aprendidos y,
potencialmente, mejorar la capacidad de generalizaci´on de los modelos, especialmente
en un contexto multilabel.
22

4. Estudio previo
4.1. Introducci´on
Esteapartadorecogelosobjetivosplanteados,lametodolog´ıaseguidayelentorno
tecnolo´gico utilizado, as´ı como la documentacio´n consultada y los datos empleados.
Tambi´en se detalla la planificacio´n inicial y real del trabajo, junto con una estimaci´on
presupuestaria.
4.2. Objetivos
Este Trabajo de Fin de Ma´ster tiene como objetivo disen˜ar e implementar un
sistema de clasificacio´n de g´eneros musicales basado en redes neuronales convolucio-
nales (Convolutional Neural Networks, CNNs [2.6]), optimizando el almacenamiento y
procesamiento de los espectrogramas generados a partir de los fragmentos de audio.
Para ello, se utilizara´ el conjunto de datos MagnaTagATune, que contiene una
amplia variedad de fragmentos musicales etiquetados con distintos g´eneros y carac-
ter´ısticas. El desarrollo de este proyecto incluira´ la preparaci´on y preprocesamiento de
los audios, la implementaci´on del modelo de clasificaci´on y la evaluacio´n de su desem-
pen˜o mediante m´etricas espec´ıficas.
A continuacio´n, se presentan los principales objetivos del estudio:
1. Disen˜ar y desarrollar un sistema de clasificacio´n automa´tica de g´eneros musicales
basado en redes neuronales convolucionales.
2. Implementar un proceso de preprocesamiento que segmente y transforme los au-
dios en espectrogramas.
3. Analizar el rendimiento del modelo utilizando m´etricas para la clasificaci´on de
audio.
4.3. Metodolog´ıa
En este apartado se describe la metodolog´ıa seguida para el desarrollo del sistema
de clasificacio´n automa´tica de g´eneros musicales. Se detallan el conjunto de datos utili-
zado, las decisiones relativas a su almacenamiento y preprocesamiento, la implementa-
cio´n del modelo, los algoritmos de aprendizaje empleados, as´ı como los procedimientos
deentrenamientoyevaluacio´n.Finalmente,seexplicanlasherramientasutilizadaspara
la documentacio´n y gesti´on del proyecto.
23

4.3.1. Documentacio´n
La documentacio´n del proyecto se redactar´a y organizar´a utilizando Overleaf,
una plataforma colaborativa basada en LaTeX que facilita la creaci´on de documentos
t´ecnicos de forma profesional. Esta herramienta permitira´ estructurar el informe, inte-
grar gr´aficos y tablas, y mantener un control de versiones efectivo del texto. Adema´s,
se utilizara´ ChatGPT-4 como apoyo para mejorar la formalidad y claridad en la re-
daccio´n, as´ı como para resolver dudas puntuales relacionadas con la implementaci´on
y depuraci´on del co´digo. El uso de control de versiones mediante Git y, adema´s, el
uso de GitHub permitira´ mantener un historial detallado tanto del c´odigo como de la
documentacio´n, garantizando trazabilidad y reproducibilidad.
4.3.2. Dataset Utilizado en el Estudio
El dataset utilizado se titula MagnaTagATune 1 y es una base de datos que se uti-
liza frecuentemente en tareas de clasificacio´n musical, como por ejemplo en los art´ıculos
de Kim et al. [27] y Won et al. [46]. El objetivo inicial de este dataset era proporcionar
anotaciones detalladas realizadas por humanos, convirti´endolo en un recurso valioso
para tareas de aprendizaje autom´atico en el a´mbito del procesamiento de audio.
Descripci´on del Dataset
El conjunto de datos MagnaTagATune [29] esta´ compuesto por un total de 25,863
clips de audio, cada uno con una duraci´on de 29 segundos. Estos fragmentos han sido
extra´ıdos de 5,223 canciones y abarcan una amplia variedad de g´eneros musicales,
incluyendo cla´sica, New Age, electr´onica, rock, pop, world, jazz, blues, metal y punk.
Cada clip de audio est´a acompan˜ado de un vector de anotaciones binarias corres-
pondientes a 188 etiquetas, las cuales fueron generadas a partir de las contribuciones
de los jugadores del juego TagATune. Estas etiquetas pueden incluir informacio´n so-
bre g´eneros, instrumentos, estado de ´animo y otras caracter´ısticas relevantes para la
identificacio´n y clasificacio´n musical.
Aunque intuitivamente esperar´ıamos que todos los fragmentos de audio perte-
necientes a una misma canci´on compartieran las mismas etiquetas de g´enero, en este
dataset no es as´ı. Debido a que las clasificaciones fueron realizadas por diferentes per-
sonas en distintos momentos, se observan inconsistencias en las etiquetas asignadas a
los fragmentos.
Los archivos de audio est´an en formato MP3, codificados a 32kbps y 16kHz, lo
queresultaenuntaman˜ototalaproximadode3GBparaelconjuntocompletodedatos.
1Esta base de datos es accesible desde https://mirg.city.ac.uk/datasets/magnatagatune/
24

| Estructura |     | y   | Organizacio´n |     | del Dataset |     |
| ---------- | --- | --- | ------------- | --- | ----------- | --- |
Los datos en MagnaTagATuneesta´n organizados en archivos de audio y archivos
con distintas anotaciones. Los archivos de audio est´an almacenados en formato MP3,
mientras que las etiquetas se presentan en un archivo separado en formato CSV, don-
de cada fila corresponde a un clip de audio y contiene su identificador junto con las
| etiquetas |     | binarias | asignadas. |     |     |     |
| --------- | --- | -------- | ---------- | --- | --- | --- |
Los archivos de audio esta´n organizados en distintas carpetas con el propo´sito
de facilitar, en etapas posteriores, la divisio´n del dataset para el entrenamiento y la
evaluaci´on de algoritmos, usando algunas carpetas como conjunto de entrenamiento
(train)yotrascomoconjuntodeprueba(test).Sinembargo,ennuestrocaso,algenerar
espectrogramas a partir de estos audios, cada carpeta contendr´a una gran cantidad de
datos derivados, lo que nos obligara´ a realizar una nueva divisio´n de la informacio´n
para asegurar una correcta separacio´n entre los conjuntos de entrenamiento y prueba.
| Motivacio´n |     | para | la  | Seleccio´n | del Dataset |     |
| ----------- | --- | ---- | --- | ---------- | ----------- | --- |
El dataset MagnaTagATune ha sido elegido para este trabajo debido a:
Variedad de anotaciones: La multitud de etiquetas detalladas (concretamente
188), proporcionan una base s´olida para realizar tareas de clasificaci´on, espec´ıfi-
|     | camente | de  | g´eneros | musicales. |     |     |
| --- | ------- | --- | -------- | ---------- | --- | --- |
Variedad de g´eneros: La presencia de gran variedad g´eneros permite evaluar
la capacidad del modelo para diferenciar entre distintas categor´ıas musicales.
Uso extendido en la comunidad cient´ıfica: Este dataset ha sido empleado
en numerosos estudios previos, lo que facilita la comparaci´on de resultados con
|     | investigaciones |     | existentes. |     |     |     |
| --- | --------------- | --- | ----------- | --- | --- | --- |
Taman˜o y accesibilidad: A pesar de su riqueza en datos, el conjunto de datos
es manejable en t´erminos de almacenamiento y procesamiento, permitiendo su
uso en un entorno de desarrollo sin requerimientos computacionales excesivos.
| 4.3.3. |     | Gestio´n |     | del almacenamiento |     | de los espectrogramas |
| ------ | --- | -------- | --- | ------------------ | --- | --------------------- |
Una de las principales dificultades de este proyecto es que plantea un desaf´ıo en
t´erminos de almacenamiento y acceso eficiente a los datos, ya que existe un elevado
nu´mero de espectrogramas generados (en torno a 300,000) que deben ser de r´apido
acceso. Para abordar este problema, he evaluado diversas opciones, teniendo en cuenta
factores como facilidad de acceso, compatibilidad y eficiencia en la carga de datos
| durante | el  | entrenamiento |              | del | modelo.   |     |
| ------- | --- | ------------- | ------------ | --- | --------- | --- |
|         | Las | opciones      | consideradas |     | han sido: |     |
Almacenamiento en plataformas en la nube (Google Drive, Dropbox,
OneDrive y Google Cloud Storage). Estas opciones ofrecen accesibilidad
remota y escalabilidad, pero pueden introducir latencias en el acceso a los datos
25

y requieren una conexi´on a internet constante, lo que puede ralentizar la consulta
| de  | los espectrogramas. |     |     |     |
| --- | ------------------- | --- | --- | --- |
Almacenamiento en formatos comprimidos (.hdf5, .npz). Estos formatos
permiten almacenar mu´ltiples espectrogramas en un u´nico archivo, facilitando el
acceso y reduciendo el taman˜o de almacenamiento, pero son lentos a la hora de
| recuperar | datos | espec´ıficos. |     |     |
| --------- | ----- | ------------- | --- | --- |
Bases de datos locales (SQLite). Esta opcio´n permite almacenar los espectro-
gramas de manera estructurada y eficiente, facilitando la indexaci´on y el acceso
| a   | los datos sin | necesidad | de depender | de la nube. |
| --- | ------------- | --------- | ----------- | ----------- |
Reduccio´n de la cantidad de datos.Unaalternativaesseleccionarunsubcon-
junto representativo del conjunto de datos original, aunque esto podr´ıa reducir
la capacidad del modelo para generalizar sobre una amplia variedad de g´eneros.
Tras evaluar los pros y los contras de cada opci´on, he decidido utilizar SQLite
como base de datos local para almacenar los espectrogramas. Esta eleccio´n se basa en
| varios factores | clave: |     |     |     |
| --------------- | ------ | --- | --- | --- |
Permite acceder a los datos de manera estructurada y eficiente sin necesidad de
| conexio´n | a internet. |     |     |     |
| --------- | ----------- | --- | --- | --- |
Optimiza el almacenamiento, ya que los espectrogramas pueden guardarse en
| formato | binario | dentro | de la base | de datos. |
| ------- | ------- | ------ | ---------- | --------- |
Facilita la consulta y recuperaci´on de datos espec´ıficos durante el entrenamiento
| del | modelo. |     |     |     |
| --- | ------- | --- | --- | --- |
Es compatible con Python y las bibliotecas que planeo utilizar en el proyecto
| como   | TensorFlow      | y PyTorch. |     |     |
| ------ | --------------- | ---------- | --- | --- |
| 4.3.4. | Implementacio´n |            |     |     |
La implementacio´n se desarrollara´ con Python, utilizando Visual Studio Code co-
mo entorno principal. El preprocesamiento de audio se realizara´ con pydub para la
conversi´on y recorte, y librosa para la extraccio´n de caracter´ısticas y generaci´on de
mel-espectrogramas. El modelo de clasificaci´on se construir´a y entrenar´a con Tensor-
Flow/Keras. El control de versiones y la gesti´on del c´odigo fuente se realizar´an con
Git, mientras que la planificaci´on y seguimiento de tareas se gestionara´n con Trello
y Clockify. La integracio´n de todas estas herramientas permitira´ un flujo de trabajo
| eficiente, | reproducible | y colaborativo. |            |     |
| ---------- | ------------ | --------------- | ---------- | --- |
| 4.3.5.     | Elecci´on    | de              | algoritmos |     |
Para la tarea de clasificaci´on autom´atica de g´eneros musicales, se utilizar´an redes
neuronales convolucionales (CNN 2.6), dada su eficacia en el procesamiento de datos
visuales como los espectrogramas. Se priorizar´a la facilidad de uso y la capacidad del
modelo para aprender patrones temporales complejos. El preprocesamiento incluira´ la
26

fragmentacio´n de los audios en segmentos de duracio´n fija y el uso de ventanas desli-
zantes, con el objetivo de incrementar el nu´mero de muestras y mejorar la capacidad
| de generalizacio´n |               | del | modelo. |               |     |     |
| ------------------ | ------------- | --- | ------- | ------------- | --- | --- |
| 4.3.6.             | Entrenamiento |     |         | y evaluaci´on |     |     |
En el entrenamiento del modelo se utilizar´an mel-espectrogramas generados a
partir de los archivos de audio preprocesados. Para asegurar la robustez de los resulta-
dos y evitar la dependencia de una u´nica partici´on de los datos, se emplear´a validaci´on
cruzada con k-folds (K-Fold Cross Validation). Durante el entrenamiento, se aplicara´
la t´ecnica de early stopping para evitar el sobreajuste. Las m´etricas utilizadas para
evaluar el rendimiento incluira´n accuracy, precision, recall, AUC y F1-score, proporcio-
nando as´ı una visio´n integral del comportamiento del modelo, especialmente relevante
| en tareas | multilabel. |     |     |     |     |     |
| --------- | ----------- | --- | --- | --- | --- | --- |
4.4. Planificaci´on
El desarrollo de este Trabajo Fin de Ma´ster se ha llevado a cabo siguiendo un
ciclo a´gil iterativo, en el cual la programacio´n y la redacci´on del documento se han
realizado de forma paralela. De esta forma he podido ir mejorando progresivamente
tantolaimplementacio´nt´ecnicacomolate´orica,yaquelosavancesenelc´odigoinflu´ıan
directamenteenlanecesidaddeadaptaryampliarseccionesdeldocumento,yviceversa.
| 4.4.1. | Metodolog´ıa |     |     | de Trabajo |     |     |
| ------ | ------------ | --- | --- | ---------- | --- | --- |
Al tratarse de un proyecto individual, se ha optado por un enfoque de desarrollo
a´gileiterativo,adaptadoalasnecesidadesycaracter´ısticasdeltrabajo.Acontinuacio´n,
| se describen | las | principales | decisiones |     | organizativas | adoptadas: |
| ------------ | --- | ----------- | ---------- | --- | ------------- | ---------- |
Planificacio´n: Se ha organizado el trabajo en iteraciones de aproximadamente
dos semanas, lo que ha facilitado una revisio´n perio´dica del progreso.
Revisiones perio´dicas: Cada dos semanas se han llevado a cabo reuniones de
seguimiento con mi tutor, donde se ha evaluado el estado del proyecto y se han
| definido |     | los siguientes |     | objetivos. |     |     |
| -------- | --- | -------------- | --- | ---------- | --- | --- |
Seguimiento diario: Al no contar con un equipo, no se han realizado reuniones
diarias. No obstante, cada d´ıa se ha llevado a cabo una revisio´n personal del
| trabajo | pendiente |     | y los | objetivos | a corto plazo. |     |
| ------- | --------- | --- | ----- | --------- | -------------- | --- |
Gestio´n de tareas: Se ha utilizado Trello como herramienta principal para la
planificacio´n y el control del progreso. En ella se ha documentado el estado de las
tareas, bloqueos detectados y soluciones aplicadas, especialmente en lo referente
alentornodedesarrolloconelfindeagilizaralresolucio´nsisevolvieraaproducir.
27

Registro del tiempo: Se ha empleado Clockify para contabilizar las horas de-
dicadas a cada fase del proyecto, lo cual ha permitido analizar la distribucio´n del
|        | tiempo | y              | ajustar | la  | planificacio´n | segu´n |          | las necesidades |     | reales. |     |
| ------ | ------ | -------------- | ------- | --- | -------------- | ------ | -------- | --------------- | --- | ------- | --- |
| 4.4.2. |        | Planificacio´n |         |     | y fases        | del    | proyecto |                 |     | inicial |     |
A continuacio´n se presenta la planificaci´on inicial para este TFM:
| Fase       |     |     |             |     |     |     |     | Periodo |       | Horas | estimadas |
| ---------- | --- | --- | ----------- | --- | --- | --- | --- | ------- | ----- | ----- | --------- |
| Formaci´on |     | y   | definici´on |     |     |     | dic | 2024    | – ene | 2025  | 40 h      |
Estado del arte y revisi´on bibliogra´fica dic 2024 – ene 2025 40 h
| Disen˜o        |     | y planificacio´n |             | t´ecnica      |               |          |     | ene | 2025      |          | 25 h  |
| -------------- | --- | ---------------- | ----------- | ------------- | ------------- | -------- | --- | --- | --------- | -------- | ----- |
| Entorno        |     | y an´alisis      |             | de datos      |               |          | ene | –   | feb 2025  |          | 35 h  |
| Espectrogramas |     |                  | y           | entrenamiento |               |          | feb | –   | mar 2025  |          | 55 h  |
| Evaluaci´on    |     | y                | validacio´n |               |               |          | mar | –   | abr 2025  |          | 35 h  |
| Redaccio´n     |     | de               | la memoria  |               |               |          | ene | –   | abr 2025  |          | 50 h  |
| Revisio´n      |     | final            | y defensa   |               |               |          | abr | –   | may 2025  |          | 20 h  |
| Total          |     |                  |             |               |               |          |     |     |           |          | 300 h |
|                |     | Cuadro           |             | 4.1:          | Distribuci´on | estimada |     | de  | horas por | fase del | TFM   |
1. Fase de formaci´on y definici´on: Durante las primeras semanas del proyecto,
se dedicar´a tiempo a la adquisici´on de conocimientos clave sobre redes neuronales
convolucionales (CNN), procesamiento de audio y generacio´n de espectrogramas.
Paralelamente, se definir´a de forma precisa el problema a abordar, los objetivos
|     | del | TFM | y los | criterios | de ´exito | del | modelo. |     |     |     |     |
| --- | --- | --- | ----- | --------- | --------- | --- | ------- | --- | --- | --- | --- |
2. Estado del arte y revisi´on bibliogr´afica: Se realizara´ una revisio´n exhaustiva
de la literatura existente, tanto en el a´mbito del reconocimiento musical como en
t´ecnicas aplicadas con deep learning. Este estudio permitira´ identificar datasets
|     | comu´nmente |     | utilizados |     | y posibles |     | ´areas | de mejora. |     |     |     |
| --- | ----------- | --- | ---------- | --- | ---------- | --- | ------ | ---------- | --- | --- | --- |
3. Disen˜o y planificacio´n t´ecnica: Una vez asentadas las bases teo´ricas, se defi-
nira´ el entorno tecnol´ogico, las herramientas de desarrollo, los recursos de compu-
tacio´n y la estructura general del sistema a implementar. Tambi´en se planificar´a
|     | la  | arquitectura |     | del | modelo. |     |     |     |     |     |     |
| --- | --- | ------------ | --- | --- | ------- | --- | --- | --- | --- | --- | --- |
4. Preparacio´n del entorno y an´alisis de datos: Se procedera´ a la instalaci´on y
configuracio´n de las librer´ıas necesarias y se realizara´ un ana´lisis exploratorio del
dataset elegido. Se definir´an las variables relevantes y se estudiara´ la viabilidad
|     | del | dataset | respecto |     | a los objetivos |     | del | proyecto. |     |     |     |
| --- | --- | ------- | -------- | --- | --------------- | --- | --- | --------- | --- | --- | --- |
5. Generacio´n de espectrogramas, desarrollo y entrenamiento de modelos:
A partir de los archivos de audio del dataset, se generar´an espectrogramas. Con
|     | los | datos | ya procesados, |     | se entrenar´a |     | el  | modelo | de  | clasificacio´n. |     |
| --- | --- | ----- | -------------- | --- | ------------- | --- | --- | ------ | --- | --------------- | --- |
6. Evaluacio´n y validaci´on del sistema: Una vez obtenidos los modelos entrena-
dos, se evaluara´ su rendimiento utilizando m´etricas adecuadas como la precisi´on,
28

|     | recall | o F1-score. |     |     |     |     |     |     |     |
| --- | ------ | ----------- | --- | --- | --- | --- | --- | --- | --- |
7. Redaccio´n y elaboracio´n de la memoria: En paralelo al desarrollo t´ecnico,
se redactara´ de forma progresiva la memoria del TFM. Esta incluira´ tanto los
fundamentos teo´ricos como las decisiones de disen˜o, los resultados obtenidos y
|     | las | conclusiones | derivadas |     | del | proyecto. |     |     |     |
| --- | --- | ------------ | --------- | --- | --- | --------- | --- | --- | --- |
8. Revisi´on final y defensa: Finalmente, se revisara´ la documentacio´n, se prepa-
rara´ la presentaci´on para la defensa y se realizara´n los u´ltimos ajustes en base a
|        | las              | recomendaciones |                 | del            | tutor    | o revisores. |       |       |       |
| ------ | ---------------- | --------------- | --------------- | -------------- | -------- | ------------ | ----- | ----- | ----- |
|        | Si representamos |                 |                 | con un         | diagrama | de Gantt:    |       |       |       |
|        |                  |                 |                 |                |          | 2024         |       | 2025  |       |
|        |                  |                 |                 |                |          | 11 12        | 01 02 | 03 04 | 05 06 |
|        |                  | Formaci´on      |                 | y definici´on  |          |              |       |       |       |
| Estado |                  | del arte        | y rev.          | bibliogr´afica |          |              |       |       |       |
|        | Disen˜o          | y               | planificacio´n  |                | t´ecnica |              |       |       |       |
|        |                  | Entorno         | y an´alisis     | de             | datos    |              |       |       |       |
|        | Espectrogramas   |                 | y entrenamiento |                |          |              |       |       |       |
|        |                  | Evaluaci´on     |                 | y validacio´n  |          |              |       |       |       |
|        |                  | Redaccio´n      | de              | la memoria     |          |              |       |       |       |
|        |                  | Revisio´n       | final           | y defensa      |          |              |       |       |       |
|        |                  | Entrega         | y               | defensa        | final    |              |       |       |       |
| 4.4.3. |                  | Planificacio´n  |                 |                | y fases  | del proyecto | real  |       |       |
El desarrollo del TFM ha seguido una estrategia iterativa y ´agil, donde la pro-
gramacio´n y la elaboracio´n de la memoria han evolucionado de forma conjunta. A
continuaci´on, se describen las fases clave del proyecto, no como bloques r´ıgidos, sino
como procesos entrelazados que se han ido mejorando y modificando de manera conti-
nua:
29

| Fase       |        |     |               |     |     |          | Periodo |          | Horas | estimadas |
| ---------- | ------ | --- | ------------- | --- | --- | -------- | ------- | -------- | ----- | --------- |
| Formaci´on | previa |     | y definicio´n |     |     | dic 2024 | –       | ene 2025 |       | 18,2 h    |
Estado del arte y revisi´on bibliogra´fica dic 2024 – ene 2025 45,7 h
| Disen˜o        | y planificacio´n |             | t´ecnica           |     |          |     | ene      | 2025 |          | 12,4 h  |
| -------------- | ---------------- | ----------- | ------------------ | --- | -------- | --- | -------- | ---- | -------- | ------- |
| Entorno        | y an´alisis      |             | de datos           |     |          | ene | – feb    | 2025 |          | 32,8 h  |
| Espectrogramas |                  | y           | entrenamiento      |     |          | feb | – mar    | 2025 |          | 57,1 h  |
| Evaluaci´on    | y                | validacio´n |                    |     |          | mar | – may    | 2025 |          | 38,4 h  |
| Redaccio´n     | de               | la memoria  |                    |     |          | ene | – jun    | 2025 |          | 83,5 h  |
| Revisio´n      | final            | y defensa   |                    |     |          | may | – jun    | 2025 |          | 18,2 h  |
| Total          |                  |             |                    |     |          |     |          |      |          | 306,7 h |
|                | Cuadro           |             | 4.2: Distribuci´on |     | estimada |     | de horas | por  | fase del | TFM     |
1. Investigacio´n inicial: Durante las primeras semanas, dediqu´e tiempo a com-
prender los conceptos fundamentales necesarios para abordar el trabajo, como las
redes neuronales convolucionales (CNN) y el uso de espectrogramas para repre-
sentar sen˜ales de audio. Esta etapa incluyo´ la visualizacio´n de v´ıdeos formativos,
lecturas de documentaci´on t´ecnica y pruebas en el entorno de desarrollo.
2. Documentacio´n para el estado del arte y fundamentos te´oricos: La re-
daccio´n del estado del arte y los fundamentos teo´ricos se desarrollaron de for-
ma paralela a la etapa anterior. A medida que analizaba trabajos relacionados,
surg´ıa la necesidad de incorporar explicaciones m´as detalladas de ciertos concep-
tos t´ecnicos. Este enfoque me permiti´o construir un marco teo´rico coherente y
bien fundamentado, ajustado a los requerimientos pr´acticos del proyecto.
3. Preparacio´n del entorno y an´alisis del dataset: En esta fase configur´e el en-
torno tecnolo´gico necesario (Python, librer´ıas de deep learning, entorno de GPU,
etc.) y realic´e un an´alisis exploratorio del dataset MagnaTagATune. Este ana´lisis
fue clave para entender la distribucio´n de los g´eneros musicales, la organizacio´n
de los datos y los posibles enfoques de clasificacio´n. Al mismo tiempo, defin´ı las
estrategias de validaci´on y segmentacio´n, identificando la posibilidad de aplicar
| validaci´on |     | cruzada. |     |     |     |     |     |     |     |     |
| ----------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
4. Procesamiento de datos y generaci´on de espectrogramas:Posteriormente,
se implementaron las herramientas necesarias para convertir los clips de audio en
espectrogramas. Este proceso gener´o una gran cantidad de datos, lo que supuso
un reto a nivel de almacenamiento. Tras evaluar distintas opciones, decid´ı utilizar
| una | base | de datos | local | (SQLite). |     |     |     |     |     |     |
| --- | ---- | -------- | ----- | --------- | --- | --- | --- | --- | --- | --- |
5. Entrenamiento de modelos y experimentaci´on: Una vez generados los
datos, comenc´e a entrenar distintos modelos de clasificacio´n, ajustando hiper-
para´metros y probando distintas configuraciones. Esta fase fue muy iterativa, con
numerosos ciclos de prueba y error, evaluacio´n de resultados y reentrenamiento.
6. Evaluaci´on de resultados y validacio´n: Con los modelos entrenados, se pro-
cedio´ a su evaluaci´on mediante m´etricas est´andar. Se analizaron los resultados
obtenidos, se identificaron posibles sesgos y se revis´o la capacidad del modelo
| para | generalizar |     | frente | a nuevos | datos. |     |     |     |     |     |
| ---- | ----------- | --- | ------ | -------- | ------ | --- | --- | --- | --- | --- |
30

7. Conclusiones y cierre del proyecto: Finalmente, se extrajeron las conclusio-
nes ma´s relevantes del trabajo, reflexionando sobre las limitaciones del enfoque
propuesto y sen˜alando posibles l´ıneas de mejora y ampliaci´on del estudio.
A continuacio´n, se presenta un diagrama de Gantt que ilustra visualmente la
planificacio´n:
|               |                |           |     | 2024 |     |     | 2025 |     |     |     |
| ------------- | -------------- | --------- | --- | ---- | --- | --- | ---- | --- | --- | --- |
|               |                |           | 11  | 12   | 01  | 02  | 03   | 04  | 05  | 06  |
|               | Investigaci´on | inicial   |     |      |     |     |      |     |     |     |
| Estado        | del arte       | y teor´ıa |     |      |     |     |      |     |     |     |
|               | Entorno        | y dataset |     |      |     |     |      |     |     |     |
| Procesamiento |                | de datos  |     |      |     |     |      |     |     |     |
| Entrenamiento | y              | pruebas   |     |      |     |     |      |     |     |     |
| Evaluaci´on   | y validacio´n  |           |     |      |     |     |      |     |     |     |
|               | Conclusiones   | y cierre  |     |      |     |     |      |     |     |     |
|               | Entrega        | del TFM   |     |      |     |     |      |     |     |     |
4.5. Presupuesto
(€)
| Concepto |            |     | Detalle |     |             |     |     | Coste | estimado |          |
| -------- | ---------- | --- | ------- | --- | ----------- | --- | --- | ----- | -------- | -------- |
|          |            |     |         | ×   | 20€/h       |     |     |       |          |          |
| Horas    | de trabajo |     | 306,7h  |     |             |     |     |       |          | 6.134,00 |
|          |            |     |         |     | × 0,13€/kWh |     | ×   |       |          |          |
306,7h
| Consumo | el´ectrico |     |     |     |     |     |     |     |     | 5,98 |
| ------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
0,15kW
|               |             |                  | Ordenador  |              | de mesa   | gama          |     |     |     |          |
| ------------- | ----------- | ---------------- | ---------- | ------------ | --------- | ------------- | --- | --- | --- | -------- |
| Amortizacio´n | de hardware |                  |            |              |           |               |     |     |     | 400,00   |
|               |             |                  | media-alta |              | (3 an˜os) |               |     |     |     |          |
| Suscripcio´n  | a GitHub    | Pro              | 12€/mes    |              | × 6 meses |               |     |     |     | 72,00    |
|               |             |                  | Parte      | proporcional |           | de 6 meses    |     |     |     |          |
| Conexio´n     | a Internet  |                  |            |              | (50€/mes) |               |     |     |     | 150,00   |
|               |             |                  | de tarifa  |              |           |               |     |     |     |          |
| Total         | estimado    |                  |            |              |           |               |     |     |     | 6.761,98 |
|               | Cuadro      | 4.3: Presupuesto |            | estimado     | para      | el desarrollo |     | del | TFM |          |
Para la elaboracio´n de este presupuesto se han tenido en cuenta diversos factores
que representan costes directos e indirectos asociados. El precio de las horas de trabajo
31

se ha obtenido a trav´es de los sueldos medios registrados para ingenieros de software
| en Espan˜a | por Talent[43] | y Glassdoor[15]2. |     |     |
| ---------- | -------------- | ----------------- | --- | --- |
En cuanto al consumo el´ectrico, se ha registrado el consumo aproximado del equi-
po durante las 306,7 horas de trabajo, con un consumo estimado de 0,1 kW (ordenador
de sobremesa de gama media-alta) y una tarifa media de 0,13 €/kWh, resultando en
| un coste | muy reducido. |     |     |     |
| -------- | ------------- | --- | --- | --- |
La amortizacio´n del hardware se calcula en funcio´n de su coste total dividido
entre el periodo total de amortizacio´n (en meses), y luego se multiplica por la duracio´n
del proyecto:
(cid:18) 1,200€(cid:19)
|     | Coste imputado | =   | ×12 = 33,33€×12 | = 400€ |
| --- | -------------- | --- | --------------- | ------ |
36
Adema´s, en el presupuesto se han incluido gastos derivados de herramientas y servicios
como GitHub Pro (12€/mes). Por u´ltimo, se ha incluido la proporcio´n correspondiente
| de la tarifa | mensual | de conexi´on | a Internet. |     |
| ------------ | ------- | ------------ | ----------- | --- |
2Talent y Glassdoor son plataformas de bu´squeda de empleo que publican estad´ısticas salariales
| basadas | en datos de usuarios | y ofertas | activas. |     |
| ------- | -------------------- | --------- | -------- | --- |
32

5. Implementaci´on
5.1. Introducci´on
En este cap´ıtulo explicaremos el proceso de investigaci´on interna del dataset y la
| posterior      | implementacio´n |              | de nuestras |     | redes | convolucionales. |           |     |
| -------------- | --------------- | ------------ | ----------- | --- | ----- | ---------------- | --------- | --- |
| 5.2. An´alisis |                 | exploratorio |             |     |       | de               | los datos |     |
Con el objetivo de identificar posibles relaciones entre los g´eneros musicales pre-
sentes en el conjunto de datos, se ha llevado a cabo un an´alisis de correlaci´on. Esta
t´ecnica estad´ıstica nos permite cuantificar la intensidad y direccio´n de la relacio´n entre
| dos variables, | en  | este caso, | distintos |     | g´eneros | musicales. |     |     |
| -------------- | --- | ---------- | --------- | --- | -------- | ---------- | --- | --- |
Se parte de la premisa de que algunos g´eneros musicales podr´ıan presentar carac-
ter´ısticas comunes, lo que dar´ıa lugar a una correlaci´on positiva entre ellos. A conti-
nuacio´n, se presentan dos tablas: la primera tabla 5.1 representa las correlaciones m´as
altas encontradas entre g´eneros, mientras que la segunda 5.2 muestra a aquellos cuya
correlacio´n es pr´acticamente nula, lo cual significa que entre ellos no existe ninguna
| relacio´n significativa. |        |            |               |          |      |       |                |           |
| ------------------------ | ------ | ---------- | ------------- | -------- | ---- | ----- | -------------- | --------- |
|                          |        | Genero     | 1             | Genero   |      | 2     | Correlacion    |           |
|                          |        | hard       | rock          | metal    |      |       | 0.534015       |           |
|                          |        | hip        | hop           | rap      |      |       | 0.525677       |           |
|                          |        | electronic |               | techno   |      |       | 0.523670       |           |
|                          |        | jazz       |               | jazzy    |      |       | 0.517258       |           |
|                          |        | heavy      | metal         | metal    |      |       | 0.413903       |           |
|                          |        | indian     |               | eastern  |      |       | 0.407724       |           |
|                          |        | hard       | rock          | heavy    |      | metal | 0.387497       |           |
|                          | Cuadro | 5.1:       | Correlaciones |          | ma´s | altas | entre g´eneros | musicales |
|                          |        | Genero     |               | 1 Genero |      | 2     | Correlacion    |           |
|                          |        | orchestra  |               | dark     |      |       | -0.000228      |           |
|                          |        | pop        |               | strange  |      |       | -0.000221      |           |
|                          |        | baroque    |               | female   |      | opera | 0.000152       |           |
|                          |        | jazz       |               | electro  |      |       | -0.000102      |           |
|                          |        | blues      |               | pop      |      |       | 0.000088       |           |
|                          |        | orchestra  |               | medieval |      |       | 0.000004       |           |
|                          |        | oriental   |               | strange  |      |       | -0.000002      |           |
|                          | Cuadro | 5.2:       | Correlaciones |          | ma´s | bajas | entre g´eneros | musicales |
33

Este ana´lisis preliminar nos ha ayudado a ver la existencia de ciertas similitudes
entreg´enerosy,porotrolado,tambi´enconfirmaquemuchosestilosmusicalespresentan
| una  | independencia | casi | total | entre s´ı.     |     |             |           |
| ---- | ------------- | ---- | ----- | -------------- | --- | ----------- | --------- |
| 5.3. | Criterios     |      | de    | clasificaci´on |     | de g´eneros | musicales |
Unavezsehaanalizadolacorrelacio´nentrelasetiquetas,planteamosreorganizar-
lasconelfindeagruparaquellasquepresentansimilitudesclaras.Estareestructuracio´n
permitir´ıa reducir la redundancia entre categor´ıas y facilitar´ıa que el modelo generalice
mejor durante el entrenamiento. En definitiva, esta nueva organizacio´n de etiquetas
contribuira´ a mejorar tanto la coherencia del conjunto de datos como la precisio´n en
la clasificaci´on.
Se han establecido cinco grandes grupos que responden a criterios de afinidad o
| contexto | cultural | comunes |     | en la categorizacio´n |     | musical: |     |
| -------- | -------- | ------- | --- | --------------------- | --- | -------- | --- |
1. G´eneros con una fuerte identidad propia: estos g´eneros esta´n claramen-
te diferenciados y sus caracter´ısticas son lo suficientemente distintivas como para ser
tratadas como clases individuales. Estos g´eneros son: Punk, Blues, Pop, Country
y Reggae.
2. G´eneros derivados de una misma familia musical: Dentro de este gru-
po agrupamos los subg´eneros que comparten rasgos comunes y poseen o no or´ıgenes
similares:
|     | Rock: hard  | rock,   | soft   | rock           |        |            |     |
| --- | ----------- | ------- | ------ | -------------- | ------ | ---------- | --- |
|     | Metal:      | heavy   | metal, | metal          |        |            |     |
|     | Jazz: jazz, | jazzy   |        |                |        |            |     |
|     | Electronic: | techno, |        | trance, house, | disco, | industrial |     |
|     | Funk: funk, | funky   |        |                |        |            |     |
|     | Hip Hop:    | hip     | hop,   | rap            |        |            |     |
3. Mu´sica basada en instrumentos acu´sticos o tradiciones: G´eneros que
| utilizan | instrumentos    |     | acu´sticos | y estructuras |     | tradicionales: |     |
| -------- | --------------- | --- | ---------- | ------------- | --- | -------------- | --- |
|          | Folk acu´stico: |     | folk,      | celtic, irish |     |                |     |
|          | Folk antiguo:   |     | medieval,  | tribal        |     |                |     |
4. Mu´sica tradicional agrupada por regi´on geogr´afica: Agrupaciones de
g´eneros musicales segu´n su origen cultural y caracter´ısticas propias:
|     | Middle  | Eastern: | middle | eastern, | arabic |     |     |
| --- | ------- | -------- | ------ | -------- | ------ | --- | --- |
|     | Indian: | indian,  | india  |          |        |     |     |
34

| Oriental: | eastern, | oriental |     |
| --------- | -------- | -------- | --- |
| Spanish:  | spanish  |          |     |
5. Mu´sica ambiental y cl´asica: G´eneros con un enfoque atmosf´erico o de
| tradicio´n | cla´sica:  |             |              |
| ---------- | ---------- | ----------- | ------------ |
| Classical: | classical, | opera       |              |
| Ambient:   | new        | age, space, | eerie, drone |
Esta nueva clasificacio´n ayudara´ al modelo a identificar patrones con mayor cla-
ridad y a mejorar la diferenciacio´n entre clases, al trabajar con categor´ıas ma´s consis-
| tentes y | menos redundantes. |     |     |
| -------- | ------------------ | --- | --- |
5.4. Visualizaci´on
Para comprender mejor los datos con los que vamos a trabajar, vamos a aplicar
diversas t´ecnicas de visualizacio´n dentro del an´alisis exploratorio. Esta etapa resulta
fundamental antes de abordar cualquier tipo de modelado, ya que permite detectar
| posibles | sesgos y desbalances. |     |     |
| -------- | --------------------- | --- | --- |
En primer lugar, la gra´fica 5.1 muestra la distribucio´n de canciones por g´enero.
Este gra´fico de barras nos representa de manera muy clara la desigualdad en la frecuen-
cia de aparicio´n entre los distintos g´eneros musicales. Adem´as, se observa una fuerte
presencia de g´eneros como classical, electronic y ambient, los cuales superan amplia-
mente en nu´mero al resto. Este desbalance es importante tenerlo en cuenta, ya que
puede influir negativamente en el entrenamiento del modelo, provocando que tienda a
favorecer las clases mayoritarias. En consecuencia, se considerara´ aplicar t´ecnicas de
| balanceo | o ponderacio´n | de clases | en fases posteriores. |
| -------- | -------------- | --------- | --------------------- |
35

Figura 5.1: Distribuci´on canciones por g´enero
La visualizaci´on tambi´en permite identificar qu´e g´eneros aparecen menos, como
puede ser reggae, punk o hip hop, que cuentan con un nu´mero significativamente menor
de ejemplos.
Por otro lado, en la imagen 5.2 podemos ver representada una matriz de corre-
lacio´n entre g´eneros. A trav´es del uso de un mapa de calor, esta imagen facilita la
deteccio´n de relaciones estad´ısticas entre pares de etiquetas. Como es esperable, los
valores m´as altos (coloreados en rojo) aparecen en la diagonal principal, ya que cada
g´enero est´a perfectamente correlacionado consigo mismo. No obstante, tambi´en pueden
apreciarse ciertas correlaciones positivas relevantes entre g´eneros distintos. Por ejem-
plo, rock y metal presentan un coeficiente de correlaci´on de 0.49, mientras que hip hop
y funk tambi´en muestran cierta proximidad.
36

Figura 5.2: Correlaci´on entre g´eneros
Estasobservacionescoincidenconlosresultadosnum´ericospresentadosenelapar-
tado anterior 5.2, donde se recogen las correlaciones m´as altas y ma´s bajas respecti-
vamente. Es interesante destacar que algunos g´eneros, como blues y pop o orchestra
y medieval, tienen una correlaci´on pra´cticamente nula, lo que refuerza la idea de que
muchos estilos musicales presentan patrones sonoros totalmente independientes.
La matriz de correlacio´n tambi´en valida la estrategia explicada en la seccio´n 5.3
para la reorganizaci´on de etiquetas.
37

|     |     | Figura 5.3: | Nu´mero | de g´eneros por | cancio´n |
| --- | --- | ----------- | ------- | --------------- | -------- |
Enlafigura5.3semuestraelnu´merodeg´enerosasignadosporcanci´on,loquenos
proporciona una visi´on ma´s precisa sobre la presencia de la multi-etiqueta en nuestro
dataset. Como se puede observar, la gran mayor´ıa de las canciones (ma´s de 10.000)
esta´n etiquetadas con un solo g´enero, lo que sugiere una fuerte tendencia hacia una
| clasificacio´n | unitaria | en el dataset | original. |     |     |
| -------------- | -------- | ------------- | --------- | --- | --- |
El hecho de que existan canciones con mu´ltiples g´eneros refuerza la necesidad
de abordar este problema como una tarea de clasificaci´on multi-etiqueta, en lugar de
una clasificaci´on tradicional. Esta distinci´on es clave, ya que afecta directamente a la
eleccio´n de la arquitectura de red, la funcio´n de p´erdida y la m´etrica de evaluaci´on. Por
lo tanto, esta visualizacio´n resulta especialmente u´til para justificar ciertas decisiones
| de disen˜o       | en la implementaci´on. |     |                |     |     |
| ---------------- | ---------------------- | --- | -------------- | --- | --- |
| 5.5. Generaci´on |                        | de  | espectrogramas |     |     |
Para que las redes convolucionales puedan trabajar con los audios, es necesario
convertir las sen˜ales sonoras en representaciones visuales que capturen sus caracter´ısti-
cas espectrales y temporales. Para ello, se opta por el uso de espectrogramas, que
permiten representar la intensidad de distintas frecuencias a lo largo del tiempo.
Existen distintas estrategias para generar estos espectrogramas a partir de los
audios del dataset, cada una con sus ventajas y desventajas. A continuacio´n se analizan
| las cuatro | principales: |     |     |     |     |
| ---------- | ------------ | --- | --- | --- | --- |
38

| Opci´on  | 1: Cortar        | el audio | en  | fragmentos  |     | de  | 3 segundos | y ge- |
| -------- | ---------------- | -------- | --- | ----------- | --- | --- | ---------- | ----- |
| nerar    | un espectrograma |          | por | fragmento   |     |     |            |       |
| Ventajas |                  |          |     | Desventajas |     |     |            |       |
M´etodo directo y f´acil de implementar. Sepierdecontinuidadentrefragmentos,lo
|     |     |     |     | que            | puede | afectar     | a la calidad | de las ca- |
| --- | --- | --- | --- | -------------- | ----- | ----------- | ------------ | ---------- |
|     |     |     |     | racter´ısticas |       | extra´ıdas. |              |            |
Cada espectrograma se corresponde con Informaci´onrelevantepuedequedarenlos
un fragmento de audio claramente defini- bordes de los cortes y no estar bien repre-
| do. |     |     |     | sentada. |     |     |     |     |
| --- | --- | --- | --- | -------- | --- | --- | --- | --- |
Bajocostecomputacional:solosegeneran Menor nu´mero de muestras por audio, lo
10 fragmentos por audio. que limita la diversidad del conjunto de
entrenamiento.
|     | Cuadro | 5.3: Corte | directo | en fragmentos |     | de 3 | segundos |     |
| --- | ------ | ---------- | ------- | ------------- | --- | ---- | -------- | --- |
Opci´on 2: Generar el espectrograma completo del audio y cor-
tarlo posteriormente
| Ventajas |     |     |     | Desventajas |     |     |     |     |
| -------- | --- | --- | --- | ----------- | --- | --- | --- | --- |
Permite un tratamiento global del audio El espectrograma completo ocupa m´as
| antes | de segmentar. |     |     | memoria |     | y es costoso | de procesar. |     |
| ----- | ------------- | --- | --- | ------- | --- | ------------ | ------------ | --- |
Facilita tareas de normalizaci´on o proce- Cortar sobre la imagen puede generar
sado sobre la sen˜al entera. fragmentos que no est´en perfectamente
alineados.
|          | Cuadro 5.4: | Segmentacio´n |     | posterior   | al espectrograma |     | completo |     |
| -------- | ----------- | ------------- | --- | ----------- | ---------------- | --- | -------- | --- |
| Opci´on  | 3: Ventanas | deslizantes   |     | sin         | solapamiento     |     |          |     |
| Ventajas |             |               |     | Desventajas |                  |     |          |     |
Aumenta el nu´mero de muestras por au- Puedehaberp´erdidadeinformaci´onenlas
| dio. |     |     |     | transiciones |     | entre | ventanas. |     |
| ---- | --- | --- | --- | ------------ | --- | ----- | --------- | --- |
Permiteuncontrolclarosobreladuraci´on No garantiza una cobertura continua del
| de cada | fragmento.  |             |               | contenido,   |              | lo que    | afecta a g´eneros | con |
| ------- | ----------- | ----------- | ------------- | ------------ | ------------ | --------- | ----------------- | --- |
|         |             |             |               | transiciones |              | r´apidas. |                   |     |
|         |             | Cuadro      | 5.5: Ventanas | sin          | solapamiento |           |                   |     |
| Opci´on | 4: Ventanas | deslizantes |               | con          | solapamiento |           |                   |     |
Esta fue la opcio´n finalmente seleccionada, al ofrecer el mejor equilibrio entre
cobertura temporal, diversidad de muestras y robustez para el aprendizaje del modelo.
39

Se aplica una ventana de 3 segundos con un solapamiento de 1.5 segundos entre
cada fragmento consecutivo. Esto significa que cada nuevo espectrograma se genera
desplazando la ventana 1.5 segundos respecto al anterior, garantizando as´ı que cada
instante de tiempo aparece representado en mu´ltiples espectrogramas.
| Ventajas |     |     |     |     | Desventajas |     |
| -------- | --- | --- | --- | --- | ----------- | --- |
Aumentaconsiderablementeelnu´meroto- Mayor coste computacional en t´erminos
tal de muestras por audio (aprox. 19 frag- de procesamiento y almacenamiento.
| mentos | por archivo | de 30 | segundos). |     |     |     |
| ------ | ----------- | ----- | ---------- | --- | --- | --- |
Mejora la cobertura temporal: cada ins- Riesgo de sobreajuste si el modelo no ge-
tante del audio aparece en varias venta- neralizabienantemuestrasmuysimilares.
nas.
| Reduce         | la p´erdida              | de informaci´on |                | en los |     |     |
| -------------- | ------------------------ | --------------- | -------------- | ------ | --- | --- |
| bordes         | gracias al solapamiento. |                 |                |        |     |     |
| Aumenta        | la diversidad            |                 | del conjunto   | de     |     |     |
| entrenamiento, | mejorando                |                 | el rendimiento |        |     |     |
del modelo.
|     | Cuadro | 5.6: | Ventanas | con | solapamiento | de 1.5 segundos |
| --- | ------ | ---- | -------- | --- | ------------ | --------------- |
Como resultado de esta configuraci´on, se generan un total de 394849 espectro-
gramas, lo que proporciona un conjunto de entrenamiento amplio y con una represen-
| tacio´n detallada | del    | contenido | temporal |     | de los audios. |           |
| ----------------- | ------ | --------- | -------- | --- | -------------- | --------- |
|                   | Figura | 5.4:      | Ejemplos | de  | espectrogramas | generados |
En la Figura 5.4 se pueden observar dos ejemplos reales de espectrogramas gene-
rados con lo explicado anteriormente. Estas ima´genes reflejan la variacio´n de energ´ıa a
trav´es del tiempo en distintas bandas de frecuencia, lo cual sera´ aprovechado por las
redes convolucionales para extraer patrones representativos de cada g´enero musical.
40

5.6. Entrenamiento
Para comenzar con el apartado de entrenamiento, el primer paso ha sido redi-
mensionar los espectrogramas a una resolucio´n fija de 128x128 p´ıxeles. Esta decisio´n
se ha tomado con el fin de garantizar que todas las entradas tengan el mismo taman˜o.
Adema´s, al escoger untaman˜orelativamentepequen˜o, sereduce el costecomputacional
y el uso de memoria durante el entrenamiento, sin necesidad de perder una cantidad
| excesiva | de informacio´n | relevante.              |                       |
| -------- | --------------- | ----------------------- | --------------------- |
|          | Figura          | 5.5: Redimensionamiento | de los espectrogramas |
Tras el redimensionado, se ha aplicado una normalizacio´n de los valores de p´ıxeles
(dividiendo entre 255) para llevar los valores de los p´ıxeles, que originalmente est´an
en el rango [0, 255], al rango [0, 1]. Esta normalizaci´on es una pra´ctica habitual en
deep learning [26] porque permite que la red neuronal entrene de forma m´as estable
y eficiente, evitando problemas derivados de escalas num´ericas muy distintas entre
distintas variables y evitando tambi´en que las redes trabajen con nu´meros muy grandes
| dificulten | el aprendizaje. |     |     |
| ---------- | --------------- | --- | --- |
La figura 5.6 muestra la distribucio´n de los valores de los p´ıxeles antes (en color
morado) y despu´es (en color amarillo) del proceso de normalizacio´n. Como puede ob-
servarse, existe una alta concentracio´n en los valores ma´ximos (255 en la escala original
| y 1.0 en | la normalizada). |     |     |
| -------- | ---------------- | --- | --- |
41

Figura 5.6: Distribuci´on de valores de p´ıxeles en un espectrograma
Como se puede ver en la figura 5.7, la red utiliza varias capas convolucionales
Conv2D, que funcionan como filtros que recorren el espectrograma buscando patrones
importantes. Despu´es de cada una de estas capas, se ha incluido una capa MaxPoo-
ling2D, que se encarga de reducir el taman˜o de las representaciones internas mante-
niendo la informaci´on ma´s relevante. Esto permite que el modelo sea ma´s ra´pido y
consuma menos recursos, pero sin perder la informaci´on que ha aprendido.
42

Figura 5.7: Capas CNN
43

Tambi´en se han incluido capas de normalizacio´n por lotes BatchNormalization,
que ayudan a que el entrenamiento sea ma´s r´apido y estable, ya que ajustan automa´ti-
camente las activaciones de las neuronas. Adema´s, se ha an˜adido una capa Dropout
con una tasa del 50% para reducir el riesgo de sobreajuste ya que evita que la red
| memorice |     | los datos | de entrenamiento. |     |     |     |     |     |
| -------- | --- | --------- | ----------------- | --- | --- | --- | --- | --- |
Despu´esdetodasestastransformaciones,seutilizaunacapaDenseconactivacio´n
sigmoid. Esta capa es la encargada de producir la salida final del modelo, que en este
caso es de tipo multilabel, es decir, una misma muestra puede tener varias etiquetas
(en nuestro caso g´eneros) activas a la vez. Por esta razo´n, tambi´en se ha elegido como
funcio´n de p´erdida binary crossentropy, que se ajusta mejor a este tipo de problemas
donde puede haber varias clases verdaderas por cada entrada. Todos los fundamentos
teo´ricos sobre las neuronas utilizadas pueden consultarse en la seccio´n 2.
| def | create_model(input_shape, |     |     |     |     | num_classes): |     |     |
| --- | ------------------------- | --- | --- | --- | --- | ------------- | --- | --- |
1
|     | model | = Sequential() |     |     |     |     |     |     |
| --- | ----- | -------------- | --- | --- | --- | --- | --- | --- |
2
| 3   | model.add(Input(shape=input_shape)) |     |     |     |     |                     |     |     |
| --- | ----------------------------------- | --- | --- | --- | --- | ------------------- | --- | --- |
|     | model.add(Conv2D(32,                |     |     | (3, | 3), | activation=’relu’)) |     |     |
4
model.add(BatchNormalization())
5
|     | model.add(MaxPooling2D(2, |     |     |     |     | 2)) |     |     |
| --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- |
6
|     | model.add(Conv2D(64, |     |     | (3, | 3), | activation=’relu’)) |     |     |
| --- | -------------------- | --- | --- | --- | --- | ------------------- | --- | --- |
7
model.add(BatchNormalization())
8
|     | model.add(MaxPooling2D(2, |     |     |     |     | 2)) |     |     |
| --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- |
9
| 10  | model.add(Conv2D(128, |     |     |     | (3, | 3), activation=’relu’)) |     |     |
| --- | --------------------- | --- | --- | --- | --- | ----------------------- | --- | --- |
model.add(BatchNormalization())
11
|     | model.add(MaxPooling2D(2, |     |     |     |     | 2)) |     |     |
| --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- |
12
model.add(Flatten())
13
|     | model.add(Dense(256, |     |     | activation=’relu’)) |     |     |     |     |
| --- | -------------------- | --- | --- | ------------------- | --- | --- | --- | --- |
14
model.add(Dropout(0.5))
15
|     | model.add(Dense(num_classes, |     |     |     |     | activation=’sigmoid’)) |     |     |
| --- | ---------------------------- | --- | --- | --- | --- | ---------------------- | --- | --- |
16
17
model.compile(optimizer=’adam’,
18
loss=’binary_crossentropy’,
19
|     |     | metrics=[’accuracy’, |     |     | tf.keras.metrics.AUC(), |     |     | tf.keras. |
| --- | --- | -------------------- | --- | --- | ----------------------- | --- | --- | --------- |
20
|     |        | metrics.Precision(), |     |     |     | tf.keras.metrics.Recall()]) |     |     |
| --- | ------ | -------------------- | --- | --- | --- | --------------------------- | --- | --- |
|     | return | model                |     |     |     |                             |     |     |
21
|     |     |     | Extracto |     | de co´digo | 5.1: | Modelo CNN |     |
| --- | --- | --- | -------- | --- | ---------- | ---- | ---------- | --- |
Como se puede ver en el fragmento de co´digo 5.2, durante el entrenamiento del
modelo, se estableci´o un ma´ximo de 30 ´epocas; tambi´en se utiliz´o la t´ecnica de early
stopping la cual detiene automa´ticamente el proceso de entrenamiento cuando el ren-
dimiento en el conjunto de validacio´n deja de mejorar durante un nu´mero determinado
de ´epocas consecutivas. Esta estrategia es fundamental para prevenir el sobreajuste,
es decir, que el modelo aprenda demasiado bien los datos de entrenamiento a costa de
| disminuir  |     | su capacidad | de generalizacio´n |     |     | a nuevos | datos. |     |
| ---------- | --- | ------------ | ------------------ | --- | --- | -------- | ------ | --- |
| image_size |     | = (128,      | 128)               |     |     |          |        |     |
1
| 2 batch_size |     | = 64 |     |     |     |     |     |     |
| ------------ | --- | ---- | --- | --- | --- | --- | --- | --- |
44

| epochs = | 30  |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- |
3
| num_folds | = 5 |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- |
4
5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
6
7
| for train_index, |     | val_index | in kf.split(data): |     |     |     |
| ---------------- | --- | --------- | ------------------ | --- | --- | --- |
8
| x_train, | x_val | = data[train_index], |     |     | data[val_index] |     |
| -------- | ----- | -------------------- | --- | --- | --------------- | --- |
9
| y_train, | y_val | = labels[train_index], |     |     | labels[val_index] |     |
| -------- | ----- | ---------------------- | --- | --- | ----------------- | --- |
10
11
12 model = create_model(input_shape=input_shape, num_classes=
num_classes)
early_stopping = EarlyStopping(patience=5, restore_best_weights
13
=True)
14
| model.fit(x_train, |     | y_train, |     |     |     |     |
| ------------------ | --- | -------- | --- | --- | --- | --- |
15
|     |     | validation_data=(x_val, |     | y_val), |     |     |
| --- | --- | ----------------------- | --- | ------- | --- | --- |
16
| 17  |     | epochs=epochs, |     |     |     |     |
| --- | --- | -------------- | --- | --- | --- | --- |
batch_size=batch_size,
18
callbacks=[early_stopping],
19
verbose=0)
20
21
| scores | = model.evaluate(x_val, |     |     | y_val, | verbose=0) |     |
| ------ | ----------------------- | --- | --- | ------ | ---------- | --- |
22
|     | Extracto | de co´digo | 5.2: Entrenamiento |     | con early | stopping |
| --- | -------- | ---------- | ------------------ | --- | --------- | -------- |
45

6. Validacio´n
6.1. Introducci´on
En este cap´ıtulo explicaremos el procedimiento seguido para validar el modelo
entrenado, as´ı como las m´etricas utilizadas y su evoluci´on a lo largo del proceso de
entrenamiento.
6.2. Validacio´n del modelo
Con el objetivo de obtener resultados ma´s robustos y evitar que dependan de una
u´nicadivisi´ondelosdatos,seutiliz´olat´ecnicadevalidacio´ncruzadacon5folds(K-Fold
Cross Validation). Esta t´ecnica divide el conjunto de datos en cinco partes distintas
y realiza cinco entrenamientos diferentes, utilizando en cada uno cuatro partes para
entrenar y una para validar, como se muestra en la figura 6.1.
Figura 6.1: K-Fold Cross Validation
Para evaluar el rendimiento del modelo entrenado, se han utilizado varias m´etri-
cas. Inicialmente, se utiliz´o u´nicamente la m´etrica de precisi´on general accuracy pero
no es adecuada en problemas de clasificaci´on multilabel, ya que solo se considera co-
rrecta una predicci´on si coincide exactamente con todas las etiquetas reales. Por esta
46

razo´n, se an˜adieron m´etricas adicionales m´as representativas del comportamiento del
| modelo | en este | tipo de tareas. |     |     |     |     |     |
| ------ | ------- | --------------- | --- | --- | --- | --- | --- |
Adema´s, se aplico´ la t´ecnica de early stopping con una paciencia de 5 ´epocas.
Estosignificaqueelentrenamientosedetieneautoma´ticamentesielmodelonomuestra
mejoras en la m´etrica de validacio´n durante cinco ´epocas seguidas. Esta medida ayuda
a evitar el sobreentrenamiento (overfitting), que ocurre cuando el modelo se ajusta
demasiado a los datos de entrenamiento y pierde capacidad para generalizar sobre
datos nuevos.
En nuestro caso, el entrenamiento se detuvo en la ´epoca 15, lo cual indica que
el modelo alcanzo´ su punto o´ptimo de rendimiento antes de completar todas las ´epo-
cas previstas. Este punto se determino´ observando las m´etricas de validaci´on como la
p´erdida (loss), la precisi´on (accuracy), la precisi´on positiva (precision), la recuperaci´on
´
(recall) y el AUC (Area Bajo la Curva ROC). A partir de la ´epoca 15, dichas m´etricas
dejaron de mostrar mejoras significativas, lo cual activo´ la condicio´n del early stopping.
En la tabla 6.1 y se muestra la evoluci´on de las principales m´etricas durante las
| primeras | 15 ´epocas | del entrenamiento |        | del    | modelo: |        |        |
| -------- | ---------- | ----------------- | ------ | ------ | ------- | ------ | ------ |
|          |            | Epoch             | Acc    | AUC    | Loss    | Prec   | Rec    |
|          |            | 1                 | 0.4638 | 0.9117 | 0.1557  | 0.7121 | 0.3333 |
|          |            | 2                 | 0.6219 | 0.9402 | 0.1281  | 0.9159 | 0.3727 |
|          |            | 3                 | 0.6360 | 0.9470 | 0.1285  | 0.9040 | 0.3296 |
|          |            | 4                 | 0.6675 | 0.9598 | 0.1074  | 0.8887 | 0.4896 |
|          |            | 5                 | 0.6243 | 0.9502 | 0.1185  | 0.8367 | 0.4930 |
|          |            | 6                 | 0.6806 | 0.9627 | 0.1021  | 0.8672 | 0.5420 |
|          |            | 7                 | 0.6493 | 0.9540 | 0.1133  | 0.8567 | 0.5110 |
|          |            | 8                 | 0.6904 | 0.9607 | 0.1051  | 0.8850 | 0.5212 |
|          |            | 9                 | 0.6909 | 0.9662 | 0.0986  | 0.8861 | 0.5396 |
|          |            | 10                | 0.6924 | 0.9658 | 0.0971  | 0.8555 | 0.5870 |
|          |            | 11                | 0.7046 | 0.9614 | 0.1024  | 0.8617 | 0.5852 |
|          |            | 12                | 0.7034 | 0.9611 | 0.1035  | 0.8535 | 0.5819 |
|          |            | 13                | 0.6559 | 0.9297 | 0.1701  | 0.8269 | 0.5708 |
|          |            | 14                | 0.6965 | 0.9558 | 0.1102  | 0.8584 | 0.5473 |
|          |            | 15                | 0.6940 | 0.9618 | 0.1023  | 0.8641 | 0.5533 |
´
|     | Cuadro | 6.1: Evoluci´on |     | de m´etricas | de  | validacio´n | (Epocas 1–15) |
| --- | ------ | --------------- | --- | ------------ | --- | ----------- | ------------- |
Los resultados obtenidos durante el entrenamiento reflejan una evolucio´n progre-
siva del rendimiento del modelo a lo largo de las 15´epocas. En las primeras iteraciones,
se observa un ra´pido incremento tanto en la precisio´n como en el AUC, lo que indica
que el modelo comienza a aprender patrones u´tiles desde fases tempranas. Por ejemplo,
ya en la ´epoca 2, la m´etrica AUC alcanza un valor de 0.9117 en entrenamiento.
Conforme avanza el entrenamiento, estas m´etricas continu´an mejorando de forma
sostenida. En la ´epoca 10, se alcanza una precisio´n del 85.55% y una sensibilidad
(recall) del 58.7%, lo que muestra que el modelo no solo acierta con mayor frecuencia,
sino que adem´as es capaz de detectar un mayor nu´mero de verdaderos positivos. Esto
47

es especialmente relevante en tareas como la nuestra, ya que cada fragmento puede
tener varios g´eneros simulta´neamente.
Sinembargo,apartirdela´epoca12comienzanaaparecerindiciosdesobreajuste.
Aunque las m´etricas de entrenamiento continu´an mejorando (con un AUC superior al
92%), los valores en el conjunto de validacio´n muestran retrocesos, como se observa
en la ca´ıda de la AUC de validacio´n a 0.9297 en la ´epoca 13. Este patr´on sugiere que
el modelo comienza a memorizar los datos de entrenamiento, en lugar de generalizar
correctamente a datos no vistos.
Gracias al mecanismo de early stopping, el entrenamiento se detuvo autom´atica-
mente en la ´epoca 15, evitando as´ı que el modelo continuara con el sobreajuste. En ese
punto, se alcanza un equilibrio con precisi´on (86.41%) y unos valores robustos tambi´en
envalidaci´on(precisio´ndel86.41%ysensibilidaddel55.33%).Estosresultadosindican
un buen equilibrio entre las distintas m´etricas y validan la eficacia del entrenamiento.
Por u´ltimo, en la figura 6.2 se ha representado la evoluci´on de las principales
m´etricas de validaci´on a lo largo de las distintas ´epocas de entrenamiento:
Figura 6.2: Evoluci´on de las m´etricas de validaci´on por ´epoca
Como vemos, el AUC se mantiene elevado durante todo el entrenamiento, lo cual
indica una buena capacidad del modelo para distinguir entre las clases. La precisi´on
presentaunaligeraca´ıdaamitaddelentrenamiento,peroengeneralsemantieneestable
en valores altos. Por otro lado, el recall y el F1-score muestran una mejora progresiva
durante las primeras ´epocas, estabiliz´andose despu´es alrededor de los valores o´ptimos
obtenidos. Estos resultados respaldan el uso del early stopping, ya que se aprecia una
ligera tendencia al sobreajuste a partir de la ´epoca 12, donde algunas m´etricas de
validaci´on comienzan a descender mientras que las de entrenamiento siguen mejorando.
48

| 6.3. Comparativa |     | con | resultados | de estudios |     | previos |
| ---------------- | --- | --- | ---------- | ----------- | --- | ------- |
En la Tabla 6.2 se compara el rendimiento de mi modelo frente a trabajos previos
(explicados en la secci´on 3.7). El rendimiento se evalu´a utilizando el a´rea bajo la curva
ROC (AUC) y, como se observa, mi modelo alcanza un AUC de 0.9618, superando de
forma notable a los modelos anteriores, cuyos valores oscilan entre 0.861 y 0.898.
|     | Referencia | An˜o |               | Modelo         | AUC    |     |
| --- | ---------- | ---- | ------------- | -------------- | ------ | --- |
|     | -          | 2025 | Mi            | modelo         | 0.9618 |     |
|     | [7]        | 2016 |               | FC-4           | 0.894  |     |
|     | [35]       | 2015 | Bag of        | features y RBM | 0.888  |     |
|     | [11]       | 2014 | Convoluciones | 1D             | 0.882  |     |
|     | [45]       | 2014 | Transferencia | de aprendizaje | 0.88   |     |
|     | [10]       | 2012 | Enfoque       | multi-escala   | 0.898  |     |
|     | [18]       | 2011 | Pooling       | MFCC           | 0.861  |     |
Cuadro 6.2: Comparativa del rendimiento entre el modelo propuesto y trabajos previos
Esta mejora puede deberse a varias diferencias respecto a los enfoques previos.
La primera diferencia podr´ıa ser que he implementado un preprocesamiento dis-
tinto al agrupar los g´eneros similares y, adema´s, fragmento los audios en segmentos
de 3 segundos y utilizo ventanas deslizantes, lo que incrementa el nu´mero de muestras
disponibles para el entrenamiento y permite al modelo aprender patrones temporales
ma´s finos.
Otradiferenciapodr´ıaserquemimodeloutilizatresbloquesdeconvolucio´nsegui-
dos de normalizaci´on por lotes y max pooling, finalizando con capas densas y dropout
para regularizacio´n. Esto contrasta con la arquitectura de Choi et al. [7], que emplea-
ba capas convolucionales mucho ma´s profundas y procesaba audios completos de 30
segundos sin ventanas deslizantes, lo que puede limitar la capacidad de generalizacio´n
| y aumentar | el riesgo | de sobreajuste. |     |     |     |     |
| ---------- | --------- | --------------- | --- | --- | --- | --- |
49

7. Conclusiones y trabajo a futuro
En este trabajo se presenta una propuesta para la clasificaci´on de fragmentos
musicales en distintos g´eneros de espectrogramas generados a partir de archivos de
audio y procesados mediante redes neuronales convolucionales.
El uso del dataset MagnaTagATune ha sido fundamental, no solo por su riqueza
enetiquetas,sinotambi´enporquemehaobligadoaenfrentarmealarealidaddeldesba-
lanceo de clases, un problema habitual en tareas de clasificacio´n musical. Para mitigar
este efecto, opt´e por agrupar g´eneros con caracter´ısticas similares, lo que me permitio´
trabajarconmenosclasesma´sequilibradasyas´ı,obtenerresultadosma´srobustos.Este
enfoque ha demostrado ser efectivo, especialmente al comparar el rendimiento de mi
modelo con el de estudios previos, donde la diferencia en las m´etricas evidencia la im-
portancia de adaptar el preprocesamiento y la estructura de los datos a las necesidades
concretas del problema.
En cuanto a los espectrogramas, he podido experimentar con distintas formas
de generarlos y he comprobado c´omo su uso como entrada para redes convolucionales
potencia la capacidad del modelo para captar patrones relevantes. La representacio´n
tiempo-frecuencia que ofrecen resulta especialmente u´til para el an´alisis musical, ya
que permite que la red detecte matices que ser´ıan dif´ıciles de identificar a partir de la
sen˜al de audio en crudo.
La naturaleza multilabel de mi clasificacio´n me llev´o a investigar y utilizar varias
m´etricas de evaluaci´on. La validacio´n cruzada con cinco folds, junto con el uso de early
stopping,hasidoclaveparaevitarelsobreajusteyasegurarquelosresultadosobtenidos
sean realmente representativos y no fruto de una particio´n afortunada de los datos.
Trabajar con redes neuronales convolucionales ha sido una oportunidad para pro-
fundizar en su funcionamiento y experimentar con diferentes arquitecturas y configu-
raciones. No solo he aprendido a ajustar para´metros y a interpretar los resultados, sino
que tambi´en he podido comparar mi enfoque con otros ma´s complejos, comprobando
que, en ocasiones, una arquitectura m´as sencilla y un buen preprocesamiento pueden
superar a modelos mucho ma´s profundos y pesados, especialmente cuando los datos
son limitados o ruidosos.
En relacio´n con la aplicabilidad de este trabajo, creo que lo desarrollado aqu´ı
podr´ıa tener un impacto real en entornos empresariales, especialmente en platafor-
mas de streaming musical, sistemas de recomendacio´n o herramientas de catalogacio´n
automa´tica. La flexibilidad del enfoque permite adaptarlo a otros tipos de datos rela-
cionados con el audio, como la clasificacio´n de emociones, la detecci´on de instrumentos
o incluso la segmentaci´on de eventos sonoros en otros contextos, como el an´alisis de
sonido ambiental o la monitorizacio´n de calidad en entornos industriales.
Aunque el uso de MagnaTagATune ha sido muy valioso, ser´ıa interesante ampliar
el estudio a otros datasets, como GTZAN, para comprobar la capacidad de generali-
zacio´n del modelo y explorar posibles mejoras. Adem´as, siempre queda margen para
probar nuevas arquitecturas, incorporar t´ecnicas de interpretabilidad que ayuden a en-
50

tender mejor las decisiones del modelo o incluso adaptar el sistema para funcionar en
tiempo real, abriendo la puerta a aplicaciones pr´acticas en streaming o en dispositivos
con recursos limitados.
En definitiva, este trabajo me ha permitido comprobar que la combinaci´on de un
buen preprocesamiento, una arquitectura adaptada y una evaluacio´n rigurosa puede
marcar la diferencia en tareas complejas como la clasificaci´on musical multilabel.
51

A. Bibliograf´ıa
[1] Safaa Allamy and Alessandro Lameiras Koerich. 1d cnn architectures for music
genre classification. arXiv preprint arXiv:2105.07302, 2021. URLhttps://arxiv.
org/abs/2105.07302.
[2] Aphex. Inteligencia artificial en la enfermedad de parkin-
son y otros trastornos del movimiento - scientific figure on re-
searchgate, 2025. URL https://www.researchgate.net/figure/
Figura-1-Esquema-de-la-arquitectura-de-las-redes-neuronales-convolucionales-Modificada
fig1 371564848.
[3] Hareesh Bahuleyan. Music genre classification using machine learning techni-
ques. arXiv preprint arXiv:1804.01149, 2018. URL https://arxiv.org/abs/
1804.01149.
[4] N. Bala Ganesh et al. Genrenet: A deep based approach for music genre classi-
fication. SN Computer Science, 5:1135, 2024. URL https://doi.org/10.1007/
s42979-024-03493-x.
[5] O.BarcelonaFerna´ndez. Deeplearning.redesneuronalesconvolucionalesaplicadas
a la detecci´on de la malaria. Universidad de Zaragoza, page 41, 2024. URL
https://zaguan.unizar.es/record/149481/files/TAZ-TFG-2024-3499.pdf.
[6] Julian Blackmore. ¿qu´e es el espectrograma?, 2023. URL https://emastered.
com/es/blog/what-is-spectrogram.
[7] Keunwoo Choi, Gy¨orgy Fazekas, Mark Sandler, and Kyunghyun Cho. Convo-
lutional recurrent neural networks for music classification. arXiv preprint ar-
Xiv:1606.00298, 2016.
[8] Keunwoo Choi, Gyorgy Fazekas, Mark Sandler, and Kyunghyun Cho. A tutorial
on deep learning for music information retrieval. arXiv preprint arXiv:1709.04396,
2017.
[9] Gobierno de Espan˜a. Qu´e es la inteligencia artificial,
2023. URL https://planderecuperacion.gob.es/noticias/
que-es-inteligencia-artificial-ia-prtr.
[10] Sander Dieleman and Benjamin Schrauwen. Multiscale approaches to music au-
dio feature learning. In Proceedings of the 14th International Society for Music
Information Retrieval Conference, ISMIR 2013, pages 116–121, Curitiba, Brazil,
2013.
[11] Sander Dieleman and Benjamin Schrauwen. End-to-end learning for music audio.
In2014 IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), pages 6964–6968. IEEE, 2014.
52

[12] Mingwen Dong. Convolutional neural network achieves human-level accuracy in
music genre classification. arXiv preprint arXiv:1802.09697, 2018. URL https:
//arxiv.org/abs/1802.09697.
[13] Timothy Dozat. Incorporating nesterov momentum into adam. arXiv preprint,
2016. URL https://arxiv.org/abs/1609.04747. arXiv:1609.04747.
[14] P. Dwivedi and B. Islam. Generative adversarial networks based framework for
music genre classification. SN Computer Science, 5:1149, 2024. URL https:
//doi.org/10.1007/s42979-024-03531-8.
[15] Glassdoor. Sueldosdeingenierodesoftware,2025. URLhttps://www.glassdoor.
es/Sueldos/software-engineer-sueldo-SRCH KO0,17.htm.
[16] Hanlin Gu, Yin Xian, Ilona Christy Unarta, and Yuan Yao. Generative adversarial
networks for robust cryo-em image denoising. arXiv preprint arXiv:2008.07307,
2020. URL https://arxiv.org/abs/2008.07307.
[17] B G´omez Pujante. Redes convolucionales. aplicacio´n a la clasificaci´on
de ima´genes m´edicas. Universidad Miguel Hern´andez de Elche, page 94,
2023. URL https://dspace.umh.es/jspui/bitstream/11000/30233/1/TFG-G%
C3%B3mez%20Pujante%2C%20Bego%C3%B1a.pdf.
[18] Philippe Hamel, Simon Lemieux, Yoshua Bengio, and Douglas Eck. Temporal
pooling and multiscale learning for automatic annotation and ranking of music
audio. In Proceedings of the 12th International Society for Music Information
Retrieval Conference, ISMIR 2011, pages 729–734, Miami, Florida, USA, 2011.
[19] Red Hat. ¿qu´e es el aprendizaje autom´atico?, 2023. URL https://www.redhat.
com/es/topics/ai/what-is-machine-learning.
[20] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition.
In CVPR, 2016.
[21] Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren
Jansen, Ron Moore, Manoj Plakal, David Platt, Rif A. Saurous, Brian Seybold,
et al. Cnn architectures for large-scale audio classification. In 2017 IEEE Inter-
national Conference on Acoustics, Speech and Signal Processing (ICASSP), pages
131–135. IEEE, 2017.
[22] G. Hinton. Neural networks for machine learning: Lecture 6a rmsprop & lecture
6b momentum, 2012.
[23] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger. Densely connected
convolutional networks. In CVPR, 2017. URL https://www.sciencedirect.com/
journal/neurocomputing.
[24] IBM. Aprendizaje no supervisado, 2023. URL https://www.ibm.com/es-es/
topics/unsupervised-learning.
53

[25] P. Jain, A. Sar, T. Choudhury, V. Singh, and K. Kotecha. Differentiation
of music genre from an audio file using neural networks. AI Technologies
for Information Systems and Management Science, 1136:482–490, 2024. doi:
10.1007/978-3-031-70789-6 40. URL https://link.springer.com/chapter/10.
1007/978-3-031-70789-6 40.
[26] Ee Hern Kheng, Chia Pao Liew, Tianhao Lan, and Kim Geok Tan. Advancing
handwritten musical notation recognition using deep learning: A convolutional
neural network-based approach with improved accuracy. International Journal of
Pattern Recognition and Artificial Intelligence, 38(3):2452007, 2024. doi: 10.1142/
S0218001424520074.
[27] Taejun Kim, Jongpil Lee, and Juhan Nam. Sample-level cnn architectures for
music auto-tagging using raw waveforms. In ICASSP 2018 - IEEE International
Conference on Acoustics, Speech and Signal Processing, pages 366–370. IEEE,
2018. URL https://arxiv.org/abs/1710.10451.
[28] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv
preprint arXiv:1412.6980, 2014.
[29] Edith Law and Luis Von Ahn. Magnatagatune dataset. https://mirg.city.ac.
uk/datasets/magnatagatune/, 2009.
[30] Hao Li, Gopi Krishnan Rajbahadur, Dayi Lin, Cor-Paul Bezemer, and Zhen Ming
Jiang. Keeping deep learning models in check: A history-based approach to mi-
tigate overfitting. arXiv preprint arXiv:2401.10359, 2024. URL https://arxiv.
org/abs/2401.10359.
[31] Y. Li, Y. Zhang, X. Wang, and Q. Liu. Accuracy assessment in convolutional
neural network-based deep learning remote sensing studies—part 1: Literature
review. Remote Sensing, 13(13):2450, 2021. doi: 10.3390/rs13132450.
[32] Caifeng Liu, Lin Feng, Guochao Liu, Huibing Wang, and Shenglan Liu. Bottom-
up broadcast neural network for music genre classification. arXiv preprint ar-
Xiv:1901.08928, 2019. URL https://arxiv.org/abs/1901.08928.
[33] Zhuang Liu, Zhiqiu Xu, Joseph Jin, Zhiqiang Shen, and Trevor Darrell. Dro-
pout reduces underfitting. arXiv preprint arXiv:2303.01500, 2023. URL https:
//arxiv.org/abs/2303.01500.
[34] Christopher Mahlich, Tobias Vente, and Joeran Beel. From theory to prac-
tice: Implementing and evaluating e-fold cross-validation. arXiv preprint ar-
Xiv:2410.09463, 2024. URL https://arxiv.org/abs/2410.09463.
[35] Juhan Nam, Jorge Herrera, and Kyogu Lee. A deep bag-of-features model for
music auto-tagging. arXiv preprint arXiv:1508.04999, 2015.
[36] Ndiatenda Ndou, Ritesh Ajoodha, and Ashwini Jadhav. Music genre classifica-
tion: A review of deep-learning and traditional machine-learning approaches. In
2021 IEEE International IOT, Electronics and Mechatronics Conference (IEM-
TRONICS), pages 1–6, 2021. doi: 10.1109/IEMTRONICS52119.2021.9422487.
54

[37] Yurii Nesterov. A method for unconstrained convex minimization problem with
the rate of convergence o(1/k2). Doklady AN USSR, 269(3):543–547, 1983.
[38] Mart´ın Ezequiel Paz, Guillermo Friedrich, and Christian Luis Galasso. Pro-
cesamiento de sen˜al visualizado sobre un espectrograma. Elektron: Ciencia y
Tecnolog´ıa en la Electr´onica de Hoy, 4(1):35–39, 2020. ISSN 2525-0159. URL
https://dialnet.unirioja.es/servlet/articulo?codigo=7468991.
[39] LawrenceR.RabinerandBiing-HwangJuang. Fundamentals of speech recognition.
Prentice Hall.
[40] PolMart´ınSahuja. Entendiendolacurvarocyelauc:dosmedidasdelrendimiento
de un clasificador binario que van de la mano. https://polmartisanahuja.com/
entendiendo-la-curva-roc-y-el-auc-dos-medidas-del-rendimiento-de-un-clasificador-binario-que-van-de-la-mano/.
[41] SoldAI. Tipos de aprendizaje automa´tico, 2021. URL https://medium.com/
soldai/tipos-de-aprendizaje-automtico-6413e3c615e2.
[42] S. S. Stevens, J. Volkmann, and E. B. Newman. A scale for the measurement
of the psychological magnitude pitch. The Journal of the Acoustical Society of
America, 8(3):185–190.
[43] Talent.com. Salario de ingeniero de software en espan˜a, 2025. URL https://es.
talent.com/salary?job=ingeniero+de+software.
[44] Telefo´nica. Aprendizaje supervisado: definici´on y aplicaciones, 2023.
URL https://www.telefonica.com/es/sala-comunicacion/blog/
aprendizaje-supervisado-definicion-aplicaciones/.
[45] Aaron Van Den Oord, Sander Dieleman, and Benjamin Schrauwen. Transfer lear-
ning by supervised pretraining for audio-based music classification. In Proceedings
of the 15th International Society for Music Information Retrieval Conference, IS-
MIR 2014, pages 29–34, Taipei, Taiwan, 2014.
[46] Minz Won, Andres Ferraro, Dmitry Bogdanov, and Xavier Serra. Evaluation of
cnn-based automatic music tagging models. Zenodo, 2020. URL https://zenodo.
org/record/3898838.
[47] Eberhard Zwicker and Hugo Fastl. Psychoacoustics: Facts and models. Springer
Science & Business Media, 2013.
55