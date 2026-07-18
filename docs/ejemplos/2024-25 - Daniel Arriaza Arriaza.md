|                |                  | TRABAJO     |               | FIN       | DE MASTER ´ |         |            |
| -------------- | ---------------- | ----------- | ------------- | --------- | ----------- | ------- | ---------- |
| An´alisis      |                  | y           | Disen˜o       |           | de          | Sistema | de         |
| Recomendaci´on |                  |             |               | Musical   |             | Basada  | en         |
|                | Caracter´ısticas |             |               |           | de          | Audio   |            |
|                |                  |             | Realizado     |           | por         |         |            |
|                |                  | Daniel      | Arriaza       |           | Arriaza     |         |            |
|                |                  | Para        | la obtencio´n |           | del t´ıtulo | de      |            |
| M´aster        | en               | Ingenier´ıa | del           | Software: | Cloud,      | Datos   | y Gesti´on |
TI
|     |     |              | Dirigido |           | por    |         |     |
| --- | --- | ------------ | -------- | --------- | ------ | ------- | --- |
|     |     | Jos´e        | Mar´ıa   | Luna      | Romera |         |     |
|     |     | Convocatoria |          | de Junio, | curso  | 2024/25 |     |

Agradecimientos
Me gustar´ıa agradecer a todas las personas que han sido un apoyo sustancial en
| el desarrollo | de este trabajo. |     |
| ------------- | ---------------- | --- |
A mi tutor, Jos´e Mar´ıa, por aliviar mis inquietudes respecto al mundo de la
investigaci´on y la docencia. Sus consejos y su cercan´ıa me han ayudado much´ısimo.
A mi familia, por escuchar todas mis quejas y problemas a lo largo del proyecto
| y del ma´ster | en general. |     |
| ------------- | ----------- | --- |
A mis amigos: Rafa, Mar´ıa y Aitor. A Rafa, por acompan˜arme casi desde el
comienzo de la carrera hasta hoy. A Mar´ıa y Aitor, por convertiros en un motivo m´as
| para ir con | ilusio´n a clase  | cada d´ıa. |
| ----------- | ----------------- | ---------- |
| A           | vosotros, gracias | por todo.  |
i

Resumen
Este trabajo se centra en investigar los sistemas de recomendacio´n musical ba-
sados en contenido y los enfoques y t´ecnicas actuales para representar y comparar
fragmentos musicales a partir de sus caracter´ısticas. Para ello, se ha utilizado el da-
taset MagnaTagATune, que contiene fragmentos de canciones anotados con etiquetas
musicales, caracter´ısticas de audio y metadatos.
El estado del arte revisado ha confirmado que, si bien los m´etodos cla´sicos de
filtrado colaborativo y basado en contenido siguen siendo relevantes, las tendencias
actuales incluyen el uso de embeddings generados con deep learning, sistemas h´ıbridos,
t´ecnicas basadas en sen˜ales musicales y estrategias centradas en las emociones del
usuario. Estos enfoques permiten una mayor personalizacio´n y abordan desaf´ıos como
la diversidad, la evoluci´on de las preferencias y la escasez de datos iniciales.
A lo largo del estudio se ha analizado co´mo atributos como el tempo, los MFCC,
los vectores chroma, y la STFT pueden utilizarse para generar recomendaciones pre-
cisas, especialmente en escenarios donde los sistemas colaborativos tienen problemas
como el “cold start”. Se han codificado vectores de alta dimensionalidad utilizando
estas caracter´ısticas y se han evaluado diversas t´ecnicas de reducci´on de dimensionali-
dad; concretamente PCA, Autoencoders y Variational Autoencoders, para mejorar la
eficiencia del c´alculo de similitud entre fragmentos.
Las recomendaciones se han generado utilizando k-Nearest Neighbors sobre el
espacio latente. Los resultados muestran que el uso combinado de caracter´ısticas de
audio y metadatos de los g´eneros y etiquetas de las canciones mejora notablemente la
calidad de las recomendaciones, sobre todo cuando se utiliza un Autoencoder. Por el
contrario, los Variational Autoencoders no han conseguido un rendimiento suficiente
y podr´ıa deberse a la complejidad del modelo o a una configuracio´n sub´optima de los
hiperpara´metros.
En conclusi´on, el uso de Autoencoders frente a Variational Autoencoders y PCA
haresultadoaltamentesatisfactorioparalareducci´ondedimensionalidadylamejorade
lacalidaddelasrecomendacionesmusicales,especialmentecuandoseintegranmu´ltiples
fuentes de informaci´on.
Palabras clave: Sistemas de recomendacio´n musical, Recomendacio´n basada
en contenido, MFCC, tempo, chroma, caracter´ısticas espectrales, Autoencoder, PCA,
Variational Autoencoder
ii

Abstract
This work focuses on the study of content-based music recommendation systems
and the current approaches and techniques for representing and comparing music frag-
ments based on their features. For this purpose, the MagnaTagATune dataset has been
used, which contains annotated song fragments with musical tags, audio features, and
metadata.
The reviewed state of the art confirms that, although classical content-based and
collaborative filtering methods remain relevant, recent trends include the use of deep
learning-generated embeddings, hybrid systems, signal-based techniques, and emotion-
driven strategies. These approaches enable greater personalization and address challen-
ges such as diversity, evolving user preferences, and the cold-start problem.
Throughout this study, we have analyzed how attributes such as tempo, MFCC,
chroma vectors, and STFT can be leveraged to generate accurate recommendations,
particularly in scenarios where collaborative filtering systems struggle with limited da-
ta. High-dimensional feature vectors have been encoded using these attributes, and
several dimensionality reduction techniques have been evaluated; namely PCA, Auto-
encoders, and Variational Autoencoders, to enhance the efficiency of similarity compu-
tation between fragments.
Recommendations have been generated using the k-Nearest Neighbors algorithm
in the latent space. The results show that combining audio features with metadata
such as genres and tags significantly improves the quality of the recommendations,
especially when using Autoencoders. In contrast, Variational Autoencoders did not
achievesatisfactoryperformance,whichmaybeduetomodelcomplexityorsuboptimal
hyperparameter configurations.
In conclusion, the use of Autoencoders, compared to Variational Autoencoders
and PCA, has proven highly effective for dimensionality reduction and enhancing the
quality of music recommendations, particularly when multiple sources of information
are integrated.
Keywords: Music recommendation systems, Content-based recommendation,
MFCC, tempo, chroma, spectral features, Autoencoder, PCA, Variational Autoenco-
der
iii

´
Indice general
1. Introduccio´n 1
1.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
1.2. Estructura de este documento . . . . . . . . . . . . . . . . . . . . . . . 2
2. Fundamentos teo´ricos 3
2.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2. Sistemas de recomendaci´on . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.3. Caracter´ısticas de audio . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2.3.1. Tempo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2.3.2. STFT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.3.3. MFCC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.3.4. Chroma Coefficients . . . . . . . . . . . . . . . . . . . . . . . . 7
2.4. Inteligencia Artificial . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
2.4.1. Embedding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
2.4.2. Machine Learning . . . . . . . . . . . . . . . . . . . . . . . . . . 9
2.4.3. Deep Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
3. Estado del arte 15
3.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
3.2. Caracter´ısticas de audio . . . . . . . . . . . . . . . . . . . . . . . . . . 15
3.3. Sistemas de recomendaci´on . . . . . . . . . . . . . . . . . . . . . . . . . 16
3.3.1. Algoritmos utilizados en la literatura . . . . . . . . . . . . . . . 17
3.3.2. Nuevos enfoques en los sistemas de recomendaci´on . . . . . . . . 18
3.4. Conclusiones . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
4. Estudio previo 22
4.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
4.2. Objetivos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
4.3. Metodolog´ıa . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
4.3.1. Documentaci´on . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.3.2. Elecci´on del dataset . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.3.3. Implementacio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
4.3.4. Elecci´on de algoritmos . . . . . . . . . . . . . . . . . . . . . . . 25
4.3.5. Procesamiento de datos . . . . . . . . . . . . . . . . . . . . . . 25
4.3.6. Entrenamiento y evaluacio´n . . . . . . . . . . . . . . . . . . . . 26
4.4. Planificacio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
4.4.1. Cronograma inicial . . . . . . . . . . . . . . . . . . . . . . . . . 27
4.4.2. Cronograma real . . . . . . . . . . . . . . . . . . . . . . . . . . 29
4.5. Presupuesto . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
4.5.1. Recursos Intelectuales . . . . . . . . . . . . . . . . . . . . . . . 31
4.5.2. Recursos materiales . . . . . . . . . . . . . . . . . . . . . . . . . 31
5. Descripcio´n de la propuesta 32
iv

5.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
5.2. Propuesta . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
5.3. Estructura del sistema de recomendacio´n . . . . . . . . . . . . . . . . . 32
5.3.1. Arquitectura final . . . . . . . . . . . . . . . . . . . . . . . . . . 33
5.4. Embeddings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
6. Implementaci´on 37
6.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
6.2. Herramientas . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
6.2.1. Entorno de ejecucio´n . . . . . . . . . . . . . . . . . . . . . . . . 37
6.2.2. Preparaci´on del entorno de desarrollo . . . . . . . . . . . . . . . 37
6.2.3. Instalaci´on de dependencias . . . . . . . . . . . . . . . . . . . . 37
6.2.4. Librer´ıas utilizadas . . . . . . . . . . . . . . . . . . . . . . . . . 38
6.3. Preprocesamiento . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
6.4. Visualizacio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
6.5. Modelos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
7. Pruebas 53
7.1. Introduccio´n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
7.2. Modelos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
7.3. Recomendaciones . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
7.3.1. Distancia eucl´ıdea . . . . . . . . . . . . . . . . . . . . . . . . . . 60
7.3.2. Distancia Manhattan . . . . . . . . . . . . . . . . . . . . . . . . 63
8. Conclusiones 68
A. Bibliograf´ıa 69
v

´
Indice de figuras
2.1. Deteccio´n de onsets. Imagen obtenida de [36]. . . . . . . . . . . . . . . 5
2.2. Ejemplo de melod´ıa y armon´ıa. Imagen obtenida de [3]. . . . . . . . . . 7
´
2.3. Areas de la Inteligencia Artificial. Imagen obtenida de [38]. . . . . . . . 8
2.4. Autoencoder. Imagen obtenida de [45]. . . . . . . . . . . . . . . . . . . 11
2.5. Espacio latente de Autoencoder. Imagen obtenida de [45]. . . . . . . . . 12
2.6. Variational Autoencoder. Imagen obtenida de [45]. . . . . . . . . . . . . 13
2.7. Espacio latente del Variational Autoencoder. Imagen obtenida de [45]. . 14
3.1. Ilustracio´n de sistemas basados en filtrado colaborativo . . . . . . . . . 16
3.2. Ilustracio´n de sistemas basados en filtrado basado en contenido . . . . . 17
4.1. Metodolog´ıa del proyecto . . . . . . . . . . . . . . . . . . . . . . . . . . 23
5.1. Arquitectura del sistema . . . . . . . . . . . . . . . . . . . . . . . . . . 34
6.1. Densidad de canciones por g´enero . . . . . . . . . . . . . . . . . . . . . 42
6.2. Densidad de canciones por etiqueta . . . . . . . . . . . . . . . . . . . . 43
6.3. Densidad de fragmentos por g´enero . . . . . . . . . . . . . . . . . . . . 44
6.4. Comparacio´n de etiquetas ma´s y menos frecuentes . . . . . . . . . . . . 45
6.5. Coocurrencia de g´eneros . . . . . . . . . . . . . . . . . . . . . . . . . . 46
6.6. Coocurrencia de etiquetas . . . . . . . . . . . . . . . . . . . . . . . . . 47
6.7. Correlacio´n entre g´eneros y etiquetas . . . . . . . . . . . . . . . . . . . 48
7.1. Varianza acumulada por nu´mero de componentes . . . . . . . . . . . . 55
7.2. Comparacio´n de resultados: (arriba) Autoencoder, (abajo) Autoencoder
full . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
7.3. Comparacio´n de resultados: (arriba) VAE, (abajo) VAE full . . . . . . 56
7.4. Comparacio´n de errores por modelo y m´etrica . . . . . . . . . . . . . . 57
7.5. Espacio latente: (izquierda) Autoencoder, (derecha) VAE . . . . . . . . 58
7.6. Comparacio´n de recomendaciones con distancia eucl´ıdea: (arriba) PCA,
(abajo) PCA full . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
7.7. Comparacio´n de recomendaciones con distancia eucl´ıdea: (arriba) Auto-
encoder, (abajo) Autoencoder full . . . . . . . . . . . . . . . . . . . . . 62
7.8. Comparacio´n de recomendaciones con distancia eucl´ıdea: (arriba) VAE,
(abajo) VAE full . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
7.9. Comparacio´n de recomendaciones con distancia Manhattan: (arriba)
PCA, (abajo) PCA full . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
7.10.Comparacio´nderecomendacionescondistanciaManhattan:(arriba)Au-
toencoder, (abajo) Autoencoder full . . . . . . . . . . . . . . . . . . . . 66
7.11.Comparacio´n de recomendaciones con distancia Manhattan: (arriba)
VAE, (abajo) VAE full . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
vi

´
Indice de extractos de co´digo
6.1. Construccio´n del autoencoder . . . . . . . . . . . . . . . . . . . . . . . 49
6.2. Construccio´n del Variational Autoencoder . . . . . . . . . . . . . . . . 50
6.3. Recomendacio´n con k-NN . . . . . . . . . . . . . . . . . . . . . . . . . 52
vii

1. Introducci´on
1.1. Introducci´on
Enlaactualidad,elvolumendeinformaci´onycontenidodisponibleenInternetha
crecidodeformaexponencial.Estaabundanciadeopcioneshageneradolanecesidadde
herramientas que ayuden a los usuarios a encontrar contenido relevante, personalizado
y de inter´es entre la gran inmensa cantidad de datos. En este contexto, los sistemas
de recomendacio´n se han convertido en una pieza clave para mejorar la experiencia del
usuario en mu´ltiples plataformas y sectores.
Plataformas de entretenimiento como Netflix, YouTube, Amazon o Spotify utili-
zansistemasderecomendaci´onavanzadosparaofrecerasususuariosproductos,v´ıdeos,
pel´ıculas o canciones adaptadas a sus gustos personales. Estos sistemas son responsa-
blesdebuenapartedelcontenidoqueelusuarioconsume,ysueficaciatieneunimpacto
directo tanto en la satisfacci´on del usuario como en los beneficios econo´micos de las
plataformas. De hecho, se estima que ma´s del 80% de las visualizaciones en Netflix
provienen de su sistema de recomendacio´n [16], y un porcentaje similar en plataformas
musicales como Spotify [34].
En el ´ambito musical, la recomendaci´on personalizada es especialmente impor-
tante debido a la gran variedad de g´eneros, estilos, artistas y nuevas canciones que
se publican constantemente. Los sistemas de recomendacio´n no solo facilitan el des-
cubrimiento de nueva mu´sica, sino que tambi´en contribuyen a fidelizar a los usuarios,
aumentar su tiempo de permanencia en la plataforma y fomentar un mayor consumo
de contenido.
Generalmente, estos sistemas utilizan enfoques de filtrado colaborativo, que se
basan en las interacciones de los usuarios (como reproducciones, valoraciones o play-
lists compartidas), o filtrado basado en contenido, que se apoya en las caracter´ısticas
intr´ınsecas de los´ıtems recomendados (por ejemplo, el g´enero de una cancio´n, el tempo
o sus atributos espectrales). Normalmente, se combinan ambos enfoques formando sis-
temas h´ıbridos para mejorar la precisi´on y superar limitaciones como el problema del
“cold start”, que es un problema que sucede cuando no hay suficientes datos previos
de un usuario o un´ıtem.
Las t´ecnicas de machine learning y deep learning han impulsado grandes avances
en este campo, permitiendo desarrollar modelos ma´s complejos y precisos que son
capaces de detectar nuevos patrones y generar recomendaciones ma´s ajustadas a los
gustos y necesidades de cada usuario.
El presente trabajo se propone realizar un estudio de la situacio´n actual de los
sistemas de recomendacio´n y las t´ecnicas utilizadas, enfoca´ndose en la recomendacio´n
musical y en los sistemas basados en contenido. Se revisara´ el estado de la literatura
en este campo, y se propondr´a la implementaci´on de un sistema de recomendacio´n
que utilice caracter´ısticas de audio para representar las canciones. Este sistema per-
mitira´ comparar diferentes estrategias de reducci´on de dimensionalidad, como PCA,
1

Autoencoders y Variational Autoencoders, y evaluara´ su rendimiento en un entorno
experimental.
| 1.2. Estructura |     | de        | este documento   |     |
| --------------- | --- | --------- | ---------------- | --- |
| La estructura   | del | documento | es la siguiente: |     |
Introduccio´n: Esta seccio´n establece el contexto del trabajo y la estructura del
documento.
Fundamentos teo´ricos: Incluye todos los fundamentos sobre sistemas de reco-
mendacio´n, audio e inteligencia artificial que sera´n necesarios saber para poder
| entender | las secciones | siguientes. |     |     |
| -------- | ------------- | ----------- | --- | --- |
Estado del Arte: Se dedica un ana´lisis de las t´ecnicas utilizadas en la literatura.
Se detallan algunos enfoques actuales y los problemas y limitaciones identificados
| para implementarlos |     | y evaluarlos. |     |     |
| ------------------- | --- | ------------- | --- | --- |
Estudio previo: Constituye la base del proyecto, donde se define la metodolog´ıa
| usada, la | planificaci´on | para | todo el estudio | y el presupuesto. |
| --------- | -------------- | ---- | --------------- | ----------------- |
Descripci´on de la propuesta: Esta seccio´n incluye todo el disen˜o de la arqui-
tectura y de las caracter´ısticas de audio utilizadas en la literatura.
Implementacio´n: Se incluyen todas las decisiones tomadas para el preprocesa-
miento de los datos y la construccio´n de los modelos, siguiendo la arquitectura
| definida | en la seccio´n | anterior. |     |     |
| -------- | -------------- | --------- | --- | --- |
Pruebas: Se presenta la evaluacio´n de los diferentes modelos y de la calidad de
las recomendaciones, comparando qu´e t´ecnicas dan mejores resultados.
Conclusiones: La seccio´n final sintetiza los puntos clave del proyecto, ofreciendo
| unas reflexiones | y   | puntos | de mejora. |     |
| ---------------- | --- | ------ | ---------- | --- |
2

2. Fundamentos te´oricos
2.1. Introducci´on
En esta seccio´n explicaremos los principales conceptos te´oricos utilizados a lo
largo de todo el desarrollo, tanto teor´ıa sobre sistemas de recomendacio´n, como carac-
ter´ısticas de audio y t´ecnicas de machine learning y deep learning.
2.2. Sistemas de recomendaci´on
Los sistemas de recomendacio´n han sido un ´area que ha ido evolucionando desde
el primer paper sobre filtrado colaborativo publicado en la d´ecada de 1990 [39]. Son
muy utilizados en diversas a´reas, aunque nos vamos a centrar en el a´mbito musical.
Labu´squedademu´sicahasidomuybeneficiadaporlaexpansi´ondeInternet,pues
ha permitido que gigantescas cantidades de canciones est´en disponibles para la gente.
Sumando el hecho de que los usuarios buscan contenidos segu´n sus gustos personales,
los sistemas de recomendacio´n se vuelven ma´s necesarios [4].
Como su nombre indica, un sistema de recomendaci´on de mu´sica (Music Recom-
mendation System o MRS en ingl´es) es un sistema capaz de proveer recomendaciones
musicales teniendo en cuenta las preferencias del usuario, de un grupo de usuarios, etc.
Los principales enfoques utilizados son tres [4]:
Sistemas Basados en Contenido: Son aquellos que, en base a las caracter´ısti-
cas de las canciones y al historial del usuario, intenta predecir qu´e quiere escuchar
y qu´e canciones similares mostrar.
Filtrado Colaborativo:Permitequelasinteraccionesdelosusuariosylosdatos
generados por estos ayuden a incrementar la calidad de las recomendaciones [14].
Ba´sicamente recopila preferencias o gustos del usuario y los compara con los
datos de personas con patrones similares para recomendar aquellas canciones que
le gusten a esos usuarios.
Filtrado H´ıbrido: Este combina los dos enfoques anteriores, teniendo en cuenta
tanto la similitud entre canciones como la similitud entre usuarios. La ventaja de
este sistema es que elimina el punto ma´s negativo del filtrado colaborativo: que se
discriminan aquellas canciones que no tengan interacciones (cold start). Es decir,
al utilizar ambos sistemas se puede evitar que solo se recomienden canciones
populares.
Un ejemplo de sistema h´ıbrido ser´ıa Spotify, que utiliza el historial de los usuarios
(filtrado colaborativo), como sus artistas favoritos, canciones ma´s escuchadas, e incluso
el momento del d´ıa en que escucha ciertos g´eneros para recomendar canciones en sec-
ciones como el “Descubrimiento Semanal” o el “Radar de Novedades” [30]. Adema´s,
3

analiza caracter´ısticas de las canciones como la letra o caracter´ısticas extra´ıdas del
audio (basado en contenido) para poder hacer recomendaciones cuando hay poca in-
formacio´ndelusuario(coldstart)oparagenerarplayliststem´aticas(porg´enero,mood,
etc.) [41].
Una vez entendidos los principales enfoques, vamos a centrarnos en los sistemas
basados en contenido, por lo que tenemos que hablar de las caracter´ısticas de audio.
| 2.3. | Caracter´ısticas |     | de audio |     |
| ---- | ---------------- | --- | -------- | --- |
Dado que las sen˜ales de audio son complejas, se emplean diversas t´ecnicas de
extraccio´n de caracter´ısticas que permiten describir propiedades como el ritmo, la ar-
| mon´ıa | o la energ´ıa. |     |     |     |
| ------ | -------------- | --- | --- | --- |
Por una parte, tendr´ıamos caracter´ısticas que miden el ritmo, como el tempo.
Adema´s, algunas miden la frecuencia, como la STFT, o el timbre, como los MFCC.
El timbre nos permite distinguir dos sonidos de igual tono (frecuencia) e igual intensi-
dad (presi´on) cuando son emitidos por dos or´ıgenes diferentes. Por ejemplo, dos voces
| cantando | la misma | nota [54]. |     |     |
| -------- | -------- | ---------- | --- | --- |
Finalmente, hay otras que miden el pitch, como los vectores chroma. El pitch es
la percepcio´n subjetiva de la frecuencia de un sonido, es decir, lo agudo o grave que
percibimoselsonido.Esunt´erminocomplejodeentenderdebidoaquedosnotasendos
instrumentos diferentes pueden tener la misma frecuencia fundamental, por ejemplo,
| un La | a 440 Hz, | pero percibirse | con diferente | timbre [58]. |
| ----- | --------- | --------------- | ------------- | ------------ |
A continuacio´n, pasaremos a ver un poco m´as en profundidad las diferentes ca-
| racter´ısticas | nombradas | en esta | secci´on. |     |
| -------------- | --------- | ------- | --------- | --- |
| 2.3.1.         | Tempo     |         |           |     |
El tempo indica la velocidad de la canci´on, es decir, cua´ntos pulsos suceden por
minuto (BPM). Se suele representar mediante una figura determinada (como una negra
o una corchea) y el nu´mero de veces que esa figura cabe en un minuto. Por ejemplo, ˇ“
= 60 indica que en un minuto caben 60 negras, por lo que cada negra ocupa 1 segundo.
Uno de los enfoques para beat tracking es la programacio´n din´amica. El objetivo
es generar una secuencia de beats que corresponda con los onsets percibidos en la sen˜al,
| y que | estos formen | un patro´n | regular [5]. |     |
| ----- | ------------ | ---------- | ------------ | --- |
Un onset es el momento exacto en el que comienza un nuevo sonido (transient).
Como se puede observar en la figura 2.1, al tocar un sonido ocurre el attack, que es la
fase donde aumenta la energ´ıa del sonido. El t´ermino transient se refiere al pico corto
y de alta energ´ıa al comienzo del sonido, por lo que tambi´en se puede definir el onset
como el instante que marca el inicio del transient, o el punto en el que un transient
| puede | ser detectado | [36]. |     |     |
| ----- | ------------- | ----- | --- | --- |
4

|     | Figura | 2.1: | Detecci´on | de onsets. | Imagen |     | obtenida | de [36]. |
| --- | ------ | ---- | ---------- | ---------- | ------ | --- | -------- | -------- |
Para ello, se utiliza una funcio´n objetivo que localiza los beats y otra que penaliza
las desviaciones:
|     |     |     |         | N        | N        |        |     |     |
| --- | --- | --- | ------- | -------- | -------- | ------ | --- | --- |
|     |     |     |         | (cid:88) | (cid:88) |        |     |     |
|     |     |     | Ξ(t ) = | O(t )+α  |          | F(t −t | ,τ  | )   |
|     |     |     | i       | i        |          | i      | i−1 | p   |
|     |     |     |         | i=1      | i=2      |        |     |     |
Donde t es la secuencia de N beats, O(t) es la fuerza del onset en el instante t, α es
i
un par´ametro para balancear la penalizacio´n de F, y τ es un tempo ideal.
p
Paraconstruirlamejorsecuenciadebeatsposibles,sepuedetomarlafo´rmularecursiva:
|     |     | Ξ∗(t) | = O(t)+ | ma´x | {αF(t−τ,τ | )+Ξ∗(τ)} |     |     |
| --- | --- | ----- | ------- | ---- | --------- | -------- | --- | --- |
p
τ=0,..,t
As´ı, para cada valor de t se decide si deber´ıa considerarse como beat teniendo en
cuenta si hay una alta activaci´on de onsets O(t) y si el tiempo desde el beat anterior τ
| es coherente | con | el tempo | esperado | τ . |     |     |     |     |
| ------------ | --- | -------- | -------- | --- | --- | --- | --- | --- |
p
| 2.3.2. | STFT |     |     |     |     |     |     |     |
| ------ | ---- | --- | --- | --- | --- | --- | --- | --- |
La Short-Time Fourier Transform (STFT) es una Transformada de Fourier que
determina c´omo var´ıa la frecuencia sinusoidal de la sen˜al a lo largo del tiempo. A
diferencia de la Transformada de Fourier (FT), que muestra el contenido espectral
global de la sen˜al, la STFT divide la sen˜al en pequen˜as ventanas de tiempo y aplica la
| transformada | de  | Fourier | en cada | una [35]. |     |     |     |     |
| ------------ | --- | ------- | ------- | --------- | --- | --- | --- | --- |
As´ı, la STFT es capaz de convertir la sen˜al de audio x[n] del dominio tiempo al
| dominio tiempo-frecuencia. |     |     | Esta | se calcula | como: |     |     |     |
| -------------------------- | --- | --- | ---- | ---------- | ----- | --- | --- | --- |
N−1
(cid:88)
|     |     | ξ(m,k) | =   | x[n+mH]·w[n]·e−2jπkn/N |     |     |     |     |
| --- | --- | ------ | --- | ---------------------- | --- | --- | --- | --- |
n=0
Donde x[n] es la sen˜al discreta de audio, w[n] es una ventana discreta de longitud
| N,  |     |     |     |     | N   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
N ∈ N es el taman˜o de la ventana, H ∈ es el taman˜o de salto y k ∈ [0 : K]. El
5

nu´merocomplejoξ(m,k)denotaelk-´esimocoeficientedeFourierparaelm-´esimomarco
temporal, donde K = N/2 es el´ındice de frecuencia correspondiente a la frecuencia de
Nyquist.
La frecuencia de Nyquist surge del Teorema de muestreo. Este establece que una
sen˜al cuya tasa de muestreo es F se puede reconstruir si contiene solamente compo-
nentes de frecuencia por debajo de la mitad: F/2. Esta frecuencia se conoce como
frecuencia de Nyquist [26]. Por ejemplo, los CD usan una tasa de muestreo de 44.100
Hz, por lo que la frecuencia de Nyquist en los CD es 22.050 Hz.
LaSTFTsesuelevisualizaratrav´esdelespectrograma,queesunarepresentaci´on
bidimensional de la magnitud al cuadrado:
θ(m,k) = |ξ(m,k)|2
Donde el eje horizontal representa el tiempo, el eje vertical representa la frecuencia y
la intensidad de color en la imagen representa el valor concreto de la frecuencia en un
punto.
2.3.3. MFCC
Los Mel Frequency Cepstral Coefficients (MFCC) permiten medir la forma del
espectro de frecuencias en la escala de Mel [52]. La escala de Mel es una escala que
intenta imitar la percepcio´n del o´ıdo humano y divide las frecuencias en intervalos
equidistantes. La siguiente funci´on transforma la frecuencia a la escala de Mel:
x
F (x) = 1127ln(1+ )
Mel
700
Para calcular los coeficientes, se calcula la STFT y se hace un mapeo entre la STFT y
la escala de Mel. Se utilizan una serie de filtros triangulares que son los que hacen esta
transformacio´n:
N−1
(cid:88)
S(n) = |ξ(k)|2 ·J (k)
m
k=0
Donde 0 ≤ m ≤ M − 1,m es el nu´mero de filtros triangulares y J (k) es el peso de
m
contribucio´n del bin k al filtro triangular m. Es decir, cu´anto aporta el bin k al filtro
m. Los bins son intervalos entre los samples en el dominio de frecuencias.
Una vez obtenida la energ´ıa espectral en la frecuencia de Mel, se puede calcular la
DiscreteCosineTransform(DCT)paraobtenerundeterminadonu´merodecoeficientes:
M−1
(cid:88) πn(m−0,5)
Λ (n) = S(n)·cos( )
m
M
m=0
Donde n ∈ N es el nu´mero de MFCC que se quieran. En concreto, en este proyecto
trabajaremos con 13 coeficientes de acuerdo con Ulloa [52].
6

| 2.3.4. | Chroma | Coefficients |     |     |
| ------ | ------ | ------------ | --- | --- |
Esta caracter´ıstica considera propiedades de la armon´ıa y la melod´ıa de la sen˜al.
Mientras que la melod´ıa es la secuencia de notas tocadas siguiendo un ritmo y en
diferentes intervalos, la armon´ıa resulta de la combinacio´n de la melod´ıa y los acordes
| (conjunto | de tres | o ma´s notas diferentes | tocadas | a la vez) [10]. |
| --------- | ------- | ----------------------- | ------- | --------------- |
En la figura 2.2 podemos observar un ejemplo simple para ver la diferencia entre
melod´ıa (ser´ıa lo que recordamos al tararear una canci´on) y la armon´ıa (es el resultado
| sonoro | de la interaccio´n | entre las | notas musicales) | [10]. |
| ------ | ------------------ | --------- | ---------------- | ----- |
Figura 2.2: Ejemplo de melod´ıa y armon´ıa. Imagen obtenida de [3].
En la mu´sica occidental, la mu´sica se divide en 12 tonos o pitches diferentes
{C,C♯,D,D♯,E,F,F♯,G,G♯,A,A♯,B},
que si traducimos del sistema de notacio´n anglosajo´n al latino [20] ser´ıa
{Do,Do♯,Re,Re♯,Mi,Fa,Fa♯,Sol,Sol♯,La,La♯,Si}
Esto se conoce como escala croma´tica [49], teniendo siete notas naturales y cinco alte-
raciones, ♯ y ♭, que elevan la nota un semitono o la disminuyen, respectivamente. En
total, en una octava hay 12 notas posibles, ya que notas como Do♯ y Re♭ son equiva-
lentes enarm´onicamente. Para calcular los coeficientes, se calcula la STFT y se obtiene
el espectrograma:
(cid:88)
|     |     | Y(n,p) | =   | |ξ(n,k)|2 |
| --- | --- | ------ | --- | --------- |
k∈P(p)
Donde p ∈ [0 : 127] es el nu´mero de la nota MIDI y representa el tono, y P(p) es una
funcio´n que mapea k a las notas MIDI. Las notas MIDI son una forma de dar una
descripcio´n de la altura de una nota a un instrumento musical electr´onico, codificando
cada nota como un nu´mero entre 0 y 127. Esto se puede codificar en 7 bits, y se podr´ıan
| cubrir | 11 octavas | [42]. |     |     |
| ------ | ---------- | ----- | --- | --- |
Una vez obtenido el espectrograma se pueden calcular los coeficientes sumando
| todos los | coeficientes | que pertenecen | al mismo | chroma: |
| --------- | ------------ | -------------- | -------- | ------- |
(cid:88)
|     |     | Ψ(n,c) | =   | Y(n,p) |
| --- | --- | ------ | --- | ------ |
p∈[0:127]:pm´od12≡c
| Siendo | c ∈ [0 : 11]. |     |     |     |
| ------ | ------------- | --- | --- | --- |
En nuestro sistema de recomendacio´n utilizaremos las cuatro caracter´ısticas de
audio descritas, por lo que ahora veremos qu´e modelos compondr´an dicho sistema.
7

2.4. Inteligencia Artificial
LaInteligenciaArtificialesunadisciplinadentrodelascienciasdelacomputaci´on
queconsisteendesarrollaruntipodealgoritmosqueimitenlainteligenciahumanapara
realizar tareas [23].
Estos algoritmos son capaces de extraer conclusiones a partir de unos datos de
entrada. El algoritmo o modelo aprende de esos datos de forma que, al introducir
nuevos datos, sea capaz de trabajar correctamente con ese nuevo dato.
Dentro de la inteligencia artificial tenemos el machine learning (ML) y el deep
learning (DL), como podemos observar en la figura 2.3. El machine learning se basa en
algoritmos que aprenden patrones a partir de datos etiquetados o no etiquetados para
tomar decisiones o hacer predicciones. Por su parte, el deep learning es una suba´rea
del ML que utiliza redes neuronales para abordar problemas complejos, como el reco-
nocimiento de voz, imagen o texto.
´
Figura 2.3: Areas de la Inteligencia Artificial. Imagen obtenida de [38].
Adema´s, es importante representar de forma num´erica conceptos complejos como
palabras, im´agenes o audio para que la IA pueda llegar a entenderlos. Aqu´ı es donde
entran en juego los embeddings, que permiten transformar estos datos en vectores de
caracter´ısticas que pueden ser procesados por modelos de ML y DL. Comprender estos
tres elementos —embeddings, machine learning y deep learning— sera´ necesario para
profundizar en el funcionamiento del sistema que se desarrolle.
2.4.1. Embedding
Un embedding es una representaci´on num´erica de los datos, ya sean im´agenes,
textooaudio,enunespaciomultidimensional.Estospermitenquelasm´aquinaspuedan
procesar de forma ma´s eficiente los datos para realizar tareas [18].
Estosembeddingsseconstruyendeformaquelosalgoritmosdemachinelearningo
deep learning sean capaces de entender las relaciones y patrones de los datos originales.
8

2.4.2. Machine Learning
Elmachinelearning(ML)esunsubconjuntodelaIAdeltipo“memorialimitada”.
Es decir, puede almacenar datos de entrada y los datos de todas las decisiones del
sistema y luego analizarlos para mejorar con el tiempo [23].
Hay tres tipos principales de algoritmos dentro del ML [23]:
Aprendizaje supervisado. En el aprendizaje supervisado una IA supervisa
activamente todo el proceso de aprendizaje. Se le proporcionar´a a la IA tanto
los datos que tendra´ que aprender, como los resultados que esos datos deber´ıan
producir. El objetivo del modelo es aprender una funcio´n que relacione entradas
con salidas de forma que pueda generalizar correctamente a nuevos datos no
vistos. Se puede utilizar aprendizaje supervisado para resolver los siguientes tipos
de problemas:
1. Clasificaci´on.Enlosproblemasdeclasificacio´nlaclaseobjetivoquesequiere
predecir es discreta (como “soleado”, “lluvioso” y “nublado” al intentar
predecir el tiempo).
2. Regresi´on. La clase objetivo a predecir es continua (como el precio de la
vivienda).
3. Series temporales. En las series temporales se utilizan datos observados en
intervalos regulares de tiempo (d´ıas, semanas, meses...) y sirven para hacer
predicciones en base a la tendencia observada en los datos (como intentar
predecir las temperaturas medias de los meses del siguiente an˜o a partir de
los 10 u´ltimos an˜os).
Un ejemplo de algoritmo supervisado ser´ıa k-NN (sera´ explicado en 2.4.2). Este
algoritmo puede ser usado tanto en problemas de clasificacio´n como de regresio´n.
Aprendizaje no supervisado: Implica que no hay intervencio´n humana duran-
te el aprendizaje. El modelo recibe unos datos de entrada e identifica patrones en
esos datos de forma independiente. Es especialmente u´til cuando no se dispone de
salidas esperadas o cuando los datos son demasiado costosos de etiquetar manual-
mente. Se puede utilizar aprendizaje no supervisado para resolver las siguientes
tareas:
1. Asociacio´n. En los problemas de asociacio´n queremos encontrar relaciones
frecuentes entre elementos en conjuntos de datos (como saber qu´e otros
productos se suelen comprar al comprar leche en el supermercado).
2. Clustering. Consiste en agrupar datos en grupos similares sin etiquetas pre-
vias (un ejemplo ser´ıa agrupar estudiantes cuyas condiciones de vida y sus
notas sean similares).
3. Resumen. Consiste en reducir o condensar datos manteniendo la mayor can-
tidad de informaci´on posible. Un ejemplo de algoritmo para esta tarea ser´ıa
PCA.
Aprendizaje por refuerzo: Es el ma´s complejo, ya que no se proporciona
ningu´n conjunto de datos para entrenar la IA, sino que el modelo debe aprender
9

interactuando con el entorno donde se encuentra. Se utilizan recompensas para
que vaya mejorando con el tiempo y maximice la cantidad de recompensas que
puede conseguir.
PCA
PrincipalComponentAnalysis(PCA)esunat´ecnicadeextracci´ondecaracter´ısti-
cas donde se combinan las entradas de una forma espec´ıfica y se eliminan algunas de
las variables menos importantes, manteniendo la parte m´as importante de todas las
variables. Asimismo, al utilizar PCA se consigue que todas las nuevas variables sean
independientes entre s´ı [6].
El algoritmo funciona de la siguiente forma:
Se estandarizan los datos de entrada restando la media de cada variable.
Se obtiene la matriz de covarianza. Esta matriz contiene estimados de co´mo cada
dimensio´n de los datos se relaciona con el resto.
Se obtienen los autovectores y autovalores de la matriz de covarianza. Cada au-
tovector indica una direccio´n del espacio (componente principal), y su autovalor
asociado representa cu´anta varianza de los datos es explicada en esa direcci´on.
Asimismo, a mayor variabilidad en una direccio´n en particular mayor informa-
cio´n, mientras que poca variabilidad indica ruido.
Se ordenan los autovalores de mayor a menor y se eligen los k autovectores que
se correspondan con los mayores autovalores.
Se construye la matriz de proyeccio´n W con los k autovectores seleccionados.
Se transforma el dataset original x usando la f´ormula y = W′·x, donde W′ es la
transpuesta de W [46].
k-NN
k-Nearest Neighbors, o k-NN, es un algoritmo de machine learning que usa la
comparacio´n de la proximidad de un punto con el resto de un conjunto de datos. Es un
algoritmo supervisado en el que k representa la cantidad de vecinos m´as cercanos que
se tienen en cuenta [13]. Para calcular esa proximidad se suelen utilizar estas funciones
de distancia:
La distancia euclidiana mide una recta entre el punto de referencia y el punto
a comparar.
La distancia de Manhattan mide el valor absoluto entre dos puntos. Se repre-
senta en una cuadr´ıcula, y su nombre viene del disen˜o de las calles de Manhattan,
que tambi´en forman una cuadr´ıcula.
La distancia de Minkowski es una generalizaci´on de la distancia euclidiana
y de Manhattan. Se define un para´metro p que define el tipo de distancia. Por
10

ejemplo, si p = 1, se usa la distancia de Manhattan y si p = 2, la distancia
euclidiana.
| 2.4.3. Deep | Learning |     |     |     |     |     |
| ----------- | -------- | --- | --- | --- | --- | --- |
El deep learning (DL) es un subconjunto del ML que intenta emular las redes
neuronales humanas. Pueden recibir, procesar y analizar grandes cantidades de infor-
macio´n no estructurada y utilizarlas para aprender sin intervencio´n humana.
Dentro de la gran variedad de arquitecturas que ofrece el DL, existen algunas
disen˜adas espec´ıficamente para extraer representaciones comprimidas y significativas
delosdatos.Unadelasma´sconocidasyutilizadasconestepropo´sitoeselautoencoder.
Autoencoder
Es una red neuronal que se entrena para comprimir y luego reconstruir datos de
entrada, aprendiendo as´ı una representaci´on m´as eficiente en un espacio ma´s reducido.
Consiste en dos partes: un encoder que reduce la dimensionalidad de los datos, forman-
do el conocido como espacio latente, y un decoder que reconstruye la entrada a partir
de los vectores latentes. En la figura 2.4 se puede ver la estructura de un Autoencoder.
|     | Figura | 2.4: Autoencoder. |     | Imagen | obtenida | de [45]. |
| --- | ------ | ----------------- | --- | ------ | -------- | -------- |
Como se puede observar tambi´en en la figura anterior, la fo´rmula de p´erdida que
| se suele utilizar | es el | Error Cuadra´tico | Medio   |        | (MSE):  |         |
| ----------------- | ----- | ----------------- | ------- | ------ | ------- | ------- |
|                   |       | ||x−xˆ||2         |         | (z)||2 |         | (x))||2 |
|                   | loss  | =                 | = ||x−d |        | = ||x−d | (e      |
|                   |       |                   |         | ϕ      |         | ϕ θ     |
11

donde x es la entrada, z es el vector latente, xˆ es la entrada reconstruida, e es el
θ
resultado de aplicar el encoder a la entrada y d es el resultado de aplicar el decoder
ϕ
| al vector | latente. |     |     |     |
| --------- | -------- | --- | --- | --- |
La idea es que la red neuronal aprenda a reducir los datos minimizando la di-
ferencia entre la entrada y la entrada reconstruida. Al minimizar estos errores, la red
aprendera´ una representaci´on latente compacta, es decir, una representacio´n que man-
| tenga | la mayor informaci´on | posible | [17]. |     |
| ----- | --------------------- | ------- | ----- | --- |
En la siguiente figura 2.5 se puede observar una representacio´n del espacio latente
para el dataset MNIST, que es un dataset que contiene im´agenes con d´ıgitos del 0 al 9
[53]. Se puede observar co´mo los puntos que corresponden a un mismo d´ıgito tienden a
ir en la misma direccio´n, perteneciendo al mismo clu´ster. Esto se observa muy bien en
casos como el nu´mero 0 o el 1, mientras que hay nu´meros que se superponen a otros
| clu´steres, | como el | 8 y el 9. |     |     |
| ----------- | ------- | --------- | --- | --- |
Figura 2.5: Espacio latente de Autoencoder. Imagen obtenida de [45].
Existenvariantesdeestaarquitectura,comolosVariationalAutoencoders(VAE),
que introducen conceptos probabil´ısticos en el espacio latente, permitiendo aplicaciones
| ma´s avanzadas | como        | la generacio´n | de nuevos | datos. |
| -------------- | ----------- | -------------- | --------- | ------ |
| Variational    | Autoencoder |                |           |        |
El Variational Autoencoder (VAE) extiende las capacidades del Autoencoder y
soluciona el hecho de que el espacio latente del Autoencoder no est´e regularizado [45].
Mientras que el autoencoder es un modelo determinista, pues codifica un u´nico
vector latente, el VAE es un modelo probabil´ıstico. Los VAE codifican las variables
12

latentes de los datos de entrenamiento no como un valor discreto fijo z, sino como un
rango continuo de posibilidades expresado como una distribucio´n de probabilidad p(z)
[25].
En la figura 2.6 se observa la estructura del VAE. Para cada variable, los VAE
calculandosvectoreslatentesdiferentes:unvectordemediasµyunvectordevarianzas
| σ.  | El VAE | tiene | las siguientes |     | partes: |     |     |     |     |     |
| --- | ------ | ----- | -------------- | --- | ------- | --- | --- | --- | --- | --- |
Encoder: El encoder calcula la probabilidad posterior p(z|x). Las salidas ser´an
un conjunto de medias y varianzas de logaritmos. Para estabilidad num´erica se
utiliza la varianza de logaritmos en lugar de la varianza directamente [21].
Decoder: El decoder calcula la probabilidad posterior p(x|z), tomando un vector
latente como entrada y generando los par´ametros para una distribuci´on condi-
|     | cional | de  | la observaci´on. |     |     |     |     |     |     |     |
| --- | ------ | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
Truco de la reparametrizaci´on: El truco de la reparametrizacio´n introduce
un nuevo par´ametro ϵ ∼ N(0,1). Este para´metro puede verse como un ruido
aleatorio para mantener la estoicidad de z [21]. Posteriormente, se reparametriza
la variable z como z = µ + σ ϵ. Esta variable z sera´ la que permita se pueda
|     |     |     |     |     | x   | x   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
optimizar mediante retropropagacio´n usando descenso del gradiente.
|     |     | Figura | 2.6: | Variational |     | Autoencoder. | Imagen | obtenida | de [45]. |     |
| --- | --- | ------ | ---- | ----------- | --- | ------------ | ------ | -------- | -------- | --- |
Asimismo, para minimizar la diferencia entre la entrada y la entrada reconstruida
| se  | utiliza | como | funcio´n | de p´erdida | dos | funciones: |     |     |     |     |
| --- | ------- | ---- | -------- | ----------- | --- | ---------- | --- | --- | --- | --- |
1. La funcio´n del error de la reconstruccio´n, que es el MSE, utilizada tambi´en en el
Autoencoder:
|     |       |                |     |        | ||x−xˆ||2 |     | (z)||2 |         |       | ϵ)||2 |
| --- | ----- | -------------- | --- | ------ | --------- | --- | ------ | ------- | ----- | ----- |
|     |       | reconstruction |     | loss   | =         | =   | ||x−d  | = ||x−d | (µ +σ |       |
|     |       |                |     |        |           |     | ϕ      |         | ϕ x   | x     |
|     | donde | µ ,σ           | = e | (x), ϵ | ∼ N(0,1). |     |        |         |       |       |
|     |       | x              | x θ |        |           |     |        |         |       |       |
13

2. La funcio´n del error de similitud, que es la divergencia de Kullback-Leibler o
divergencia KL entre la distribucio´n del espacio latente y la gaussiana:
|     | similarity | loss | = D (N(µ | ,σ ) || | N(0,1)) |
| --- | ---------- | ---- | -------- | ------- | ------- |
|     |            |      | KL x     | x       |         |
Sin embargo, al utilizar la divergencia KL para la inferencia variacional se tiene
que el denominador de la ecuacio´n es intratable, por lo que llevar´ıa una cantidad
de tiempo teo´ricamente infinita calcularlo directamente [25]. Para solucionar este
problema, los VAE se aproximan a la minimizaci´on de la divergencia KL maximi-
zando el l´ımite inferior de evidencia (ELBO). La ecuacio´n final que maximizar´ıa
| el ELBO | ser´ıa [55]: |     |     |     |     |
| ------- | ------------ | --- | --- | --- | --- |
1 (cid:88)
|     | similarity | loss = | − (1+log(σ2)−µ2 |     | −σ2) |
| --- | ---------- | ------ | --------------- | --- | ---- |
|     |            |        | 2               | j   | j j  |
j
| donde j | es la dimensio´n | del espacio | latente. |     |     |
| ------- | ---------------- | ----------- | -------- | --- | --- |
La figura 2.7 visualiza el espacio latente del VAE, que esta´ ma´s regularizado
que el del Autoencoder, pues no hay espacio entre los clu´steres (aunque hay cierto
solapamiento) y se asemeja a la distribucio´n gaussiana multivariante.
Figura 2.7: Espacio latente del Variational Autoencoder. Imagen obtenida de [45].
14

3. Estado del arte
3.1. Introducci´on
Los sistemas de recomendaci´on musical han ido evolucionando a lo largo de las
u´ltimas d´ecadas, siendo muy importantes en las plataformas de streaming y servicios
digitales.Conelaugedelmachinelearning,estossistemashanpasadodefiltradobasado
en contenido y filtrado colaborativo a modelos ma´s complejos que aprovechan t´ecnicas
de deep learning, procesamiento de sen˜ales y an´alisis de patrones de escucha.
Actualmente, los sistemas inteligentes de recomendaci´on musical emplean dife-
rentes t´ecnicas, como modelos basados en contenido, filtrado colaborativo y modelos
h´ıbridos, adema´s de t´ecnicas avanzadas como embeddings generados mediante redes
neuronales y m´etodos de clustering para identificar similitudes entre canciones y usua-
rios.
Este estado del arte analiza las principales t´ecnicas utilizadas en la literatura,
destacando los avances ma´s recientes en el uso de deep learning, sistemas h´ıbridos
y, principalmente, en modelos basados en sen˜ales musicales. Adema´s, se revisar´an los
puntosfuertesdeestossistemasylosdesaf´ıosactuales,comolaeficienciayladiversidad
en las recomendaciones.
3.2. Caracter´ısticas de audio
Antes de hablar de los sistemas de recomendacio´n musical, es importante co-
mentar una de las bases en las que se basan muchas aplicaciones de tema´tica musical
que utilizan modelos de machine learning. Estos sistemas suelen utilizar diferentes ca-
racter´ısticas, como el tempo o la melod´ıa, para poder realizar diversas tareas, como
clasificar el g´enero de una cancio´n o realizar recomendaciones.
Para entender qu´e caracter´ısticas y t´ecnicas se han utilizado, tambi´en se ha re-
visado literatura relacionada con la clasificaci´on de canciones segu´n el g´enero. Las
caracter´ısticas principales utilizadas son:
STFT (Short Time-Frequency Transform): Es una Transformada de Fou-
rier diferente a la tradicional que descompone la sen˜al de audio en ventanas de
tiempo ma´s cortas y calcula la FFT (Fast Fourier Transform) a cada una de ellas.
As´ı, es posible obtener un conjunto de coeficientes complejos que representan las
frecuencias activas en cada instante de tiempo [1].
Chroma vectors: Es un vector doce-dimensional donde cada dimensi´on repre-
senta una clase de pitch que se calcula a partir de la STFT. El pitch es la per-
cepcio´n subjetiva de la frecuencia de un sonido, es decir, lo agudo o grave que
percibimos el sonido. Es un t´ermino complejo de entender debido a que dos notas
15

en dos instrumentos diferentes pueden tener la misma frecuencia fundamental,
por ejemplo un La a 440 Hz, pero percibirse con diferente timbre [58].
MFCC (Mel Frequency Cepstral Coefficients): Son coeficientes que repre-
sentan las caracter´ısticas esta´ticas del audio en la Escala Mel, que se considera la
aproximacio´n ma´s cercana a la percepcio´n auditiva humana [1]. Estos coeficientes
se calculan a partir de la STFT, y opcionalmente se puede an˜adir las deltas o las
deltas-deltas, que representan la velocidad y la aceleracio´n con la que cambian
los MFCC [56].
BPM (Beats Per Minute): Es una medida que define el tempo de un audio
contandoelnu´merodebeats(baser´ıtmica)quecabenenunminuto.Porejemplo,
ˇ“
= 60 indica que en un minuto caben 60 negras, por lo que cada negra ocupa 1
segundo.
ZCR (Zero crossing rate): Es una caracter´ıstica que cuenta cua´ntas veces
la onda cruza el eje X [37]. As´ı podemos observar la suavidad de la onda, y
la principal utilidad de esta caracter´ıstica es distinguir entre sonidos agudos y
sonidos de percusio´n. Asimismo, hay diferencias entre g´eneros musicales, pues
una cancio´n de un g´enero como el metal tendra´ un ZCR mayor a una canci´on de
blues [12].
Estas propiedades, us´andolas de forma combinada, podr´an permitirnos recoger
la informaci´on m´as importante de los audios de las canciones para construir el sistema
de recomendacio´n.
3.3. Sistemas de recomendaci´on
Los sistemas de recomendaci´on musical son sistemas que permiten mostrar al
usuario canciones, a´lbumes, artistas o listas de reproduccio´n en funci´on de sus prefe-
rencias, historial de escucha o caracter´ısticas de las canciones.
Recomendada a
A
A
Escuchada
C D
por B
B
Figura 3.1: Ilustraci´on de sistemas basados en filtrado colaborativo
En general, un sistema de recomendaci´on compara el perfil de un usuario con
algunas caracter´ısticas de referencia. Las t´ecnicas ma´s utilizadas son el filtrado colabo-
rativo y el filtrado basado en contenido. El filtrado colaborativo utiliza la l´ogica de que
16

a los usuarios similares les gustara´n las mismas canciones. Dos usuarios ser´ıan similares
si escuchan o les gustan canciones parecidas. En la figura 3.1 podemos observar dicha
lo´gica. Si al usuario A y B les gusta la cancio´n C, es probable que a A le guste la
cancio´n D porque tambi´en le gusta al usuario B.
Por otra parte, el filtrado basado en contenido utiliza la lo´gica de que si a un
usuario le ha gustado o ha escuchado una canci´on, podr´ıan gustarle canciones similares.
Estos sistemas se basan exclusivamente en las preferencias pasadas del usuario, en
contraposicio´n con el sistema anterior que se basa en las preferencias de otros usuarios.
En la figura 3.2 podemos observar dicha l´ogica: al usuario A le ha gustado la cancio´n
B, y las canciones C y B son parecidas. Por ello, es probable que a A tambi´en le guste
C.
A escucha B
B
Similares
A
Se recomienda C
C
Figura 3.2: Ilustraci´on de sistemas basados en filtrado basado en contenido
Para calcular la similitud entre dos canciones se suele utilizar la similitud coseno
o la distancia eucl´ıdea. La similitud coseno es el coseno del a´ngulo que forman dos
vectores, de forma que si el coseno es cercano a 1, los vectores son muy similares, y si
es cercano a -1, son muy diferentes. En cambio, la distancia eucl´ıdea mide la distancia
entre dos puntos vectoriales; cuanto menor sea esta distancia, m´as similares son los
vectores.
Muchos sistemas de recomendaci´on utilizan ambas t´ecnicas; son los conocidos
como sistemas h´ıbridos.
3.3.1. Algoritmos utilizados en la literatura
Para implementar los sistemas de recomendaci´on, se suelen utilizar diversos al-
goritmos de machine learning. Algunos de los ma´s destacados en la literatura son:
SVM (Support Vector Machine): Algoritmo de clasificaci´on u´til para carac-
ter´ısticas de grandes dimensiones que es capaz de trabajar con datos no lineales.
Tiene gran utilidad a la hora de encontrar patrones y hacer recomendaciones
basadas en las emociones del usuario [8].
17

KNN (k-Nearest Neighbour): Algoritmo de clasificacio´n que utiliza la etique-
ta asociada a los K vecinos m´as cercanos para determinar la etiqueta del ejemplo
actual. Este algoritmo parte de la idea de que los puntos cercanos en el espacio
vectorial ser´an m´as similares entre s´ı. Este algoritmo ha sido utilizado para la
clasificacio´n de g´eneros [1].
Logistic Regression: Algoritmo de clasificaci´on para predecir una variable ca-
tego´rica en funcio´n de las variables predictoras [1]. Este algoritmo fue utilizado
para predecir la emoci´on del usuario para recomendarle mu´sica segu´n su estado
de ´animo [8].
Random Forest: Algoritmo ampliamente utilizado que combina mu´ltiples ´arbo-
les de decisio´n para llegar a un resultado con un rendimiento similar a la del
boosting [8].
A su vez, han ido surgiendo sistemas m´as complejos que utilizan modelos de
deep learning para intentar detectar nuevos patrones. Algunos de dichos algoritmos o
t´ecnicas son los siguientes:
Multilayer Perceptron: Red neuronal artificial que soluciona los problemas
del perceptr´on simple y es capaz de resolver problemas que no son linealmente
separables [1] [37].
Recurrent Neural Networks (RNN): Red neuronal profunda entrenada con
datos secuenciales o de series temporales para crear un modelo que pueda hacer
predicciones secuenciales o conclusiones basadas en entradas secuenciales [24] [1].
Convolutional Neural Networks (CNN): Red neuronal que usa datos tridi-
mensionales para tareas de clasificacio´n y aprendizaje de patrones en ima´genes
[33].
Embedding:T´ecnicaparatransformarlosdatosinicialesendatosnum´ericosque
permita que puedan ser manipulados matem´aticamente, y as´ı poder encontrar la
similitud entre los datos [57].
Autoencoder: Red neuronal que se entrena para comprimir datos y posterior-
mente reconstruirlos, por lo que es capaz de detectar patrones y caracter´ısticas
de los datos trabajando con un espacio de menor dimensionalidad [57].
3.3.2. Nuevos enfoques en los sistemas de recomendacio´n
Dejando de lado los algoritmos utilizados, tambi´en se han explorado nuevos tipos
de sistemas de recomendacio´n. El primer ejemplo de esto ser´ıa un sistema basado en
emociones [8]. Este sistema utiliza Natural Language Processing (NLP) para predecir
el estado de a´nimo del usuario, de forma que se le recomiende mu´sica que tenga como
etiqueta ese mismo estado en la base de datos. No es el primer sistema que intenta
recomendar mu´sica basado en las emociones del usuario, pero s´ı es un sistema muy
poco intrusivo. Una persona solo debe escribir c´omo se siente y el sistema se encargar´a
de predecir su estado de a´nimo y la mu´sica que deber´ıa escuchar.
18

Otro ejemplo ser´ıa un Cross-domain Recommendation System (CDRS), disen˜ado
para abordar problemas como la escasez de datos, el cold start y la falta de diversidad
en las recomendaciones [57]. Estos sistemas aprovechan el conocimiento de dominios
fuente con mayor cantidad de informaci´on para mejorar las recomendaciones en domi-
nios con datos limitados. Un ejemplo de caso pr´actico de estos sistemas podr´ıa ser la
integracio´n de informaci´on proveniente de redes sociales, donde se analicen las interac-
ciones y los intereses de los usuarios para enriquecer las recomendaciones de contenido
en plataformas de streaming musical.
Otroenfoqueenlossistemasderecomendaci´onmusicalsebasaenelusodehuellas
digitalesdeaudio(audiofingerprints)paraidentificarsimilitudesentrecanciones.Ulloa
[52]propusolacombinacio´ndediferentescaracter´ısticasdeaudio,comolosCoeficientes
Cepstrales en las Frecuencias de Mel (Mel Frequency Cepstral Coefficients, MFCC),
los chroma coefficients o el tempo, para construir un vector de alta dimensio´n que
representa la huella digital de una cancio´n. Luego, se aplic´o Ana´lisis de Componentes
Principales (PCA) para reducir la dimensionalidad y calcular una matriz de similitud
entre canciones. La evaluaci´on del sistema mostr´o una precisi´on del 89% al recomendar
canciones del mismo g´enero que la cancio´n objetivo, lo que sugiere que este m´etodo
puede ser eficaz para la recomendaci´on basada en contenido.
Por otro lado, tendr´ıamos los sistemas de recomendaci´on secuencial, que han ga-
nado inter´es en los u´ltimos an˜os, especialmente en plataformas de streaming de mu´sica.
Un estudio reciente presento´ Psychology-Informed Session embedding using ACT-R
(PISA), un sistema que combina modelos de Transformers con la arquitectura cogniti-
va Adaptive Control of Thought-Rational (ACT-R) para capturar patrones din´amicos
y repetitivos en el comportamiento de los usuarios [50]. Este modelo considera la im-
portancia de la repetici´on en la escucha musical y genera recomendaciones que reflejan
tanto las preferencias cambiantes del usuario como su tendencia a volver a escuchar
canciones anteriores.
Finalmente, otra l´ınea de investigaci´on en sistemas de recomendacio´n musical
combina t´ecnicas de clasificacio´n de g´eneros con caracter´ısticas extra´ıdas directamen-
te de las canciones [28]. Este integra tres factores principales: clasificacio´n de g´enero
mediante una red neuronal, MFCC y el tempo de la cancio´n. Se evaluaron cuatro es-
trategias diferentes de recomendacio´n, donde se ponderaban estos factores de manera
distinta. Los resultados mostraron que los MFCC ten´ıan el mayor impacto en la calidad
de las recomendaciones, seguidos por la clasificacio´n de g´enero y el tempo.
Aunque el uso de deep learning ha permitido que se desarrollen sistemas ma´s
refinados, Ndou et al. [37] consiguieron superar el rendimiento de los algoritmos de
deep learning usando machine learning. Aunque el objetivo del art´ıculo es diferente al
de este trabajo, es relevante tenerlo en cuenta y abre puertas a realizar comparaciones
entre algoritmos de diferente tipo o a priorizar los algoritmos de machine learning.
Un enfoque muy interesante es el de los autoencoders, que permiten aprender
una representacio´n de los datos de entrada, reduciendo su dimensionalidad y recons-
truy´endolos para minimizar la diferencia entre la entrada y la reconstruccio´n. Tiene
una funcionalidad parecida a PCA, con la ventaja de que un autoencoder involucra
operaciones no lineales con una reducci´on mayor de la dimensionalidad [29]. Quijada
[40] propone un sistema de recomendaci´on basado en contenidos utilizando autoenco-
19

ders de 5 y 7 capas, y Saini and Singh [43] propone un sistema basado en contenidos
combinando un stacked Long Short-Term Memory (LSTM) y un autoencoder basado
en atenci´on.
Un enfoque generativo similar al de los Variational Autoencoders (VAE) es el de
Chen et al. [9], donde proponen un sistema de recomendacio´n basado en un modelo de
difusio´n condicionado. Otro sistema interesante es el de Salas [44], donde se utilizan
redes siamesas en diferentes a´mbitos.
Asimismo, la bibliograf´ıa que m´as impacto tendra´ en el sistema de recomendaci´on
que se implementara´ sera´n los estudios de Kostrzewa et al. [28], Ulloa [52], Quijada [40]
y Saini and Singh [43]. Esto se debe a que dependen directamente de caracter´ısticas
objetivas, obtenidas a trav´es del audio de la cancio´n y a que utilizan metodolog´ıas
directamente aplicables al objetivo del trabajo. Otros sistemas como el de Behura
et al. [8] dependen de la emoci´on con la que se haya catalogado cada cancio´n de la
base de datos, lo que es una propiedad bastante subjetiva. Por otra parte, replicar un
sistema como el que propone Zhu et al. [57] ser´ıa muy complicado al necesitar datos de
diferentes dominios que se pudieran combinar, y para replicar el estudio de Tran et al.
[50] y Chen et al. [9] se necesitar´ıan tanto datos de las canciones como de los usuarios
en plataformas de streaming o p´aginas musicales.
3.4. Conclusiones
En este cap´ıtulo se ha revisado la evolucio´n y el estado actual de los sistemas de
recomendacio´n musical, identificando las principales t´ecnicas utilizadas en la literatura
y los avances m´as recientes en la personalizacio´n de recomendaciones.
Se ha constatado que los m´etodos tradicionales de filtrado basado en conteni-
do y filtrado colaborativo siguen siendo primordiales en estos sistemas. Sin embargo,
la integraci´on de t´ecnicas ma´s avanzadas, como el uso de embeddings generados con
deep learning, modelos h´ıbridos y enfoques basados en sen˜ales musicales, ha permitido
mejorar la precisi´on y la adaptabilidad de las recomendaciones.
El ana´lisis de las caracter´ısticas de audio ha demostrado que propiedades como
los MFCC, los vectores crom´aticos y la STFT son esenciales para extraer informaci´on
relevante y objetiva de las canciones. Estas caracter´ısticas han sido ampliamente utili-
zadas en los sistemas de recomendacio´n, as´ı como en tareas de clasificacio´n de g´eneros
musicales y detecci´on de patrones.
Asimismo, se han identificado nuevas tendencias en la literatura, como los siste-
mas de recomendaci´on basados en emociones, los CDRS, el uso de huellas digitales de
audio y los sistemas secuenciales que incorporan modelos cognitivos. Estas estrategias
han permitido abordar desaf´ıos en la recomendacio´n musical, como el problema del
cold start, la falta de diversidad y la adaptacio´n a los cambios en las preferencias de
los usuarios.
Finalmente, se destaca que, aunque el deep learning ha impulsado avances signi-
ficativos en la personalizaci´on de recomendaciones, algunos estudios han demostrado
que algoritmos cla´sicos de machine learning pueden ser igualmente efectivos en ciertos
20

contextos. Adema´s, la eleccio´n del enfoque dependera´ en gran medida de los datos
utilizados, la interpretabilidad del modelo y la eficiencia computacional.
Los estudios revisados en este cap´ıtulo servira´n de base para el desarrollo del
sistema de recomendaci´on que se implementar´a en este trabajo, priorizando aquellos
enfoques que utilizan caracter´ısticas extra´ıdas directamente del audio y m´etodos de
reduccio´n de dimensionalidad para la mejora de la precisio´n en la recomendacio´n.
21

4. Estudio previo
4.1. Introducci´on
En este cap´ıtulo presentaremos los objetivos, la metodolog´ıa, la planificaci´on y
el presupuesto del proyecto. Se abordara´n aspectos clave para la consecucio´n de los
objetivos del trabajo.
4.2. Objetivos
Este trabajo final de ma´ster explora las tendencias de los sistemas de recomenda-
cio´n musicales, con el objetivo de implementar un sistema que utilice algunos de estos
enfoques y demuestre su utilidad para detectar las necesidades e intereses musicales de
los usuarios. Los diferentes objetivos a cumplir son los siguientes:
1. Revisi´on de la literatura: Se realizar´a una revisi´on de la literatura con el fin de
identificar, clasificar y analizar las principales t´ecnicas de recomendaci´on musical.
Esta revisio´n incluir´a m´etodos cla´sicos y actuales de Machine Learning y Deep
Learning, as´ı como m´etricas utilizadas para evaluar su rendimiento.
2. Disen˜o y construcci´on del sistema de recomendaci´on: Se disen˜ara´ y cons-
truira´ un sistema de recomendacio´n musical que utilice representaciones vecto-
riales derivadas de caracter´ısticas de audio. Para ello, se aplicar´an t´ecnicas de
Machine Learning (como PCA y k-NN) y Deep Learning (como Autoencoders y
Variational Autoencoders).
3. Comparacio´n y evaluacio´n: Comparar y evaluar cuantitativamente el rendi-
miento de las diversas t´ecnicas implementadas. Adema´s, se analizara´ el impacto
de diferentes m´etricas de similitud y se discutir´an los resultados obtenidos.
4.3. Metodolog´ıa
A continuaci´on, se va a comentar la metodolog´ıa utilizada a lo largo del proyecto.
Como se ha comentado en los apartados anteriores, se va a realizar un sistema de
recomendacio´n musical basado en diferentes algoritmos de machine learning y deep
learning. En la figura 4.1 se ven los pasos seguidos en la metodolog´ıa.
22

Inicio Final
Documentación del
proceso
PCA
MagnaTagATune
AE
Elección del dataset GTZAN p E a le r c a c im ión p l d em el e e n n t t a o c r i n ó o n P V y S th C o o n d e y Elección de algoritmos Pre d p a r t o o c s e e s n a m S i Q e L n i t t o e de Ent e r v e a n l a u m ac ie ió n n to y
VAE
FMA
k-NN
Figura 4.1: Metodolog´ıa del proyecto
4.3.1. Documentacio´n
Todo el proceso de investigacio´n e implementaci´on del sistema sera´ documentado
de forma exhaustiva y paralela al desarrollo del proyecto. Se utilizar´a Overleaf como
editor LaTeX para esta tarea, principalmente por su amplio uso en la redacci´on de
documentos cient´ıficos, su comodidad para estructurar la informaci´on y su facilidad
para utilizar plantillas y crear entornos colaborativos frente a otros editores de texto
deusogeneralcomoWord.Asimismo,seutilizar´aGPT-4comoapoyoparalaredacci´on,
mejorando la calidad de los textos y la profesionalidad de estos.
4.3.2. Elecci´on del dataset
Laelecci´ondeldataseteselprimerpasoparapoderplanteareldisen˜odelsistema.
Para ello, es necesario que los datos cumplan ciertos requisitos:
Ficheros de audio. Hay datasets que contienen la informacio´n ya procesada de
los audios, lo que puede llegar a limitar el desarrollo si se quiere llegar a utilizar
caracter´ısticas diferentes del audio. Por ello, es recomendable poder tener los
ficheros de audio directamente, ya sea en formato WAV o MP3. Asimismo, si se
llegara a desplegar el sistema ser´ıa recomendable que el usuario pudiera escuchar
las canciones recomendadas, y para ello se necesitar´ıan los ficheros.
Metadatos de canciones. Tambi´en hay datasets que cumplen el apartado an-
terior pero que no dan informacio´n ninguna sobre el nombre de la cancio´n o el
artista. Para poder dar recomendaciones con cierto sentido ser´ıa importante que
tambi´en este tipo de informaci´on.
23

Taman˜o manejable. Para garantizar que los modelos hayan sido entrenados
con una cantidad suficiente de datos sin que el rendimiento de estos se vean
afectados ser´a importante no elegir un dataset demasiado grande. En el caso de
que el dataset escogido tenga un mal rendimiento se podr´ıa intentar reducir el
taman˜o, pero esto podr´ıa provocar un desbalanceo de los datos si no se hiciera
correctamente.
De acuerdo con los puntos anteriores, se analizaron tres datasets:
GTZAN: Dataset que contiene 1.000 pistas de audio de 30 segundos cada una.
Hay 10 g´eneros, 100 pistas por cada g´enero y ocupa 1GB [51].
MagnaTagATune: Dataset cuyos datos fueron obtenidos a trav´es del juego Ta-
gATune y las canciones de Magnatune. Contiene 3GB de fragmentos de canciones
de 30segundos. Adem´as,tiene variosficheros conmetadatos de las canciones [31].
FMA: Dataset de gran escala que tiene diferentes versiones segu´n el taman˜o.
La versi´on del dataset ma´s pequen˜a ocupa 7GB y contiene 8.000 fragmentos de
canciones balanceados para 8 g´eneros diferentes [11].
De estos tres, se descarto´ el primero porque no conten´ıa informaci´on relativa a los
metadatos de las canciones, y el tercero ten´ıa un taman˜o m´ınimo de 7 GB y m´aximo
de 879 GB, dejando como mejor candidato el dataset de MagnaTagATune.
Este dataset contiene 3 GB de audios de diferentes canciones, fraccionados en
fragmentos de 30 segundos de duracio´n, as´ı como ficheros con los metadatos de cada
una de las canciones. Asimismo, contiene una gran variedad de etiquetas que aportan
bastante informacio´n que sera´ u´til m´as adelante.
4.3.3. Implementacio´n
Para el desarrollo del co´digo del sistema de recomendacio´n se utilizar´a un IDE
(Entorno de Desarrollo Integrado). Este entorno se elegir´a teniendo en cuenta el len-
guaje y las librer´ıas que se vayan a utilizar.
En primer lugar, los dos lenguajes de programaci´on ma´s utilizados en el campo
de Data Science son R y Python. R es un lenguaje espec´ıfico de dominio (DSL) amplia-
mente utilizado en estad´ıstica y ana´lisis de datos, pero que se ve superado por Python
al tener este una mayor cantidad de bibliotecas para machine learning y mayor soporte
para integrarse con otros lenguajes y frameworks, o bases de datos SQL y NoSQL. Por
estos motivos, es que se utilizar´a Python como lenguaje de desarrollo.
A su vez, tenemos la librer´ıa librosa, ampliamente utilizada en la literatura para
el procesamiento de audio, que es otra ventaja que tendr´ıa utilizar Python frente a R.
Una vez decidido el lenguaje, tenemos que decidir el entorno. Algunos de los ma´s
utilizados ser´ıan Eclipse, Visual Studio Code y PyCharm.
Eclipse: Este entorno cuenta con el plugin PyDev que permite el desarrollo en
Python. Sin embargo, es un entorno m´as complejo y menos intuitivo que otros
entornos ma´s optimizados para Python [19].
24

Visual Studio Code: VSCode es un editor ligero con soporte para Python a
trav´es de extensiones. Tiene compatibilidad con Jupyter Notebooks y con entor-
nos virtuales, lo que lo convierte en una buena herramienta para Data Science
[32].
PyCharm: Es un IDE de pago espec´ıficamente disen˜ado para Python que cuen-
ta con un plan gratuito para estudiantes. Incluye herramientas avanzadas para
ana´lisis de co´digo y bibliotecas de Data Science y machine learning [27].
Por otra parte, tendr´ıamos Google Colab, que es un servicio alojado de Jupyter
Notebook que no requiere configuraci´on.
De entre estos entornos, VSCode y PyCharm son notablemente mejores para
el objetivo del proyecto. Por lo tanto, y por tener una mayor experiencia en su uso,
utilizaremosVSCode.Encasodequeeltiempodeejecuci´ondelosmodelosseaexcesivo,
se probar´an los modelos en Google Colab.
4.3.4. Elecci´on de algoritmos
Para la implementacio´n del sistema de recomendaci´on se utilizara´n t´ecnicas de
Machine Learning y Deep Learning, aunque se dara´ m´as relevancia al Deep Learning.
Se utilizar´an principalmente algunos de los algoritmos vistos en la literatura,
explicados en la seccio´n 2. Los algoritmos de ML que se usara´n son PCA y k-NN, y de
DL se usara´n Autoencoders y Variational Autoencoders.
Aunque los algoritmos son un paso importante, no se debe descuidar la calidad
de los datos. Segu´n el concepto “Garbage in, Garbage out (GIGO)”, la calidad de los
datos de salida de los algoritmos dependera´ principalmente de la calidad de los datos
de entrada. Esto sucede incluso si la l´ogica de los algoritmos es precisa. Por ello, tanto
la calidad de los datos de entrada como la de los algoritmos son igual de importantes
[7].
4.3.5. Procesamiento de datos
Una vez elegido el dataset, es necesario establecer la estrategia a seguir para el
procesamiento de los datos. Como se explico´ en el apartado anterior, hay 3 GB de
audios de canciones en fragmentos de 30 segundos y ficheros con los metadatos de las
canciones.
Las canciones vienen comprimidas en formato ZIP y los datos en CSV. Por lo
tanto, se deber´a tener en cuenta c´omo se almacenara´n los datos, pues esto condicionar´a
la forma de implementar el sistema.
Para ello, hay dos opciones, una de ellas ser´ıa trabajar directamente con los
ficheros como se han descargado. As´ı pues, solo se necesitar´ıa tener un directorio donde
guardarlosficherosydescomprimirlascanciones.Ladesventajadeestoser´ıaquehabr´ıa
que trabajar con dataframes muy grandes y no tendr´ıamos un mecanismo sencillo de
JOIN como en una base de datos (BBDD) relacional.
25

Teniendo esto en cuenta, la otra opcio´n ser´ıa trabajar con una BBDD y un alma-
cenamiento interno o externo. La BBDD podr´ıa ser relacional o no relacional. Las tres
alternativas de BBDD planteadas son:
1. SQLite: Base de datos relacional ligera que permite realizar peticiones de forma
ra´pida porque no necesita un servidor dedicado. Al no tener que configurar un
servidor, se utilizan simplemente archivos “.db”. No escala bien si hay muchas
consultas simulta´neas.
2. PostgreSQL: Sistema relacional m´as robusto que puede realizar consultas com-
plejas y puede escalar mejor que SQLite.
3. MongoDB: Base de datos no relacional que almacena los datos en formato
BSON (Binary JSON) en lugar de tablas con filas y columnas. La principal ven-
tajaespoderescalarlaslecturasr´apidamente,perotieneproblemassisenecesitan
consultas SQL avanzadas.
Como se realizar´a el sistema en un entorno local, se intentar´a buscar la solucio´n
ma´s sencilla. Por ello, se utilizara´ en un principio SQLite. Como no se necesita confi-
guracio´n ninguna, podra´ ser reemplazada fa´cilmente por otra BBDD si su rendimiento
llega a ser ineficiente ma´s adelante.
Asimismo, se pueden utilizar dos tipos de almacenamiento para los audios:
1. Interno: Es la forma m´as sencilla de almacenar los audios, utilizando el sistema
de ficheros para tener un directorio donde est´en todas las canciones.
2. Externo: Es una forma ma´s compleja de almacenamiento, pues se tendr´ıa que
utilizar Azure Blob Storage o Amazon S3, y esto an˜adir´ıa problemas como la
latencia o almacenar temporalmente las canciones descargadas para entrenar los
modelos. Sin embargo, permitir´ıa reducir de forma significativa el taman˜o del
sistema, al no necesitar tener descargado todo el tiempo los 3 GB de canciones.
Al igual que con las BBDD, se elegira´ la soluci´on ma´s sencilla de implementar;
por ello, se utilizara´ el almacenamiento interno para las canciones.
Paratrabajarconlascanciones,seprocesar´anlosaudiosde30segundosusandola
librer´ıa librosa y guardando los resultados de las diferentes caracter´ısticas extra´ıdas en
ficheros separados. De esta forma, tendr´ıamos, por una parte, los audios originales; por
otra, las caracter´ısticas, es decir, la informacio´n relevante; y, por otra, los metadatos.
4.3.6. Entrenamiento y evaluaci´on
Para el entrenamiento y evaluacio´n de los modelos se dividira´ el dataset en dos
conjuntos: un 80% de los datos para entrenamiento y el 20% restante para evaluaci´on.
Esta t´ecnica permite garantizar que los datos de entrenamiento y de prueba sean inde-
pendientes. Sin embargo, hay que tener en cuenta que el dataset viene preparado para
poder utilizar K-fold Cross Validation, teniendo separadas en diferentes carpetas (don-
de cada una corresponder´ıa a un pliegue o ‘fold‘) una parte equilibrada de las canciones
totales. El problema est´a en que en el preprocesamiento de los datos vamos a tener
que eliminar canciones enteras que est´en corruptas o que tengan muchos fragmentos
26

sin g´enero, y esto acabar´ıa haciendo que las carpetas no tuvieran una cantidad similar
de fragmentos. Por ello, para evitar evaluar el modelo de forma desequilibrada se utili-
zara´n estos conjuntos de entrenamiento y evaluacio´n. Una vez terminada la evaluacio´n,
se entrenara´n los modelos finales con todo el dataset.
4.4. Planificaci´on
En un primer momento, se decidio´ planificar el proyecto utilizando una metodo-
log´ıa iterativa e incremental. Como el sistema que se va a desarrollar depender´a de la
literatura revisada, las fases del proyecto se repetira´n en cada iteracio´n.
El objetivo al utilizar este marco de trabajo es que en cada fase se tengan nuevos
modelos, y poder realizar mejoras sobre los modelos ya existentes.
Se dividira´ el proyecto en iteraciones, cada una con una serie de fases, con las
tareas a realizar y una duraci´on estimada.
4.4.1. Cronograma inicial
En un inicio, se estimaron 2 iteraciones repartidas a lo largo del proyecto. Al
planificar, el objetivo es que se termine el proyecto para la convocatoria de junio de
2025.
Antes de la planificaci´on de las iteraciones se realizo´ una fase previa en diciembre
de 2024 donde se plante´o la metodolog´ıa y se empez´o el estado del arte. Esta fase
comenzo´ una vez adjudicados el tutor y la tema´tica del trabajo, y acabo´ a comienzos
de enero.
La primera iteracio´n servir´ıa para tener un sistema inicial y la segunda servir´ıa
como revisi´on para an˜adir nuevos modelos o t´ecnicas de evaluacio´n.
En la primera iteracio´n se realizar´ıan estas tareas:
Estado del arte: Se hara´ una revisio´n que servira´ para las fases posteriores
y condicionar´a tanto el disen˜o como la construccio´n del sistema. Esta revisi´on
incluira´ fundamentos sobre los sistemas de recomendaci´on y t´ecnicas de machine
learning y deep learning utilizadas en estos, dando mayor importancia a aquellas
utilizadas en recomendaciones musicales y explorando los l´ımites y las oportuni-
dades de estas.
Disen˜o: Una vez priorizada la literatura relacionada, se definir´a el entorno tec-
nolo´gico, las herramientas y librer´ıas a utilizar y la arquitectura del sistema de
recomendacio´n.
Implementacio´n: Se implementar´an y evaluar´an los modelos del sistema di-
sen˜ado. La implementaci´on incluir´a tanto la obtencio´n y preprocesamiento del
dataset como el entrenamiento de los modelos. Para la evaluacio´n se debera´ uti-
lizar m´etricas adecuadas que reflejen el rendimiento de las recomendaciones.
27

Como la segunda iteraci´on servir´ıa como revisio´n, sera´ una iteraci´on m´as corta al
tener menor grueso de trabajo. Se realizar´ıan las mismas tareas que en la anterior:
Estado del arte: Se revisar´a tanto la literatura de la iteracio´n anterior como
nueva literatura. Se intentar´a buscar nuevas t´ecnicas que complementen las ya
implementadas para an˜adir complejidad al sistema y poder hacer ma´s compara-
|     | ciones | del rendimiento |     | del | sistema | si  | se ve necesario. |     |     |     |
| --- | ------ | --------------- | --- | --- | ------- | --- | ---------------- | --- | --- | --- |
Disen˜o: Se modificar´ıa el disen˜o acorde con la nueva literatura.
Implementacio´n: Se construir´ıan y evaluar´ıan los nuevos modelos.
La redacci´on se har´ıa de forma transversal a lo largo del proyecto, construyendo
| por | secciones | el trabajo      | realizado.       |     |          |          |          |                |          |       |
| --- | --------- | --------------- | ---------------- | --- | -------- | -------- | -------- | -------------- | -------- | ----- |
|     |           | Fase            |                  |     | Per´ıodo |          |          | Estimacio´n    | de       | Horas |
|     |           | Fase previa     |                  | dic | 2024     | - ene    | 2025     |                |          | 30 h  |
|     |           | Iteracio´n      | 1                |     | ene      | - mar    | 2025     |                |          | 100 h |
|     |           | Estado          | del arte         |     | ene      | 2025     |          |                |          | 40 h  |
|     |           | Disen˜o         |                  |     | feb      | 2025     |          |                |          | 15 h  |
|     |           | Implementacio´n |                  |     | feb -    | mar      | 2025     |                |          | 45 h  |
|     |           | Iteracio´n      | 2                |     | abr -    | may      | 2025     |                |          | 70 h  |
|     |           | Estado          | del arte         |     | abr      | 2025     |          |                |          | 30 h  |
|     |           | Disen˜o         |                  |     | abr      | 2025     |          |                |          | 10 h  |
|     |           | Implementacio´n |                  |     | abr -    | may      | 2025     |                |          | 30 h  |
|     |           | Redaccio´n      |                  | dic | 2024     | - may    | 2025     |                |          | 100 h |
|     |           | Total           |                  |     |          |          |          |                |          | 300 h |
|     |           | Cuadro          | 4.1: Estimaci´on |     |          | inicial  | de horas | por fase del   | proyecto |       |
|     | A         | continuacio´n,  | podemos          |     | ver el   | diagrama | de       | Gantt inicial: |          |       |
28

|     |     | 2024 |       | 2025     |       |
| --- | --- | ---- | ----- | -------- | ----- |
|     |     | 11   | 12 01 | 02 03 04 | 05 06 |
Fase previa
|     | Iteracio´n | 1    |     |     |     |
| --- | ---------- | ---- | --- | --- | --- |
|     | Estado del | arte |     |     |     |
Disen˜o
Implementacio´n
|     | Iteracio´n | 2    |     |     |     |
| --- | ---------- | ---- | --- | --- | --- |
|     | Estado del | arte |     |     |     |
Disen˜o
Implementacio´n
Redaccio´n
| Entrega | del        | TFM |      |     |     |
| ------- | ---------- | --- | ---- | --- | --- |
| 4.4.2.  | Cronograma |     | real |     |     |
Como se comento´ en el cronograma inicial, en un principio se realiz´o una fase
previa en diciembre, donde se planteo´ la metodolog´ıa y se comenz´o el estado del arte.
Una vez terminada dicha fase previa que sirvio´ para plantear el proyecto de forma
general, se empezaron las iteraciones. En cada iteracio´n se llevo´ a cabo una revisi´on de
| la literatura, | disen˜o | si proced´ıa | e implementaci´on. |     |     |
| -------------- | ------- | ------------ | ------------------ | --- | --- |
Durante la primera iteracio´n, se planeo´ realizar un sistema de recomendaci´on
al estilo de Kostrzewa et al. [28]. Con esta idea en mente, se empezo´ a implementar
la obtencio´n y preprocesamiento de los datos. Sin embargo, cuando se termino´ esa
implementacio´n, se decidi´o empezar una iteraci´on nueva porque se consider´o que el
| enfoque | de Ulloa [52] | podr´ıa | llegar a ser m´as | interesante. |     |
| ------- | ------------- | ------- | ----------------- | ------------ | --- |
En esta segunda iteracio´n se realiz´o de nuevo una revisio´n de literatura relaciona-
da con t´ecnicas de reduccio´n de dimensionalidad y formas de calcular la similitud entre
embeddings. Posteriormente, se disen˜´o la estructura de los vectores y la arquitectura
del sistema. Finalmente, se implemento´ y evalu´o dicha arquitectura.
De forma transversal al proyecto, se fue documentando todo el proyecto, escri-
29

biendo de forma incremental los diferentes apartados del presente documento.
|                  | Fase            |         |             |     | Per´ıodo    |       | Estimaci´on |          | de       | Horas   |
| ---------------- | --------------- | ------- | ----------- | --- | ----------- | ----- | ----------- | -------- | -------- | ------- |
|                  | Fase previa     |         |             | dic | 2024        | - ene | 2025        |          |          | 15,3 h  |
|                  | Iteracio´n      | 1       |             | ene | - mar       | 2025  |             |          |          | 74,6 h  |
|                  | Estado          | del     | arte        |     | ene         | 2025  |             |          |          | 31,3 h  |
|                  | Implementacio´n |         |             | feb | - mar       | 2025  |             |          |          | 43,3 h  |
|                  | Iteracio´n      | 2       |             | abr | - may       | 2025  |             |          | 119,6    | h       |
|                  | Estado          | del     | arte        |     | abr         | 2025  |             |          |          | 21,5 h  |
|                  | Disen˜o         |         |             |     | abr         | 2025  |             |          |          | 12,8 h  |
|                  | Implementacio´n |         |             | abr | - may       | 2025  |             |          |          | 85,3 h  |
|                  | Redaccio´n      |         |             | dic | 2024        | - jun | 2025        |          |          | 95 h    |
|                  | Total           |         |             |     |             |       |             |          |          | 304,5 h |
|                  | Cuadro          | 4.2:    | Estimaci´on |     | real        | de    | horas por   | fase del | proyecto |         |
| A continuacio´n, |                 | podemos |             | ver | el diagrama |       | de Gantt    | real:    |          |         |
|                  |                 |         | 2024        |     |             |       | 2025        |          |          |         |
|                  |                 |         | 11          | 12  | 01          | 02    | 03 04       | 05       | 06       |         |
|                  | Fase            | previa  |             |     |             |       |             |          |          |         |
|                  | Iteracio´n      |         | 1           |     |             |       |             |          |          |         |
| Estado           | del             | arte    |             |     |             |       |             |          |          |         |
Implementacio´n
|        | Iteracio´n |      | 2   |     |     |     |     |     |     |     |
| ------ | ---------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
| Estado | del        | arte |     |     |     |     |     |     |     |     |
Disen˜o
Implementacio´n
Redaccio´n
| Entrega | del | TFM |     |     |     |     |     |     |     |     |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
30

4.5. Presupuesto
Para la elaboracio´n del presupuesto se han tenido en cuenta diversos factores que
| representan |     | costes   | directos | e             | indirectos |     | en el proyecto: |     |     |
| ----------- | --- | -------- | -------- | ------------- | ---------- | --- | --------------- | --- | --- |
| 4.5.1.      |     | Recursos |          | Intelectuales |            |     |                 |     |     |
El precio de los perfiles inform´aticos se ha obtenido mirando el sueldo medio de
los Data Scientist Junior en Espan˜a a trav´es de [22] y [47]. Por ello, para las 300 horas
estimadas del proyecto se ha estimado un coste por hora de 15 euros, lo que da un coste
total de 4.500 euros. Este coste representa el valor del trabajo intelectual invertido en
este trabajo.
| 4.5.2. |     | Recursos |     | materiales |     |     |     |     |     |
| ------ | --- | -------- | --- | ---------- | --- | --- | --- | --- | --- |
Se ha utilizado un port´atil con un coste de 950 euros para todo el desarrollo
del proyecto. A este coste, se ha tenido que tener en cuenta el coste de la conexio´n a
Internet y del consumo el´ectrico. A su vez, se ha tenido que tener en cuenta los costes
derivados del uso de GitHub Pro y el uso de Microsoft 365 incorporados en el plan de
estudiante.
Para el consumo se ha mirado el consumo de energ´ıa de la bater´ıa del port´atil
Asus Rog Strix que es de 56 W/h, por lo que consumo = 56W/h = 0,056kW/h. En el
siguiente cuadro 4.3 podemos ver el presupuesto estimado para el TFM:
|     | Concepto |     |     |     |     |     | Detalle |     | Coste |
| --- | -------- | --- | --- | --- | --- | --- | ------- | --- | ----- |
estimado
|     | Horas         |     | de trabajo |        |           | 304,5h     | x         | 15e/h     | 4.567,5e |
| --- | ------------- | --- | ---------- | ------ | --------- | ---------- | --------- | --------- | -------- |
|     | Amortizacio´n |     |            | de     | Ordenador |            | port´atil | gama      | 316,67e  |
|     | hardware      |     | a 3        | an˜os  |           | media-alta |           | (1 an˜o)  |          |
|     | Suscripcio´n  |     | a          | GitHub |           | 12e/mes    |           | x 6 meses | 72e      |
Pro
|     | Conexio´n      |     | a Internet |       | 6 meses  |        | de fibra   | (20e/mes) | 120e |
| --- | -------------- | --- | ---------- | ----- | -------- | ------ | ---------- | --------- | ---- |
|     | Almacenamiento |     |            | en la | OneDrive |        | (Microsoft | 365:      | 12e  |
|     | nube           |     |            |       |          | 2e/mes | x          | 6 meses)  |      |
5.088,17e
Total
|     |     |     |     | Cuadro | 4.3: | Presupuesto |     | del proyecto |     |
| --- | --- | --- | --- | ------ | ---- | ----------- | --- | ------------ | --- |
31

5. Descripcio´n de la propuesta
5.1. Introducci´on
En este cap´ıtulo explicaremos el sistema a desarrollar y se presentara´ la propuesta
de disen˜o, sin entrar en detalles t´ecnicos de la implementacio´n.
5.2. Propuesta
El propo´sito principal del sistema es que sea capaz de recomendar canciones
utilizando las caracter´ısticas de audio de las canciones. Este sistema ser´ıa entonces
un sistema basado en contenido (content-based) que aprendiera las relaciones entre
las diferentes caracter´ısticas extra´ıdas de los audios. Con este sistema, se pretende
poder recomendar mu´sica sin necesidad de conocer informacio´n de usuario o historial
de canciones escuchadas, necesarios en sistemas basados en filtrado colaborativo o
sistemas h´ıbridos.
5.3. Estructura del sistema de recomendaci´on
Una vez tenemos clara la propuesta del sistema de recomendacio´n que vamos a
disen˜ar e implementar, podemos discutir qu´e partes tendra´. Como queremos hacer un
sistema basado en contenido que utilice las caracter´ısticas de audio, podemos trabajar
con vectores de baja dimensionalidad aplicando reductores como PCA. Estos vectores
ser´ıan del tipo 5.1.
x ∈ Rn (5.1)
Sin embargo, PCA utiliza operaciones lineales de todas las dimensiones de los
vectores. Ah´ı es donde entran en juego los Autoencoders (AE o VAE). Estos apren-
der´ıan relaciones no lineales entre las caracter´ısticas y proporcionar´ıan vectores de
menor dimensionalidad. El AE simplemente proporciona vectores latentes que contie-
nen la informacio´n m´ınima de las dimensiones relevantes para ser capaz de reconstruir
las entradas a partir de estos vectores, mientras que el VAE trabaja con distribuciones
normales, es decir, el espacio latente que origina sigue una distribuci´on probabil´ıstica
N(0,1).
El resultado obtenido del Autoencoder ser´ıa de la forma Z = [z ,z ,...,z ] donde
1 2 m
cadazser´ıaelembeddingdecadacancio´ndeldataset.Paradefinirladistancia,tenemos
varias opciones:
Distancia en el espacio latente (k-NN): Para una cancio´n query z , cal-
q
cularemos la distancia (eucl´ıdea, coseno del a´ngulo, etc.) contra todas las z ∈
i
32

Z = [z ,z ,...,z ]. La ventaja principal es que es un m´etodo sencillo que permi-
|     |     | 1 2 | m   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
te ajustar el nu´mero de vecinos k fa´cilmente, aunque no utiliza la funcio´n de
|     | similitud | ´optima | para | el embedding. |     |     |     |
| --- | --------- | ------- | ---- | ------------- | --- | --- | --- |
Perceptro´n multicapa (MLP) que aprenda una funcio´n de similitud: Es
un modelo supervisado que aprende si dos canciones son similares o no en funcio´n
de sus embeddings. El modelo deber´ıa dar una puntuacio´n de similitud a partir
de dos embeddings z y z . La principal ventaja es que el MLP aprender´ıa una
|     |     |     |     | 1 2 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
funcio´n no lineal de similitud teniendo en cuenta nuestros datos, pero necesitar´ıa
entrenar con pares de canciones etiquetadas como similares y alguna heur´ıstica
|     | para construirlos. |     |     |     |     |     |     |
| --- | ------------------ | --- | --- | --- | --- | --- | --- |
Red siamesa: Una red siamesa permite comparar a la vez varios elementos.
En nuestro caso, comparar´ıamos las canciones, minimizando la distancia entre
|     | canciones | similares |     | y maximiza´ndola |     | entre las | no similares. |
| --- | --------- | --------- | --- | ---------------- | --- | --------- | ------------- |
Este pipeline utilizar´ıa una definicio´n similar de los embeddings de Ulloa [52],
concatenando los vectores de las features que queremos utilizar. Para an˜adir mayor
complejidad al sistema, se utilizar´ıa un Autoencoder en lugar de PCA, y se an˜adir´ıan
diferentes estrategias para la similitud. Entre ellas, se utilizar´ıa k-NN como en [43] y
se podr´ıa llegar a an˜adir una red siamesa con el objetivo de predecir la similitud entre
| canciones, | similar |     | al de | Salas [44]. |     |     |     |
| ---------- | ------- | --- | ----- | ----------- | --- | --- | --- |
En la arquitectura final se recortar´a la adicio´n del MLP y la red siamesa, pues
habr´ıamuchascombinacionesdemodelosqueprobarparacompararcu´alesdanmejores
resultados. Adema´s, la solucio´n para calcular la distancia que ma´s se adapta a nuestros
modelos ser´ıa utilizar k-Nearest Neighbors, pudiendo comparar diferentes funciones de
| distancia | para         | medir | cua´l | da mejores | resultados. |     |     |
| --------- | ------------ | ----- | ----- | ---------- | ----------- | --- | --- |
| 5.3.1.    | Arquitectura |       |       | final      |             |     |     |
Ya hemos discutido los aspectos clave de la estructura principal de nuestro siste-
| ma. | Este seguira´ |     | el flujo | visible en | la figura | 5.1: |     |
| --- | ------------- | --- | -------- | ---------- | --------- | ---- | --- |
33

Inicio Reductores de
dimensionalidad Canción de
referencia
PCA
(M O a b g t n e a n T e a r g d A a T to u s ne Preproc d es a a to m s iento de Autoencoder K-NN T r o e p c - o K m c e a n n d c a io d n a e s s
Dataset)
Variational
Autoencoder
Figura 5.1: Arquitectura del sistema
1. Preprocesamiento: Transformamos los datos de los audios del dataset para
tener, por una parte, informaci´on de los g´eneros, las etiquetas y otros metadatos;
y por la otra, las caracter´ısticas de audio extra´ıdas.
2. Construcci´on de embeddings: Construimos para cada fragmento de audio
un embedding que tenga toda la informaci´on importante relativo a este. Los
embeddingsera´nunaconcatenacio´ndelaShortTimeFourierTransform(STFT),
los Mel Frequency Cepstral Coefficients (MFCC), los Chroma Coefficients y el
Tempo. Esto nos dar´ıa vectores de alta dimensionalidad con las propiedades ma´s
relevantes de los audios.
3. Reductores de dimensionalidad:Paraevitartrabajarconvectoresdemasiado
grandes, y que el ruido empeore el rendimiento de las recomendaciones, reduci-
mos la dimensionalidad de los embeddings. Analizaremos el rendimiento de PCA,
un Autoencoder (AE) y un Variational Autoencoder (VAE). Los autoencoders
aprendera´n a comprimir la informaci´on obtenida de las entradas en vectores la-
tentes, reduciendo la diferencia entre la entrada y la salida que se obtendr´ıa
reconstruyendo la entrada a partir del vector latente. El VAE nos dara´ unos
vectores latentes distribuidos de acuerdo con una distribuci´on normal N(0,1).
4. K-NN: Una vez obtenidos los embeddings reducidos utilizamos k-NN para reco-
mendar las k canciones ma´s cercanas a una dada segu´n una m´etrica de distancia
como la distancia eucl´ıdea.
En resumen, el sistema utilizar´ıa aprendizaje no supervisado para comprimir la
informacio´n al no utilizar etiquetas y dejar que los modelos descubran por s´ı mismos
qu´e caracter´ısticas de audio son importantes. Adema´s, un autoencoder bien entrenado
eliminar´ıa aquellas dimensiones redundantes o irrelevantes, reduciendo el ruido que
podr´ıa empeorar las recomendaciones. Por u´ltimo, k-NN nos permitir´ıa recomendar de
34

forma sencilla y efectiva canciones similares entre s´ı.
5.4. Embeddings
Como se coment´o en la seccio´n anterior, tendremos un embedding por cada frag-
mento del dataset. Se generara´n dos embeddings por modelo, uno exclusivamente con
caracter´ısticas de audio (STFT, Chroma Coefficients, MFCC y tempo) y otro con las
caracter´ısticas de audio, los g´eneros y las etiquetas codificadas.
Vamos a explicar qu´e informaci´on representara´ cada caracter´ıstica, siguiendo las
definiciones de las caracter´ısticas utilizadas por Ulloa [52]:
Short-Time Fourier Transform: Con la Short-Time Fourier Transform
(STFT) queremos convertir la sen˜al de audio x[n] del dominio tiempo al dominio
tiempo-frecuencia. Esta se calcula como:
N−1
(cid:88)
ξ(m,k) = x[n+mH]·w[n]·e−2jπkn/N
n=0
Donde x[n] es la sen˜al discreta de audio, w[n] es la ventana, N es el taman˜o de
la ventana, H es el salto, m es el ´ındice del frame temporal y k ∈ [0,N −1] es
el´ındice del bin de frecuencia. Tras esto tomamos el espectrograma elevando al
cuadrado
θ(m,k) = |ξ(m,k)|2
donde el eje horizontal representa el tiempo y el eje vertical representa la frecuen-
cia. Una vez tenemos el espectrograma, podemos calcular la media en el tiempo
de la energ´ıa en cada frecuencia para tener un vector que represente co´mo de
fuerte es la energ´ıa en promedio a lo largo de cada fragmento.
Mel Frequency Cepstral Coefficients: Con los Mel Frequency Cepstral Coef-
ficients (MFCC) podemos medir la forma del espectro de frecuencias en la escala
de Mel (escala perceptiva del o´ıdo humano). Los coeficientes tienen la siguiente
forma:
N−1
(cid:88)
S(n) = |ξ(k)|2 ·J (k)
m
k=0
Donde 0 ≤ m ≤ M−1,m es el filtro triangular y J (k) es el peso de contribuci´on
m
del bin k al filtro triangular m. Es decir, cu´anto aporta el bin k al filtro m. Una
vez tenemos la energ´ıa espectral en la frecuencia de Mel, calculamos la Discrete
Cosine Transform (DCT) para obtener un determinado nu´mero de coeficientes:
M−1
(cid:88) πn(m−0,5)
Λ (n) = S(n)·cos( )
m
M
m=0
Donden ∈ Neselnu´merodeMFCCquequeremos,enconcreto,13coeficientesde
acuerdo con Ulloa [52]. El u´ltimo paso ser´ıa calcular la media de cada coeficiente
en el tiempo.
35

Chroma Coefficients: Esta caracter´ıstica considera propiedades de la armon´ıa
y la melod´ıa de la sen˜al. En la mu´sica occidental, la mu´sica se divide en 12 tonos
| o pitches | diferentes |     |     |     |     |     |
| --------- | ---------- | --- | --- | --- | --- | --- |
{C,C♯,D,D♯,E,F,F♯,G,G♯,A,A♯,B},
que si traducimos del sistema de notacio´n anglosajo´n al latino ser´ıa
{Do,Do♯,Re,Re♯,Mi,Fa,Fa♯,Sol,Sol♯,La,La♯,Si}
Esto se conoce como escala croma´tica, teniendo siete notas naturales y cinco alte-
raciones, ♯ y ♭, que elevan la nota un semitono o la disminuyen, respectivamente.
En total, en una octava hay 12 notas posibles, ya que notas como Do♯ y Re♭ son
equivalentes enarmo´nicamente. Para calcular los coeficientes, se calcula la STFT
| y se obtiene | el  | espectrograma: |     |     |     |     |
| ------------ | --- | -------------- | --- | --- | --- | --- |
(cid:88)
|     |     |     | Y(n,p) | =   | |ξ(n,k)|2 |     |
| --- | --- | --- | ------ | --- | --------- | --- |
k∈P(p)
Donde p ∈ [0 : 127] es el nu´mero de la nota MIDI, y representa el tono, y P(p) es
una funcio´n que mapea k a las notas MIDI. Las notas MIDI son una forma de dar
una descripcio´n de la altura de una nota a un instrumento musical electr´onico,
codificando cada nota como un nu´mero entre 0 y 127. Esto se puede codificar en
| 7 bits, | y podr´ıamos | cubrir | 11 octavas. |     |     |     |
| ------- | ------------ | ------ | ----------- | --- | --- | --- |
Una vez hemos obtenido este espectrograma podemos calcular los coeficientes
sumando todos los coeficientes que pertenecen al mismo chroma:
(cid:88)
|     |     | Ψ(n,c) | =   |     | Y(n,p) |     |
| --- | --- | ------ | --- | --- | ------ | --- |
p∈[0:127]:pm´od12≡c
Siendo c ∈ [0 : 11]. Finalmente calculamos la media temporal para cada nota.
Tempo: El tempo nos indica la velocidad de la cancio´n, es decir, cuantos pulsos
suceden por minuto (BPM). Uno de los enfoques para beat tracking es la progra-
macio´n dina´mica. Nuestro objetivo es detectar los onsets en la sen˜al que tienen
un ritmo regular. Para ello, se utiliza una funci´on objetivo que localiza los beats
| y otra | que penaliza | las desviaciones: |          |     |          |       |
| ------ | ------------ | ----------------- | -------- | --- | -------- | ----- |
|        |              |                   | N        |     | N        |       |
|        |              |                   | (cid:88) |     | (cid:88) |       |
|        |              | Ξ(t ) =           | O(t      | )+α | F(t −t   | ,τ )  |
|        |              | i                 |          | i   | i        | i−1 p |
|        |              |                   | i=1      |     | i=2      |       |
Donde t es la secuencia de N beats, O(t) es la fuerza de onset en el instante t, α
i
es un para´metro para balancear la penalizaci´on de F, y τ es un tempo ideal.
p
Paraconstruirlamejorsecuenciadebeatsposibles,tomamoslaf´ormularecursiva:
|     |     | Ξ∗(t) = | O(t)+ | ma´x {αF(t−τ,τ | )+Ξ∗(τ)} |     |
| --- | --- | ------- | ----- | -------------- | -------- | --- |
p
τ=0,..,t
As´ı, para cada valor de t se decide si deber´ıa considerarse como beat teniendo
en cuenta si hay una alta activacio´n de onset O(t) y si el tiempo desde el beat
| anterior | τ es coherente | con | el tempo | esperado | τ . |     |
| -------- | -------------- | --- | -------- | -------- | --- | --- |
p
36

6. Implementaci´on
6.1. Introducci´on
En este cap´ıtulo explicaremos el proceso de investigaci´on y construcci´on del sis-
tema de recomendacio´n musical. Hablaremos, por una parte, de las herramientas uti-
lizadas, del preprocesamiento y visualizacio´n realizados, y de los modelos entrenados,
por otra.
6.2. Herramientas
En primer lugar, vamos a comentar el entorno tecnolo´gico utilizado, co´mo se ha
| preparado | dicho entorno | y las librer´ıas | utilizadas. |
| --------- | ------------- | ---------------- | ----------- |
| 6.2.1.    | Entorno       | de ejecucio´n    |             |
El sistema ha sido desarrollado en una ma´quina con Windows 11 en Visual Studio
Code. Los modelos se han implementado utilizando un AMD Ryzen 7 6000 series y
una gra´fica Nvidia 3060 RTX; es decir, se han implementado teniendo en mente su
| ejecucio´n | en un entorno | local.      |               |
| ---------- | ------------- | ----------- | ------------- |
| 6.2.2.     | Preparacio´n  | del entorno | de desarrollo |
Antesdeempezaratrabajarenlaimplementaci´on,setuvoqueprepararelentorno
detrabajocontodaslasconfiguracionesnecesarias.ElIDEelegidohasidoVisualStudio
| Code (VSCode) | por          | su simplicidad. |     |
| ------------- | ------------ | --------------- | --- |
| 6.2.3.        | Instalacio´n | de dependencias |     |
El lenguaje de programacio´n fue Python 3.11.0. Para aislar las dependencias,
tambi´en se configuro´ un entorno virtual. Para ello, se utilizo´ el siguiente comando:
python -m venv myenv
Donde myenv es el nombre del entorno virtual. Las librer´ıas utilizadas y sus versiones
esta´n disponibles en el fichero “requirements.txt” y se pueden instalar con el siguiente
| comando | usando pip: |             |                     |
| ------- | ----------- | ----------- | ------------------- |
|         |             | pip install | -r requirements.txt |
37

6.2.4. Librer´ıas utilizadas
A lo largo del proyecto se han utilizado diversas librer´ıas. Para el procesamiento
de las sen˜ales de audio ha sido clave librosa, que tiene herramientas para poder cargar
audios, extraer caracter´ısticas y poder visualizarlos y reproducirlos.
En cuanto a los modelos, por una parte ha sido importante el uso de scikit-
learn para construir el modelo k-NN y las m´etricas. Por otra, tensorflow y keras han
permitido construir y entrenar los Autoencoders y Variational Autoencoders.
Otraslibrer´ıasu´tileshansidopandas ynumpy paratrabajarcondatostabularesy
vectores;matplotlib yseaborn paragra´ficos;rich paraembellecerlasalidaenconsola;os
para la manipulacio´n de ficheros y directorios; y requests para automatizar la descarga
del dataset.
6.3. Preprocesamiento
La primera tarea que se ha tenido que realizar ha sido el preprocesamiento de los
datos del dataset. Para la obtencio´n de los datos se desarrollo´ un mo´dulo llamado “01-
download.py” que permite descargar tanto las tres partes del dataset en formato ZIP y
descomprimirlas,comodescargarlosCSVdelosmetadatosdelascanciones.Alejecutar
dicho script se crea una carpeta llamada dataset que contendra´ toda la informaci´on.
El objetivo de los siguientes pasos ser´a conseguir almacenar de forma local; es decir,
en la propia ma´quina, tanto los audios de las canciones como la informacio´n relativa a
los metadatos y las caracter´ısticas extra´ıdas en bases de datos SQLite.
Acontinuacio´n,sevaaexplicarqu´epasossehanseguidoparatrabajarconlosda-
tos. Todo el preprocesamiento esta´ disponible en un mo´dulo llamado “02-database.py”,
que preprocesa y crea todas las tablas de nuestra base de datos. Los pasos realizados
han sido los siguientes:
1. Separar etiquetas de g´eneros: El CSV que contiene los g´eneros contiene
adema´s diferentes etiquetas que las personas que participaron en el juego de
TagATune an˜adieron, lo que implica que es necesario un buen tratamiento de es-
tas etiquetas. Sabiendo esto, se observ´o que hab´ıa muchos g´eneros y subg´eneros
mezclados, adema´s de muchas etiquetas que ten´ıan significados muy parecidos.
En total ten´ıamos 190 columnas, de las cuales 188 eran booleanas.
Otro problema a tener en cuenta es la dificultad para decir a qu´e g´enero perte-
nece una canci´on, pues no es una propiedad completamente objetiva del sonido,
sino una construcci´on cultural con l´ımites difusos al haber muchas canciones que
combinan varios g´eneros, o que dentro de un g´enero hay muchos subg´eneros con
diferencias notables. A esto hay que sumarle el hecho de que una canci´on puede
ser de un g´enero diferente para dos personas, lo que para uno es “rock alterna-
tivo”, para otra puede ser “indie rock”. Estas dificultades hacen que haya que
tomarunabuenaestrategiaparaintentarminimizarelimpactodequelosg´eneros
asignados no tengan mucha representacio´n en el esquema global o que hayan sido
mal impuestas por los participantes de TagATune.
38

Como solucio´n a este problema, en un principio se separo´ el CSV en dos BBDD,
una para los g´eneros y otra para las etiquetas. Para los g´eneros se tuvo que
agrupar muchos subg´eneros en sus g´eneros ma´s grandes utilizando la funcio´n OR
para evitar tener muchos g´eneros con baja representacio´n, y, adema´s se tuvo que
an˜adir algunas etiquetas que permitir´ıan identificar de mejor forma ese g´enero.
Tambi´en cabe destacar que se agruparon t´erminos con similar sema´ntica. Vamos
a pasar a comentar algunos de los g´eneros cuya fusio´n ha sido m´as compleja:
Mu´sica cl´asica: En el caso de la mu´sica cl´asica se fusionaron palabras como
“clasical” o “classic” pero tambi´en se agrupo´ el subg´enero barroco y las
canciones catalogadas como orquestales.
´
Arabe: Se fusiono´ con la mu´sica de oriente medio, al comprender muchos de
los pa´ıses que dan origen a este g´enero musical.
Electr´onica: Se agruparon subg´eneros como el techno, el house o el dance.
´
Opera: Se separo´ este g´enero de la mu´sica cla´sica debido a que al incorporar
gran cantidad de voces podr´ıa tener bastantes diferencias con las piezas
musicales plenamente instrumentales. En este caso se fusionaron las ´operas
masculinas y femeninas, as´ı como las piezas interpretadas por sopranos.
2. Transformar los CSV en BBDD Sqlite: Tras tener los dataframes con la
informacio´n separada en g´eneros y etiquetas se guardaron en dos BBDD SQLite
diferentes.
De forma transversal, se tuvo que realizar el tratamiento de valores nulos y
outliers. Este tratamiento es importante para evitar que haya muchas canciones que
an˜adan ruido a los algoritmos y que impidan un correcto funcionamiento de estos.
Cuando se visualizo´ si hab´ıa algu´n problema con los datos despu´es del proce-
samiento, se observo´ que hab´ıa m´as de 10.000 fragmentos sin g´enero (ningu´n g´enero
siendo True) y m´as de 6.000 sin etiqueta. Es decir, que ten´ıamos sobre un 40% de
fragmentos sin g´enero y sobre un 25% sin etiqueta.
Por lo tanto, hab´ıa que encontrar una soluci´on a este problema. Las alternativas
que se plantearon son las siguientes:
Modelo preentrenado: Utilizar un modelo ya entrenado nos permitir´ıa obtener
algunos g´eneros para todas las canciones, y podr´ıamos desechar el CSV de los
g´eneros. Esto implicar´ıa tambi´en que no se utilicen las etiquetas, y por lo tanto,
la detecci´on del g´enero tome m´as peso en el sistema de recomendaci´on. Algunas
opciones ser´ıan Musicnn1, que utiliza CNN con espectrogramas, o VGGish2 para
obtener embeddings de los audios, y entrenar despu´es un SVM o k-NN para
predecir los g´eneros.
Web scraping: Utilizando web scraping podr´ıamos obtener informaci´on de las
canciones de la base de datos de Magnatune. Sin embargo, probando de forma
manual se ha observado que hay ´albumes que ya no est´an disponibles, por lo que
esta alternativa queda descartada porque las canciones son poco conocidas y ser´ıa
1Musicnn: https://github.com/jordipons/musicnn
2VGGish: https://www.kaggle.com/models/google/vggish
39

dif´ıcil encontrar un u´nico sitio web aparte de Magnatune con todas las canciones
del dataset.
Eliminar canciones sin g´enero: Aunque dr´astica, esta opcio´n nos permitir´ıa
tener el dataset m´as limpio, y no arrastrar´ıamos errores como podr´ıa pasar con
las dos primeras opciones. Es cierto que se perder´ıa un porcentaje considerable de
instancias del dataset, pero podr´ıa ser una p´erdida asumible teniendo en cuenta
la cantidad de canciones disponibles.
Asignar g´enero mayoritario a las canciones: Se podr´ıa designar para un
fragmento, el g´enero mayoritario asignado a otros fragmentos de su misma can-
cio´n. Adema´s, se podr´ıan plantear t´ecnicas para asignar solamente canciones con
una buena representaci´on de g´eneros en sus canciones.
Finalmente, las opciones elegidas han sido la de asignar el g´enero mayoritario a
las canciones sin g´enero y eliminar las que no se puedan asignar. Son las opciones que
nos dejara´n unos resultados ma´s fiables, ya que cualquier opci´on tendr´ıa un margen de
error.
Tras un an´alisis de los fragmentos de audio se ha visto que hay bastantes can-
ciones que solamente tienen algunos fragmentos sin g´enero, es decir, que la mayor´ıa
de las canciones poseen al menos algu´n fragmento con g´enero asignado. Por lo tanto,
tambi´en podemos asignar el g´enero mayoritario de los fragmentos de la cancio´n a los
fragmentos sin g´enero y borrar las canciones que, tras esta asignaci´on, sigan sin g´enero.
La estrategia que se seguir´a ser´a asignar el g´enero mayoritario a aquellos fragmentos
que pertenezcan a una canci´on con g´eneros en al menos el 50% de sus partes.
Adema´s, aunque no tiene por qu´e cumplirse siempre, normalmente si varios frag-
mentos se perciben de un g´enero en concreto, el resto de la canci´on deber´ıa percibirse
del mismo modo. As´ı, si una cancio´n de 9 fragmentos tiene g´eneros en 7 de estos, y
todos son clasificados como cl´asica, los otros dos los asignar´ıamos como mu´sica cla´sica
tambi´en.
Una vez realizadas estas acciones para limpiar el dataset, podr´ıamos utilizar li-
brosa para extraer audio features como los MFCC o el tempo. Las caracter´ısticas que
se han extra´ıdo son los Chroma Coefficients, STFT, MFCC y el tempo. Despu´es de
todos estos pasos, tendr´ıamos cuatro archivos sqlite, uno con los g´eneros, otro con las
etiquetas, otro con otros metadatos y el u´ltimo con las audio features.
6.4. Visualizaci´on
La siguiente tarea realizada es la visualizacio´n, donde podemos observar de forma
gra´fica los datos para comprenderlos mejor. Tambi´en ayuda a visualizar outliers o
patrones en los datos. Todo el co´digo est´a disponible en el Notebook de Jupyter “03-
visualize”.
En primer lugar, tenemos que saber con cua´ntos datos contamos despu´es de todo
el preprocesamiento. En un principio ten´ıamos 25.863 fragmentos, 50 g´eneros y 133
40

etiquetas. Tras el preprocesamiento, nos quedamos con un total de 17.783 fragmentos
| de audio, | 19 g´eneros | diferentes | y 76 etiquetas. |     |     |
| --------- | ----------- | ---------- | --------------- | --- | --- |
Todos estos fragmentos forman un total de 4.825 canciones, de las cuales 2.192
tienen so´lo un g´enero, 1.539 tienen dos, 668 tienen tres y 426 tienen ma´s de tres. En
la siguiente tabla 6.1 podemos ver cua´ntas canciones tienen estos g´eneros. Podemos
|     |     |        | G´enero      | Canciones    |             |
| --- | --- | ------ | ------------ | ------------ | ----------- |
|     |     |        | ambient      | 882          |             |
|     |     |        | arabic       | 148          |             |
|     |     |        | blues        | 117          |             |
|     |     |        | classical    | 1872         |             |
|     |     |        | country      | 316          |             |
|     |     |        | eastern      | 235          |             |
|     |     |        | electronic   | 1329         |             |
|     |     |        | folk         | 563          |             |
|     |     |        | funk         | 188          |             |
|     |     |        | hip hop      | 73           |             |
|     |     |        | indian       | 481          |             |
|     |     |        | jazz         | 293          |             |
|     |     |        | metal        | 259          |             |
|     |     |        | opera        | 531          |             |
|     |     |        | oriental     | 130          |             |
|     |     |        | pop          | 675          |             |
|     |     |        | punk         | 89           |             |
|     |     |        | reggae       | 32           |             |
|     |     |        | rock         | 981          |             |
|     |     | Cuadro | 6.1: Nu´mero | de canciones | por g´enero |
observar que el dataset tiene cierto desbalanceo, teniendo canciones con alta represen-
tacio´n como la mu´sica cl´asica o la electro´nica frente a otros g´eneros escasos como el
punk o el hip hop. Esto se puede ver mejor en la siguiente figura 6.1.
41

Figura 6.1: Densidad de canciones por g´enero
Podemos observar que la mu´sica cl´asica representa de por s´ı el 20% de todas las
canciones, mientras que el reggae representa casi el 0%. Sera´ importante tener esto en
cuenta por si hiciera falta eliminar aquellos g´eneros raros, y con ello, las canciones que
se queden sin asignar.
En cuanto a las etiquetas, tenemos 373 canciones con so´lo una, 564 con dos, 767
con tres y 3.046 con m´as de tres. Cabe destacar que hay canciones que pueden no tener
ninguna etiqueta, pero no se ha tomado una decisi´on de eliminar aquellas canciones
porque no se ha considerado tan importante como el caso de los g´eneros.
Para representar esto, y como son tantas las etiquetas que tenemos en nuestro
dataset, vamos a visualizar el siguiente diagrama de sectores 6.2.
42

Figura 6.2: Densidad de canciones por etiqueta
Este diagrama representa de forma ordenada las 20 etiquetas ma´s frecuentes en
las canciones, y vemos que las ma´s frecuentes ser´ıan “calm” o “guitar”, mientras que
el 21% restante representar´ıa las 51 etiquetas restantes. Por lo tanto, las etiquetas
tambi´en sufren un desbalanceo que habr´a que estudiar si ser´ıa necesario solucionar o
se podr´ıa aceptar simplemente.
Si en vez de observar las canciones observamos los fragmentos que las conforman,
esdecir,lasfilasdenuestrodataset,podemosobservarenlafigura6.3quelaproporci´on
de densidad se mantiene.
43

Figura 6.3: Densidad de fragmentos por g´enero
La mu´sica cla´sica y electro´nica seguir´ıan siendo los m´as frecuentes, con un 18%
de representacio´n aproximadamente, y el reggae representa menos del 1% de los frag-
mentos.
En cuanto a las etiquetas, ser´ıa imposible representar tantas etiquetas, por lo
que vamos a comparar las etiquetas m´as frecuentes con las ma´s raras. En 6.4 podemos
observar que hay etiquetas presentes casi en el 0% del dataset, frente a etiquetas como
“guitar” y “calm” (que estaban en el 8% de canciones del dataset) que esta´n en el 18%
de todo el dataset.
44

|     | Figura 6.4: | Comparaci´on | de etiquetas | ma´s y menos | frecuentes |
| --- | ----------- | ------------ | ------------ | ------------ | ---------- |
Vamos a ver la coocurrencia de los g´eneros en el dataset, es decir, qu´e g´eneros
suelenaparecerjuntoseneldataset.En6.5sepuedeverunamatrizcuadradasim´etrica,
cuya diagonal representa la cantidad de fragmentos que tienen dicho g´enero. Cuanto
ma´s claro sea el color, mayor es la coocurrencia entre dos g´eneros. Teniendo esto en
cuenta, podemos ver co´mo g´eneros como o´pera y cla´sica, rock y metal, o ambient y
| electro´nica | suelen aparecer | juntos. |     |     |     |
| ------------ | --------------- | ------- | --- | --- | --- |
45

|     |     | Figura 6.5: | Coocurrencia | de g´eneros |
| --- | --- | ----------- | ------------ | ----------- |
A continuaci´on, vamos a visualizar la coocurrencia de las etiquetas, ya que hay
etiquetas que tienen sentido que suelan aparecer juntas. En 6.6 podemos observar que
las etiquetas que m´as salen juntas son “guitar” con “strings” y “calm”, “piano” con
“strings” que tiene sentido en muchas obras de mu´sica cla´sica y “beats” con “drum”,
algo que tiene mucho sentido pues est´an relacionadas. Adema´s, tambi´en hay otras
coocurrencias que no tienen tanto peso pero tambi´en tienen sentido como la presencia
de “guitar” con “electric guitar” o “classical guitar”, y la ausencia de que “guitar”
| aparezca con | “no guitar” | que ser´ıa | una contradiccio´n. |     |
| ------------ | ----------- | ---------- | ------------------- | --- |
46

| Figura 6.6: | Coocurrencia | de etiquetas |
| ----------- | ------------ | ------------ |
Finalmente, vamos a analizar la correlacio´n de Pearson entre los g´eneros y las
etiquetas ma´s frecuentes. En concreto, se han elegido 15 g´eneros y 30 etiquetas por
visibilidad de la matriz.
47

|     |     | Figura | 6.7: Correlaci´on | entre | g´eneros | y etiquetas |
| --- | --- | ------ | ----------------- | ----- | -------- | ----------- |
En 6.7 podemos ver que hay etiquetas muy relacionadas con ciertos g´eneros. Los
casos ma´s destacables ser´ıan “sitar” con la mu´sica india, “strings” y “piano” con la
| mu´sica | cla´sica, | “heavy” con | el metal | o “choral” | con | la ´opera. |
| ------- | --------- | ----------- | -------- | ---------- | --- | ---------- |
6.5. Modelos
Una vez que tenemos el dataset preparado, podemos hablar de la implementacio´n
de la estructura que tendr´a nuestro sistema de recomendacio´n. Como comentamos en
la seccio´n anterior 5, este sistema estara´ formado por un pipeline donde tendr´ıamos las
| canciones | representadas | con | un vector | num´erico | del tipo | 6.1. |
| --------- | ------------- | --- | --------- | --------- | -------- | ---- |
x ∈ Rn (6.1)
Estos vectores tendr´ıan caracter´ısticas en diferentes escalas (ej. tempo vs MFCC) por
lo que tendr´ıamos que normalizarlos por dimensio´n. Una vez hecho esto, los vectores
entrar´ıan en un Autoencoder (AE o VAE). El encoder maximizar´ıa la compacidad del
48

vector, minimizando la diferencia entre el vector de entrada y la entrada reconstruida
por el decoder. La principal ventaja de utilizar este tipo de redes neuronales frente a
t´ecnicas ma´s simples como PCA es que PCA utiliza operaciones lineales, lo que puede
llegar a ser una limitacio´n dependiendo de los datos. Los vectores del espacio latente
sera´n los que utilizaremos para decidir la similitud entre las canciones. El resultado
obtenido del Autoencoder ser´ıa de la forma Z = [z ,z ,...,z ] donde cada z ser´ıa el
|           |     |     |               |              |     |     | 1 2 m |
| --------- | --- | --- | ------------- | ------------ | --- | --- | ----- |
| embedding |     | de  | cada cancio´n | del dataset. |     |     |       |
El primer modelo que se ha construido ha sido un PCA que reduzca la dimensi´on
de los embeddings de entrada a 128 para compararlo con el resto de modelos.
Para la construccio´n del Autoencoder se ha optado por una estructura sencilla
[48], cuyo co´digo 6.1 esta´ disponible en el Notebook “04-models”. El encoder est´a for-
mado por dos capas: la capa de entrada que recibe el embedding de taman˜o input_dim
y una capa oculta de 512 neuronas con funcio´n de activacio´n ReLU, cuya salida pasa a
unau´ltima capa de 128 neuronas (tambi´en con activacio´n ReLU) que generara´ el vector
en el espacio latente y la entrada del decoder. El decoder consta de una capa oculta
de 512 neuronas y una capa de salida con activaci´on lineal, que intenta reconstruir la
| entrada |                              | original | a partir | del vector | latente.     |     |     |
| ------- | ---------------------------- | -------- | -------- | ---------- | ------------ | --- | --- |
| def     | build_autoencoder(input_dim, |          |          |            | latent_dim): |     |     |
1
|     | """Construye |     | un autoencoder |     | simple""" |     |     |
| --- | ------------ | --- | -------------- | --- | --------- | --- | --- |
2
|     | input_layer |     | = Input(shape=(input_dim,)) |     |     |     |     |
| --- | ----------- | --- | --------------------------- | --- | --- | --- | --- |
3
|     | encoded |     | = Dense(512, | activation=’relu’)(input_layer) |     |     |     |
| --- | ------- | --- | ------------ | ------------------------------- | --- | --- | --- |
4
| 5   | encoded |     | = Dense(latent_dim, |     | activation=’relu’)(encoded) |     |     |
| --- | ------- | --- | ------------------- | --- | --------------------------- | --- | --- |
6
|     | decoded |     | = Dense(512, | activation=’relu’)(encoded) |     |     |     |
| --- | ------- | --- | ------------ | --------------------------- | --- | --- | --- |
7
| 8   | decoded |     | = Dense(input_dim, |     | activation=’linear’)(decoded) |     |     |
| --- | ------- | --- | ------------------ | --- | ----------------------------- | --- | --- |
9
| 10  | autoencoder |     | = Model(inputs=input_layer, |     |     |                  | outputs=decoded) |
| --- | ----------- | --- | --------------------------- | --- | --- | ---------------- | ---------------- |
|     | encoder     |     | = Model(inputs=input_layer, |     |     | outputs=encoded) |                  |
11
12
13 autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss= ’mse’)
|     | return |     | autoencoder, | encoder |     |     |     |
| --- | ------ | --- | ------------ | ------- | --- | --- | --- |
14
15
autoencoder, encoder = build_autoencoder(input_dim=X.shape[1], latent_dim
16
=128)
17 print(encoder.summary())
|     |     |     | Extracto | de co´digo | 6.1: Construcci´on |     | del autoencoder |
| --- | --- | --- | -------- | ---------- | ------------------ | --- | --------------- |
En el caso del Variational Autoencoder, se ha tenido en cuenta una arquitectura
ma´s compleja que un Autoencoder tradicional, como se puede observar en 6.2. La ar-
quitectura esta´ compuesta por tres partes principales: el encoder, el bloque de sampling
| (reparametrizacio´n) |     |     | y el | decoder. |     |     |     |
| -------------------- | --- | --- | ---- | -------- | --- | --- | --- |
El encoder comienza con una capa de entrada que recibe el embedding de dimen-
sio´n input_dim. A continuaci´on, una capa densa oculta con 512 neuronas y activacio´n
ReLU proyecta los datos a una representaci´on intermedia que se bifurca en dos ramas
paralelas:
Una capa que genera el vector de medias z_mean del espacio latente.
Una capa que produce la varianza de logaritmos z_log_var para cada dimensio´n
49

latente.
Estas dos salidas definen una distribuci´on gaussiana multivariada por cada entrada. A
diferencia de un autoencoder cl´asico, el VAE no codifica directamente en un punto del
espacio latente, sino que aprende una distribuci´on desde la cual se puede muestrear.
Posteriormente,parapermitirlaretropropagaci´onatrav´esdelmuestreoestoc´asti-
co se utiliza la reparametrizaci´on. Para ello, definimos una capa llamada Sampling, que
| genera | una | muestra | z   | mediante: |     |     |     |     |
| ------ | --- | ------- | --- | --------- | --- | --- | --- | --- |
1
|     |     |     |     | z   | = µ+exp( | logσ2)·ϵ |     |     |
| --- | --- | --- | --- | --- | -------- | -------- | --- | --- |
2
donde µ = z , logσ2 = z y ϵ ∼ N(0,1) [45]. Esta operacio´n permite el paso
|     |           | mean |         | log var       |     |     |     |     |
| --- | --------- | ---- | ------- | ------------- | --- | --- | --- | --- |
| del | gradiente | a    | trav´es | del muestreo. |     |     |     |     |
Finalmente, el decoder recibe como entrada el vector muestreado del espacio
latente. A partir de este, intenta reconstruir la entrada original utilizando una capa
densa oculta de 512 neuronas con activaci´on ReLU, seguida de una capa de salida con
| activaci´on |     | sigmoide. |     |     |     |     |     |     |
| ----------- | --- | --------- | --- | --- | --- | --- | --- | --- |
Para la funcio´n de p´erdida se ha combinado por una parte la p´erdida de re-
construccio´n (reconstruction loss), que mide la similitud de la salida reconstruida con
respecto a la entrada original utilizando el Error Cuadr´atico Medio (MSE), y la p´erdida
KL (KL divergence), que penaliza la diferencia entre la distribucio´n latente aprendida
| y   | una distribucio´n |     | normal | est´andar | [55]. | Se calcula | como: |     |
| --- | ----------------- | --- | ------ | --------- | ----- | ---------- | ----- | --- |
1 (cid:88)
|     |     |     |     | KL = − |     | (1+logσ2 | −µ2 | −σ2) |
| --- | --- | --- | --- | ------ | --- | -------- | --- | ---- |
2
En el entrenamiento se ha utilizado early stopping para evitar sobreajuste y se ha
utilizado el optimizador Adam con una tasa de aprendizaje de 0.001 y normalizaci´on
| del   | gradiente                        | para | evitar | explosiones |     | de gradiente. |     |     |
| ----- | -------------------------------- | ---- | ------ | ----------- | --- | ------------- | --- | --- |
| class | Sampling(tf.keras.layers.Layer): |      |        |             |     |               |     |     |
1
| 2   | """Clase |            | de muestreo | para     | el  | VAE""" |     |     |
| --- | -------- | ---------- | ----------- | -------- | --- | ------ | --- | --- |
|     | def      | call(self, |             | inputs): |     |        |     |     |
3
|     |     | z_mean, | z_log_var | =   | inputs |     |     |     |
| --- | --- | ------- | --------- | --- | ------ | --- | --- | --- |
4
|     |     | batch | = tf.shape(z_mean)[0] |     |     |     |     |     |
| --- | --- | ----- | --------------------- | --- | --- | --- | --- | --- |
5
|     |     | dim = | tf.shape(z_mean)[1] |     |     |     |     |     |
| --- | --- | ----- | ------------------- | --- | --- | --- | --- | --- |
6
| 7   |     | epsilon | = tf.random.normal(shape=(batch, |              |     |              |     | dim))     |
| --- | --- | ------- | -------------------------------- | ------------ | --- | ------------ | --- | --------- |
|     |     | return  | z_mean                           | + tf.exp(0.5 |     | * z_log_var) |     | * epsilon |
8
9
| def | build_vae(input_dim, |     |     | latent_dim): |     |     |     |     |
| --- | -------------------- | --- | --- | ------------ | --- | --- | --- | --- |
10
|     | # Encoder |     |     |     |     |     |     |     |
| --- | --------- | --- | --- | --- | --- | --- | --- | --- |
11
| 12  | inputs  | =   | Input(shape=(input_dim,), |                            |     | name=’encoder_input’) |     |     |
| --- | ------- | --- | ------------------------- | -------------------------- | --- | --------------------- | --- | --- |
|     | encoded | =   | Dense(512,                | activation=’relu’)(inputs) |     |                       |     |     |
13
|     | z_mean | =   | Dense(latent_dim, |     | name=’z_mean’)(encoded) |     |     |     |
| --- | ------ | --- | ----------------- | --- | ----------------------- | --- | --- | --- |
14
| 15  | z_log_var |                             | = Dense(latent_dim, |     |     | name=’z_log_var’)(encoded) |     |     |
| --- | --------- | --------------------------- | ------------------- | --- | --- | -------------------------- | --- | --- |
|     | z =       | Sampling(name=’z’)([z_mean, |                     |     |     | z_log_var])                |     |     |
16
17
encoder = Model(inputs, [z_mean, z_log_var, z], name=’encoder’)
18
19
| 20  | # Decoder |     |     |     |     |     |     |     |
| --- | --------- | --- | --- | --- | --- | --- | --- | --- |
latent_inputs = Input(shape=(latent_dim,), name=’z_sampling’)
21
50

|     | h_decoded |     | = Dense(512, |     | activation=’relu’)(latent_inputs) |     |     |     |
| --- | --------- | --- | ------------ | --- | --------------------------------- | --- | --- | --- |
22
23 outputs = Dense(input_dim, activation=’sigmoid’)(h_decoded)
24
| 25  | decoder |     | = Model(latent_inputs, |     |     |     | outputs, | name=’decoder’) |
| --- | ------- | --- | ---------------------- | --- | --- | --- | -------- | --------------- |
26
|     | # VAE |     |     |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- |
27
| 28  | outputs |                 | = decoder(encoder(inputs)[2]) |     |          |             |     |     |
| --- | ------- | --------------- | ----------------------------- | --- | -------- | ----------- | --- | --- |
|     | vae     | = Model(inputs, |                               |     | outputs, | name=’vae’) |     |     |
29
30
|     | return |     | vae, encoder, |     | decoder |     |     |     |
| --- | ------ | --- | ------------- | --- | ------- | --- | --- | --- |
31
32
| 33 def | reconstruction_loss(x, |     |                       |     | x_decoded): |            |     |     |
| ------ | ---------------------- | --- | --------------------- | --- | ----------- | ---------- | --- | --- |
|        | return                 |     | MeanSquaredError()(x, |     |             | x_decoded) |     |     |
34
35
| 36 def | kl_loss(z_mean, |     |     | z_log_var): |     |     |     |     |
| ------ | --------------- | --- | --- | ----------- | --- | --- | --- | --- |
return -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(
37
|     | z_mean) |     | - tf.exp(z_log_var), |     |     | axis=1)) |     |     |
| --- | ------- | --- | -------------------- | --- | --- | -------- | --- | --- |
38
| def | vae_loss(x, |     | x_decoded, |     | z_mean, |     | z_log_var): |     |
| --- | ----------- | --- | ---------- | --- | ------- | --- | ----------- | --- |
39
| 40  | r_loss |     | = reconstruction_loss(x, |     |     |            | x_decoded) |     |
| --- | ------ | --- | ------------------------ | --- | --- | ---------- | ---------- | --- |
|     | k_loss |     | = kl_loss(z_mean,        |     |     | z_log_var) |            |     |
41
| 42  | return |     | r_loss | + k_loss |     |     |     |     |
| --- | ------ | --- | ------ | -------- | --- | --- | --- | --- |
43
early_stopping = EarlyStopping(monitor=’val_loss’, patience=5,
44
restore_best_weights=True)
45
46 vae, vae_encoder, vae_decoder = build_vae(input_dim=X.shape[1],
latent_dim=128)
47
48 vae.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss=
lambda x, x_decoded: vae_loss(x, x_decoded, vae_encoder(x)[0],
vae_encoder(x)[1]))
49
vae.summary()
50
51 vae.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True,
validation_data=(X_test, X_test), callbacks=[early_stopping])
Extracto de co´digo 6.2: Construcci´on del Variational Autoencoder
Una vez construidos los modelos, estos se entrenan con el conjunto de entrena-
miento y se evalu´an con el de pruebas. Tras evaluar los reductores de dimensionalidad,
entrenaremos los modelos con todo el dataset y los guardaremos para calcular las re-
comendaciones.
Paradefinirladistanciaentrevectoreslatentes,utilizaremosk-NearestNeighbors.
Todo el co´digo esta´ disponible en el Notebook “05-recommender”. En el c´odigo 6.3
podemos observar la implementacio´n. La idea principal es buscar como m´aximo los X
fragmentos en el espacio latente cercanos al fragmento de referencia, y a partir de ellos,
| seleccionar |     | las | canciones | que | corresponden |     | a dichos | fragmentos. |
| ----------- | --- | --- | --------- | --- | ------------ | --- | -------- | ----------- |
Dada una matriz de embeddings de fragmentos de audio, recibimos como entrada
el ´ındice en el dataset de un fragmento fragment-index. Como algoritmo para k-NN
se utiliza un a´rbol de particio´n de tipo ball tree, eficiente para nuestros vectores de alta
51

| dimensio´n. |     | Como | funcio´n |     | de distancia | utilizamos |     | Minkowski: |     |
| ----------- | --- | ---- | -------- | --- | ------------ | ---------- | --- | ---------- | --- |
N
|     |     |     |     |     |        |     | (cid:88) |         | 1   |
| --- | --- | --- | --- | --- | ------ | --- | -------- | ------- | --- |
|     |     |     |     |     | D(x,y) | =   | ( |x     | −y |p)p |     |
|     |     |     |     |     |        |     |          | i i     |     |
i=1
Cuyo valor por defecto del para´metro p es 2. Esto significa que, por defecto, se est´a
| utilizando |     | la  | distancia | euclidiana. |     |     |     |     |     |
| ---------- | --- | --- | --------- | ----------- | --- | --- | --- | --- | --- |
Una vez obtenidos los search_pool vecinos ma´s cercanos al fragmento de referen-
cia, filtramos aquellos fragmentos cuya cancio´n a la que pertenecen (song_id) coincida
con el del fragmento base y guardamos las recomendaciones en un diccionario para
mantener la unicidad de las canciones recomendadas, preservando el orden segu´n la
distancia. Verdaderamente, search_pool no nos puede garantizar que vayamos a obte-
nern_songsu´nicasporquepuedehaberfragmentosquepertenezcanalamismacancio´n
y que la cancio´n base se descarte directamente, por lo que si entre los search_pool
vecinos hay muchas canciones repetidas, no llegaremos a n_songs.
1 def recommend_songs_by_fragment_knn(fragment_index, embeddings,
|     | fragment_to_song, |     |     |     | n_songs=5, | search_pool=50, |     |     | n_jobs=6): |
| --- | ----------------- | --- | --- | --- | ---------- | --------------- | --- | --- | ---------- |
"""
2
Devuelve hasta ‘n_songs‘ diferentes basados en fragmentos similares.
3
- fragment_index: ´ındice del fragmento base en ‘embeddings‘.
4
| 5   | -   | embeddings:       |     | matriz | de embeddings |     | de    | fragmentos. |              |
| --- | --- | ----------------- | --- | ------ | ------------- | --- | ----- | ----------- | ------------ |
|     | -   | fragment_to_song: |     |        | lista         | que | mapea | fragmento   | -> canci´on. |
6
|     | -   | n_songs: | cantidad |     | de canciones |     | ´unicas | deseadas. |     |
| --- | --- | -------- | -------- | --- | ------------ | --- | ------- | --------- | --- |
7
- search_pool: cu´antos vecinos buscar inicialmente (se expandir´a si
8
|     | no  | se alcanzan |     | n_songs). |     |     |     |     |     |
| --- | --- | ----------- | --- | --------- | --- | --- | --- | --- | --- |
9 """
10
knn = NearestNeighbors(n_neighbors=search_pool, algorithm=’ball_tree’
11
|     | ,   | n_jobs=n_jobs, |     | metric=’minkowski’) |     |     |     |     |     |
| --- | --- | -------------- | --- | ------------------- | --- | --- | --- | --- | --- |
knn.fit(embeddings)
12
13 distances, indices = knn.kneighbors(embeddings[fragment_index].
|     | reshape(1, |     | -1), | n_neighbors=search_pool) |     |     |     |     |     |
| --- | ---------- | --- | ---- | ------------------------ | --- | --- | --- | --- | --- |
14
| 15  | base_song_id |     |     | = fragment_to_song[fragment_index] |     |     |     |     |     |
| --- | ------------ | --- | --- | ---------------------------------- | --- | --- | --- | --- | --- |
16
| 17  | seen_songs |       | =   | OrderedDict() |                   |     |              |     |     |
| --- | ---------- | ----- | --- | ------------- | ----------------- | --- | ------------ | --- | --- |
|     | for        | dist, | idx | in            | zip(distances[0], |     | indices[0]): |     |     |
18
|     |     | song_id |     | = fragment_to_song[idx] |     |     |     |     |     |
| --- | --- | ------- | --- | ----------------------- | --- | --- | --- | --- | --- |
19
| 20  |     | if  | song_id | ==  | base_song_id: |     |     |     |     |
| --- | --- | --- | ------- | --- | ------------- | --- | --- | --- | --- |
continue
21
|     |     | if  | song_id | not | in seen_songs: |     |     |     |     |
| --- | --- | --- | ------- | --- | -------------- | --- | --- | --- | --- |
22
|     |     |     | mp3_path |     | = metadata_df.iloc[idx][’mp3_path’] |     |     |     |     |
| --- | --- | --- | -------- | --- | ----------------------------------- | --- | --- | --- | --- |
23
|     |     |     | seen_songs[song_id] |     |     | =   | (mp3_path, |     | dist) |
| --- | --- | --- | ------------------- | --- | --- | --- | ---------- | --- | ----- |
24
| 25  |     | if  | len(seen_songs) |     | >=  | n_songs: |     |     |     |
| --- | --- | --- | --------------- | --- | --- | -------- | --- | --- | --- |
break
26
27
return [(song_id, mp3_path, dist) for song_id, (mp3_path, dist) in
28
seen_songs.items()]
|     |     |     | Extracto |     | de co´digo | 6.3: | Recomendaci´on |     | con k-NN |
| --- | --- | --- | -------- | --- | ---------- | ---- | -------------- | --- | -------- |
En la siguiente seccio´n 7, compararemos las m´etricas conseguidas con las diferen-
tesfuncionesparacadaunodelosembeddingsobtenidosdelosdiferentesAutoencoders
utilizados.
52

7. Pruebas
7.1. Introducci´on
Este cap´ıtulo esta´ dedicado a la evaluacio´n tanto de los modelos de reducci´on de
dimensio´n desarrollados en la secci´on anterior como del sistema de recomendaci´on.
7.2. Modelos
Para evaluar los diferentes modelos se han utilizado como medidas el Error
Cuadra´tico Medio (MSE), la Ra´ız del Error Cuadra´tico Medio (RMSE) y el Error
Absoluto Medio (MAE). Para calcular el MSE utilizamos la fo´rmula 7.1:
n
1 (cid:88)
−y˜)2
|     |     | MSE = | (y    |     | (7.1) |
| --- | --- | ----- | ----- | --- | ----- |
|     |     |       | n i i |     |       |
i=1
Donde y es el valor real, y˜ es el valor reconstruido por el modelo y n es el nu´mero
|     | i   | i   |     |     |     |
| --- | --- | --- | --- | --- | --- |
de valores de entrada. Esta m´etrica nos permite ver la distorsi´on entre los embeddings
| originales | y las reconstrucciones. | El RMSE | se mide como | 7.2: |     |
| ---------- | ----------------------- | ------- | ------------ | ---- | --- |
√
|     |     | RMSE | = MSE |     | (7.2) |
| --- | --- | ---- | ----- | --- | ----- |
Al ser la ra´ız cuadrada del MSE, no penaliza tanto los errores. El MAE utiliza la
| siguiente | fo´rmula 7.3: |     |     |     |     |
| --------- | ------------- | --- | --- | --- | --- |
n
1 (cid:88)
|     |     | MAE = | |y −y˜| |     | (7.3) |
| --- | --- | ----- | ------- | --- | ----- |
|     |     |       | n i i   |     |       |
i=1
Donde y es el valor real, y˜ es el valor reconstruido por el modelo y n es el nu´mero de
|         | i i         |     |     |     |     |
| ------- | ----------- | --- | --- | --- | --- |
| valores | de entrada. |     |     |     |     |
En la siguiente tabla 7.1 podemos ver el rendimiento de los diferentes modelos
evaluados mediante el MSE. Para cada t´ecnica se han entrenado dos variantes, una
utilizando exclusivamente los embeddings a partir de las caracter´ısticas de audio selec-
cionadas y otra (denominada full) que adema´s incorpora informacio´n sobre los g´eneros
y etiquetas.
Los mejores resultados los obtienen los modelos PCA y Autoencoder, al tener
las menores medias y desviaciones esta´ndar, seguidos de las versiones full de ambos
modelos. Por otro lado, tanto VAE como VAE full muestran un rendimiento significa-
tivamente inferior, con errores medios y desviaciones mucho mayores con respecto al
| resto de | modelos. |     |     |     |     |
| -------- | -------- | --- | --- | --- | --- |
53

|     |     |             | Modelo      |        | Media    | Desviacio´n |
| --- | --- | ----------- | ----------- | ------ | -------- | ----------- |
|     |     |             | PCA         |        | 0.0849   | 0.4780      |
|     |     |             | PCA full    |        | 0.1513   | 0.6628      |
|     |     |             | Autoencoder |        | 0.0964   | 0.4772      |
|     |     | Autoencoder |             | full   | 0.1230   | 0.4471      |
|     |     |             | VAE         |        | 0.9573   | 6.5291      |
|     |     |             | VAE full    |        | 0.9637   | 5.9890      |
|     |     |             |             | Cuadro | 7.1: MSE |             |
A continuaci´on, en la tabla 7.2 se presentan los resultados obtenidos para el
RMSE. Al igual que en el caso anterior, comparamos distintas t´ecnicas de reducci´on
de dimensionalidad, tanto en su versio´n ba´sica como en la versi´on full.
Observamos que los modelos PCA y Autoencoder vuelven a obtener los mejores
resultados, con valores medios de RMSE ma´s bajos y menor desviaci´on. En cambio, los
modelos VAE y VAE full vuelven a tener un rendimiento bastante inferior, reflejado en
| una media | de error | y una       | desviacio´n | muy    | superiores. |             |
| --------- | -------- | ----------- | ----------- | ------ | ----------- | ----------- |
|           |          |             | Modelo      |        | Media       | Desviacio´n |
|           |          |             | PCA         |        | 0.2000      | 0.2114      |
|           |          |             | PCA full    |        | 0.3096      | 0.2353      |
|           |          |             | Autoencoder |        | 0.2192      | 0.2200      |
|           |          | Autoencoder |             | full   | 0.2735      | 0.2194      |
|           |          |             | VAE         |        | 0.6943      | 0.6902      |
|           |          |             | VAE full    |        | 0.7351      | 0.6506      |
|           |          |             |             | Cuadro | 7.2: RMSE   |             |
Finalmente,enlatabla7.3vemoslosresultadosobtenidosparaelMAE,siguiendo
| la misma | estructura | que         | las dos tablas |        | anteriores. |             |
| -------- | ---------- | ----------- | -------------- | ------ | ----------- | ----------- |
|          |            |             | Modelo         |        | Media       | Desviacio´n |
|          |            |             | PCA            |        | 0.1146      | 0.1242      |
|          |            |             | PCA full       |        | 0.1553      | 0.1320      |
|          |            |             | Autoencoder    |        | 0.1463      | 0.1423      |
|          |            | Autoencoder |                | full   | 0.1748      | 0.1413      |
|          |            |             | VAE            |        | 0.4940      | 0.4326      |
|          |            |             | VAE full       |        | 0.4792      | 0.3962      |
|          |            |             |                | Cuadro | 7.3: MAE    |             |
Vamos a analizar la varianza explicada para las 128 dimensiones de nuestros em-
beddings reducidos con PCA y PCA full. La varianza explicada por cada componente
principal representa cu´anta informaci´on (o variabilidad) de los datos originales cap-
tura esa componente. Por tanto, una alta varianza acumulada implica que el modelo
54

estara´ reteniendo una representacio´n fiel de los datos en menos dimensiones, lo cual es
deseable.
Con PCA hemos conseguido retener un 92.70% de la varianza total del dataset
original,mientrasque conPCAfull un86.74%.Esto indicaque,en amboscasos,la ma-
yor parte de la informaci´on original ha sido conservada en el espacio reducido, aunque
la versio´n full, al incluir m´as atributos, distribuye la varianza entre ma´s dimensiones,
lo que puede indicarnos por qu´e se ha retenido menor varianza. Podemos observar la
evolucio´n de la varianza para diferentes valores de dimensionalidad en la figura 7.1.
Figura 7.1: Varianza acumulada por nu´mero de componentes
Vamos a centrarnos ahora en los autoencoders. Vamos a comparar la distribucio´n
de errores de las m´etricas de los cuadros anteriores para ver do´nde hay outliers. En la
figura 7.2 observamos diferencias en la distribuci´on de RMSE y de MAE, teniendo una
mayor cantidad de instancias con errores cercanos a cero en el autoencoder normal.
La mayor´ıa de los errores son cercanos a 0, lo que es un buen signo. Sin embargo, es
posible ver algunos outliers en la gr´afica del autoencoder full, cuyo valor de RMSE es
superior a 2, lo que nos da una indicaci´on de por qu´e el eje X de la gr´afica de MSE
tiene valores por encima de 20. En total, hay 41 instancias cuyo RMSE es mayor a 1,
lo que nos indica que son posibles outliers.
55

Figura 7.2: Comparacio´n de resultados: (arriba) Autoencoder, (abajo) Autoencoder
full
En la figura 7.3 podemos ver la distribuci´on de los errores en los modelos VAE
y VAE full. Como ha sucedido en los autoencoders, la mayor´ıa de los errores han sido
cercanos a 0, teniendo ma´s instancias con errores mayores. Esto se puede ver en la
escala del eje X, teniendo en el caso del RMSE instancias que llegan hasta 17.5 y en el
MSE instancias que llegan a errores de 300. Al tener m´as outliers, es entendible el mal
rendimiento observado en los cuadros anteriores.
Figura 7.3: Comparaci´on de resultados: (arriba) VAE, (abajo) VAE full
56

En la siguiente figura 7.4 podemos observar claramente los outliers en las m´etri-
cas. La m´etrica que mejor los evidencia es el MSE, donde se aprecia que los outliers de
los Autoencoders (AE) tienen una diferencia enorme con los de los Variational Auto-
| encoders (VAE). |     |     |     |     |     |
| --------------- | --- | --- | --- | --- | --- |
Asimismo, tanto en el RMSE como en el MAE, los AE muestran errores muy
pro´ximos a cero en pra´cticamente todas las instancias, mientras que los VAE presen-
tan mu´ltiples casos con errores considerablemente mayores. Esto explica por qu´e en
el MSE, que penaliza m´as los errores grandes, los outliers del VAE alcanzan valores
| exponencialmente | superiores. |              |            |            |            |
| ---------------- | ----------- | ------------ | ---------- | ---------- | ---------- |
|                  | Figura 7.4: | Comparaci´on | de errores | por modelo | y m´etrica |
Asimismo, en la figura 7.5 se observa el espacio latente del Autoencoder y el VAE.
Se ha coloreado cada fragmento combinando los colores de cada uno de sus g´eneros.
En el espacio del Autoencoder vemos que hay algunos g´eneros que suelen moverse en
ciertas partes del espacio, aunque no es algo estricto porque el g´enero no es una m´etrica
completamente objetiva que nos permita separar las canciones de forma totalmente
clara.
En el caso del Variational Autoencoder, al estar regularizado el espacio, vemos
que los puntos esta´n centrados sobre el origen y que las canciones de un mismo g´enero
| esta´n distribuidas | por | todo el espacio. |     |     |     |
| ------------------- | --- | ---------------- | --- | --- | --- |
57

Figura 7.5: Espacio latente: (izquierda) Autoencoder, (derecha) VAE
7.3. Recomendaciones
Para evaluar el rendimiento de las recomendaciones de nuestro sistema, evalua-
remos la eficacia de k-NN para recomendar las canciones utilizando los embeddings
reducidos con cada una de las t´ecnicas previamente descritas. Como trabajamos con
fragmentos de canciones y queremos recomendar canciones, tenemos que evaluar las
recomendaciones dando relevancia a que las canciones compartan g´enero. Antes de
| definir | las | m´etricas, | definimos: |              |     |             |     |
| ------- | --- | ---------- | ---------- | ------------ | --- | ----------- | --- |
|         | Sea | G es el    | conjunto   | de canciones |     | relevantes. |     |
Sea R = {r ,r ,...,r } el conjunto ordenado de las k canciones recomendadas.
|     |     | k   | 1 1 | k   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
I(·)
Sea la funcio´n que determina si una canci´on es relevante (1) o no (0).
Sea rel(i) = I(r ∈ G) la relevancia binaria de la canci´on recomendada en la
i
|     | posici´on | i   |          |         |            |       |     |
| --- | --------- | --- | -------- | ------- | ---------- | ----- | --- |
| Las | m´etricas | que | usaremos | son las | siguientes | [15]: |     |
Precision@k: Proporcio´n de canciones recomendadas entre las k primeras que
comparten en su fragmento al menos una etiqueta con el fragmento de la canci´on
base. Mide la precisio´n inmediata del sistema en las k primeras posiciones del
ranking.
k
1 (cid:88)
|     |     |     |     | Precision@k |     | = I(r | ∈ G) |
| --- | --- | --- | --- | ----------- | --- | ----- | ---- |
i
k
i=1
|     | Una | alta precisio´n |     | indica que | el sistema | recomienda | bien. |
| --- | --- | --------------- | --- | ---------- | ---------- | ---------- | ----- |
58

Recall@k: Proprocio´n de canciones relevantes totales que fueron encontradas en
las k recomendaciones. Evalu´a la capacidad de recuperacio´n del sistema.
k
|     |     |          | 1 (cid:88) |          |
| --- | --- | -------- | ---------- | -------- |
|     |     | Recall@k | =          | I(r ∈ G) |
i
|G|
i=1
HitRate@k: Indica si al menos una canci´on relevante fue recomendada. Es una
| m´etrica | binaria (0 o | 1) por cada | consulta. |     |
| -------- | ------------ | ----------- | --------- | --- |
(cid:40)
|     |     |           | 1, si  | R ∩G ̸= ∅ |
| --- | --- | --------- | ------ | --------- |
|     |     | HitRate@k | =      | k         |
|     |     |           | 0, siR | ∩G = ∅    |
k
DondeR eselconjuntodecancionesrecomendadaseneltop-kyGeselconjunto
k
| de canciones | relevantes. |     |     |     |
| ------------ | ----------- | --- | --- | --- |
F1@k: Media armo´nica entre precision y recall. Equilibra ambos factores en una
sola m´etrica.
Precision@k·Recall@k
F1@k = 2·
Precision@k+Recall@k
AveragePrecision@k:Promediodeprecisionesacumuladascuandoapareceuna
cancio´n relevante. Parte de la premisa de que el orden en las recomendaciones
importa, por lo que las canciones ma´s relevantes deber´ıan ser las primeras que se
recomienden. Penaliza que las relevantes est´en muy abajo en el ranking.
k
|     |                    |     | 1   | (cid:88)           |
| --- | ------------------ | --- | --- | ------------------ |
|     | AveragePrecision@k |     | =   | Precision@i·rel(i) |
m´ın(|G|,k)
i=1
Donde:
|     |     |     | 1   | i   |
| --- | --- | --- | --- | --- |
(cid:88)
|     |     | Precision@i | =   | I(r ∈ G) |
| --- | --- | ----------- | --- | -------- |
j
i
j=1
nDCG@k (Normalized Discounted Cumulative Gain): Esta m´etrica mide
la ganancia acumulada de las canciones relevantes, descontada logar´ıtmicamente
segu´n su posicio´n, y la normaliza con respecto al mejor ranking posible. Evalu´a
tanto la relevancia como la posici´on dentro del top-k. Primero definimos el DCG:
k
|     |     |       | (cid:88) | rel(i) |
| --- | --- | ----- | -------- | ------ |
|     |     | DCG@k | =        |        |
log (i+1)
|     |     |     | i=1 | 2   |
| --- | --- | --- | --- | --- |
Luego, definimos el IDCG (Ideal DCG, con todas las canciones relevantes al
principio):
m´ın(|G|,k)
|     |     |        | (cid:88) | 1   |
| --- | --- | ------ | -------- | --- |
|     |     | IDCG@k | =        |     |
log (i+1)
2
i=1
Finalmente:
DCG@k
|     |     |     | nDCG@k = |     |
| --- | --- | --- | -------- | --- |
IDCG@k
Donde IDCG@k es el DCG ideal (orden perfecto). Una nDCG cercana a 1 indica
que las canciones relevantes esta´n en las primeras posiciones del ranking, en el
| mejor orden | posible | [2]. |     |     |
| ----------- | ------- | ---- | --- | --- |
59

A continuacio´n, vamos a estudiar el rendimiento de nuestras recomendaciones
| para   | diferentes |           | funciones | de        | distancia. |     |     |     |     |
| ------ | ---------- | --------- | --------- | --------- | ---------- | --- | --- | --- | --- |
| 7.3.1. |            | Distancia |           | eucl´ıdea |            |     |     |     |     |
En la siguiente tabla 7.4 podemos ver las recomendaciones con cada uno de los
| embeddings |             |      | reducidos | para        | k=5.     |        |           |        |        |
| ---------- | ----------- | ---- | --------- | ----------- | -------- | ------ | --------- | ------ | ------ |
|            | Modelo      |      |           | Precision@5 | Recall@5 | F1@5   | HitRate@5 | AP@5   | nDCG@5 |
|            |             | PCA  |           | 0.7026      | 0.0031   | 0.0062 | 0.9374    | 0.8059 | 0.6332 |
|            | PCA         | full |           | 0.8396      | 0.0041   | 0.0081 | 0.9771    | 0.9001 | 0.7616 |
|            | Autoencoder |      |           | 0.7087      | 0.0032   | 0.0063 | 0.9389    | 0.8080 | 0.6378 |
|            | Autoencoder |      | full      | 0.8685      | 0.0045   | 0.0088 | 0.9854    | 0.9192 | 0.8010 |
|            |             | VAE  |           | 0.3145      | 0.0012   | 0.0024 | 0.7803    | 0.4608 | 0.2598 |
|            | VAE         | full |           | 0.2986      | 0.0012   | 0.0023 | 0.7728    | 0.4480 | 0.2433 |
Cuadro 7.4: Evaluacio´n de las recomendaciones con distancia eucl´ıdea
Los embeddings que mejor resultado han dado han sido los del Autoencoder
full, que supera al resto de modelos en todas las m´etricas, incluso ligeramente por
encima de PCA full. Asimismo, todos los modelos, con excepci´on del VAE, mejoran
significativamente al usar tanto los g´eneros como las etiquetas (versio´n full):
PCA full vs PCA:HayunamejoraclaraenPrecision,AveragePrecision,nDCG
y HitRate.
Autoencoder full vs Autoencoder: La mejora es au´n mayor, lo que nos in-
dica que los autoencoders se benefician con m´as informaci´on para aprender a
|     | comprimir |     | la  | informaci´on | de forma | ma´s efectiva. |     |     |     |
| --- | --------- | --- | --- | ------------ | -------- | -------------- | --- | --- | --- |
VAE full vs VAE: Empeora ligeramente, y los resultados son muy bajos con
respectoalosotrosm´etodos.Estemalrendimientopuededeberseaqueelenfoque
probabil´ıstico del VAE introduce ruido que afecta negativamente a la calidad del
espacio latente, o a que se ha utilizado una mala configuracio´n de la arquitectura.
Por otro lado, en todos los casos se ve un Recall@5 muy bajo, lo cual es esperable
dado que hay una gran cantidad de canciones relevantes posibles, mientras que so´lo
evaluamos 5 recomendaciones. Por lo tanto, esta m´etrica sera´ baja, y no nos aportar´a
mucha informacio´n teniendo el resto de m´etricas. En consecuencia, el F1@5 tambi´en
| sera´ | bajo, | aunque | Precision@5 |     | sea muy alta. |     |     |     |     |
| ----- | ----- | ------ | ----------- | --- | ------------- | --- | --- | --- | --- |
Tambi´en vamos a ver la evolucio´n de estas m´etricas para valores de k ∈ [1 : 10].
En 7.6 vemos c´omo el HitRate y el F1 aumentan con mayores valores de k, mientras
que el resto de m´etricas se queda estables o descienden. Tambi´en observamos que PCA
full esta´ mejor entrenado, mostrando una mayor estabilizacio´n y valores ma´s altos para
| todas | las | m´etricas. |     |     |     |     |     |     |     |
| ----- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
60

Figura 7.6: Comparaci´on de recomendaciones con distancia eucl´ıdea: (arriba) PCA,
(abajo) PCA full
En la figura 7.7 sucede casi lo mismo que en el caso anterior. Como se coment´o
anteriormente, las versiones full de los modelos dan mejores rendimientos que sus ver-
siones est´andar.
61

Figura 7.7: Comparaci´on de recomendaciones con distancia eucl´ıdea: (arriba) Autoen-
| coder, (abajo) | Autoencoder | full |
| -------------- | ----------- | ---- |
Finalmente, en el caso del VAE en la figura 7.8, observamos una curva muy
pronunciada en el caso del HitRate, llegando a ∼ 0,93 en k = 10 lo que nos indica que
al menos una cancio´n relevante suele aparecer en el top-k. Sin embargo, en los modelos
anteriores se alcanzan valores cercanos a 1 mucho antes. Asimismo, vemos c´omo la
Precision y el nDCG se mantienen bajos y casi estables a lo largo de la curva, lo que
nos indica que el VAE no esta´ priorizando correctamente las canciones m´as relevantes
| en las primeras | posiciones | del ranking. |
| --------------- | ---------- | ------------ |
62

Figura 7.8: Comparacio´n de recomendaciones con distancia eucl´ıdea: (arriba) VAE,
| (abajo) | VAE full  |           |
| ------- | --------- | --------- |
| 7.3.2.  | Distancia | Manhattan |
En la siguiente tabla 7.5 podemos ver las recomendaciones con cada uno de los
embeddings reducidos para k = 5. Vemos que los resultados son muy similares a los
obtenidos con la distancia eucl´ıdea, teniendo un leve aumento en general en PCA full,
| Autoencoder | y Autoencoder | full. |
| ----------- | ------------- | ----- |
63

| Modelo      |      | Precision@5 | Recall@5 | F1@5   | HitRate@5 | AP@5   | nDCG@5 |
| ----------- | ---- | ----------- | -------- | ------ | --------- | ------ | ------ |
| PCA         |      | 0.6920      | 0.0031   | 0.0061 | 0.9310    | 0.7960 | 0.6264 |
| PCA         | full | 0.8469      | 0.0042   | 0.0083 | 0.9816    | 0.9067 | 0.7724 |
| Autoencoder |      | 0.7036      | 0.0032   | 0.0063 | 0.9396    | 0.8090 | 0.6350 |
| Autoencoder | full | 0.8658      | 0.0045   | 0.0088 | 0.9846    | 0.9171 | 0.7992 |
| VAE         |      | 0.3110      | 0.0012   | 0.0024 | 0.7840    | 0.4652 | 0.2565 |
| VAE         | full | 0.3027      | 0.0012   | 0.0024 | 0.7897    | 0.4590 | 0.2478 |
Cuadro 7.5: Evaluacio´n de las recomendaciones con distancia Manhattan
Como las tablas son muy parecidas, mantenemos que las versiones full de los
modelos aprenden mejor a construir unos embeddings robustos y que el Recall@5 es
una m´etrica que no nos aporta mucha informaci´on porque lo normal ser´a que tome
| valores muy | cercanos | a 0. |     |     |     |     |     |
| ----------- | -------- | ---- | --- | --- | --- | --- | --- |
Tambi´en vamos a ver la evolucio´n de estas m´etricas para valores de k ∈ [1 : 10].
En las figuras 7.9, 7.10 y 7.11 no se aprecian diferencias significativas con respecto a
| las gr´aficas | obtenidas | con distancia | eucl´ıdea. |     |     |     |     |
| ------------- | --------- | ------------- | ---------- | --- | --- | --- | --- |
Esto nos sugiere, por una parte, que el espacio latente aprendido es robusto a
pequen˜as variaciones en la m´etrica de similitud. Por otra, la distancia Manhattan
tiende a ser m´as tolerante a valores at´ıpicos y menos sensible a grandes diferencias
en pocas dimensiones, lo que podr´ıa explicar la pequen˜a mejora observada en algunos
modelos.
En consecuencia, parece que los modelos con mejor desempen˜o (como el Autoen-
coder full) esta´n siendo capaces de representar relaciones relevantes entre los embed-
dings de forma efectiva, independientemente de la m´etrica utilizada.
64

Figura 7.9: Comparacio´n de recomendaciones con distancia Manhattan: (arriba) PCA,
(abajo) PCA full
65

Figura 7.10: Comparaci´on de recomendaciones con distancia Manhattan: (arriba) Au-
| toencoder, | (abajo) Autoencoder | full |
| ---------- | ------------------- | ---- |
66

Figura7.11:Comparacio´nderecomendacionescondistanciaManhattan:(arriba)VAE,
(abajo) VAE full
67

8. Conclusiones
En este proyecto hemos conseguido implementar un sistema de recomendaci´on
basado en caracter´ısticas de audio y metadatos, codificados en embeddings. Para resol-
ver el problema de la alta dimensionalidad de dichos embeddings, hemos implementado
PCA, Autoencoders y Variational Autoencoders, siendo los dos primeros los que ofre-
cieron mejores resultados.
A la hora de comprimir la informaci´on, los que mejores resultados han dado han
sido el PCA y el Autoencoder entrenados con las caracter´ısticas de audio.
No obstante, al evaluar la calidad de las recomendaciones generadas con k-NN
sobre el espacio latente, se observo´ que el Autoencoder que combinaba tanto las carac-
ter´ısticas como los metadatos de las canciones tuvo los mejores resultados en todas las
m´etricas evaluadas. Esto sugiere que aportar ma´s informaci´on a los Autoencoders me-
jora el espacio latente generado y, en consecuencia, la calidad de las recomendaciones.
Por el contrario, los Variational Autoencoders han dado un resultado deficiente.
Este resultado podr´ıa deberse a diversos factores, como la complejidad del modelo, que
el modelo no se haya adaptado a los datos utilizados, o una mala configuracio´n de los
hiperpara´metros.
Cabe destacar que estos resultados de las recomendaciones se han obtenido consi-
derando que un fragmento de una canci´on es relevante si comparte al menos un g´enero
con el fragmento de referencia. Por lo tanto, cambiar el criterio de relevancia podr´ıa
modificar las m´etricas.
Como puntos de mejora, se proponen los siguientes:
Evaluar el sistema con otros de los datasets que se consideraron al principio
del proyecto como GTZAN o FMA, para visualizar si el rendimiento variar´ıa
dra´sticamente.
An˜adirmecanismosparasolucionareldesbalanceodelosg´eneroseneldataset,eli-
minandolos g´eneroscon menor presenciaoaplicando t´ecnicasdeunder-sampling.
Implementar modelos descartados para la similitud entre canciones como el MLP
o la red siamesa, comparando el rendimiento del k-NN ya implementado.
Cambiar el criterio de relevancia. Un criterio como que una canci´on es relevante
para recomendar si tiene tanto algu´n g´enero o alguna etiqueta en comu´n podr´ıa
cambiar de forma notoria las m´etricas.
Implementar en un sistema real el sistema de recomendacio´n disen˜ado y analizar
la evolucio´n del rendimiento conforme aumenta el taman˜o del dataset al an˜adir
nuevas canciones.
Enconclusi´on,elusodeAutoencodershasidomuysatisfactorioalahoraderedu-
cir la dimensionalidad de los embeddings y mejorar la calidad de las recomendaciones
musicales.
68

A. Bibliograf´ıa
[1] Moataz Ahmed, Sherif Fadel, Manal Helal, and Abdel Moneim Wahdan. Arabic
music genre identification. Journal of Advanced Research in Applied Sciences and
Engineering Technology, 2024. URL https://doi.org/10.37934/araset.46.1.
187200.
[2] Evidently AI. Normalized discounted cumulative gain (ndcg) explained, 2023.
| URL | https://www.evidentlyai.com/ranking-metrics/ndcg-metric. |     |     |     |     |     |     |
| --- | -------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
https://archialpizar.
| [3] Archi | Alp´ızar. | Melod´ıa | y duracio´n, | mayo 2015. | URL |     |     |
| --------- | --------- | -------- | ------------ | ---------- | --- | --- | --- |
wordpress.com/2015/05/27/melodia-y-duracion/.
[4] Jos´e Alvarado-Garc´ıa, Janet Herna´ndez-Garc´ıa, Esau´ Villatoro-Tello, Gabriela
Ramirez-de-laRosa,andChristianSa´nchez-Sa´nchez. Sistemaderecomendaci´onde
mu´sica basado en aprendizaje semi-supervisado. Research in Computing Science,
| 94:97–109, |     | 12 2015. | doi: 10.13053/rcs-94-1-8. |     |     |     |     |
| ---------- | --- | -------- | ------------------------- | --- | --- | --- | --- |
[5] Daniel P. W. Ellis and. Beat tracking by dynamic programming. Journal of New
Music Research, 36(1):51–60, 2007. doi: 10.1080/09298210701653344.
[6] Aprende Machine Learning. Comprende principal component analy-
https://www.aprendemachinelearning.com/
| sis | (pca), | 2018. | URL |     |     |     |     |
| --- | ------ | ----- | --- | --- | --- | --- | --- |
comprende-principal-component-analysis/.
| [7] Rahul | Awat. |     | garbage | in, garbage | out (gigo), | 2023. | URL |
| --------- | ----- | --- | ------- | ----------- | ----------- | ----- | --- |
https://www.techtarget.com/searchsoftwarequality/definition/
garbage-in-garbage-out.
[8] Subhranshu Behura, Arham Alam, Nishtha Phutela, Atul Mishra, and Goldie Ga-
brani. Tune into your feelings: Nlp-powered emotion driven music recommender
system. In Amita Dev, Arun Sharma, S. S. Agrawal, and Ritu Rani, editors, Ar-
tificial Intelligence and Speech Technology, pages 426–439, Cham, 2025. Springer
|        |              |     | https://doi.org/10.1007/978-3-031-75164-6 |     |     |     | 32. |
| ------ | ------------ | --- | ----------------------------------------- | --- | --- | --- | --- |
| Nature | Switzerland. |     | URL                                       |     |     |     |     |
[9] Ruixin Chen, Jianping Fan, Meiqin Wu, and Sining Ma. Conditional diffusion
model for recommender systems. Neural Networks, 185:107204, 2025. ISSN 0893-
6080. doi: https://doi.org/10.1016/j.neunet.2025.107204. URL https://www.
sciencedirect.com/science/article/pii/S0893608025000838.
[10] Cifra Club. Teor´ıa musical para principiantes: armon´ıa, me-
| lod´ıa | y   | ritmo, | s.f. | URL | https://www.cifraclub.com/blog/ |     |     |
| ------ | --- | ------ | ---- | --- | ------------------------------- | --- | --- |
teoria-musical-para-principiantes-armonia-melodia-ritmo/.
[11] Micha¨el Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. Fma:
A dataset for music analysis, 2017. URL https://arxiv.org/abs/1612.01840.
69

[12] Divy Dwivedi, Ashutosh Ganguly, and V.V. Haragopal. 6 - contrast between
simple and complex classification algorithms. In Tilottama Goswami and G.R.
Sinha, editors, Statistical Modeling in Machine Learning, pages 93–110. Aca-
demic Press, 2023. ISBN 978-0-323-91776-6. doi: https://doi.org/10.1016/
B978-0-323-91776-6.00016-6. URL https://www.sciencedirect.com/science/
article/pii/B9780323917766000166.
[13] Elastic. ¿qu´e es knn?, s.f. URL https://www.elastic.co/es/what-is/knn.
[14] Graph Everywhere. Sistemas de recomendaci´on: qu´e son, ti-
| pos | y ejemplos, |     | 2023. |     | URL | https://www.grapheverywhere.com/ |     |     |     |     |
| --- | ----------- | --- | ----- | --- | --- | -------------------------------- | --- | --- | --- | --- |
sistemas-de-recomendacion-que-son-tipos-y-ejemplos/.
[15] Evidently AI. Precision@k and recall@k – ranking evaluation metrics, s.f. URL
https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k.
[16] Flatiron School. The (data) science behind netflix recom-
| mendations, |     | 2021. |     |     | URL | https://flatironschool.com/blog/ |     |     |     |     |
| ----------- | --- | ----- | --- | --- | --- | -------------------------------- | --- | --- | --- | --- |
science-behind-netflix-recommendations.
[17] Foqum. ¿qu´e es un autoencoder?, s.f.. URL https://foqum.io/blog/termino/
autoencoder/.
[18] Foqum. ¿qu´e es un embedding?, s.f.. URL https://foqum.io/blog/termino/
embedding/.
[19] Eclipse Foundation. Eclipse ide: The leading open platform for professional deve-
| lopers,     | 2025.    | URL | https://eclipseide.org. |     |         |            |                           |         |     |     |
| ----------- | -------- | --- | ----------------------- | --- | ------- | ---------- | ------------------------- | ------- | --- | --- |
| [20] Marcos | Garc´ıa. |     |                         | El  | cifrado | americano: |                           | sistema | de  | no- |
| tacio´n     | musical, |     | 2021.                   |     |         | URL        | https://marcosgarcia.net/ |         |     |     |
el-cifrado-americano-sistema-de-notacion-musical/.
[21] Geek Culture. Variational autoencoder (vae), 2021. URL https://medium.com/
geekculture/variational-autoencoder-vae-9b8ce5475f68.
[22] Glassdoor. Sueldo: Ingeniero de software, 2025. URL https://www.glassdoor.
| es/Sueldos/ingeniero-de-software-sueldo-SRCH |            |     |       |     |        |                                 | KO0,21.htm. |         |        |     |
| -------------------------------------------- | ---------- | --- | ----- | --- | ------ | ------------------------------- | ----------- | ------- | ------ | --- |
| [23] Red                                     | Hat.       |     | What  | is  | ai/ml, | and                             | why         | does it | matter | to  |
| your                                         | business?, |     | 2023. |     | URL    | https://www.redhat.com/es/blog/ |             |         |        |     |
what-aiml-and-why-does-it-matter-your-business.
https://www.ibm.com/es-es/topics/
| [24] IBM. | ¿qu´e | es una | rnn?, | 2024. | URL |     |     |     |     |     |
| --------- | ----- | ------ | ----- | ----- | --- | --- | --- | --- | --- | --- |
recurrent-neural-networks.
[25] IBM. Variational autoencoder (vae), s.f. URL https://www.ibm.com/es-es/
think/topics/variational-autoencoder.
[26] Gatan Inc. Nyquist frequency, 2020. URL https://www.gatan.com/
nyquist-frequency.
70

[27] JetBrains. Pycharm: El ide de python para profesionales de datos y web, 2025.
URL https://www.jetbrains.com/pycharm/.
[28] Daniel Kostrzewa, Jonatan Chrobak, and Robert Brzeski. Attributes relevance in
content-based music recommendation system. Applied Sciences, 14(2), 2024. ISSN
2076-3417. doi: 10.3390/app14020855. URL https://www.mdpi.com/2076-3417/
14/2/855.
[29] Mark A. Kramer. Nonlinear principal component analysis using autoassociative
neural networks. AIChE Journal, 37(2):233–243, 1991. doi: https://doi.org/10.
1002/aic.690370209. URL https://aiche.onlinelibrary.wiley.com/doi/abs/
10.1002/aic.690370209.
[30] Bas Larrosa. Spotify algorithm: la gu´ıa definitiva, 2023. URL https://www.
larrosa.pro/post/spotify-algorithm-la-gua-definitiva.
[31] Edith Law, Kris West, Michael Mandel, Mert Bay, and J. Downie. Evaluation of
algorithms using games: The case of music tagging. In 10th International Society
for Music Information Retrieval Conference (ISMIR 2009), pages 387–392, 01
2009.
[32] Microsoft. Visual studio code, 2025. URL https://code.visualstudio.com.
[33] Himadri Mukherjee, Matteo Marciano, Ankita Dhar, and Kaushik Roy. Duf-
calf: Instilling sentience in computerized song analysis. Lecture Notes in
Computer Science (including subseries Lecture Notes in Artificial Intelli-
gence and Lecture Notes in Bioinformatics), 15300 LNAI:277 – 292, 2025.
doi: 10.1007/978-3-031-78014-1 21. URL https://www.scopus.com/inward/
record.uri?eid=2-s2.0-85210870138&doi=10.1007%2f978-3-031-78014-1 21&
partnerID=40&md5=0a9dc31e871292ec0cc9393d1cdb8600.
[34] Music Tomorrow. How spotify recommendation system works: A com-
plete guide (2022), 2022. URL https://www.music-tomorrow.com/blog/
how-spotify-recommendation-system-works-a-complete-guide-2022.
[35] Meinard Mu¨ller. Lab course: Short-time fourier transform (stft), 2016.
URL https://www.audiolabs-erlangen.de/content/05 fau/professor/
00 mueller/02 teaching/2016s apl/LabCourse STFT.pdf.
[36] MeinardMu¨ller. Onsetdetection,2023. URLhttps://www.audiolabs-erlangen.
de/resources/MIR/FMP/C6/C6S1 OnsetDetection.html.
[37] Ndiatenda Ndou, Ritesh Ajoodha, and Ashwini Jadhav. Music genre classifica-
tion: A review of deep-learning and traditional machine-learning approaches. In
2021 IEEE International IOT, Electronics and Mechatronics Conference (IEM-
TRONICS), pages 1–6, 2021. doi: 10.1109/IEMTRONICS52119.2021.9422487.
[38] Neurosnap. Understanding the differences between ai, machine lear-
ning, and deep learning, 2023. URL https://neurosnap.ai/blog/post/
understanding-the-differences-between-ai-machine-learning-and-deep-learning/
64279cadfeb3e5ca5ba0904a.
71

[39] Han-Saem Park, Ji-Oh Yoo, and Sung-Bae Cho. S.b.: A context-aware music
recommendation system using fuzzy bayesian networks with utility theory. In
Fuzzy Systems and Knowledge Discovery, volume 4223, pages 970–979, 09 2006.
ISBN 978-3-540-45916-3. doi: 10.1007/11881599 121.
[40] Adr´ıan Quijada. Sistema de recomendacio´n musical para ehealth. Master’s the-
sis, Universitat Oberta de Catalunya, 2020. URL https://openaccess.uoc.edu/
bitstream/10609/120146/9/aquijadagTFM0620memory.pdf.
[41] Kel Rodolfo. Spotify y las matema´ticas: as´ı es la fo´rmula de sus
recomendaciones, 2023. URL https://www.linkedin.com/pulse/
spotify-y-las-matemticas-as-es-la-frmula-de-sus-rodolfo-kel6f/.
[42] Ignacio Rodr´ıguez. Formato midi, 2002. URL https://www.lpi.tel.uva.es/
∼nacho/docencia/ing ond 1/trabajos 01 02/formatos audio digital/html/
midiformat.htm.
[43] Kapil Saini and Ajmer Singh. A content-based recommender system using stacked
lstmandanattention-basedautoencoder. Measurement: Sensors,31:100975,2024.
ISSN 2665-9174. doi: https://doi.org/10.1016/j.measen.2023.100975. URL https:
//www.sciencedirect.com/science/article/pii/S2665917423003112.
[44] Nicol´as Serrano Salas. Estudio y an´alisis del uso de redes siamesas en estrategias
de recomendacio´n basadas en contenido y de filtrado colaborativo. Master’s thesis,
Universidad Aut´onoma de Madrid, 2023. URL https://abellogin.github.io/
2022/NSS.pdf.
[45] Data Science. Difference between autoencoder (ae) and variational
autoencoder (vae), 2020. URL https://medium.com/data-science/
difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2.
[46] Jonathon Shlens. The mathematics behind principal
component analysis. https://medium.com/data-science/
the-mathematics-behind-principal-component-analysis-fff2d7f4b643,
2019.
[47] Talent.com. Salario ingeniero de software en espan˜a, 2025. URL https://es.
talent.com/salary?job=ingeniero+de+software.
[48] TensorFlow. Autoencoder,2024. URLhttps://www.tensorflow.org/tutorials/
generative/autoencoder?hl=es-419.
[49] Tonus M´exico. Escala croma´tica y construcci´on de esca-
las mayores, 2022. URL https://www.tonus.com.mx/blog/
escala-cromatica-y-construccion-escalas-mayores.
[50] Viet-Anh Tran, Guillaume Salha-Galvan, Bruno Sguerra, and Romain Hennequin.
Transformersmeetact-r:Repeat-awareandsequentiallisteningsessionrecommen-
dation, 2024. URL https://arxiv.org/abs/2408.16578.
72

[51] George Tzanetakis, Georg Essl, and Perry Cook. Automatic musical genre
classification of audio signals, 2001. URL http://ismir2001.ismir.net/pdf/
tzanetakis.pdf.
[52] Diego Saldan˜a Ulloa. Music recommendation based on audio fingerprint, 2023.
URL https://arxiv.org/abs/2310.17655.
[53] Ultralytics. Clasificacio´n de im´agenes con el dataset mnist, s.f. URL https:
//docs.ultralytics.com/es/datasets/classify/mnist/.
[54] Universidad Virtual de Quilmes. Duracio´n, tono, intensidad y timbre,
n.d. URL https://libros.uvq.edu.ar/spm/321 duracin tono intensidad y
timbre.html.
[55] Charel van Hoof. Learn by example variational autoencoder, 2019. URL https://
www.kaggle.com/code/charel/learn-by-example-variational-autoencoder.
[56] Wikipedia. Mfcc, 2024. URL https://es.wikipedia.org/wiki/MFCC.
[57] Feng Zhu, Yan Wang, Chaochao Chen, Jun Zhou, Longfei Li, and Guanfeng Liu.
Cross-domain recommendation: Challenges, progress, and prospects. CoRR, ab-
s/2103.01696, 2021. URL https://arxiv.org/abs/2103.01696.
´
[58] In˜aki Ucar. Conceptos musicales: nota, tono, figura y
‘pitch’, 2009. URL https://www.enchufa2.es/archives/
conceptos-musicales-nota-tono-figura-y-pitch.html.
73