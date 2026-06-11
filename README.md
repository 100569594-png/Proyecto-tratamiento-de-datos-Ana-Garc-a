## Análisis de desinformación y polarización lingüística en noticias mediante modelos de lenguaje
**Alumna:** Ana Gabriela García Rivas  
**Asignatura:** Tratamiento de Datos  
**Dataset:** WELFake 
**Problema:** Clasificación binaria de noticias reales y falsas  

---

## 1. Descripción del problema

La desinformación se trata de uno de los principales retos del sistema infnformativo actual. La difusión de noticias falsas ha llegado a alterar la percepción pública de la realidad, reforzando creencias  y contribuyendo de forma directa  a la polarización social e ideológica. De esta forma, es relevante estudiar hasta qué punto las técnicas de procesamiento del lenguaje natural (NLP) permiten detectar automáticamente patrones asociados a noticias falsas.

En este proyecto se abordará la clasificación automática de noticias reales y falsas utilizando el dataset **WELFake**. Cuyo objetivo principal es comparar diferentes representaciones vectoriales de texto y distintos modelos supervisados para evaluar cuáles son más adecuados en la detección de desinformación.

El trabajo se ha organizado en tres notebooks:

| Notebook | Contenido |
|---|---|
| `Trabajo datos_notebook1` | Análisis exploratorio, limpieza, detección de posibles atajos léxicos y creación de hipótesis  |
| `Trabajo datos_notebook2` | Vectorización con TF-IDF, Word2Vec y embeddings BERT |
| `Trabajo_datos_notebook3` | Modelado, evaluación, fine-tuning, pruebas de robustez y extensión |


## **2.-Conjunto de datos**

 hemos contado con una base de datos con noticias recopiladas de diferentes fuentes y etiquetadas como 1-True o 0-False/Partially False. Concretamente, hay un total de 3119 noticias, repartidas entre 2061 verdaderas y 1058 falsas (como muestra el histograma a continuación), de las que no solo se muestra la noticia y la etiqueta, sino también el título y la categoría (True/False/Partially False). 

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/f19ed717-adc3-4503-b334-b4a20e5a54a9" />

En cuanto a esta subdivisión de las noticias falsas entre False y Partially False, dado que el número de noticias verdaderas es casi el doble, hemos podido comprobar que provoca una degradación en las prestaciones de algunos de los algoritmos de clasificación al intentar clasificar entre 3 etiquetas diferentes. Es por ello que se ha simplificado el problema a un clasificador binario de noticias true/false sobre el COVID-19 eliminando la categoría de Partially False.

Las noticias de nuestro dataset tienen de media unos 2250 caracteres y 365 palabras, es decir, que son noticias bastante largas. En el histograma de la siguiente figura se muestra la distribución del número de caracteres y palabras y la frecuencia con la que aparecen en las noticias. Encontramos noticias de hasta 5500 palabras y 32000 caracteres, luego para futuras secciones, aunque se reducirá considerablemente el vocabulario empleado, nos quedaremos con unas 8000 palabras. Es algo más de lo habitual, pero el objetivo es no perder información en nuestras noticias tan grandes.

<img width="1624" height="573" alt="image" src="https://github.com/user-attachments/assets/e0ed4707-3508-4b86-95c3-74e5e58e23ca" />

Como hipótesis y resultados que podríamos esperar, podemos decir que, al haber casi el doble de noticias verdaderas que de noticias falsas, es probable que sea para la clase maypritaria (verdaderas) para la que cometamos menos errores al clasificar las noticias. Además, no hemos podido identificar en el primer cuaderno (de análisis de la base de datos) ninguna palabra característica o patrón importante que diferencie con claridad las noticias verdaderas y falsas.

Como trabajo previo a la presentación de las técnicas de vectorización y algoritmos clasificadores, se ha realizado una limpieza y homogeneización de este conjunto de noticias para evitar que caracteres, urls y palabras concretas (como cambiar *viru* por *virus*) entren a nuestros modelos e introduzcan ruido que dificulte el trabajo de clasificación.

## **3.-Herramientas empleadas**

### **3.1. Técnicas de vectorización:**

Se ha presentado en el punto anterior el dataset (cuaderno 1 en github) con noticias desde el que partimos para entrenar, validar y testear nuestros modelos. Sin embargo, entrando ya en el cuaderno 2 subido a github, para entrenar un algoritmo de clasificación como SVM, redes neuronales, etc. necesitamos convertir el texto en representaciones numéricas, para lo cual empleamos 3 técnicas de vectorización diferentes que se presentan a continuación. Destacar que en este apartado del proyecto, tán solo hemos definido y experimentado con estos tres modelos aplicados sobre la base de datos global y será en el siguiente apartado donde dividiremos el dataset en *train*,*test* y *validation*.

- ***TF-IDF:*** Esta técnica transforma cada una de nuestras noticias en un vector basándose únicamente en la importancia de las palabras en las noticias. Para implementarlo, se han propuesto 2 alternativas con dos librerías de python: gensim (partiendo de una descripción *Bag of Words*) y un modelo ya definido de sklearn. Las prestaciones en ambos casos hemos comprobado que son prácticamente iguales y por simplicidad, se ha utilizado el modelo de sklearn, que internamente realiza la tokenización de las noticias, vectorización y reducción del vocabulario. Esto es un punto importante, pues el vocabulario original de la base de datos está en torno a 37000 palabras, lo cual es difícil (e innecesario) de manejar por los modelos. Es por ello que se han suprimido aquellas palabras que han aparecido en más del 60% de las noticias o que aparecen en menos de 10.
  
- ***Word2Vec:*** Es un modelo que aprende embeddings semánticos de palabras, es decir, vectores densos donde palabras con significados similares tienen representaciones similares. Es una técnica a priori más compleja que TF-IDF (que solo se basa en frecuencia de aparición de las palabras) y esto se ve reflejado en los resultados que se presentarán posteriormente. A diferencia de TF-IDF, que partía de las noticias "limpias y homogeneizadas", en este caso se parte de los tokens de dichas noticias. Igual que antes, se ha realizado una reducción del vocabulario antes de calcular embeddings y para facilitar tareas futuras.

- ***Embeddings contextuales:*** Son vectores generados por modelos donde la representación de una palabra depende del contexto en el que aparece, es decir, capturan relaciones complejas entre palabras y son adecuados para representaciones profundas. En nuestro caso, igual que con TF-IDF, se ha realizado una comparativa entre 3 posibles modelos: BERT, RoBERTa y distilBERT. En este caso, para la elección del modelo más adecuado, hemos realizado una pequeña comparativa con una porción de la base de datos para ver cuál es el que ofrece mejores prestaciones. El resultado es que, para nuestra aplicación de clasificador *true*/*false*, es el modelo de RoBERTa el que mejor funciona, luego ha sido el que se ha escogido para los clasificadores que se presentan en el siguiente punto.

### **3.2. Modelos de clasificación:**

Las técnicas de vectorización planteadas se han utilizado para adecuar las noticias de nuestra base de datos a entradas válidas que sirvan para 3 modelos que se han evaluado: 2 algoritmos de Scikit-Learn (Regresión Logística y SVM) y una red neuronal que hemos diseñado. Por otra parte, se ha utilizado fine-tunning sobre el modelo de RoBERTa hallado en el anterior punto (aunque aquí no se utilizan vectorizaciones). En cada uno de los casos, se han ido utilizando diferentes métricas de evaluación de resultados destacando *accuracy* para Regresión Logística y SVM y *matriz de confusión*, *f1-score* y *recall* para la red neuronal y modelo con fine-tunning.

Además, se ha dividido la base de datos en 3 conjuntos de *train*,*test* y *validation*, con una cantidad de noticias del 60%, 20% y 20% del total respectivamente. Así, tanto para la red neuronal como para Regresión Logística y SVM, se han utilizado los modelos definidos en el punto anterior aplicados sobre los tres conjuntos, de forma que cada técnica de vectorización tenga su dataset de *train*,*test* y *validation*.

- ***Regresión Logística:*** Es un modelo estadístico lineal que predice la probabilidad de que una instancia pertenezca a una clase usando una función sigmoide. A priori es el clasificador más simple de los utilizados, pero como veremos en el siguiente punto, las prestaciones que ofrece son realmente buenas.

- ***Supporrt Vector Machine (SVM):*** Es un modelo de aprendizaje supervisado que busca un límite óptimo que separe las clases con el mayor margen posible (en este caso nuestras noticias *true*/*false*). Se ha escogido por ser un modelo bastante diferente del de Regresión Logística y, en principio, más complejos. En el siguiente apartado se ofrecen los resultados obtenidos en comparación con Regresión Logística.
  
- ***Red Neuronal:*** En este caso se ha diseñado un modelo supervisado compuesto por capas de nodos (neuronas) para clasificar noticias sobre COVID-19 como verdaderas o falsas usando vectores TF-IDF, Word2Vec y Embeddings Contextuales (RoBERTa).

  Hemos implementado una arquitectura con tres capas ocultas (de 1024, 512 y 128 neuronas), con ReLU, Batch Normalization, Dropout progresivo para prevenir sobreajuste y una capa de salida con sigmoide para clasificación binaria. Dado que es binaria, se ha establecido la red con una única salida para reducir la complejidad de la misma, pues es suficiente para deducir si la noticia es verdadera (salida=1) o falsa (salida=0).

  Para el entrenamiento, hemos utilizado un optimizador Adam y BCEWithLogitsLoss como función de pérdidas, que combina sigmoide + Binary Cross Entropy. Los datos se procesan en mini-batches con validación para controlar el rendimiento y se han ido ajustando los parámetros de learning rate, número de neuronas de las capas ocultas, dropout, weight decay, etc. hasta obtener los resultados que se presentan en la siguiente sección.
  
- ***RoBERTa+Fine-Tunning:*** Se ha realizado fine-tunning al modelo preentrenado de RoBERTa. El fine-tuning consiste en ajustar un modelo preentrenado a una tarea específica usando tus propios datos. En nuestro caso, hemos entrenado RoBERTa con noticias sobre COVID-19 para que aprenda a clasificar automáticamente noticias verdaderas y falsas, combinando su conocimiento general del lenguaje con los patrones específicos de nuestro dataset. Los resultados han sido positivos y, nuevamente, se presentan y comparan en la siguiente sección.
  
- ***TRABAJO DE EXTENSIÓN:*** Como trabajo extra para complementar los resultados obtenidos, se ha realizado un análisis del sentimiento (positivo, negativo o neutro) para cada una de las noticias. el objetivo es verificar si la hipótesis de la que partíamos, que en las noticias falsas suele predominar un sentimiento negativo (dentro de que durante el COVID-19, la mayor parte de las noticias eran negativas) es verdad o no. Para ello, se han realizado 2 análisis con 2 librerías diferentes: VADER y Flair. Con la primera se han analizado los títulos de las noticias, que hasta este momento no se habían utilizado, clasificando el sentimiento como positivo, negativo o neutro. Con Flair, se han categorizado las noticias analizando el cuerpo de las mismas y suprimiendo la categoría neutra. El motivo por el que se han utilizado librerías diferentes es que con VADER, para textos muy largos (como el caso de nuestras noticias), el sentimiento suele caer en neutro y por lo tanto no íbamos a poder extraer demasiadas conclusiones. Esta información nos permite explorar patrones adicionales que podrían ayudar a distinguir noticias verdaderas de falsas.







