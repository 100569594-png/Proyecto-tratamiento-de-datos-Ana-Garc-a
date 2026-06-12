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



## 2. Conjunto de datos

Para evaluar la situación, se ha contado con el  dataset denominado **WELFake**, compuesto por artículos de noticias etiquetados como reales o falsos. Para ello, se plantea una tarea de clasificación binaria con la siguiente convención:

| Etiqueta | Clase |
|---:|---|
| `0` | Real |
| `1` | Fake |

Una vez cargado inicialmente el dataset, se realiza una limpieza valores nulos y duplicados. Este paso es  especialmente importante, debido a que la existencia de textos repetidos puede llegar a  producir solapamiento entre entrenamiento y test, inflando artificialmente las métricas.

El resultado de la limpieza es:

| Etapa | Nº noticias |
|---|---:|
| Filas iniciales | 72.134 |
| Filas eliminadas por nulos o texto vacío | 39 |
| Duplicados exactos eliminados | 8.458 |
| Filas finales | 63.637 |

La distribución de clases tras la limpieza es relativamente equilibrada:

| Clase | Nº noticias | Porcentaje |
|---|---:|---:|
| Real | 34.790 | 54,7 % |
| Fake | 28.847 | 45,3 % |

De esta forma, debido al tamaño del dataset se utiliza la siguiente división estratégica **80/10/10**:

| Split | Nº noticias | Uso |
|---|---:|---|
| Train | 50.909 | Ajuste de vectorizadores y entrenamiento |
| Validation | 6.364 | Comparación y selección de modelos |
| Test | 6.364 | Evaluación final |

Esta división de los datos fue la elegida debido a que el dataset es bastante grande, donde validation y test mantienen más de 6.000 ejemplos cada uno, mientras que se conserva el mayor número posible de noticias para entrenar modelos complejos, especialmente el fine-tuning de Transformers.

Además, se verifica explícitamente que no existe solapamiento entre particiones:

| Comparación | Solapamiento |
|---|---:|
| Train-Val | 0 |
| Train-Test | 0 |
| Val-Test | 0 |

---

## 3. Análisis y control de leakage

A lo largo del  Notebook 1 se ha realizado un análisis detallado del dataset: distribución de clases, longitudes de texto, palabras frecuentes, nubes de palabras y ejemplos por clase.

La parte central del análisis consiste en revisar posibles fuentes de **data leakage**, ya que en un dataset de estas caracteristicas  es especialmente importante comprobar si existen palabras que simplificaran artificialmente la tarea, como nombres de fuentes muy parecidos a una clase.

### 3.1. Posibles atajos de fuente

El análisis detecta términos de fuente muy asociados a una clase. Como por ejemplo:

| Término | Interpretación |
|---|---|
| `reuters` | Aparece mayoritariamente en noticias reales |
| `infowars` | Aparece mayoritariamente en noticias fake |
| `natural news` | Asociado a noticias fake |
| `yournewswire` | Asociado a noticias fake |
| `beforeitsnews` | Asociado a noticias fake |

Estos términos pueden actuar como atajos directos,  el modelo podría clasificar por fuente en lugar de aprender patrones generales de desinformación.

### 3.2. Posibles palabras-etiqueta

De la misma forma se revisan las palabras que podrían coincidir directamente con las etiquetas del problema:

```text
fake, false, partially false, true, real, label, category
```

Estas palabras no aparecen de forma completamente exclusiva en una sola clase, por lo que no constituyen una fuga directa absoluta. Sin embargo, se eliminan posteriormente como medida conservadora.

### 3.3. Mitigación aplicada

Con el objetivo de reducir el riesgo de aprendizaje por atajos, se eliminan términos de fuente y palabras-etiqueta de alto riesgo en las técnicas utilizadas.

- TF-IDF.
- Word2Vec.
- Embeddings BERT.
- Fine-tuning de DistilBERT.

Esta mitigación reduce el riesgo de leakage explícito, aunque se reconoce como limitación que pueden permanecer sesgos indirectos de fuente, estilo o temática.


### 4.1. Representaciones vectoriales 

Antes de aplicar los modelos de clasificación, es necesario transformar cada noticia formada por carácteres textuales en una representación numérica. Esto se realiza debido a que los algoritmos de machine learning no trabajan directamente con palabras o frases, por lo que cada noticia en su conjunto debe ser convertida en un vector. Asimismo, se representará el texto mediante las siguientes técnicas: **TF-IDF**, **Word2Vec** y **embeddings contextuales BERT**.

La comparación entre estas representaciones es una parte central del trabajo, porque cada técnica captura un tipo de información diferente:

| Representación | Carácteristica principal | Tipo de vector | Ventaja  | Limitación |
|---|---|---|---|---|
| TF-IDF | Importancia estadística de palabras y bigramas | Disperso y de alta dimensión |Altamente eficaz para señales léxicas | No entiende contexto ni orden |
| Word2Vec | Similitud semántica entre palabras | Denso y de baja dimensión | Agrupa palabras con significado similar| Pierde estructura al promediar palabras |
| BERT embeddings | Información contextual| Denso contextual | Captura información semántica y contextual | Coste computacional y truncamiento |

Además, en cada representación se controla el posible uso de términos asociados a leakage, como nombres de fuentes o palabras cercanas a la etiqueta (`reuters`, `infowars`, `fake`, `false`, `true`, `real`, etc.). El objetivo es evitar que los modelos aprendan únicamente atajos evidentes y forzar una comparación más realista entre métodos.

#### 4.1.1. TF-IDF

Convierte cada noticia en un vector donde cada posición representa una palabra o combinación de palabras, y el valor asociado mide su importancia relativa dentro del corpus.

- Si una palabra aparece muchas veces en una noticia, puede ser relevante para describirla.
- Si esa palabra aparece en casi todas las noticias, deja de ser informativa y se penaliza.

Por ejemplo, palabras muy frecuentes y generales como `the`, `said` o `people` aportan poca información discriminante. En cambio, términos más específicos o combinaciones de palabras pueden ayudar a diferenciar noticias reales y falsas.

Se toman los siguientes parámetros para definir la representación. 

| Parámetro | Valor | Justificación |
|---|---:|---|
| `max_features` | 50.000 | Conserva un vocabulario amplio sin hacerlo inmanejable |
| `min_df` | 3 | Elimina términos demasiado especificos |
| `max_df` | 0.90 | Elimina términos demasiado frecuentes y poco informativos |
| `ngram_range` | (1, 2) | Incluye palabras individuales y pares de palabras |
| `sublinear_tf` | True | Suaviza el efecto de palabras repetidas muchas veces |

La elección de **50.000 términos** no es arbitraria. En el Notebook 2 se analiza el vocabulario disponible tras aplicar filtros de frecuencia y se comprueba que el corpus contiene un número elevado de términos útiles. Reducir demasiado el vocabulario podría eliminar señales relevantes, mientras que mantenerlo completo aumentaría el coste computacional y el ruido.

Esta técnica se ajusta únicamente con el conjunto de entrenamiento, evitando que el vocabulario o las frecuencias del conjunto de test influyan en el entrenamiento. Posteriormente validation y test se transforman usando exactamente el vocabulario aprendido en train.

TF-IDF acaba siendo una de las representaciones con mejores resultados del dataset. Combinando el alto rendimiento de **SVM + TF-IDF** y **Logistic Regression + TF-IDF**, donde se señala de forma directa como el conjunto de noticias tienen un conjunto de señales léxicas fuertes.

#### 4.1.2. Word2Vec

Esta técnica de embeddings aprende vectores densos para palabras. A diferencia de TF-IDF, Word2Vec no representa una palabra por su frecuencia aislada, sino por los contextos en los que aparece.

La intuición es que las palabras que aparecen en contextos parecidos tienden a tener significados relacionados. Por ejemplo, los términos como `president`, `government` o  `election` pueden quedar cercanos en el espacio vectorial debido a que suelen aparecer en noticias de temática política.

Word2Vec se entrenará  con los textos del conjunto de entrenamiento. Cada palabra obtiene un vector de dimensión **300**, y posteriormente cada noticia se representa como el promedio de los vectores de las palabras que contiene.

| Parámetro                   |         Valor  | Justificación                                                                                                                                                                        |
| --------------------------- | ---------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `vector_size`               |                  `300` | Cada palabra se representa mediante un vector denso de 300 dimensiones. Es un tamaño que permite capturar relaciones semánticas sin generar vectores excesivamente grandes. |
| `window`                    |                    `5` | Se considera un contexto de 5 palabras alrededor de cada término. Este valor permite aprender relaciones locales entre palabras sin introducir demasiado ruido contextual.    |
| `min_count`                 |                    `3` | Se ignoran palabras que aparecen menos de 3 veces.                                |
| `sg`                        |                    `1` | Capturar relaciones semánticas de palabras menos frecuentes.     |
| `epochs`                    |                    `5` | Se entrena con 5 épocas, suficiente para aprender relaciones semánticas generales sin aumentar excesivamente el coste computacional.                                                 |
| `workers`                   |                    `4` | Paralelización para acelerar el entrenamiento.                                                                                                                                |
| `seed`                      |            `SEED = 42` | Reproducibilidad de resultados.                                                                                                                                        |
| Corpus de entrenamiento     |     Solo `X_train_tok` | Word2Vec se entrena únicamente con los textos tokenizados de train. Validation y test solo se transforman usando el vocabulario aprendido.                                           |
| Representación de documento | Promedio de embeddings | Cada noticia se representa como la media de los vectores de sus palabras conocidas. Es una forma sencilla y eficiente de pasar de embeddings de palabra a embeddings de documento.   |

La ventaja de esta técnica es que genera vectores densos y mucho más compactos que TF-IDF. Cada noticia se representa mediante un vector de 300 componentes, reduciendo el coste computacional y permitiendo capturar cierta información semántica.

Sin embargo, esta técnica tiene una limitación, y es que al promediar los vectores de las palabras, se pierde gran parte de la estructura del texto. El modelo deja de saber qué palabra aparece antes, qué palabras forman una frase, qué relaciones sintácticas existen o qué partes del documento son más importantes.

Por ejemplo, dos noticias con palabras similares pero con sentido opuesto podrían terminar teniendo vectores parecidos si se usa simplemente el promedio. Por eso Word2Vec obtiene buenos resultados, pero queda por debajo de TF-IDF y de BERT fine-tuned.Aunque es destacable que Word2Vec es muy útil como representación intermedia, ya que es más semántica que TF-IDF, pero menos compleja y costosa que BERT. 

#### 4.1.3. Embeddings contextuales BERT

Se trata de la tercera representación empleada,  generada con `distilbert-base-uncased`, cuyo objetivo es obtener una representación contextual de cada una de las noticias. DistilBERT es utilizado como extractor de características, donde genera vectores teniendo en cuenta el contexto en el que aparecen las palabras.   

La representación de una palabra depende de la frase en la que aparece. Por ejemplo, una palabra como `claim` no tiene exactamente el mismo significado en todos los contextos; BERT puede generar representaciones diferentes según el uso concreto.

| Parámetro         |              Valor  | Justificación                                                                                                                                                                         |
| ----------------- | --------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MODEL_NAME`      | `'distilbert-base-uncased'` |  DistilBERT  es una versión más ligera de BERT,  menor coste computacional y buena capacidad para capturar contexto lingüístico.                                   |
| `MAX_LEN`         |                       `256` | Se limita cada noticia a 256 tokens. Compromiso entre conservar suficiente información del texto y poder procesar el dataset completo. 
| `BATCH_SIZE`      |                        `16` | Procesado de varios textos en paralelo manteniendo el uso de memoria GPU bajo control.                                                                                            |
| `N_SAMPLE`        |                      `3000` |Genera una muestra de embeddings para visualizaciones exploratorias como PCA/t-SNE sin tener que representar todo el dataset en esas gráficas.                                     |
| Embedding usado   |               Token `[CLS]` | Se usa el embedding del primer token como representación global del documento.            |
| `truncation`      |                      `True` | Los textos que superan `MAX_LEN` se recortan para tener longitudes manejables.                                                                                     |
| `padding`         |                      `True` | Se rellenan los textos de cada batch para que tengan la misma longitud y puedan procesarse.                                                                                    |
| `torch.no_grad()` |                    Activado | Se extraen embeddings sin calcular gradientes,  DistilBERT es un extractor fijo y no se entrena.                                                             |
| Limpieza previa   |    Limpieza mínima filtrada | Se eliminan URLs y términos de fuente/etiqueta de alto riesgo, pero se conserva la estructura general del texto. Esto evita repetir el error de aplicar una limpieza agresiva a BERT. |


De esta forma, cada noticia se tokeniza y se pasa por el modelo preentrenado. Después se obtiene un vector de dimensión **768**, asociado a la representación global del documento.


A diferencia del preprocesamiento clásico, para BERT se utiliza una **limpieza mínima**. Esto es importante porque los Transformers están preentrenados con texto natural y pueden aprovechar puntuación, estructura y contexto. Una limpieza demasiado agresiva podría eliminar información útil. Por eso no se aplica el mismo preprocesamiento que en TF-IDF o Word2Vec.

Cabe destacar que aunque se utiliza una limpieza mínima para no eliminar información útil, sí se eliminan términos de fuente y palabras-etiqueta de alto riesgo para reducir posibles atajos. De esta forma, el texto usado por BERT conserva su estructura, pero se filtran términos como:

```text
reuters, infowars, natural news, fake, false, true, real, label, category
```

A lo largo de la tarea de clasificación, la representación BERT tiene dos usos distintos:

1. **Embeddings BERT:** se usa DistilBERT como extractor fijo de vectores y después se entrenan modelos externos, como Logistic Regression, SVM o MLP.
2. **Fine-tuning de DistilBERT:** se ajusta directamente el modelo Transformer completo para la tarea de clasificación.

Esta diferencia es fundamental para interpretar los resultados. Los embeddings BERT son útiles, pero no siempre superan a TF-IDF. En cambio, el fine-tuning de DistilBERT obtiene el mejor resultado global porque adapta el modelo completo al dataset WELFake.


#### 4.1.4. Comparación metodológica entre representaciones

Las tres representaciones permiten analizar el problema desde perspectivas distintas:

- **TF-IDF**  señala las palabras y expresiones  que  aparecen y su importancia estadística. 
- **Word2Vec** indica el  significado aproximado tienen las palabras dentro del documento.
- **BERT** dicta que información contextual y semántica contiene el texto completo.

Por ende, no se trata  unicamente de escoger la técnica más compleja, sino de comprobar qué tipo de información resulta más útil para el dataset empleado.

Los resultados muestran que TF-IDF es muy competitivo porque el dataset contiene señales léxicas fuertes. Word2Vec funciona peor porque al promediar palabras pierde información estructural. BERT como extractor fijo obtiene buenos resultados, pero su máximo potencial aparece cuando se realiza fine-tuning.

En conjunto, esta comparación permite concluir que la representación del texto tiene un impacto directo en el rendimiento del clasificador. En WELFake, las señales léxicas son muy potentes, pero el mejor rendimiento se alcanza cuando se combinan representación contextual y ajuste supervisado mediante DistilBERT fine-tuned.
---

## 5. Modelos de clasificación

Una vez obtenidas las representaciones vectoriales del texto, se pasa  al entrenamiento, comparación y evaluacion de distintos modelos de clasificación supervisada. 

El objetivo es analizar cómo cambia el rendimiento al aplicar la combinación de diferentes tipos de representaciones con modelos de complejidad distinta.

El problema se plantea como una clasificación binaria:

| Etiqueta | Clase |
|---:|---|
| `0` | Real |
| `1` | Fake |

El flujo seguido en cada bloque  es el mismo. Cada modelo se entrena con el conjunto de **train**, se compara con el conjunto de **validation** y se evalúa finalmente en **test**. Esta separación es importante porque evita escoger el mejor modelo mirando directamente el conjunto de test.

| Conjunto | Uso |
|---|---|
| Train | Entrenar los modelos y ajustar parámetros internos |
| Validation | Comparar modelos y seleccionar la mejor configuración |
| Test | Evaluación final, una vez elegido el modelo |

Las métricas utilizadas son **Accuracy**, **F1-score de la clase fake** y **ROC-AUC**. Aunque la accuracy permite medir el porcentaje global de aciertos, en este proyecto se presta una mayor atención  `F1_fake`, debido a  que el objetivo principal es detectar correctamente noticias falsas. Esta métrica combina precisión y recall de la clase fake, por lo que penaliza tanto clasificar noticias reales como falsas como no detectar noticias falsas reales.

### 5.1. Modelos clásicos de Scikit-learn

Para empezar, se entrenan tres modelos  de aprendizaje supervisado: **Logistic Regression**, **Linear SVM** y **Random Forest**. Esto permitirá realizar una comparación robústa frente a modelos neuronales complejos.

| Modelo | Representación | Utilización|
|---|---|---|
| Logistic Regression | TF-IDF, Word2Vec, BERT embeddings | Base lineal, sencilla y interpretable |
| Linear SVM | TF-IDF, Word2Vec, BERT embeddings | Modelo lineal eficaz en texto de alta dimensión |
| Random Forest | TF-IDF SVD300, Word2Vec, BERT embeddings | Modelo no lineal basado en árboles, contraste de información|

#### Logistic Regression

Se emplea como primer clasificador de referencia. Es modelo relativamente simple que resulta adecuado para problemas de texto, especialmente cuando se combinan representaciones del tipo TF-IDF. 

| Parámetro | Valor | Justificación |
|---|---:|---|
| `penalty` | `l2` | Reduce sobreajuste penalizando pesos excesivamente grandes |
| `C` | `1.0` | Valor estándar, mantiene un equilibrio entre ajuste y regularización |
| `solver` | `saga` | Adecuado para datasets grandes y matrices dispersas como TF-IDF |
| `max_iter` | `2000` | Permite la convergencia con representaciones de alta dimensión |
| `n_jobs` | `-1` | Usa todos los núcleos disponibles para acelerar el entrenamiento |

Sirve como un punto de referencia en la evaluación.  Si un modelo más complejo no supera a Logistic Regression, entonces su mayor coste computacional no estaría justificado. En los resultados finales, Logistic Regression con TF-IDF obtiene un rendimiento muy alto, lo que confirma que WELFake contiene señales léxicas muy discriminativas.

#### Linear SVM

Este clasificador busca una frontera de separación con el mayor margen posible entre clases. Además, es especialmente útil en clasificación de texto debido a que no le supone un esfuerzo trabajar con vectores de alta dimensión, como los generados por TF-IDF.


| Parámetro | Valor | Justificación |
|---|---:|---|
| `max_iter` | `5000` | Aumenta el número máximo de iteraciones para asegurar convergencia |
| `random_state` | `42` | Permite reproducir los resultados |
| Tipo de SVM | Lineal | Adecuado para texto de alta dimensión |

Al evaluar la clasificación del dataset en su totalidad, Linear SVM toma un papel protagonista ya que termina siendo el mejor modelo clásico. El buen redimiento con TF-IDF indica que gran parte de la información necesaria para clasificar noticias reales y falsas está presente en la distribución de palabras y bigramas. 

#### Random Forest

Se incluye como modelo no lineal que combina múltiples árboles entrenados sobre subconjuntos de datos y características. 

De esta forma, no es el modelo más natural para texto de alta dimensión. Por lo tanto, cuando se utiliza con TF-IDF no se emplea la matriz completa de 50.000 términos, sino una versión reducida. Esta reducción hace que el entrenamiento sea más manejable y comparable en tamaño a Word2Vec.

| Parámetro | Valor | Justificación |
|---|---:|---|
| `n_estimators` | `300` | Número suficiente de árboles para estabilizar el modelo |
| `min_samples_leaf` | `2` | Reduce sobreajuste evitando hojas con muy pocos ejemplos |
| `class_weight` | `balanced` | Compensa posibles diferencias de frecuencia entre clases |
| TF-IDF usado | `SVD300` | Reduce coste computacional y dimensionalidad |
| `n_jobs` | `-1` | Paraleliza el entrenamiento |

Random Forest se incluye como contraste frente a modelos lineales y neuronales. Obtiene resultados  más bajos, lo cual es esperable, ya quelos modelos basados en árboles no suelen explotar tan bien las representaciones textuales dispersas o embeddings de alta dimensión.

### 5.2. Red neuronal MLP en PyTorch

Se implementa una **red neuronal multicapa** en PyTorch. Esta red se aplica sobre tres representaciones: TF-IDF reducido mediante SVD300, Word2Vec y embeddings BERT.

La arquitectura utilizada tiene tres capas ocultas:

```text
Entrada → 512 neuronas → 256 neuronas → 128 neuronas → salida binaria
```


| Elemento | Valor | Justificación |
|---|---:|---|
| Capas ocultas | `(512, 256, 128)` | Arquitectura que comprime progresivamente la información |
| Activación | `ReLU` | Introduce no linealidad y facilita el entrenamiento |
| Batch Normalization | Sí | Estabiliza el aprendizaje entre capas |
| Dropout | `0.3` | Reduce sobreajuste apagando neuronas aleatoriamente |
| Función de pérdida | `BCEWithLogitsLoss` | Adecuada para clasificación binaria, estable numéricamente |
| Optimizador | `Adam` | Optimizador robusto y eficiente para redes neuronales |
| Learning rate | `1e-3` | Valor habitual para MLP con Adam |
| Batch size | `512` | Permite entrenamiento eficiente en GPU/CPU |
| Early stopping | `patience = 4` | Detiene el entrenamiento si no mejora la validación |
| Scheduler | `ReduceLROnPlateau` | Reduce el learning rate si la pérdida de validación se estanca |

La red neuronal multicapa hace posible la evaluación entre  una arquitectura neuronal simple y modelos clásicos. 

Al analizar los resultados, la red neuronal funciona muy bien con embeddings BERT y con TF-IDF reducido,  indicando  que es capaz de aprovechar representaciones densas y no lineales. Sin embargo, no llega a superar al fine-tuning de DistilBERT, porque el MLP solo aprende sobre vectores ya generados, mientras que el fine-tuning ajusta directamente el modelo de lenguaje completo.

### 5.3. Fine-tuning de DistilBERT

Se trata del modelo más complejo del proyecto, **fine-tuning de DistilBERT** mediante Hugging Face Transformers.Para ello, el texto tokenizado entra directamente al Transformer y se ajustan los pesos del modelo preentrenado para la tarea concreta de clasificar noticias reales y falsas.


| Parámetro | Valor | Justificación |
|---|---:|---|
| Modelo base | `distilbert-base-uncased` | Transformer ligero y eficiente para fine-tuning |
| `MAX_LEN` | `256` | Compromiso entre conservar información y limitar memoria GPU |
| Batch size | `8` en GPU | Evita saturar memoria durante el cálculo de gradientes |
| Épocas | `3` | Adaptar el modelo sin entrenar en exceso |
| Learning rate | `2e-5` | Valor típico y estable para fine-tuning de Transformers |
| Weight decay | `0.01` | Regularización para reducir sobreajuste |
| Warmup | 10 % de pasos | Estabiliza el inicio del entrenamiento |
| Evaluación | Cada época | Permite monitorizar validation en el entrenamiento |
| Selección del modelo | Mejor `F1` en validation | Prioriza la detección de noticias fake |
| `fp16` | Activado con GPU | Reduce memoria y acelera entrenamiento |

Antes del fine-tuning se aplica una limpieza mínima filtrada, eliminando URLs y términos de alto riesgo asociados a fuentes o etiquetas, como `reuters`, `infowars`, `fake`, `false`, `true` o `real`, pero se conserva el resto de la estructura textual. Reduciendo atajos explícitos sin destruir el contexto.

El fine-tuning es el enfoque que obtiene el mejor resultado global porque adapta el modelo completo a WELFake. Mientras que los embeddings BERT solo usan el conocimiento lingüístico general del modelo preentrenado, el fine-tuning permite que DistilBERT aprenda patrones específicos del dataset, incluyendo estilo, estructura, temática y relaciones contextuales.

### 5.4. Comparación metodológica entre los modelos

Los modelos se han escogido para cubrir distintos niveles de complejidad:

          
| Modelo | Tipo de enfoque | Justificación|
| --- | --- | --- |
| **Logistic Regression** | Modelo lineal clásico | Permite comprobar si una relación lineal entre las palabras y la clase es suficiente para clasificar noticias reales y falsas. |
| **Linear SVM** | Modelo lineal de margen máximo |Buen funcionamiento con TF-IDF y permite comprobar si maximizar el margen mejora la clasificación frente a Logistic Regression. |
| **Random Forest** | Modelo no lineal basado en árboles | Permite evaluar si combinar muchos árboles aporta ventajas frente a modelos lineales en representaciones reducidas como Word2Vec, BERT o TF-IDF con SVD. |
| **MLP PyTorch** | Red neuronal sobre vectores | Sirve para comprobar si una red neuronal densa puede aprovechar mejor las representaciones vectoriales que los clasificadores clásicos. |
| **DistilBERT fine-tuning** | Transformer ajustado extremo a extremo | Permite evaluar si adaptar un modelo de lenguaje completo a la tarea mejora de forma evidente los resultados |                  |


Esta comparación es especialmente importante para decidir si realmente  para detectar desinformación  es suficiente con técnicas estadísticas clásicas o si realmente es necesario utilizar modelos de lenguaje avanzados. 

Los resultados muestran la siguiente conclusión. Por un lado, los modelos clásicos, como **Linear SVM con TF-IDF**, ofrece un rendimiento muy alto y son más baratos computacionalmente. Por otro lado, el mejor resultado global se alcanza con **DistilBERT fine-tuned**, lo que confirma que adaptar un Transformer a la tarea permite capturar patrones más complejos.

Por tanto, la elección del modelo depende del objetivo. Si se busca eficiencia, interpretabilidad y bajo coste, SVM con TF-IDF es una opción muy sólida. Si se busca maximizar el rendimiento y se dispone de GPU, el fine-tuning de DistilBERT es la alternativa más potente.


## 6. Resultados experimentales

### 6.1. Resultados finales en test

| Ranking | Modelo | Representación | Accuracy | F1_fake | ROC-AUC |
|---:|---|---|---:|---:|---:|
| 1 | DistilBERT fine-tuning | Texto filtrado | 0.9943 | 0.9938 | 0.9998 |
| 2 | Linear SVM | TF-IDF | 0.9667 | 0.9631 | 0.9942 |
| 3 | MLP PyTorch | BERT-emb | 0.9648 | 0.9609 | 0.9944 |
| 4 | MLP PyTorch | TF-IDF SVD300 | 0.9580 | 0.9536 | 0.9930 |
| 5 | Logistic Regression | TF-IDF | 0.9569 | 0.9522 | 0.9911 |
| 6 | MLP PyTorch | Word2Vec | 0.9452 | 0.9384 | 0.9875 |
| 7 | Linear SVM | BERT-emb | 0.9408 | 0.9343 | 0.9839 |
| 8 | Logistic Regression | BERT-emb | 0.9408 | 0.9343 | 0.9837 |
| 9 | Random Forest | TF-IDF SVD300 | 0.9230 | 0.9128 | 0.9756 |
| 10 | Linear SVM | Word2Vec | 0.9178 | 0.9085 | 0.9714 |
| 11 | Logistic Regression | Word2Vec | 0.9136 | 0.9035 | 0.9702 |
| 12 | Random Forest | BERT-emb | 0.8905 | 0.8772 | 0.9571 |
| 13 | Random Forest | Word2Vec | 0.8875 | 0.8753 | 0.9559 |

### 6.2. Interpretación de resultados

El mejor modelo global es **DistilBERT fine-tuned sobre texto filtrado**, con `F1_fake = 0.9938`. Este resultado indica que el ajuste supervisado del Transformer permite capturar patrones textuales muy discriminativos del dataset.

El mejor modelo clásico es **Linear SVM con TF-IDF**, con `F1_fake = 0.9631`. Esto confirma que TF-IDF es una representación muy fuerte para WELFake, debido a que existen señales léxicas importantes entre noticias reales y falsas.

Los embeddings BERT también son útiles, especialmente al ser utilizados con la red neuronal MLP, sin embargo, no llegan a  alcanzar el rendimiento del fine-tuning. Esto demuestra la utilización de BERT como extractor fijo no es equivalente a ajustar el Transformer completo a la tarea.

Por otro lado, Word2Vec queda por debajo de TF-IDF y BERT. Este comportamiento es esperable porque representa cada noticia mediante el promedio de embeddings de palabras, perdiendo orden, estructura y contexto global.

Random Forest es el modelo más débil. Esto no es sorprendente, debido a que no suele ser el método más adecuado para texto de alta dimensión. Además, para TF-IDF se usa una reducción SVD a 300 dimensiones por razones computacionales, lo que hace que su comparación con Logistic Regression y SVM no sea completamente equivalente.

---

## 7. Pruebas de robustez frente a leakage

Dado que los resultados son muy elevados, se realizan comprobaciones adicionales para descartar fugas de información.

El objetivo es  evaluar la fiabilidad metodológica del trabajo y comprobar si los modelos están aprendiendo patrones reales del texto. 

### 7.1. Entrenamiento con etiquetas aleatorias

La primera prueba consiste en entrenar un modelo utilizando las etiquetas de entrenamiento mezcladas aleatoriamente. De esta forma se rompe la relación real entre cada noticia y su clase.

| Prueba | Accuracy | F1_fake | ROC-AUC |
|---|---:|---:|---:|
| Etiquetas aleatorias | 0.5085 | 0.3518 | 0.4817 |

Esto indica que el pipeline no aprende patrones útiles cuando se rompe la relación entre texto y etiqueta.

### 7.2. Baseline basada solo en términos de fuente o etiqueta

La segunda prueba mide cuánto rendimiento se puede obtener empleando únicamente la lista de términos potencialmente problemáticos, como nombres de fuentes o dominios relacionados con  determinadas clases, así como palabras cercanas a las etiquetas

| Prueba | Accuracy | F1_fake | ROC-AUC |
|---|---:|---:|---:|
| Solo términos fuente/etiqueta | 0.7792 | 0.7997 | 0.8137 |

Este resultado es relativamente alto, lo que confirma que el dataset contiene señales de fuente y etiqueta. Sin embargo, queda claramente por debajo de SVM + TF-IDF y DistilBERT fine-tuned. Por tanto, estos términos explican parte del sesgo del dataset, pero no justifican por sí solos el rendimiento final.

### 7.3. Comprobación del filtrado BERT

Se comprueba que los términos de alto riesgo no aparecen en el texto filtrado usado por BERT. El resultado es:

| Prueba | Resultado |
|---|---|
| Términos restantes tras filtrado | 0 |

Esto confirma que el fine-tuning no recibe directamente los términos explícitos definidos como alto riesgo.


## 8. Proyecto de extensión: análisis temático y polarización lingüística

Como trabajo de extensión se realiza un análisis de interpretación del dataset mediante clustering y medición de indicadores lingüísticos de polarización.

El objetivo  es estudiar si las noticias falsas se concentran en determinados grupos temáticos y si presentan mayor carga política, emocional o confrontativa.

### 8.1. Clustering temático con BERT

Se utilizan los embeddings BERT generados en el Notebook 2 y se aplica KMeans. El número de clusters se selecciona de forma exploratoria mediante silhouette score, obteniendo:

```text
k = 6
```

El clustering se ajusta solo sobre train y después se asignan clusters a validation y test.

La distribución de noticias fake por cluster muestra que la desinformación no se reparte de forma homogénea:

| Cluster | % Fake |
|---:|---:|
| 5 | 82,68 % |
| 2 | 63,47 % |
| 3 | 46,14 % |
| 1 | 25,27 % |
| 4 | 11,55 % |
| 0 | 11,15 % |

El cluster 5, muy dominado por noticias fake, contiene palabras clave relacionadas con política, por ejemplo:

```text
trump, clinton, hillary, president, donald, video, obama, twitter
```

Esto sugiere que parte de la desinformación del dataset se concentra en temas políticos polarizantes.

La silhouette obtenida es baja, por lo tanto, los clusters no deben interpretarse como categorías temáticas separadas de forma perfecta. Sin embargo, permiten una exploración útil de la estructura temática del dataset.

### 8.2. Indicadores de polarización lingüística

Se calculan tres indicadores por cada 1000 palabras:

| Indicador | Descripción |
|---|---|
| `political_rate` | Frecuencia de términos políticos |
| `conflict_rate` | Frecuencia de términos de conflicto |
| `emotive_rate` | Frecuencia de términos emocionales |

Los resultados por clase son:

| Clase | political_rate | conflict_rate | emotive_rate |
|---|---:|---:|---:|
| Fake | 23,69 | 2,63 | 1,26 |
| Real | 21,42 | 1,87 | 0,45 |

Las noticias falsas tienden a presentar más términos políticos, más vocabulario de conflicto y una carga emocional mayor que las noticias reales. 

No obstante, el análisis por clusters muestra que el lenguaje conflictivo no equivale automáticamente a desinformación, ya que algunos clusters con bajo porcentaje de fake también presentan términos de conflicto.

---


## 9. Relación con las hipótesis iniciales

Al inicio del proyecto se plantean diferentes hipótesis sobre el comportamiento esperado del dataset y sobre el rendimiento de las distintas técnicas de clasificación. A continuación se revisa cada hipótesis a partir de los resultados obtenidos en los tres notebooks.

### Hipótesis 1: Diferencias léxicas entre noticias reales y falsas

**Contexto de la hipótesis:**
Noticias reales y falsas utilizan exactamente el mismo tipo de vocabulario. En concreto, se planteaba que podían existir diferencias en las palabras frecuentes, expresiones utilizadas, nombres propios, fuentes mencionadas o estilo de redacción.

**Resultado obtenido:**
Esta hipótesis se cumple. Los modelos basados en TF-IDF obtienen resultados muy altos, especialmente Linear SVM + TF-IDF, con un `F1_fake` cercano a `0.963`. Esto indica que las palabras y combinaciones de palabras presentes en los textos contienen mucha información para separar noticias reales y falsas.

---

### Hipótesis 2: Clasificación automática de modelos supervisados. 

**Contexto de la hipótesis:**
Se plantea que en caso de la existencia de  patrones lingüísticos y temáticos consistentes entre noticias reales y falsas, los modelos de aprendizaje supervisado serían capaces de aprenderlos y obtener buenos resultados.

**Resultado obtenido:**
Esta hipótesis se cumple. Todos los modelos entrenados superan ampliamente el azar, incluso las alternativas más simples. Los mejores resultados se obtienen con DistilBERT fine-tuning, seguido de Linear SVM con TF-IDF y MLP con embeddings BERT.

---

### Hipótesis 3: Ventaja de modelos contextuales cuando se ajustan directamente a la tarea.

**Contexto de la hipótesis:**
Modelos tipo BERT pueden aportar ventajas frente a representaciones más simples porque capturan el contexto de las palabras. Sin embargo, también se quería comprobar si bastaba con usar BERT como extractor de embeddings o si era necesario hacer fine-tuning.

**Resultado obtenido:**
La hipótesis se cumple, pero con matizada. Los embeddings BERT ofrecen buenos resultados, especialmente cuando se combinan con una red neuronal MLP. Sin embargo, no siempre superan a TF-IDF con modelos lineales.

La mejora más clara aparece al hacer fine-tuning de DistilBERT. En este caso, el modelo no solo genera vectores, sino que ajusta sus pesos internos al problema concreto de clasificación de noticias reales y falsas. Por eso consigue el mejor resultado global del proyecto.

---

### Hipótesis 4: Relación entre desinformación y polarización lingüística.

**Contexto de la hipótesis:**
Las noticias falsas presentan un lenguaje más emocional, político o conflictivo. Esta hipótesis se estudia especialmente en la extensión del proyecto mediante clustering temático e indicadores de polarización lingüística.

**Resultado obtenido:**///////////////////////////////////////////////////////////////
La hipótesis se cumple parcialmente. La extensión indica que las noticias falsas presentan, de media, mayor presencia de términos políticos, conflictivos y emocionales que las noticias reales. La diferencia más clara aparece en la tasa de términos emocionales, que es superior en la clase fake.

Además, el clustering con embeddings BERT muestra que la desinformación no se distribuye de forma homogénea. Algunos clusters concentran un porcentaje mucho mayor de noticias falsas, especialmente aquellos relacionados con temas políticos y figuras públicas.

No obstante, la relación no es perfecta. Algunos clusters con bajo porcentaje de fake también presentan lenguaje conflictivo, lo cual puede deberse a noticias reales sobre guerra, crisis, terrorismo o política internacional.

///////////////////////////////////////////////////////////////////////////////////

---


## 10. Conclusiones

Este proyecto demuestra como la tarea de  clasificación automática de noticias falsas en el dataset WELfake es viable y llega a alcanzar resultados muy elevados. El modelo con mejores resultados obtenidos es DistilBERT fine-tuned sobre texto filtrado, seguido de Linear SVM con TF-IDF.

Estos resultados indican  de forma directa que WELFake contiene patrones textuales, temáticos y estilísticos muy discriminativos. TF-IDF es una representación muy competitiva, lo que confirma la importancia de las señales léxicas. Sin embargo, el mejor rendimiento se obtiene al ajustar un Transformer. 

También se demuestra que los resultados altos no pueden atribuirse directamente a una fuga de información. Se eliminan duplicados, se verifica la ausencia de solapamiento entre particiones, se filtran términos de fuente y palabras-etiqueta, y se realizan pruebas de robustez adicionales.

Además, la extensión demuestra como la desinformación no se distribuye de forma  homogénea, donde algunos clusters concentran un porcentaje mucho mayor de noticias fake. Además, las fake presentan mayor carga política, emocional y confrontativa, lo que conecta el análisis de desinformación con la polarización lingüística.

---

## 11. Limitaciones

Este trabajo presenta varias limitaciones:

1. **Sesgos de fuente y estilo.** Aunque se filtran términos explícitos como `reuters`, `infowars`, `fake` o `false`, pueden permanecer señales indirectas de procedencia, estilo o temática.
2. **Generalización externa.** Los resultados se obtienen sobre WELFake y no garantizan el mismo rendimiento en otros datasets o dominios.
3. **Truncamiento en BERT.** Los modelos BERT usan una longitud máxima de 256 tokens, por lo que parte de las noticias largas puede quedar fuera.
4. **TF-IDF reducido en algunos modelos.** Random Forest y MLP usan TF-IDF reducido mediante SVD a 300 dimensiones por coste computacional, por lo que su comparación con LR/SVM no es completamente equivalente.
5. **Clustering exploratorio.** La extensión de clustering tiene una silhouette baja, por lo que los clusters deben entenderse como agrupaciones exploratorias y no como categorías temáticas cerradas.

---

