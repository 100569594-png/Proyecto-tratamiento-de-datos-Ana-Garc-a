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

En el Notebook 3 se entrenan y comparan los siguientes modelos:

| Bloque | Modelos |
|---|---|
| Scikit-learn | Logistic Regression, Linear SVM, Random Forest |
| PyTorch | MLP con tres capas ocultas |
| Hugging Face | Fine-tuning de DistilBERT |

La evaluación se realiza separando claramente validation y test:

- **Validation** se usa para comparar modelos y seleccionar el mejor.
- **Test** se reserva para la evaluación final.

Las métricas utilizadas son:

- Accuracy.
- F1-score sobre la clase fake (`F1_fake`).
- ROC-AUC.

Se presta especial atención a `F1_fake`, ya que el objetivo principal es detectar correctamente noticias falsas.

---



