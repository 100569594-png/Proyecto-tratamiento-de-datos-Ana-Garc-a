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






