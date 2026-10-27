# 4. Proyecto básico (2,25 puntos)

Para obtener la calificación mínima de aprobado, el trabajo debe incluir todos los apartados siguientes.

## 4.1. Análisis exploratorio

Antes de aplicar cualquier técnica de procesamiento del lenguaje, realizad un breve análisis exploratorio:

- [ ] Descripción general: número de instancias, variables, tipos de datos y valores ausentes.
- [ ] Distribución de la variable objetivo una vez construida según la Sección 3. ¿Cuántas clases quedan y cuál es su grado de desbalance? ¿Qué rendimiento obtendría un clasificador que siempre predijera la clase mayoritaria? Ese valor es el **baseline** que todo lo demás debe superar.
- [ ] Estadísticas básicas y visualizaciones sencillas: longitud de los resúmenes, proyectos por año y por país.
- [ ] Análisis preliminar del texto: términos más frecuentes, nube de palabras y ejemplos por clase.

## 4.2. Representación vectorial

Construid **tres** representaciones diferentes de los documentos:

- [ ] **TF-IDF**: frecuencia de término y frecuencia inversa de documento.
- [ ] **Word2Vec**: media de los embeddings de las palabras del documento. Indicad si entrenáis los embeddings con este corpus o usáis vectores preentrenados, y cómo tratáis las palabras fuera de vocabulario (OOV).
- [ ] **Embeddings contextuales** de Transformers preentrenados *encoder-only* (BERT, RoBERTa...). Explicad cómo convertís la salida a nivel de token en un único vector por documento y qué hacéis con los resúmenes que superan la longitud máxima de entrada del modelo.

Notas:

- Para TF-IDF es necesario implementar un pipeline de preprocesamiento (NLTK, Gensim, SpaCy u otra). Documentad las decisiones: minúsculas, stopwords, lematización, frecuencias mínima y máxima de documento.
- El pipeline de preprocesamiento es opcional para Word2Vec si entrenáis vosotros los embeddings.
- Para los embeddings contextuales debe usarse **únicamente el texto sin preprocesar**.
- El Apéndice A (`05_apendice_A_representaciones.md`) resume TF-IDF y Word2Vec.

## 4.3. Modelado

Entrenad, ajustad y evaluad modelos de clasificación con, como mínimo:

- [ ] Al menos un algoritmo de **Scikit-learn** (k-NN, SVM, random forest, regresión logística...).
- [ ] Una **red neuronal en PyTorch**, diseñada e implementada por vosotros.
- [ ] Comparación con un **Transformer preentrenado ajustado mediante fine-tuning** con Hugging Face Transformers.

Requisitos de protocolo:

- La selección de hiperparámetros debe usar un procedimiento de validación adecuado, **sin usar el test** para tomar decisiones de modelado.
- El informe debe indicar claramente el **tamaño de train, validación y test**.
- El código debe permitir reproducir los experimentos.

**Fine-tuning vs. embeddings contextuales:** en los embeddings el Transformer actúa como extractor de características y sus pesos permanecen congelados; en el fine-tuning los parámetros del Transformer se actualizan con las etiquetas del problema. El fine-tuning de un modelo base sobre decenas de miles de resúmenes es viable con una GPU gratuita de Google Colab; no se espera ejecutarlo en el ordenador personal.

## 4.4. Evaluación comparativa

Se espera que:

- [ ] Evaluéis con métricas adecuadas para multiclase **desbalanceada**, seleccionadas y justificadas. El accuracy por sí solo no basta: deben calcularse y usarse en la discusión el **macro-F1** y el **desglose por clase**.
- [ ] Comparéis el rendimiento entre las distintas representaciones y entre los distintos algoritmos.
- [ ] Repitáis vuestra **mejor configuración en el primer nivel** (seis clases) y comparéis. ¿Cambia el ranking de las tres representaciones al variar la granularidad?
- [ ] **No interpretéis diferencias dentro de la incertidumbre del test.** Indicad qué magnitud debería tener una diferencia para ser significativa y aplicad ese criterio a vuestras conclusiones.
- [ ] Analicéis **dónde fallan** los modelos: ¿confusiones entre categorías hermanas de la taxonomía (razonable) o entre categorías no relacionadas?

El objetivo del informe no es solo presentar métricas, sino justificar las decisiones de modelado y analizar críticamente los resultados, apoyando las conclusiones con tablas, figuras y análisis de errores.
