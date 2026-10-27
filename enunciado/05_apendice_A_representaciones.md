# Apéndice A. Representaciones de entrada: preprocesamiento, TF-IDF y Word2Vec

Resumen práctico de lo necesario para construir las representaciones TF-IDF y Word2Vec antes de que se cubran en clase. No sustituye a las clases.

## A.1. Preprocesamiento de texto

TF-IDF trata un documento como una bolsa de términos, por lo que la primera decisión es qué se considera un término. Un pipeline estándar aplica, en este orden:

1. **Tokenización:** dividir el texto en tokens. Una expresión regular sobre caracteres de palabra puede bastar; SpaCy o NLTK gestionan mejor guiones, acrónimos y unidades.
2. **Normalización de mayúsculas/minúsculas:** todo a minúsculas. Se pierde la distinción entre *US* y *us*, pero el vocabulario es mucho más pequeño.
3. **Eliminación de tokens no informativos:** puntuación, números, tokens de uno o dos caracteres y stopwords («the», «of», «which»), con la lista estándar en inglés de cada biblioteca.
4. **Reducción de flexiones:** mapear *networks*, *networking* y *network* a un único término. El *stemming* (Porter, Snowball) elimina sufijos, es rápido y puede producir formas que no son palabras reales; la *lematización* (SpaCy, WordNet) devuelve formas de diccionario y es más lenta. Cualquiera es aceptable; indicad cuál usasteis.
5. **Poda del vocabulario por frecuencia de documento:** eliminar términos en menos de *n* documentos (ruido, erratas, nombres aislados) y términos en casi todos los documentos (no discriminan).

Pipeline mínimo con SpaCy, aplicado a la columna `summary`:

```python
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean(doc):
    return [t.lemma_.lower() for t in doc
            if t.is_alpha and not t.is_stop and len(t) > 2]

docs = [clean(d) for d in nlp.pipe(summaries, batch_size=256, n_process=4)]
```

**Observación específica de este corpus:** los resúmenes contienen muchas expresiones repetitivas («this project», «work package», «state of the art», «consortium»). Se comportan como stopwords específicas del dominio, y la poda mediante frecuencia máxima de documento permite eliminarlas sin enumerarlas a mano.

## A.2. TF-IDF

Sea D un corpus de N documentos y V el vocabulario tras el preprocesamiento. El peso del término *t* en el documento *d* es:

```
tfidf(t, d) = tf(t, d) · idf(t),     idf(t) = log(N / df(t))
```

donde `tf(t, d)` es el número de apariciones de *t* en *d* y `df(t)` el número de documentos que contienen *t*. El primer factor indica que un término repetido es importante para ese documento; el segundo, que un término que aparece en todas partes no discrimina. Después cada vector se normaliza a norma ℓ2 unitaria para que resúmenes largos y cortos sean comparables. Existen variantes: scikit-learn usa por defecto un IDF suavizado y normalización ℓ2.

El resultado es una matriz N × |V| con menos del 1 % de entradas distintas de cero. Mantenedla en formato disperso (`scipy.sparse`).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(min_df=5,           # descarta términos en < 5 documentos
                      max_df=0.5,         # descarta términos en > 50% de documentos
                      ngram_range=(1,1),  # (1,2) añade también bigramas
                      sublinear_tf=True)
X_tr = vec.fit_transform(docs_tr)         # ajustar SOLO con conjunto de entrenamiento
X_te = vec.transform(docs_te)
```

Hiperparámetros más relevantes: `min_df` y `max_df` (controlan |V| y, por tanto, todo el problema), `ngram_range` (los bigramas capturan «machine learning» o «climate change» a costa de multiplicar el vocabulario) y `max_features`. En Gensim, los objetos equivalentes son `Dictionary` (con `filter_extremes`) y `TfidfModel`.

**Nota práctica:** los modelos lineales (regresión logística, SVM lineal) son muy eficaces con TF-IDF disperso y de alta dimensionalidad, mientras que k-NN y los ensembles de árboles tienden a tener dificultades. Si los usáis, reducid antes la dimensionalidad.

## A.3. Word2Vec

Word2Vec representa cada palabra como un vector denso de dimensión *d* (normalmente 100–300), aprendido de un corpus grande de modo que palabras en contextos similares obtengan vectores similares. Hay dos objetivos de entrenamiento: CBOW (predice una palabra a partir de su contexto) y skip-gram (predice el contexto a partir de la palabra), pero no hace falta conocer sus detalles. El contraste con TF-IDF: las dimensiones no son términos ni tienen significado individual, los vectores son densos y palabras semánticamente relacionadas (*photovoltaic*, *solar*) quedan próximas aunque sean términos distintos.

Hay dos opciones, **no equivalentes**:

- **Entrenar los embeddings con este corpus.** El vocabulario coincide exactamente con el vuestro, los términos técnicos están cubiertos y no hay OOV; sin embargo, 35.000 resúmenes son un corpus pequeño para estimar buenos embeddings.
- **Usar vectores preentrenados** (GloVe, fastText, `word2vec-google-news-300`, disponibles vía `gensim.downloader`). Están mucho mejor estimados (miles de millones de palabras), pero faltarán términos específicos del dominio y tendréis que decidir cómo tratarlos.

Se necesita un único vector por documento. El procedimiento estándar es promediar los vectores de sus palabras, opcionalmente ponderando por TF-IDF:

```python
import numpy as np
from gensim.models import Word2Vec

w2v = Word2Vec(docs_tr, vector_size=200, window=5, min_count=5, sg=1, workers=4)

def doc_vector(tokens, kv, dim):
    vs = [kv[t] for t in tokens if t in kv]   # omite palabras OOV
    return np.mean(vs, axis=0) if vs else np.zeros(dim)

Z = np.vstack([doc_vector(d, w2v.wv, 200) for d in docs])
```

Indicad cómo tratáis las palabras OOV y con qué frecuencia un documento termina sin ningún token utilizable. El promedio destruye el orden de las palabras y diluye el contenido de documentos largos (una de las razones de existir de los embeddings contextuales). Con resúmenes de 100–200 palabras funciona razonablemente; con tweets no. La matriz resultante es densa y sus características tienen escalas muy distintas, por lo que debe estandarizarse (`StandardScaler`) antes de usar modelos basados en distancias o regularizados.

## A.4. Advertencia aplicable a las tres representaciones

El vocabulario, los pesos IDF, cualquier embedding entrenado con el corpus y cualquier scaler son **parámetros estimados a partir de los datos**. Ajustadlos **únicamente con la partición de entrenamiento** y aplicadlos después a validación y test. Ejecutar el vectorizador sobre todo el dataset antes de particionar es el error metodológico más frecuente en este proyecto y produce resultados artificialmente inflados. Usar un `sklearn.pipeline.Pipeline` dentro de la validación cruzada hace mucho más difícil cometer este error.
