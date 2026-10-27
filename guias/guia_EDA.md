# Guía para entender el Análisis Exploratorio de Datos (EDA)

Guía pensada para quien hace por primera vez un proyecto de Machine Learning. Está escrita para este proyecto (clasificación de resúmenes de proyectos H2020), pero las ideas valen para cualquier otro.

**Índice**

1. [Qué es el EDA, con una analogía](#1-qué-es-el-eda)
2. [Por qué se hace antes de modelar](#2-por-qué-se-hace-antes-de-modelar)
3. [Vocabulario básico](#3-vocabulario-básico)
4. [Bloque 1: descripción general](#4-bloque-1-descripción-general)
5. [Bloque 2: la variable objetivo y el baseline](#5-bloque-2-la-variable-objetivo-y-el-baseline)
6. [Bloque 3: estadísticas y gráficas sencillas](#6-bloque-3-estadísticas-y-gráficas-sencillas)
7. [Bloque 4: análisis preliminar del texto](#7-bloque-4-análisis-preliminar-del-texto)
8. [Un detalle importante: EDA y fuga de información](#8-un-detalle-importante-eda-y-fuga-de-información)
9. [Cómo escribirlo en el README](#9-cómo-escribirlo-en-el-readme)
10. [Errores típicos](#10-errores-típicos)
11. [Checklist final](#11-checklist-final)

---

## 1. Qué es el EDA

**EDA** significa *Exploratory Data Analysis*, en español **análisis exploratorio de datos**.

Es la fase en la que **miras los datos con calma antes de construir ningún modelo**, para entender qué contienen, cómo son y qué problemas tienen.

### Analogía: el cocinero y la despensa

Imagina que te piden preparar una cena y te dan una despensa que no conoces. Antes de cocinar harías algo así:

- Abrir los armarios y ver **qué hay** y **cuánto** de cada cosa.
- Comprobar **qué falta** o está caducado.
- Fijarte en que tienes 20 kg de patatas y solo 2 huevos: eso condiciona el menú.
- Probar un poco de cada cosa.

Hasta que no sabes eso no puedes decidir bien la receta. En un proyecto de datos:

| Cocina | Proyecto |
|---|---|
| La despensa | El fichero `projects.xlsx` |
| Ver qué hay y cuánto | Número de filas, columnas, tipos |
| Lo que falta o caduca | Valores ausentes, datos raros |
| 20 kg de patatas y 2 huevos | Clases muy desbalanceadas |
| Probar un poco | Leer resúmenes de ejemplo |
| La receta | Los modelos (TF-IDF, Word2Vec, BERT...) |

**El EDA no consiste en hacer gráficas bonitas.** Consiste en **hacerte preguntas sobre los datos y responderlas con números y figuras**, y que esas respuestas te ayuden a tomar decisiones.

## 2. Por qué se hace antes de modelar

Tres razones.

### 2.1 Para no equivocarte sin darte cuenta
Si no miras los datos, puedes entrenar un modelo sobre datos rotos y no enterarte. Ejemplo típico: la columna con las categorías viene como texto (`"[155, 56306972]"`) y no como lista. Si no lo ves, el programa puede dar un resultado absurdo sin avisar.

### 2.2 Para justificar tus decisiones
El enunciado insiste en que lo que se evalúa es la **justificación metodológica**. El EDA es donde nacen las justificaciones:

- «Usamos macro-F1 **porque** el EDA muestra que las clases están muy desbalanceadas.»
- «Truncamos los textos a X palabras **porque** el EDA muestra que la mayoría mide menos de X.»

Sin EDA, esas frases no tienen apoyo.

### 2.3 Para tener un punto de referencia (el baseline)
El EDA te permite calcular el resultado de un «modelo tonto» que sirve de listón. Si tu modelo sofisticado no lo supera claramente, algo va mal. Se explica en el bloque 2.

## 3. Vocabulario básico

| Término | Significado sencillo | En este proyecto |
|---|---|---|
| **Instancia / muestra / fila** | Un ejemplo individual | Un proyecto H2020 |
| **Variable / característica / columna** | Una propiedad de cada instancia | `title`, `summary`, `totalCost`... |
| **Variable objetivo (target)** | Lo que queremos predecir | La categoría de nivel 2 |
| **Clase** | Cada valor posible del objetivo | «medical biotechnology», «chemical sciences»... |
| **Clasificación multiclase** | Predecir una entre varias clases | Elegir 1 entre ~40 áreas |
| **Valor ausente (NaN)** | Dato que falta | Un proyecto sin `patentID` |
| **Desbalance de clases** | Unas clases tienen muchos ejemplos y otras pocos | Hay áreas con miles de proyectos y otras con decenas |
| **Baseline** | Resultado de un modelo trivial, usado como listón | Predecir siempre la clase más frecuente |
| **Corpus** | Colección de textos | Los 35.378 resúmenes |
| **Token** | Unidad de texto, normalmente una palabra | «project», «energy» |
| **Stopword** | Palabra muy común que aporta poco («the», «of») | Se suele eliminar |

## 4. Bloque 1: descripción general

El enunciado pide: *«número de instancias, variables, tipos de datos y valores ausentes»*.

### Qué preguntas respondes

1. ¿Cuántas filas y cuántas columnas hay?
2. ¿De qué tipo es cada columna (número, texto, fecha)?
3. ¿Qué columnas tienen valores que faltan y cuántos?
4. ¿Hay filas duplicadas?

### Lo que ya sabemos de vuestros datos

Se comprobó leyendo el fichero:

- **35.378 filas** y **19 columnas**.
- Las columnas son las 13 del enunciado más `masterCall`, `subCall`, `fundingScheme`, `rcn`, `topic` y `topic_title`.
- Valores ausentes:

| Columna | Ausentes | Cómo interpretarlo |
|---|---|---|
| `startDate`, `endDate` | 11 | Muy pocos. Se pueden ignorar en las gráficas por año |
| `masterCall` | 20.358 | No nos importa; no la usaremos |
| `euroSciVocCode` | 3.326 | **Importante:** son proyectos **sin etiqueta**. No se pueden usar para entrenar |
| `publicationID` | 18.166 | Probablemente significa «sin publicaciones», no «dato perdido» |
| `patentID` | 34.499 | Casi todos los proyectos no tienen patentes |
| `summary`, `title` | 0 | Perfecto: el texto está completo |

Dos ideas que se aprenden aquí:

- **Un NaN no siempre significa lo mismo.** En `euroSciVocCode` es información que falta. En `patentID` casi seguro es «cero patentes». Hay que pensar qué significa cada uno.
- **Los tipos importan.** `euroSciVocCode` es texto con forma de lista. Hay que convertirlo antes de usarlo.

### Cómo se hace (código de ejemplo)

```python
import pandas as pd

df = pd.read_excel("projects.xlsx")

df.shape                 # (filas, columnas)
df.dtypes                # tipo de cada columna
df.isna().sum()          # ausentes por columna
(df.isna().mean() * 100) # lo mismo, en porcentaje
df.duplicated("projectID").sum()   # duplicados por identificador
df.head()                # primeras filas
```

### Qué escribir en el informe
Una **tabla** con una fila por variable: nombre, tipo, número de ausentes, y una frase de qué significa. Y una frase de conclusión como: «El 9,4 % de los proyectos no tiene código EuroSciVoc y no podrá usarse».

## 5. Bloque 2: la variable objetivo y el baseline

Este es el bloque **más importante** del EDA para este proyecto.

### Primero hay que construir la etiqueta

Antes de estudiar la distribución del objetivo, hay que **crearlo**, como explica la Sección 3 del enunciado. Resumen:

1. Cada proyecto tiene una lista de códigos (`[155, 56306972, 325]`).
2. Cada código corresponde a un concepto dentro de una jerarquía (como carpetas dentro de carpetas).
3. Para cada código se sube hasta el **nivel 2** de la jerarquía.
4. Se usa **el primer código** que llegue a nivel 2 como etiqueta.
5. Los proyectos sin etiqueta válida se descartan (debería quedar ~90,31 %).

Por eso la construcción de etiquetas (`labels.py`) viene **antes** de esta parte del EDA.

### Qué preguntas respondes

**a) ¿Cuántas clases hay?**
En la taxonomía hay 41 categorías de nivel 2 en total. En vuestros datos pueden ser algunas menos. Se cuenta con `df["y2"].nunique()`.

**b) ¿Cuántos ejemplos tiene cada clase?**
Se hace un gráfico de barras ordenado de mayor a menor:

```python
conteo = df["y2"].value_counts()
conteo.plot.barh()
```

**c) ¿Qué tan desbalanceadas están?**
Una cifra sencilla es la **razón de desbalance**:

```
razón = nº de ejemplos de la clase más grande / nº de la más pequeña
```

Si sale 500, significa que una clase tiene 500 veces más ejemplos que otra. Eso es un problema serio para el modelo, porque tenderá a ignorar las clases pequeñas.

**d) ¿Qué baseline sale?**

### Qué es un baseline y por qué importa

Un **baseline** es el resultado de un modelo ridículamente simple que sirve de **suelo**. El más habitual: *«predecir siempre la clase mayoritaria»*.

Ejemplo ilustrativo (**números inventados para explicar**): si la clase más frecuente representa el 18 % de los proyectos, un modelo que siempre responda esa clase acertará el 18 % de las veces. Ese 18 % es el baseline de accuracy.

> **Por qué es útil:** si un día tu modelo saca 70 % de accuracy, parece mucho. Pero la pregunta correcta es *«¿cuánto mejor que 18 %?»*. Y si sacara 20 %, sabrías que no ha aprendido casi nada.

### Dos métricas, dos baselines

- **Accuracy** = proporción de aciertos. El baseline es la proporción de la clase mayoritaria.
- **Macro-F1** = media de la puntuación F1 de **cada clase por igual**, sea grande o pequeña.

Importante: el baseline en macro-F1 sale **casi cero**. Un modelo que siempre dice «clase A» acierta bastante en A, pero en las otras 40 clases tiene F1 = 0, y la media las cuenta todas por igual. Fórmula: si la clase mayoritaria tiene proporción *p* entre *K* clases, el macro-F1 del baseline es **(2p/(1+p)) / K**.

Con los números inventados de antes (*p* = 0,18, *K* = 40) sale ≈ 0,0076, menos de un 1 %.

**Qué enseña esto:** una misma «respuesta tonta» puede parecer decente en accuracy (18 %) y desastrosa en macro-F1 (<1 %). Por eso el enunciado exige macro-F1 y no solo accuracy: **macro-F1 no se deja engañar por las clases grandes**.

### Qué escribir en el informe
- Una figura de barras con todas las clases.
- Una tabla con: nº de clases, clase mayoritaria y su %, clase minoritaria y su n.º, razón de desbalance, baseline de accuracy y baseline de macro-F1.
- Una conclusión: «Las clases están desbalanceadas (razón X), por eso usaremos macro-F1 y el desglose por clase».

### Ambigüedad de las etiquetas (extra muy valorable)
El enunciado avisa de que solo ~32 % de los proyectos tienen todos sus códigos en la misma categoría de nivel 2. Es decir, **muchos proyectos pertenecen realmente a varias áreas** y vosotros solo usáis la primera. Calcular cuántos proyectos tienen 1, 2, 3... áreas distintas y mostrarlo en una gráfica demuestra que entendéis el problema, y apoya la idea del «techo» de rendimiento.

## 6. Bloque 3: estadísticas y gráficas sencillas

El enunciado pide tres: **longitud de resúmenes**, **proyectos por año**, **proyectos por país**.

### 6.1 Longitud de los resúmenes

- **Qué es:** el número de palabras de cada `summary`.
- **Cómo:** `df["summary"].str.split().str.len()`
- **Gráfica:** histograma (barras que muestran cuántos resúmenes tienen cada longitud).
- **Estadísticas:** media, mediana, mínimo, máximo, percentiles 5 y 95.
- **Por qué importa:** BERT y similares aceptan un máximo de unos 512 tokens (≈ 350-400 palabras). Si muchos resúmenes lo superan, tendréis que decidir qué hacer (el enunciado lo pregunta explícitamente). Y si hay resúmenes cortísimos o vacíos, son un problema para cualquier modelo.
- **Conclusión tipo:** «El X % de los resúmenes supera las N palabras, así que...».

### 6.2 Proyectos por año

- **Qué es:** cuántos proyectos empiezan cada año.
- **Cómo:** `df["startDate"].dt.year.value_counts().sort_index()`
- **Gráfica:** barras o línea.
- **Por qué importa:** H2020 va de 2014 a 2020. Si hay años con muy pocos proyectos, hay que saber por qué. También avisa de si los proyectos recientes tienen menos publicaciones (aún no han terminado), algo clave si alguien hace la extensión de producción científica.

### 6.3 Proyectos por país

- **Qué es:** cuántos proyectos coordina cada país (`coordinatorCountry`).
- **Gráfica:** barras horizontales con los 15-20 países principales.
- **Por qué importa:** es contexto. Ayuda a ver si hay sesgo (por ejemplo, mucho peso de unos pocos países) y es la variable que usaría la extensión de metadatos.

### Opcionales, baratas y útiles
- Coste total y contribución UE (usar escala logarítmica porque hay valores enormes).
- Duración del proyecto en meses (`endDate - startDate`).
- Número de países participantes (se obtiene de `countryContr`).

## 7. Bloque 4: análisis preliminar del texto

Aquí se mira el texto **antes de aplicar ninguna técnica de NLP sofisticada**. El objetivo es entender qué vocabulario tienen los resúmenes y cómo son.

### 7.1 Términos más frecuentes

- **Idea:** contar qué palabras aparecen más.
- **Cómo:** separar por palabras, pasar a minúsculas, quitar stopwords y contar.
- **Qué vas a ver:** palabras como *project*, *research*, *new*, *development*, *innovation*... Son muy frecuentes en **todas** las clases, así que no ayudan a distinguirlas. El Apéndice A del enunciado las llama «stopwords específicas del dominio».
- **Qué se aprende:** que hay que podar el vocabulario (con `max_df`) porque lo más frecuente suele ser lo menos útil.

### 7.2 Nube de palabras

- Es una imagen con las palabras más frecuentes en tamaño grande.
- **Honestidad:** aporta poca información precisa. Es visual y se espera en el enunciado.
- **Más útil:** hacer una nube **por clase** para 3 o 4 clases muy distintas (por ejemplo, una médica y una de ingeniería). Así ves que el vocabulario sí difiere entre clases, lo cual **justifica que la clasificación sea posible**.

### 7.3 Ejemplos por clase

- **Qué es:** leer 2 o 3 resúmenes reales de varias clases.
- **Por qué es lo más valioso:** te da intuición. Verás si la etiqueta parece razonable, qué estilo tienen los textos, si hay textos raros (idiomas distintos, resúmenes casi vacíos, texto repetido).
- **Qué poner en el informe:** una pequeña tabla con clase y un fragmento corto de un resumen.

## 8. Un detalle importante: EDA y fuga de información

La **fuga de información** (*data leakage*) ocurre cuando información del conjunto de **test** influye de alguna manera en tus decisiones de modelado. Inflaría los resultados y es lo que más penaliza el enunciado.

Regla práctica para el EDA:

- **Describir todo el dataset está bien.** Contar clases, ver longitudes, hacer gráficas generales es solo *describir*.
- **Pero no uses el EDA para ajustar cosas del modelo mirando el test.** Por ejemplo, no elijas el vocabulario o los umbrales de `min_df`/`max_df` basándote en estadísticas que incluyan el test.
- Más adelante, todo lo que se *aprende de los datos* (vocabulario, pesos IDF, escalado) se calcula **solo con train**. Ver Apéndice A.4 del enunciado.

Resumen: el EDA sirve para **entender**; las decisiones de modelado se toman con **train y validación**.

## 9. Cómo escribirlo en el README

Una estructura posible de la sección:

```
## 1. Datos y análisis exploratorio
### 1.1 Descripción general
   Tabla de variables, tipos y ausentes + 2-3 frases.
### 1.2 Construcción de la variable objetivo
   Cómo se obtiene la etiqueta, % de proyectos conservados.
### 1.3 Distribución de las clases y baseline
   Figura de barras + tabla (nº clases, desbalance, baselines).
### 1.4 Estadísticas descriptivas
   Longitud de resúmenes, proyectos por año, por país.
### 1.5 Análisis preliminar del texto
   Términos frecuentes, nube(s), tabla de ejemplos.
### 1.6 Conclusiones del EDA
   3-4 frases: qué implica esto para el modelado.
```

**Regla de oro del enunciado:** *toda afirmación debe estar respaldada por una figura o tabla incluida en el README*. Guardad las figuras en `figures/` y enlazadlas con `![descripción](figures/nombre.png)`.

## 10. Errores típicos

| Error | Por qué es un problema | Qué hacer |
|---|---|---|
| Gráficas sin conclusión | Parecen relleno | Cada figura termina con una frase de «por tanto...» |
| Olvidar el baseline | No se puede interpretar ningún resultado | Calcularlo en el EDA, de accuracy y de macro-F1 |
| Mirar solo accuracy | Engaña con clases desbalanceadas | Reportar siempre macro-F1 y desglose por clase |
| Ignorar los ausentes | Pueden romper el entrenamiento o sesgarlo | Documentar cuántos hay y qué se hace con ellos |
| No leer ningún texto | Se pierde intuición | Leer ejemplos de varias clases |
| Figuras solo en el notebook | El enunciado exige que estén en el README | Guardar como PNG y enlazar |
| Usar el test para decidir | Fuga de información | EDA para entender, modelado con train/val |

## 11. Checklist final

- [ ] Tabla con filas, columnas, tipos y ausentes por variable.
- [ ] Explicado qué se hace con los proyectos sin etiqueta.
- [ ] Etiqueta construida y comprobado que se conserva ≈ 90,31 %.
- [ ] Gráfica de distribución de clases (ordenada).
- [ ] Número de clases y razón de desbalance.
- [ ] Baseline de accuracy **y** de macro-F1.
- [ ] Histograma de longitud de resúmenes (con % que supera 512 tokens aprox.).
- [ ] Proyectos por año y por país.
- [ ] Palabras más frecuentes y al menos una nube de palabras.
- [ ] Ejemplos de resúmenes de varias clases.
- [ ] (Extra) Gráfica de cuántas áreas de nivel 2 tiene cada proyecto.
- [ ] Conclusión de 3-4 frases sobre qué implica el EDA para el modelado.
- [ ] Todas las figuras guardadas en `figures/` y enlazadas en el README.
