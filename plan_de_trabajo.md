# Plan de trabajo (equipo de 3)

Arranque: **10 nov 2026** · Entrega oficial: **11 dic 23:59** · Presentación individual: semana del 14 dic.
Objetivo del equipo: **terminar todo el 30 nov** (resultados el 25, código congelado el 27, README y diapositivas el 30). Diciembre queda como margen, para revisión y ensayo de la presentación.

Miembros: **A**, **B**, **C** (sustituir por nombres).

---

## 0. Cómo está pensado el reparto

Principios:

1. **Equidad.** Cada persona tiene: una representación vectorial, un requisito de modelado del enunciado (§4.3), una pieza de infraestructura común, una parte de la extensión, una parte del análisis y sus secciones del README. La carga está equilibrada (ver §2.4).
2. **Paralelismo.** Una vez fijado el «contrato» común (§1), cada persona trabaja con sus propios notebooks y ficheros de resultados. Nadie espera a nadie salvo en los puntos de sincronía de §4.
3. **Todos aprenden de todo.** La presentación es **individual** y la metodología se evalúa en las preguntas. Por eso, cada persona revisa el trabajo de otra (§5) y debe saber explicar el pipeline completo, no solo su parte.

Resumen del reparto:

| | Representación | Requisito de modelado (§4.3) | Infraestructura común | Extensión | Análisis |
|---|---|---|---|---|---|
| **A** | TF-IDF (+ preprocesamiento SpaCy) | **Scikit-learn** (LogReg / SVM lineal), también sobre los embeddings de C | `labels.py` + `splits.npz`, EDA (§4.1) | Clasificador **solo metadatos** | Significancia estadística (bootstrap), **análisis de errores y de ambigüedad** |
| **B** | Word2Vec | **Red PyTorch** propia | `config.py`, `eval.py` | Clasificador **combinado** y comparación **texto vs. metadatos vs. ambos** | Curvas de aprendizaje y de validación |
| **C** | Embeddings contextuales (BERT/SciBERT) | **Fine-tuning** Hugging Face (Colab) | Extracción de embeddings compartida (`.npy`) | Comparación **BERT genérico vs. SciBERT** | Nivel 1 de sus modelos |

**Dificultad real, tarea por tarea:** ver §2.4. Resumen honesto: la carga total es parecida (≈ 45–55 h por persona), pero **C tiene la dificultad más alta y el mayor riesgo** (GPU, Colab), A tiene más volumen que dificultad y B está en un punto medio. Por eso C no lleva análisis de errores ni sklearn sobre embeddings.

---

## 1. El contrato común (día 1–2, lo hacen los tres en paralelo)

Estas decisiones se toman **juntos el día 1** (30 min) y las tres piezas se construyen **a la vez**. Si cada uno parte los datos o mide a su manera, las comparaciones no son válidas.

### 1.1 Decisiones fijas (todos las respetan)

| Decisión | Valor |
|---|---|
| Etiquetas (nivel 2 y nivel 1) | Se construyen **solo** con `src/labels.py`. Comprobar que se conserva ≈ 90,31 % |
| Partición | Estratificada por clase, semilla fija, 70/15/15. Índices guardados en `data/splits.npz` (arrays `train_idx`, `val_idx`, `test_idx`) |
| Métrica principal | macro-F1, y además accuracy y F1 por clase. Baseline: clase mayoritaria |
| Hiperparámetros | Solo con validación (o CV sobre train). **Test se toca una vez por modelo final** |
| Ajuste de transformaciones | TF-IDF, vocabulario, Word2Vec y `StandardScaler` se ajustan **solo con train** |
| Formato de predicciones | `results/preds_<representacion>_<modelo>_<nivel>.csv` con columnas `id, y_true, y_pred` (y opcionalmente `proba_<clase>`) y un `split` (`val` o `test`) |
| Formato de métricas | `results/metrics.csv` con columnas `representacion, modelo, nivel, split, macro_f1, accuracy, ic_low, ic_high` |
| Extensión elegida | «¿Qué aporta el texto?» (metadatos vs. texto vs. ambos). Es barata, reutiliza lo ya construido y no necesita LLMs ni GPU extra |

Con ese formato, `eval.py`, el análisis de errores y la extensión funcionan con las predicciones de cualquier modelo sin conocer su código.

### 1.2 Las tres piezas del día 1–2

| Pieza | Responsable | Entregable | Plazo |
|---|---|---|---|
| Configuración y entorno | **B** | `requirements.txt`, `config.py` (ruta a los datos y semilla), comprobar que `.gitignore` incluye `projects.xlsx`, estructura de carpetas (§6) y una rama por persona (`a/…`, `b/…`, `c/…`). El repo de GitHub ya está creado | Día 1 |
| Etiquetas y splits | **A** | `src/labels.py` (niveles 1 y 2) y `data/splits.npz`. Notebook corto que demuestra el ≈ 90,31 % y las clases resultantes | Día 2 |
| `eval.py` mínimo | **B** | `evaluate(y_true, y_pred)` → macro-F1, accuracy, F1 por clase; `save_results(...)` que añade una fila a `results/metrics.csv` y guarda el CSV de predicciones. El bootstrap lo añade después con la misma interfaz | Día 2 |

### 1.3 Qué hacen C y los demás mientras tanto (sin esperar)

- **C** extrae los embeddings contextuales de **todo** el corpus y los guarda en `.npy` (ordenados por `id`). No depende de los splits ni produce fuga, porque el encoder está preentrenado y no ve las etiquetas. Es la tarea más pesada de arranque, así que empieza el día 1.
- **B**, en cuanto termina `config.py` y `eval.py`, prototipa Word2Vec en una muestra pequeña (el entrenamiento definitivo, solo con train, espera a `splits.npz`).
- **A**, una vez entregados etiquetas y splits, empieza el EDA y el preprocesamiento.

**Punto de sincronía 0 (fin del día 2):** los tres confirmáis que `labels.py`, `splits.npz`, `eval.py` y el formato de CSV están cerrados. Si cambian después, hay que rehacer todo.

---

## 2. Qué tiene que hacer cada persona

### 2.1 Persona A: EDA, TF-IDF y modelo de Scikit-learn

**Responsabilidad principal:** el análisis exploratorio, la representación TF-IDF y el algoritmo de Scikit-learn.

**Entregables:**

1. **Etiquetas y splits** (día 1–2): `labels.py`, `splits.npz`, notebook de verificación (ver §1.2).
2. **EDA** (`01_eda.ipynb`, enunciado §4.1):
   - Número de instancias, variables, tipos y valores ausentes.
   - Distribución de la variable objetivo; número de clases y desbalance; **baseline de clase mayoritaria**.
   - Longitud de los resúmenes, proyectos por año y por país.
   - Términos más frecuentes, nube de palabras y ejemplos por clase.
   - Figuras guardadas en `figures/` para el README.
3. **Preprocesamiento + TF-IDF** (`src/preprocess.py`, `02_tfidf_sklearn.ipynb`):
   - Pipeline SpaCy: minúsculas, stopwords, lematización, frecuencias mínima y máxima de documento. **Documentar y justificar cada decisión.**
   - TF-IDF ajustado **solo con train**.
4. **Modelo Scikit-learn**: regresión logística y/o SVM lineal. Búsqueda de hiperparámetros con validación (o CV sobre train). Guardar predicciones y métricas con `eval.save_results`.
5. **Scikit-learn sobre los embeddings de C:** mismo clasificador y mismo protocolo que con TF-IDF, usando `data/emb_*.npy` (C los genera). Así A y C comparan representaciones con el mismo algoritmo.
6. **Nivel 1** (6 clases): repetir la mejor configuración de TF-IDF y de embeddings y registrar los resultados.
7. **Análisis de significancia:** definir el criterio de «diferencia significativa» (qué magnitud debe tener una diferencia de macro-F1 dado el IC del test) y aplicarlo a todas las comparaciones. Añadir el **bootstrap** a `eval.py` (acordado con B, que es su dueño).
8. **Análisis de errores y de ambigüedad** (con las predicciones de todos, bloque 3): matriz de confusión de la mejor configuración; ¿confunde categorías **hermanas** de la taxonomía o **no relacionadas**? (A ya conoce la jerarquía por `labels.py`); fracción de errores donde la clase predicha ∈ categorías descartadas; ejemplos concretos.
9. **Extensión, parte A:** clasificador **solo con metadatos** (coste, contribución de la UE, duración, país coordinador, nº de países). Procesar los metadatos (nulos, categóricas, escalado solo con train) en `src/metadata.py`, y entregar la matriz de metadatos lista para que B la combine con el texto.

**Depende de:** nadie, salvo B para `eval.py` (día 2); C para los embeddings (bloque 1) y todos para las predicciones del análisis de errores.
**Lo necesitan de A:** B y C (`labels.py` y `splits.npz`, día 2); B (matriz de metadatos, bloque 2).

### 2.2 Persona B: Word2Vec, red PyTorch e infraestructura

**Responsabilidad principal:** la representación Word2Vec y la red neuronal de PyTorch (requisito explícito del enunciado), más la configuración común y el módulo de evaluación.

**Entregables:**

1. **Configuración y entorno** (día 1: `config.py`, `requirements.txt`, ramas) y **`eval.py`** (día 2), ver §1.2. Después, mantener `eval.py`: añadir el bootstrap (con A) y la matriz de confusión.
2. **Word2Vec** (`03_word2vec_pytorch.ipynb`):
   - Decidir y justificar: **entrenar con este corpus o usar vectores preentrenados**. Si se entrena, solo con train.
   - Vector de documento = media de los vectores de sus palabras.
   - **Tratamiento de OOV** documentado; contar cuántos documentos quedan **sin ningún token válido**.
3. **Red neuronal en PyTorch** (diseñada e implementada por B, no un modelo copiado):
   - MLP sobre Word2Vec; opcionalmente, el mismo MLP sobre TF-IDF y sobre los embeddings de BERT (usa las salidas de A y C).
   - Búsqueda de hiperparámetros (capas, dropout, lr, weight decay) con validación; *early stopping* con validación.
   - Manejo del desbalance (pesos de clase o muestreo), justificado.
4. **Curvas de aprendizaje y de validación** (pérdida y macro-F1 por época) para la mejor configuración, en `figures/`.
5. **Nivel 1:** repetir la mejor configuración y registrar resultados.
6. **Extensión, parte B:** clasificador **combinado** (metadatos + texto). Une los metadatos de A con la mejor representación de texto (decidir si por concatenación de vectores o por *late fusion*). Después responde a la pregunta de la extensión: **¿cuánto de lo que consigue el texto se obtendría sin los resúmenes?** Compara solo-metadatos (de A), solo-texto y combinado, también por clase.

**Depende de:** `splits.npz` (A, día 2) para el entrenamiento definitivo; la matriz de metadatos de A para la extensión; los embeddings de C si hace el MLP sobre BERT.
**Lo necesitan de B:** A y C (`config.py` y `eval.py`, día 2).

### 2.3 Persona C: Embeddings contextuales, fine-tuning y análisis de errores

**Responsabilidad principal:** la representación con Transformers, el fine-tuning con Hugging Face y el análisis de dónde fallan los modelos.

**Entregables:**

1. **Embeddings contextuales** (`04_bert_embeddings.ipynb`, desde el día 1):
   - Modelo encoder-only (BERT, RoBERTa o SciBERT), sobre el **texto sin preprocesar**.
   - Documentar **cómo se pasa de vectores por token a uno por documento** (`[CLS]`, media de tokens, etc.) y **qué se hace con los resúmenes que superan la longitud máxima** (truncado o *chunking* con promedio).
   - Guardar embeddings de todo el corpus en `data/emb_<modelo>.npy` alineados por `id`.
2. **Fine-tuning con Hugging Face** (`05_finetuning_colab.ipynb`): niveles 2 y 1, en Google Colab con GPU.
   - **Empezar el script en cuanto haya embeddings** y probarlo con una submuestra pequeña. Es el mayor riesgo: límites de GPU y desconexiones.
   - Guardar *checkpoints* en Drive y registrar semilla y tiempos.
   - Entrenar con train, elegir época con validación, evaluar en test una sola vez.
3. **Comparación de embeddings (extensión, parte C):** BERT genérico vs. SciBERT (o similar) como extractores, con el mismo clasificador de A, y si hay tiempo también como base del fine-tuning. Explicar el efecto sobre el rendimiento.
4. **Nivel 1** de sus modelos (fine-tuning y embeddings): registrar resultados.

**Qué NO lleva C (decisión de equilibrio):** el clasificador sklearn sobre sus embeddings lo ejecuta A, y el análisis de errores también. C ya tiene la tarea más difícil y arriesgada.

**Depende de:** nada hasta el bloque 3 (los embeddings no necesitan splits); `splits.npz` para el fine-tuning.
**Lo necesitan de C:** A (embeddings, bloque 1), B (si usa BERT en el MLP) y el análisis final.

### 2.4 Dificultad real y equilibrio de la carga

Las horas son **estimaciones aproximadas** (no medidas) para alguien con base de Python y sklearn. Dificultad: ★ fácil (rutina) · ★★ media (hay que pensar y depurar) · ★★★ alta (conceptos nuevos, errores difíciles de depurar o riesgo de infraestructura).

| Persona | Tarea | Dificultad | Horas aprox. |
|---|---|---|---|
| **A** | `labels.py` + splits | ★★ (hay que entender la taxonomía y reproducir el 90,31 %) | 4 |
| | EDA (§4.1) | ★ | 8 |
| | Preprocesamiento SpaCy | ★★ (lento sobre decenas de miles de resúmenes) | 5 |
| | TF-IDF + sklearn con tuning | ★ | 8 |
| | sklearn sobre embeddings de C | ★ | 3 |
| | Nivel 1 | ★ | 3 |
| | Bootstrap + criterio de significancia | ★★ (concepto estadístico) | 6 |
| | Análisis de errores y ambigüedad | ★★ | 10 |
| | Metadatos solo | ★★ | 5 |
| | **Total A** | **mayormente ★–★★; más volumen que dificultad** | **≈ 52** |
| **B** | `config.py`, `requirements.txt`, ramas | ★ | 1 |
| | `eval.py` (métricas, guardado, confusión) | ★★ (tiene que ser robusto: todos lo usan) | 5 |
| | Word2Vec (entrenar/preentrenado, OOV) | ★★ | 8 |
| | Red PyTorch + tuning + early stopping | ★★★ (diseño propio, desbalance, depuración) | 16 |
| | Curvas de aprendizaje | ★ | 3 |
| | Nivel 1 | ★ | 3 |
| | Modelo combinado + comparación texto/metadatos/ambos | ★★ | 10 |
| | **Total B** | **mezcla ★★ con una parte ★★★** | **≈ 46** |
| **C** | Extracción de embeddings (token → documento, chunking) | ★★ (decisiones y tiempos de GPU) | 8 |
| | Fine-tuning HF en Colab, niveles 2 y 1 | ★★★ (límites de GPU, desconexiones, checkpoints, entrenar dos veces) | 22 |
| | BERT genérico vs. SciBERT | ★★ | 8 |
| | Nivel 1 | ★★ (reentrenar) | 4 |
| | **Total C** | **la más difícil y la más arriesgada** | **≈ 42** |

**Lectura honesta:**
- **A** tiene más horas pero la tarea más rutinaria; cuando A acaba, es quien puede apoyar a los demás.
- **B** tiene un bloque difícil (la red PyTorch), pero depurable en local y sin dependencias externas.
- **C** tiene menos horas, pero **el riesgo de bloqueo más alto**: Colab es impredecible y el fine-tuning no se puede acelerar. Por eso se le han quitado tareas para dejarle holgura de tiempo.
- Si A termina antes, **A ayuda a C** con la preparación del script de fine-tuning y el nivel 1.
- Si queréis cambiar el reparto de dificultad, la rotación natural es **intercambiar B y C** en la parte difícil; cualquier cambio debe mantener que cada persona hace un requisito de §4.3.

Reevaluad estas estimaciones el día 5. Si alguien va un 30 % por encima o por debajo, reasignad figuras, README o pruebas en la reunión.

---

## 3. Calendario por bloques (todos en paralelo)

| Bloque | Fechas | A | B | C | Hito |
|---|---|---|---|---|---|
| **0** | 10–11 nov | `labels.py` y `splits.npz` | `config.py`, `eval.py` mínimo | Empieza a extraer embeddings de todo el corpus | **Contrato cerrado (fin del día 2)** |
| **1** | 12–16 nov | EDA, preprocesamiento, TF-IDF | Word2Vec con train, doc-vectors, OOV | Embeddings terminados; script de fine-tuning probado con una submuestra | **Las 3 representaciones generadas** |
| **2** | 17–21 nov | Tuning sklearn (TF-IDF y embeddings), metadatos solo | Tuning MLP PyTorch, curvas | **Fine-tuning en Colab** (nivel 2) | **Primera tabla comparativa (val)** |
| **3** | 22–25 nov | Nivel 1, bootstrap, significancia, análisis de errores | Nivel 1, extensión combinada | Fine-tuning nivel 1, BERT vs. SciBERT | **Resultados finales en test (25 nov)** |
| **4** | 26–30 nov | README (sus secciones), figuras, diapositivas | README (sus secciones), diapositivas | README (sus secciones), diapositivas | **Código congelado el 27 nov**; **README y diapositivas listos el 30 nov** |
| – | 1–10 dic | Revisión cruzada, reproducibilidad desde cero, subir la entrega | idem | idem | **Subir el 4–5 dic** (margen de una semana antes del 11) |
| – | 14–18 dic | Ensayo y preguntas | | | Presentación individual |

**Ritmo:** el calendario es apretado: el bloque 2 solo tiene 5 días y concentra el fine-tuning. Si el 21 nov alguien va retrasado, el primer recorte es el modelo combinado de la extensión y las tareas opcionales (MLP sobre BERT o TF-IDF, SciBERT en fine-tuning), nunca el protocolo ni el análisis.

**Margen:** el fine-tuning de C es el mayor riesgo. Si en el bloque 2 hay problemas con Colab, A (que termina antes) ayuda con las pruebas de que el pipeline corre antes de lanzar el entrenamiento largo.

---

## 4. Puntos de sincronía (cuándo necesitáis algo de otro)

| Cuándo | Quién entrega | Qué | Quién lo necesita |
|---|---|---|---|
| Fin del día 1 | B | `config.py` y `requirements.txt` | A y C |
| Fin del día 2 | A | `labels.py`, `splits.npz` | B y C |
| Fin del día 2 | B | `eval.py` mínimo | A y C |
| Bloque 1 | C | `data/emb_*.npy` | A (sklearn sobre embeddings), B (MLP sobre BERT, opcional) |
| Bloque 2 | A | Matriz de metadatos (`src/metadata.py`) | B (extensión combinada) |
| Fin del bloque 2 | A, B, C | Predicciones y métricas de val de cada modelo | Todos (tabla comparativa) |
| Bloque 3 | A, B, C | Predicciones de test de cada modelo | A (errores y tablas finales) |

Fuera de estos puntos, no hace falta esperar a nadie.

---

## 5. Reglas del equipo

- **Una rama por persona**, `pull request` revisado por otro miembro antes de fusionar a `main`. Revisión cruzada: **A revisa a B, B revisa a C, C revisa a A.** El revisor ejecuta el notebook y comprueba que no se usa test para decidir.
- **Cada persona edita solo lo suyo:** sus notebooks y sus ficheros en `results/`. Los módulos compartidos (`labels.py`, `eval.py`, `config.py`) los toca su dueño; los demás piden cambios por PR o mensaje.
- **Reunión de 15 minutos**, dos por semana, solo para desbloquear; un canal para dudas.
- Cada resultado se guarda como CSV en `results/` (§1.1) para generar las tablas del README sin repetir cálculos.
- **README y diapositivas:** cada uno escribe la sección de lo que ha hecho **según avanza**, no al final:

| Sección del README | Autor |
|---|---|
| Problema, datos, EDA, baseline | A |
| Preprocesamiento, TF-IDF, sklearn | A |
| Protocolo (splits, métricas, bootstrap, criterio de significancia) | A, B (revisa) |
| Word2Vec, red PyTorch, curvas | B |
| Extensión (metadatos, combinado, texto vs. metadatos) | A (metadatos), B (combinado y comparación) |
| Embeddings contextuales, fine-tuning, BERT vs. SciBERT | C |
| Comparación entre representaciones, nivel 1 | A (tablas), todos (discusión) |
| Análisis de errores y de ambigüedad | A |
| Conclusiones y limitaciones | Redacta C, revisan A y B |

- La **declaración de contribuciones** la redacta quien suba la entrega; debe reflejar este reparto. Las **diapositivas** las prepara cada uno para sus secciones y las revisa todo el equipo.
- **Dudas** sobre la extensión o la corrección de la etiqueta: consultar al profesor **antes del bloque 2 (17 nov)**.

---

## 6. Estructura del repositorio

```
├── README.md              # el informe (figuras y tablas incrustadas)
├── requirements.txt
├── config.py              # ruta a los datos, semilla
├── src/
│   ├── labels.py          # A: construcción de etiquetas nivel 1/2
│   ├── preprocess.py      # A: pipeline SpaCy
│   ├── metadata.py        # A: procesado de metadatos
│   ├── eval.py            # B: métricas, bootstrap, confusión, guardado de resultados
│   └── ...
├── notebooks/
│   ├── 01_eda.ipynb                # A
│   ├── 02_tfidf_sklearn.ipynb      # A
│   ├── 03_word2vec_pytorch.ipynb   # B
│   ├── 04_bert_embeddings.ipynb    # C
│   ├── 05_finetuning_colab.ipynb   # C
│   └── 06_extension.ipynb          # A, B, C (una sección cada uno)
├── figures/               # PNG que usa el README
├── results/               # metrics.csv y preds_*.csv
└── data/                  # splits.npz SÍ se sube (es pequeño y todos lo necesitan igual);
                           # NO se suben (.gitignore): projects.xlsx y emb_*.npy (pesados)
```

---

## 7. Checklist de trampas metodológicas (lo que sí puntúa)

- [ ] TF-IDF, vocabulario, Word2Vec entrenado y `StandardScaler` ajustados **solo con train**.
- [ ] Hiperparámetros elegidos **sin tocar test**; test usado una vez por modelo final.
- [ ] Todos los modelos evaluados sobre **el mismo test** (mismos índices).
- [ ] Tamaños de train/val/test indicados en el README.
- [ ] Baseline de clase mayoritaria reportado.
- [ ] macro-F1 y F1 por clase, no solo accuracy.
- [ ] Incertidumbre del test (bootstrap/IC) y criterio explícito de «diferencia significativa».
- [ ] Embeddings contextuales sobre texto **sin preprocesar**; explicado el manejo de resúmenes largos.
- [ ] Documento Word2Vec sin tokens válidos: contar cuántos.
- [ ] Repetido en nivel 1; discutido si cambia el ranking.
- [ ] Confusiones: ¿hermanas o no relacionadas? Fracción de errores donde la predicción es una categoría descartada.
- [ ] Si una puntuación es muy alta: discutir que las etiquetas se generaron a partir del mismo texto.
- [ ] Figuras y tablas **en el README**, no solo en notebooks.
- [ ] Código externo citado; `projects.xlsx` no subido; repo público.
- [ ] Declaración de contribuciones y diapositivas subidas a Aula Global.
- [ ] Cada miembro sabe explicar el pipeline completo (la presentación es individual).
