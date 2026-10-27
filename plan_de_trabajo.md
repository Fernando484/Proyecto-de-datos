# Plan de trabajo (equipo de 3)

Hoy: 27 oct 2026 · Entrega: **11 dic 23:59** (~6,5 semanas) · Presentación: semana del 14 dic.
Recomendación: **congelar el código el 6 dic** y dejar la última semana para README, diapositivas y ensayo.

Miembros: **A**, **B**, **C** (sustituir por nombres).

## 1. Decisiones que hay que tomar la primera semana (todo el equipo)

Todo lo demás depende de esto; si cada uno parte los datos a su manera, las comparaciones no son válidas.

| Decisión | Propuesta |
|---|---|
| Construcción de etiquetas (nivel 2 y nivel 1) | Un único módulo `src/labels.py`, comprobar que se conserva ≈ 90,31 % |
| Partición train/val/test | Estratificada por clase, semilla fija, p. ej. 70/15/15. **Guardar los índices en disco** (`splits.npz`) y que todos los usen |
| Métrica principal | macro-F1 (+ accuracy, F1 por clase). Intervalo de confianza por bootstrap sobre test |
| Selección de hiperparámetros | Solo con validación (o CV sobre train). Test se toca una vez por modelo final |
| Extensión | **Propuesta: «¿Qué aporta el texto?»** (metadatos vs. texto vs. ambos). Es barata, usa lo ya construido y no depende de LLMs ni GPU extra. Si sobra tiempo: comparar SciBERT vs. BERT genérico |
| Entorno | Repo GitHub público, `requirements.txt`, datos fuera del repo (ruta en `config.py` o variable de entorno), `.gitignore` con `projects.xlsx` |

## 2. Reparto de responsabilidades

Cada persona es responsable de **una representación completa + su modelo** y revisa el trabajo de otra.

| | Representación | Modelo | Extra propio |
|---|---|---|---|
| **A** | EDA + TF-IDF (preprocesamiento con SpaCy) | Regresión logística / SVM lineal (sklearn) | Etiquetas y splits, baseline, análisis de ambigüedad (clase predicha ∈ categorías descartadas) |
| **B** | Word2Vec (decidir: entrenado vs. preentrenado, OOV) | Red neuronal en PyTorch (MLP sobre Word2Vec; opcionalmente también sobre TF-IDF y BERT) | Extensión: clasificador solo-metadatos y combinado |
| **C** | Embeddings contextuales (BERT/RoBERTa/SciBERT; pooling; truncado o *chunking*) | Mismo clasificador sklearn sobre los embeddings | **Fine-tuning** con Hugging Face en Colab |

Tareas compartidas:

- **Evaluación comparativa (4.4):** módulo común `src/eval.py` (métricas, bootstrap, matriz de confusión). Escribe A; B y C lo usan.
- **Nivel 1 (6 clases):** cada uno repite su mejor configuración; compara A.
- **Análisis de errores** (hermanas de taxonomía vs. no relacionadas): C, con apoyo de A para la jerarquía.
- **README, diapositivas, declaración de contribuciones:** reparto por secciones (ver §4); revisa todo el equipo.

## 3. Calendario

| Semana | Fechas | A | B | C | Hito |
|---|---|---|---|---|---|
| 1 | 27 oct – 2 nov | Repo, `labels.py`, splits, EDA inicial | Montar entorno, leer Apéndice A, probar Word2Vec | Probar extracción de embeddings en pocas muestras; acceso a Colab GPU | **Splits y etiquetas congelados** |
| 2 | 3 – 9 nov | EDA completo (figuras), baseline, preprocesamiento + TF-IDF | Word2Vec sobre train, doc-vectors, OOV | Embeddings BERT de todo el corpus (guardar en `.npy`) | Las 3 representaciones generadas |
| 3 | 10 – 16 nov | Tuning LogReg/SVM sobre TF-IDF | Tuning MLP PyTorch | Tuning sklearn sobre embeddings; preparar script de fine-tuning | Primera tabla comparativa (val) |
| 4 | 17 – 23 nov | `eval.py` final, bootstrap, nivel 1 | Extensión: metadatos / texto / combinado | Fine-tuning en Colab (nivel 2 y nivel 1) | Todos los modelos entrenados |
| 5 | 24 – 30 nov | Evaluación final en test, tablas, matrices de confusión | Curvas de aprendizaje, resultados de extensión | Análisis de errores y ambigüedad | **Resultados finales (test)** |
| 6 | 1 – 6 dic | Redactar README: problema, datos, EDA | README: modelos, protocolo, extensión | README: resultados, errores, conclusiones | **Congelar código (6 dic)** |
| 7 | 7 – 11 dic | Revisión cruzada, reproducibilidad (ejecutar desde cero), diapositivas | idem | idem | **Entrega 11 dic, objetivo 10 dic** |
| – | 14 – 18 dic | Ensayo de presentación y preguntas | | | Presentación individual |

Margen: el fine-tuning (C) es el mayor riesgo por Colab (límites de GPU, desconexiones). Guardar *checkpoints* y empezar el script en la semana 3.

## 4. Estructura del repositorio

```
├── README.md              # el informe (figuras y tablas incrustadas)
├── requirements.txt
├── config.py              # ruta a los datos, semilla
├── src/
│   ├── labels.py          # construcción de etiquetas nivel 1/2
│   ├── preprocess.py      # pipeline SpaCy
│   ├── eval.py            # métricas, bootstrap, confusión
│   └── ...
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_tfidf_sklearn.ipynb
│   ├── 03_word2vec_pytorch.ipynb
│   ├── 04_bert_embeddings.ipynb
│   ├── 05_finetuning_colab.ipynb
│   └── 06_extension.ipynb
├── figures/               # PNG que usa el README
├── results/               # CSV con métricas por modelo
└── data/                  # vacío; en .gitignore
```

## 5. Checklist de trampas metodológicas (lo que sí puntúa)

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

## 6. Organización del equipo

- Una rama por persona; *pull request* revisado por otro miembro antes de fusionar a `main`.
- Reunión corta (15 min) dos veces por semana; un canal para dudas de bloqueo.
- Cada resultado se guarda como CSV en `results/` (modelo, representación, nivel, macro-F1, accuracy, IC) para generar las tablas del README sin repetir cálculos.
- Dudas sobre la extensión o la corrección de la etiqueta: consultar al profesor **antes de la semana 4**.
