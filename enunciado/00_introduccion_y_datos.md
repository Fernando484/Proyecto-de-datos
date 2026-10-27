# Proyecto final: Análisis temático de proyectos de investigación

**Procesamiento de Datos** · Máster en Ingeniería de Telecomunicación · 21 de octubre de 2026

## Índice del enunciado

| Fichero | Contenido |
|---|---|
| `00_introduccion_y_datos.md` | Introducción y conjunto de datos (este fichero) |
| `01_problema_clasificacion.md` | Definición del problema multiclase, advertencias y techo |
| `02_proyecto_basico.md` | Proyecto básico: EDA, representaciones, modelado, evaluación |
| `03_extension.md` | Ideas de extensión |
| `04_entrega_y_evaluacion.md` | Entrega, rúbrica y qué no se evalúa |
| `05_apendice_A_representaciones.md` | Apéndice A: preprocesamiento, TF-IDF y Word2Vec |

---

## 1. Introducción

En este proyecto aplicaréis las técnicas de Machine Learning estudiadas durante la asignatura a un problema de clasificación de documentos de texto. En concreto, trabajaréis con los resúmenes de proyectos de investigación financiados por la Comisión Europea dentro del programa Horizonte 2020 (H2020). El objetivo será formular y resolver un problema de clasificación multiclase, comparando distintas formas de representar el texto y distintos modelos de clasificación, y analizar de forma crítica su rendimiento.

- El proyecto se realizará en **equipos de 3 o 4 estudiantes**.
- Puntuación máxima: **3 puntos**:
  - Proyecto básico: **2,25 puntos**.
  - Extensión: **0,75 puntos**.
- **Fecha límite de entrega: 11 de diciembre de 2026 a las 23:59.**
- Presentación individual: semana del 14 de diciembre de 2026.

## 2. El conjunto de datos

El conjunto de datos se proporciona en el fichero Excel `projects.xlsx`. Contiene **35.378 registros** correspondientes a proyectos financiados por la Comisión Europea dentro de H2020. Para cada proyecto se proporciona:

| Campo | Descripción |
|---|---|
| `projectID` | Identificador del proyecto |
| `acronym` | Acrónimo del proyecto |
| `title` | Título del proyecto |
| `summary` | Resumen del proyecto |
| `startDate`, `endDate` | Fechas de inicio y fin de ejecución |
| `totalCost` | Coste total del proyecto |
| `ecMaxContribution` | Financiación de la UE |
| `countryContr` | Financiación recibida por cada país |
| `coordinatorCountry` | País coordinador del proyecto |
| `euroSciVocCode` | Código de la categoría temática. Las categorías asociadas a cada código están en `SciVocCodes.xlsx` |
| `publicationID` | Identificadores de las publicaciones asociadas |
| `patentID` | Identificadores de las patentes asociadas |

La base de datos se ha construido a partir de la información publicada por CORDIS en el portal de datos de la Unión Europea: <https://data.europa.eu/data/datasets/cordish2020projects>
