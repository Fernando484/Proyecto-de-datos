# 6. Entrega del proyecto

La entrega utiliza **GitHub** como herramienta de colaboración y control de versiones, y debe incluir:

- [ ] Código completamente funcional y documentado.
- [ ] Un `README.md` (o sitio de GitHub Pages) que actúe como **informe del proyecto**. No es una guía de instalación; debe contener el informe de la implementación y el análisis de resultados:
  - Descripción del problema y del conjunto de datos.
  - Explicación de las metodologías, con detalle suficiente para saber qué se ejecutó realmente: preprocesamiento, representaciones, modelos, búsqueda de hiperparámetros y protocolo train/validación/test.
  - Resultados experimentales y discusión. Las gráficas y tablas deben estar **en el propio README**, no solo en notebooks: curvas de aprendizaje o validación, comparación entre representaciones y modelos, matriz de confusión de la mejor configuración y cualquier otro resultado usado en la argumentación. **Toda afirmación del análisis debe estar respaldada por una figura o tabla consultable sin ejecutar el código.**
  - Conclusiones, incluyendo las limitaciones de las que seáis conscientes.
- [ ] Todos los ficheros o scripts necesarios para reproducir los resultados. **No subir `projects.xlsx`**; incluir el código que lea los datos desde una ruta local.

Condiciones:

- El repositorio debe ser **público**.
- Las páginas de documentación no podrán editarse tras la fecha límite; se evalúa el **último commit anterior a la fecha límite**.

## Entrega en Aula Global

Solo un estudiante por grupo sube:

- [ ] El enlace al repositorio de GitHub.
- [ ] Una breve declaración de contribuciones (qué hizo cada miembro).
- [ ] Las diapositivas de la presentación (se permiten pequeños cambios antes de la presentación final).

Sin excepción, la documentación debe respetar el principio de reconocimiento de autoría: si usáis fragmentos de código externos o material de otras fuentes, indicadlo claramente en el informe.

---

# 7. Evaluación

Se evalúa a partir de la documentación entregada y de una **presentación individual** (semana del 14 de diciembre de 2026). La calidad metodológica se evalúa en ambos sitios: en la documentación y en cómo defendáis vuestras decisiones ante las preguntas.

| Criterio | Puntuación |
|---|---|
| **Proyecto básico** | **2,25** |
| Metodología | 0,75 |
| Calidad de la documentación | 0,30 |
| Calidad del código | 0,30 |
| Calidad de la presentación | 0,50 |
| Respuestas a comentarios/preguntas de la presentación | 0,40 |
| **Extensión** | **0,75** |
| Originalidad | 0,25 |
| Calidad del trabajo | 0,50 |

## Qué NO se evalúa

**La exactitud alcanzada.** Ningún criterio recompensa un rendimiento alto. El dataset, la etiqueta y el techo (Sección 3) son iguales para todos, y unos pocos puntos de macro-F1 dicen más sobre la arbitrariedad de la etiqueta que sobre la calidad del trabajo.

Lo que se evalúa:

- Corrección metodológica: protocolo limpio de train/validación/test.
- Hiperparámetros ajustados sin usar el test.
- Métricas adecuadas para multiclase desbalanceada.
- Comparaciones realmente comparables.
- Discusión crítica de los resultados, sean cuales sean.

Un resultado modesto bien medido, comprendido y explicado vale más que una cifra mejor que no podáis justificar. Informar de un resultado negativo (una representación que no ayuda, una extensión que no supera el baseline) es perfectamente válido si el análisis explica por qué.

Esto se aplica igual al proyecto básico y a la extensión, y tanto al README como a la presentación. **No dediquéis la presentación a enumerar cifras**: centradla en qué hicisteis, por qué y qué conclusiones sacasteis.
